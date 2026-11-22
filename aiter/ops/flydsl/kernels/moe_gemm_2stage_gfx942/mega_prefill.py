# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Persistent prefill MoE for gfx942 PTPC FP8 (step 1: gate-up + down in one launch).

One 256-thread workgroup per CU loops over tickets from its XCD's queue. Each XCD owns the
metadata blocks b with (b // 8) % 8 == xcd, so a block's gate-up (which writes the fp8 h rows
and scales) and its down (which reads them) run on the same XCD and h never crosses L2s.
Queue order with lag L: G0 .. G(L-1), then D(i), G(i+L) alternating, then the last downs;
a down only waits for a gate-up with a smaller ticket of its own queue, so it never deadlocks.
The block bodies are the ones the per-block kernels use (gemm1.gateup_prefill_block,
gemm2_1x4.down_1x4_block), so the output is bit-identical to the separate launches.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm, vector
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from .. import buffer_ops
from . import common as fxh
from .gemm1 import _scf_if_then, compile_moe_gemm1
from .gemm2_1x4 import _build_moe_gemm2_1x4

_XCD = 4  # MI308X: 4 XCDs x 20 CUs
_GROUP = 8  # consecutive metadata blocks per XCD turn
_SPIN_MAX = 2_000_000


def _gptr(addr_i64):
    ptr = buffer_ops.create_llvm_ptr(fx.Int64(addr_i64), address_space=1)
    return ptr._value if hasattr(ptr, "_value") else ptr


def _atomic_add(addr_i64, val):
    return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, _gptr(addr_i64), _raw(fx.Int32(val)),
                                     llvm.AtomicOrdering.monotonic, syncscope="agent",
                                     alignment=4).res)


def _load_volatile(addr_i64):
    return fx.Int32(llvm.LoadOp(T.i32, _gptr(addr_i64), alignment=4,
                                ordering=llvm.AtomicOrdering.monotonic, syncscope="agent",
                                volatile_=True).result)


def _fence(ordering):
    llvm.FenceOp(ordering, syncscope="agent")


def _xcc_id():
    v = llvm.InlineAsmOp(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)", "=s",
                         has_side_effects=True).result
    return fx.Int32(v)


def _rfl(v):
    """Wave-uniform value into an SGPR (keeps loop-invariant scalars out of VGPRs)."""
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


def _seg_ptr(tab_i64, idx, elem_ty, align=4):
    """Pointer idx of the segment table (int64 addresses), as a wave-uniform pointer."""
    addr = fx.Int64(_load_i64(tab_i64 + fx.Int64(idx * 8)))
    lo = _rfl(fx.Int32(addr & 0xFFFFFFFF))
    hi = _rfl(fx.Int32(addr >> 32))
    a64 = (fx.Int64(hi) << 32) | (fx.Int64(lo) & 0xFFFFFFFF)
    return fx.inttoptr(fx.PointerType.get(elem_ty, address_space=fx.AddressSpace.Global,
                                          alignment=align), a64)


def _load_i64(addr_i64):
    return fx.Int64(llvm.LoadOp(T.i64, _gptr(addr_i64), alignment=8).result)


def _wg_sync():
    """s_barrier with workgroup-scope release/acquire, so LDS accesses cannot move across it."""
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="workgroup")
    gpu.barrier()
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")


def _inv_l1():
    llvm.InlineAsmOp(None, [], "buffer_inv sc0", "", has_side_effects=True)


def _hw_id():
    v = llvm.InlineAsmOp(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s",
                         has_side_effects=True).result
    return fx.Int32(v)


def _acq(dbg):
    if (dbg & 128) == 0:
        _fence(llvm.AtomicOrdering.acquire)


def _inv(dbg):
    if (dbg & 128) == 0:
        _inv_l1()


def _inv_l1_l2():
    llvm.InlineAsmOp(None, [], "buffer_inv sc0 sc1\ns_waitcnt vmcnt(0)", "~{memory}", has_side_effects=True)


def _memtime_lo():
    v = llvm.InlineAsmOp(T.i64, [], "s_memtime $0\ns_waitcnt lgkmcnt(0)", "=s", has_side_effects=True).result
    return fx.Int32(fx.Int64(v) & 0x7FFFFFFF)


def _store_wt(addr_i64, val):
    # plain 32-bit store; the writer's release fence (L2 writeback) before its sync signal
    # makes it visible to the other XCDs
    llvm.StoreOp(_raw(fx.Int32(val)), _gptr(addr_i64), alignment=4)


def _lds_ptr3(base_i64, off):
    ptr = buffer_ops.create_llvm_ptr(fx.Int64(base_i64) + fx.Int64(off), address_space=3)
    return ptr._value if hasattr(ptr, "_value") else ptr


def _lds_add(base_i64, off, val):
    return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, _lds_ptr3(base_i64, off), _raw(fx.Int32(val)),
                                     llvm.AtomicOrdering.monotonic, syncscope="workgroup",
                                     alignment=4).res)


def _lds_ld(base_i64, off):
    return fx.Int32(llvm.LoadOp(T.i32, _lds_ptr3(base_i64, off), alignment=4).result)


def _lds_st(base_i64, off, val):
    llvm.StoreOp(_raw(fx.Int32(val)), _lds_ptr3(base_i64, off), alignment=4)


class _Carved:
    """Stands in for a SharedAllocator storage: .peek().view(layout) over a carved pointer."""

    def __init__(self, it):
        self.it = it

    def peek(self):
        return self

    def view(self, layout):
        return fx.make_view(self.it, layout)


def _carver(base_i8):
    state = {"off": 0}

    def alloc(dtype, count, align):
        a = align or 16
        off = (state["off"] + a - 1) // a * a
        state["off"] = off + count * dtype.width // 8
        return _Carved(fx.recast_iter(dtype, base_i8 + off))

    return alloc


@functools.lru_cache(maxsize=8)
def build_mega_prefill(*, model_dim, inter_dim, topk, experts, lag=2, grid=80, dbg=0, fuse_sum=True, store_cache=18):
    N1 = 2 * inter_dim
    gu = compile_moe_gemm1(
        N=N1, K=model_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=N1, tile_k=128, alg="prefill_1x4", E=experts,
        act_quant_type="ptpc", activation="silu", METADATA_TILE_SIZE_M=64, FUSE_QUANT=True,
        WAVES_PER_EU=2 if N1 <= 384 else 0, XCD_GROUP=0,
    )
    dn = _build_moe_gemm2_1x4(
        N=model_dim, K=inter_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256, stage="down", alg="prefill_1x4", E=experts,
        USE_ATOMIC_WRITE=False, act_quant_type="ptpc", down_path="1x4_64x256",
        down_output_padding_bytes=0, METADATA_TILE_SIZE_M=64,
        _store_cache=store_cache if fuse_sum else 2,
    )
    gu_block, GuStorage, dn_block = gu.block, gu.storage, dn.block
    K = model_dim
    FP8 = fx.Float8E4M3FNUZ

    @flyc.kernel(known_block_size=[256, 1, 1])
    def mega_moe_prefill(
        p_xq: fx.Pointer, p_w13: fx.Pointer, p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer, p_seid: fx.Pointer, p_nv: fx.Pointer,
        p_s13: fx.Pointer, p_xscale: fx.Pointer, M: fx.Int32,
        sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32, alim: fx.Float32,
        i64_h: fx.Int64, i64_hscale: fx.Int64, p_h: fx.Pointer, p_hscale: fx.Pointer,
        p_w2: fx.Pointer, p_s2: fx.Pointer, p_gemm2_out: fx.Pointer, p_ctl: fx.Pointer,
        p_loc: fx.Pointer, p_out: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        sa = fx.SharedAllocator()
        lds = sa.allocate(GuStorage)
        tick = sa.allocate(fx.Array[fx.Int32, 4, 16]).peek()
        tick_v = tick.view(fx.make_layout(4, 1))
        lds_base = fx.recast_iter(fx.Int8, lds.sorted_lds.peek().ptr)
        ctl = fx.Int64(fx.ptrtoint(p_ctl))
        nv = fxh.view_as_torch_tensor(p_nv, (1,), fx.Int32)[0]
        nblk = nv // 64
        xcd = _xcc_id() & (_XCD - 1)
        # blocks of this XCD: b = g * 64 + xcd * 8 + j for group g, j < 8
        full_groups = nblk // (_XCD * _GROUP)
        rem = nblk - full_groups * (_XCD * _GROUP)
        tail = fx.Int32(arith_min(arith_max(rem - xcd * _GROUP, 0), _GROUP))
        n_x = full_groups * _GROUP + tail
        lag_x = arith_min(fx.Int32(lag), n_x)
        n_gd = 2 * n_x
        n_tasks = n_gd if dbg != 8 else n_gd * 0

        xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)), fx.make_layout((M, K), (K, 1)))
        w13_it = fx.recast_iter(FP8, fxh._as_ptr(p_w13))

        if tid == 0:
            _atomic_add(ctl + fx.Int64(48 + (xcd & 3) * 4), 1)
            tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
        gpu.barrier()
        t = tick_v[0]
        n_dn = fx.Int32(0)
        while t < n_tasks:
            # decode ticket -> (is_down, local index)
            u = t - lag_x
            pair_end = 2 * (n_x - lag_x)
            is_down = (t >= lag_x) & ((u >= pair_end) | (u % 2 == 0))
            loc = fx.Int32(arith_select(t < lag_x, t,
                                        arith_select(u < pair_end,
                                                     arith_select(u % 2 == 0, u // 2, u // 2 + lag_x),
                                                     n_x - lag_x + (u - pair_end))))
            b = (loc // _GROUP) * (_XCD * _GROUP) + xcd * _GROUP + loc % _GROUP
            is_sum = t < 0
            if is_down & (dbg != 2) & (dbg != 3):
                if (tid == 0) & (dbg != 1) & (dbg != 7):
                    cur = _load_volatile(ctl + fx.Int64(64 + b * 4))
                    spins = fx.Int32(0)
                    while (cur < 1) & (spins < _SPIN_MAX):
                        rocdl.s_sleep(1)
                        spins = spins + 1
                        cur = _load_volatile(ctl + fx.Int64(64 + b * 4))
                    if spins >= _SPIN_MAX:
                        _atomic_add(ctl + fx.Int64(40), 1)
                    _fence(llvm.AtomicOrdering.acquire)
                gpu.barrier()
                dn_block(b, p_h, p_w2, p_gemm2_out, p_sorted_ids,
                         p_sorted_weights, p_seid, p_s2, p_hscale, M, _carver(lds_base))
                n_dn = n_dn + 1
            if (is_down == False) & (is_sum == False) & (dbg != 3) & (dbg != 7) & ((dbg != 5) | (t == 0)):  # noqa: E712
                gu_block(lds, b, 0, tid, M, xq_view, w13_it, p_xq, p_sorted_ids,
                         p_sorted_weights, p_seid, p_s13, p_xscale, sb, sbr, slb, slbr, alim,
                         i64_h, i64_hscale)
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == 0:
                    _fence(llvm.AtomicOrdering.release)
                    _atomic_add(ctl + fx.Int64(64 + b * 4), 1)
            gpu.barrier()
            if tid == 0:
                tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
            gpu.barrier()
            t = tick_v[0]

        if const_expr(fuse_sum):
            # publish this WG's downs once, then wait for every down (all XCDs) to land
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tid == 0:
                _fence(llvm.AtomicOrdering.release)
                _atomic_add(ctl + fx.Int64(32), n_dn)
            if (tid == 0) & (dbg != 8):
                cur = _load_volatile(ctl + fx.Int64(32))
                spins = fx.Int32(0)
                while (cur < nblk) & (spins < _SPIN_MAX):
                    rocdl.s_sleep(1)
                    spins = spins + 1
                    cur = _load_volatile(ctl + fx.Int64(32))
                if spins >= _SPIN_MAX:
                    _atomic_add(ctl + fx.Int64(44), 1)
                _fence(llvm.AtomicOrdering.acquire)
            gpu.barrier()
            wg = fx.Int32(gpu.block_idx.x)
            chunks = (M + 3) // 4
            c0 = wg * chunks // grid
            c1 = (wg + 1) * chunks // grid
            c = c0
            while c < c1:
                _sum_tokens(c, tid, M, p_loc, p_gemm2_out, p_out)
                c = c + 1

    def _sum_tokens(chunk, tid, M, p_loc, p_g2, p_out):
        # Same math and order as moe_reduce.sorted_sum: f32 sum over the TOPK routes in slot
        # order (merged routes have loc -1), then one conversion to bf16. One wave per token.
        lane = tid % 64
        tok = chunk * 4 + tid // 64
        tok_c = arith_min(tok, M - 1)
        locs = [p_loc[tok_c * topk + k] for k in range(topk)]
        valid = [loc >= 0 for loc in locs]
        g2 = fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_g2))
        rows = [g2 + fx.Int64(valid[k].select(locs[k], 0)) * model_dim for k in range(topk)]
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
        copy_atom_b = fx.make_copy_atom(fx.rocdl.BufferCopy(128), fx.BFloat16)
        out_row = fx.make_view(fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_out)) + fx.Int64(tok) * model_dim,
                               fx.make_layout(model_dim, 1))
        out_buf = fx.rocdl.make_buffer_tensor(
            out_row, max_size=False,
            num_records_bytes=(tok < M).select(fx.Int32(model_dim * 2), fx.Int32(0)))
        n_chunk = model_dim // 8 // 64
        frags = []
        for i in range(n_chunk):  # issue every load first: one wave per CU-SIMD needs the MLP
            col = (lane + 64 * i) * 8
            row_frags = []
            for k in range(topk):
                atom = fx.make_view(rows[k] + col, fx.make_layout(8, 1))
                frag = fx.make_fragment_like(atom)
                fx.copy(copy_atom, atom, frag)
                row_frags.append(frag)
            frags.append(row_frags)
        for i in range(n_chunk):
            col = (lane + 64 * i) * 8
            acc = valid[0].select(frags[i][0].load().to(fx.Float32), 0.0)
            for k in range(1, topk):
                acc += valid[k].select(frags[i][k].load().to(fx.Float32), 0.0)
            dst = fx.make_view(fx.get_iter(out_buf) + col, fx.make_layout(8, 1))
            of = fx.make_fragment_like(dst)
            of.store(acc.to(fx.BFloat16))
            fx.copy(copy_atom_b, of, dst)

    @flyc.jit
    def launch(p_xq: fx.Pointer, p_w13: fx.Pointer, p_sorted_ids: fx.Pointer,
               p_sorted_weights: fx.Pointer, p_seid: fx.Pointer, p_nv: fx.Pointer,
               p_s13: fx.Pointer, p_xscale: fx.Pointer, M: fx.Int32,
               sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, i64_h: fx.Int64, i64_hscale: fx.Int64,
               p_h: fx.Pointer, p_hscale: fx.Pointer, p_w2: fx.Pointer, p_s2: fx.Pointer,
               p_gemm2_out: fx.Pointer, p_ctl: fx.Pointer, p_loc: fx.Pointer, p_out: fx.Pointer,
               stream: fx.Stream):
        mega_moe_prefill(p_xq, p_w13, p_sorted_ids, p_sorted_weights, p_seid, p_nv, p_s13,
                         p_xscale, M, sb, sbr, slb, slbr, alim, i64_h, i64_hscale, p_h, p_hscale, p_w2, p_s2,
                         p_gemm2_out, p_ctl, p_loc, p_out,
                         value_attrs={"rocdl.waves_per_eu": 1},
                         ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch




@functools.lru_cache(maxsize=8)
def build_mega_prefill_v2(*, model_dim, inter_dim, topk, experts, lag=8, grid=160, dbg=0, fuse_sum=True, store_cache=18, gu_bn=128, h_stride=None, lean=True):
    H_STRIDE = h_stride or (inter_dim + 127) // 128 * 128  # fp8 h row stride: whole cache lines
    HS_STRIDE = 32  # one 128-byte line per route scale: no line is shared between blocks
    N1 = 2 * inter_dim
    import os as _os
    _os.environ["MOE_GATEUP_MIN_LDS"] = _os.environ.get("MEGA_LDS", str(30 * 1024))  # down carves ~29.7 KB
    gu = compile_moe_gemm1(
        N=N1, K=model_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=gu_bn, tile_k=128, alg="prefill_1x4", E=experts,
        act_quant_type="ptpc", activation="silu", METADATA_TILE_SIZE_M=64, FUSE_QUANT=False,
        XCD_GROUP=0,
    )
    _os.environ.pop("MOE_GATEUP_MIN_LDS", None)
    _os.environ["MOE_DOWN_A_STRIDE"] = str(H_STRIDE)
    _os.environ["MOE_DOWN_AS_STRIDE"] = str(HS_STRIDE)
    dn = _build_moe_gemm2_1x4(
        N=model_dim, K=inter_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256, stage="down", alg="prefill_1x4", E=experts,
        USE_ATOMIC_WRITE=False, act_quant_type="ptpc", down_path="1x4_64x256",
        down_output_padding_bytes=0, METADATA_TILE_SIZE_M=64,
        _store_cache=2 if lean else (store_cache if fuse_sum else 2),
    )
    _os.environ.pop("MOE_DOWN_A_STRIDE", None)
    _os.environ.pop("MOE_DOWN_AS_STRIDE", None)
    gu_block, GuStorage, dn_block = gu.block, gu.storage, dn.block
    NT = N1 // gu_bn  # gate-up N tiles per block
    TILES_PER_TASK = int(_os.environ.get("MEGA_TILES_PER_TASK", str(NT)))
    NTT = NT // TILES_PER_TASK  # gate-up tasks per block
    LDS_PAD = 14 * 1024  # down carves ~29 KB from the start of the gate-up storage
    K = model_dim
    FP8 = fx.Float8E4M3FNUZ

    @flyc.kernel(known_block_size=[256, 1, 1])
    def mega_moe_prefill_v2(
        p_xq: fx.Pointer, p_w13: fx.Pointer, p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer, p_seid: fx.Pointer, p_nv: fx.Pointer,
        p_s13: fx.Pointer, p_xscale: fx.Pointer, M: fx.Int32,
        sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32, alim: fx.Float32,
        i64_h: fx.Int64, i64_hscale: fx.Int64, p_h: fx.Pointer, p_hscale: fx.Pointer,
        p_w2: fx.Pointer, p_s2: fx.Pointer, p_gemm2_out: fx.Pointer, p_ctl: fx.Pointer,
        p_loc: fx.Pointer, p_out: fx.Pointer, p_hbf: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        sa = fx.SharedAllocator()
        tick = sa.allocate(fx.Array[fx.Int32, 4, 16]).peek()
        lds = sa.allocate(GuStorage)
        tick_v = tick.view(fx.make_layout(4, 1))
        lds_base = fx.recast_iter(fx.Int8, lds.sorted_lds.peek().ptr)
        ctl = fx.Int64(fx.ptrtoint(p_ctl))
        nv = fxh.view_as_torch_tensor(p_nv, (1,), fx.Int32)[0]
        nblk = nv // 64
        xcd = _xcc_id() & (_XCD - 1)
        if const_expr(dbg & 4096):
            if tid == 0:
                _atomic_add(ctl + fx.Int64(240), arith_select(xcd == fx.Int32(gpu.block_idx.x) % _XCD, 0, 1))
        # blocks of this XCD: b = g * 64 + xcd * 8 + j for group g, j < 8
        full_groups = nblk // (_XCD * _GROUP)
        rem = nblk - full_groups * (_XCD * _GROUP)
        tail = fx.Int32(arith_min(arith_max(rem - xcd * _GROUP, 0), _GROUP))
        n_x = full_groups * _GROUP + tail
        lag_x = arith_min(fx.Int32(lag), n_x)
        n_gd = (NTT + 1) * n_x
        n_tasks = n_gd

        xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)), fx.make_layout((M, K), (K, 1)))
        w13_it = fx.recast_iter(FP8, fxh._as_ptr(p_w13))

        if const_expr(dbg & 16384):
            # G-only ticket loop with the static loop's body (no branches)
            if tid == 0:
                tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
            gpu.barrier()
            tq = _rfl(tick_v[0])
            iq = fx.Int32(0)
            while (tq < NT * n_x) & (iq < 1000000):
                iq = iq + 1
                lq = tq // NT
                bq = (lq // _GROUP) * (_XCD * _GROUP) + xcd * _GROUP + lq % _GROUP
                if tid == 0:
                    _atomic_add(ctl + fx.Int64(262144 + xcd * 16384 + tq * 4), bq * 8 + tq % NT + 1)
                if const_expr((dbg & 32768) == 0 or (dbg & 65536)):
                    gu_block(lds, bq, tq % NT, tid, M, xq_view, w13_it, p_hbf, p_sorted_ids,
                             p_sorted_weights, p_seid, p_s13, p_xscale, sb, sbr, slb, slbr, alim,
                             i64_h, i64_hscale, sync_before_gemm=True)
                gpu.barrier()
                if tid == 0:
                    _atomic_add(ctl + fx.Int64(248), arith_select(tick_v[0] == tq, 0, 1))
                    tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
                gpu.barrier()
                tq = _rfl(tick_v[0])
            n_tasks = n_tasks * 0
        if const_expr(dbg & 256):
            # static G-only schedule, no tickets: WG w takes (block, tile) w, w + grid, ...
            st = fx.Int32(gpu.block_idx.x)
            if const_expr(dbg & 8192):
                sk = fx.Int32(0)
                while sk < (st % 7) * 300:
                    rocdl.s_sleep(8)
                    sk = sk + 1
            while st < nblk * NT:
                gu_block(lds, st // NT, st % NT, tid, M, xq_view, w13_it, p_hbf, p_sorted_ids,
                         p_sorted_weights, p_seid, p_s13, p_xscale, sb, sbr, slb, slbr, alim,
                         i64_h, i64_hscale, sync_before_gemm=True)
                gpu.barrier()
                st = st + grid
            n_tasks = n_tasks * 0
        # Static per-XCD schedule: rank r within this XCD (one atomic), wait until every WG
        # registered, then tasks r, r + W, ... of this XCD's sequence (W = WGs on this XCD).
        # Every WG is resident and a task only waits for earlier tasks, so it cannot deadlock.
        if tid == 0:
            tick_v[0] = _atomic_add(ctl + fx.Int64(48 + xcd * 4), 1)
            _atomic_add(ctl + fx.Int64(16), 1)
        if tid < 64:
            creg = _load_volatile(ctl + fx.Int64(16))
            while creg < grid:
                rocdl.s_sleep(1)
                creg = _load_volatile(ctl + fx.Int64(16))
            if tid == 0:
                tick_v[1] = _load_volatile(ctl + fx.Int64(48 + xcd * 4))
        gpu.barrier()
        t = _rfl(tick_v[0])
        W = _rfl(tick_v[1])
        n_dn = fx.Int32(0)
        tk0 = _memtime_lo()
        iters = fx.Int32(0)
        while (t < n_tasks) & (iters < 1000000):
            iters = iters + 1
            ti0 = _memtime_lo()
            # tasks: G(0..lag-1) | [D(i), G(i+lag)] ... | last downs; a G task = all NT tiles
            u = t - NTT * lag_x
            pair_end = (NTT + 1) * (n_x - lag_x)
            in_pairs = (t >= NTT * lag_x) & (u < pair_end)
            is_down = (t >= NTT * lag_x) & ((u >= pair_end) | (u % (NTT + 1) == 0))
            loc = fx.Int32(arith_select(t < NTT * lag_x, t // NTT, arith_select(
                in_pairs, arith_select(u % (NTT + 1) == 0, u // (NTT + 1), u // (NTT + 1) + lag_x),
                n_x - lag_x + (u - pair_end))))
            tile = fx.Int32(arith_select(t < NTT * lag_x, t % NTT, u % (NTT + 1) - 1))
            if const_expr(dbg & 1024):
                is_down = t < 0
                loc = t // NT
                tile = t % NT
            b = (loc // _GROUP) * (_XCD * _GROUP) + xcd * _GROUP + loc % _GROUP
            is_sum = t < 0
            if is_down & (dbg != 2) & (dbg != 3):
                if (tid < 64) & (dbg != 1) & (dbg != 7):
                    cur = _load_volatile(ctl + fx.Int64(64 + b * 4))
                    spins = fx.Int32(0)
                    while (cur < NTT) & (spins < _SPIN_MAX):
                        rocdl.s_sleep(1)
                        spins = spins + 1
                        cur = _load_volatile(ctl + fx.Int64(64 + b * 4))
                    if spins >= _SPIN_MAX:
                        _atomic_add(ctl + fx.Int64(40), 1)
                    if const_expr(not lean):
                        _acq(dbg)
                gpu.barrier()
                tda = _memtime_lo()
                _quant_block(b, tid, M, p_sorted_ids, p_hbf, i64_h, i64_hscale)
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if const_expr(not lean):
                    _inv(dbg)
                tdb = _memtime_lo()
                dn_block(b, p_h, p_w2, p_gemm2_out, p_sorted_ids,
                         p_sorted_weights, p_seid, p_s2, p_hscale, M, _carver(lds_base))
                if const_expr(dbg & 262144):
                    gpu.barrier()
                    if tid == 0:
                        wgd = fx.Int32(gpu.block_idx.x) * 4
                        _atomic_add(ctl + fx.Int64(139264 + wgd), tda - ti0)
                        _atomic_add(ctl + fx.Int64(143360 + wgd), tdb - tda)
                        _atomic_add(ctl + fx.Int64(147456 + wgd), _memtime_lo() - tdb)
                n_dn = n_dn + 1
            if (is_down == False) & (is_sum == False) & (dbg != 3) & (dbg != 7) & ((dbg != 5) | (t == 0)):  # noqa: E712
                if const_expr((dbg & (256 | 16384)) == 0):
                    t0c = _memtime_lo()
                    for tl in range(TILES_PER_TASK):
                        gu_block(lds, b, tile * TILES_PER_TASK + tl, tid, M, xq_view, w13_it, p_hbf,
                                 p_sorted_ids, p_sorted_weights, p_seid, p_s13, p_xscale, sb, sbr,
                                 slb, slbr, alim, i64_h, i64_hscale, sync_before_gemm=True)
                        gpu.barrier()
                    if const_expr(dbg & 262144):
                        gpu.barrier()
                        if tid == 0:
                            _atomic_add(ctl + fx.Int64(65536 + fx.Int32(gpu.block_idx.x) * 8), _memtime_lo() - t0c)
                            _atomic_add(ctl + fx.Int64(65540 + fx.Int32(gpu.block_idx.x) * 8), 1)
                if const_expr(dbg & 64):
                    if tid == 0:
                        _atomic_add(ctl + fx.Int64(65536 + (b * NT + tile) * 8), _hw_id())
                        _atomic_add(ctl + fx.Int64(65540 + (b * NT + tile) * 8),
                                    fx.Int32(gpu.block_idx.x) + 1)
                        _atomic_add(ctl + fx.Int64(131072 + (b * NT + tile) * 4), 1)
                if const_expr((dbg & 131072) == 0):
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    if tid == 0:
                        if const_expr((dbg & 512) == 0 and not lean):
                            _fence(llvm.AtomicOrdering.release)
                        _atomic_add(ctl + fx.Int64(64 + b * 4), 1)
            gpu.barrier()
            if const_expr(dbg & 262144):
                if tid == 0:
                    dti = _memtime_lo() - ti0
                    _atomic_add(ctl + fx.Int64(arith_select(is_down, 131072, 135168) + fx.Int32(gpu.block_idx.x) * 4), dti)
            t = t + W

        if const_expr(dbg & 262144):
            if tid == 0:
                _atomic_add(ctl + fx.Int64(98304 + fx.Int32(gpu.block_idx.x) * 4), _memtime_lo() - tk0)
        if const_expr(fuse_sum and (dbg & 2048) == 0):
            # publish this WG's downs once, then wait for every down (all XCDs) to land
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tid == 0:
                _fence(llvm.AtomicOrdering.release)
                _atomic_add(ctl + fx.Int64(32), n_dn)
            if (tid == 0) & (dbg != 8):
                cur = _load_volatile(ctl + fx.Int64(32))
                spins = fx.Int32(0)
                while (cur < nblk) & (spins < _SPIN_MAX):
                    rocdl.s_sleep(1)
                    spins = spins + 1
                    cur = _load_volatile(ctl + fx.Int64(32))
                if spins >= _SPIN_MAX:
                    _atomic_add(ctl + fx.Int64(44), 1)
                _acq(dbg)
            gpu.barrier()
            # g2 rows come from every XCD (write-through stores); drop any line this XCD's L2
            # still holds for those addresses (e.g. from an earlier launch) before reading them
            _inv_l1_l2()
            gpu.barrier()
            wg = fx.Int32(gpu.block_idx.x)
            chunks = (M + 3) // 4
            c0 = wg * chunks // grid
            c1 = (wg + 1) * chunks // grid
            c = c0
            while c < c1:
                _sum_tokens(c, tid, M, p_loc, p_gemm2_out, p_out)
                c = c + 1

    def _quant_block(b, tid, M, p_sid, p_hbf, i64_q, i64_qs):
        # Per-route fp8 quant of block b's 64 bf16 h rows, the math of the fused gate-up
        # epilogue (gemm1 _quant_rows_epilogue): 4 lanes per row, 8 channels per 32-channel
        # slice each, amax over the row, scale = amax / 240, q = clamp(x * rcp(scale)).
        inter = inter_dim
        quarter = tid % 4
        row = tid // 4
        sid = fx.Int32(p_sid[b * 64 + row])
        token = sid & fx.Int32(0xFFFFFF)
        slot = sid >> fx.Int32(24)
        valid = token < M
        route = valid.select(token * topk + slot, fx.Int32(0))
        hb_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(p_hbf)), num_records_bytes=fx.Int64(M) * fx.Int64(topk * inter * 2))
        q_rsrc = buffer_ops.create_buffer_resource_from_addr(
            i64_q, num_records_bytes=fx.Int64(M) * fx.Int64(topk * H_STRIDE))
        s_rsrc = buffer_ops.create_buffer_resource_from_addr(
            i64_qs, num_records_bytes=fx.Int64(M) * fx.Int64(topk * 4 * HS_STRIDE))
        f32 = T.f32
        vals = []
        amax = fx.Float32(0.0)
        for sl in range(inter // 32):
            off = (route * inter + sl * 32 + quarter * 8) // 2  # in i32 words
            w4 = Vec(buffer_ops.buffer_load(hb_rsrc, off, vec_width=4, dtype=T.i32))
            v8 = w4.bitcast(fx.BFloat16)
            f8 = [v8[j].to(fx.Float32) for j in range(8)]
            for j in range(8):
                amax = amax.maximumf(fx.Float32(llvm.call_intrinsic(f32, "llvm.fabs.f32", [_raw(f8[j])], [], [])))
            vals.append(f8)
        for dist in (1, 2):
            amax = amax.maximumf(amax.shuffle_xor(fx.Int32(dist), fx.Int32(64)))
        scale = amax * fx.Float32(float(1.0 / 240.0))
        inv_scale = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
        buffer_ops.buffer_store(_raw(scale), s_rsrc, route * fx.Int32(4 * HS_STRIDE),
                                mask=_raw((quarter == fx.Int32(0)) & valid), offset_is_bytes=True)
        fm_fast = fx.arith.FastMathFlags.fast
        for sl in range(inter // 32):
            f8 = vals[sl]
            q = [fx.math.clampf(f8[j] * inv_scale, fx.Float32(-240.0), fx.Float32(240.0), fastmath=fm_fast)
                 for j in range(8)]
            words = []
            for wd in range(2):
                lo = rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[4 * wd]), _raw(q[4 * wd + 1]), fx.Int32(0), False)
                words.append(rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[4 * wd + 2]), _raw(q[4 * wd + 3]), lo, True))
            vec2 = vector.from_elements(T.vec(2, T.i32), words)
            buffer_ops.buffer_store(vec2, q_rsrc, route * H_STRIDE + sl * 32 + quarter * 8,
                                    mask=_raw(valid), offset_is_bytes=True)

    def _sum_tokens(chunk, tid, M, p_loc, p_g2, p_out):
        # Same math and order as moe_reduce.sorted_sum: f32 sum over the TOPK routes in slot
        # order (merged routes have loc -1), then one conversion to bf16. One wave per token.
        lane = tid % 64
        tok = chunk * 4 + tid // 64
        tok_c = arith_min(tok, M - 1)
        locs = [p_loc[tok_c * topk + k] for k in range(topk)]
        valid = [loc >= 0 for loc in locs]
        g2 = fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_g2))
        rows = [g2 + fx.Int64(valid[k].select(locs[k], 0)) * model_dim for k in range(topk)]
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
        copy_atom_b = fx.make_copy_atom(fx.rocdl.BufferCopy(128), fx.BFloat16)
        out_row = fx.make_view(fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_out)) + fx.Int64(tok) * model_dim,
                               fx.make_layout(model_dim, 1))
        out_buf = fx.rocdl.make_buffer_tensor(
            out_row, max_size=False,
            num_records_bytes=(tok < M).select(fx.Int32(model_dim * 2), fx.Int32(0)))
        n_chunk = model_dim // 8 // 64
        for i in range(n_chunk):  # 2 waves/SIMD: one column chunk (11 loads) at a time
            col = (lane + 64 * i) * 8
            frags_i = []
            for k in range(topk):
                atom = fx.make_view(rows[k] + col, fx.make_layout(8, 1))
                frag = fx.make_fragment_like(atom)
                fx.copy(copy_atom, atom, frag)
                frags_i.append(frag)
            acc = valid[0].select(frags_i[0].load().to(fx.Float32), 0.0)
            for k in range(1, topk):
                acc += valid[k].select(frags_i[k].load().to(fx.Float32), 0.0)
            dst = fx.make_view(fx.get_iter(out_buf) + col, fx.make_layout(8, 1))
            of = fx.make_fragment_like(dst)
            of.store(acc.to(fx.BFloat16))
            fx.copy(copy_atom_b, of, dst)

    @flyc.jit
    def launch(p_xq: fx.Pointer, p_w13: fx.Pointer, p_sorted_ids: fx.Pointer,
               p_sorted_weights: fx.Pointer, p_seid: fx.Pointer, p_nv: fx.Pointer,
               p_s13: fx.Pointer, p_xscale: fx.Pointer, M: fx.Int32,
               sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, i64_h: fx.Int64, i64_hscale: fx.Int64,
               p_h: fx.Pointer, p_hscale: fx.Pointer, p_w2: fx.Pointer, p_s2: fx.Pointer,
               p_gemm2_out: fx.Pointer, p_ctl: fx.Pointer, p_loc: fx.Pointer, p_out: fx.Pointer,
               p_hbf: fx.Pointer, stream: fx.Stream):
        mega_moe_prefill_v2(p_xq, p_w13, p_sorted_ids, p_sorted_weights, p_seid, p_nv, p_s13,
                         p_xscale, M, sb, sbr, slb, slbr, alim, i64_h, i64_hscale, p_h, p_hscale, p_w2, p_s2,
                         p_gemm2_out, p_ctl, p_loc, p_out, p_hbf,
                         value_attrs={"rocdl.waves_per_eu": 2},
                         ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch

@functools.lru_cache(maxsize=8)
def build_mega_prefill_2e(*, model_dim, inter_dim, topk, experts, lag=2, grid=80, dbg=0, fuse_sum=True, store_cache=18):
    N1 = 2 * inter_dim
    gu = compile_moe_gemm1(
        N=N1, K=model_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=N1, tile_k=128, alg="prefill_1x4", E=experts,
        act_quant_type="ptpc", activation="silu", METADATA_TILE_SIZE_M=64, FUSE_QUANT=True,
        WAVES_PER_EU=2 if N1 <= 384 else 0, XCD_GROUP=0,
    )
    dn = _build_moe_gemm2_1x4(
        N=model_dim, K=inter_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256, stage="down", alg="prefill_1x4", E=experts,
        USE_ATOMIC_WRITE=False, act_quant_type="ptpc", down_path="1x4_64x256",
        down_output_padding_bytes=0, METADATA_TILE_SIZE_M=64,
        _store_cache=store_cache if fuse_sum else 2,
    )
    gu_block, GuStorage, dn_block = gu.block, gu.storage, dn.block
    K = model_dim
    FP8 = fx.Float8E4M3FNUZ

    @flyc.kernel(known_block_size=[256, 1, 1])
    def mega_moe_prefill_2e(
        p_xq: fx.Pointer, p_w13: fx.Pointer, p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer, p_seid: fx.Pointer, p_nv: fx.Pointer,
        p_s13: fx.Pointer, p_xscale: fx.Pointer, M: fx.Int32,
        sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32, alim: fx.Float32,
        i64_h: fx.Int64, i64_hscale: fx.Int64, p_h: fx.Pointer, p_hscale: fx.Pointer,
        p_w2: fx.Pointer, p_s2: fx.Pointer, p_gemm2_out: fx.Pointer, p_ctl: fx.Pointer,
        p_locA: fx.Pointer, p_outA: fx.Pointer, p_locB: fx.Pointer, p_outB: fx.Pointer,
        nbA: fx.Int32, MA: fx.Int32, MB: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        sa = fx.SharedAllocator()
        lds = sa.allocate(GuStorage)
        tick = sa.allocate(fx.Array[fx.Int32, 4, 16]).peek()
        tick_v = tick.view(fx.make_layout(4, 1))
        lds_base = fx.recast_iter(fx.Int8, lds.sorted_lds.peek().ptr)
        ctl = fx.Int64(fx.ptrtoint(p_ctl))
        nv = fxh.view_as_torch_tensor(p_nv, (1,), fx.Int32)[0]
        nblk = nv // 64
        xcd = _xcc_id() & (_XCD - 1)
        # blocks of this XCD: b = g * 64 + xcd * 8 + j for group g, j < 8
        full_groups = nblk // (_XCD * _GROUP)
        rem = nblk - full_groups * (_XCD * _GROUP)
        tail = fx.Int32(arith_min(arith_max(rem - xcd * _GROUP, 0), _GROUP))
        n_x = full_groups * _GROUP + tail
        lag_x = arith_min(fx.Int32(lag), n_x)
        n_gd = 2 * n_x
        n_tasks = n_gd
        # blocks [0, nbA) carry segment A (pushed to the peer): local count and the ticket of
        # this queue's last A down (D(i) is ticket lag + 2 i, or n_x + i in the drain)
        fgA = nbA // (_XCD * _GROUP)
        remA = nbA - fgA * (_XCD * _GROUP)
        nA_x = fgA * _GROUP + fx.Int32(arith_min(arith_max(remA - xcd * _GROUP, 0), _GROUP))
        iA = nA_x - 1
        T_A = arith_select(nA_x == 0, -1, arith_select(iA < n_x - lag_x, lag_x + 2 * iA, n_x + iA))
        wg = fx.Int32(gpu.block_idx.x)

        xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)), fx.make_layout((M, K), (K, 1)))
        w13_it = fx.recast_iter(FP8, fxh._as_ptr(p_w13))

        if tid == 0:
            _atomic_add(ctl + fx.Int64(48 + (xcd & 3) * 4), 1)
            tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
        gpu.barrier()
        t = tick_v[0]
        n_dnA = fx.Int32(0)
        n_dnB = fx.Int32(0)
        sa_done = fx.Int32(0)
        while (t < n_tasks) | (sa_done == 0):
            if ((t > T_A) | (t >= n_tasks)) & (sa_done == 0):
                # every A ticket of this queue is taken: publish, wait for all A downs, sum A
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == 0:
                    _fence(llvm.AtomicOrdering.release)
                    _atomic_add(ctl + fx.Int64(32), n_dnA)
                n_dnA = n_dnA * 0
                if tid < 64:
                    curA = _load_volatile(ctl + fx.Int64(32))
                    while curA < nbA:
                        rocdl.s_sleep(1)
                        curA = _load_volatile(ctl + fx.Int64(32))
                    _fence(llvm.AtomicOrdering.acquire)
                gpu.barrier()
                chA = (MA + 3) // 4
                ca = wg * chA // grid
                ca_end = (wg + 1) * chA // grid
                while ca < ca_end:
                    _sum_tokens(ca, tid, MA, p_locA, p_gemm2_out, p_outA)
                    ca = ca + 1
                sa_done = sa_done + 1
            # decode ticket -> (is_down, local index)
            u = t - lag_x
            pair_end = 2 * (n_x - lag_x)
            is_down = (t >= lag_x) & ((u >= pair_end) | (u % 2 == 0))
            loc = fx.Int32(arith_select(t < lag_x, t,
                                        arith_select(u < pair_end,
                                                     arith_select(u % 2 == 0, u // 2, u // 2 + lag_x),
                                                     n_x - lag_x + (u - pair_end))))
            b = (loc // _GROUP) * (_XCD * _GROUP) + xcd * _GROUP + loc % _GROUP
            is_sum = t < 0
            if is_down & (t < n_tasks):
                if (tid == 0) & (dbg != 1) & (dbg != 7):
                    cur = _load_volatile(ctl + fx.Int64(64 + b * 4))
                    spins = fx.Int32(0)
                    while (cur < 1) & (spins < _SPIN_MAX):
                        rocdl.s_sleep(1)
                        spins = spins + 1
                        cur = _load_volatile(ctl + fx.Int64(64 + b * 4))
                    if spins >= _SPIN_MAX:
                        _atomic_add(ctl + fx.Int64(40), 1)
                    _fence(llvm.AtomicOrdering.acquire)
                gpu.barrier()
                dn_block(b, p_h, p_w2, p_gemm2_out, p_sorted_ids,
                         p_sorted_weights, p_seid, p_s2, p_hscale, M, _carver(lds_base))
                n_dnA = n_dnA + arith_select(b < nbA, 1, 0)
                n_dnB = n_dnB + arith_select(b < nbA, 0, 1)
            if (is_down == False) & (t < n_tasks):  # noqa: E712
                gu_block(lds, b, 0, tid, M, xq_view, w13_it, p_xq, p_sorted_ids,
                         p_sorted_weights, p_seid, p_s13, p_xscale, sb, sbr, slb, slbr, alim,
                         i64_h, i64_hscale)
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == 0:
                    _fence(llvm.AtomicOrdering.release)
                    _atomic_add(ctl + fx.Int64(64 + b * 4), 1)
            gpu.barrier()
            if tid == 0:
                tick_v[0] = arith_select(t < n_tasks, _atomic_add(ctl + fx.Int64(xcd * 4), 1), t)
            gpu.barrier()
            t = tick_v[0]

        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            _fence(llvm.AtomicOrdering.release)
            _atomic_add(ctl + fx.Int64(36), n_dnB)
        if tid < 64:
            curB = _load_volatile(ctl + fx.Int64(36))
            while curB < nblk - nbA:
                rocdl.s_sleep(1)
                curB = _load_volatile(ctl + fx.Int64(36))
            _fence(llvm.AtomicOrdering.acquire)
        gpu.barrier()
        chB = (MB + 3) // 4
        cb = wg * chB // grid
        cb_end = (wg + 1) * chB // grid
        while cb < cb_end:
            _sum_tokens(cb, tid, MB, p_locB, p_gemm2_out, p_outB)
            cb = cb + 1

    def _sum_tokens(chunk, tid, M, p_loc, p_g2, p_out):
        # Same math and order as moe_reduce.sorted_sum: f32 sum over the TOPK routes in slot
        # order (merged routes have loc -1), then one conversion to bf16. One wave per token.
        lane = tid % 64
        tok = chunk * 4 + tid // 64
        tok_c = arith_min(tok, M - 1)
        locs = [p_loc[tok_c * topk + k] for k in range(topk)]
        valid = [loc >= 0 for loc in locs]
        g2 = fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_g2))
        rows = [g2 + fx.Int64(valid[k].select(locs[k], 0)) * model_dim for k in range(topk)]
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
        copy_atom_b = fx.make_copy_atom(fx.rocdl.BufferCopy(128), fx.BFloat16)
        out_row = fx.make_view(fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_out)) + fx.Int64(tok) * model_dim,
                               fx.make_layout(model_dim, 1))
        out_buf = fx.rocdl.make_buffer_tensor(
            out_row, max_size=False,
            num_records_bytes=(tok < M).select(fx.Int32(model_dim * 2), fx.Int32(0)))
        n_chunk = model_dim // 8 // 64
        frags = []
        for i in range(n_chunk):  # issue every load first: one wave per CU-SIMD needs the MLP
            col = (lane + 64 * i) * 8
            row_frags = []
            for k in range(topk):
                atom = fx.make_view(rows[k] + col, fx.make_layout(8, 1))
                frag = fx.make_fragment_like(atom)
                fx.copy(copy_atom, atom, frag)
                row_frags.append(frag)
            frags.append(row_frags)
        for i in range(n_chunk):
            col = (lane + 64 * i) * 8
            acc = valid[0].select(frags[i][0].load().to(fx.Float32), 0.0)
            for k in range(1, topk):
                acc += valid[k].select(frags[i][k].load().to(fx.Float32), 0.0)
            dst = fx.make_view(fx.get_iter(out_buf) + col, fx.make_layout(8, 1))
            of = fx.make_fragment_like(dst)
            of.store(acc.to(fx.BFloat16))
            fx.copy(copy_atom_b, of, dst)

    @flyc.jit
    def launch(p_xq: fx.Pointer, p_w13: fx.Pointer, p_sorted_ids: fx.Pointer,
               p_sorted_weights: fx.Pointer, p_seid: fx.Pointer, p_nv: fx.Pointer,
               p_s13: fx.Pointer, p_xscale: fx.Pointer, M: fx.Int32,
               sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, i64_h: fx.Int64, i64_hscale: fx.Int64,
               p_h: fx.Pointer, p_hscale: fx.Pointer, p_w2: fx.Pointer, p_s2: fx.Pointer,
               p_gemm2_out: fx.Pointer, p_ctl: fx.Pointer, p_locA: fx.Pointer, p_outA: fx.Pointer,
               p_locB: fx.Pointer, p_outB: fx.Pointer, nbA: fx.Int32, MA: fx.Int32, MB: fx.Int32,
               stream: fx.Stream):
        mega_moe_prefill_2e(p_xq, p_w13, p_sorted_ids, p_sorted_weights, p_seid, p_nv, p_s13,
                         p_xscale, M, sb, sbr, slb, slbr, alim, i64_h, i64_hscale, p_h, p_hscale, p_w2, p_s2,
                         p_gemm2_out, p_ctl, p_locA, p_outA, p_locB, p_outB, nbA, MA, MB,
                         value_attrs={"rocdl.waves_per_eu": 1},
                         ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch

def arith_select(c, a, b):
    return fx.Int32(fx.Boolean(c).select(fx.Int32(a), fx.Int32(b))) if not isinstance(c, bool) else (a if c else b)


def arith_min(a, b):
    return arith_select(fx.Int32(a) < fx.Int32(b), a, b)


def arith_max(a, b):
    return arith_select(fx.Int32(a) > fx.Int32(b), a, b)


@flyc.jit
def _wait_ge(addr_i64, target, err_addr_i64):
    cur = _load_volatile(addr_i64)
    spins = fx.Int32(0)
    while (cur < target) & (spins < _SPIN_MAX):
        rocdl.s_sleep(1)
        spins = spins + 1
        cur = _load_volatile(addr_i64)
    if spins >= _SPIN_MAX:
        _atomic_add(err_addr_i64, 1)
    _fence(llvm.AtomicOrdering.acquire)


@functools.lru_cache(maxsize=8)
def build_mega_prefill_seg(*, model_dim, inter_dim, topk, experts, grid=80, store_cache=18, dbg=0):
    """Step 2: phases per WG, every task of a phase from this XCD's queue.

    G  : gate-up of the full-sort blocks of this XCD's experts (h stays in this XCD's L2)
    DA : down of segment A (the tokens pushed to the peer) for the same experts
    SA : top-k sum of segment A tokens (static token range per WG), after every DA landed
    DB, SB : the same for segment B (this rank's own tokens)
    plan[x] = [g0, g1, a0, a1, b0, b1] (block ranges per XCD, expert-aligned).
    ctl bytes: 0+4x G tickets, 16+4x DA tickets, 32+4x DB tickets, 64+4x G done per XCD,
    128 DA done, 132 DB done, 200 spin errors.
    """
    N1 = 2 * inter_dim
    gu = compile_moe_gemm1(
        N=N1, K=model_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=N1, tile_k=128, alg="prefill_1x4", E=experts,
        act_quant_type="ptpc", activation="silu", METADATA_TILE_SIZE_M=64, FUSE_QUANT=True,
        WAVES_PER_EU=2 if N1 <= 384 else 0, XCD_GROUP=0,
    )
    dn = _build_moe_gemm2_1x4(
        N=model_dim, K=inter_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256, stage="down", alg="prefill_1x4", E=experts,
        USE_ATOMIC_WRITE=False, act_quant_type="ptpc", down_path="1x4_64x256",
        down_output_padding_bytes=0, METADATA_TILE_SIZE_M=64, _store_cache=store_cache,
    )
    gu_block, GuStorage, dn_block = gu.block, gu.storage, dn.block
    K = model_dim
    FP8 = fx.Float8E4M3FNUZ
    H = model_dim

    def _sum_tokens(chunk, tid, M, p_loc, p_g2, p_out):
        lane = tid % 64
        tok = chunk * 4 + tid // 64
        tok_c = arith_min(tok, M - 1)
        locs = [p_loc[tok_c * topk + k] for k in range(topk)]
        valid = [loc >= 0 for loc in locs]
        g2 = fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_g2))
        rows = [g2 + fx.Int64(valid[k].select(locs[k], 0)) * H for k in range(topk)]
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
        copy_atom_b = fx.make_copy_atom(fx.rocdl.BufferCopy(128), fx.BFloat16)
        out_row = fx.make_view(fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_out)) + fx.Int64(tok) * H,
                               fx.make_layout(H, 1))
        out_buf = fx.rocdl.make_buffer_tensor(
            out_row, max_size=False,
            num_records_bytes=(tok < M).select(fx.Int32(H * 2), fx.Int32(0)))
        n_chunk = H // 8 // 64
        frags = []
        for i in range(n_chunk):
            col = (lane + 64 * i) * 8
            row_frags = []
            for k in range(topk):
                atom = fx.make_view(rows[k] + col, fx.make_layout(8, 1))
                frag = fx.make_fragment_like(atom)
                fx.copy(copy_atom, atom, frag)
                row_frags.append(frag)
            frags.append(row_frags)
        for i in range(n_chunk):
            col = (lane + 64 * i) * 8
            acc = valid[0].select(frags[i][0].load().to(fx.Float32), 0.0)
            for k in range(1, topk):
                acc += valid[k].select(frags[i][k].load().to(fx.Float32), 0.0)
            dst = fx.make_view(fx.get_iter(out_buf) + col, fx.make_layout(8, 1))
            of = fx.make_fragment_like(dst)
            of.store(acc.to(fx.BFloat16))
            fx.copy(copy_atom_b, of, dst)

    SA_CHUNKS = 8  # 4-token chunks per SA task

    @flyc.kernel(known_block_size=[256, 1, 1])
    def mega_moe_prefill_seg(
        p_segtab: fx.Pointer, p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer,
        p_sw: fx.Pointer, p_seid: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
        M: fx.Int32, MA: fx.Int32, MB: fx.Int32,
        sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32, alim: fx.Float32,
        i64_h: fx.Int64, i64_hscale: fx.Int64,
    ):
        # Every pointer comes from p_segtab (int64 addresses), read where it is used: few
        # kernel-argument SGPRs stay live across the task loop.
        # One ticket loop (queue per XCD): [G][DA][SA][DB], then SB with a static split.
        # Completion counts are published when a WG moves on to another task kind.
        tid = gpu.thread_idx.x
        sa = fx.SharedAllocator()
        lds = sa.allocate(GuStorage)
        tick = sa.allocate(fx.Array[fx.Int32, 4, 16]).peek()
        tick_v = tick.view(fx.make_layout(4, 1))
        lds_base = fx.recast_iter(fx.Int8, lds.sorted_lds.peek().ptr)
        tab = fx.Int64(fx.ptrtoint(p_segtab))
        ctl = fx.Int64(_load_i64(tab + fx.Int64(25 * 8)))
        p_plan = _seg_ptr(tab, 26, T.i32)

        def seg(si, j, ty, al=4):
            return _seg_ptr(tab, si * 8 + j, ty, al)
        xcd = _xcc_id() & (_XCD - 1)
        g0 = _rfl(p_plan[xcd * 8 + 0])
        nG = _rfl(p_plan[xcd * 8 + 1]) - g0
        a0 = _rfl(p_plan[xcd * 8 + 2])
        nDA = _rfl(p_plan[xcd * 8 + 3]) - a0
        b0 = _rfl(p_plan[xcd * 8 + 4])
        nDB = _rfl(p_plan[xcd * 8 + 5]) - b0
        totA = _rfl(p_plan[(_XCD - 1) * 8 + 3])
        totB = _rfl(p_plan[(_XCD - 1) * 8 + 5])
        sa_tasks = ((MA + 3) // 4 + SA_CHUNKS - 1) // SA_CHUNKS
        sb_tasks = ((MB + 3) // 4 + SA_CHUNKS - 1) // SA_CHUNKS
        nSA = (sa_tasks - xcd + _XCD - 1) // _XCD
        nSB = (sb_tasks - xcd + _XCD - 1) // _XCD
        # sections of this XCD's queue: 0 G | 1 DA | 2 SA | 3 DB | 4 SB
        e_g = nG
        e_a = e_g + nDA
        e_s = e_a + nSA
        e_b = e_s + nDB
        n_tot = e_b + nSB
        g_done = ctl + fx.Int64(64 + xcd * 4)
        a_done = ctl + fx.Int64(128)
        b_done = ctl + fx.Int64(132)

        if tid == 0:
            tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
        _wg_sync()
        t = tick_v[0]
        pend_g = fx.Int32(0)
        pend_a = fx.Int32(0)
        pend_b = fx.Int32(0)
        prev_sec = fx.Int32(0)
        iters = fx.Int32(0)
        while (t < n_tot) & (iters < 1000000):
            iters = iters + 1
            sec = arith_select(t < e_g, 0, arith_select(t < e_a, 1, arith_select(
                t < e_s, 2, arith_select(t < e_b, 3, 4))))
            if (sec != prev_sec) & ((pend_g + pend_a + pend_b) > 0):
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == 0:
                    _fence(llvm.AtomicOrdering.release)
                    _atomic_add(g_done, pend_g)
                    _atomic_add(a_done, pend_a)
                    _atomic_add(b_done, pend_b)
                pend_g = pend_g * 0
                pend_a = pend_a * 0
                pend_b = pend_b * 0
            prev_sec = sec
            if t < e_g:
                xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)),
                                       fx.make_layout((M, K), (K, 1)))
                w13_it = fx.recast_iter(FP8, fxh._as_ptr(p_w13))
                if const_expr((dbg & 1) == 0):
                    gblk = g0 + t
                    if const_expr(dbg & 512):
                        gblk = (t // _GROUP) * (_XCD * _GROUP) + xcd * _GROUP + t % _GROUP
                    gu_block(lds, gblk, 0, tid, M, xq_view, w13_it, p_xq, p_sid, p_sw, p_seid,
                             p_s13, p_xscale, sb, sbr, slb, slbr, alim, i64_h, i64_hscale)
                pend_g = pend_g + 1
            if t >= e_g:
                # one wait site: D waits for this XCD's gate-ups, S for every down of its segment
                is_d = (sec == 1) | (sec == 3)
                w_addr = fx.Int64(arith_select(is_d, 64 + xcd * 4, arith_select(sec == 2, 128, 132)))
                w_val = arith_select(is_d, nG, arith_select(sec == 2, totA, totB))
                if tid < 64:
                    cw = _load_volatile(ctl + w_addr)
                    spw = fx.Int32(0)
                    while (cw < w_val) & (spw < 200000):
                        spw = spw + 1
                        rocdl.s_sleep(1)
                        cw = _load_volatile(ctl + w_addr)
                    _fence(llvm.AtomicOrdering.acquire)
                gpu.barrier()
                si = arith_select(sec < 3, 0, 1)
                Ms = arith_select(sec < 3, MA, MB)
                if is_d & ((dbg & 128) == 0):
                    blk = arith_select(sec == 1, a0 + t - e_g, b0 + t - e_s)
                    if const_expr((dbg & 16) == 0):
                      dn_block(blk, seg(si, 4, FP8.ir_type, 16), seg(2, 7, FP8.ir_type, 16),
                             seg(si, 6, T.bf16, 16), seg(si, 0, T.i32), seg(si, 1, T.f32),
                             seg(si, 2, T.i32), seg(3, 0, T.f32), seg(si, 5, T.f32), Ms,
                             _carver(lds_base))
                    pend_a = pend_a + arith_select(sec == 1, 1, 0)
                    pend_b = pend_b + arith_select(sec == 3, 1, 0)
                if (is_d == False) & ((dbg & 256) == 0):  # noqa: E712
                    task = (t - arith_select(sec == 2, e_a, e_b)) * _XCD + xcd
                    for j in range(SA_CHUNKS if (dbg & 2) == 0 else 0):
                        _sum_tokens(task * SA_CHUNKS + j, tid, Ms, seg(si, 3, T.i32),
                                    seg(si, 6, T.bf16, 16), seg(si, 7, T.bf16, 16))
            _wg_sync()
            if tid == 0:
                tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
            _wg_sync()
            t = tick_v[0]

        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            _fence(llvm.AtomicOrdering.release)
            _atomic_add(g_done, pend_g)
            _atomic_add(a_done, pend_a)
            _atomic_add(b_done, pend_b)
            wgi = fx.Int32(gpu.block_idx.x)
            _atomic_add(ctl + fx.Int64(4096 + wgi * 16), iters)
            _atomic_add(ctl + fx.Int64(4100 + wgi * 16), t)
            _atomic_add(ctl + fx.Int64(4104 + wgi * 16), n_tot)
            _atomic_add(ctl + fx.Int64(4108 + wgi * 16), prev_sec)

    @flyc.jit
    def launch(p_segtab: fx.Pointer, p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer,
               p_sw: fx.Pointer, p_seid: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
               M: fx.Int32, MA: fx.Int32, MB: fx.Int32,
               sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, i64_h: fx.Int64, i64_hscale: fx.Int64, stream: fx.Stream):
        mega_moe_prefill_seg(
            p_segtab, p_xq, p_w13, p_sid, p_sw, p_seid, p_s13, p_xscale, M, MA, MB, sb, sbr, slb, slbr, alim, i64_h, i64_hscale,
            value_attrs={"rocdl.waves_per_eu": 1},
        ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


@functools.lru_cache(maxsize=8)
def build_mega_prefill_v3(*, model_dim, inter_dim, topk, experts, grid=160, gu_bn=128,
                          push=False, prof=False, fuse_xq=False, fuse_sort=False):
    """Persistent prefill MoE, 2 WGs per CU (all bodies <= 256 VGPR, <= 30 KB LDS).

    Per XCD (the blocks b with (b // 8) % 4 == xcd) the task sequence is
    [gate-up of each block (all N tiles, bf16 h)] [down of each block (per-row fp8 quant of
    its h rows, then down 1x4)], statically strided over the XCD's WGs. Blocks [0, nbA)
    carry segment A (tokens whose partial sums go to the peer; sort experts with ids
    e + E * own so those blocks come first). Once a WG is past its queue's last A down it
    sums its share of the A tokens (all A downs landed); with push, the last WG to finish
    posts one SDMA put of the A sums to the peer. After the loop: B sums (own tokens).
    h rows and their scales are padded to whole cache lines: blocks never share a line.
    par (int32): [nbA, MA, MB, peer, queue, _, dev_comm lo/hi, dst lo/hi, src lo/hi, ...],
    then from byte 64 int64 addresses [locA, outA, locB, outB].
    ctl (int32, zero): [xcd] rank counters, [4] registered WGs, [8] A downs, [9] B downs,
    [10] finished A sums, [16 + b] gate-up done of block b.
    """
    import os as _os

    N1 = 2 * inter_dim
    NT = N1 // gu_bn
    H = model_dim
    K = model_dim
    H_STRIDE = (inter_dim + 127) // 128 * 128
    HS_STRIDE = 32
    _os.environ["MOE_GATEUP_MIN_LDS"] = str(30 * 1024)
    gu = compile_moe_gemm1(
        N=N1, K=model_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=gu_bn, tile_k=128, alg="prefill_1x4", E=experts,
        act_quant_type="ptpc", activation="silu", METADATA_TILE_SIZE_M=64, FUSE_QUANT=False,
        XCD_GROUP=0,
    )
    _os.environ.pop("MOE_GATEUP_MIN_LDS", None)
    _os.environ["MOE_DOWN_A_STRIDE"] = str(H_STRIDE)
    _os.environ["MOE_DOWN_AS_STRIDE"] = str(HS_STRIDE)
    dn = _build_moe_gemm2_1x4(
        N=model_dim, K=inter_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256, stage="down", alg="prefill_1x4", E=experts,
        USE_ATOMIC_WRITE=False, act_quant_type="ptpc", down_path="1x4_64x256",
        down_output_padding_bytes=0, METADATA_TILE_SIZE_M=64, _store_cache=2,
    )
    _os.environ.pop("MOE_DOWN_A_STRIDE", None)
    _os.environ.pop("MOE_DOWN_AS_STRIDE", None)
    gu_block, GuStorage, dn_block = gu.block, gu.storage, dn.block
    FP8 = fx.Float8E4M3FNUZ
    if push:
        import mori.cco.device.flydsl as cco

    def _quant_block(b, tid, M, p_sid, p_hbf, i64_q, i64_qs):
        inter = inter_dim
        quarter = tid % 4
        row = tid // 4
        sid = fx.Int32(p_sid[b * 64 + row])
        token = sid & fx.Int32(0xFFFFFF)
        slot = sid >> fx.Int32(24)
        valid = token < M
        route = valid.select(token * topk + slot, fx.Int32(0))
        hb_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(p_hbf)), num_records_bytes=fx.Int64(M) * fx.Int64(topk * inter * 2))
        q_rsrc = buffer_ops.create_buffer_resource_from_addr(
            i64_q, num_records_bytes=fx.Int64(M) * fx.Int64(topk * H_STRIDE))
        s_rsrc = buffer_ops.create_buffer_resource_from_addr(
            i64_qs, num_records_bytes=fx.Int64(M) * fx.Int64(topk * 4 * HS_STRIDE))
        f32 = T.f32
        vals = []
        amax = fx.Float32(0.0)
        for sl in range(inter // 32):
            off = (route * inter + sl * 32 + quarter * 8) // 2
            w4 = Vec(buffer_ops.buffer_load(hb_rsrc, off, vec_width=4, dtype=T.i32))
            v8 = w4.bitcast(fx.BFloat16)
            f8 = [v8[j].to(fx.Float32) for j in range(8)]
            for j in range(8):
                amax = amax.maximumf(fx.Float32(llvm.call_intrinsic(f32, "llvm.fabs.f32", [_raw(f8[j])], [], [])))
            vals.append(f8)
        for dist in (1, 2):
            amax = amax.maximumf(amax.shuffle_xor(fx.Int32(dist), fx.Int32(64)))
        scale = amax * fx.Float32(float(1.0 / 240.0))
        inv_scale = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
        buffer_ops.buffer_store(_raw(scale), s_rsrc, route * fx.Int32(4 * HS_STRIDE),
                                mask=_raw((quarter == fx.Int32(0)) & valid), offset_is_bytes=True)
        fm_fast = fx.arith.FastMathFlags.fast
        for sl in range(inter // 32):
            f8 = vals[sl]
            q = [fx.math.clampf(f8[j] * inv_scale, fx.Float32(-240.0), fx.Float32(240.0), fastmath=fm_fast)
                 for j in range(8)]
            words = []
            for wd in range(2):
                lo = rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[4 * wd]), _raw(q[4 * wd + 1]), fx.Int32(0), False)
                words.append(rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[4 * wd + 2]), _raw(q[4 * wd + 3]), lo, True))
            vec2 = vector.from_elements(T.vec(2, T.i32), words)
            buffer_ops.buffer_store(vec2, q_rsrc, route * H_STRIDE + sl * 32 + quarter * 8,
                                    mask=_raw(valid), offset_is_bytes=True)

    def _quant_x_token(tok, lane, M, x_i64, xq_i64, xs_i64):
        # aiter dynamic_per_token_scaled_quant's math (the fused gate-up epilogue's formula):
        # scale = amax(|x|) / 240, q = clamp(x * rcp(scale)); one wave per token.
        valid = tok < M
        tok_c = arith_min(tok, M - 1)
        x_rsrc = buffer_ops.create_buffer_resource_from_addr(
            x_i64, num_records_bytes=fx.Int64(M) * fx.Int64(H * 2))
        q_rsrc = buffer_ops.create_buffer_resource_from_addr(
            xq_i64, num_records_bytes=fx.Int64(M) * fx.Int64(H))
        s_rsrc = buffer_ops.create_buffer_resource_from_addr(
            xs_i64, num_records_bytes=fx.Int64(M) * fx.Int64(4))
        f32 = T.f32
        vals = []
        amax = fx.Float32(0.0)
        for i in range(H // 8 // 64):
            col = (lane + 64 * i) * 8
            w4 = Vec(buffer_ops.buffer_load(x_rsrc, (tok_c * H + col) // 2, vec_width=4, dtype=T.i32))
            v8 = w4.bitcast(fx.BFloat16)
            f8 = [v8[j].to(fx.Float32) for j in range(8)]
            for j in range(8):
                amax = amax.maximumf(fx.Float32(llvm.call_intrinsic(f32, "llvm.fabs.f32", [_raw(f8[j])], [], [])))
            vals.append(f8)
        for dist in (1, 2, 4, 8, 16, 32):
            amax = amax.maximumf(amax.shuffle_xor(fx.Int32(dist), fx.Int32(64)))
        scale = amax * fx.Float32(float(1.0 / 240.0))
        inv_scale = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
        buffer_ops.buffer_store(_raw(scale), s_rsrc, tok_c * fx.Int32(4),
                                mask=_raw((lane == fx.Int32(0)) & valid), cache_modifier=17,
                                offset_is_bytes=True)
        fm_fast = fx.arith.FastMathFlags.fast
        for i in range(H // 8 // 64):
            col = (lane + 64 * i) * 8
            f8 = vals[i]
            q = [fx.math.clampf(f8[j] * inv_scale, fx.Float32(-240.0), fx.Float32(240.0), fastmath=fm_fast)
                 for j in range(8)]
            words = []
            for wd in range(2):
                lo = rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[4 * wd]), _raw(q[4 * wd + 1]), fx.Int32(0), False)
                words.append(rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[4 * wd + 2]), _raw(q[4 * wd + 3]), lo, True))
            vec2 = vector.from_elements(T.vec(2, T.i32), words)
            buffer_ops.buffer_store(vec2, q_rsrc, tok_c * H + col, mask=_raw(valid),
                                    cache_modifier=17, offset_is_bytes=True)

    SORT_HIST, SORT_CUR, SORT_OFF = 81920, 90112, 98304  # ctl byte offsets (2E bins each)
    SORT_META, SORT_SYNC = 106496, 106512  # [nblk, nbA], [sync after S2, sync after S3]

    def _sum_tokens(chunk, tid, M, p_loc, p_g2, p_out, wt):
        # moe_reduce.sorted_sum's math and order; one wave per token, 4 tokens per call.
        lane = tid % 64
        tok = chunk * 4 + tid // 64
        tok_c = arith_min(tok, M - 1)
        locs = [p_loc[tok_c * topk + k] for k in range(topk)]
        valid = [loc >= 0 for loc in locs]
        g2 = fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_g2))
        rows = [g2 + fx.Int64(valid[k].select(locs[k], 0)) * H for k in range(topk)]
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
        copy_atom_b = fx.make_copy_atom(
            fx.rocdl.BufferCopy(128, cache_modifier=17) if wt else fx.rocdl.BufferCopy(128), fx.BFloat16)
        out_row = fx.make_view(fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_out)) + fx.Int64(tok) * H,
                               fx.make_layout(H, 1))
        out_buf = fx.rocdl.make_buffer_tensor(
            out_row, max_size=False,
            num_records_bytes=(tok < M).select(fx.Int32(H * 2), fx.Int32(0)))
        for i in range(H // 8 // 64):
            col = (lane + 64 * i) * 8
            frags_i = []
            for k in range(topk):
                atom = fx.make_view(rows[k] + col, fx.make_layout(8, 1))
                frag = fx.make_fragment_like(atom)
                fx.copy(copy_atom, atom, frag)
                frags_i.append(frag)
            acc = valid[0].select(frags_i[0].load().to(fx.Float32), 0.0)
            for k in range(1, topk):
                acc += valid[k].select(frags_i[k].load().to(fx.Float32), 0.0)
            dst = fx.make_view(fx.get_iter(out_buf) + col, fx.make_layout(8, 1))
            of = fx.make_fragment_like(dst)
            of.store(acc.to(fx.BFloat16))
            fx.copy(copy_atom_b, of, dst)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def mega_moe_prefill_v3(
        p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
        p_seid: fx.Pointer, p_nv: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
        M: fx.Int32, sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
        alim: fx.Float32, p_h: fx.Pointer, p_hs: fx.Pointer,
        p_w2: fx.Pointer, p_s2: fx.Pointer, p_g2: fx.Pointer, p_ctl: fx.Pointer,
        p_hbf: fx.Pointer, p_par: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        i64_h = fx.Int64(fx.ptrtoint(p_h))
        i64_hs = fx.Int64(fx.ptrtoint(p_hs))
        par = fx.Int64(fx.ptrtoint(p_par))

        def par_ptr(i, ty, al=4):
            # pointers used once per phase live in the int64 slots 8.. of par
            return _seg_ptr(par + fx.Int64(64), i, ty, al)

        sa = fx.SharedAllocator()
        tick = sa.allocate(fx.Array[fx.Int32, 4, 16]).peek()
        if const_expr(fuse_sort):
            scan_v = sa.allocate(fx.Array[fx.Int32, 256, 16]).peek().view(fx.make_layout(256, 1))
        lds = sa.allocate(GuStorage)
        tick_v = tick.view(fx.make_layout(4, 1))
        lds_base = fx.recast_iter(fx.Int8, lds.sorted_lds.peek().ptr)
        ctl = fx.Int64(fx.ptrtoint(p_ctl))
        if const_expr(not fuse_sort):
            nblk = fxh.view_as_torch_tensor(p_nv, (1,), fx.Int32)[0] // 64
        xcd = _xcc_id() & (_XCD - 1)
        MA = _rfl(p_par[1])
        MB = _rfl(p_par[2])
        xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)), fx.make_layout((M, K), (K, 1)))
        w13_it = fx.recast_iter(FP8, fxh._as_ptr(p_w13))
        wg = fx.Int32(gpu.block_idx.x)

        _ts0 = _memtime_lo()
        if const_expr(fuse_xq):
            # phase 0: per-token fp8 quant of x (every WG, one wave per token, written through)
            x_i64 = fx.Int64(_load_i64(par + fx.Int64(64 + 4 * 8)))
            wv = fx.Int32(tid // 64)
            tq0 = wg * 4 + wv
            while tq0 < M:
                _quant_x_token(tq0, tid % 64, M, x_i64, fx.Int64(fx.ptrtoint(p_xq)),
                               fx.Int64(fx.ptrtoint(p_xscale)))
                tq0 = tq0 + grid * 4
            rocdl.s_waitcnt(0)
            gpu.barrier()
        if const_expr(prof):
            if tid == 0:
                _atomic_add(ctl + fx.Int64(65536 + wg * 32 + 7 * 4), _memtime_lo() - _ts0)
        if const_expr(fuse_sort):
            # S1: per-WG LDS histogram of virtual experts e + E * own over a contiguous route
            # chunk, then one global add per non-empty bin (the LDS bins are kept for S3)
            ids_p = par_ptr(5, T.i32)
            own_lo = _rfl(p_par[12])
            own_n = _rfl(p_par[13])
            R = M * topk
            NV = 2 * experts
            lb = fx.Int64(fx.ptrtoint(lds_base))
            vz = tid
            while vz < NV:
                _lds_st(lb, vz * 4, 0)
                vz = vz + 256
            gpu.barrier()
            rc0 = wg * R // grid
            rc1 = (wg + 1) * R // grid
            rr = rc0 + tid
            while rr < rc1:
                tk = rr // topk
                vv = ids_p[rr] + experts * arith_select((tk >= own_lo) & (tk < own_lo + own_n), 1, 0)
                _lds_add(lb, vv * 4, 1)
                rr = rr + 256
            gpu.barrier()
            vz = tid
            while vz < NV:
                ch = _lds_ld(lb, vz * 4)
                if ch > 0:
                    _atomic_add(ctl + fx.Int64(SORT_HIST + vz * 4), ch)
                vz = vz + 256
            rocdl.s_waitcnt(0)
            gpu.barrier()
        # rank within this XCD, then wait for every WG so W (WGs on this XCD) is final (with
        # fuse_xq this is also the grid barrier after phase 0)
        if tid == 0:
            tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
            _atomic_add(ctl + fx.Int64(16), 1)
        if tid < 64:
            creg = _load_volatile(ctl + fx.Int64(16))
            while creg < grid:
                rocdl.s_sleep(1)
                creg = _load_volatile(ctl + fx.Int64(16))
            if tid == 0:
                tick_v[1] = _load_volatile(ctl + fx.Int64(xcd * 4))
        gpu.barrier()
        if const_expr(fuse_xq or fuse_sort):
            _inv_l1_l2()
            gpu.barrier()
        if const_expr(prof):
            if tid == 0:
                _atomic_add(ctl + fx.Int64(65536 + 8192 + wg * 16 + 0), _memtime_lo() - _ts0)
        if const_expr(fuse_sort):
            NV = 2 * experts
            per = (NV + 255) // 256
            # S2 (WG 0): padded block offsets, nblk, nbA
            if wg == 0:
                loc_blocks = fx.Int32(0)
                for jj in range(per):
                    vj = tid * per + jj
                    cj = arith_select(vj < NV, _load_volatile(ctl + fx.Int64(SORT_HIST + vj * 4)), 0)
                    loc_blocks = loc_blocks + (cj + 63) // 64
                scan_v[tid] = loc_blocks
                gpu.barrier()
                if tid == 0:
                    acc = fx.Int32(0)
                    accA = fx.Int32(0)
                    for ii in range(256):
                        cur_ii = scan_v[ii]
                        scan_v[ii] = acc
                        acc = acc + cur_ii
                    _atomic_add(ctl + fx.Int64(SORT_META), acc)
                gpu.barrier()
                run = scan_v[tid]
                nbA_part = fx.Int32(0)
                for jj in range(per):
                    vj = tid * per + jj
                    cj = arith_select(vj < NV, _load_volatile(ctl + fx.Int64(SORT_HIST + vj * 4)), 0)
                    _atomic_add(ctl + fx.Int64(SORT_OFF + vj * 4), arith_select(vj < NV, run, 0))
                    nbA_part = nbA_part + arith_select(vj < experts, (cj + 63) // 64, 0)
                    run = run + (cj + 63) // 64
                _atomic_add(ctl + fx.Int64(SORT_META + 4), nbA_part)
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == 0:
                    _fence(llvm.AtomicOrdering.release)
                    _atomic_add(ctl + fx.Int64(SORT_SYNC), grid)
            if tid < 64:
                cs = _load_volatile(ctl + fx.Int64(SORT_SYNC))
                while cs < grid:
                    rocdl.s_sleep(1)
                    cs = _load_volatile(ctl + fx.Int64(SORT_SYNC))
            gpu.barrier()
            _inv_l1_l2()
            gpu.barrier()
            if const_expr(prof):
                if tid == 0:
                    _atomic_add(ctl + fx.Int64(65536 + 8192 + wg * 16 + 4), _memtime_lo() - _ts0)
            # S2b: block expert ids and padding rows; S3: scatter routes
            sid_w = fx.Int64(fx.ptrtoint(p_sid))
            sw_w = fx.Int64(fx.ptrtoint(p_sw))
            seid_w = fx.Int64(fx.ptrtoint(p_seid))
            loc_w = fx.Int64(_load_i64(par + fx.Int64(64 + 7 * 8)))  # loc base (all routes)
            gt = wg * 256 + tid
            vv2 = gt
            while vv2 < NV:
                c2 = _load_volatile(ctl + fx.Int64(SORT_HIST + vv2 * 4))
                o2 = _load_volatile(ctl + fx.Int64(SORT_OFF + vv2 * 4))
                nb2 = (c2 + 63) // 64
                jb = fx.Int32(0)
                while jb < nb2:
                    _store_wt(seid_w + fx.Int64((o2 + jb) * 4), vv2 % experts)
                    jb = jb + 1
                pp = o2 * 64 + c2
                while pp < (o2 + nb2) * 64:
                    _store_wt(sid_w + fx.Int64(pp * 4), M | (topk << 24))
                    _store_wt(sw_w + fx.Int64(pp * 4), 0)
                    pp = pp + 1
                vv2 = vv2 + grid * 256
            w_p = par_ptr(6, T.f32)
            # S3: reserve this WG's range per expert (one global atomic per non-empty bin),
            # then place each route with an LDS cursor
            vz = tid
            while vz < NV:
                ch = _lds_ld(lb, vz * 4)
                if ch > 0:
                    o3 = _load_volatile(ctl + fx.Int64(SORT_OFF + vz * 4))
                    _lds_st(lb, vz * 4, o3 * 64 + _atomic_add(ctl + fx.Int64(SORT_CUR + vz * 4), ch))
                vz = vz + 256
            gpu.barrier()
            rr = rc0 + tid
            while rr < rc1:
                tk = rr // topk
                sl = rr - tk * topk
                vv = ids_p[rr] + experts * arith_select((tk >= own_lo) & (tk < own_lo + own_n), 1, 0)
                pos = _lds_add(lb, vv * 4, 1)
                _store_wt(sid_w + fx.Int64(pos * 4), tk | (sl << 24))
                _store_wt(sw_w + fx.Int64(pos * 4), fx.Int32(w_p[rr].bitcast(fx.Int32)))
                _store_wt(loc_w + fx.Int64(rr * 4), pos)
                rr = rr + 256
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tid == 0:
                _fence(llvm.AtomicOrdering.release)
                _atomic_add(ctl + fx.Int64(SORT_SYNC + 4), 1)
            if tid < 64:
                cs2 = _load_volatile(ctl + fx.Int64(SORT_SYNC + 4))
                while cs2 < grid:
                    rocdl.s_sleep(1)
                    cs2 = _load_volatile(ctl + fx.Int64(SORT_SYNC + 4))
            gpu.barrier()
            _inv_l1_l2()
            gpu.barrier()
            if const_expr(prof):
                if tid == 0:
                    _atomic_add(ctl + fx.Int64(65536 + 8192 + wg * 16 + 8), _memtime_lo() - _ts0)
            nblk = _rfl(_load_volatile(ctl + fx.Int64(SORT_META)))
            nbA = _rfl(_load_volatile(ctl + fx.Int64(SORT_META + 4)))
        else:
            nbA = _rfl(p_par[0])
        full_groups = nblk // (_XCD * _GROUP)
        rem = nblk - full_groups * (_XCD * _GROUP)
        n_x = full_groups * _GROUP + fx.Int32(arith_min(arith_max(rem - xcd * _GROUP, 0), _GROUP))
        fgA = nbA // (_XCD * _GROUP)
        remA = nbA - fgA * (_XCD * _GROUP)
        nA_x = fgA * _GROUP + fx.Int32(arith_min(arith_max(remA - xcd * _GROUP, 0), _GROUP))
        n_tasks = 2 * n_x
        T_A = arith_select(nA_x == 0, -1, n_x + nA_x - 1)  # this queue's last A down
        t = _rfl(tick_v[0])
        W = _rfl(tick_v[1])
        _tk = _memtime_lo()
        n_dnA = fx.Int32(0)
        n_dnB = fx.Int32(0)
        sa_done = fx.Int32(0)
        iters = fx.Int32(0)
        while ((t < n_tasks) | (sa_done == 0)) & (iters < 1000000):
            iters = iters + 1
            _t0 = _memtime_lo()
            if ((t > T_A) | (t >= n_tasks)) & (sa_done == 0):
                # every A task of this queue is taken: publish A downs, wait for all, sum A
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == 0:
                    _fence(llvm.AtomicOrdering.release)
                    _atomic_add(ctl + fx.Int64(32), n_dnA)
                n_dnA = n_dnA * 0
                if tid < 64:
                    cA = _load_volatile(ctl + fx.Int64(32))
                    while cA < nbA:
                        rocdl.s_sleep(1)
                        cA = _load_volatile(ctl + fx.Int64(32))
                    _fence(llvm.AtomicOrdering.acquire)
                gpu.barrier()
                _inv_l1_l2()
                gpu.barrier()
                if const_expr(prof):
                    if tid == 0:
                        _atomic_add(ctl + fx.Int64(65536 + wg * 32 + 2 * 4), _memtime_lo() - _t0)
                _t1 = _memtime_lo()
                chA = (MA + 3) // 4
                ca = wg * chA // grid
                ca_end = (wg + 1) * chA // grid
                while ca < ca_end:
                    _sum_tokens(ca, tid, MA, par_ptr(0, T.i32), p_g2, par_ptr(1, T.bf16, 16), push)
                    ca = ca + 1
                if const_expr(push):
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    if tid == 0:
                        seen = _atomic_add(ctl + fx.Int64(40), 1)
                        if seen == grid - 1:
                            dev_comm = (fx.Int64(p_par[7]) << 32) | (fx.Int64(p_par[6]) & 0xFFFFFFFF)
                            dst = (fx.Int64(p_par[9]) << 32) | (fx.Int64(p_par[8]) & 0xFFFFFFFF)
                            src = (fx.Int64(p_par[11]) << 32) | (fx.Int64(p_par[10]) & 0xFFFFFFFF)
                            sdma = cco.DevComm(dev_comm).sdma()
                            sdma.put_ptr(p_par[3], dst, src, fx.Int64(MA) * fx.Int64(H * 2),
                                         p_par[4], remote_signal=True)
                if const_expr(prof):
                    if tid == 0:
                        _atomic_add(ctl + fx.Int64(65536 + wg * 32 + 3 * 4), _memtime_lo() - _t1)
                sa_done = sa_done + 1
            _t2 = _memtime_lo()
            if t < n_tasks:
                is_down = t >= n_x
                loc = arith_select(is_down, t - n_x, t)
                b = (loc // _GROUP) * (_XCD * _GROUP) + xcd * _GROUP + loc % _GROUP
                if is_down:
                    if tid < 64:
                        cg = _load_volatile(ctl + fx.Int64(64 + b * 4))
                        while cg < 1:
                            rocdl.s_sleep(1)
                            cg = _load_volatile(ctl + fx.Int64(64 + b * 4))
                    gpu.barrier()
                    _quant_block(b, tid, M, p_sid, p_hbf, i64_h, i64_hs)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    dn_block(b, p_h, p_w2, p_g2, p_sid, p_sw, p_seid, p_s2, p_hs, M,
                             _carver(lds_base))
                    if const_expr(prof):
                        if tid == 0:
                            _atomic_add(ctl + fx.Int64(65536 + wg * 32 + 1 * 4), _memtime_lo() - _t2)
                    n_dnA = n_dnA + arith_select(b < nbA, 1, 0)
                    n_dnB = n_dnB + arith_select(b < nbA, 0, 1)
                if is_down == False:  # noqa: E712
                    for tl in range(NT):
                        gu_block(lds, b, tl, tid, M, xq_view, w13_it, p_hbf, p_sid, p_sw, p_seid,
                                 p_s13, p_xscale, sb, sbr, slb, slbr, alim, i64_h, i64_hs,
                                 sync_before_gemm=True)
                        gpu.barrier()
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    if tid == 0:
                        _atomic_add(ctl + fx.Int64(64 + b * 4), 1)
                        if const_expr(prof):
                            _atomic_add(ctl + fx.Int64(65536 + wg * 32 + 0 * 4), _memtime_lo() - _t2)
            gpu.barrier()
            t = t + W

        # B (own tokens): every B down landed, then a static token range per WG
        _t3 = _memtime_lo()
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            _fence(llvm.AtomicOrdering.release)
            _atomic_add(ctl + fx.Int64(36), n_dnB)
        if tid < 64:
            cB = _load_volatile(ctl + fx.Int64(36))
            while cB < nblk - nbA:
                rocdl.s_sleep(1)
                cB = _load_volatile(ctl + fx.Int64(36))
            _fence(llvm.AtomicOrdering.acquire)
        gpu.barrier()
        _inv_l1_l2()
        gpu.barrier()
        if const_expr(prof):
            if tid == 0:
                _atomic_add(ctl + fx.Int64(65536 + wg * 32 + 4 * 4), _memtime_lo() - _t3)
        _t4 = _memtime_lo()
        chB = (MB + 3) // 4
        cb = wg * chB // grid
        cb_end = (wg + 1) * chB // grid
        while cb < cb_end:
            _sum_tokens(cb, tid, MB, par_ptr(2, T.i32), p_g2, par_ptr(3, T.bf16, 16), False)
            cb = cb + 1
        if const_expr(prof):
            if tid == 0:
                _atomic_add(ctl + fx.Int64(65536 + wg * 32 + 5 * 4), _memtime_lo() - _t4)
                _atomic_add(ctl + fx.Int64(65536 + wg * 32 + 6 * 4), _memtime_lo() - _tk)

    @flyc.jit
    def launch(p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
               p_seid: fx.Pointer, p_nv: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
               M: fx.Int32, sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, p_h: fx.Pointer,
               p_hs: fx.Pointer, p_w2: fx.Pointer, p_s2: fx.Pointer, p_g2: fx.Pointer,
               p_ctl: fx.Pointer, p_hbf: fx.Pointer, p_par: fx.Pointer, stream: fx.Stream):
        mega_moe_prefill_v3(
            p_xq, p_w13, p_sid, p_sw, p_seid, p_nv, p_s13, p_xscale, M, sb, sbr, slb, slbr, alim,
            p_h, p_hs, p_w2, p_s2, p_g2, p_ctl, p_hbf, p_par, value_attrs={"rocdl.waves_per_eu": 2},
        ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
