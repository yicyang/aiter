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
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T
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
        p_segtab: fx.Pointer, M: fx.Int32, MA: fx.Int32, MB: fx.Int32,
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
        gpu.barrier()
        t = _rfl(tick_v[0])
        pend_g = fx.Int32(0)
        pend_a = fx.Int32(0)
        pend_b = fx.Int32(0)
        prev_sec = fx.Int32(0)
        while t < n_tot:
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
            if sec == 0:
                p_xq = seg(2, 0, FP8.ir_type, 16)
                xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)),
                                       fx.make_layout((M, K), (K, 1)))
                w13_it = fx.recast_iter(FP8, fxh._as_ptr(seg(2, 1, FP8.ir_type, 16)))
                gu_block(lds, g0 + t, 0, tid, M, xq_view, w13_it, p_xq, seg(2, 2, T.i32),
                         seg(2, 3, T.f32), seg(2, 4, T.i32), seg(2, 5, T.f32),
                         seg(2, 6, T.f32), sb, sbr, slb, slbr, alim, i64_h, i64_hscale)
                pend_g = pend_g + 1
            if sec != 0:
                # one wait site: D waits for this XCD's gate-ups, S for every down of its segment
                is_d = (sec == 1) | (sec == 3)
                w_addr = fx.Int64(arith_select(is_d, 64 + xcd * 4, arith_select(sec == 2, 128, 132)))
                w_val = arith_select(is_d, nG, arith_select(sec == 2, totA, totB))
                if tid < 64:
                    cw = _load_volatile(ctl + w_addr)
                    spw = fx.Int32(0)
                    while (cw < w_val) & (spw < 4000000):
                        spw = spw + 1
                        rocdl.s_sleep(1)
                        cw = _load_volatile(ctl + w_addr)
                    _fence(llvm.AtomicOrdering.acquire)
                gpu.barrier()
                si = arith_select(sec < 3, 0, 1)
                Ms = arith_select(sec < 3, MA, MB)
                if is_d:
                    blk = arith_select(sec == 1, a0 + t - e_g, b0 + t - e_s)
                    dn_block(blk, seg(si, 4, FP8.ir_type, 16), seg(2, 7, FP8.ir_type, 16),
                             seg(si, 6, T.bf16, 16), seg(si, 0, T.i32), seg(si, 1, T.f32),
                             seg(si, 2, T.i32), seg(3, 0, T.f32), seg(si, 5, T.f32), Ms,
                             _carver(lds_base))
                    pend_a = pend_a + arith_select(sec == 1, 1, 0)
                    pend_b = pend_b + arith_select(sec == 3, 1, 0)
                if is_d == False:  # noqa: E712
                    task = (t - arith_select(sec == 2, e_a, e_b)) * _XCD + xcd
                    for j in range(SA_CHUNKS):
                        _sum_tokens(task * SA_CHUNKS + j, tid, Ms, seg(si, 3, T.i32),
                                    seg(si, 6, T.bf16, 16), seg(si, 7, T.bf16, 16))
            gpu.barrier()
            if tid == 0:
                tick_v[0] = _atomic_add(ctl + fx.Int64(xcd * 4), 1)
            gpu.barrier()
            t = _rfl(tick_v[0])

        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            _fence(llvm.AtomicOrdering.release)
            _atomic_add(g_done, pend_g)
            _atomic_add(a_done, pend_a)
            _atomic_add(b_done, pend_b)

    @flyc.jit
    def launch(p_segtab: fx.Pointer, M: fx.Int32, MA: fx.Int32, MB: fx.Int32,
               sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, i64_h: fx.Int64, i64_hscale: fx.Int64, stream: fx.Stream):
        mega_moe_prefill_seg(
            p_segtab, M, MA, MB, sb, sbr, slb, slbr, alim, i64_h, i64_hscale,
            value_attrs={"rocdl.waves_per_eu": 1},
        ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
