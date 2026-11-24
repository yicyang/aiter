"""Probes for the 8-wave megamoe layout (experiments, not production).

build_gu_probe: persistent gate-up (BN128, bf16 h, no fused quant) over every block of a
sort, in one of three shapes:
  "2wg"  256 threads, 2 WGs per CU (LDS padded to 30 KB), like mega v3
  "4wg"  256 threads, LDS 16 KB (occupancy limited by VGPR only)
  "8w"   512 threads, 1 WG per CU; waves 0-3 and 4-7 are two groups running gate-up of two
         different blocks in lockstep (gate-up barriers are workgroup-wide), each group with
         its own 16 KB LDS region
"""
import contextlib
import types

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import gpu, rocdl, const_expr, range_constexpr
from flydsl.expr.primitive import PointerType
from . import common as fxh

from .gemm1 import compile_moe_gemm1
from .mega_prefill import _rfl, arith_select, arith_min


class _LdsPtr:
    def __init__(self, ptr):
        self.ptr = ptr

    def peek(self):
        return self


def _aligned(elem, p, align=16):
    """p reinterpreted as elem* with a stated alignment (runtime offsets drop it to 1)."""
    return fx.recast_iter(PointerType.get(elem.ir_type, p.memspace, align), p)


def _group_lds(base_i8, off_bytes, a_bytes):
    """Duck-typed gemm1 SharedStorage at base + off (off a multiple of 16):
    sorted_lds unioned with a_ping/a_pong."""
    p = base_i8 + off_bytes
    gemm = types.SimpleNamespace(
        a_ping=_LdsPtr(_aligned(fx.Float8E4M3FNUZ, p)),
        a_pong=_LdsPtr(_aligned(fx.Float8E4M3FNUZ, p + a_bytes)),
    )
    gemm.peek = lambda: gemm
    return types.SimpleNamespace(sorted_lds=_LdsPtr(_aligned(fx.Int32, p)), gemm=gemm)


def build_gu_probe(*, model_dim, inter_dim, topk, experts, mode="8w", stagger=0):
    import os as _os

    N1 = 2 * inter_dim
    BN = 128
    NT = N1 // BN
    K = model_dim
    nwg = int(mode[:-2]) if mode.endswith("wg") else 1
    if nwg in (2, 3):
        _os.environ["MOE_GATEUP_MIN_LDS"] = str({2: 30 * 1024, 3: 20 * 1024}[nwg])
    gu = compile_moe_gemm1(
        N=N1, K=model_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=BN, tile_k=128, alg="prefill_1x4", E=experts,
        act_quant_type="ptpc", activation="silu", METADATA_TILE_SIZE_M=64, FUSE_QUANT=False,
        XCD_GROUP=0,
    )
    _os.environ.pop("MOE_GATEUP_MIN_LDS", None)
    gu_block, GuStorage = gu.block, gu.storage
    FP8 = fx.Float8E4M3FNUZ
    A_BYTES = 64 * 128  # one A buffer (BM x TILE_K fp8)
    G_BYTES = 2 * A_BYTES
    threads = 512 if mode.startswith("8w") else 256
    NOLOOP = _os.environ.get("GU_PROBE_NOLOOP") == "1"
    grid = 80 if mode.startswith("8w") else 80 * nwg

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gu_probe(
        p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
        p_seid: fx.Pointer, p_nv: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
        M: fx.Int32, sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
        alim: fx.Float32, p_hbf: fx.Pointer,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        wg = fx.Int32(gpu.block_idx.x)
        nblk = _rfl(fxh.view_as_torch_tensor(p_nv, (1,), fx.Int32)[0] // 64)
        xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)), fx.make_layout((M, K), (K, 1)))
        w13_it = fx.recast_iter(FP8, fxh._as_ptr(p_w13))
        zero64 = fx.Int64(0)
        if const_expr(mode.startswith("8w")):
            group = _rfl(tid // 256)
            gtid = tid % 256
            sa = fx.SharedAllocator()
            raw = sa.allocate(fx.Array[fx.Int8, 2 * G_BYTES, 16]).peek()
            lds = _group_lds(fx.recast_iter(fx.Int8, raw.ptr), group * G_BYTES, A_BYTES)
            b0 = 2 * wg
            for _s in range(stagger):
                if group == 1:
                    gpu.barrier()
            while b0 < nblk:
                b = arith_select(b0 + group < nblk, b0 + group, b0)
                for tl in range(NT):
                    gu_block(lds, b, tl, gtid, M, xq_view, w13_it, p_hbf, p_sid, p_sw, p_seid,
                             p_s13, p_xscale, sb, sbr, slb, slbr, alim, zero64, zero64,
                             sync_before_gemm=True, group_tid=True)
                    gpu.barrier()
                b0 = b0 + 2 * grid
            for _s in range(stagger):
                if group == 0:
                    gpu.barrier()
        else:
            lds = fx.SharedAllocator().allocate(GuStorage)
            b = wg
            if const_expr(NOLOOP):
                for tl in range(NT):
                    gu_block(lds, b, tl, tid, M, xq_view, w13_it, p_hbf, p_sid, p_sw, p_seid,
                             p_s13, p_xscale, sb, sbr, slb, slbr, alim, zero64, zero64,
                             sync_before_gemm=True)
                    gpu.barrier()
            else:
                while b < nblk:
                    for tl in range(NT):
                        gu_block(lds, b, tl, tid, M, xq_view, w13_it, p_hbf, p_sid, p_sw, p_seid,
                                 p_s13, p_xscale, sb, sbr, slb, slbr, alim, zero64, zero64,
                                 sync_before_gemm=True)
                        gpu.barrier()
                    b = b + grid

    @flyc.jit
    def launch(p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
               p_seid: fx.Pointer, p_nv: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
               M: fx.Int32, sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, p_hbf: fx.Pointer, stream: fx.Stream):
        gu_probe(p_xq, p_w13, p_sid, p_sw, p_seid, p_nv, p_s13, p_xscale, M, sb, sbr, slb, slbr,
                 alim, p_hbf, value_attrs={"rocdl.waves_per_eu": 2 if mode.startswith("8w") else nwg},
                 ).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


def build_dn_probe(*, model_dim, inter_dim, topk, experts, nwg=2):
    """Persistent down (1x4 64x256) over every block of a sort, nwg WGs per CU (256 threads),
    LDS carved from one 30 KB array like the mega kernel. MOE_DOWN_PREFETCH_TILE=1 enables
    the full-tile B prefetch (read at build time)."""
    from .gemm2_1x4 import _build_moe_gemm2_1x4
    from .mega_prefill import _carver

    dn = _build_moe_gemm2_1x4(
        N=model_dim, K=inter_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256, stage="down", alg="prefill_1x4", E=experts,
        USE_ATOMIC_WRITE=False, act_quant_type="ptpc", down_path="1x4_64x256",
        down_output_padding_bytes=0, METADATA_TILE_SIZE_M=64, _store_cache=2,
    )
    dn_block = dn.block
    grid = 80 * nwg

    @flyc.kernel(known_block_size=[256, 1, 1])
    def dn_probe(
        p_h: fx.Pointer, p_w2: fx.Pointer, p_g2: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
        p_seid: fx.Pointer, p_nv: fx.Pointer, p_s2: fx.Pointer, p_hs: fx.Pointer, M: fx.Int32,
    ):
        wg = fx.Int32(gpu.block_idx.x)
        nblk = _rfl(fxh.view_as_torch_tensor(p_nv, (1,), fx.Int32)[0] // 64)
        raw = fx.SharedAllocator().allocate(fx.Array[fx.Int8, 30 * 1024, 16]).peek()
        base = fx.recast_iter(fx.Int8, raw.ptr)
        b = wg
        while b < nblk:
            dn_block(b, p_h, p_w2, p_g2, p_sid, p_sw, p_seid, p_s2, p_hs, M, _carver(base))
            gpu.barrier()
            b = b + grid

    @flyc.jit
    def launch(p_h: fx.Pointer, p_w2: fx.Pointer, p_g2: fx.Pointer, p_sid: fx.Pointer,
               p_sw: fx.Pointer, p_seid: fx.Pointer, p_nv: fx.Pointer, p_s2: fx.Pointer,
               p_hs: fx.Pointer, M: fx.Int32, stream: fx.Stream):
        dn_probe(p_h, p_w2, p_g2, p_sid, p_sw, p_seid, p_nv, p_s2, p_hs, M,
                 value_attrs={"rocdl.waves_per_eu": nwg,
                              "passthrough": [["target-features", "-packed-fp32-ops"]]},
                 ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


# ---- software barrier for a 4-wave compute group (role-split WGs) ----
_LGKM0 = 15 | (7 << 4) | (0 << 8) | (3 << 14)  # s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(0)


def _lds_helpers():
    from .mega_prefill import _lds_add, _lds_ld_volatile, _lds_st
    return _lds_add, _lds_ld_volatile, _lds_st


@flyc.jit
def _sw_group_barrier(lb, off, n_waves):
    """Barrier over n_waves waves through one monotonically increasing LDS counter [off]:
    one LDS atomic per wave (lane 0), then spin until the counter reaches the end of this
    round. Same memory semantics as gpu.barrier here (lgkmcnt(0) first)."""
    _lds_add, _lds_ld_volatile, _lds_st = _lds_helpers()
    rocdl.s_waitcnt(_LGKM0)
    lane = fx.Int32(gpu.thread_idx.x) % 64
    old_v = fx.Int32(0)
    if lane == 0:
        old_v = _lds_add(lb, off, 1)
    old = _rfl(old_v)
    target = (old // n_waves + 1) * n_waves
    cur = old + 1
    it = fx.Int32(0)
    while (cur < target) & (it < 16777216):
        cur = _rfl(_lds_ld_volatile(lb, off))
        it = it + 1
    rocdl.s_waitcnt(_LGKM0)


@contextlib.contextmanager
def _group_barrier(lb, off, n_waves=4):
    """While tracing a compute body, route gpu.barrier() to the software group barrier."""
    import flydsl.expr.gpu as _gpum
    saved = _gpum.barrier
    _gpum.barrier = lambda *a, **k: _sw_group_barrier(lb, off, n_waves)
    try:
        yield
    finally:
        _gpum.barrier = saved


def build_gu_role_probe(*, model_dim, inter_dim, topk, experts, hw_mirror=False):
    """8 waves, 1 WG per CU: waves 0-3 run gate-up (BN128) with a software barrier among
    themselves; waves 4-7 are idle helpers (exit at once). Measures the software barrier's cost."""
    import os as _os
    N1 = 2 * inter_dim
    NT = N1 // 128
    K = model_dim
    gu = compile_moe_gemm1(
        N=N1, K=model_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128, tile_k=128, alg="prefill_1x4", E=experts,
        act_quant_type="ptpc", activation="silu", METADATA_TILE_SIZE_M=64, FUSE_QUANT=False,
        XCD_GROUP=0,
    )
    gu_block, GuStorage = gu.block, gu.storage
    FP8 = fx.Float8E4M3FNUZ
    grid = 80

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gu_role_probe(
        p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
        p_seid: fx.Pointer, p_nv: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
        M: fx.Int32, sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
        alim: fx.Float32, p_hbf: fx.Pointer,
    ):
        _lds_add, _lds_ld_volatile, _lds_st = _lds_helpers()
        tid = fx.Int32(gpu.thread_idx.x)
        wg = fx.Int32(gpu.block_idx.x)
        nblk = _rfl(fxh.view_as_torch_tensor(p_nv, (1,), fx.Int32)[0] // 64)
        xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)), fx.make_layout((M, K), (K, 1)))
        w13_it = fx.recast_iter(FP8, fxh._as_ptr(p_w13))
        zero64 = fx.Int64(0)
        sa = fx.SharedAllocator()
        bw = sa.allocate(fx.Array[fx.Int32, 4, 16]).peek()
        lds = sa.allocate(GuStorage)
        lb = fx.Int64(fx.ptrtoint(bw.ptr))
        if tid < 4:
            _lds_st(lb, tid * 4, 0)
        gpu.barrier()  # all 8 waves, before the roles split
        if const_expr(hw_mirror):
            # compute waves keep the hardware barrier; helper waves execute exactly the same
            # number of s_barriers per task (counted while tracing the compute body)
            import flydsl.expr.gpu as _gpum
            nbar = [0]
            _real = _gpum.barrier

            def _counting(*a, **k):
                nbar[0] += 1
                return _real(*a, **k)
            if tid < 256:
                _gpum.barrier = _counting
                try:
                    b = wg
                    while b < nblk:
                        for tl in range(NT):
                            gu_block(lds, b, tl, tid, M, xq_view, w13_it, p_hbf, p_sid, p_sw,
                                     p_seid, p_s13, p_xscale, sb, sbr, slb, slbr, alim, zero64,
                                     zero64, sync_before_gemm=True)
                            gpu.barrier()
                        b = b + grid
                finally:
                    _gpum.barrier = _real
            else:
                b2 = wg
                while b2 < nblk:
                    for tl2 in range(NT):
                        for _q in range_constexpr(nbar[0]):
                            gpu.barrier()
                    b2 = b2 + grid
        else:
          if tid < 256:
            with _group_barrier(lb, 0, 4):
                b = wg
                while b < nblk:
                    for tl in range(NT):
                        gu_block(lds, b, tl, tid, M, xq_view, w13_it, p_hbf, p_sid, p_sw, p_seid,
                                 p_s13, p_xscale, sb, sbr, slb, slbr, alim, zero64, zero64,
                                 sync_before_gemm=True)
                        gpu.barrier()
                    b = b + grid

    @flyc.jit
    def launch(p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
               p_seid: fx.Pointer, p_nv: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
               M: fx.Int32, sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, p_hbf: fx.Pointer, stream: fx.Stream):
        gu_role_probe(p_xq, p_w13, p_sid, p_sw, p_seid, p_nv, p_s13, p_xscale, M, sb, sbr, slb,
                      slbr, alim, p_hbf, value_attrs={"rocdl.waves_per_eu": 2},
                      ).launch(grid=(grid, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


def _sum_tokens_g(chunk, tid, M, p_loc, p_g2, p_out, wt, H, topk):
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
        if const_expr(False and i > 0):
            # keep each slice's 11 row loads (44 VGPRs) from being hoisted together, so the
            # sum phase stays inside the 168-VGPR budget of 3 WGs per CU
            rocdl.sched_barrier(0)
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


def build_gu_role_sum_probe(*, model_dim, inter_dim, topk, experts, mode="rolesum"):
    """8 waves, 1 WG per CU. Waves 0-3: gate-up (hardware barrier). Waves 4-7: top-k sums of
    an independent g2 (one 4-token chunk at the start of each gate-up tile), executing the
    same number of s_barriers as the compute waves. mode="sumonly": the sums alone."""
    N1 = 2 * inter_dim
    NT = N1 // 128
    K = model_dim
    H = model_dim
    gu = compile_moe_gemm1(
        N=N1, K=model_dim, weight_dtype="fp8", weight_quant_type="ptpc", TOPK=topk,
        BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128, tile_k=128, alg="prefill_1x4", E=experts,
        act_quant_type="ptpc", activation="silu", METADATA_TILE_SIZE_M=64, FUSE_QUANT=False,
        XCD_GROUP=0,
    )
    gu_block, GuStorage = gu.block, gu.storage
    FP8 = fx.Float8E4M3FNUZ
    grid = 80

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gu_role_sum_probe(
        p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
        p_seid: fx.Pointer, p_nv: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
        M: fx.Int32, sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
        alim: fx.Float32, p_hbf: fx.Pointer, p_loc: fx.Pointer, p_g2: fx.Pointer,
        p_out: fx.Pointer,
    ):
        import flydsl.expr.gpu as _gpum
        tid = fx.Int32(gpu.thread_idx.x)
        wg = fx.Int32(gpu.block_idx.x)
        nblk = _rfl(fxh.view_as_torch_tensor(p_nv, (1,), fx.Int32)[0] // 64)
        xq_view = fx.make_view(fx.recast_iter(FP8, fxh._as_ptr(p_xq)), fx.make_layout((M, K), (K, 1)))
        w13_it = fx.recast_iter(FP8, fxh._as_ptr(p_w13))
        zero64 = fx.Int64(0)
        lds = fx.SharedAllocator().allocate(GuStorage)
        n_ch = (M + 3) // 4
        ch0 = wg * n_ch // grid
        ch1 = (wg + 1) * n_ch // grid
        if const_expr(mode == "sumonly"):
            if tid >= 256:
                ch = ch0
                while ch < ch1:
                    _sum_tokens_g(ch, tid - 256, M, p_loc, p_g2, p_out, False, H, topk)
                    ch = ch + 1
        else:
            nbar = [0]
            _real = _gpum.barrier

            def _counting(*a, **k):
                nbar[0] += 1
                return _real(*a, **k)
            if tid < 256:
                _gpum.barrier = _counting
                try:
                    b = wg
                    while b < nblk:
                        for tl in range(NT):
                            gu_block(lds, b, tl, tid, M, xq_view, w13_it, p_hbf, p_sid, p_sw,
                                     p_seid, p_s13, p_xscale, sb, sbr, slb, slbr, alim, zero64,
                                     zero64, sync_before_gemm=True)
                            gpu.barrier()
                        b = b + grid
                finally:
                    _gpum.barrier = _real
            elif const_expr(mode == "rolepipe"):
                # one 4-token chunk per gate-up tile, software-pipelined across the tile's
                # barriers: slice j's loads issued after barrier j, summed after barrier j + LAG
                LAG = 3
                NS = H // 8 // 64
                L0 = 2  # stages between the row-index loads and the first slice's loads
                assert nbar[0] >= L0 + NS + LAG, nbar[0]
                hl = (tid - 256) % 64
                hw = (tid - 256) // 64
                ch = ch0
                b2 = wg
                while b2 < nblk:
                    for tl2 in range(NT):
                        tok = arith_select(ch < ch1, ch * 4 + hw, M)
                        pend = {}
                        locs = None
                        for j in range_constexpr(nbar[0]):
                            if const_expr(j == 0):
                                locs = _sum_locs(tok, M, p_loc, topk)
                            if const_expr(L0 <= j < L0 + NS):
                                pend[j - L0] = _sum_issue(locs, hl, p_g2, H, topk, j - L0)
                            if const_expr(L0 + LAG <= j < L0 + NS + LAG):
                                fr, va = pend.pop(j - L0 - LAG)
                                _sum_consume(fr, va, tok, hl, M, p_out, H, topk, j - L0 - LAG)
                            # keep each stage's helper code between its own barriers (the
                            # scheduler otherwise hoists slice loads next to their index loads)
                            rocdl.sched_barrier(0)
                            gpu.barrier()
                            rocdl.sched_barrier(0)
                        ch = ch + 1
                    b2 = b2 + grid
            else:
                ch = ch0
                b2 = wg
                while b2 < nblk:
                    for tl2 in range(NT):
                        if ch < ch1:
                            _sum_tokens_g(ch, tid - 256, M, p_loc, p_g2, p_out, False, H, topk)
                        ch = ch + 1
                        for _q in range_constexpr(nbar[0]):
                            gpu.barrier()
                    b2 = b2 + grid
                while ch < ch1:  # compute waves are done: no more barriers
                    _sum_tokens_g(ch, tid - 256, M, p_loc, p_g2, p_out, False, H, topk)
                    ch = ch + 1

    @flyc.jit
    def launch(p_xq: fx.Pointer, p_w13: fx.Pointer, p_sid: fx.Pointer, p_sw: fx.Pointer,
               p_seid: fx.Pointer, p_nv: fx.Pointer, p_s13: fx.Pointer, p_xscale: fx.Pointer,
               M: fx.Int32, sb: fx.Float32, sbr: fx.Float32, slb: fx.Float32, slbr: fx.Float32,
               alim: fx.Float32, p_hbf: fx.Pointer, p_loc: fx.Pointer, p_g2: fx.Pointer,
               p_out: fx.Pointer, stream: fx.Stream):
        gu_role_sum_probe(p_xq, p_w13, p_sid, p_sw, p_seid, p_nv, p_s13, p_xscale, M, sb, sbr,
                          slb, slbr, alim, p_hbf, p_loc, p_g2, p_out,
                          value_attrs={"rocdl.waves_per_eu": 2},
                          ).launch(grid=(grid, 1, 1), block=(512, 1, 1), stream=stream)

    return launch



def _sum_locs(tok, M, p_loc, topk):
    """Pipelined top-k sum, part 0: the token's topk row indices (issued one stage early).
    Tokens >= M (idle stages) get -1 everywhere, so their slice loads all hit row 0 (L2)."""
    tok_c = arith_min(tok, M - 1)
    return [arith_select(tok < M, p_loc[tok_c * topk + k], -1) for k in range(topk)]


def _sum_issue(locs, lane, p_g2, H, topk, i):
    """Pipelined top-k sum, part 1: load slice i (8 columns per lane) of the topk rows
    (one wave per token). Returns (fragments, valid flags)."""
    valid = [loc >= 0 for loc in locs]
    g2 = fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_g2))
    copy_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
    col = (lane + 64 * i) * 8
    frags = []
    for k in range(topk):
        row = g2 + fx.Int64(valid[k].select(locs[k], 0)) * H
        atom = fx.make_view(row + col, fx.make_layout(8, 1))
        frag = fx.make_fragment_like(atom)
        fx.copy(copy_atom, atom, frag)
        frags.append(frag)
    return frags, valid


def _sum_consume(frags, valid, tok, lane, M, p_out, H, topk, i):
    """Pipelined top-k sum, part 2: add the topk fragments in slot order and store slice i."""
    copy_atom_b = fx.make_copy_atom(fx.rocdl.BufferCopy(128), fx.BFloat16)
    out_row = fx.make_view(fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_out)) + fx.Int64(tok) * H,
                           fx.make_layout(H, 1))
    out_buf = fx.rocdl.make_buffer_tensor(
        out_row, max_size=False, num_records_bytes=(tok < M).select(fx.Int32(H * 2), fx.Int32(0)))
    col = (lane + 64 * i) * 8
    acc = valid[0].select(frags[0].load().to(fx.Float32), 0.0)
    for k in range(1, topk):
        acc += valid[k].select(frags[k].load().to(fx.Float32), 0.0)
    dst = fx.make_view(fx.get_iter(out_buf) + col, fx.make_layout(8, 1))
    of = fx.make_fragment_like(dst)
    of.store(acc.to(fx.BFloat16))
    fx.copy(copy_atom_b, of, dst)



def build_sum_probe(*, hidden, topk, fp8=False, grid=160):
    """Top-k sum of g2 rows over all M tokens (one wave per token, the mega kernel's SA math);
    fp8=True writes sglang fp8 RS records (as the megamoe fp8 push) instead of bf16 rows."""
    import flydsl.expr.math  # noqa: F401
    from flydsl._mlir.dialects import llvm, vector
    from flydsl.expr.typing import T
    from flydsl.expr.utils.arith import _to_raw as _raw

    from .. import buffer_ops

    H = hidden
    REC = H + H // 32 * 2

    def records(acc, lane, tok, M, col, rec_rsrc):
        f32 = T.f32
        vb = acc.to(fx.BFloat16)
        f8 = [vb[j].to(fx.Float32) for j in range(8)]
        amax = fx.Float32(0.0)
        for j in range(8):
            amax = amax.maximumf(fx.Float32(llvm.call_intrinsic(f32, "llvm.fabs.f32", [_raw(f8[j])], [], [])))
        for dist in (1, 2):
            amax = amax.maximumf(amax.shuffle_xor(fx.Int32(dist), fx.Int32(64)))
        scale_b = (amax / fx.Float32(240.0)).to(fx.BFloat16)
        scale = scale_b.to(fx.Float32)
        inv = (scale > fx.Float32(0.0)).select(fx.Float32(1.0) / scale, fx.Float32(0.0))
        q = [fx.math.clampf(f8[j] * inv, fx.Float32(-240.0), fx.Float32(240.0)) for j in range(8)]
        words = []
        for wd in range(2):
            lo = rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[4 * wd]), _raw(q[4 * wd + 1]), fx.Int32(0), False)
            words.append(rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[4 * wd + 2]), _raw(q[4 * wd + 3]), lo, True))
        row = arith_min(tok, M - 1) * REC
        buffer_ops.buffer_store(vector.from_elements(T.vec(2, T.i32), words), rec_rsrc, row + col,
                                mask=_raw(tok < M), cache_modifier=17, offset_is_bytes=True)
        buffer_ops.buffer_store(_raw(scale_b), rec_rsrc, row + H + col // 32 * 2,
                                mask=_raw((tok < M) & (lane % 4 == 0)), cache_modifier=17,
                                offset_is_bytes=True)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def sum_probe(p_loc: fx.Pointer, p_g2: fx.Pointer, p_out: fx.Pointer, p_rec: fx.Pointer, M: fx.Int32):
        tid = fx.Int32(gpu.thread_idx.x)
        wg = fx.Int32(gpu.block_idx.x)
        lane = tid % 64
        n_ch = (M + 3) // 4
        ch = wg * n_ch // grid
        ch1 = (wg + 1) * n_ch // grid
        rec_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(p_rec)), num_records_bytes=fx.Int64(M) * fx.Int64(REC))
        while ch < ch1:
            tok = ch * 4 + tid // 64
            tok_c = arith_min(tok, M - 1)
            locs = [p_loc[tok_c * topk + k] for k in range_constexpr(topk)]
            valid = [loc >= 0 for loc in locs]
            g2 = fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_g2))
            rows = [g2 + fx.Int64(valid[k].select(locs[k], 0)) * H for k in range_constexpr(topk)]
            copy_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
            copy_atom_b = fx.make_copy_atom(fx.rocdl.BufferCopy(128, cache_modifier=17), fx.BFloat16)
            out_row = fx.make_view(fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_out)) + fx.Int64(tok) * H,
                                   fx.make_layout(H, 1))
            out_buf = fx.rocdl.make_buffer_tensor(
                out_row, max_size=False, num_records_bytes=(tok < M).select(fx.Int32(H * 2), fx.Int32(0)))
            for i in range_constexpr(H // 8 // 64):
                col = (lane + 64 * i) * 8
                frags_i = []
                for k in range_constexpr(topk):
                    atom = fx.make_view(rows[k] + col, fx.make_layout(8, 1))
                    frag = fx.make_fragment_like(atom)
                    fx.copy(copy_atom, atom, frag)
                    frags_i.append(frag)
                acc = valid[0].select(frags_i[0].load().to(fx.Float32), 0.0)
                for k in range_constexpr(1, topk):
                    acc += valid[k].select(frags_i[k].load().to(fx.Float32), 0.0)
                if const_expr(fp8):
                    records(acc, lane, tok, M, col, rec_rsrc)
                else:
                    dst = fx.make_view(fx.get_iter(out_buf) + col, fx.make_layout(8, 1))
                    of = fx.make_fragment_like(dst)
                    of.store(acc.to(fx.BFloat16))
                    fx.copy(copy_atom_b, of, dst)
            ch = ch + 1

    @flyc.jit
    def launch(p_loc: fx.Pointer, p_g2: fx.Pointer, p_out: fx.Pointer, p_rec: fx.Pointer,
               M: fx.Int32, stream: fx.Stream):
        sum_probe(p_loc, p_g2, p_out, p_rec, M, value_attrs={"rocdl.waves_per_eu": 2}).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
