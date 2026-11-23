"""Probes for the 8-wave megamoe layout (experiments, not production).

build_gu_probe: persistent gate-up (BN128, bf16 h, no fused quant) over every block of a
sort, in one of three shapes:
  "2wg"  256 threads, 2 WGs per CU (LDS padded to 30 KB), like mega v3
  "4wg"  256 threads, LDS 16 KB (occupancy limited by VGPR only)
  "8w"   512 threads, 1 WG per CU; waves 0-3 and 4-7 are two groups running gate-up of two
         different blocks in lockstep (gate-up barriers are workgroup-wide), each group with
         its own 16 KB LDS region
"""
import types

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import gpu, rocdl, const_expr
from flydsl.expr.primitive import PointerType
from . import common as fxh

from .gemm1 import compile_moe_gemm1
from .mega_prefill import _rfl, arith_select


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
