# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Linear Attention APIs."""

from __future__ import annotations


import os
import csv
import torch


from pathlib import Path
from flydsl.runtime.device import get_rocm_arch
from .kernels.gdr_decode import create_vk_gdr_decode_kernel
from .kernels.tensor_shim import get_dtype_str, _run_compiled

__all__ = [
    "flydsl_gdr_decode",
    "flydsl_gdr_mtp",
]


GDR_GLOBAL_CONFIG_MAP = None
GDR_GPU_ARCH = get_rocm_arch()


def get_default_kwargs(
    dtype_str,
    state_dtype_str,
    batch_size,
    seq_length,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    d = {}
    d["NUM_BLOCKS_PER_V_DIM"] = 1
    d["NUM_WARPS"] = 4
    d["WARP_THREADS_K"] = 8
    global GDR_GLOBAL_CONFIG_MAP
    global GDR_GPU_ARCH
    if GDR_GLOBAL_CONFIG_MAP is None:
        _dict = {}
        fname = os.path.join(Path(__file__).resolve().parent, "gdr_decode_tuned.csv")
        with open(fname, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = dict(row)
                arch, b, sq, nkh, nvh, khd, vhd = (
                    obj["arch"],
                    int(obj["b"]),
                    int(obj["sq"]),
                    int(obj["num_k_heads"]),
                    int(obj["num_v_heads"]),
                    int(obj["head_k_dim"]),
                    int(obj["head_v_dim"]),
                )
                d_str, sd_str = obj["dtype"], obj["state_dtype"]
                if float(obj["duration"]) < 10000.0:
                    _dict[(d_str, sd_str, arch, b, sq, nkh, nvh, khd, vhd)] = {
                        "NUM_BLOCKS_PER_V_DIM": int(obj["NUM_BLOCKS_PER_V_DIM"]),
                        "NUM_WARPS": int(obj["NUM_WARPS"]),
                        "WARP_THREADS_K": int(obj["WARP_THREADS_K"]),
                    }
        GDR_GLOBAL_CONFIG_MAP = _dict
    config = GDR_GLOBAL_CONFIG_MAP.get(
        (
            dtype_str,
            state_dtype_str,
            GDR_GPU_ARCH,
            batch_size,
            seq_length,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        ),
        None,
    )
    if config:
        d.update(config)
    return d


def flydsl_gdr_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    stream: torch.cuda.Stream = None,
    min_valid_index: int = 0,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    device = query.device
    dtype = query.dtype
    for input in [query, key, value, a, b, dt_bias, A_log, indices, out]:
        assert input.device == device
    assert state.data_ptr() % 16 == 0
    for input in [key, value, a, b, dt_bias, out]:
        assert input.dtype == dtype
    assert state.dtype in [torch.float, torch.bfloat16]
    assert A_log.dtype in [torch.float, torch.bfloat16]
    assert indices.dtype == torch.int32

    if need_shuffle_state:
        state_ = state.permute(0, 1, 3, 2).contiguous()
    else:
        state_ = state
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]
    kwargs_ = get_default_kwargs(
        str(dtype),
        str(state_.dtype),
        batch_size,
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    _launch_vk_gdr(
        query, key, value, a, b, dt_bias, A_log, indices, indices, state_, out,
        use_qk_l2norm, stream, min_valid_index, kwargs_, mtp_chain=False,
    )
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)


def _nearest_tuned_kwargs(
    dtype_str, state_dtype_str, batch_size, seq_length, num_k_heads, num_v_heads,
    head_k_dim, head_v_dim,
):
    """Tuned kwargs for the closest tuned batch size (same shape/dtype/arch/sq);
    CUDA-graph batch sizes are not all present in gdr_decode_tuned.csv."""
    kwargs_ = get_default_kwargs(
        dtype_str, state_dtype_str, batch_size, seq_length, num_k_heads,
        num_v_heads, head_k_dim, head_v_dim,
    )
    rest = (seq_length, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    tuned = {
        k[3]: v
        for k, v in GDR_GLOBAL_CONFIG_MAP.items()
        if k[:3] == (dtype_str, state_dtype_str, GDR_GPU_ARCH) and k[4:] == rest
    }
    if tuned and batch_size not in tuned:
        kwargs_.update(tuned[min(tuned, key=lambda n: abs(n - batch_size))])
    return kwargs_


def _launch_vk_gdr(
    query, key, value, a, b, dt_bias, A_log, indices, num_accepted, state, out,
    use_qk_l2norm, stream, min_valid_index, kwargs_, mtp_chain,
):
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads, head_v_dim = value.shape[-2], value.shape[-1]
    qkv = (query, key, value)
    # Strided q/k/v views (unit inner stride) are read in place; anything else is
    # made contiguous as before.
    strided = not all(t.is_contiguous() for t in qkv) and all(
        t.stride(-1) == 1 for t in qkv
    )
    qkv = qkv if strided else tuple(t.contiguous() for t in qkv)
    if not mtp_chain:
        indices = indices.contiguous()
    exe = create_vk_gdr_decode_kernel(
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        get_dtype_str(state.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state.stride(),
        a.stride(),
        b.stride(),
        use_qk_l2norm,
        **kwargs_,
        qkv_strides=tuple(t.stride() for t in qkv) if strided else None,
        min_valid_index=min_valid_index,
        mtp_chain=mtp_chain,
        indices_strides=tuple(indices.stride()) if mtp_chain else None,
    )
    with torch.cuda.device(query.device.index):
        _run_compiled(
            exe,
            *qkv,
            a,
            b,
            dt_bias.contiguous(),
            A_log.contiguous(),
            indices,
            state,
            out,
            num_accepted,
            batch_size,
            stream,
        )


def flydsl_gdr_mtp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    stream: torch.cuda.Stream = None,
    min_valid_index: int = 1,
):
    """Gated delta rule over T draft tokens per sequence with vLLM's MTP-verify
    contract (same as vLLM's fused_sigmoid_gating_delta_rule_update with
    num_accepted_tokens): the initial state is read from
    ``ssm_state_indices[n, num_accepted_tokens[n] - 1]`` and the state after
    token t is written to ``ssm_state_indices[n, t]``. Slots below
    ``min_valid_index`` (vLLM's null block 0) are skipped.

    Shapes: query/key [N, T, Hk, K], value [N, T, Hv, V], a/b [N, T, Hv],
    ssm_state_indices [N, T] int32, num_accepted_tokens [N] int32,
    state [slots, Hv, V, K] (VK layout), out [N, T, Hv, V].
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    device = query.device
    for t in (key, value, a, b, dt_bias, A_log, ssm_state_indices, num_accepted_tokens, state, out):
        assert t.device == device
    assert state.data_ptr() % 16 == 0 and state.dtype in (torch.float, torch.bfloat16)
    assert ssm_state_indices.dtype == torch.int32 and ssm_state_indices.dim() == 2
    assert num_accepted_tokens.dtype == torch.int32
    assert ssm_state_indices.shape == query.shape[:2]
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    kwargs_ = _nearest_tuned_kwargs(
        str(query.dtype), str(state.dtype), batch_size, seq_length, num_k_heads,
        value.shape[-2], head_k_dim, value.shape[-1],
    )
    _launch_vk_gdr(
        query, key, value, a, b, dt_bias, A_log, ssm_state_indices,
        num_accepted_tokens.contiguous(), state, out, use_qk_l2norm, stream,
        min_valid_index, kwargs_, mtp_chain=True,
    )
