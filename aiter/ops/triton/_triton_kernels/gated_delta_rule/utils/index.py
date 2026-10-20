# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Index preparation utilities for variable-length sequence processing.

This module provides functions for preparing various indices needed for
chunk-based and variable-length sequence operations.
"""

import torch
import torch.nn.functional as F
import triton

from ..gated_delta_rule_utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Compute sequence lengths from cumulative sequence lengths."""
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """Compute sequence lengths from a boolean mask."""
    return mask.sum(dim=-1, dtype=torch.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    """Convert sequence lengths to cumulative sequence lengths."""
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    """Convert a boolean mask to cumulative sequence lengths."""
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


@tensor_cache
def prepare_lens_from_cu_seqlens(
    cu_seqlens: torch.LongTensor,
) -> torch.LongTensor:
    """Extract sequence lengths from cumulative sequence lengths."""
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Generate position IDs for each sequence."""
    return torch.cat(
        [
            torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            for n in prepare_lens(cu_seqlens).unbind()
        ]
    )


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Generate sequence IDs indicating which sequence each token belongs to."""
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Generate (sequence_id, position_id) pairs for each token."""
    position_ids = prepare_position_ids(cu_seqlens)
    return torch.stack([prepare_sequence_ids(cu_seqlens), position_ids], 1).to(
        cu_seqlens
    )


def _prefill_lens(
    cu_seqlens: torch.LongTensor,
    num_decodes: int,
) -> torch.LongTensor:
    """Per-sequence token counts for the prefill slice, computed directly
    from the original (cache-stable) ``cu_seqlens`` without materialising an
    intermediate sliced view. Equivalent to
    ``prepare_lens(cu_seqlens[num_decodes:])`` but skips the call to
    ``prepare_lens`` on a sliced tensor (which would miss its own cache).
    """
    if num_decodes == 0:
        return prepare_lens(cu_seqlens)
    return cu_seqlens[num_decodes + 1 :] - cu_seqlens[num_decodes:-1]


_HOST_SEQ_LENS_ATTR = "_aiter_host_seq_lens"


def register_host_seq_lens(cu_seqlens: torch.Tensor, seq_lens_cpu) -> None:
    """Attach host-side per-sequence lengths to a device ``cu_seqlens`` tensor.

    ``prepare_chunk_indices`` / ``prepare_num_chunks`` then derive their
    results from these host values instead of reading ``cu_seqlens`` back
    from the device, so the prefill path issues no device->host sync.
    ``seq_lens_cpu`` must include any leading decode-only sequences, matching
    ``cu_seqlens`` element for element.
    """
    lens = tuple(int(n) for n in seq_lens_cpu)
    if len(lens) != cu_seqlens.numel() - 1:
        raise ValueError(
            f"seq_lens_cpu has {len(lens)} entries but cu_seqlens describes "
            f"{cu_seqlens.numel() - 1} sequences."
        )
    setattr(cu_seqlens, _HOST_SEQ_LENS_ATTR, lens)


def _host_prefill_lens(cu_seqlens: torch.Tensor, num_decodes: int):
    lens = getattr(cu_seqlens, _HOST_SEQ_LENS_ATTR, None)
    return None if lens is None else lens[num_decodes:]


def _to_device_async(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if like.device.type == "cpu" or t.device.type != "cpu":
        return t.to(like)
    return t.pin_memory().to(device=like.device, dtype=like.dtype, non_blocking=True)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
) -> torch.LongTensor:
    """
    Prepare chunk indices for variable-length sequences.

    When the caller is the GDN mixed-batch dispatcher,
    ``num_decodes`` / ``num_decode_tokens`` indicate that the prefill slice
    starts at ``cu_seqlens[num_decodes]`` rather than ``cu_seqlens[0]``; we
    account for that internally so the cache key stays
    ``(metadata_cu_seqlens_id, chunk_size, num_decodes, num_decode_tokens)``
    -- stable across forward calls when the scheduler reuses the metadata
    ``cu_seqlens`` tensor. This is what eliminates the per-forward
    ``.tolist()`` D2H: passing the ORIGINAL stable tensor (never a freshly
    sliced one) keeps every subsequent call a cache hit.

    ``num_decode_tokens`` does not affect the output (the chunk indices are
    rebase-invariant) but is part of the cache key for symmetry with
    ``prepare_chunk_offsets``.

    Args:
        cu_seqlens: Cumulative sequence lengths [N+1] (original, unsliced)
        chunk_size: Size of each chunk
        num_decodes: number of leading decode-only sequences to skip
        num_decode_tokens: number of leading decode tokens (cache key only)

    Returns:
        Tensor of shape [num_chunks, 2] where each row is [sequence_id, chunk_idx_in_seq]
    """
    _ = num_decode_tokens  # in cache key only
    host_lens = _host_prefill_lens(cu_seqlens, num_decodes)
    if host_lens is not None:
        counts = [triton.cdiv(n, chunk_size) for n in host_lens]
    else:
        counts = triton.cdiv(_prefill_lens(cu_seqlens, num_decodes), chunk_size).tolist()
    # Explicit CPU: callers may run under torch.set_default_device("cuda").
    indices = torch.cat([torch.arange(n, device="cpu") for n in counts])
    out = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
    return _to_device_async(out, cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
) -> torch.LongTensor:
    """
    Prepare cumulative chunk offsets for variable-length sequences.

    See ``prepare_chunk_indices`` for the decode-prefix slicing semantics.

    Args:
        cu_seqlens: Cumulative sequence lengths [N+1] (original, unsliced)
        chunk_size: Size of each chunk
        num_decodes: number of leading decode-only sequences to skip
        num_decode_tokens: number of leading decode tokens (cache key only)

    Returns:
        Cumulative chunk offsets [N_prefill+1]
    """
    _ = num_decode_tokens  # in cache key only
    host_lens = _host_prefill_lens(cu_seqlens, num_decodes)
    if host_lens is not None:
        counts = torch.tensor(
            [0] + [triton.cdiv(n, chunk_size) for n in host_lens], device="cpu"
        )
        return _to_device_async(counts.cumsum(-1), cu_seqlens)
    return torch.cat(
        [
            cu_seqlens.new_zeros(1),
            triton.cdiv(_prefill_lens(cu_seqlens, num_decodes), chunk_size),
        ]
    ).cumsum(-1)


@tensor_cache
def prepare_rebased_cu_seqlens(
    cu_seqlens: torch.LongTensor,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
) -> torch.LongTensor:
    """Rebase cumulative sequence lengths to the prefill slice.

    Produces the kernel-facing offsets
    ``cu_seqlens[num_decodes:] - num_decode_tokens`` for callers that need an
    actual rebased ``cu_seqlens`` tensor (e.g. as a kernel arg, or to derive
    the prefill sequence count). ``@tensor_cache`` keys on the ORIGINAL
    ``cu_seqlens`` identity + the two ints, so the slice/subtract runs once
    per (cu_seqlens_id, num_decodes, num_decode_tokens) tuple.

    When ``num_decodes == 0`` and ``num_decode_tokens == 0`` (pure-prefill
    batch) the input tensor is returned unchanged (same object).
    """
    if num_decodes == 0 and num_decode_tokens == 0:
        return cu_seqlens
    return cu_seqlens[num_decodes:] - num_decode_tokens


@tensor_cache
def prepare_num_chunks(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
) -> int:
    """Total number of chunks across the prefill sequences (host int).

    Equals the last element of the (decode-aware) cumulative chunk offsets,
    i.e. ``sum_i cdiv(prefill_seqlen_i, chunk_size)``. Reuses the cached
    ``prepare_chunk_offsets``; the single ``int()`` D2H is memoized per
    ``(cu_seqlens identity, chunk_size, num_decodes, num_decode_tokens)``, so
    it runs once per shape rather than once per forward call.
    """
    host_lens = _host_prefill_lens(cu_seqlens, num_decodes)
    if host_lens is not None:
        return sum(triton.cdiv(n, chunk_size) for n in host_lens)
    return int(
        prepare_chunk_offsets(cu_seqlens, chunk_size, num_decodes, num_decode_tokens)[
            -1
        ]
    )


@tensor_cache
def get_max_num_splits(cu_seqlens: torch.LongTensor, chunk_size: int) -> int:
    """Get maximum number of splits (chunks) across all sequences."""
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)
