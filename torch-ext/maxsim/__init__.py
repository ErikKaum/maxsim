"""Thin Python wrapper around the compiled MaxSim kernel.

This module exposes:

* :func:`score_pairs_packed` -- the canonical, kernel-facing packed API.
* :func:`score_candidates_padded` -- an ergonomic padded wrapper that
  converts ``[B, Lq, D]`` / ``[B, C, Ld, D]`` inputs into the packed
  representation and delegates to the same kernel.
* :func:`maxsim_reference` -- a pure-PyTorch reference for tests / benchmarks.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ._ops import ops
from .reference import (
    maxsim_reference,
    score_candidates_padded_reference,
    score_pairs_packed_reference,
)

__all__ = [
    "score_pairs_packed",
    "score_candidates_padded",
    "maxsim_reference",
    "score_pairs_packed_reference",
    "score_candidates_padded_reference",
]


_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_INDEX_DTYPES = (torch.int32, torch.int64)


def _check_float(name: str, t: torch.Tensor) -> None:
    if t.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            f"{name} must be float32, float16, or bfloat16; got {t.dtype}"
        )


def _check_index(name: str, t: torch.Tensor) -> None:
    if t.dtype not in _INDEX_DTYPES:
        raise TypeError(f"{name} must be int32 or int64; got {t.dtype}")


def _check_same_device(tensors: dict) -> None:
    devices = {name: t.device for name, t in tensors.items()}
    first_name, first_dev = next(iter(devices.items()))
    for name, dev in devices.items():
        if dev != first_dev:
            raise RuntimeError(
                f"all tensors must be on the same device; {first_name} is on "
                f"{first_dev} but {name} is on {dev}"
            )


def score_pairs_packed(
    queries: torch.Tensor,
    query_offsets: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
    pair_query_ids: torch.Tensor,
    pair_document_ids: torch.Tensor,
    *,
    max_q_len: int | None = None,
) -> torch.Tensor:
    """Compute MaxSim scores for a packed ragged batch of (query, document) pairs.

    Args:
        queries: ``[total_q_tokens, dim]`` (fp32 / fp16 / bf16).
        query_offsets: ``[num_queries + 1]`` (int32 / int64). Must start at 0,
            end at ``queries.shape[0]``, be strictly monotonically increasing
            (no empty query segments).
        documents: ``[total_d_tokens, dim]`` with the same dtype as ``queries``.
        document_offsets: ``[num_documents + 1]`` (int32 / int64), same rules
            as ``query_offsets``.
        pair_query_ids: ``[num_pairs]`` of query ids in ``[0, num_queries)``.
        pair_document_ids: ``[num_pairs]`` of document ids in
            ``[0, num_documents)``.
        max_q_len: optional pre-computed maximum query segment length. When
            provided the kernel skips its CPU-side validation pass (4 fewer
            device->host syncs per call); pass it whenever you've already
            computed it (e.g. ``_pack_padded`` returns it for free). When
            ``None`` the kernel falls back to the safer / slower validating
            path.

    Returns:
        ``[num_pairs]`` fp32 tensor of MaxSim scores on the same device.
    """
    if queries.dim() != 2:
        raise ValueError(
            f"queries must be 2-D [total_q_tokens, dim]; got shape {tuple(queries.shape)}"
        )
    if documents.dim() != 2:
        raise ValueError(
            f"documents must be 2-D [total_d_tokens, dim]; got shape {tuple(documents.shape)}"
        )
    if queries.shape[1] != documents.shape[1]:
        raise ValueError(
            "queries.dim and documents.dim must match; got "
            f"{queries.shape[1]} vs {documents.shape[1]}"
        )
    if queries.dtype != documents.dtype:
        raise TypeError(
            "queries and documents must have the same dtype; got "
            f"{queries.dtype} vs {documents.dtype}"
        )

    _check_float("queries", queries)
    _check_float("documents", documents)
    _check_index("query_offsets", query_offsets)
    _check_index("document_offsets", document_offsets)
    _check_index("pair_query_ids", pair_query_ids)
    _check_index("pair_document_ids", pair_document_ids)

    _check_same_device(
        dict(
            queries=queries,
            query_offsets=query_offsets,
            documents=documents,
            document_offsets=document_offsets,
            pair_query_ids=pair_query_ids,
            pair_document_ids=pair_document_ids,
        )
    )

    if pair_query_ids.shape != pair_document_ids.shape:
        raise ValueError(
            "pair_query_ids and pair_document_ids must have the same shape; "
            f"got {tuple(pair_query_ids.shape)} vs {tuple(pair_document_ids.shape)}"
        )
    if pair_query_ids.dim() != 1:
        raise ValueError(
            f"pair_query_ids must be 1-D; got shape {tuple(pair_query_ids.shape)}"
        )

    return ops.maxsim_forward(
        queries.contiguous(),
        query_offsets.contiguous(),
        documents.contiguous(),
        document_offsets.contiguous(),
        pair_query_ids.contiguous(),
        pair_document_ids.contiguous(),
        -1 if max_q_len is None else int(max_q_len),
    )


# ---------------------------------------------------------------------------
# Padded -> packed conversion + ergonomic wrapper.
# ---------------------------------------------------------------------------


def _pack_padded(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
    *,
    validate: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, int,
]:
    """Convert padded ``[B, Lq, D]`` / ``[B, C, Ld, D]`` inputs to packed form.

    Returns:
        ``(queries_packed, query_offsets, documents_packed, document_offsets,
        pair_query_ids, pair_document_ids, max_q_len_int)``.
        ``max_q_len_int`` is a Python int suitable for passing through to
        :func:`score_pairs_packed`'s ``max_q_len`` argument so the kernel can
        skip its own CPU-side validation.

    Pair ordering is row-major ``(batch, candidate)`` so the output of the
    packed call can be reshaped to ``[B, C]``.

    ``validate=False`` (default) skips the ``.any().item()`` length checks --
    those would force a device->host sync on every call. Pass
    ``validate=True`` to enable them (e.g. for first-time / debug usage).
    """
    if queries.dim() != 3:
        raise ValueError(
            f"queries must be 3-D [B, Lq, D]; got shape {tuple(queries.shape)}"
        )
    if documents.dim() != 4:
        raise ValueError(
            f"documents must be 4-D [B, C, Ld, D]; got shape {tuple(documents.shape)}"
        )
    B, Lq_max, D = queries.shape
    Bd, C, Ld_max, Dd = documents.shape
    if B != Bd:
        raise ValueError(
            f"batch dim mismatch: queries B={B} but documents B={Bd}"
        )
    if D != Dd:
        raise ValueError(
            f"embedding dim mismatch: queries D={D} but documents D={Dd}"
        )
    if query_lengths.shape != (B,):
        raise ValueError(
            f"query_lengths must have shape [B={B}]; got {tuple(query_lengths.shape)}"
        )
    if doc_lengths.shape != (B, C):
        raise ValueError(
            f"doc_lengths must have shape [B={B}, C={C}]; got {tuple(doc_lengths.shape)}"
        )

    device = queries.device
    qlen = query_lengths.to(device=device, dtype=torch.int32)
    dlen = doc_lengths.to(device=device, dtype=torch.int32)

    if validate:
        # These three checks each force a device->host sync.
        if (qlen <= 0).any().item():
            raise ValueError("query_lengths must all be > 0")
        if (dlen <= 0).any().item():
            raise ValueError("doc_lengths must all be > 0")
        if (qlen > Lq_max).any().item() or (dlen > Ld_max).any().item():
            raise ValueError("a length exceeds the padded extent")

    # Vectorised pack: build a boolean mask of valid (non-padded) positions
    # and gather them in row-major order. The same row-major order is used
    # for the offsets so they stay consistent.
    q_arange = torch.arange(Lq_max, device=device, dtype=torch.int32)
    q_mask = q_arange[None, :] < qlen[:, None]                # [B, Lq_max]
    queries_packed = queries.reshape(B * Lq_max, D)[q_mask.reshape(-1)]

    d_arange = torch.arange(Ld_max, device=device, dtype=torch.int32)
    d_mask = d_arange[None, None, :] < dlen[:, :, None]       # [B, C, Ld_max]
    documents_packed = documents.reshape(B * C * Ld_max, D)[d_mask.reshape(-1)]

    query_offsets = torch.zeros(B + 1, dtype=torch.int32, device=device)
    query_offsets[1:] = qlen.cumsum(0)

    document_offsets = torch.zeros(B * C + 1, dtype=torch.int32, device=device)
    document_offsets[1:] = dlen.reshape(-1).cumsum(0)

    # Pair ordering: (b, c) -> pair index b * C + c, query id b, doc id b * C + c.
    pair_indices = torch.arange(B * C, dtype=torch.int32, device=device)
    pair_query_ids = (pair_indices // C).contiguous()
    pair_document_ids = pair_indices.contiguous()

    # One sync to pull max query length to CPU so the kernel can skip its own
    # validation pass. This is unavoidable today because we need an int to
    # size threadgroup memory; doing it here amortises one sync against the
    # four the C++ kernel would otherwise do.
    max_q_len_int = int(qlen.max().item())

    return (
        queries_packed,
        query_offsets,
        documents_packed,
        document_offsets,
        pair_query_ids,
        pair_document_ids,
        max_q_len_int,
    )


def score_candidates_padded(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute MaxSim scores directly on padded inputs.

    This is the recommended high-throughput entry point: it dispatches to a
    dedicated Metal kernel that reads ``queries`` and ``documents`` in place
    via padded strides, so there's no pack/gather, no ``cumsum``, and no
    device->host syncs in the hot path.

    Args:
        queries: ``[B, Lq, dim]``.
        documents: ``[B, C, Ld, dim]``.
        query_lengths: ``[B]`` (int32 / int64). Each value in ``(0, Lq]``.
        doc_lengths: ``[B, C]`` (int32 / int64). Each value in ``(0, Ld]``.

    Returns:
        ``[B, C]`` fp32 tensor of MaxSim scores on the same device.
    """
    if queries.dim() != 3:
        raise ValueError(
            f"queries must be 3-D [B, Lq, D]; got shape {tuple(queries.shape)}"
        )
    if documents.dim() != 4:
        raise ValueError(
            f"documents must be 4-D [B, C, Ld, D]; got shape {tuple(documents.shape)}"
        )
    B, Lq_max, D = queries.shape
    Bd, C, Ld_max, Dd = documents.shape
    if B != Bd:
        raise ValueError(
            f"batch dim mismatch: queries B={B} but documents B={Bd}"
        )
    if D != Dd:
        raise ValueError(
            f"embedding dim mismatch: queries D={D} but documents D={Dd}"
        )
    if query_lengths.shape != (B,):
        raise ValueError(
            f"query_lengths must have shape [B={B}]; got {tuple(query_lengths.shape)}"
        )
    if doc_lengths.shape != (B, C):
        raise ValueError(
            f"doc_lengths must have shape [B={B}, C={C}]; got {tuple(doc_lengths.shape)}"
        )
    if queries.dtype != documents.dtype:
        raise TypeError(
            "queries and documents must have the same dtype; got "
            f"{queries.dtype} vs {documents.dtype}"
        )
    _check_float("queries", queries)
    _check_float("documents", documents)
    _check_index("query_lengths", query_lengths)
    _check_index("doc_lengths", doc_lengths)
    _check_same_device(
        dict(
            queries=queries,
            documents=documents,
            query_lengths=query_lengths,
            doc_lengths=doc_lengths,
        )
    )

    queries_flat = queries.reshape(B * Lq_max, D).contiguous()
    documents_flat = documents.reshape(B * C * Ld_max, D).contiguous()
    qlen = query_lengths.to(dtype=torch.int32).contiguous()
    dlen = doc_lengths.reshape(B * C).to(dtype=torch.int32).contiguous()

    flat = ops.maxsim_padded_forward(
        queries_flat,
        qlen,
        documents_flat,
        dlen,
        int(Lq_max),
        int(Ld_max),
        int(C),
    )
    return flat.view(B, C)
