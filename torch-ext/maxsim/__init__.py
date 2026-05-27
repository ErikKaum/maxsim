"""Thin Python wrapper around the compiled MaxSim kernel.

Three scoring surfaces, each with an inference function, a ``_with_argmax``
variant (also returns the winning document-token index per query token), and a
``_train`` variant wired into PyTorch autograd:

* :func:`score_candidates_padded` -- padded reranking. Reads ``[B, Lq, D]``
  queries and ``[B, K, Ld, D]`` candidates directly; the common inference path.
* :func:`score_contrastive` -- all-pairs ``[Nq, Nb]`` scoring with packed
  documents; what in-batch contrastive training losses consume.
* :func:`score_pairs_packed` -- the lowest-level, kernel-facing API over
  arbitrary ``(query, document)`` pair grids on ragged inputs.

Pure-PyTorch references (:func:`maxsim_reference`, ``score_*_reference``) are
also exported for tests and benchmarks.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ._ops import ops
from .reference import (
    maxsim_reference,
    maxsim_reference_with_argmax,
    score_candidates_padded_reference,
    score_candidates_padded_with_argmax_reference,
    score_contrastive_backward_reference,
    score_contrastive_reference,
    score_contrastive_with_argmax_reference,
    score_pairs_packed_backward_reference,
    score_pairs_packed_reference,
    score_pairs_packed_with_argmax_reference,
)

__all__ = [
    "score_pairs_packed",
    "score_pairs_packed_with_argmax",
    "score_pairs_packed_train",
    "score_candidates_padded",
    "score_candidates_padded_with_argmax",
    "score_candidates_padded_train",
    "score_contrastive",
    "score_contrastive_with_argmax",
    "score_contrastive_train",
    "maxsim_reference",
    "maxsim_reference_with_argmax",
    "score_pairs_packed_reference",
    "score_pairs_packed_with_argmax_reference",
    "score_candidates_padded_reference",
    "score_candidates_padded_with_argmax_reference",
    "score_contrastive_reference",
    "score_contrastive_with_argmax_reference",
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


def _validate_length_bounds(
    lengths: torch.Tensor,
    *,
    max_len: int,
    name: str,
) -> None:
    values = lengths.detach().to(device="cpu", dtype=torch.int64)
    if (values <= 0).any().item():
        raise ValueError(f"{name} must contain values > 0")
    if (values > max_len).any().item():
        raise ValueError(
            f"{name} values must be <= padded length {max_len}"
        )


def _check_padded_shapes(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> tuple[int, int, int, int, int]:
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
    _validate_length_bounds(query_lengths, max_len=Lq_max, name="query_lengths")
    _validate_length_bounds(doc_lengths, max_len=Ld_max, name="doc_lengths")
    return B, C, Lq_max, Ld_max, D


def _check_pair_ids(
    pair_query_ids: torch.Tensor,
    pair_document_ids: torch.Tensor,
) -> None:
    if pair_query_ids.shape != pair_document_ids.shape:
        raise ValueError(
            "pair_query_ids and pair_document_ids must have the same shape; "
            f"got {tuple(pair_query_ids.shape)} vs {tuple(pair_document_ids.shape)}"
        )
    if pair_query_ids.dim() != 1:
        raise ValueError(
            f"pair_query_ids must be 1-D; got shape {tuple(pair_query_ids.shape)}"
        )


def _validate_packed_layout(
    queries: torch.Tensor,
    query_offsets: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
    pair_query_ids: torch.Tensor,
    pair_document_ids: torch.Tensor,
) -> int:
    """Validate packed offsets and ids, returning max query segment length.

    This is intentionally a host-side sync so public APIs fail clearly before
    launching a native kernel with invalid layout metadata.
    """

    def _validate_offsets(
        offsets: torch.Tensor,
        total_tokens: int,
        name: str,
    ) -> tuple[list[int], int]:
        values = offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
        if len(values) < 2:
            raise RuntimeError(f"{name} must have length >= 2")
        if values[0] != 0:
            raise RuntimeError(f"{name}[0] must equal 0, got {values[0]}")
        if values[-1] != total_tokens:
            raise RuntimeError(
                f"{name}[-1] ({values[-1]}) must equal total token count "
                f"({total_tokens})"
            )
        max_len = 0
        for i, (start, end) in enumerate(zip(values, values[1:])):
            diff = end - start
            if diff <= 0:
                raise RuntimeError(f"empty segment in {name} at index {i}")
            max_len = max(max_len, diff)
        return values, max_len

    def _validate_ids(ids: torch.Tensor, upper: int, name: str) -> None:
        values = ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        for i, value in enumerate(values):
            if value < 0 or value >= upper:
                raise RuntimeError(
                    f"{name}[{i}] = {value} out of range [0, {upper})"
                )

    q_offsets, max_q_len = _validate_offsets(
        query_offsets, queries.shape[0], "query_offsets"
    )
    d_offsets, _ = _validate_offsets(
        document_offsets, documents.shape[0], "document_offsets"
    )
    _validate_ids(pair_query_ids, len(q_offsets) - 1, "pair_query_ids")
    _validate_ids(pair_document_ids, len(d_offsets) - 1, "pair_document_ids")
    return max_q_len


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
            provided it is checked against ``query_offsets`` before launch; it
            must be at least the actual maximum query segment length.

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

    _check_pair_ids(pair_query_ids, pair_document_ids)

    actual_max_q_len = _validate_packed_layout(
        queries,
        query_offsets,
        documents,
        document_offsets,
        pair_query_ids,
        pair_document_ids,
    )
    if max_q_len is None:
        mql = actual_max_q_len
    else:
        if max_q_len <= 0:
            raise ValueError(f"max_q_len must be > 0; got {max_q_len}")
        if max_q_len < actual_max_q_len:
            raise ValueError(
                f"max_q_len ({max_q_len}) must be >= actual max query "
                f"segment length ({actual_max_q_len})"
            )
        mql = int(max_q_len)

    return ops.maxsim_forward(
        queries.contiguous(),
        query_offsets.contiguous(),
        documents.contiguous(),
        document_offsets.contiguous(),
        pair_query_ids.contiguous(),
        pair_document_ids.contiguous(),
        mql,
    )

def _check_packed_shapes_for_argmax(
    queries: torch.Tensor,
    query_offsets: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
    pair_query_ids: torch.Tensor,
    pair_document_ids: torch.Tensor,
) -> None:
    if queries.dim() != 2:
        raise ValueError(
            f"queries must be 2-D; got shape {tuple(queries.shape)}"
        )
    if documents.dim() != 2:
        raise ValueError(
            f"documents must be 2-D; got shape {tuple(documents.shape)}"
        )
    if queries.shape[1] != documents.shape[1]:
        raise ValueError(
            f"queries.D ({queries.shape[1]}) must match documents.D "
            f"({documents.shape[1]})"
        )
    if queries.dtype != documents.dtype:
        raise TypeError(
            f"queries and documents must share dtype; got {queries.dtype} "
            f"vs {documents.dtype}"
        )
    _check_float("queries", queries)
    _check_float("documents", documents)
    _check_index("query_offsets", query_offsets)
    _check_index("document_offsets", document_offsets)
    _check_index("pair_query_ids", pair_query_ids)
    _check_index("pair_document_ids", pair_document_ids)
    _check_same_device(dict(
        queries=queries, query_offsets=query_offsets,
        documents=documents, document_offsets=document_offsets,
        pair_query_ids=pair_query_ids,
        pair_document_ids=pair_document_ids,
    ))
    _check_pair_ids(pair_query_ids, pair_document_ids)


def score_pairs_packed_with_argmax(
    queries: torch.Tensor,
    query_offsets: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
    pair_query_ids: torch.Tensor,
    pair_document_ids: torch.Tensor,
    *,
    max_q_len: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`score_pairs_packed` but also returns the per-q-tok argmax
    positions.

    Returns:
        ``(scores, argmax)`` — ``scores`` is fp32 ``[num_pairs]``; ``argmax``
        is int32 ``[num_pairs, max_q_len]``. Slots beyond a pair's Lq are 0.
        First-index-wins tiebreak.
    """
    _check_packed_shapes_for_argmax(
        queries, query_offsets, documents, document_offsets,
        pair_query_ids, pair_document_ids,
    )
    actual_max_q_len = _validate_packed_layout(
        queries, query_offsets, documents, document_offsets,
        pair_query_ids, pair_document_ids,
    )
    if max_q_len is None:
        mql = actual_max_q_len
    else:
        if max_q_len <= 0:
            raise ValueError(f"max_q_len must be > 0; got {max_q_len}")
        if max_q_len < actual_max_q_len:
            raise ValueError(
                f"max_q_len ({max_q_len}) must be >= actual max query "
                f"segment length ({actual_max_q_len})"
            )
        mql = int(max_q_len)
    return ops.maxsim_packed_forward_with_argmax(
        queries.contiguous(),
        query_offsets.contiguous(),
        documents.contiguous(),
        document_offsets.contiguous(),
        pair_query_ids.contiguous(),
        pair_document_ids.contiguous(),
        mql,
    )


class _ScorePairsPacked(torch.autograd.Function):
    """Differentiable wrapper for the packed forward+argmax / backward
    pair. Same fp32-grad convention as the padded and contrastive autograd
    Functions."""

    @staticmethod
    def forward(
        ctx,
        queries: torch.Tensor,
        query_offsets: torch.Tensor,
        documents: torch.Tensor,
        document_offsets: torch.Tensor,
        pair_query_ids: torch.Tensor,
        pair_document_ids: torch.Tensor,
        max_q_len: int,
    ) -> torch.Tensor:
        q_c = queries.contiguous()
        d_c = documents.contiguous()
        qoff = query_offsets.contiguous()
        doff = document_offsets.contiguous()
        qids = pair_query_ids.contiguous()
        dids = pair_document_ids.contiguous()
        mql = int(max_q_len)

        scores, argmax = ops.maxsim_packed_forward_with_argmax(
            q_c, qoff, d_c, doff, qids, dids, mql
        )
        ctx.save_for_backward(q_c, qoff, d_c, doff, qids, dids, argmax)
        ctx.max_q_len = mql
        return scores

    @staticmethod
    def backward(ctx, dscore: torch.Tensor):
        q_c, qoff, d_c, doff, qids, dids, argmax = ctx.saved_tensors
        dscore_f32 = dscore.contiguous().to(torch.float32)
        dq, dd = ops.maxsim_packed_backward(
            dscore_f32, q_c, qoff, d_c, doff, qids, dids, argmax,
            ctx.max_q_len,
        )
        # Only queries/documents are differentiable.
        return dq, None, dd, None, None, None, None


def score_pairs_packed_train(
    queries: torch.Tensor,
    query_offsets: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
    pair_query_ids: torch.Tensor,
    pair_document_ids: torch.Tensor,
    *,
    max_q_len: int | None = None,
) -> torch.Tensor:
    """Differentiable packed MaxSim — the training entry point.

    Same forward semantics as :func:`score_pairs_packed` but plugged into
    PyTorch autograd. Gradients are fp32 (cast at the call site if needed).

    Args:
        queries: ``[total_q_tokens, dim]``. ``requires_grad=True`` to
            receive grads.
        documents: ``[total_d_tokens, dim]``. Same.
        query_offsets / document_offsets / pair_query_ids / pair_document_ids:
            non-differentiable layout tensors.
        max_q_len: optional pre-computed max query segment length. When
            provided it is checked against ``query_offsets`` before launch; it
            must be at least the actual maximum query segment length.

    Returns:
        ``[num_pairs]`` fp32 tensor of MaxSim scores, end-to-end
        differentiable.
    """
    _check_packed_shapes_for_argmax(
        queries, query_offsets, documents, document_offsets,
        pair_query_ids, pair_document_ids,
    )
    actual_max_q_len = _validate_packed_layout(
        queries, query_offsets, documents, document_offsets,
        pair_query_ids, pair_document_ids,
    )
    if max_q_len is None:
        mql = actual_max_q_len
    else:
        if max_q_len <= 0:
            raise ValueError(f"max_q_len must be > 0; got {max_q_len}")
        if max_q_len < actual_max_q_len:
            raise ValueError(
                f"max_q_len ({max_q_len}) must be >= actual max query "
                f"segment length ({actual_max_q_len})"
            )
        mql = int(max_q_len)
    return _ScorePairsPacked.apply(
        queries, query_offsets, documents, document_offsets,
        pair_query_ids, pair_document_ids, mql,
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
        :func:`score_pairs_packed`'s ``max_q_len`` argument. The public API
        still checks it against ``query_offsets`` before launching the native
        kernel.

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
    via padded strides, so there's no pack/gather or ``cumsum``. The Python
    wrapper validates that the real lengths fit inside the padded extents
    before launching the kernel.

    Args:
        queries: ``[B, Lq, dim]``.
        documents: ``[B, C, Ld, dim]``.
        query_lengths: ``[B]`` (int32 / int64). Each value in ``(0, Lq]``.
        doc_lengths: ``[B, C]`` (int32 / int64). Each value in ``(0, Ld]``.

    Returns:
        ``[B, C]`` fp32 tensor of MaxSim scores on the same device.
    """
    B, C, Lq_max, Ld_max, D = _check_padded_shapes(
        queries, documents, query_lengths, doc_lengths
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


def score_candidates_padded_with_argmax(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`score_candidates_padded` but also returns argmax positions
    per query token. This is the forward pass kernel needed for backward /
    training: the int32 argmax buffer is the small saved-for-backward
    payload (95-205x smaller than materialising ``[Lq, Ld]``).

    Args:
        queries: ``[B, Lq, dim]``.
        documents: ``[B, C, Ld, dim]``.
        query_lengths: ``[B]`` (int32 / int64). Each value in ``(0, Lq]``.
        doc_lengths: ``[B, C]`` (int32 / int64). Each value in ``(0, Ld]``.

    Returns:
        ``(scores, argmax)`` — ``scores`` is fp32 ``[B, C]``; ``argmax`` is
        int32 ``[B, C, Lq]`` with the document-token index that won the
        max per query token. PyTorch's first-index-wins tiebreak applies;
        slots beyond ``query_lengths[b]`` are 0.
    """
    B, C, Lq_max, Ld_max, D = _check_padded_shapes(
        queries, documents, query_lengths, doc_lengths
    )

    queries_flat = queries.reshape(B * Lq_max, D).contiguous()
    documents_flat = documents.reshape(B * C * Ld_max, D).contiguous()
    qlen = query_lengths.to(dtype=torch.int32).contiguous()
    dlen = doc_lengths.reshape(B * C).to(dtype=torch.int32).contiguous()

    flat_scores, flat_argmax = ops.maxsim_padded_forward_with_argmax(
        queries_flat,
        qlen,
        documents_flat,
        dlen,
        int(Lq_max),
        int(Ld_max),
        int(C),
    )
    return flat_scores.view(B, C), flat_argmax.view(B, C, Lq_max)


# ---------------------------------------------------------------------------
# Training-mode (differentiable) entry point.
# ---------------------------------------------------------------------------


class _ScoreCandidatesPadded(torch.autograd.Function):
    """Differentiable wrapper around the padded forward+argmax / backward
    kernel pair. Forward computes scores and saves (q_flat, d_flat, qlen,
    dlen, argmax_flat, dims) for backward; backward routes the incoming
    grad via the saved argmax.

    Gradient dtype is always fp32 (the kernel accumulates atomically in
    fp32 to avoid the bf16-atomic-only-on-Hopper hardware gap). Returning
    fp32 grads lines up with how AMP / mixed-precision training stacks
    expect to receive gradients.
    """

    @staticmethod
    def forward(
        ctx,
        queries: torch.Tensor,        # [B, Lq, D]
        documents: torch.Tensor,      # [B, C, Ld, D]
        query_lengths: torch.Tensor,  # [B]
        doc_lengths: torch.Tensor,    # [B, C]
    ) -> torch.Tensor:
        B, Lq, D = queries.shape
        _, C, Ld, _ = documents.shape
        q_flat = queries.reshape(B * Lq, D).contiguous()
        d_flat = documents.reshape(B * C * Ld, D).contiguous()
        qlen = query_lengths.to(dtype=torch.int32).contiguous()
        dlen = doc_lengths.reshape(B * C).to(dtype=torch.int32).contiguous()

        scores_flat, argmax_flat = ops.maxsim_padded_forward_with_argmax(
            q_flat, qlen, d_flat, dlen, int(Lq), int(Ld), int(C)
        )

        # Save for backward.
        ctx.save_for_backward(q_flat, d_flat, qlen, dlen, argmax_flat)
        ctx.shapes = (B, C, Lq, Ld, D)
        return scores_flat.view(B, C)

    @staticmethod
    def backward(ctx, dscore: torch.Tensor):
        B, C, Lq, Ld, D = ctx.shapes
        q_flat, d_flat, qlen, dlen, argmax_flat = ctx.saved_tensors
        dscore_flat = dscore.reshape(B * C).contiguous().to(torch.float32)

        dq_flat, dd_flat = ops.maxsim_padded_backward(
            dscore_flat,
            q_flat,
            d_flat,
            qlen,
            dlen,
            argmax_flat,
            int(Lq),
            int(Ld),
            int(C),
        )
        # Return one grad per forward input (None for non-tensor / non-
        # differentiable inputs query_lengths and doc_lengths).
        return dq_flat.view(B, Lq, D), dd_flat.view(B, C, Ld, D), None, None


def score_candidates_padded_train(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> torch.Tensor:
    """Differentiable padded MaxSim — the training entry point.

    Same forward semantics as :func:`score_candidates_padded` but plugged
    into PyTorch autograd so ``scores.sum().backward()`` (or any other
    downstream loss) propagates gradients back to ``queries`` and
    ``documents``. The kernel accumulates gradients in fp32; PyTorch stores
    ``.grad`` in the input tensor's dtype for fp16/bf16 inputs.

    Args:
        queries: ``[B, Lq, dim]``. Must have ``requires_grad=True`` to
            receive gradients.
        documents: ``[B, C, Ld, dim]``. Same.
        query_lengths: ``[B]`` (int32 / int64). Non-differentiable.
        doc_lengths: ``[B, C]`` (int32 / int64). Non-differentiable.

    Returns:
        ``[B, C]`` fp32 tensor of MaxSim scores, end-to-end differentiable.
    """
    _check_padded_shapes(queries, documents, query_lengths, doc_lengths)
    return _ScoreCandidatesPadded.apply(
        queries, documents, query_lengths, doc_lengths
    )


# ---------------------------------------------------------------------------
# Contrastive (all-pairs) API: every query in `queries[Nq, Lq, D]` is scored
# against every doc in `documents[total_d_tokens, D]` packed via document_offsets.
# This is the in-batch contrastive layout standard ColBERT-style training
# loops use (flash-maxsim's killer feature).
# ---------------------------------------------------------------------------


def _check_contrastive_shapes(
    queries: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
) -> None:
    _check_float("queries", queries)
    _check_float("documents", documents)
    _check_index("document_offsets", document_offsets)
    if queries.dtype != documents.dtype:
        raise TypeError(
            f"queries and documents must share dtype; "
            f"got {queries.dtype} and {documents.dtype}"
        )
    if queries.dim() != 3:
        raise ValueError(
            f"queries must be [Nq, Lq, D]; got shape {tuple(queries.shape)}"
        )
    if documents.dim() != 2:
        raise ValueError(
            f"documents must be [total_d_tokens, D] (packed); got shape "
            f"{tuple(documents.shape)}"
        )
    if document_offsets.dim() != 1:
        raise ValueError(
            f"document_offsets must be 1-D [Nb+1]; got shape {tuple(document_offsets.shape)}"
        )
    if queries.shape[2] != documents.shape[1]:
        raise ValueError(
            f"queries.D ({queries.shape[2]}) must match documents.D "
            f"({documents.shape[1]})"
        )
    if queries.shape[1] <= 0:
        raise ValueError(f"Lq must be > 0; got {queries.shape[1]}")
    if queries.shape[2] <= 0:
        raise ValueError(f"dim must be > 0; got {queries.shape[2]}")
    if document_offsets.numel() < 2:
        raise ValueError(
            f"document_offsets must have length >= 2 (at least one doc); "
            f"got {document_offsets.numel()}"
        )
    _check_same_device(
        dict(queries=queries, documents=documents, document_offsets=document_offsets)
    )
    # Invariant checks on document_offsets. These pay one host sync, but catching
    # bad offsets here gives a clear error instead of a kernel crash or
    # silent wrong-answer.
    cu_cpu = document_offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
    total_d_tokens = documents.shape[0]
    if cu_cpu[0] != 0:
        raise ValueError(
            f"document_offsets[0] must equal 0 (CSR offset convention); "
            f"got {cu_cpu[0]}"
        )
    if cu_cpu[-1] != total_d_tokens:
        raise ValueError(
            f"document_offsets[-1] ({cu_cpu[-1]}) must equal total doc tokens "
            f"({total_d_tokens}). The packed documents tensor and the "
            f"offsets must agree on the total length."
        )
    for i in range(len(cu_cpu) - 1):
        if cu_cpu[i + 1] <= cu_cpu[i]:
            raise ValueError(
                f"document_offsets must be strictly increasing; "
                f"got document_offsets[{i + 1}] ({cu_cpu[i + 1]}) <= "
                f"document_offsets[{i}] ({cu_cpu[i]})"
            )


def score_contrastive(
    queries: torch.Tensor,    # [Nq, Lq, D]
    documents: torch.Tensor,  # [total_d_tokens, D]
    document_offsets: torch.Tensor, # [Nb + 1]
) -> torch.Tensor:
    """Contrastive MaxSim: score every query against every doc.

    Args:
        queries: ``[Nq, Lq, dim]`` fp16 / bf16.
        documents: ``[total_d_tokens, dim]`` packed, same dtype as queries.
        document_offsets: ``[Nb + 1]`` int32 cumulative doc-length offsets.

    Returns:
        ``[Nq, Nb]`` fp32 score tensor.
    """
    _check_contrastive_shapes(queries, documents, document_offsets)
    document_offsets_i32 = document_offsets.to(dtype=torch.int32).contiguous()
    return ops.maxsim_contrastive_forward(
        queries.contiguous(),
        documents.contiguous(),
        document_offsets_i32,
    )


def score_contrastive_with_argmax(
    queries: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`score_contrastive` but also returns the per-q-tok argmax
    positions.

    Returns:
        ``(scores, argmax)`` — ``scores`` is fp32 ``[Nq, Nb]``; ``argmax``
        is int32 ``[Nq, Nb, Lq]``. First-index-wins tiebreak.
    """
    _check_contrastive_shapes(queries, documents, document_offsets)
    document_offsets_i32 = document_offsets.to(dtype=torch.int32).contiguous()
    return ops.maxsim_contrastive_forward_with_argmax(
        queries.contiguous(),
        documents.contiguous(),
        document_offsets_i32,
    )


class _ScoreContrastive(torch.autograd.Function):
    """Differentiable wrapper around the contrastive forward+argmax /
    backward kernel pair. The native kernels accumulate gradients in fp32.
    """

    @staticmethod
    def forward(
        ctx,
        queries: torch.Tensor,    # [Nq, Lq, D]
        documents: torch.Tensor,  # [total_d_tokens, D]
        document_offsets: torch.Tensor, # [Nb + 1]
    ) -> torch.Tensor:
        q_c = queries.contiguous()
        d_c = documents.contiguous()
        cu = document_offsets.to(dtype=torch.int32).contiguous()

        scores, argmax = ops.maxsim_contrastive_forward_with_argmax(
            q_c, d_c, cu
        )
        ctx.save_for_backward(q_c, d_c, cu, argmax)
        return scores

    @staticmethod
    def backward(ctx, dscore: torch.Tensor):
        q_c, d_c, cu, argmax = ctx.saved_tensors
        dscore_f32 = dscore.contiguous().to(torch.float32)
        dq, dd = ops.maxsim_contrastive_backward(
            dscore_f32, q_c, d_c, cu, argmax
        )
        return dq, dd, None


def score_contrastive_train(
    queries: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
) -> torch.Tensor:
    """Differentiable contrastive MaxSim — the training entry point.

    Same forward semantics as :func:`score_contrastive` but plugged into
    PyTorch autograd so ``scores.sum().backward()`` (or any downstream
    loss like InfoNCE / triplet) propagates gradients to ``queries`` and
    ``documents``. The kernel accumulates gradients in fp32; PyTorch stores
    ``.grad`` in the input tensor's dtype for fp16/bf16 inputs.

    Args:
        queries: ``[Nq, Lq, dim]``. ``requires_grad=True`` to receive grads.
        documents: ``[total_d_tokens, dim]`` packed. Same.
        document_offsets: ``[Nb + 1]`` int32. Non-differentiable.

    Returns:
        ``[Nq, Nb]`` fp32 tensor of contrastive MaxSim scores.
    """
    _check_contrastive_shapes(queries, documents, document_offsets)
    return _ScoreContrastive.apply(queries, documents, document_offsets)
