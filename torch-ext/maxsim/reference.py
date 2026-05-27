"""Pure-PyTorch reference implementations of MaxSim.

These are intentionally simple and slow (they materialize the full
`[Lq, Ld]` similarity matrix). They exist so that:

* tests can compare the kernel against an obviously-correct baseline, and
* benchmarks can show the speed and memory wins of the kernel against the
  natural way someone would write MaxSim in PyTorch.

The public function is `maxsim_reference`, which mirrors the formula

    score(q, d) = sum_i  max_j  dot(q_i, d_j)
"""

from __future__ import annotations

import torch


def maxsim_reference(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Reference MaxSim score for a single (query, document) pair.

    Args:
        q: ``[Lq, dim]``
        d: ``[Ld, dim]``

    Returns:
        Scalar fp32 score tensor on the same device.
    """
    sim = (q.float() @ d.float().transpose(-1, -2))  # [Lq, Ld]
    return sim.max(dim=-1).values.sum()


def maxsim_reference_with_argmax(
    q: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`maxsim_reference` but also returns the argmax document index
    per query token, with PyTorch's first-index-wins tiebreak semantics.

    Returns:
        ``(score, argmax)`` where ``score`` is a scalar fp32 tensor and
        ``argmax`` is an int32 tensor of shape ``[Lq]``.
    """
    sim = q.float() @ d.float().transpose(-1, -2)  # [Lq, Ld]
    # torch.max returns first occurrence on ties — matches what we want.
    vals, idx = sim.max(dim=-1)
    return vals.sum(), idx.to(torch.int32)


def score_pairs_packed_reference(
    queries: torch.Tensor,
    query_offsets: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
    pair_query_ids: torch.Tensor,
    pair_document_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of :func:`score_pairs_packed`.

    Loops over pairs and calls :func:`maxsim_reference`. Always returns fp32.
    """
    qoff = query_offsets.to(torch.int64).cpu().tolist()
    doff = document_offsets.to(torch.int64).cpu().tolist()
    qids = pair_query_ids.to(torch.int64).cpu().tolist()
    dids = pair_document_ids.to(torch.int64).cpu().tolist()

    out = torch.empty(len(qids), dtype=torch.float32, device=queries.device)
    for k, (qi, di) in enumerate(zip(qids, dids)):
        q = queries[qoff[qi] : qoff[qi + 1]]
        d = documents[doff[di] : doff[di + 1]]
        out[k] = maxsim_reference(q, d)
    return out


def score_pairs_packed_with_argmax_reference(
    queries: torch.Tensor,
    query_offsets: torch.Tensor,
    documents: torch.Tensor,
    document_offsets: torch.Tensor,
    pair_query_ids: torch.Tensor,
    pair_document_ids: torch.Tensor,
    max_q_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`score_pairs_packed_reference` but also returns argmax
    positions per query token. Out-of-range slots (q_tok >= Lq for this pair)
    are filled with 0 so the buffer has a uniform shape.

    Returns:
        ``(scores, argmax)`` — ``scores`` is fp32 ``[num_pairs]``; ``argmax``
        is int32 ``[num_pairs, max_q_len]``. Tiebreak is PyTorch's
        first-index-wins.
    """
    qoff = query_offsets.to(torch.int64).cpu().tolist()
    doff = document_offsets.to(torch.int64).cpu().tolist()
    qids = pair_query_ids.to(torch.int64).cpu().tolist()
    dids = pair_document_ids.to(torch.int64).cpu().tolist()

    n = len(qids)
    scores = torch.empty(n, dtype=torch.float32, device=queries.device)
    argmax = torch.zeros(
        (n, max_q_len), dtype=torch.int32, device=queries.device
    )
    for k, (qi, di) in enumerate(zip(qids, dids)):
        q = queries[qoff[qi] : qoff[qi + 1]]
        d = documents[doff[di] : doff[di + 1]]
        s, a = maxsim_reference_with_argmax(q, d)
        scores[k] = s
        argmax[k, : a.numel()] = a
    return scores, argmax


def score_candidates_padded_reference(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of :func:`score_candidates_padded`.

    Args:
        queries: ``[B, Lq, dim]``
        documents: ``[B, C, Ld, dim]``
        query_lengths: ``[B]``
        doc_lengths: ``[B, C]``

    Returns:
        ``[B, C]`` fp32 tensor on the same device as ``queries``.
    """
    B, C = doc_lengths.shape
    out = torch.empty((B, C), dtype=torch.float32, device=queries.device)
    qlen = query_lengths.to(torch.int64).cpu().tolist()
    dlen = doc_lengths.to(torch.int64).cpu().tolist()
    for b in range(B):
        q = queries[b, : qlen[b]]
        for c in range(C):
            d = documents[b, c, : dlen[b][c]]
            out[b, c] = maxsim_reference(q, d)
    return out


def score_candidates_padded_backward_reference(
    dscore: torch.Tensor,        # [B, C] fp32, incoming gradient
    queries: torch.Tensor,       # [B, Lq, dim] - forward input
    documents: torch.Tensor,     # [B, C, Ld, dim] - forward input
    query_lengths: torch.Tensor, # [B]
    doc_lengths: torch.Tensor,   # [B, C]
    argmax: torch.Tensor,        # [B, C, Lq] int32 - from forward
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference backward for ``score_candidates_padded``.

    Routes ``g = dscore[b, c]`` to ``dq[b, q] += g * d[b, c, j]`` and
    ``dd[b, c, j] += g * q[b, q]`` where ``j = argmax[b, c, q]`` for each
    valid (b, c, q).

    Always returns fp32 gradients regardless of input dtype (matches the
    kernel's behavior; downstream can cast).
    """
    B, C = doc_lengths.shape
    Lq = queries.shape[1]
    Ld = documents.shape[2]

    dq = torch.zeros_like(queries, dtype=torch.float32)
    dd = torch.zeros_like(documents, dtype=torch.float32)

    qlen = query_lengths.to(torch.int64).cpu().tolist()
    dlen = doc_lengths.to(torch.int64).cpu().tolist()
    argmax_cpu = argmax.to(torch.int64).cpu()

    q_f = queries.float()
    d_f = documents.float()
    g_f = dscore.float()

    for b in range(B):
        for c in range(C):
            g = g_f[b, c].item()
            for i in range(qlen[b]):
                j = int(argmax_cpu[b, c, i].item())
                if j < 0 or j >= dlen[b][c]:
                    continue
                dq[b, i] += g * d_f[b, c, j]
                dd[b, c, j] += g * q_f[b, i]

    return dq, dd


def score_pairs_packed_backward_reference(
    dscore: torch.Tensor,             # [num_pairs] fp32 incoming gradient
    queries: torch.Tensor,            # [total_q_tokens, dim]
    query_offsets: torch.Tensor,      # [num_queries + 1]
    documents: torch.Tensor,          # [total_d_tokens, dim]
    document_offsets: torch.Tensor,   # [num_documents + 1]
    pair_query_ids: torch.Tensor,     # [num_pairs]
    pair_document_ids: torch.Tensor,  # [num_pairs]
    argmax: torch.Tensor,             # [num_pairs, max_q_len] int32 from forward
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference backward for the packed maxsim.

    Routes ``g = dscore[k]`` to ``dq[q_start + i] += g * d[d_start + j]`` and
    ``dd[d_start + j] += g * q[q_start + i]`` where ``j = argmax[k, i]``
    and (q_start, d_start) are derived from the offset arrays.

    Returns fp32 gradients matching the kernel.
    """
    qoff = query_offsets.to(torch.int64).cpu().tolist()
    doff = document_offsets.to(torch.int64).cpu().tolist()
    qids = pair_query_ids.to(torch.int64).cpu().tolist()
    dids = pair_document_ids.to(torch.int64).cpu().tolist()
    argmax_cpu = argmax.to(torch.int64).cpu()
    q_f = queries.float()
    d_f = documents.float()
    g_f = dscore.float()

    dq = torch.zeros_like(queries, dtype=torch.float32)
    dd = torch.zeros_like(documents, dtype=torch.float32)

    for k, (qi, di) in enumerate(zip(qids, dids)):
        q_start, q_end = qoff[qi], qoff[qi + 1]
        d_start, d_end = doff[di], doff[di + 1]
        Lq_i = q_end - q_start
        Ld_i = d_end - d_start
        g = g_f[k].item()
        for i in range(Lq_i):
            j = int(argmax_cpu[k, i].item())
            if j < 0 or j >= Ld_i:
                continue
            dq[q_start + i] += g * d_f[d_start + j]
            dd[d_start + j] += g * q_f[q_start + i]

    return dq, dd


def score_contrastive_reference(
    queries: torch.Tensor,    # [Nq, Lq, dim]
    documents: torch.Tensor,  # [total_d_toks, dim]  (packed)
    document_offsets: torch.Tensor, # [Nb + 1] int32       (CSR offsets)
) -> torch.Tensor:
    """Reference for the contrastive maxsim: every query scored against
    every doc.

    Returns:
        ``[Nq, Nb]`` fp32 on the same device as ``queries``.
    """
    Nq = queries.shape[0]
    Nb = document_offsets.numel() - 1
    out = torch.empty((Nq, Nb), dtype=torch.float32, device=queries.device)
    offs = document_offsets.to(torch.int64).cpu().tolist()
    for qi in range(Nq):
        q = queries[qi]
        for di in range(Nb):
            d = documents[offs[di] : offs[di + 1]]
            out[qi, di] = maxsim_reference(q, d)
    return out


def score_contrastive_with_argmax_reference(
    queries: torch.Tensor,    # [Nq, Lq, dim]
    documents: torch.Tensor,  # [total_d_toks, dim]
    document_offsets: torch.Tensor, # [Nb + 1] int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`score_contrastive_reference` but also returns the
    argmax positions per query token.

    Returns:
        ``(scores, argmax)`` — ``scores`` is fp32 ``[Nq, Nb]``; ``argmax``
        is int32 ``[Nq, Nb, Lq]`` with first-index-wins tiebreak.
    """
    Nq, Lq, _ = queries.shape
    Nb = document_offsets.numel() - 1
    scores = torch.empty((Nq, Nb), dtype=torch.float32, device=queries.device)
    argmax = torch.zeros(
        (Nq, Nb, Lq), dtype=torch.int32, device=queries.device
    )
    offs = document_offsets.to(torch.int64).cpu().tolist()
    for qi in range(Nq):
        q = queries[qi]
        for di in range(Nb):
            d = documents[offs[di] : offs[di + 1]]
            s, a = maxsim_reference_with_argmax(q, d)
            scores[qi, di] = s
            argmax[qi, di] = a
    return scores, argmax


def score_contrastive_backward_reference(
    dscore: torch.Tensor,     # [Nq, Nb] fp32, incoming gradient
    queries: torch.Tensor,    # [Nq, Lq, dim]
    documents: torch.Tensor,  # [total_d_toks, dim]
    document_offsets: torch.Tensor, # [Nb + 1] int32
    argmax: torch.Tensor,     # [Nq, Nb, Lq] int32 from forward
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference backward for the contrastive maxsim.

    Routes ``g = dscore[qi, di]`` to ``dq[qi, i] += g * d[di, j]`` and
    ``dd[d_offset + j] += g * q[qi, i]`` where ``j = argmax[qi, di, i]``.

    Both gradients are fp32 (matches kernel).
    """
    Nq, Lq, _ = queries.shape
    Nb = document_offsets.numel() - 1

    dq = torch.zeros_like(queries, dtype=torch.float32)
    dd = torch.zeros_like(documents, dtype=torch.float32)

    offs = document_offsets.to(torch.int64).cpu().tolist()
    argmax_cpu = argmax.to(torch.int64).cpu()
    q_f = queries.float()
    d_f = documents.float()
    g_f = dscore.float()

    for qi in range(Nq):
        for di in range(Nb):
            g = g_f[qi, di].item()
            d_start = offs[di]
            d_end = offs[di + 1]
            Ld_i = d_end - d_start
            for i in range(Lq):
                j = int(argmax_cpu[qi, di, i].item())
                if j < 0 or j >= Ld_i:
                    continue
                dq[qi, i] += g * d_f[d_start + j]
                dd[d_start + j] += g * q_f[qi, i]

    return dq, dd


def score_candidates_padded_with_argmax_reference(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`score_candidates_padded_reference` but also returns argmax
    positions per query token. Tiebreak is PyTorch's first-index-wins.

    Args:
        queries: ``[B, Lq, dim]``
        documents: ``[B, C, Ld, dim]``
        query_lengths: ``[B]``
        doc_lengths: ``[B, C]``

    Returns:
        ``(scores, argmax)`` — ``scores`` is fp32 ``[B, C]``; ``argmax`` is
        int32 ``[B, C, Lq]``. Slots beyond ``query_lengths[b]`` are filled
        with 0.
    """
    B, C = doc_lengths.shape
    Lq = queries.shape[1]
    scores = torch.empty((B, C), dtype=torch.float32, device=queries.device)
    argmax = torch.zeros(
        (B, C, Lq), dtype=torch.int32, device=queries.device
    )
    qlen = query_lengths.to(torch.int64).cpu().tolist()
    dlen = doc_lengths.to(torch.int64).cpu().tolist()
    for b in range(B):
        q = queries[b, : qlen[b]]
        for c in range(C):
            d = documents[b, c, : dlen[b][c]]
            s, a = maxsim_reference_with_argmax(q, d)
            scores[b, c] = s
            argmax[b, c, : a.numel()] = a
    return scores, argmax
