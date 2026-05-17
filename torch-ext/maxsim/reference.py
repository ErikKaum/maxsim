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
