"""Tests for the forward + argmax kernel.

The kernel ships the per-q-tok argmax document index alongside the score so
backward / training can route gradients without materialising the full
``[Lq, Ld]`` similarity matrix. Tiebreak should match PyTorch's
first-index-wins (``torch.max(dim=-1)`` returns the first index on ties).
"""

from __future__ import annotations

import math

import pytest
import torch

import maxsim
from ._helpers import DEVICE, TOLERANCES, supported_float_dtypes


def _argmax_dtypes():
    """argmax kernel currently supports fp16/bf16 across all backends; the
    CUDA fp32 path is future work (scalar fallback)."""
    return [d for d in supported_float_dtypes()
            if d in (torch.float16, torch.bfloat16)]


@pytest.mark.parametrize("dtype", _argmax_dtypes())
def test_padded_argmax_matches_reference_random(dtype) -> None:
    """Random padded batch: scores and argmax must match the reference."""
    torch.manual_seed(0)
    B, C, Lq, Ld, D = 4, 6, 32, 64, 128

    queries = torch.randn(B, Lq, D, device=DEVICE, dtype=dtype)
    documents = torch.randn(B, C, Ld, D, device=DEVICE, dtype=dtype)
    query_lengths = torch.randint(
        low=8, high=Lq + 1, size=(B,), dtype=torch.int32, device=DEVICE
    )
    doc_lengths = torch.randint(
        low=16, high=Ld + 1, size=(B, C), dtype=torch.int32, device=DEVICE
    )

    scores, argmax = maxsim.score_candidates_padded_with_argmax(
        queries, documents, query_lengths, doc_lengths
    )
    ref_scores, ref_argmax = (
        maxsim.score_candidates_padded_with_argmax_reference(
            queries, documents, query_lengths, doc_lengths
        )
    )

    torch.testing.assert_close(scores, ref_scores, **TOLERANCES[dtype])

    # Argmax comparison must hold on all VALID slots — i.e. q_tok < Lq[b].
    # Padding slots are 0 in both the kernel and the reference.
    qlen_cpu = query_lengths.to(torch.int64).cpu()
    for b in range(B):
        valid = int(qlen_cpu[b].item())
        assert torch.equal(
            argmax[b, :, :valid].cpu(),
            ref_argmax[b, :, :valid].cpu(),
        ), (
            f"argmax mismatch in batch b={b}; valid slots [0, {valid})"
        )


def test_padded_argmax_first_index_wins_on_ties() -> None:
    """Construct deliberate ties and verify kernel matches PyTorch (first
    index wins). All doc tokens are identical, so ``<q_i, d_j>`` is the same
    for every j and the argmax must be 0 for every query token.

    Shapes chosen to satisfy CUDA's WMMA eligibility gate (Lq_max % 16 == 0,
    dim % 16 == 0). Metal accepts more, but we keep the test cross-backend.
    """
    dtype = torch.float16
    B, C, Lq, Ld, D = 1, 1, 16, 16, 16

    # All d_j identical -> every (q_i, d_j) dot product equals <q_i, d_0>.
    base = torch.randn(D, device=DEVICE, dtype=dtype)
    documents = base.view(1, 1, 1, D).expand(B, C, Ld, D).contiguous()
    queries = torch.randn(B, Lq, D, device=DEVICE, dtype=dtype)
    query_lengths = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
    doc_lengths = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

    _, argmax = maxsim.score_candidates_padded_with_argmax(
        queries, documents, query_lengths, doc_lengths
    )

    # Every q_tok should pick j=0 because it's the FIRST index that achieves
    # the (tied-with-everyone) max.
    assert torch.all(argmax == 0), f"got argmax = {argmax.tolist()}"
