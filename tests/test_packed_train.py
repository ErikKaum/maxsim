"""Tests for the packed (ragged) training API.

Mirrors test_backward.py and test_contrastive.py but for the packed form:
ragged queries, ragged docs, arbitrary (q_id, d_id) pair mapping. Scalar
argmax variant; WMMA argmax for packed is post-V2.
"""

from __future__ import annotations

import pytest
import torch

import maxsim
from maxsim._ops import ops
from ._helpers import (
    DEVICE,
    TOLERANCES,
    make_random_packed_batch,
    supported_float_dtypes,
)


def _packed_dtypes():
    """fp16 + bf16. fp32 forward is fast; the argmax path supports all
    three but we keep tests symmetric with the other training surfaces."""
    return [dt for dt in supported_float_dtypes() if dt != torch.float32]


@pytest.mark.parametrize("dtype", _packed_dtypes())
def test_packed_argmax_matches_reference(dtype) -> None:
    batch = make_random_packed_batch(
        num_queries=3,
        num_documents=4,
        num_pairs=6,
        dim=64,
        q_len_range=(8, 24),
        d_len_range=(16, 96),
        dtype=dtype,
        seed=0,
    )
    scores_k, argmax_k = maxsim.score_pairs_packed_with_argmax(*batch.args())
    max_q_len = int(
        (batch.query_offsets[1:].to(torch.int64)
         - batch.query_offsets[:-1].to(torch.int64)).max().item()
    )
    scores_ref, argmax_ref = (
        maxsim.score_pairs_packed_with_argmax_reference(
            *batch.args(), max_q_len=max_q_len,
        )
    )

    torch.testing.assert_close(scores_k, scores_ref, **TOLERANCES[dtype])
    # argmax must match exactly on valid slots.
    qoff = batch.query_offsets.to(torch.int64).cpu()
    qids = batch.pair_query_ids.to(torch.int64).cpu()
    for k in range(int(batch.pair_query_ids.numel())):
        Lq_k = int(qoff[qids[k] + 1] - qoff[qids[k]])
        assert torch.equal(
            argmax_k[k, :Lq_k].cpu(),
            argmax_ref[k, :Lq_k].cpu(),
        ), f"argmax mismatch at pair {k}"


@pytest.mark.parametrize("dtype", _packed_dtypes())
def test_packed_first_index_wins_on_ties(dtype) -> None:
    """All-identical doc tokens for one query: argmax must be 0 for every
    valid q_tok (PyTorch first-index-wins). Scalar kernel — no WMMA/scalar
    precision asymmetry to worry about; any Ld works."""
    dim = 32
    Lq, Ld = 12, 47
    base = torch.randn(dim, device=DEVICE, dtype=dtype)
    q = torch.randn(Lq, dim, device=DEVICE, dtype=dtype)
    d = base.unsqueeze(0).expand(Ld, dim).contiguous()

    queries = q
    documents = d
    query_offsets = torch.tensor([0, Lq], dtype=torch.int32, device=DEVICE)
    document_offsets = torch.tensor([0, Ld], dtype=torch.int32, device=DEVICE)
    pair_query_ids = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    pair_document_ids = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    _, argmax = maxsim.score_pairs_packed_with_argmax(
        queries, query_offsets, documents, document_offsets,
        pair_query_ids, pair_document_ids,
    )
    assert torch.all(argmax[:, :Lq] == 0), \
        f"first-index-wins violated: argmax = {argmax.tolist()}"


def _autograd_grads_via_reference(batch, dscore):
    """Reference grads via torch.autograd through the pure-PyTorch forward."""
    q_ref = batch.queries.detach().clone().float().requires_grad_(True)
    d_ref = batch.documents.detach().clone().float().requires_grad_(True)
    scores = maxsim.score_pairs_packed_reference(
        q_ref, batch.query_offsets,
        d_ref, batch.document_offsets,
        batch.pair_query_ids, batch.pair_document_ids,
    )
    (scores * dscore).sum().backward()
    return q_ref.grad.detach(), d_ref.grad.detach()


@pytest.mark.parametrize("dtype", _packed_dtypes())
def test_packed_backward_matches_autograd(dtype) -> None:
    batch = make_random_packed_batch(
        num_queries=3,
        num_documents=4,
        num_pairs=6,
        dim=64,
        q_len_range=(8, 24),
        d_len_range=(16, 96),
        dtype=dtype,
        seed=1,
    )
    dscore = torch.randn(
        int(batch.pair_query_ids.numel()), device=DEVICE, dtype=torch.float32
    )
    _, argmax = maxsim.score_pairs_packed_with_argmax(*batch.args())
    max_q_len = int(
        (batch.query_offsets[1:].to(torch.int64)
         - batch.query_offsets[:-1].to(torch.int64)).max().item()
    )
    dq_k, dd_k = ops.maxsim_packed_backward(
        dscore.contiguous(),
        batch.queries.contiguous(),
        batch.query_offsets.contiguous(),
        batch.documents.contiguous(),
        batch.document_offsets.contiguous(),
        batch.pair_query_ids.contiguous(),
        batch.pair_document_ids.contiguous(),
        argmax,
        max_q_len,
    )
    dq_ref, dd_ref = _autograd_grads_via_reference(batch, dscore)
    tol = TOLERANCES[dtype]
    torch.testing.assert_close(dq_k, dq_ref, **tol)
    torch.testing.assert_close(dd_k, dd_ref, **tol)


@pytest.mark.parametrize("dtype", _packed_dtypes())
def test_score_pairs_packed_train_end_to_end(dtype) -> None:
    batch = make_random_packed_batch(
        num_queries=3,
        num_documents=4,
        num_pairs=6,
        dim=64,
        q_len_range=(8, 24),
        d_len_range=(16, 96),
        dtype=dtype,
        seed=2,
    )
    # Promote queries/documents to require_grad.
    q = batch.queries.detach().clone().requires_grad_(True)
    d = batch.documents.detach().clone().requires_grad_(True)
    scores = maxsim.score_pairs_packed_train(
        q, batch.query_offsets,
        d, batch.document_offsets,
        batch.pair_query_ids, batch.pair_document_ids,
    )
    weights = torch.randn(
        scores.shape, device=DEVICE, dtype=torch.float32
    )
    (scores * weights).sum().backward()

    dq_ref, dd_ref = _autograd_grads_via_reference(batch, weights)
    tol = TOLERANCES[dtype]
    torch.testing.assert_close(q.grad.float(), dq_ref, **tol)
    torch.testing.assert_close(d.grad.float(), dd_ref, **tol)
