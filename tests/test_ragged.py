"""Ragged-shape stress tests for `score_pairs_packed`."""

from __future__ import annotations

import pytest
import torch

import maxsim

from ._helpers import DEVICE, TOLERANCES, make_random_packed_batch


@pytest.mark.parametrize(
    "q_len_range,d_len_range",
    [
        ((1, 1), (1, 1)),          # smallest segments
        ((1, 64), (1, 64)),        # extremely ragged short
        ((4, 32), (16, 1024)),     # long doc stress
        ((32, 32), (128, 128)),    # uniform (regression check)
    ],
)
def test_ragged_shapes(q_len_range, d_len_range) -> None:
    batch = make_random_packed_batch(
        num_queries=5,
        num_documents=7,
        num_pairs=13,
        dim=64,
        q_len_range=q_len_range,
        d_len_range=d_len_range,
        dtype=torch.float32,
        seed=123,
    )
    actual = maxsim.score_pairs_packed(*batch.args())
    expected = maxsim.score_pairs_packed_reference(*batch.args())
    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.float32])


def test_single_query_many_pairs() -> None:
    """Many pairs that all reference the same query id."""
    batch = make_random_packed_batch(
        num_queries=1,
        num_documents=32,
        num_pairs=32,
        dim=128,
        q_len_range=(8, 8),
        d_len_range=(4, 200),
        dtype=torch.float32,
        seed=7,
    )
    pq = torch.zeros_like(batch.pair_query_ids)
    pd = torch.arange(32, dtype=torch.int32, device=DEVICE)

    actual = maxsim.score_pairs_packed(
        batch.queries, batch.query_offsets, batch.documents, batch.document_offsets,
        pq, pd,
    )
    expected = maxsim.score_pairs_packed_reference(
        batch.queries, batch.query_offsets, batch.documents, batch.document_offsets,
        pq, pd,
    )
    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.float32])


def test_rejects_empty_query_segment() -> None:
    queries = torch.randn(4, 32, dtype=torch.float32, device=DEVICE)
    # Second segment is empty (1 -> 1).
    q_off = torch.tensor([0, 1, 1, 4], dtype=torch.int32, device=DEVICE)
    documents = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    d_off = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEVICE)
    pq = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    pd = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError, match="empty segment"):
        maxsim.score_pairs_packed(queries, q_off, documents, d_off, pq, pd)


def test_rejects_out_of_range_pair_id() -> None:
    queries = torch.randn(4, 32, dtype=torch.float32, device=DEVICE)
    q_off = torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE)
    documents = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    d_off = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEVICE)
    pq = torch.tensor([5], dtype=torch.int32, device=DEVICE)  # only 2 queries
    pd = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError, match="out of range"):
        maxsim.score_pairs_packed(queries, q_off, documents, d_off, pq, pd)


def test_rejects_dtype_mismatch_between_q_and_d() -> None:
    queries = torch.randn(4, 32, dtype=torch.float32, device=DEVICE)
    documents = torch.randn(4, 32, dtype=torch.float16, device=DEVICE)
    q_off = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
    d_off = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
    pq = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    pd = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises(TypeError, match="same dtype"):
        maxsim.score_pairs_packed(queries, q_off, documents, d_off, pq, pd)


def test_rejects_dim_mismatch() -> None:
    queries = torch.randn(4, 32, dtype=torch.float32, device=DEVICE)
    documents = torch.randn(4, 16, dtype=torch.float32, device=DEVICE)
    q_off = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
    d_off = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
    pq = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    pd = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises(ValueError, match="dim"):
        maxsim.score_pairs_packed(queries, q_off, documents, d_off, pq, pd)
