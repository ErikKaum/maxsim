"""Numerical correctness of `score_pairs_packed` against the pure PyTorch reference."""

from __future__ import annotations

import pytest
import torch

import maxsim

from ._helpers import (
    DEVICE,
    TOLERANCES,
    make_random_packed_batch,
)


@pytest.mark.parametrize("dim", [64, 96, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_matches_reference_fp32(dim: int, seed: int) -> None:
    batch = make_random_packed_batch(
        num_queries=6,
        num_documents=11,
        num_pairs=17,
        dim=dim,
        q_len_range=(4, 32),
        d_len_range=(8, 200),
        dtype=torch.float32,
        seed=seed,
    )

    actual = maxsim.score_pairs_packed(*batch.args())
    expected = maxsim.score_pairs_packed_reference(*batch.args())

    assert actual.dtype == torch.float32
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.float32])


def test_single_pair_single_token() -> None:
    """Smallest non-trivial input: 1 query token, 1 doc token, 1 pair."""
    q = torch.randn(1, 16, dtype=torch.float32, device=DEVICE)
    d = torch.randn(1, 16, dtype=torch.float32, device=DEVICE)
    q_off = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    d_off = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    pq = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    pd = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    actual = maxsim.score_pairs_packed(q, q_off, d, d_off, pq, pd)
    expected = (q.float() @ d.float().T).max(dim=-1).values.sum().reshape(1)
    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.float32])


def test_empty_pairs_returns_empty_tensor() -> None:
    batch = make_random_packed_batch(
        num_queries=2,
        num_documents=3,
        num_pairs=0,
        dim=32,
        q_len_range=(2, 8),
        d_len_range=(2, 16),
        dtype=torch.float32,
    )
    out = maxsim.score_pairs_packed(*batch.args())
    assert out.shape == (0,)
    assert out.dtype == torch.float32


def test_shared_queries_and_documents() -> None:
    """Multiple pairs sharing the same query or document id should all be valid."""
    gen = torch.Generator().manual_seed(42)
    dim = 32
    queries = torch.randn(20, dim, generator=gen).to(DEVICE)
    documents = torch.randn(30, dim, generator=gen).to(DEVICE)

    # 4 queries, 3 documents.
    q_off = torch.tensor([0, 4, 9, 14, 20], dtype=torch.int32, device=DEVICE)
    d_off = torch.tensor([0, 10, 20, 30], dtype=torch.int32, device=DEVICE)

    # 6 pairs, some sharing query / document.
    pq = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int32, device=DEVICE)
    pd = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.int32, device=DEVICE)

    actual = maxsim.score_pairs_packed(queries, q_off, documents, d_off, pq, pd)
    expected = maxsim.score_pairs_packed_reference(
        queries, q_off, documents, d_off, pq, pd
    )
    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.float32])
