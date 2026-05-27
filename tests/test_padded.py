"""Correctness of the padded API."""

from __future__ import annotations

import pytest
import torch

import maxsim

from ._helpers import (
    DEVICE,
    TOLERANCES,
    make_random_padded_batch,
    supported_float_dtypes,
)


@pytest.mark.parametrize("dtype", supported_float_dtypes())
def test_padded_matches_reference(dtype: torch.dtype) -> None:
    queries, documents, qlen, dlen = make_random_padded_batch(
        batch=4,
        candidates=5,
        q_len_max=32,
        d_len_max=128,
        dim=64,
        dtype=dtype,
        seed=2024,
    )
    actual = maxsim.score_candidates_padded(queries, documents, qlen, dlen)
    expected = maxsim.score_candidates_padded_reference(
        queries, documents, qlen, dlen
    )
    assert actual.shape == expected.shape == (4, 5)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])


def test_padded_ignores_pad_positions() -> None:
    """Garbage past the declared lengths must not affect the result."""
    B, C, Lq, Ld, D = 2, 3, 8, 16, 32
    queries = torch.randn(B, Lq, D, dtype=torch.float32, device=DEVICE)
    documents = torch.randn(B, C, Ld, D, dtype=torch.float32, device=DEVICE)
    qlen = torch.tensor([4, 6], dtype=torch.int32, device=DEVICE)
    dlen = torch.tensor(
        [[10, 8, 4], [16, 1, 12]], dtype=torch.int32, device=DEVICE
    )

    baseline = maxsim.score_candidates_padded(queries, documents, qlen, dlen)

    # Overwrite padded positions with huge values: result must not change.
    queries_corrupt = queries.clone()
    queries_corrupt[0, 4:] = 1e4
    queries_corrupt[1, 6:] = -1e4
    documents_corrupt = documents.clone()
    documents_corrupt[0, 0, 10:] = 1e4
    documents_corrupt[1, 1, 1:] = -1e4

    out = maxsim.score_candidates_padded(
        queries_corrupt, documents_corrupt, qlen, dlen
    )
    torch.testing.assert_close(out, baseline, **TOLERANCES[torch.float32])
