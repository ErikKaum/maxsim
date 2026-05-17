"""Dtype coverage tests: fp32, fp16, bf16 inputs all match the fp32 reference."""

from __future__ import annotations

import pytest
import torch

import maxsim

from ._helpers import TOLERANCES, make_random_packed_batch, supported_float_dtypes


@pytest.mark.parametrize("dtype", supported_float_dtypes())
def test_packed_dtype(dtype: torch.dtype) -> None:
    batch = make_random_packed_batch(
        num_queries=4,
        num_documents=6,
        num_pairs=9,
        dim=128,
        q_len_range=(8, 32),
        d_len_range=(16, 256),
        dtype=dtype,
        seed=99,
    )

    actual = maxsim.score_pairs_packed(*batch.args())
    expected = maxsim.score_pairs_packed_reference(*batch.args())

    assert actual.dtype == torch.float32, "kernel must always return fp32"
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", supported_float_dtypes())
def test_packed_dtype_large_doc(dtype: torch.dtype) -> None:
    """Longer docs trigger multiple iterations of the tile loop per thread."""
    batch = make_random_packed_batch(
        num_queries=2,
        num_documents=3,
        num_pairs=4,
        dim=128,
        q_len_range=(8, 16),
        d_len_range=(512, 1024),
        dtype=dtype,
        seed=101,
    )
    actual = maxsim.score_pairs_packed(*batch.args())
    expected = maxsim.score_pairs_packed_reference(*batch.args())
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])
