"""Smoke test for the public maxsim API.

Heavier numerical / dtype / shape coverage lives in the sibling test files.
"""

from __future__ import annotations

import torch

import maxsim

from ._helpers import DEVICE, TOLERANCES, make_random_packed_batch


def test_score_pairs_packed_smoke() -> None:
    batch = make_random_packed_batch(
        num_queries=3,
        num_documents=4,
        num_pairs=5,
        dim=64,
        q_len_range=(8, 16),
        d_len_range=(16, 64),
        dtype=torch.float32,
        seed=0,
    )

    scores = maxsim.score_pairs_packed(*batch.args())

    assert scores.shape == (5,)
    assert scores.dtype == torch.float32
    assert scores.device.type == DEVICE.type

    expected = maxsim.score_pairs_packed_reference(*batch.args())
    torch.testing.assert_close(scores, expected, **TOLERANCES[torch.float32])
