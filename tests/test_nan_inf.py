"""NaN / inf input behavior.

If a user's encoder produces non-finite values upstream (e.g. fp16
gradient blow-up before clipping, or autocast quirks), what does the
maxsim kernel do? This file pins down the actual contract:

  * **NaN in queries → -inf scores** for every pair touching that query.
    This diverges from ``torch.max``, which propagates NaN. The cause is
    that the kernel's tile cascade uses ``v > my_max`` to update the
    running max; NaN never satisfies ``>``, so NaN d_tok candidates are
    skipped and my_max stays at its initialisation sentinel (-inf).
    Documented divergence; fix is post-V2 work.

  * **+inf in queries (aligned-sign documents) → +inf scores.**
    Monotone behaviour through the kernel.

  * **Backward through non-finite scores → non-finite gradients.**
    Tested explicitly so users debugging NaN loss know to look upstream.

Anyone debugging a non-finite loss should pre-screen their embeddings;
this test pins down what to expect.
"""

from __future__ import annotations

import pytest
import torch

import maxsim
from ._helpers import DEVICE


@pytest.mark.parametrize("api", ["padded", "contrastive", "packed"])
def test_nan_in_input_produces_neg_inf_scores(api) -> None:
    """A NaN-poisoned query token causes every score touching that query
    to come out as -inf (NOT NaN). This is because the kernel cascade uses
    ``v > my_max`` to update; NaN fails that compare, so my_max stays at
    its -inf init.

    This diverges from ``torch.max``, which would propagate NaN. Documented
    behaviour; if your encoder emits NaN it is still a bug, just one whose
    symptom is a -inf score rather than a NaN one."""
    torch.manual_seed(0)
    dtype = torch.float16

    if api == "padded":
        B, C, Lq, Ld, D = 2, 3, 16, 32, 64
        q = torch.randn(B, Lq, D, device=DEVICE, dtype=dtype)
        d = torch.randn(B, C, Ld, D, device=DEVICE, dtype=dtype)
        qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
        dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)
        # Poison the second batch's first query token.
        q[1, 0, 0] = float("nan")
        scores = maxsim.score_candidates_padded(q, d, qlen, dlen)
        assert torch.isinf(scores[1]).all() and (scores[1] < 0).all(), \
            f"expected -inf in scores[1], got {scores[1]}"
        assert torch.isfinite(scores[0]).all(), \
            f"-inf leaked to scores[0]: {scores[0]}"

    elif api == "contrastive":
        Nq, Nb, Lq, Ld, D = 2, 3, 16, 32, 64
        q = torch.randn(Nq, Lq, D, device=DEVICE, dtype=dtype)
        d = torch.randn(Nb * Ld, D, device=DEVICE, dtype=dtype)
        cu = torch.arange(0, (Nb + 1) * Ld, Ld,
                          dtype=torch.int32, device=DEVICE)
        q[1, 0, 0] = float("nan")
        scores = maxsim.score_contrastive(q, d, cu)
        assert torch.isinf(scores[1]).all() and (scores[1] < 0).all(), \
            f"expected -inf in scores[1], got {scores[1]}"
        assert torch.isfinite(scores[0]).all(), \
            f"-inf leaked to scores[0]: {scores[0]}"

    else:  # packed
        Nq, Nb, Lq, Ld, D = 2, 3, 16, 32, 64
        q = torch.randn(Nq * Lq, D, device=DEVICE, dtype=dtype)
        d = torch.randn(Nb * Ld, D, device=DEVICE, dtype=dtype)
        qoff = torch.arange(0, (Nq + 1) * Lq, Lq,
                            dtype=torch.int32, device=DEVICE)
        doff = torch.arange(0, (Nb + 1) * Ld, Ld,
                            dtype=torch.int32, device=DEVICE)
        qids = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        dids = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
        # Poison query 1's first token.
        q[Lq] = float("nan")
        scores = maxsim.score_pairs_packed(
            q, qoff, d, doff, qids, dids, max_q_len=Lq,
        )
        # Pair 0 uses query 0 (clean); pair 1 uses query 1 (poisoned).
        assert torch.isfinite(scores[0]), \
            f"non-finite leaked to clean pair: scores[0]={scores[0]}"
        assert torch.isinf(scores[1]) and scores[1] < 0, \
            f"expected -inf in scores[1], got {scores[1]}"


@pytest.mark.parametrize("api", ["padded", "contrastive"])
def test_inf_in_input_propagates(api) -> None:
    """+inf in a query embedding produces +inf in the score (since maxsim is
    a sum of maxes, both monotone in inputs)."""
    torch.manual_seed(1)
    dtype = torch.float16

    if api == "padded":
        B, C, Lq, Ld, D = 1, 2, 16, 32, 64
        q = torch.randn(B, Lq, D, device=DEVICE, dtype=dtype)
        d = torch.randn(B, C, Ld, D, device=DEVICE, dtype=dtype)
        qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
        dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)
        # Align signs so dot product is +inf (rather than -inf or NaN).
        q[0, 0] = float("inf")
        d[0, :, :, :] = d[0, :, :, :].abs()
        scores = maxsim.score_candidates_padded(q, d, qlen, dlen)
        assert torch.isinf(scores).all(), f"expected +inf scores, got {scores}"
        assert (scores > 0).all()

    else:  # contrastive
        Nq, Nb, Lq, Ld, D = 1, 2, 16, 32, 64
        q = torch.randn(Nq, Lq, D, device=DEVICE, dtype=dtype)
        d = torch.randn(Nb * Ld, D, device=DEVICE, dtype=dtype).abs()
        cu = torch.arange(0, (Nb + 1) * Ld, Ld,
                          dtype=torch.int32, device=DEVICE)
        q[0, 0] = float("inf")
        scores = maxsim.score_contrastive(q, d, cu)
        assert torch.isinf(scores).all(), f"expected +inf scores, got {scores}"
        assert (scores > 0).all()


def test_nan_in_backward_propagates_to_grads() -> None:
    """If the kernel's forward produces a NaN score and we backward through
    a finite dscore, gradients land on whatever the kernel multiplies the
    dscore with — which for NaN-poisoned inputs ends up non-finite."""
    torch.manual_seed(2)
    dtype = torch.float16
    Nq, Nb, Lq, Ld, D = 1, 2, 16, 32, 64
    q = torch.randn(Nq, Lq, D, device=DEVICE, dtype=dtype, requires_grad=True)
    d = torch.randn(Nb * Ld, D, device=DEVICE, dtype=dtype, requires_grad=True)
    cu = torch.arange(0, (Nb + 1) * Ld, Ld,
                      dtype=torch.int32, device=DEVICE)
    # Poison one query token.
    with torch.no_grad():
        q[0, 0, 0] = float("nan")
    scores = maxsim.score_contrastive_train(q, d, cu)
    scores.sum().backward()
    # We don't make claims about every grad slot, only that the kernel
    # didn't silently return zeros for the poisoned path.
    assert not torch.isfinite(q.grad).all() or not torch.isfinite(d.grad).all(), \
        "backward through NaN forward gave all-finite grads — kernel may be sanitising silently"
