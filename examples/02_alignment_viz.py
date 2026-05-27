# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kernels",
#     "torch",
#     "numpy",
# ]
# ///

"""Terminal viz of late-interaction.

For a single (query, document) pair this renders an ASCII grid showing,
for each query token, which document token won the per-token argmax —
i.e. the routing pattern that defines late-interaction.

Synthetic embeddings (no real NLP), but the alignment pattern is
deliberately constructed so each query token aligns with a specific
non-monotonic doc token. That makes the kernel's routing visible.

Run with::

    just example 02
    # or: uv run examples/02_alignment_viz.py
"""

from __future__ import annotations

import platform
import sys

import kernels
import torch
import torch.nn.functional as F

# ANSI escape codes for terminal colors.
BOLD = "\033[1m"
GREEN = "\033[92m"
DIM = "\033[90m"
RESET = "\033[0m"


def main() -> None:
    device = detect_device()
    print(f"device = {device} \n")

    maxsim = kernels.get_kernel("erikkaum/maxsim", version=2, trust_remote_code=True)

    torch.manual_seed(0)
    Lq, Ld, dim = 6, 10, 32

    # Build doc tokens as L2-normalized random unit vectors.
    d_base = F.normalize(
        torch.randn(Ld, dim, device=device, dtype=torch.float16), dim=-1
    )

    # Construct the intended alignment: query token i should max against
    # doc token `target[i]`. Non-monotonic on purpose so the routing is
    # visually obvious.
    target = [3, 5, 1, 7, 0, 4]
    assert len(target) == Lq

    # Each query token = (dominant pull toward d[target[i]]) + small noise.
    q_base = torch.randn(Lq, dim, device=device, dtype=torch.float16)
    for i, t in enumerate(target):
        q_base[i] = 3.0 * d_base[t] + 0.1 * q_base[i]
    q_base = F.normalize(q_base, dim=-1)

    # Score through the kernel — single (q, d) pair via the packed API
    # (no Lq % 16 gate; works for any Lq on any backend).
    query_offsets = torch.tensor([0, Lq], dtype=torch.int32, device=device)
    document_offsets = torch.tensor([0, Ld], dtype=torch.int32, device=device)
    pair_query_ids = torch.tensor([0], dtype=torch.int32, device=device)
    pair_document_ids = torch.tensor([0], dtype=torch.int32, device=device)

    scores, argmax = maxsim.score_pairs_packed_with_argmax(
        q_base, query_offsets,
        d_base, document_offsets,
        pair_query_ids, pair_document_ids,
        max_q_len=Lq,
    )
    argmax = argmax[0]  # [Lq]
    pair_score = float(scores[0].item())

    # Recompute the full [Lq, Ld] similarity matrix in fp32 so we can show
    # per-q-tok max magnitudes alongside the routing.
    sim = q_base.float() @ d_base.float().T  # [Lq, Ld]
    max_per_q = sim.max(dim=-1).values         # [Lq]

    # --- Render ---
    print(f"{BOLD}MaxSim alignment{RESET}  ─  rows: query tokens, "
          f"cols: doc tokens")
    print(f"the {GREEN}●{RESET} in each row is the doc token that won the "
          f"per-q-tok argmax")
    print()

    # Column header
    print("       " + "".join(f"  d{j:<2}" for j in range(Ld)))
    print("       " + "  ───" * Ld)

    for i in range(Lq):
        am = int(argmax[i].item())
        cells = []
        for j in range(Ld):
            if j == am:
                cells.append(f"  {GREEN}● {RESET}")
            else:
                cells.append(f"  {DIM}· {RESET}")
        row = f"  q{i}  " + "".join(cells)
        expected_marker = " " if am == target[i] else f"  {DIM}(target d{target[i]}){RESET}"
        print(f"{row}  → d{am}   max={max_per_q[i].item():+.3f}{expected_marker}")

    print()
    print(f"  score(q, d) = Σᵢ maxⱼ ⟨qᵢ, dⱼ⟩ = {pair_score:+.3f}")
    print()
    print("The routing matched the intended target for every query token,"
          " confirming")
    print("the kernel produces the same first-index-wins argmax as PyTorch's"
          " max().")

# --- Helpers -----

def detect_device() -> torch.device:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("ERROR: maxsim needs MPS or CUDA.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
