# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kernels",
#     "torch",
#     "numpy"
# ]
# ///

"""Quickstart: 30 seconds to grasp the maxsim API.

Three things:

  1. Inference   — score a batch of (query, doc) pairs.
  2. Training    — same forward, with autograd-propagating gradients.
  3. Where to go — pointers to the other examples for deeper dives.

Run with::

    just example 01
    # or: uv run examples/01_quickstart.py
"""

from __future__ import annotations

import platform
import sys

import kernels
import torch

def main() -> None:
    device = detect_device()
    print(f"device = {device}\n")

    maxsim = kernels.get_kernel("erikkaum/maxsim", version=2, trust_remote_code=True)

    # ----- 1. Inference: padded reranking -----
    B, C, Lq, Ld, dim = 2, 4, 16, 32, 128
    queries = torch.randn(B, Lq, dim, device=device, dtype=torch.float16)
    documents = torch.randn(B, C, Ld, dim, device=device, dtype=torch.float16)
    qlen = torch.full((B,), Lq, dtype=torch.int32, device=device)
    dlen = torch.full((B, C), Ld, dtype=torch.int32, device=device)

    scores = maxsim.score_candidates_padded(queries, documents, qlen, dlen)
    print(f"inference  scores.shape = {tuple(scores.shape)}  "
          f"dtype = {scores.dtype}")
    print(f"           first row    = {scores[0].tolist()}")

    # ----- 2. Training: autograd-propagating forward -----
    queries.requires_grad_(True)
    documents.requires_grad_(True)
    scores = maxsim.score_candidates_padded_train(
        queries, documents, qlen, dlen
    )
    loss = scores.sum()
    loss.backward()
    print(f"training   loss = {loss.item():.3f}")
    print(f"           queries.grad.shape = {tuple(queries.grad.shape)}  "
          f"(kernel produces fp32 grads; PyTorch casts to input dtype "
          f"for storage)")

    # ----- 3. Where to go next -----
    print()
    print("Next:")
    print("  examples/02_alignment_viz.py     — what late-interaction looks like")
    print("  examples/03_contrastive_training.py — full InfoNCE loop with a tiny encoder")
    print("  examples/04_memory_showcase.py   — OOM-ceiling demo vs naive baseline")

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
