# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kernels",
#     "torch",
# ]
# ///

"""End-to-end training: tiny encoder + in-batch contrastive InfoNCE.

A learnable embedding "encoder" (stand-in for a transformer) is trained
to push paired (query, document) examples together and everything else
apart. Shows the standard ColBERT recipe minus the real BERT — same
training loop you'd plug a real encoder into.

Note the mixed-precision pattern: fp32 weights (AdamW's exponential
moving averages need the dynamic range) with fp16 only at the kernel
boundary. Real ColBERT training uses this same shape.

Run with::

    just example 03
    # or: uv run examples/03_contrastive_training.py
"""

from __future__ import annotations

import platform
import sys
import time

import kernels
import torch
import torch.nn.functional as F

class TinyEncoder(torch.nn.Module):
    """Stand-in for a real transformer: token-id → per-token embedding."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(ids))


def main() -> None:
    device = detect_device()
    print(f"device = {device}\n")

    maxsim = kernels.get_kernel("erikkaum/maxsim", version=2, trust_remote_code=True)
    
    # fp32 weights, fp16 only at the kernel boundary. Matches real
    # mixed-precision ColBERT training.
    encoder_dtype = torch.float32
    kernel_dtype = torch.float16

    Nq, Nb, Lq, Ld, dim = 16, 16, 32, 64, 128
    vocab = 100
    n_steps = 100
    lr = 1e-3

    torch.manual_seed(42)
    encoder_q = TinyEncoder(vocab, dim).to(device, dtype=encoder_dtype)
    encoder_d = TinyEncoder(vocab, dim).to(device, dtype=encoder_dtype)
    optimizer = torch.optim.AdamW(
        list(encoder_q.parameters()) + list(encoder_d.parameters()),
        lr=lr,
    )

    # Synthetic in-batch contrastive data. For each i in [0, Nq) we
    # make query_i and doc_i share an overlapping token prefix so the
    # diagonal pair has a learnable signal.
    q_ids = torch.randint(0, vocab, (Nq, Lq), device=device)
    d_ids = torch.randint(0, vocab, (Nb, Ld), device=device)
    overlap = min(Lq, Ld) // 4
    for i in range(min(Nq, Nb)):
        d_ids[i, :overlap] = q_ids[i, :overlap]

    targets = torch.arange(Nq, device=device)  # diagonal positives

    # Packed document_offsets. Fixed Ld here for simplicity, but ragged is fine.
    document_offsets = torch.arange(
        0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=device
    )

    losses: list[float] = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        # Encoder forward — fp32 throughout.
        q_emb = encoder_q(q_ids)                          # [Nq, Lq, dim]
        d_emb = encoder_d(d_ids).reshape(Nb * Ld, dim)    # packed

        # Cast to the kernel dtype at the kernel boundary. Gradients flow
        # back through .to() automatically.
        scores = maxsim.score_contrastive_train(
            q_emb.to(kernel_dtype),
            d_emb.to(kernel_dtype),
            document_offsets,
        )                                                  # [Nq, Nb]

        # InfoNCE. The 1/sqrt(Nb) scale keeps softmax temperature reasonable
        # (scores are O(Lq) magnitude before scaling).
        logits = scores / (Nb ** 0.5)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        if step % 10 == 0 or step == n_steps - 1:
            print(f"step {step:3d}  loss = {loss.item():.4f}")

    elapsed = time.perf_counter() - t0
    per_step_ms = elapsed * 1000 / n_steps
    print()
    print(f"trained {n_steps} steps in {elapsed:.2f}s "
          f"({per_step_ms:.2f} ms/step including encoder forward/backward)")
    print(f"loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    if losses[-1] >= losses[0]:
        print("WARNING: loss did not decrease — something is off "
              "(encoder dtype? lr?)")
        sys.exit(1)

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
