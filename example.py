# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kernels",
#     "numpy",
#     "torch",
# ]
# ///

"""End-to-end example for the testing MaxSim kernel.

Builds two tiny query/document segments, scores one pair through the
kernel, and compares against the pure PyTorch reference.
"""

import platform
from pathlib import Path

import kernels
import torch


def pick_device() -> torch.device:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = pick_device()
    print(f"Using device: {device}")

    kernel = kernels.get_kernel("erikkaum/maxsim", version=1)

    torch.manual_seed(0)
    dim = 64

    # Two queries (lengths 3 and 5) and three documents (lengths 6, 10, 8).
    queries = torch.randn(3 + 5, dim, device=device, dtype=torch.float32)
    query_offsets = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
    documents = torch.randn(6 + 10 + 8, dim, device=device, dtype=torch.float32)
    document_offsets = torch.tensor([0, 6, 16, 24], dtype=torch.int32, device=device)

    # Score four pairs: every query against documents 0 and 2.
    pair_query_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    pair_document_ids = torch.tensor([0, 2, 0, 2], dtype=torch.int32, device=device)

    scores = kernel.score_pairs_packed(
        queries,
        query_offsets,
        documents,
        document_offsets,
        pair_query_ids,
        pair_document_ids,
    )
    print(f"Kernel scores: {scores.tolist()}")

    expected = kernel.score_pairs_packed_reference(
        queries,
        query_offsets,
        documents,
        document_offsets,
        pair_query_ids,
        pair_document_ids,
    )
    print(f"Reference   : {expected.tolist()}")

    torch.testing.assert_close(scores, expected, rtol=1e-5, atol=1e-4)
    print("Success!")


if __name__ == "__main__":
    main()
