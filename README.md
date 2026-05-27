<div align="center">

# MaxSim

<img src="assets/maxsim.png" alt="maxsim banner" />

---

[[Install]](#install) [[Common workflows]](#common-workflows)
[[Reference]](#reference) [[Example scripts]](#example-scripts)
[[Benchmarks]](#benchmarks)

</div>

## Introduction

A fast, memory-efficient exact MaxSim kernel for ColBERT-style late-interaction
retrieval and training. It's a drop-in replacement for PyTorch's MaxSim that's
**14× faster** in training and saves **98% of the backward memory**.
[See benchmarks.](#benchmarks)

The kernel is packaged as a
[Hugging Face `kernels`](https://github.com/huggingface/kernels) repo so you can
`kernels.get_kernel(...)` it in your own code with minimal dependencies.

The kernel computes

```
score(q, d) = sum_i max_j  <q_i, d_j>
```

over batches of `(query, document)` pairs without materialising the full
`[Lq, Ld]` similarity matrix. Most of those values get discarded by max anyway
so the kernel never computes them in forward, and saves only the surviving
argmax positions (`[Lq]` int32 per pair) for backward. This is where the speed
and memory savings come from.

## Install

The kernel is ahead-of-time compiled and distributed as a Hugging Face kernel
repo. This means that you can use the kernel without any compile step and just
with a minimal dependency. Using the kernel is as easy as:

```bash
uv add kernels        # or: pip install kernels
```

```python
from kernels import get_kernel
maxsim = get_kernel("erikkaum/maxsim", version=2, trust_remote_code=True)
```

## Common workflows

The MaxSim kernel has three entrypoints with an inference and training variant
of each:

- **Contrastive**: every query scored against every document. This is the
  classic ColBERT-style paradigm that's mostly used in training and fine-tuning.
- **Padded**: each query has a fixed number `C` of candidates, queries and
  documents are both padded to max-`Lq` / max-`Ld`. This is the most common
  inference workflow as a second stage exact rerank.
- **Packed**: arbitrary `(query, document)` pair grids over ragged queries and
  documents. The lowest-level surface; the most flexible, but also the most
  boilerplate to set up. You build the offset arrays and pair-id arrays
  yourself.

The two most common ways you will use the kernel are contrastive training and
padded reranking. The packed API is lower-level and is covered in the reference
section below.

### Contrastive training

Use this when training or fine-tuning a ColBERT-style model with in-batch
negatives. `score_contrastive_train` scores every query in the batch against
every document in the batch, producing an `[Nq, Nb]` score matrix.

These are the abbreviations used in this section:

```text
Nq  = number of queries in the batch
Nb  = number of documents in the batch
q   = query index
d   = document index
Lq  = query length after padding/truncation
dim = embedding dimension from the encoder
```

For example, in the common in-batch setup:

```text
scores[q, d] = MaxSim(queries[q], documents[d])
```

This scores query `q` against document `d`. The batch is arranged so that
`queries[i]` and `documents[i]` are the positive pair for training example `i`,
so the diagonal entries are positives and the off-diagonal entries are in-batch
negatives:

```text
    d=0    d=1    d=2    d=3
q=0  +      -      -      -
q=1  -      +      -      -
q=2  -      -      +      -
q=3  -      -      -      +
```

That is why the result can be passed directly to cross-entropy with diagonal
targets, like:

```python
targets = torch.arange(Nq, device=scores.device)
loss = torch.nn.functional.cross_entropy(scores, targets)
```

The contrastive API uses a mixed layout: queries are rectangular, while
documents are packed.

Queries are usually short and padded/truncated to one fixed length for the
batch. Each filled square is one token embedding:

```text
queries: [Nq, Lq, dim]

     t0 t1 t2 t3 ... t31
q0   ■  ■  ■  ■       ■
q1   ■  ■  ■  ■       ■
q2   ■  ■  ■  ■       ■
```

In contrastive mode, all `Lq` query positions participate in MaxSim; there is no
separate `query_lengths` tensor. This matches the usual ColBERT training setup
where queries are encoded to a fixed query length.

Documents are often longer and have more variable lengths. If we padded every
document to the longest document in the batch, some of the work would go to
padding:

```text
padded documents: [Nb, max_Ld, dim]

     t0 t1 t2 ... t79 t80 ... t255
d0   ■  ■  ■      ■   ■       ■     256 tokens
d1   ■  ■  ■      ■   □       □      80 tokens
d2   ■  ■  ■  ... □   □       □      64 tokens

■ = real token embedding
□ = padding
```

Instead, contrastive mode packs only the real document token embeddings into one
flat tensor:

```text
documents: [total_d_tokens, dim]

idx     0 1 2 ... 255 | 256 257 ... 335 | 336 337 ... 399
token   ■ ■ ■      ■  |  ■   ■       ■  |  ■   ■       ■
doc          doc 0    |      doc 1      |      doc 2
```

The companion `document_offsets` tensor marks the boundaries between documents:

```text
document_offsets = [0, 256, 336, 400]

document 0 = documents[0:256]     # 256 tokens
document 1 = documents[256:336]   # 80 tokens
document 2 = documents[336:400]   # 64 tokens
```

A typical training step looks like this:

```python
import torch
import torch.nn.functional as F
from kernels import get_kernel

maxsim = get_kernel("erikkaum/maxsim", version=2, trust_remote_code=True)

# Encoder outputs.
queries = encoder_q(query_ids)      # [Nq, Lq, dim]
doc_emb = encoder_d(document_ids)   # [Nb, Ld, dim] before packing

# Naive/illustrative packing according to real token lengths.
documents = torch.cat(
    [doc_emb[i, :doc_lengths[i]] for i in range(Nb)],
    dim=0,
)  # [total_d_tokens, dim]

document_offsets = torch.zeros(Nb + 1, dtype=torch.int32, device=doc_emb.device)
document_offsets[1:] = doc_lengths.cumsum(0)

scores = maxsim.score_contrastive_train(
    queries,
    documents,
    document_offsets,
)  # [Nq, Nb] fp32

# Diagonal positives: query i should match document i.
targets = torch.arange(Nq, device=scores.device)

# MaxSim scores grow with Lq, so training loops often use a temperature/scale.
temperature = 1.0
loss = F.cross_entropy(scores / temperature, targets)
loss.backward()
```

Typical ColBERT-style values are:

```text
Nq, Nb = 16 to 64
Lq     = 32
Ld     = 80 to 256 document tokens before packing
dim    = encoder output width, often 64 / 96 / 128
```

> [!NOTE]
> On CUDA, the contrastive training kernel currently uses the WMMA path and
> expects `Lq % 16 == 0` and `dim % 16 == 0`. Typical ColBERT shapes such as
> `Lq=32`, `dim=128` satisfy this.

The kernel saves only the winning document-token index per query token for
backward, rather than retaining the full `[Nq, Nb, Lq, Ld]` similarity tensor.
This is where most of the training-time memory saving comes from.

### Padded inference

A typical ColBERT/late-interaction retrieval pipeline usually has two stages.

1. A retriever such as PLAID first gives you `K` candidate documents for each
   query.
2. Then rerank those `K` with exact MaxSim, through
   `score_candidates_padded(...)`. This computes the exact MaxSim for every
   query-candidate pair and returns the scores as a `[B, K]` matrix.

The inputs come from your retriever, e.g. PLAID:

```
queries        [B, Lq, dim]       fp16/bf16
candidates     [B, K, Ld, dim]    same dtype as queries
query_lengths  [B]                int32, real per-query token count (≤ Lq)
doc_lengths    [B, K]             int32, real per-doc token count   (≤ Ld)
```

`Lq` and `Ld` are the padded lengths of the tensors, while `query_lengths` and
`doc_lengths` are the actual per-query and per-document token counts. Padding
positions are ignored when computing MaxSim. In practice, `Lq` is usually the
longest query length in the batch, and `Ld` is the longest document length in
the batch.

These are the abbreviations used in this section:

```text
B   = number of queries in the batch
K   = number of candidate documents per query
Lq  = padded query length
Ld  = padded document length
dim = embedding dimension from your encoder, e.g. 64 / 96 / 128
```

We feed these into the `score_candidates_padded`, which returns a `[B,K]` matrix
with dtype fp32, regardless of the input dtype.

```python
# Exact MaxSim score per (query, candidate)
scores = maxsim.score_candidates_padded(queries, candidates, query_lengths, doc_lengths)
```

And sort them to get the reranked order

```python
reranked_ids = scores.argsort(dim=1, descending=True)
```

For typical shapes (B=32, K=50), this is ~3–7× faster than PyTorch einsum+max.
[See Benchmarks.](#benchmarks)

## Reference

### Inference

```python
# Padded: per-query reranking with a fixed K candidates/documents each. Inputs are
# padded to max-Lq / max-Ld; *_lengths give the real per-row length.
scores = maxsim.score_candidates_padded(
    queries,        # [B, Lq, dim]      fp32 / fp16 / bf16
    documents,      # [B, K, Ld, dim]   same dtype as queries
    query_lengths,  # [B]               int32 — real query length (≤ Lq)
    doc_lengths,    # [B, K]            int32 — real doc length   (≤ Ld)
)  # -> [B, K] fp32

# Contrastive: every query × every document. Returns the full score
# matrix [Nq, Nb] — what most contrastive losses (InfoNCE, etc.) want.
scores = maxsim.score_contrastive(
    queries,           # [Nq, Lq, dim]          fp16 / bf16 on CUDA; fp32 also on Metal
    documents,         # [total_d_tokens, dim]  flat (all docs concatenated)
    document_offsets,  # [Nb + 1]   int32       documents[offsets[i]:offsets[i+1]] = doc i
)  # -> [Nq, Nb] fp32

# Packed: arbitrary (q, d) pair grids. You pick which pairs to score via
# pair_query_ids / pair_document_ids; the kernel computes only those.
scores = maxsim.score_pairs_packed(
    queries,            # [total_q_tokens, dim]   fp32 / fp16 / bf16
    query_offsets,      # [num_queries + 1]       int32 — CSR offsets into queries
    documents,          # [total_d_tokens, dim]   same dtype as queries
    document_offsets,   # [num_documents + 1]     int32 — CSR offsets into documents
    pair_query_ids,     # [num_pairs]             int32 — which query to score
    pair_document_ids,  # [num_pairs]             int32 — which document to score
)  # -> [num_pairs] fp32
```

### Argmax helpers

Each inference API also has a `_with_argmax` variant. It returns the same fp32
scores plus the document-token index that won the max for each query token.
These helpers are useful for debugging alignments, visualizing which document
tokens matched, and inspecting the compact state used by the training kernels.

```python
scores, argmax = maxsim.score_candidates_padded_with_argmax(
    queries,        # [B, Lq, dim]
    documents,      # [B, K, Ld, dim]
    query_lengths,  # [B]
    doc_lengths,    # [B, K]
)  # scores: [B, K] fp32; argmax: [B, K, Lq] int32

scores, argmax = maxsim.score_contrastive_with_argmax(
    queries,           # [Nq, Lq, dim]
    documents,         # [total_d_tokens, dim]
    document_offsets,  # [Nb + 1]
)  # scores: [Nq, Nb] fp32; argmax: [Nq, Nb, Lq] int32

scores, argmax = maxsim.score_pairs_packed_with_argmax(
    queries,
    query_offsets,
    documents,
    document_offsets,
    pair_query_ids,
    pair_document_ids,
)  # scores: [num_pairs] fp32; argmax: [num_pairs, max_q_len] int32
```

### Training

Each inference API has a `_train` variant with the same input layout and output
shape, but connected to PyTorch autograd. The `_train` variants return fp32
scores and propagate gradients to `queries` and `documents`; length, offset, and
pair-id tensors are metadata and do not receive gradients. Backward uses fp32
accumulation.

```python
# Padded training.
scores = maxsim.score_candidates_padded_train(
    queries,        # [B, Lq, dim]      fp32 / fp16 / bf16
    documents,      # [B, K, Ld, dim]   same dtype as queries
    query_lengths,  # [B]               int32 / int64
    doc_lengths,    # [B, K]            int32 / int64
)  # -> [B, K] fp32

# Contrastive training.
scores = maxsim.score_contrastive_train(
    queries,           # [Nq, Lq, dim]          fp16 / bf16 on CUDA; fp32 also on Metal
    documents,         # [total_d_tokens, dim]  same dtype as queries
    document_offsets,  # [Nb + 1]               int32 / int64
)  # -> [Nq, Nb] fp32

# Packed training.
scores = maxsim.score_pairs_packed_train(
    queries,            # [total_q_tokens, dim]   fp32 / fp16 / bf16
    query_offsets,      # [num_queries + 1]       int32 / int64
    documents,          # [total_d_tokens, dim]   same dtype as queries
    document_offsets,   # [num_documents + 1]     int32 / int64
    pair_query_ids,     # [num_pairs]             int32 / int64
    pair_document_ids,  # [num_pairs]             int32 / int64
)  # -> [num_pairs] fp32
```

## Example scripts

The `examples/` directory is the quickest way to run some code locally:

```bash
just example 01   # quickstart: padded inference + padded training
just example 02   # argmax / alignment visualization
just example 03   # tiny contrastive InfoNCE training loop
just example 04   # retained-memory showcase vs naive PyTorch
```

## Benchmarks

All numbers below are medians over 30 iterations × 3 independent runs
(`MAXSIM_BENCH_REPEATS=3`), generated by `just cuda-bench-matrix`. The baseline
is a straightforward PyTorch implementation that uses `torch.einsum` to
materialise the full similarity tensor before `max`. Raw outputs are committed
under [`bench_results/v2/`](./bench_results/v2/).

### ColBERT training on A100

The headline workload uses a common ColBERT-style in-batch contrastive shape:
`Nq=Nb=32, Lq=32, Ld=80, dim=128`. **10.3× faster training step, with 1/80th the
retained backward state.**

| dtype | maxsim step | naive step | full step× | bwd alone× | peak mem× | retained state |
| ----- | ----------: | ---------: | ---------: | ---------: | --------: | -------------: |
| fp16  |    0.337 ms |   3.477 ms | **10.31×** |     16.62× |     1.78× |           1/80 |
| bf16  |    0.334 ms |   3.464 ms | **10.37×** |     16.84× |     1.78× |           1/80 |

`NVIDIA A100-SXM4-80GB` (`sm_80`, Torch 2.6.0 + CUDA 12.4). Run-to-run spread
over 3 repeats: ≤0.31× on the contrastive workloads. Raw JSON artifact:
[`bench_results/v2/a100-sxm4-80gb.json`](./bench_results/v2/a100-sxm4-80gb.json).

The backward speedup is the load-bearing one — training is backward-dominated,
and that's where the kernel's argmax-only save pays off:

```text
naive contrastive backward:  [Nq, Nb, Lq, Ld] fp32 similarities
maxsim contrastive backward: [Nq, Nb, Lq]     int32 argmax
maxsim retained state = 1/Ld of naive
```

|   Ld | naive retained | maxsim retained | maxsim / naive |
| ---: | -------------: | --------------: | -------------: |
|   75 |         9.8 MB |         0.13 MB |           1/75 |
|  128 |        16.8 MB |         0.13 MB |          1/128 |
|  256 |        33.6 MB |         0.13 MB |          1/256 |
|  512 |        67.1 MB |         0.13 MB |          1/512 |
| 1024 |       134.2 MB |         0.13 MB |         1/1024 |

### Cross-Device Contrastive fp16

This table is rendered from the GPU JSON artifacts available under
[`bench_results/v2/`](./bench_results/v2/).

<!-- BENCH:cross-gpu-contrastive -->

| GPU            | sm    | maxsim step | naive step | speedup |
| -------------- | ----- | ----------: | ---------: | ------: |
| A100 SXM4 80GB | sm_80 |    0.338 ms |   3.523 ms |  10.41× |

<!-- /BENCH -->

### Full A100 benchmark matrix

The full matrix includes every benchmark workload present in the rendered JSON
artifact.

<!-- BENCH:full-matrix-a100 -->

| Surface           | Preset      | Shape                                | dtype |   maxsim |  PyTorch | speedup | padded |   bwd× | peak× | retained state |
| ----------------- | ----------- | ------------------------------------ | ----- | -------: | -------: | ------: | -----: | -----: | ----: | -------------: |
| contrastive_train | Contrastive | `Nq=32, Nb=32, Lq=32, Ld=80, D=128`  | fp16  | 0.338 ms | 3.523 ms |  10.41× |      — | 16.83× | 1.78× |           1/80 |
| contrastive_train | Contrastive | `Nq=32, Nb=32, Lq=32, Ld=80, D=128`  | bf16  | 0.340 ms | 3.530 ms |  10.40× |      — | 16.77× | 1.78× |           1/80 |
| contrastive_train | LongDocs    | `Nq=32, Nb=32, Lq=32, Ld=512, D=128` | fp16  | 0.421 ms | 3.499 ms |   8.30× |      — | 33.15× | 3.24× |          1/512 |
| contrastive_train | LongDocs    | `Nq=32, Nb=32, Lq=32, Ld=512, D=128` | bf16  | 0.425 ms | 3.502 ms |   8.24× |      — | 33.12× | 3.24× |          1/512 |
| contrastive_train | BigBatch    | `Nq=64, Nb=64, Lq=32, Ld=128, D=128` | fp16  | 0.500 ms | 6.317 ms |  12.65× |      — | 33.36× | 4.15× |          1/128 |
| contrastive_train | BigBatch    | `Nq=64, Nb=64, Lq=32, Ld=128, D=128` | bf16  | 0.511 ms | 6.302 ms |  12.34× |      — | 33.23× | 4.15× |          1/128 |

<!-- /BENCH -->

Packed rows compare `score_pairs_packed` against both the equivalent padded
kernel layout and the PyTorch einsum baseline.

### Reproducing

```bash
just cuda-bench-matrix a100-large 3 30   # flavor, repeats, iters
just cuda-bench-matrix l40sx1     3 30   # optional additional GPU
just cuda-bench-matrix a10g-small 3 30   # optional additional GPU
just bench-matrix-metal           3 30   # optional Apple Silicon / MPS run
just bench-render                        # regenerate README tables from JSON
```

The CUDA bench recipe submits an HF Jobs run; `bench-matrix-metal` runs the same
matrix against the Metal/MPS build. Both write one JSON artifact per device
class to [`bench_results/v2/`](./bench_results/v2/). The artifacts capture the
kernel commit, device/torch/CUDA versions, timing + variance per workload, and a
structured shape per row. `just bench-render` rewrites the tables above from
those JSONs — same data, no copy-paste drift.

## Supported

| Backend | Devices                              | Input dtypes       | Backward accumulation |
| ------- | ------------------------------------ | ------------------ | --------------------- |
| Metal   | Apple Silicon (MPS, MSL 3.1+)        | fp32 / fp16 / bf16 | fp32                  |
| CUDA    | sm_80, sm_86, sm_89, sm_90 (PTX JIT) | fp32 / fp16 / bf16 | fp32                  |

The table is package-level support. A few API-specific constraints are listed
below.

## Limitations and edge cases

- CUDA contrastive APIs (`score_contrastive`, `score_contrastive_train`, and
  `score_contrastive_with_argmax`) currently use the WMMA path directly. They
  require fp16/bf16 inputs, `Lq % 16 == 0`, and `dim % 16 == 0`.
- Padded inference (`score_candidates_padded`) accepts fp32/fp16/bf16 and
  arbitrary `Lq`, `Ld`, and `dim` on both backends. CUDA uses the WMMA fast path
  for fp16/bf16 when `Lq % 16 == 0` and `dim % 16 == 0`; otherwise it uses the
  scalar path.
- Padded training on CUDA uses the argmax WMMA path and currently requires
  fp16/bf16 inputs, `Lq >= 16`, `Lq % 16 == 0`, and `dim % 16 == 0`.
- Metal handles arbitrary `Lq` for padded and contrastive paths. Its
  `simdgroup_matrix` path fires when `dim % 8 == 0`; otherwise it uses scalar
  work. Very large `Lq * dim` shapes can hit Metal threadgroup-memory limits.
- Packed CUDA paths are scalar today: `score_pairs_packed` uses a scalar
  score-only kernel, and packed training uses scalar argmax/backward kernels
  rather than WMMA. Which means that there isn't really a performance win over
  native PyTorch. Use packed for arbitrary or sparse pair grids; for fixed-`K`
  reranking, prefer the padded API.
- Backward kernels accumulate gradients with fp32 atomics. The results are
  numerically stable for training, but not bit-exact deterministic across
  repeated backward passes because floating-point additions can arrive in a
  different order.
- NaN inputs do not follow PyTorch's exact `torch.max` NaN propagation
  semantics. Kernel comparisons use `v > max`, so NaN candidates are skipped. A
  NaN in a query token usually drives the whole score for that pair to `-inf`; a
  NaN in one document token may simply be ignored if another document token wins
  the max. Treat non-finite encoder outputs as upstream bugs and pre-screen with
  `torch.isfinite(...)` if you need strict behavior.
- Hopper (`sm_90`) currently runs via PTX forward compatibility. Native WGMMA
  tuning is future work.

## Related projects and acknowledgements

- [Flash-MaxSim](https://github.com/roipony/flash-maxsim) is a fused Triton
  MaxSim kernel for ColBERT/ColPali scoring.

## Source

Source: <https://github.com/erikkaum/maxsim>.

Kernel repo: <https://huggingface.co/kernels/erikkaum/maxsim>.

License: Apache-2.0.
