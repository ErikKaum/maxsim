// CUDA backend — Stage 1: naive scalar kernel mirroring the original Metal
// "step 1" structure (one block per (pair), warps split q_toks, lanes split
// d_toks, warp-shuffle reduce gives per-q-tok max, single warp sums maxes
// into the per-pair score). Lays the correctness baseline. Optimization
// (WMMA tiles, multi-tile batching, etc.) comes in Stages 2+.
//
// The host functions match `torch-ext/torch_binding.h`. Same signature is
// also exposed by `maxsim_cuda/dev_binding.cpp` for the cpp_extension-based
// dev path used on HF Jobs.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <mma.h>
#include <torch/all.h>

namespace {

constexpr int kWarpSize     = 32;
constexpr int kBlockThreads = 128;
constexpr int kNumWarps     = kBlockThreads / kWarpSize;

// WMMA tile (Ampere mma.sync 16x16x16 fp16/bf16 -> fp32 accumulator)
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// ---- Element load helpers (fp16/bf16/fp32 -> float promotion) ------------

template <typename T>
__device__ inline float load_f(const T* p) { return static_cast<float>(*p); }

template <>
__device__ inline float load_f<__half>(const __half* p) {
  return __half2float(*p);
}

template <>
__device__ inline float load_f<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}

// Typed zero (bf16 has no constructor from float; need explicit conversion).
template <typename T>
__device__ inline T zero_v() { return T(0); }

template <>
__device__ inline __half zero_v<__half>() { return __float2half(0.0f); }

template <>
__device__ inline __nv_bfloat16 zero_v<__nv_bfloat16>() {
  return __float2bfloat16(0.0f);
}

// ---- Core inner loop (per simdgroup-equivalent: per-warp slice of q_toks) -

template <typename T>
__device__ inline float dot_row(const T* qrow, const T* drow, int dim) {
  float acc = 0.0f;
  for (int k = 0; k < dim; ++k) {
    acc += load_f(qrow + k) * load_f(drow + k);
  }
  return acc;
}

// ---- Kernels --------------------------------------------------------------

template <typename T>
__global__ void maxsim_packed_kernel(
    const T* __restrict__       queries,
    const int* __restrict__     query_offsets,
    const T* __restrict__       documents,
    const int* __restrict__     document_offsets,
    const int* __restrict__     pair_query_ids,
    const int* __restrict__     pair_document_ids,
    float* __restrict__         scores,
    int                         dim) {
  const int pair_idx = blockIdx.x;
  const int tid      = threadIdx.x;
  const int lane     = tid & (kWarpSize - 1);
  const int warp_id  = tid >> 5;

  const int q_id    = pair_query_ids[pair_idx];
  const int d_id    = pair_document_ids[pair_idx];
  const int q_start = query_offsets[q_id];
  const int q_end   = query_offsets[q_id + 1];
  const int d_start = document_offsets[d_id];
  const int d_end   = document_offsets[d_id + 1];
  const int Lq      = q_end - q_start;
  const int Ld      = d_end - d_start;

  // shared: per-q-tok running max
  extern __shared__ float maxes[];
  for (int i = tid; i < Lq; i += kBlockThreads) {
    maxes[i] = -CUDART_INF_F;
  }
  __syncthreads();

  // each warp owns a stride-`kNumWarps` slice of q_toks
  for (int q_tok = warp_id; q_tok < Lq; q_tok += kNumWarps) {
    const T* qrow = queries + static_cast<size_t>(q_start + q_tok) * dim;
    float my_max  = -CUDART_INF_F;
    for (int d_tok = lane; d_tok < Ld; d_tok += kWarpSize) {
      const T* drow = documents + static_cast<size_t>(d_start + d_tok) * dim;
      my_max = fmaxf(my_max, dot_row(qrow, drow, dim));
    }
    // butterfly reduce within the warp
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, off));
    }
    if (lane == 0) maxes[q_tok] = my_max;
  }
  __syncthreads();

  // warp 0 sums maxes -> scores[pair_idx]
  if (warp_id == 0) {
    float s = 0.0f;
    for (int i = lane; i < Lq; i += kWarpSize) s += maxes[i];
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
    }
    if (lane == 0) scores[pair_idx] = s;
  }
}

// ---- Stage 2: WMMA-based padded kernel ----------------------------------
//
// One block per pair. Each warp owns a 16-row q-block; iterates db over the
// document in 16-wide tiles, doing a `dim/16`-deep K-loop of mma.sync 16x16x16
// fp16 accumulating into an fp32 fragment. After each tile, the 16x16
// fragment is stored to per-warp shared scratch and row-max'd into `maxes`.
// Final reduction: warp 0 sums `maxes[0..Lq]` -> `scores[pair_idx]`.
//
// Constraints (host gates these): dim % 16 == 0, Lq_max >= 16, fp16/bf16.

template <typename T_mma>
__global__ void maxsim_padded_wmma_kernel(
    const T_mma* __restrict__   queries,
    const int* __restrict__     query_lengths,
    const T_mma* __restrict__   documents,
    const int* __restrict__     doc_lengths,
    float* __restrict__         scores,
    int                         dim,
    int                         Lq_max,
    int                         Ld_max,
    int                         num_candidates) {
  using namespace nvcuda::wmma;

  const int pair_idx = blockIdx.x;
  const int tid      = threadIdx.x;
  const int lane     = tid & (kWarpSize - 1);
  const int warp_id  = tid >> 5;
  const int num_warps = blockDim.x >> 5;

  const int q_id    = pair_idx / num_candidates;
  const int d_id    = pair_idx;
  const int q_start = q_id * Lq_max;
  const int d_start = d_id * Ld_max;
  const int Lq      = query_lengths[q_id];
  const int Ld      = doc_lengths[d_id];

  // Pad Lq up to multiple of kWmmaM; rows beyond Lq are zero so they
  // contribute nothing to the per-row max (and we don't sum them anyway).
  const int Lq_aligned = ((Lq + kWmmaM - 1) / kWmmaM) * kWmmaM;
  const int Ld_aligned = (Ld / kWmmaN) * kWmmaN;

  extern __shared__ unsigned char smem[];
  float* maxes      = reinterpret_cast<float*>(smem);
  T_mma* Q_shared   = reinterpret_cast<T_mma*>(maxes + Lq_max);
  float* frag_store = reinterpret_cast<float*>(
      Q_shared + static_cast<size_t>(Lq_max) * dim);
  // frag_store layout: [num_warps][kWmmaM * kWmmaN] floats.

  // Init maxes for the valid q range.
  for (int i = tid; i < Lq; i += blockDim.x) {
    maxes[i] = -CUDART_INF_F;
  }
  // Cooperatively stage Q[q_start, 0..Lq) and zero-pad rows [Lq, Lq_aligned).
  for (int i = tid; i < Lq * dim; i += blockDim.x) {
    Q_shared[i] = queries[static_cast<size_t>(q_start) * dim + i];
  }
  for (int i = Lq * dim + tid; i < Lq_aligned * dim; i += blockDim.x) {
    Q_shared[i] = zero_v<T_mma>();
  }
  __syncthreads();

  const int qb = warp_id * kWmmaM;
  if (qb < Lq_aligned) {
    const T_mma* Q_qb   = Q_shared + static_cast<size_t>(qb) * dim;
    const T_mma* D_base = documents + static_cast<size_t>(d_start) * dim;
    // 4 slot region per warp (cascade up to 4-tile batching).
    constexpr int kSlot = kWmmaM * kWmmaN;
    float* slot = frag_store + warp_id * (4 * kSlot);

    // Cascade: 4-tile -> 2-tile -> 1-tile -> scalar.
    const int db_quad_end =
        (Ld_aligned / (4 * kWmmaN)) * (4 * kWmmaN);
    const int db_pair_end =
        db_quad_end +
        ((Ld_aligned - db_quad_end) / (2 * kWmmaN)) * (2 * kWmmaN);

    // Helper macros that fan out fragments without resorting to arrays
    // (wmma::fragment types aren't default-constructible into stack arrays
    // in a clean way across CUDA versions).
    #define LOAD_A(K) \
      fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, T_mma, row_major> a_frag; \
      load_matrix_sync(a_frag, Q_qb + (K), dim);
    #define MMA_TILE(C, DB, K) \
      { \
        fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, T_mma, col_major> b_frag; \
        load_matrix_sync(b_frag, \
            D_base + static_cast<size_t>(DB) * dim + (K), dim); \
        mma_sync(C, a_frag, b_frag, C); \
      }
    #define ROW_MAX_OVER_N(N_TILES) \
      { \
        const int row    = lane >> 1; \
        const int col_lo = (lane & 1) << 3; \
        float lane_max = -CUDART_INF_F; \
        _Pragma("unroll") \
        for (int t = 0; t < (N_TILES); ++t) { \
          const float* row_ptr = slot + t * kSlot + row * kWmmaN; \
          _Pragma("unroll") \
          for (int c = 0; c < 8; ++c) { \
            lane_max = fmaxf(lane_max, row_ptr[col_lo + c]); \
          } \
        } \
        lane_max = fmaxf(lane_max, __shfl_xor_sync(0xffffffff, lane_max, 1)); \
        if ((lane & 1) == 0) { \
          const int q_tok = qb + row; \
          if (q_tok < Lq) maxes[q_tok] = fmaxf(maxes[q_tok], lane_max); \
        } \
      }

    // ---- 4-tile path: one A-load shared across 4 MMAs ----
    for (int db = 0; db < db_quad_end; db += 4 * kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0, c1, c2, c3;
      fill_fragment(c0, 0.0f);
      fill_fragment(c1, 0.0f);
      fill_fragment(c2, 0.0f);
      fill_fragment(c3, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db + 0 * kWmmaN, k);
        MMA_TILE(c1, db + 1 * kWmmaN, k);
        MMA_TILE(c2, db + 2 * kWmmaN, k);
        MMA_TILE(c3, db + 3 * kWmmaN, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 1 * kSlot, c1, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 2 * kSlot, c2, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 3 * kSlot, c3, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_OVER_N(4);
      __syncwarp();
    }

    // ---- 2-tile path for any remaining (0 or 1) pair of tiles ----
    for (int db = db_quad_end; db < db_pair_end; db += 2 * kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0, c1;
      fill_fragment(c0, 0.0f);
      fill_fragment(c1, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db + 0 * kWmmaN, k);
        MMA_TILE(c1, db + 1 * kWmmaN, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 1 * kSlot, c1, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_OVER_N(2);
      __syncwarp();
    }

    // ---- 1-tile path for any single leftover 16-wide tile ----
    for (int db = db_pair_end; db < Ld_aligned; db += kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0;
      fill_fragment(c0, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_OVER_N(1);
      __syncwarp();
    }

    #undef LOAD_A
    #undef MMA_TILE
    #undef ROW_MAX_OVER_N

    // ---- Scalar tail for Ld % 16 leftover d_toks ----
    for (int q_off = 0; q_off < kWmmaM; ++q_off) {
      const int q_tok = qb + q_off;
      if (q_tok >= Lq) break;
      const T_mma* qrow = Q_qb + static_cast<size_t>(q_off) * dim;
      float my_max = -CUDART_INF_F;
      for (int d_tok = Ld_aligned + lane; d_tok < Ld; d_tok += kWarpSize) {
        const T_mma* drow = D_base + static_cast<size_t>(d_tok) * dim;
        my_max = fmaxf(my_max, dot_row(qrow, drow, dim));
      }
      for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, off));
      }
      if (lane == 0) maxes[q_tok] = fmaxf(maxes[q_tok], my_max);
    }
  }

  __syncthreads();

  if (warp_id == 0) {
    float s = 0.0f;
    for (int i = lane; i < Lq; i += kWarpSize) s += maxes[i];
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
    }
    if (lane == 0) scores[pair_idx] = s;
  }
}

// =============================================================================
// Padded forward + argmax (training mode). Same 4-tile WMMA cascade as
// `maxsim_padded_wmma_kernel` above, but additionally tracks per-q-tok
// argmax positions with PyTorch's first-index-wins tiebreak (strict `>` on
// every cross-tile update, lower-arg-wins on butterfly ties). Output
// argmax_out has shape [num_pairs, Lq_max] int32; slots beyond Lq are 0.
// =============================================================================

template <typename T_mma>
__global__ void maxsim_padded_wmma_argmax_kernel(
    const T_mma* __restrict__   queries,
    const int* __restrict__     query_lengths,
    const T_mma* __restrict__   documents,
    const int* __restrict__     doc_lengths,
    float* __restrict__         scores,
    int* __restrict__           argmax_out,
    int                         dim,
    int                         Lq_max,
    int                         Ld_max,
    int                         num_candidates) {
  using namespace nvcuda::wmma;

  const int pair_idx  = blockIdx.x;
  const int tid       = threadIdx.x;
  const int lane      = tid & (kWarpSize - 1);
  const int warp_id   = tid >> 5;
  const int num_warps = blockDim.x >> 5;
  (void)num_warps;

  const int q_id    = pair_idx / num_candidates;
  const int d_id    = pair_idx;
  const int q_start = q_id * Lq_max;
  const int d_start = d_id * Ld_max;
  const int Lq      = query_lengths[q_id];
  const int Ld      = doc_lengths[d_id];

  const int Lq_aligned = ((Lq + kWmmaM - 1) / kWmmaM) * kWmmaM;
  const int Ld_aligned = (Ld / kWmmaN) * kWmmaN;

  // Shared mem layout (all 4-byte aligned slabs; Q_shared falls on a
  // 16-byte boundary because maxes + argmax_buf together are a multiple
  // of 8 bytes per Lq_max entry):
  //   maxes      : Lq_max floats
  //   argmax_buf : Lq_max int32
  //   Q_shared   : Lq_max * dim T_mma
  //   frag_store : 4 * num_warps * kWmmaM * kWmmaN floats
  extern __shared__ unsigned char smem[];
  float* maxes      = reinterpret_cast<float*>(smem);
  int*   argmax_buf = reinterpret_cast<int*>(maxes + Lq_max);
  T_mma* Q_shared   = reinterpret_cast<T_mma*>(argmax_buf + Lq_max);
  float* frag_store = reinterpret_cast<float*>(
      Q_shared + static_cast<size_t>(Lq_max) * dim);

  for (int i = tid; i < Lq; i += blockDim.x) {
    maxes[i]      = -CUDART_INF_F;
    argmax_buf[i] = 0;
  }
  for (int i = tid; i < Lq * dim; i += blockDim.x) {
    Q_shared[i] = queries[static_cast<size_t>(q_start) * dim + i];
  }
  for (int i = Lq * dim + tid; i < Lq_aligned * dim; i += blockDim.x) {
    Q_shared[i] = zero_v<T_mma>();
  }
  __syncthreads();

  const int qb = warp_id * kWmmaM;
  if (qb < Lq_aligned) {
    const T_mma* Q_qb   = Q_shared + static_cast<size_t>(qb) * dim;
    const T_mma* D_base = documents + static_cast<size_t>(d_start) * dim;
    constexpr int kSlot = kWmmaM * kWmmaN;
    float* slot = frag_store + warp_id * (4 * kSlot);

    const int db_quad_end =
        (Ld_aligned / (4 * kWmmaN)) * (4 * kWmmaN);
    const int db_pair_end =
        db_quad_end +
        ((Ld_aligned - db_quad_end) / (2 * kWmmaN)) * (2 * kWmmaN);

    #define LOAD_A(K) \
      fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, T_mma, row_major> a_frag; \
      load_matrix_sync(a_frag, Q_qb + (K), dim);
    #define MMA_TILE(C, DB, K) \
      { \
        fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, T_mma, col_major> b_frag; \
        load_matrix_sync(b_frag, \
            D_base + static_cast<size_t>(DB) * dim + (K), dim); \
        mma_sync(C, a_frag, b_frag, C); \
      }
    // Row-reduce over N_TILES contiguous 16x16 fragments, carrying the
    // (max, argmax) tuple. Strict > everywhere (cross-tile, butterfly,
    // global update) so the lowest tied d-token wins. `db` is captured
    // from the enclosing loop and provides the global d-tok base.
    #define ROW_MAX_ARGMAX_OVER_N(N_TILES) \
      { \
        const int row    = lane >> 1; \
        const int col_lo = (lane & 1) << 3; \
        float lane_max = -CUDART_INF_F; \
        int   lane_arg = 0; \
        _Pragma("unroll") \
        for (int t = 0; t < (N_TILES); ++t) { \
          const float* row_ptr = slot + t * kSlot + row * kWmmaN; \
          _Pragma("unroll") \
          for (int c = 0; c < 8; ++c) { \
            float v = row_ptr[col_lo + c]; \
            int   d = db + t * kWmmaN + col_lo + c; \
            if (v > lane_max) { lane_max = v; lane_arg = d; } \
          } \
        } \
        float pm = __shfl_xor_sync(0xffffffff, lane_max, 1); \
        int   pa = __shfl_xor_sync(0xffffffff, lane_arg, 1); \
        bool take = pm > lane_max || (pm == lane_max && pa < lane_arg); \
        if (take) { lane_max = pm; lane_arg = pa; } \
        if ((lane & 1) == 0) { \
          const int q_tok = qb + row; \
          if (q_tok < Lq && lane_max > maxes[q_tok]) { \
            maxes[q_tok]      = lane_max; \
            argmax_buf[q_tok] = lane_arg; \
          } \
        } \
      }

    // ---- 4-tile path ----
    for (int db = 0; db < db_quad_end; db += 4 * kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0, c1, c2, c3;
      fill_fragment(c0, 0.0f);
      fill_fragment(c1, 0.0f);
      fill_fragment(c2, 0.0f);
      fill_fragment(c3, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db + 0 * kWmmaN, k);
        MMA_TILE(c1, db + 1 * kWmmaN, k);
        MMA_TILE(c2, db + 2 * kWmmaN, k);
        MMA_TILE(c3, db + 3 * kWmmaN, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 1 * kSlot, c1, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 2 * kSlot, c2, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 3 * kSlot, c3, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_ARGMAX_OVER_N(4);
      __syncwarp();
    }

    // ---- 2-tile path ----
    for (int db = db_quad_end; db < db_pair_end; db += 2 * kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0, c1;
      fill_fragment(c0, 0.0f);
      fill_fragment(c1, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db + 0 * kWmmaN, k);
        MMA_TILE(c1, db + 1 * kWmmaN, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 1 * kSlot, c1, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_ARGMAX_OVER_N(2);
      __syncwarp();
    }

    // ---- 1-tile path ----
    for (int db = db_pair_end; db < Ld_aligned; db += kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0;
      fill_fragment(c0, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_ARGMAX_OVER_N(1);
      __syncwarp();
    }

    #undef LOAD_A
    #undef MMA_TILE
    #undef ROW_MAX_ARGMAX_OVER_N

    // ---- Scalar tail for Ld % 16 leftover d_toks ----
    for (int q_off = 0; q_off < kWmmaM; ++q_off) {
      const int q_tok = qb + q_off;
      if (q_tok >= Lq) break;
      const T_mma* qrow = Q_qb + static_cast<size_t>(q_off) * dim;
      float my_max = -CUDART_INF_F;
      int   my_arg = 0;
      // Each lane visits a stride-kWarpSize slice of d_toks. Within a lane
      // the visit order is ascending so strict > picks the earliest tie.
      for (int d_tok = Ld_aligned + lane; d_tok < Ld; d_tok += kWarpSize) {
        const T_mma* drow = D_base + static_cast<size_t>(d_tok) * dim;
        float v = dot_row(qrow, drow, dim);
        if (v > my_max) { my_max = v; my_arg = d_tok; }
      }
      // 32-lane butterfly reduce with tiebreak by lower arg.
      for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        float pm = __shfl_xor_sync(0xffffffff, my_max, off);
        int   pa = __shfl_xor_sync(0xffffffff, my_arg, off);
        bool take = pm > my_max || (pm == my_max && pa < my_arg);
        if (take) { my_max = pm; my_arg = pa; }
      }
      if (lane == 0 && my_max > maxes[q_tok]) {
        maxes[q_tok]      = my_max;
        argmax_buf[q_tok] = my_arg;
      }
    }
  }

  __syncthreads();

  // Sum maxes -> scores.
  if (warp_id == 0) {
    float s = 0.0f;
    for (int i = lane; i < Lq; i += kWarpSize) s += maxes[i];
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
    }
    if (lane == 0) scores[pair_idx] = s;
  }

  // Write argmax_buf to global. Slots beyond Lq are 0 (matches reference).
  int* argmax_pair = argmax_out + static_cast<size_t>(pair_idx) * Lq_max;
  for (int i = tid; i < Lq_max; i += blockDim.x) {
    argmax_pair[i] = (i < Lq) ? argmax_buf[i] : 0;
  }
}

// =============================================================================
// Padded backward. Given the per-q-tok argmax positions saved by the forward
// kernel, routes the incoming `dscore` gradient back to dqueries and
// ddocuments. fp32 accumulation everywhere so the same kernel works for
// fp16 and bf16 inputs (and would for fp32 inputs too). Heavy use of
// atomicAdd is unavoidable: queries are shared across the C candidates
// (so dq accumulates across pairs of the same query), and within a single
// pair multiple q_toks may pick the same j_star (so dd accumulates).
//
// fp32 atomicAdd is supported on every arch we target (sm_80+); the
// fp16/bf16 atomicAdd story is messier (bf16 needs Hopper) which is why
// we materialize gradients in fp32 unconditionally.
// =============================================================================

template <typename T_mma>
__global__ void maxsim_padded_backward_kernel(
    const T_mma* __restrict__   queries,
    const T_mma* __restrict__   documents,
    const int* __restrict__     query_lengths,
    const int* __restrict__     doc_lengths,
    const int* __restrict__     argmax,
    const float* __restrict__   dscore,
    float* __restrict__         dqueries,
    float* __restrict__         ddocuments,
    int                         dim,
    int                         Lq_max,
    int                         Ld_max,
    int                         num_candidates) {
  const int pair_idx = blockIdx.x;
  const int tid      = threadIdx.x;
  const int lane     = tid & (kWarpSize - 1);
  const int warp_id  = tid >> 5;
  const int warps_per_block = blockDim.x >> 5;

  const int q_id    = pair_idx / num_candidates;
  const int d_id    = pair_idx;
  const int q_start = q_id * Lq_max;
  const int d_start = d_id * Ld_max;
  const int Lq      = query_lengths[q_id];
  const int Ld      = doc_lengths[d_id];

  const float g = dscore[pair_idx];

  // Each warp owns a stride-`warps_per_block` slice of q_toks. Lanes within
  // a warp split the dim dimension.
  for (int q_tok = warp_id; q_tok < Lq; q_tok += warps_per_block) {
    const int j = argmax[static_cast<size_t>(pair_idx) * Lq_max + q_tok];
    // Bounds guard: argmax should always be in [0, Ld) for valid q_toks,
    // but be defensive in case of garbage padding.
    if (j < 0 || j >= Ld) continue;

    const T_mma* q_row  = queries   + static_cast<size_t>(q_start + q_tok) * dim;
    const T_mma* d_row  = documents + static_cast<size_t>(d_start + j) * dim;
    float*       dq_row = dqueries  + static_cast<size_t>(q_start + q_tok) * dim;
    float*       dd_row = ddocuments + static_cast<size_t>(d_start + j) * dim;

    for (int k = lane; k < dim; k += kWarpSize) {
      const float q_v = load_f(q_row + k);
      const float d_v = load_f(d_row + k);
      atomicAdd(&dq_row[k], g * d_v);
      atomicAdd(&dd_row[k], g * q_v);
    }
  }
}

template <typename T_mma>
void launch_padded_backward(
    const torch::Tensor& queries,
    const torch::Tensor& documents,
    const torch::Tensor& query_lengths,
    const torch::Tensor& doc_lengths,
    const torch::Tensor& argmax,
    const torch::Tensor& dscore,
    torch::Tensor& dqueries,
    torch::Tensor& ddocuments,
    int dim, int Lq_max, int Ld_max, int num_candidates,
    int total_pairs, cudaStream_t stream) {
  // 4 warps per block — splits q_toks 4-way per pair. 8 warps and K-pair
  // packing (K=4, dq accumulated in registers) were both tried and gave
  // no improvement (the latter actually regressed). The kernel is close to
  // the L2-atomic-throughput floor on these shapes; cutting the block count
  // costs more in parallelism than it saves in atomic ops. See plan.md.
  constexpr int kBackwardWarps = 4;
  const int block_threads = kBackwardWarps * kWarpSize;
  maxsim_padded_backward_kernel<T_mma>
      <<<total_pairs, block_threads, 0, stream>>>(
          reinterpret_cast<const T_mma*>(queries.data_ptr()),
          reinterpret_cast<const T_mma*>(documents.data_ptr()),
          query_lengths.data_ptr<int>(),
          doc_lengths.data_ptr<int>(),
          argmax.data_ptr<int>(),
          dscore.data_ptr<float>(),
          dqueries.data_ptr<float>(),
          ddocuments.data_ptr<float>(),
          dim, Lq_max, Ld_max, num_candidates);
}

template <typename T>
__global__ void maxsim_padded_kernel(
    const T* __restrict__       queries,        // [B * Lq_max, dim]
    const int* __restrict__     query_lengths,  // [B]
    const T* __restrict__       documents,      // [B * C * Ld_max, dim]
    const int* __restrict__     doc_lengths,    // [B * C]
    float* __restrict__         scores,         // [B * C]
    int                         dim,
    int                         Lq_max,
    int                         Ld_max,
    int                         num_candidates) {
  const int pair_idx = blockIdx.x;
  const int tid      = threadIdx.x;
  const int lane     = tid & (kWarpSize - 1);
  const int warp_id  = tid >> 5;

  const int q_id    = pair_idx / num_candidates;
  const int d_id    = pair_idx;
  const int q_start = q_id * Lq_max;
  const int d_start = d_id * Ld_max;
  const int Lq      = query_lengths[q_id];
  const int Ld      = doc_lengths[d_id];

  extern __shared__ float maxes[];
  for (int i = tid; i < Lq; i += kBlockThreads) {
    maxes[i] = -CUDART_INF_F;
  }
  __syncthreads();

  for (int q_tok = warp_id; q_tok < Lq; q_tok += kNumWarps) {
    const T* qrow = queries + static_cast<size_t>(q_start + q_tok) * dim;
    float my_max  = -CUDART_INF_F;
    for (int d_tok = lane; d_tok < Ld; d_tok += kWarpSize) {
      const T* drow = documents + static_cast<size_t>(d_start + d_tok) * dim;
      my_max = fmaxf(my_max, dot_row(qrow, drow, dim));
    }
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, off));
    }
    if (lane == 0) maxes[q_tok] = my_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    float s = 0.0f;
    for (int i = lane; i < Lq; i += kWarpSize) s += maxes[i];
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
    }
    if (lane == 0) scores[pair_idx] = s;
  }
}

// ---- Dtype dispatch helper ------------------------------------------------

template <typename T>
void launch_packed(const torch::Tensor& queries,
                   const torch::Tensor& query_offsets,
                   const torch::Tensor& documents,
                   const torch::Tensor& document_offsets,
                   const torch::Tensor& pair_query_ids,
                   const torch::Tensor& pair_document_ids,
                   torch::Tensor& scores,
                   int dim,
                   int max_q_len,
                   cudaStream_t stream) {
  const int num_pairs = static_cast<int>(pair_query_ids.size(0));
  const size_t smem   = static_cast<size_t>(max_q_len) * sizeof(float);
  maxsim_packed_kernel<T><<<num_pairs, kBlockThreads, smem, stream>>>(
      reinterpret_cast<const T*>(queries.data_ptr()),
      query_offsets.data_ptr<int>(),
      reinterpret_cast<const T*>(documents.data_ptr()),
      document_offsets.data_ptr<int>(),
      pair_query_ids.data_ptr<int>(),
      pair_document_ids.data_ptr<int>(),
      scores.data_ptr<float>(),
      dim);
}

template <typename T>
void launch_padded(const torch::Tensor& queries,
                   const torch::Tensor& query_lengths,
                   const torch::Tensor& documents,
                   const torch::Tensor& doc_lengths,
                   torch::Tensor& scores,
                   int dim,
                   int Lq_max,
                   int Ld_max,
                   int num_candidates,
                   int total_pairs,
                   cudaStream_t stream) {
  const size_t smem = static_cast<size_t>(Lq_max) * sizeof(float);
  maxsim_padded_kernel<T><<<total_pairs, kBlockThreads, smem, stream>>>(
      reinterpret_cast<const T*>(queries.data_ptr()),
      query_lengths.data_ptr<int>(),
      reinterpret_cast<const T*>(documents.data_ptr()),
      doc_lengths.data_ptr<int>(),
      scores.data_ptr<float>(),
      dim, Lq_max, Ld_max, num_candidates);
}

template <typename T_mma>
void launch_padded_wmma_argmax(const torch::Tensor& queries,
                               const torch::Tensor& query_lengths,
                               const torch::Tensor& documents,
                               const torch::Tensor& doc_lengths,
                               torch::Tensor& scores,
                               torch::Tensor& argmax,
                               int dim,
                               int Lq_max,
                               int Ld_max,
                               int num_candidates,
                               int total_pairs,
                               cudaStream_t stream) {
  const int num_warps = (Lq_max + kWmmaM - 1) / kWmmaM;
  const int block_threads = num_warps * kWarpSize;
  const size_t maxes_bytes =
      static_cast<size_t>(Lq_max) * sizeof(float);
  const size_t argmax_bytes =
      static_cast<size_t>(Lq_max) * sizeof(int);
  const size_t Q_bytes =
      static_cast<size_t>(Lq_max) * dim * sizeof(T_mma);
  const size_t frag_bytes =
      static_cast<size_t>(num_warps) * 4 * kWmmaM * kWmmaN * sizeof(float);
  const size_t smem_bytes =
      maxes_bytes + argmax_bytes + Q_bytes + frag_bytes;
  maxsim_padded_wmma_argmax_kernel<T_mma>
      <<<total_pairs, block_threads, smem_bytes, stream>>>(
          reinterpret_cast<const T_mma*>(queries.data_ptr()),
          query_lengths.data_ptr<int>(),
          reinterpret_cast<const T_mma*>(documents.data_ptr()),
          doc_lengths.data_ptr<int>(),
          scores.data_ptr<float>(),
          argmax.data_ptr<int>(),
          dim, Lq_max, Ld_max, num_candidates);
}

template <typename T_mma>
void launch_padded_wmma(const torch::Tensor& queries,
                        const torch::Tensor& query_lengths,
                        const torch::Tensor& documents,
                        const torch::Tensor& doc_lengths,
                        torch::Tensor& scores,
                        int dim,
                        int Lq_max,
                        int Ld_max,
                        int num_candidates,
                        int total_pairs,
                        cudaStream_t stream) {
  const int num_warps = (Lq_max + kWmmaM - 1) / kWmmaM;
  const int block_threads = num_warps * kWarpSize;
  const size_t maxes_bytes = static_cast<size_t>(Lq_max) * sizeof(float);
  const size_t Q_bytes =
      static_cast<size_t>(Lq_max) * dim * sizeof(T_mma);
  // 4 slots per warp to back the 4-tile cascade in maxsim_padded_wmma_kernel.
  const size_t frag_bytes =
      static_cast<size_t>(num_warps) * 4 * kWmmaM * kWmmaN * sizeof(float);
  const size_t smem_bytes = maxes_bytes + Q_bytes + frag_bytes;
  maxsim_padded_wmma_kernel<T_mma>
      <<<total_pairs, block_threads, smem_bytes, stream>>>(
          reinterpret_cast<const T_mma*>(queries.data_ptr()),
          query_lengths.data_ptr<int>(),
          reinterpret_cast<const T_mma*>(documents.data_ptr()),
          doc_lengths.data_ptr<int>(),
          scores.data_ptr<float>(),
          dim, Lq_max, Ld_max, num_candidates);
}

}  // namespace

// ---- Host entry points (match torch-ext/torch_binding.h) -----------------

torch::Tensor maxsim_forward(torch::Tensor queries,
                             torch::Tensor query_offsets,
                             torch::Tensor documents,
                             torch::Tensor document_offsets,
                             torch::Tensor pair_query_ids,
                             torch::Tensor pair_document_ids,
                             int64_t max_q_len) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device(),
              "documents must be on the same device as queries");
  TORCH_CHECK(queries.dim() == 2 && documents.dim() == 2,
              "queries/documents must be 2-D [tokens, dim]");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim must match documents.dim");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");

  queries           = queries.contiguous();
  documents         = documents.contiguous();
  query_offsets     = query_offsets.contiguous().to(torch::kInt32);
  document_offsets  = document_offsets.contiguous().to(torch::kInt32);
  pair_query_ids    = pair_query_ids.contiguous().to(torch::kInt32);
  pair_document_ids = pair_document_ids.contiguous().to(torch::kInt32);

  const int64_t num_pairs = pair_query_ids.size(0);
  const int     dim       = static_cast<int>(queries.size(1));

  auto scores = torch::zeros(
      {num_pairs},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));
  if (num_pairs == 0) return scores;

  // max_q_len < 0 => infer it from query_offsets and validate offsets +
  // pair ids on the CPU side. Mirrors maxsim_metal/maxsim.mm so tests can
  // hit the same validation paths regardless of backend.
  int max_q_len_i;
  if (max_q_len < 0) {
    auto qoff_cpu = query_offsets.to(torch::kCPU);
    auto doff_cpu = document_offsets.to(torch::kCPU);
    auto qids_cpu = pair_query_ids.to(torch::kCPU);
    auto dids_cpu = pair_document_ids.to(torch::kCPU);

    auto validate_offsets =
        [](const torch::Tensor& off_cpu, int64_t total, const char* name) {
          const int32_t* p = off_cpu.data_ptr<int32_t>();
          int64_t n = off_cpu.size(0);
          TORCH_CHECK(n >= 2, name, " must have length >= 2");
          TORCH_CHECK(p[0] == 0, name, "[0] must equal 0, got ", p[0]);
          TORCH_CHECK((int64_t)p[n - 1] == total, name, "[-1] (",
                      p[n - 1], ") must equal total token count (", total, ")");
          int m = 0;
          for (int64_t i = 0; i + 1 < n; ++i) {
            int diff = p[i + 1] - p[i];
            TORCH_CHECK(diff > 0, "empty segment in ", name, " at index ", i);
            if (diff > m) m = diff;
          }
          return m;
        };
    auto validate_ids =
        [](const torch::Tensor& ids_cpu, int64_t upper, const char* name) {
          const int32_t* p = ids_cpu.data_ptr<int32_t>();
          int64_t n = ids_cpu.size(0);
          for (int64_t i = 0; i < n; ++i) {
            TORCH_CHECK(p[i] >= 0 && (int64_t)p[i] < upper, name, "[", i,
                        "] = ", p[i], " out of range [0, ", upper, ")");
          }
        };

    max_q_len_i = validate_offsets(qoff_cpu, queries.size(0), "query_offsets");
    (void)validate_offsets(doff_cpu, documents.size(0), "document_offsets");
    validate_ids(qids_cpu, qoff_cpu.size(0) - 1, "pair_query_ids");
    validate_ids(dids_cpu, doff_cpu.size(0) - 1, "pair_document_ids");
  } else {
    TORCH_CHECK(max_q_len > 0, "max_q_len must be > 0; got ", max_q_len);
    max_q_len_i = static_cast<int>(max_q_len);
  }

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (queries.scalar_type()) {
    case at::ScalarType::Float:
      launch_packed<float>(queries, query_offsets, documents, document_offsets,
                           pair_query_ids, pair_document_ids, scores, dim,
                           max_q_len_i, stream);
      break;
    case at::ScalarType::Half:
      launch_packed<__half>(queries, query_offsets, documents, document_offsets,
                            pair_query_ids, pair_document_ids, scores, dim,
                            max_q_len_i, stream);
      break;
    case at::ScalarType::BFloat16:
      launch_packed<__nv_bfloat16>(queries, query_offsets, documents,
                                   document_offsets, pair_query_ids,
                                   pair_document_ids, scores, dim,
                                   max_q_len_i, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "queries/documents dtype must be float32, float16, or bfloat16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

torch::Tensor maxsim_padded_forward(torch::Tensor queries,
                                    torch::Tensor query_lengths,
                                    torch::Tensor documents,
                                    torch::Tensor doc_lengths,
                                    int64_t Lq_max,
                                    int64_t Ld_max,
                                    int64_t num_candidates) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device(),
              "documents must be on the same device as queries");
  TORCH_CHECK(queries.dim() == 2 && documents.dim() == 2,
              "queries/documents must be 2-D");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim must match documents.dim");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(Lq_max > 0 && Ld_max > 0 && num_candidates > 0,
              "Lq_max/Ld_max/num_candidates must all be > 0");

  queries       = queries.contiguous();
  documents     = documents.contiguous();
  query_lengths = query_lengths.contiguous().to(torch::kInt32);
  doc_lengths   = doc_lengths.contiguous().to(torch::kInt32);

  const int64_t total_q_rows = queries.size(0);
  const int64_t total_d_rows = documents.size(0);
  TORCH_CHECK(total_q_rows % Lq_max == 0, "queries rows must be a multiple of Lq_max");
  TORCH_CHECK(total_d_rows % Ld_max == 0, "documents rows must be a multiple of Ld_max");
  const int64_t B           = total_q_rows / Lq_max;
  const int64_t total_pairs = total_d_rows / Ld_max;
  TORCH_CHECK(total_pairs == B * num_candidates,
              "documents shape inconsistent with B * num_candidates");

  const int dim = static_cast<int>(queries.size(1));

  auto scores = torch::zeros(
      {total_pairs},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));
  if (total_pairs == 0) return scores;

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // Stage 2: WMMA path. Eligible when input is fp16/bf16, dim is a multiple
  // of the 16-wide K-tile, and Lq_max is a multiple of the 16-wide M-tile
  // (which keeps `Q_shared` sizing matched between host and kernel — no
  // overflow when zero-padding rows beyond Lq).
  const bool wmma_eligible = (dim % kWmmaK == 0)
                          && (Lq_max % kWmmaM == 0)
                          && (Lq_max >= kWmmaM)
                          && (Ld_max >= 1);

  switch (queries.scalar_type()) {
    case at::ScalarType::Float:
      launch_padded<float>(queries, query_lengths, documents, doc_lengths,
                           scores, dim, static_cast<int>(Lq_max),
                           static_cast<int>(Ld_max),
                           static_cast<int>(num_candidates),
                           static_cast<int>(total_pairs), stream);
      break;
    case at::ScalarType::Half:
      if (wmma_eligible) {
        launch_padded_wmma<__half>(queries, query_lengths, documents,
                                   doc_lengths, scores, dim,
                                   static_cast<int>(Lq_max),
                                   static_cast<int>(Ld_max),
                                   static_cast<int>(num_candidates),
                                   static_cast<int>(total_pairs), stream);
      } else {
        launch_padded<__half>(queries, query_lengths, documents, doc_lengths,
                              scores, dim, static_cast<int>(Lq_max),
                              static_cast<int>(Ld_max),
                              static_cast<int>(num_candidates),
                              static_cast<int>(total_pairs), stream);
      }
      break;
    case at::ScalarType::BFloat16:
      if (wmma_eligible) {
        launch_padded_wmma<__nv_bfloat16>(queries, query_lengths, documents,
                                          doc_lengths, scores, dim,
                                          static_cast<int>(Lq_max),
                                          static_cast<int>(Ld_max),
                                          static_cast<int>(num_candidates),
                                          static_cast<int>(total_pairs),
                                          stream);
      } else {
        launch_padded<__nv_bfloat16>(queries, query_lengths, documents,
                                     doc_lengths, scores, dim,
                                     static_cast<int>(Lq_max),
                                     static_cast<int>(Ld_max),
                                     static_cast<int>(num_candidates),
                                     static_cast<int>(total_pairs), stream);
      }
      break;
    default:
      TORCH_CHECK(false,
                  "queries/documents dtype must be float32, float16, or bfloat16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

// =============================================================================
// Padded forward + argmax (training mode). Identical validation to
// maxsim_padded_forward but dispatches the argmax-aware WMMA kernel and
// returns both score and argmax tensors. fp32 inputs and shapes that don't
// satisfy the WMMA eligibility gate (dim % 16 != 0 or Lq_max % 16 != 0) are
// rejected for now -- the scalar argmax fallback is future work, and our
// target workloads (ColBERT-style training) all use fp16/bf16 with dim=128
// or similar.
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor> maxsim_padded_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor query_lengths,
    torch::Tensor documents,
    torch::Tensor doc_lengths,
    int64_t Lq_max,
    int64_t Ld_max,
    int64_t num_candidates) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device(),
              "documents must be on the same device as queries");
  TORCH_CHECK(queries.dim() == 2 && documents.dim() == 2,
              "queries/documents must be 2-D");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim must match documents.dim");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(Lq_max > 0 && Ld_max > 0 && num_candidates > 0,
              "Lq_max/Ld_max/num_candidates must all be > 0");

  queries       = queries.contiguous();
  documents     = documents.contiguous();
  query_lengths = query_lengths.contiguous().to(torch::kInt32);
  doc_lengths   = doc_lengths.contiguous().to(torch::kInt32);

  const int64_t total_q_rows = queries.size(0);
  const int64_t total_d_rows = documents.size(0);
  TORCH_CHECK(total_q_rows % Lq_max == 0,
              "queries rows must be a multiple of Lq_max");
  TORCH_CHECK(total_d_rows % Ld_max == 0,
              "documents rows must be a multiple of Ld_max");
  const int64_t B = total_q_rows / Lq_max;
  const int64_t total_pairs = total_d_rows / Ld_max;
  TORCH_CHECK(total_pairs == B * num_candidates,
              "documents shape inconsistent with B * num_candidates");

  const int dim = static_cast<int>(queries.size(1));

  auto scores = torch::zeros(
      {total_pairs},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));
  auto argmax = torch::zeros(
      {total_pairs, Lq_max},
      torch::TensorOptions().dtype(torch::kInt32).device(queries.device()));
  if (total_pairs == 0) return std::make_tuple(scores, argmax);

  // WMMA-only path: requires fp16/bf16 + dim and Lq_max both 16-aligned.
  TORCH_CHECK(dim % kWmmaK == 0,
              "argmax kernel requires dim % 16 == 0; got dim = ", dim);
  TORCH_CHECK(Lq_max % kWmmaM == 0,
              "argmax kernel requires Lq_max % 16 == 0; got Lq_max = ",
              Lq_max);
  TORCH_CHECK(Lq_max >= kWmmaM,
              "argmax kernel requires Lq_max >= 16; got Lq_max = ", Lq_max);

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (queries.scalar_type()) {
    case at::ScalarType::Half:
      launch_padded_wmma_argmax<__half>(
          queries, query_lengths, documents, doc_lengths, scores, argmax,
          dim, static_cast<int>(Lq_max), static_cast<int>(Ld_max),
          static_cast<int>(num_candidates),
          static_cast<int>(total_pairs), stream);
      break;
    case at::ScalarType::BFloat16:
      launch_padded_wmma_argmax<__nv_bfloat16>(
          queries, query_lengths, documents, doc_lengths, scores, argmax,
          dim, static_cast<int>(Lq_max), static_cast<int>(Ld_max),
          static_cast<int>(num_candidates),
          static_cast<int>(total_pairs), stream);
      break;
    default:
      TORCH_CHECK(false,
                  "argmax kernel currently supports float16 / bfloat16 only "
                  "(fp32 scalar argmax fallback is future work)");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(scores, argmax);
}

// =============================================================================
// Padded backward. Takes (dscore, queries, documents, qlen, dlen, argmax)
// from a previous forward+argmax call and returns (dqueries, ddocuments).
// Both grad tensors are fp32 regardless of input dtype.
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor> maxsim_padded_backward(
    torch::Tensor dscore,
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor query_lengths,
    torch::Tensor doc_lengths,
    torch::Tensor argmax,
    int64_t Lq_max,
    int64_t Ld_max,
    int64_t num_candidates) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              dscore.device() == queries.device() &&
              argmax.device() == queries.device() &&
              query_lengths.device() == queries.device() &&
              doc_lengths.device() == queries.device(),
              "all backward inputs must share a device");
  TORCH_CHECK(queries.dim() == 2 && documents.dim() == 2,
              "queries/documents must be 2-D");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim must match documents.dim");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(dscore.scalar_type() == at::ScalarType::Float,
              "dscore must be fp32");
  TORCH_CHECK(argmax.scalar_type() == at::ScalarType::Int,
              "argmax must be int32");
  TORCH_CHECK(Lq_max > 0 && Ld_max > 0 && num_candidates > 0,
              "Lq_max/Ld_max/num_candidates must all be > 0");

  queries       = queries.contiguous();
  documents     = documents.contiguous();
  query_lengths = query_lengths.contiguous().to(torch::kInt32);
  doc_lengths   = doc_lengths.contiguous().to(torch::kInt32);
  argmax        = argmax.contiguous();
  dscore        = dscore.contiguous();

  const int64_t total_q_rows = queries.size(0);
  const int64_t total_d_rows = documents.size(0);
  TORCH_CHECK(total_q_rows % Lq_max == 0,
              "queries rows must be a multiple of Lq_max");
  TORCH_CHECK(total_d_rows % Ld_max == 0,
              "documents rows must be a multiple of Ld_max");
  const int64_t B = total_q_rows / Lq_max;
  const int64_t total_pairs = total_d_rows / Ld_max;
  TORCH_CHECK(total_pairs == B * num_candidates,
              "documents shape inconsistent with B * num_candidates");
  TORCH_CHECK(argmax.numel() == total_pairs * Lq_max,
              "argmax must have shape [total_pairs, Lq_max]");
  TORCH_CHECK(dscore.numel() == total_pairs,
              "dscore must have shape [total_pairs]");

  const int dim = static_cast<int>(queries.size(1));

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto dqueries   = torch::zeros({total_q_rows, dim}, fp32_opts);
  auto ddocuments = torch::zeros({total_d_rows, dim}, fp32_opts);
  if (total_pairs == 0) return std::make_tuple(dqueries, ddocuments);

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (queries.scalar_type()) {
    case at::ScalarType::Half:
      launch_padded_backward<__half>(
          queries, documents, query_lengths, doc_lengths, argmax, dscore,
          dqueries, ddocuments, dim,
          static_cast<int>(Lq_max), static_cast<int>(Ld_max),
          static_cast<int>(num_candidates),
          static_cast<int>(total_pairs), stream);
      break;
    case at::ScalarType::BFloat16:
      launch_padded_backward<__nv_bfloat16>(
          queries, documents, query_lengths, doc_lengths, argmax, dscore,
          dqueries, ddocuments, dim,
          static_cast<int>(Lq_max), static_cast<int>(Ld_max),
          static_cast<int>(num_candidates),
          static_cast<int>(total_pairs), stream);
      break;
    default:
      TORCH_CHECK(false,
                  "backward kernel currently supports fp16 / bf16 only");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(dqueries, ddocuments);
}

// =============================================================================
// Contrastive (all-pairs) MaxSim
// =============================================================================
//
// Cross-product of every query in `queries[Nq, Lq, D]` against every doc in
// `documents[total_d_tokens, D]` (packed via document_offsets[Nb+1]). The block
// layout is the same per-pair pattern as the padded kernel (one block per
// (q_idx, d_idx)); the only meaningful difference is how each block resolves
// q_start, d_start, and Ld. Lq is uniform across queries by API contract.
//
// The 4/2/1-tile WMMA cascade body is mechanically identical to the padded
// kernels above. We accept the duplication for now; once the design has
// settled and we've profiled, the cascade can be factored into a shared
// __device__ helper.

template <typename T_mma>
__global__ void maxsim_contrastive_wmma_kernel(
    const T_mma* __restrict__   queries,    // [Nq * Lq, D]
    const T_mma* __restrict__   documents,  // [total_d_tokens, D]
    const int* __restrict__     document_offsets, // [Nb + 1]
    float* __restrict__         scores,     // [Nq * Nb]
    int                         dim,
    int                         Lq,
    int                         Nb) {
  using namespace nvcuda::wmma;

  const int pair_idx  = blockIdx.x;
  const int tid       = threadIdx.x;
  const int lane      = tid & (kWarpSize - 1);
  const int warp_id   = tid >> 5;

  const int q_idx   = pair_idx / Nb;
  const int d_idx   = pair_idx - q_idx * Nb;
  const int q_start = q_idx * Lq;
  const int d_start = document_offsets[d_idx];
  const int Ld      = document_offsets[d_idx + 1] - d_start;

  const int Lq_aligned = ((Lq + kWmmaM - 1) / kWmmaM) * kWmmaM;
  const int Ld_aligned = (Ld / kWmmaN) * kWmmaN;

  extern __shared__ unsigned char smem[];
  float* maxes = reinterpret_cast<float*>(smem);
  T_mma* Q_shared = reinterpret_cast<T_mma*>(maxes + Lq);
  float* frag_store = reinterpret_cast<float*>(
      Q_shared + static_cast<size_t>(Lq_aligned) * dim);

  for (int i = tid; i < Lq; i += blockDim.x) {
    maxes[i] = -CUDART_INF_F;
  }
  for (int i = tid; i < Lq * dim; i += blockDim.x) {
    Q_shared[i] = queries[static_cast<size_t>(q_start) * dim + i];
  }
  for (int i = Lq * dim + tid; i < Lq_aligned * dim; i += blockDim.x) {
    Q_shared[i] = zero_v<T_mma>();
  }
  __syncthreads();

  const int qb = warp_id * kWmmaM;
  if (qb < Lq_aligned) {
    const T_mma* Q_qb   = Q_shared + static_cast<size_t>(qb) * dim;
    const T_mma* D_base = documents + static_cast<size_t>(d_start) * dim;
    constexpr int kSlot = kWmmaM * kWmmaN;
    float* slot = frag_store + warp_id * (4 * kSlot);

    const int db_quad_end =
        (Ld_aligned / (4 * kWmmaN)) * (4 * kWmmaN);
    const int db_pair_end =
        db_quad_end +
        ((Ld_aligned - db_quad_end) / (2 * kWmmaN)) * (2 * kWmmaN);

    #define LOAD_A(K) \
      fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, T_mma, row_major> a_frag; \
      load_matrix_sync(a_frag, Q_qb + (K), dim);
    #define MMA_TILE(C, DB, K) \
      { \
        fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, T_mma, col_major> b_frag; \
        load_matrix_sync(b_frag, \
            D_base + static_cast<size_t>(DB) * dim + (K), dim); \
        mma_sync(C, a_frag, b_frag, C); \
      }
    #define ROW_MAX_OVER_N(N_TILES) \
      { \
        const int row    = lane >> 1; \
        const int col_lo = (lane & 1) << 3; \
        float lane_max = -CUDART_INF_F; \
        _Pragma("unroll") \
        for (int t = 0; t < (N_TILES); ++t) { \
          const float* row_ptr = slot + t * kSlot + row * kWmmaN; \
          _Pragma("unroll") \
          for (int c = 0; c < 8; ++c) { \
            lane_max = fmaxf(lane_max, row_ptr[col_lo + c]); \
          } \
        } \
        lane_max = fmaxf(lane_max, __shfl_xor_sync(0xffffffff, lane_max, 1)); \
        if ((lane & 1) == 0) { \
          const int q_tok = qb + row; \
          if (q_tok < Lq) maxes[q_tok] = fmaxf(maxes[q_tok], lane_max); \
        } \
      }

    for (int db = 0; db < db_quad_end; db += 4 * kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0, c1, c2, c3;
      fill_fragment(c0, 0.0f);
      fill_fragment(c1, 0.0f);
      fill_fragment(c2, 0.0f);
      fill_fragment(c3, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db + 0 * kWmmaN, k);
        MMA_TILE(c1, db + 1 * kWmmaN, k);
        MMA_TILE(c2, db + 2 * kWmmaN, k);
        MMA_TILE(c3, db + 3 * kWmmaN, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 1 * kSlot, c1, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 2 * kSlot, c2, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 3 * kSlot, c3, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_OVER_N(4);
      __syncwarp();
    }

    for (int db = db_quad_end; db < db_pair_end; db += 2 * kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0, c1;
      fill_fragment(c0, 0.0f);
      fill_fragment(c1, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db + 0 * kWmmaN, k);
        MMA_TILE(c1, db + 1 * kWmmaN, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 1 * kSlot, c1, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_OVER_N(2);
      __syncwarp();
    }

    for (int db = db_pair_end; db < Ld_aligned; db += kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0;
      fill_fragment(c0, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_OVER_N(1);
      __syncwarp();
    }

    #undef LOAD_A
    #undef MMA_TILE
    #undef ROW_MAX_OVER_N

    for (int q_off = 0; q_off < kWmmaM; ++q_off) {
      const int q_tok = qb + q_off;
      if (q_tok >= Lq) break;
      const T_mma* qrow = Q_qb + static_cast<size_t>(q_off) * dim;
      float my_max = -CUDART_INF_F;
      for (int d_tok = Ld_aligned + lane; d_tok < Ld; d_tok += kWarpSize) {
        const T_mma* drow = D_base + static_cast<size_t>(d_tok) * dim;
        my_max = fmaxf(my_max, dot_row(qrow, drow, dim));
      }
      for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, off));
      }
      if (lane == 0) maxes[q_tok] = fmaxf(maxes[q_tok], my_max);
    }
  }

  __syncthreads();

  if (warp_id == 0) {
    float s = 0.0f;
    for (int i = lane; i < Lq; i += kWarpSize) s += maxes[i];
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
    }
    if (lane == 0) scores[pair_idx] = s;
  }
}

template <typename T_mma>
__global__ void maxsim_contrastive_wmma_argmax_kernel(
    const T_mma* __restrict__   queries,    // [Nq * Lq, D]
    const T_mma* __restrict__   documents,  // [total_d_tokens, D]
    const int* __restrict__     document_offsets, // [Nb + 1]
    float* __restrict__         scores,     // [Nq * Nb]
    int* __restrict__           argmax_out, // [Nq * Nb, Lq]
    int                         dim,
    int                         Lq,
    int                         Nb) {
  using namespace nvcuda::wmma;

  const int pair_idx  = blockIdx.x;
  const int tid       = threadIdx.x;
  const int lane      = tid & (kWarpSize - 1);
  const int warp_id   = tid >> 5;

  const int q_idx   = pair_idx / Nb;
  const int d_idx   = pair_idx - q_idx * Nb;
  const int q_start = q_idx * Lq;
  const int d_start = document_offsets[d_idx];
  const int Ld      = document_offsets[d_idx + 1] - d_start;

  const int Lq_aligned = ((Lq + kWmmaM - 1) / kWmmaM) * kWmmaM;
  const int Ld_aligned = (Ld / kWmmaN) * kWmmaN;

  // Same smem layout as padded_with_argmax kernel, sized by Lq (fixed):
  //   maxes      : Lq floats
  //   argmax_buf : Lq int32
  //   Q_shared   : Lq_aligned * dim T_mma  (zero-padded rows up to Lq_aligned)
  //   frag_store : num_warps * 4 * kWmmaM * kWmmaN floats
  extern __shared__ unsigned char smem[];
  float* maxes      = reinterpret_cast<float*>(smem);
  int*   argmax_buf = reinterpret_cast<int*>(maxes + Lq);
  T_mma* Q_shared   = reinterpret_cast<T_mma*>(argmax_buf + Lq);
  float* frag_store = reinterpret_cast<float*>(
      Q_shared + static_cast<size_t>(Lq_aligned) * dim);

  for (int i = tid; i < Lq; i += blockDim.x) {
    maxes[i]      = -CUDART_INF_F;
    argmax_buf[i] = 0;
  }
  // Stage queries[q_idx, :, :] — already contiguous Lq rows of dim each.
  for (int i = tid; i < Lq * dim; i += blockDim.x) {
    Q_shared[i] = queries[static_cast<size_t>(q_start) * dim + i];
  }
  for (int i = Lq * dim + tid; i < Lq_aligned * dim; i += blockDim.x) {
    Q_shared[i] = zero_v<T_mma>();
  }
  __syncthreads();

  const int qb = warp_id * kWmmaM;
  if (qb < Lq_aligned) {
    const T_mma* Q_qb   = Q_shared + static_cast<size_t>(qb) * dim;
    const T_mma* D_base = documents + static_cast<size_t>(d_start) * dim;
    constexpr int kSlot = kWmmaM * kWmmaN;
    float* slot = frag_store + warp_id * (4 * kSlot);

    const int db_quad_end =
        (Ld_aligned / (4 * kWmmaN)) * (4 * kWmmaN);
    const int db_pair_end =
        db_quad_end +
        ((Ld_aligned - db_quad_end) / (2 * kWmmaN)) * (2 * kWmmaN);

    #define LOAD_A(K) \
      fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, T_mma, row_major> a_frag; \
      load_matrix_sync(a_frag, Q_qb + (K), dim);
    #define MMA_TILE(C, DB, K) \
      { \
        fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, T_mma, col_major> b_frag; \
        load_matrix_sync(b_frag, \
            D_base + static_cast<size_t>(DB) * dim + (K), dim); \
        mma_sync(C, a_frag, b_frag, C); \
      }
    #define ROW_MAX_ARGMAX_OVER_N(N_TILES) \
      { \
        const int row    = lane >> 1; \
        const int col_lo = (lane & 1) << 3; \
        float lane_max = -CUDART_INF_F; \
        int   lane_arg = 0; \
        _Pragma("unroll") \
        for (int t = 0; t < (N_TILES); ++t) { \
          const float* row_ptr = slot + t * kSlot + row * kWmmaN; \
          _Pragma("unroll") \
          for (int c = 0; c < 8; ++c) { \
            float v = row_ptr[col_lo + c]; \
            int   d = db + t * kWmmaN + col_lo + c; \
            if (v > lane_max) { lane_max = v; lane_arg = d; } \
          } \
        } \
        float pm = __shfl_xor_sync(0xffffffff, lane_max, 1); \
        int   pa = __shfl_xor_sync(0xffffffff, lane_arg, 1); \
        bool take = pm > lane_max || (pm == lane_max && pa < lane_arg); \
        if (take) { lane_max = pm; lane_arg = pa; } \
        if ((lane & 1) == 0) { \
          const int q_tok = qb + row; \
          if (q_tok < Lq && lane_max > maxes[q_tok]) { \
            maxes[q_tok]      = lane_max; \
            argmax_buf[q_tok] = lane_arg; \
          } \
        } \
      }

    for (int db = 0; db < db_quad_end; db += 4 * kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0, c1, c2, c3;
      fill_fragment(c0, 0.0f);
      fill_fragment(c1, 0.0f);
      fill_fragment(c2, 0.0f);
      fill_fragment(c3, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db + 0 * kWmmaN, k);
        MMA_TILE(c1, db + 1 * kWmmaN, k);
        MMA_TILE(c2, db + 2 * kWmmaN, k);
        MMA_TILE(c3, db + 3 * kWmmaN, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 1 * kSlot, c1, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 2 * kSlot, c2, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 3 * kSlot, c3, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_ARGMAX_OVER_N(4);
      __syncwarp();
    }

    for (int db = db_quad_end; db < db_pair_end; db += 2 * kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0, c1;
      fill_fragment(c0, 0.0f);
      fill_fragment(c1, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db + 0 * kWmmaN, k);
        MMA_TILE(c1, db + 1 * kWmmaN, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      store_matrix_sync(slot + 1 * kSlot, c1, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_ARGMAX_OVER_N(2);
      __syncwarp();
    }

    for (int db = db_pair_end; db < Ld_aligned; db += kWmmaN) {
      fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0;
      fill_fragment(c0, 0.0f);
      for (int k = 0; k < dim; k += kWmmaK) {
        LOAD_A(k);
        MMA_TILE(c0, db, k);
      }
      store_matrix_sync(slot + 0 * kSlot, c0, kWmmaN, mem_row_major);
      __syncwarp();
      ROW_MAX_ARGMAX_OVER_N(1);
      __syncwarp();
    }

    #undef LOAD_A
    #undef MMA_TILE
    #undef ROW_MAX_ARGMAX_OVER_N

    // Scalar tail.
    for (int q_off = 0; q_off < kWmmaM; ++q_off) {
      const int q_tok = qb + q_off;
      if (q_tok >= Lq) break;
      const T_mma* qrow = Q_qb + static_cast<size_t>(q_off) * dim;
      float my_max = -CUDART_INF_F;
      int   my_arg = 0;
      for (int d_tok = Ld_aligned + lane; d_tok < Ld; d_tok += kWarpSize) {
        const T_mma* drow = D_base + static_cast<size_t>(d_tok) * dim;
        float v = dot_row(qrow, drow, dim);
        if (v > my_max) { my_max = v; my_arg = d_tok; }
      }
      for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        float pm = __shfl_xor_sync(0xffffffff, my_max, off);
        int   pa = __shfl_xor_sync(0xffffffff, my_arg, off);
        bool take = pm > my_max || (pm == my_max && pa < my_arg);
        if (take) { my_max = pm; my_arg = pa; }
      }
      if (lane == 0 && my_max > maxes[q_tok]) {
        maxes[q_tok]      = my_max;
        argmax_buf[q_tok] = my_arg;
      }
    }
  }

  __syncthreads();

  if (warp_id == 0) {
    float s = 0.0f;
    for (int i = lane; i < Lq; i += kWarpSize) s += maxes[i];
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
    }
    if (lane == 0) scores[pair_idx] = s;
  }

  // Write argmax_buf to global. argmax_out has shape [Nq*Nb, Lq] — exactly
  // Lq slots per pair, no padding (unlike padded which uses Lq_max).
  int* argmax_pair = argmax_out + static_cast<size_t>(pair_idx) * Lq;
  for (int i = tid; i < Lq; i += blockDim.x) {
    argmax_pair[i] = argmax_buf[i];
  }
}

template <typename T_mma>
void launch_contrastive(const torch::Tensor& queries,
                        const torch::Tensor& documents,
                        const torch::Tensor& document_offsets,
                        torch::Tensor& scores,
                        int dim, int Lq, int Nb, int total_pairs,
                        cudaStream_t stream) {
  const int num_warps = (Lq + kWmmaM - 1) / kWmmaM;
  const int block_threads = num_warps * kWarpSize;
  const int Lq_aligned = num_warps * kWmmaM;
  const size_t maxes_bytes = static_cast<size_t>(Lq) * sizeof(float);
  const size_t Q_bytes =
      static_cast<size_t>(Lq_aligned) * dim * sizeof(T_mma);
  const size_t frag_bytes =
      static_cast<size_t>(num_warps) * 4 * kWmmaM * kWmmaN * sizeof(float);
  const size_t smem_bytes = maxes_bytes + Q_bytes + frag_bytes;
  maxsim_contrastive_wmma_kernel<T_mma>
      <<<total_pairs, block_threads, smem_bytes, stream>>>(
          reinterpret_cast<const T_mma*>(queries.data_ptr()),
          reinterpret_cast<const T_mma*>(documents.data_ptr()),
          document_offsets.data_ptr<int>(),
          scores.data_ptr<float>(),
          dim, Lq, Nb);
}

template <typename T_mma>
void launch_contrastive_argmax(const torch::Tensor& queries,
                               const torch::Tensor& documents,
                               const torch::Tensor& document_offsets,
                               torch::Tensor& scores,
                               torch::Tensor& argmax,
                               int dim, int Lq, int Nb, int total_pairs,
                               cudaStream_t stream) {
  const int num_warps = (Lq + kWmmaM - 1) / kWmmaM;
  const int block_threads = num_warps * kWarpSize;
  const int Lq_aligned = num_warps * kWmmaM;
  const size_t maxes_bytes  = static_cast<size_t>(Lq) * sizeof(float);
  const size_t argmax_bytes = static_cast<size_t>(Lq) * sizeof(int);
  const size_t Q_bytes =
      static_cast<size_t>(Lq_aligned) * dim * sizeof(T_mma);
  const size_t frag_bytes =
      static_cast<size_t>(num_warps) * 4 * kWmmaM * kWmmaN * sizeof(float);
  const size_t smem_bytes = maxes_bytes + argmax_bytes + Q_bytes + frag_bytes;
  maxsim_contrastive_wmma_argmax_kernel<T_mma>
      <<<total_pairs, block_threads, smem_bytes, stream>>>(
          reinterpret_cast<const T_mma*>(queries.data_ptr()),
          reinterpret_cast<const T_mma*>(documents.data_ptr()),
          document_offsets.data_ptr<int>(),
          scores.data_ptr<float>(),
          argmax.data_ptr<int>(),
          dim, Lq, Nb);
}

std::tuple<torch::Tensor, torch::Tensor> maxsim_contrastive_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor document_offsets) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              document_offsets.device() == queries.device(),
              "all inputs must share a device");
  TORCH_CHECK(queries.dim() == 3, "queries must be [Nq, Lq, D]");
  TORCH_CHECK(documents.dim() == 2, "documents must be [total_d_tokens, D]");
  TORCH_CHECK(document_offsets.dim() == 1, "document_offsets must be 1-D");
  TORCH_CHECK(queries.size(2) == documents.size(1),
              "queries.D must match documents.D");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");

  queries    = queries.contiguous();
  documents  = documents.contiguous();
  document_offsets = document_offsets.contiguous().to(torch::kInt32);

  const int Nq  = static_cast<int>(queries.size(0));
  const int Lq  = static_cast<int>(queries.size(1));
  const int dim = static_cast<int>(queries.size(2));
  const int Nb  = static_cast<int>(document_offsets.size(0)) - 1;
  TORCH_CHECK(Nb >= 0, "document_offsets must have at least 1 element");
  TORCH_CHECK(Lq > 0, "Lq must be > 0");
  TORCH_CHECK(dim > 0, "dim must be > 0");
  TORCH_CHECK(Lq % kWmmaM == 0,
              "Lq must be a multiple of 16 for the CUDA WMMA path "
              "(got Lq=", Lq, "; scalar fp32 fallback is future work)");
  TORCH_CHECK(dim % kWmmaK == 0,
              "dim must be a multiple of 16 for the CUDA WMMA path "
              "(got dim=", dim, "; the typical ColBERT dims 32/64/96/128 "
              "all satisfy this)");

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(queries.device());
  auto scores = torch::empty({Nq, Nb}, fp32_opts);
  auto argmax = torch::empty({Nq, Nb, Lq}, i32_opts);

  const int total_pairs = Nq * Nb;
  if (total_pairs == 0) return std::make_tuple(scores, argmax);

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (queries.scalar_type()) {
    case at::ScalarType::Half:
      launch_contrastive_argmax<__half>(
          queries, documents, document_offsets, scores, argmax,
          dim, Lq, Nb, total_pairs, stream);
      break;
    case at::ScalarType::BFloat16:
      launch_contrastive_argmax<__nv_bfloat16>(
          queries, documents, document_offsets, scores, argmax,
          dim, Lq, Nb, total_pairs, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "contrastive forward+argmax currently supports fp16 / bf16 "
                  "only");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(scores, argmax);
}

torch::Tensor maxsim_contrastive_forward(torch::Tensor queries,
                                         torch::Tensor documents,
                                         torch::Tensor document_offsets) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              document_offsets.device() == queries.device(),
              "all inputs must share a device");
  TORCH_CHECK(queries.dim() == 3, "queries must be [Nq, Lq, D]");
  TORCH_CHECK(documents.dim() == 2, "documents must be [total_d_tokens, D]");
  TORCH_CHECK(document_offsets.dim() == 1, "document_offsets must be 1-D");
  TORCH_CHECK(queries.size(2) == documents.size(1),
              "queries.D must match documents.D");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");

  queries    = queries.contiguous();
  documents  = documents.contiguous();
  document_offsets = document_offsets.contiguous().to(torch::kInt32);

  const int Nq  = static_cast<int>(queries.size(0));
  const int Lq  = static_cast<int>(queries.size(1));
  const int dim = static_cast<int>(queries.size(2));
  const int Nb  = static_cast<int>(document_offsets.size(0)) - 1;
  TORCH_CHECK(Nb >= 0, "document_offsets must have at least 1 element");
  TORCH_CHECK(Lq > 0, "Lq must be > 0");
  TORCH_CHECK(dim > 0, "dim must be > 0");
  TORCH_CHECK(Lq % kWmmaM == 0,
              "Lq must be a multiple of 16 for the CUDA WMMA path "
              "(got Lq=", Lq, "; scalar fp32 fallback is future work)");
  TORCH_CHECK(dim % kWmmaK == 0,
              "dim must be a multiple of 16 for the CUDA WMMA path "
              "(got dim=", dim, "; the typical ColBERT dims 32/64/96/128 "
              "all satisfy this)");

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto scores = torch::empty({Nq, Nb}, fp32_opts);

  const int total_pairs = Nq * Nb;
  if (total_pairs == 0) return scores;

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (queries.scalar_type()) {
    case at::ScalarType::Half:
      launch_contrastive<__half>(
          queries, documents, document_offsets, scores,
          dim, Lq, Nb, total_pairs, stream);
      break;
    case at::ScalarType::BFloat16:
      launch_contrastive<__nv_bfloat16>(
          queries, documents, document_offsets, scores,
          dim, Lq, Nb, total_pairs, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "contrastive forward currently supports fp16 / bf16 only");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

// Contrastive backward. One block per (q_idx, d_idx); warps split q_toks,
// lanes split dim. Both gradient tensors are fp32 (matches padded backward).
//
// Atomic contention pattern (different from padded!):
//   dq[q_idx, q_tok, k]   is shared by Nb docs  -> Nb-fold cross-block contention
//   dd[d_offset + j*, k]  is shared by Nq queries -> up to Nq-fold contention
// Both sides have contention here, where padded only had it on dq.
template <typename T_mma>
__global__ void maxsim_contrastive_backward_kernel(
    const T_mma* __restrict__   queries,    // [Nq * Lq, D]
    const T_mma* __restrict__   documents,  // [total_d_tokens, D]
    const int* __restrict__     document_offsets, // [Nb + 1]
    const int* __restrict__     argmax,     // [Nq * Nb, Lq]
    const float* __restrict__   dscore,     // [Nq * Nb]
    float* __restrict__         dqueries,   // [Nq * Lq, D]   fp32
    float* __restrict__         ddocuments, // [total_d_tokens, D] fp32
    int                         dim,
    int                         Lq,
    int                         Nb) {
  const int pair_idx = blockIdx.x;
  const int tid      = threadIdx.x;
  const int lane     = tid & (kWarpSize - 1);
  const int warp_id  = tid >> 5;
  const int warps_per_block = blockDim.x >> 5;

  const int q_idx   = pair_idx / Nb;
  const int d_idx   = pair_idx - q_idx * Nb;
  const int q_start = q_idx * Lq;
  const int d_start = document_offsets[d_idx];
  const int Ld      = document_offsets[d_idx + 1] - d_start;

  const float g = dscore[pair_idx];

  for (int q_tok = warp_id; q_tok < Lq; q_tok += warps_per_block) {
    const int j = argmax[static_cast<size_t>(pair_idx) * Lq + q_tok];
    if (j < 0 || j >= Ld) continue;

    const T_mma* q_row  = queries   + static_cast<size_t>(q_start + q_tok) * dim;
    const T_mma* d_row  = documents + static_cast<size_t>(d_start + j) * dim;
    float*       dq_row = dqueries  + static_cast<size_t>(q_start + q_tok) * dim;
    float*       dd_row = ddocuments + static_cast<size_t>(d_start + j) * dim;

    for (int k = lane; k < dim; k += kWarpSize) {
      const float q_v = load_f(q_row + k);
      const float d_v = load_f(d_row + k);
      atomicAdd(&dq_row[k], g * d_v);
      atomicAdd(&dd_row[k], g * q_v);
    }
  }
}

template <typename T_mma>
void launch_contrastive_backward(const torch::Tensor& queries,
                                 const torch::Tensor& documents,
                                 const torch::Tensor& document_offsets,
                                 const torch::Tensor& argmax,
                                 const torch::Tensor& dscore,
                                 torch::Tensor& dqueries,
                                 torch::Tensor& ddocuments,
                                 int dim, int Lq, int Nb, int total_pairs,
                                 cudaStream_t stream) {
  constexpr int kBackwardWarps = 4;
  const int block_threads = kBackwardWarps * kWarpSize;
  maxsim_contrastive_backward_kernel<T_mma>
      <<<total_pairs, block_threads, 0, stream>>>(
          reinterpret_cast<const T_mma*>(queries.data_ptr()),
          reinterpret_cast<const T_mma*>(documents.data_ptr()),
          document_offsets.data_ptr<int>(),
          argmax.data_ptr<int>(),
          dscore.data_ptr<float>(),
          dqueries.data_ptr<float>(),
          ddocuments.data_ptr<float>(),
          dim, Lq, Nb);
}

std::tuple<torch::Tensor, torch::Tensor> maxsim_contrastive_backward(
    torch::Tensor dscore,
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor argmax) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              document_offsets.device() == queries.device() &&
              argmax.device() == queries.device() &&
              dscore.device() == queries.device(),
              "all inputs must share a device");
  TORCH_CHECK(queries.dim() == 3, "queries must be [Nq, Lq, D]");
  TORCH_CHECK(documents.dim() == 2, "documents must be [total_d_tokens, D]");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(dscore.scalar_type() == at::ScalarType::Float,
              "dscore must be fp32");
  TORCH_CHECK(argmax.scalar_type() == at::ScalarType::Int,
              "argmax must be int32");

  queries    = queries.contiguous();
  documents  = documents.contiguous();
  document_offsets = document_offsets.contiguous().to(torch::kInt32);
  argmax     = argmax.contiguous();
  dscore     = dscore.contiguous();

  const int Nq  = static_cast<int>(queries.size(0));
  const int Lq  = static_cast<int>(queries.size(1));
  const int dim = static_cast<int>(queries.size(2));
  const int Nb  = static_cast<int>(document_offsets.size(0)) - 1;
  const int total_pairs = Nq * Nb;

  TORCH_CHECK(argmax.numel() == static_cast<int64_t>(total_pairs) * Lq,
              "argmax must have shape [Nq * Nb, Lq] (or equivalent)");
  TORCH_CHECK(dscore.numel() == total_pairs,
              "dscore must have shape [Nq, Nb] (or [Nq * Nb])");

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto dqueries   = torch::zeros_like(queries, fp32_opts);
  auto ddocuments = torch::zeros_like(documents, fp32_opts);
  if (total_pairs == 0) return std::make_tuple(dqueries, ddocuments);

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (queries.scalar_type()) {
    case at::ScalarType::Half:
      launch_contrastive_backward<__half>(
          queries, documents, document_offsets, argmax, dscore,
          dqueries, ddocuments, dim, Lq, Nb, total_pairs, stream);
      break;
    case at::ScalarType::BFloat16:
      launch_contrastive_backward<__nv_bfloat16>(
          queries, documents, document_offsets, argmax, dscore,
          dqueries, ddocuments, dim, Lq, Nb, total_pairs, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "contrastive backward currently supports fp16 / bf16 only");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(dqueries, ddocuments);
}

// =============================================================================
// Packed forward + per-q-tok argmax (training mode).
//
// Companion to `maxsim_packed_kernel`. Tracks (max, argmax) per q_tok with
// PyTorch's first-index-wins tiebreak. Output `argmax_out` has shape
// [num_pairs, max_q_len] int32 — slots beyond this pair's Lq are 0.
//
// Scalar argmax variant (matches the existing scalar packed forward). A
// WMMA argmax for packed is post-V2 work.
// =============================================================================

template <typename T>
__global__ void maxsim_packed_argmax_kernel(
    const T* __restrict__       queries,
    const int* __restrict__     query_offsets,
    const T* __restrict__       documents,
    const int* __restrict__     document_offsets,
    const int* __restrict__     pair_query_ids,
    const int* __restrict__     pair_document_ids,
    float* __restrict__         scores,
    int* __restrict__           argmax_out,
    int                         dim,
    int                         max_q_len) {
  const int pair_idx = blockIdx.x;
  const int tid      = threadIdx.x;
  const int lane     = tid & (kWarpSize - 1);
  const int warp_id  = tid >> 5;

  const int q_id    = pair_query_ids[pair_idx];
  const int d_id    = pair_document_ids[pair_idx];
  const int q_start = query_offsets[q_id];
  const int q_end   = query_offsets[q_id + 1];
  const int d_start = document_offsets[d_id];
  const int d_end   = document_offsets[d_id + 1];
  const int Lq      = q_end - q_start;
  const int Ld      = d_end - d_start;

  // Shared mem: maxes[Lq] floats, then argmax_buf[Lq] int32.
  extern __shared__ unsigned char smem_packed[];
  float* maxes      = reinterpret_cast<float*>(smem_packed);
  int*   argmax_buf = reinterpret_cast<int*>(maxes + Lq);

  for (int i = tid; i < Lq; i += kBlockThreads) {
    maxes[i]      = -CUDART_INF_F;
    argmax_buf[i] = 0;
  }
  __syncthreads();

  // Each warp owns a stride-`kNumWarps` slice of q_toks; no cross-warp
  // combine needed for this layout. Lane-local scan visits d_toks in
  // ascending order so strict `>` keeps the lowest index on ties.
  for (int q_tok = warp_id; q_tok < Lq; q_tok += kNumWarps) {
    const T* qrow = queries + static_cast<size_t>(q_start + q_tok) * dim;
    float my_max  = -CUDART_INF_F;
    int   my_arg  = 0;
    for (int d_tok = lane; d_tok < Ld; d_tok += kWarpSize) {
      const T* drow = documents + static_cast<size_t>(d_start + d_tok) * dim;
      float v = dot_row(qrow, drow, dim);
      if (v > my_max) { my_max = v; my_arg = d_tok; }
    }
    // Butterfly reduce: tie-break by smaller arg.
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      float pm = __shfl_xor_sync(0xffffffff, my_max, off);
      int   pa = __shfl_xor_sync(0xffffffff, my_arg, off);
      bool take = pm > my_max || (pm == my_max && pa < my_arg);
      if (take) { my_max = pm; my_arg = pa; }
    }
    if (lane == 0) {
      maxes[q_tok]      = my_max;
      argmax_buf[q_tok] = my_arg;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float s = 0.0f;
    for (int i = lane; i < Lq; i += kWarpSize) s += maxes[i];
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
    }
    if (lane == 0) scores[pair_idx] = s;
  }

  // Write argmax to global. Slots beyond Lq are 0.
  int* argmax_pair = argmax_out + static_cast<size_t>(pair_idx) * max_q_len;
  for (int i = tid; i < max_q_len; i += kBlockThreads) {
    argmax_pair[i] = (i < Lq) ? argmax_buf[i] : 0;
  }
}

template <typename T>
void launch_packed_argmax(const torch::Tensor& queries,
                          const torch::Tensor& query_offsets,
                          const torch::Tensor& documents,
                          const torch::Tensor& document_offsets,
                          const torch::Tensor& pair_query_ids,
                          const torch::Tensor& pair_document_ids,
                          torch::Tensor& scores,
                          torch::Tensor& argmax,
                          int dim, int max_q_len,
                          cudaStream_t stream) {
  const int num_pairs = static_cast<int>(pair_query_ids.size(0));
  const size_t smem =
      (size_t)max_q_len * sizeof(float) + (size_t)max_q_len * sizeof(int);
  maxsim_packed_argmax_kernel<T>
      <<<num_pairs, kBlockThreads, smem, stream>>>(
          reinterpret_cast<const T*>(queries.data_ptr()),
          query_offsets.data_ptr<int>(),
          reinterpret_cast<const T*>(documents.data_ptr()),
          document_offsets.data_ptr<int>(),
          pair_query_ids.data_ptr<int>(),
          pair_document_ids.data_ptr<int>(),
          scores.data_ptr<float>(),
          argmax.data_ptr<int>(),
          dim, max_q_len);
}

std::tuple<torch::Tensor, torch::Tensor> maxsim_packed_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor query_offsets,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor pair_query_ids,
    torch::Tensor pair_document_ids,
    int64_t max_q_len) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              query_offsets.device() == queries.device() &&
              document_offsets.device() == queries.device() &&
              pair_query_ids.device() == queries.device() &&
              pair_document_ids.device() == queries.device(),
              "all packed inputs must share a device");
  TORCH_CHECK(queries.dim() == 2 && documents.dim() == 2,
              "queries/documents must be 2-D");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim must match documents.dim");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(max_q_len > 0, "max_q_len must be > 0");

  queries           = queries.contiguous();
  documents         = documents.contiguous();
  query_offsets     = query_offsets.contiguous().to(torch::kInt32);
  document_offsets  = document_offsets.contiguous().to(torch::kInt32);
  pair_query_ids    = pair_query_ids.contiguous().to(torch::kInt32);
  pair_document_ids = pair_document_ids.contiguous().to(torch::kInt32);

  const int dim       = static_cast<int>(queries.size(1));
  const int num_pairs = static_cast<int>(pair_query_ids.size(0));

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(queries.device());
  auto scores = torch::empty({num_pairs}, fp32_opts);
  auto argmax = torch::zeros({num_pairs, max_q_len}, i32_opts);
  if (num_pairs == 0) return std::make_tuple(scores, argmax);

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (queries.scalar_type()) {
    case at::ScalarType::Float:
      launch_packed_argmax<float>(
          queries, query_offsets, documents, document_offsets,
          pair_query_ids, pair_document_ids, scores, argmax,
          dim, static_cast<int>(max_q_len), stream);
      break;
    case at::ScalarType::Half:
      launch_packed_argmax<__half>(
          queries, query_offsets, documents, document_offsets,
          pair_query_ids, pair_document_ids, scores, argmax,
          dim, static_cast<int>(max_q_len), stream);
      break;
    case at::ScalarType::BFloat16:
      launch_packed_argmax<__nv_bfloat16>(
          queries, query_offsets, documents, document_offsets,
          pair_query_ids, pair_document_ids, scores, argmax,
          dim, static_cast<int>(max_q_len), stream);
      break;
    default:
      TORCH_CHECK(false,
                  "packed forward+argmax dtype must be fp32 / fp16 / bf16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(scores, argmax);
}

// =============================================================================
// Packed backward. Given (dscore, argmax) from a forward pass plus the
// original (queries, documents, offsets, pair_ids), routes the incoming
// `dscore` gradient back to `dqueries` and `ddocuments` via atomic-add.
// Both grad tensors are fp32 regardless of input dtype.
//
// Atomic contention pattern:
//   dq[q_start + i]   shared by all pairs with this q_id (Nb-ish-fold)
//   dd[d_start + j*]  shared by all pairs with this d_id AND any q_tok of
//                     any pair that argmaxed there.
// =============================================================================

template <typename T>
__global__ void maxsim_packed_backward_kernel(
    const T* __restrict__       queries,
    const int* __restrict__     query_offsets,
    const T* __restrict__       documents,
    const int* __restrict__     document_offsets,
    const int* __restrict__     pair_query_ids,
    const int* __restrict__     pair_document_ids,
    const int* __restrict__     argmax,
    const float* __restrict__   dscore,
    float* __restrict__         dqueries,
    float* __restrict__         ddocuments,
    int                         dim,
    int                         max_q_len) {
  const int pair_idx = blockIdx.x;
  const int tid      = threadIdx.x;
  const int lane     = tid & (kWarpSize - 1);
  const int warp_id  = tid >> 5;
  const int warps_per_block = blockDim.x >> 5;

  const int q_id    = pair_query_ids[pair_idx];
  const int d_id    = pair_document_ids[pair_idx];
  const int q_start = query_offsets[q_id];
  const int q_end   = query_offsets[q_id + 1];
  const int d_start = document_offsets[d_id];
  const int d_end   = document_offsets[d_id + 1];
  const int Lq      = q_end - q_start;
  const int Ld      = d_end - d_start;

  const float g = dscore[pair_idx];

  for (int q_tok = warp_id; q_tok < Lq; q_tok += warps_per_block) {
    const int j = argmax[static_cast<size_t>(pair_idx) * max_q_len + q_tok];
    if (j < 0 || j >= Ld) continue;

    const T* q_row  = queries   + static_cast<size_t>(q_start + q_tok) * dim;
    const T* d_row  = documents + static_cast<size_t>(d_start + j) * dim;
    float*   dq_row = dqueries  + static_cast<size_t>(q_start + q_tok) * dim;
    float*   dd_row = ddocuments + static_cast<size_t>(d_start + j) * dim;

    for (int k = lane; k < dim; k += kWarpSize) {
      const float q_v = load_f(q_row + k);
      const float d_v = load_f(d_row + k);
      atomicAdd(&dq_row[k], g * d_v);
      atomicAdd(&dd_row[k], g * q_v);
    }
  }
}

template <typename T>
void launch_packed_backward(const torch::Tensor& queries,
                            const torch::Tensor& query_offsets,
                            const torch::Tensor& documents,
                            const torch::Tensor& document_offsets,
                            const torch::Tensor& pair_query_ids,
                            const torch::Tensor& pair_document_ids,
                            const torch::Tensor& argmax,
                            const torch::Tensor& dscore,
                            torch::Tensor& dqueries,
                            torch::Tensor& ddocuments,
                            int dim, int max_q_len, int num_pairs,
                            cudaStream_t stream) {
  constexpr int kBackwardWarps = 4;
  const int block_threads = kBackwardWarps * kWarpSize;
  maxsim_packed_backward_kernel<T>
      <<<num_pairs, block_threads, 0, stream>>>(
          reinterpret_cast<const T*>(queries.data_ptr()),
          query_offsets.data_ptr<int>(),
          reinterpret_cast<const T*>(documents.data_ptr()),
          document_offsets.data_ptr<int>(),
          pair_query_ids.data_ptr<int>(),
          pair_document_ids.data_ptr<int>(),
          argmax.data_ptr<int>(),
          dscore.data_ptr<float>(),
          dqueries.data_ptr<float>(),
          ddocuments.data_ptr<float>(),
          dim, max_q_len);
}

std::tuple<torch::Tensor, torch::Tensor> maxsim_packed_backward(
    torch::Tensor dscore,
    torch::Tensor queries,
    torch::Tensor query_offsets,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor pair_query_ids,
    torch::Tensor pair_document_ids,
    torch::Tensor argmax,
    int64_t max_q_len) {
  TORCH_CHECK(queries.device().is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              dscore.device() == queries.device() &&
              argmax.device() == queries.device() &&
              query_offsets.device() == queries.device() &&
              document_offsets.device() == queries.device() &&
              pair_query_ids.device() == queries.device() &&
              pair_document_ids.device() == queries.device(),
              "all backward inputs must share a device");
  TORCH_CHECK(queries.dim() == 2 && documents.dim() == 2,
              "queries/documents must be 2-D");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim must match documents.dim");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(dscore.scalar_type() == at::ScalarType::Float,
              "dscore must be fp32");
  TORCH_CHECK(argmax.scalar_type() == at::ScalarType::Int,
              "argmax must be int32");
  TORCH_CHECK(max_q_len > 0, "max_q_len must be > 0");

  queries           = queries.contiguous();
  documents         = documents.contiguous();
  query_offsets     = query_offsets.contiguous().to(torch::kInt32);
  document_offsets  = document_offsets.contiguous().to(torch::kInt32);
  pair_query_ids    = pair_query_ids.contiguous().to(torch::kInt32);
  pair_document_ids = pair_document_ids.contiguous().to(torch::kInt32);
  argmax            = argmax.contiguous();
  dscore            = dscore.contiguous();

  const int dim       = static_cast<int>(queries.size(1));
  const int num_pairs = static_cast<int>(pair_query_ids.size(0));

  TORCH_CHECK(argmax.numel() == (int64_t)num_pairs * max_q_len,
              "argmax must have shape [num_pairs, max_q_len]");
  TORCH_CHECK(dscore.numel() == num_pairs,
              "dscore must have shape [num_pairs]");

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto dqueries   = torch::zeros_like(queries, fp32_opts);
  auto ddocuments = torch::zeros_like(documents, fp32_opts);
  if (num_pairs == 0) return std::make_tuple(dqueries, ddocuments);

  const c10::cuda::CUDAGuard device_guard(queries.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (queries.scalar_type()) {
    case at::ScalarType::Float:
      launch_packed_backward<float>(
          queries, query_offsets, documents, document_offsets,
          pair_query_ids, pair_document_ids, argmax, dscore,
          dqueries, ddocuments, dim, static_cast<int>(max_q_len),
          num_pairs, stream);
      break;
    case at::ScalarType::Half:
      launch_packed_backward<__half>(
          queries, query_offsets, documents, document_offsets,
          pair_query_ids, pair_document_ids, argmax, dscore,
          dqueries, ddocuments, dim, static_cast<int>(max_q_len),
          num_pairs, stream);
      break;
    case at::ScalarType::BFloat16:
      launch_packed_backward<__nv_bfloat16>(
          queries, query_offsets, documents, document_offsets,
          pair_query_ids, pair_document_ids, argmax, dscore,
          dqueries, ddocuments, dim, static_cast<int>(max_q_len),
          num_pairs, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "packed backward dtype must be fp32 / fp16 / bf16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(dqueries, ddocuments);
}
