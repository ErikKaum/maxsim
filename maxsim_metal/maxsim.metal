// Exact, memory-efficient MaxSim Metal kernel for late-interaction
// retrieval / reranking.
//
//   score(q, d) = sum_i  max_j  dot(q_i, d_j)
//
// Grid layout (one threadgroup per (pair, query_token)):
//   threadgroups.x = num_pairs
//   threadgroups.y = max_q_len   (over all query segments)
//   threads        = 64          (power of two, used for tg reduction)
//
// Threads in a threadgroup cooperatively scan the document tokens of the
// pair's document segment. Each thread keeps a running max of dot(q_i, d_j)
// over its slice of doc tokens; a tree reduction in threadgroup memory
// produces the per-query-token max. The result is atomic-added into the
// per-pair score so we never materialize the [Lq, Ld] similarity matrix.
//
// fp32 accumulation regardless of input dtype; output is fp32.

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

template <typename T>
inline void maxsim_kernel_impl(
    device const T*       queries,
    device const int*     query_offsets,
    device const T*       documents,
    device const int*     document_offsets,
    device const int*     pair_query_ids,
    device const int*     pair_document_ids,
    device atomic<float>* scores,
    constant uint&        dim,
    threadgroup float*    shared,
    uint                  pair_idx,
    uint                  q_tok,
    uint                  tid,
    uint                  tg_size)
{
    int q_id = pair_query_ids[pair_idx];
    int d_id = pair_document_ids[pair_idx];

    int q_start = query_offsets[q_id];
    int q_end   = query_offsets[q_id + 1];
    int Lq      = q_end - q_start;

    // Out of range query token for this pair: this threadgroup has no work.
    if ((int)q_tok >= Lq) {
        return;
    }

    int d_start = document_offsets[d_id];
    int d_end   = document_offsets[d_id + 1];
    int Ld      = d_end - d_start;

    device const T* qptr = queries + (size_t)(q_start + (int)q_tok) * (size_t)dim;

    float my_max = -INFINITY;
    for (int j = (int)tid; j < Ld; j += (int)tg_size) {
        device const T* dptr = documents + (size_t)(d_start + j) * (size_t)dim;
        float acc = 0.0f;
        for (uint k = 0; k < dim; ++k) {
            acc += (float)qptr[k] * (float)dptr[k];
        }
        my_max = max(my_max, acc);
    }

    shared[tid] = my_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in threadgroup memory (tg_size assumed power of two).
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        atomic_fetch_add_explicit(&scores[pair_idx], shared[0],
                                  memory_order_relaxed);
    }
}

// Metal 4 requires all attribute-decorated stage-input parameters in a kernel
// to have the same shape (all scalar or all uintN), so we take everything as
// uint3 and use only the .x components for the 1-D threadgroup.
#define MAXSIM_KERNEL_ENTRY(NAME, T)                                         \
    kernel void NAME(                                                        \
        device const T*       queries           [[buffer(0)]],               \
        device const int*     query_offsets     [[buffer(1)]],               \
        device const T*       documents         [[buffer(2)]],               \
        device const int*     document_offsets  [[buffer(3)]],               \
        device const int*     pair_query_ids    [[buffer(4)]],               \
        device const int*     pair_document_ids [[buffer(5)]],               \
        device atomic<float>* scores            [[buffer(6)]],               \
        constant uint&        dim               [[buffer(7)]],               \
        threadgroup float*    shared            [[threadgroup(0)]],          \
        uint3                 tgid              [[threadgroup_position_in_grid]], \
        uint3                 tid               [[thread_position_in_threadgroup]], \
        uint3                 tg_size           [[threads_per_threadgroup]]) \
    {                                                                        \
        maxsim_kernel_impl<T>(queries, query_offsets, documents,             \
                              document_offsets, pair_query_ids,              \
                              pair_document_ids, scores, dim, shared,        \
                              tgid.x, tgid.y, tid.x, tg_size.x);             \
    }

MAXSIM_KERNEL_ENTRY(maxsim_kernel_fp32, float)
MAXSIM_KERNEL_ENTRY(maxsim_kernel_fp16, half)

#if __METAL_VERSION__ >= 310
MAXSIM_KERNEL_ENTRY(maxsim_kernel_bf16, bfloat)
#endif

#define MMA8 8u

// 8×8 token block: for each K-tile of 8, multiply simdgroup 8×8 slices so
// Acc += Q[qb:qb+8, k:k+8] × D[db:db+8, k:k+8]ᵀ (fp32 accum). Requires dim % 8 == 0.
inline void maxsim_mma_tile_8x8_float(
    threadgroup const float* Q_base,
    threadgroup const float* D_base,
    uint                       dim,
    threadgroup float*         mma_store,
    uint                       simd_id)
{
    simdgroup_float8x8 Acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_float8x8 A_frag;
        simdgroup_float8x8 B_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_load(B_frag, D_base + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc, A_frag, B_frag, Acc);
    }
    simdgroup_store(Acc, mma_store + simd_id * 256, 8, ulong2(0, 0), false);
}

inline void maxsim_mma_tile_8x8_half(
    threadgroup const half* Q_base,
    threadgroup const half* D_base,
    uint                    dim,
    threadgroup float*      mma_store,
    uint                    simd_id)
{
    simdgroup_float8x8 Acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_half8x8 A_frag;
        simdgroup_half8x8 B_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_load(B_frag, D_base + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc, A_frag, B_frag, Acc);
    }
    simdgroup_store(Acc, mma_store + simd_id * 256, 8, ulong2(0, 0), false);
}

#if __METAL_VERSION__ >= 310
inline void maxsim_mma_tile_8x8_bfloat(
    threadgroup const bfloat* Q_base,
    threadgroup const bfloat* D_base,
    uint                      dim,
    threadgroup float*        mma_store,
    uint                      simd_id)
{
    simdgroup_float8x8 Acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_bfloat8x8 A_frag;
        simdgroup_bfloat8x8 B_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_load(B_frag, D_base + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc, A_frag, B_frag, Acc);
    }
    simdgroup_store(Acc, mma_store + simd_id * 256, 8, ulong2(0, 0), false);
}
#endif

inline void maxsim_mma_dispatch_tile(
    threadgroup const float* Q_base,
    threadgroup const float* D_base,
    uint                     dim,
    threadgroup float*       mma_store,
    uint                     simd_id)
{
    maxsim_mma_tile_8x8_float(Q_base, D_base, dim, mma_store, simd_id);
}

inline void maxsim_mma_dispatch_tile(
    threadgroup const half* Q_base,
    threadgroup const half* D_base,
    uint                    dim,
    threadgroup float*      mma_store,
    uint                    simd_id)
{
    maxsim_mma_tile_8x8_half(Q_base, D_base, dim, mma_store, simd_id);
}

#if __METAL_VERSION__ >= 310
inline void maxsim_mma_dispatch_tile(
    threadgroup const bfloat* Q_base,
    threadgroup const bfloat* D_base,
    uint                      dim,
    threadgroup float*        mma_store,
    uint                      simd_id)
{
    maxsim_mma_tile_8x8_bfloat(Q_base, D_base, dim, mma_store, simd_id);
}
#endif

// =============================================================================
// Device-source variants: B comes from `device const T*` (skipping the
// threadgroup-memory staging of D). Q (A) still lives in threadgroup memory
// because it's reused across many d-tiles per simdgroup.
// =============================================================================

inline void maxsim_mma_tile_8x8_float_dev(
    threadgroup const float* Q_base,
    device const float*      D_base,
    uint                     dim,
    threadgroup float*       mma_store,
    uint                     simd_id)
{
    simdgroup_float8x8 Acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_float8x8 A_frag;
        simdgroup_float8x8 B_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_load(B_frag, D_base + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc, A_frag, B_frag, Acc);
    }
    simdgroup_store(Acc, mma_store + simd_id * 256, 8, ulong2(0, 0), false);
}

inline void maxsim_mma_tile_8x8_half_dev(
    threadgroup const half* Q_base,
    device const half*      D_base,
    uint                    dim,
    threadgroup float*      mma_store,
    uint                    simd_id)
{
    simdgroup_float8x8 Acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_half8x8 A_frag;
        simdgroup_half8x8 B_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_load(B_frag, D_base + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc, A_frag, B_frag, Acc);
    }
    simdgroup_store(Acc, mma_store + simd_id * 256, 8, ulong2(0, 0), false);
}

#if __METAL_VERSION__ >= 310
inline void maxsim_mma_tile_8x8_bfloat_dev(
    threadgroup const bfloat* Q_base,
    device const bfloat*      D_base,
    uint                      dim,
    threadgroup float*        mma_store,
    uint                      simd_id)
{
    simdgroup_float8x8 Acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_bfloat8x8 A_frag;
        simdgroup_bfloat8x8 B_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_load(B_frag, D_base + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc, A_frag, B_frag, Acc);
    }
    simdgroup_store(Acc, mma_store + simd_id * 256, 8, ulong2(0, 0), false);
}
#endif

inline void maxsim_mma_dispatch_tile_dev(
    threadgroup const float* Q_base,
    device const float*      D_base,
    uint                     dim,
    threadgroup float*       mma_store,
    uint                     simd_id)
{
    maxsim_mma_tile_8x8_float_dev(Q_base, D_base, dim, mma_store, simd_id);
}

inline void maxsim_mma_dispatch_tile_dev(
    threadgroup const half* Q_base,
    device const half*      D_base,
    uint                    dim,
    threadgroup float*      mma_store,
    uint                    simd_id)
{
    maxsim_mma_tile_8x8_half_dev(Q_base, D_base, dim, mma_store, simd_id);
}

#if __METAL_VERSION__ >= 310
inline void maxsim_mma_dispatch_tile_dev(
    threadgroup const bfloat* Q_base,
    device const bfloat*      D_base,
    uint                      dim,
    threadgroup float*        mma_store,
    uint                      simd_id)
{
    maxsim_mma_tile_8x8_bfloat_dev(Q_base, D_base, dim, mma_store, simd_id);
}
#endif

// =============================================================================
// 2x variant: compute Acc0 += Q @ D0ᵀ AND Acc1 += Q @ D1ᵀ in lockstep,
// sharing the A-fragment load across the two MMAs. Halves the A-loads and
// the per-iteration barrier count (one row-max over the combined 8×16
// scratch). Writes both 8×8 results into a single per-simdgroup slot:
//   mma_store[simd_id * 256 + 0..63]   <- Acc0 (8×8 row-major)
//   mma_store[simd_id * 256 + 64..127] <- Acc1 (8×8 row-major)
// =============================================================================

inline void maxsim_mma_tile_8x8_float_dev_2x(
    threadgroup const float* Q_base,
    device const float*      D_base0,
    device const float*      D_base1,
    uint                     dim,
    threadgroup float*       mma_store,
    uint                     simd_id)
{
    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_float8x8 A_frag;
        simdgroup_float8x8 B0_frag;
        simdgroup_float8x8 B1_frag;
        simdgroup_load(A_frag,  Q_base  + k, dim, ulong2(0, 0), false);
        simdgroup_load(B0_frag, D_base0 + k, dim, ulong2(0, 0), true);
        simdgroup_load(B1_frag, D_base1 + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc0, A_frag, B0_frag, Acc0);
        simdgroup_multiply_accumulate(Acc1, A_frag, B1_frag, Acc1);
    }
    simdgroup_store(Acc0, mma_store + simd_id * 256 + 0,  8, ulong2(0, 0), false);
    simdgroup_store(Acc1, mma_store + simd_id * 256 + 64, 8, ulong2(0, 0), false);
}

inline void maxsim_mma_tile_8x8_half_dev_2x(
    threadgroup const half* Q_base,
    device const half*      D_base0,
    device const half*      D_base1,
    uint                    dim,
    threadgroup float*      mma_store,
    uint                    simd_id)
{
    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_half8x8 A_frag;
        simdgroup_half8x8 B0_frag;
        simdgroup_half8x8 B1_frag;
        simdgroup_load(A_frag,  Q_base  + k, dim, ulong2(0, 0), false);
        simdgroup_load(B0_frag, D_base0 + k, dim, ulong2(0, 0), true);
        simdgroup_load(B1_frag, D_base1 + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc0, A_frag, B0_frag, Acc0);
        simdgroup_multiply_accumulate(Acc1, A_frag, B1_frag, Acc1);
    }
    simdgroup_store(Acc0, mma_store + simd_id * 256 + 0,  8, ulong2(0, 0), false);
    simdgroup_store(Acc1, mma_store + simd_id * 256 + 64, 8, ulong2(0, 0), false);
}

#if __METAL_VERSION__ >= 310
inline void maxsim_mma_tile_8x8_bfloat_dev_2x(
    threadgroup const bfloat* Q_base,
    device const bfloat*      D_base0,
    device const bfloat*      D_base1,
    uint                      dim,
    threadgroup float*        mma_store,
    uint                      simd_id)
{
    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_bfloat8x8 A_frag;
        simdgroup_bfloat8x8 B0_frag;
        simdgroup_bfloat8x8 B1_frag;
        simdgroup_load(A_frag,  Q_base  + k, dim, ulong2(0, 0), false);
        simdgroup_load(B0_frag, D_base0 + k, dim, ulong2(0, 0), true);
        simdgroup_load(B1_frag, D_base1 + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc0, A_frag, B0_frag, Acc0);
        simdgroup_multiply_accumulate(Acc1, A_frag, B1_frag, Acc1);
    }
    simdgroup_store(Acc0, mma_store + simd_id * 256 + 0,  8, ulong2(0, 0), false);
    simdgroup_store(Acc1, mma_store + simd_id * 256 + 64, 8, ulong2(0, 0), false);
}
#endif

inline void maxsim_mma_dispatch_tile_dev_2x(
    threadgroup const float* Q_base,
    device const float*      D_base0,
    device const float*      D_base1,
    uint                     dim,
    threadgroup float*       mma_store,
    uint                     simd_id)
{
    maxsim_mma_tile_8x8_float_dev_2x(
        Q_base, D_base0, D_base1, dim, mma_store, simd_id);
}

inline void maxsim_mma_dispatch_tile_dev_2x(
    threadgroup const half* Q_base,
    device const half*      D_base0,
    device const half*      D_base1,
    uint                    dim,
    threadgroup float*      mma_store,
    uint                    simd_id)
{
    maxsim_mma_tile_8x8_half_dev_2x(
        Q_base, D_base0, D_base1, dim, mma_store, simd_id);
}

#if __METAL_VERSION__ >= 310
inline void maxsim_mma_dispatch_tile_dev_2x(
    threadgroup const bfloat* Q_base,
    device const bfloat*      D_base0,
    device const bfloat*      D_base1,
    uint                      dim,
    threadgroup float*        mma_store,
    uint                      simd_id)
{
    maxsim_mma_tile_8x8_bfloat_dev_2x(
        Q_base, D_base0, D_base1, dim, mma_store, simd_id);
}
#endif

// =============================================================================
// 4x variant: one A-fragment shared across four MMAs (Acc0..Acc3) against
// four consecutive 8-row slices of D. Quarter the A-loads and barrier-pair
// count vs 1x; half vs 2x. Writes four 8×8 results into a single
// per-simdgroup 256-float slot in mma_store at offsets {0, 64, 128, 192}.
// =============================================================================

inline void maxsim_mma_tile_8x8_float_dev_4x(
    threadgroup const float* Q_base,
    device const float*      D_base,   // D[db..db+32, :], we slice internally
    uint                     dim,
    threadgroup float*       mma_store,
    uint                     simd_id)
{
    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_float8x8 A_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_float8x8 B0, B1, B2, B3;
        simdgroup_load(B0, D_base + (size_t)0  * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B1, D_base + (size_t)8  * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B2, D_base + (size_t)16 * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B3, D_base + (size_t)24 * dim + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc0, A_frag, B0, Acc0);
        simdgroup_multiply_accumulate(Acc1, A_frag, B1, Acc1);
        simdgroup_multiply_accumulate(Acc2, A_frag, B2, Acc2);
        simdgroup_multiply_accumulate(Acc3, A_frag, B3, Acc3);
    }
    threadgroup float* slot = mma_store + simd_id * 256;
    simdgroup_store(Acc0, slot + 0,   8, ulong2(0, 0), false);
    simdgroup_store(Acc1, slot + 64,  8, ulong2(0, 0), false);
    simdgroup_store(Acc2, slot + 128, 8, ulong2(0, 0), false);
    simdgroup_store(Acc3, slot + 192, 8, ulong2(0, 0), false);
}

inline void maxsim_mma_tile_8x8_half_dev_4x(
    threadgroup const half* Q_base,
    device const half*      D_base,
    uint                    dim,
    threadgroup float*      mma_store,
    uint                    simd_id)
{
    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_half8x8 A_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_half8x8 B0, B1, B2, B3;
        simdgroup_load(B0, D_base + (size_t)0  * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B1, D_base + (size_t)8  * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B2, D_base + (size_t)16 * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B3, D_base + (size_t)24 * dim + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc0, A_frag, B0, Acc0);
        simdgroup_multiply_accumulate(Acc1, A_frag, B1, Acc1);
        simdgroup_multiply_accumulate(Acc2, A_frag, B2, Acc2);
        simdgroup_multiply_accumulate(Acc3, A_frag, B3, Acc3);
    }
    threadgroup float* slot = mma_store + simd_id * 256;
    simdgroup_store(Acc0, slot + 0,   8, ulong2(0, 0), false);
    simdgroup_store(Acc1, slot + 64,  8, ulong2(0, 0), false);
    simdgroup_store(Acc2, slot + 128, 8, ulong2(0, 0), false);
    simdgroup_store(Acc3, slot + 192, 8, ulong2(0, 0), false);
}

#if __METAL_VERSION__ >= 310
inline void maxsim_mma_tile_8x8_bfloat_dev_4x(
    threadgroup const bfloat* Q_base,
    device const bfloat*      D_base,
    uint                      dim,
    threadgroup float*        mma_store,
    uint                      simd_id)
{
    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < dim; k += MMA8) {
        simdgroup_bfloat8x8 A_frag;
        simdgroup_load(A_frag, Q_base + k, dim, ulong2(0, 0), false);
        simdgroup_bfloat8x8 B0, B1, B2, B3;
        simdgroup_load(B0, D_base + (size_t)0  * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B1, D_base + (size_t)8  * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B2, D_base + (size_t)16 * dim + k, dim, ulong2(0, 0), true);
        simdgroup_load(B3, D_base + (size_t)24 * dim + k, dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Acc0, A_frag, B0, Acc0);
        simdgroup_multiply_accumulate(Acc1, A_frag, B1, Acc1);
        simdgroup_multiply_accumulate(Acc2, A_frag, B2, Acc2);
        simdgroup_multiply_accumulate(Acc3, A_frag, B3, Acc3);
    }
    threadgroup float* slot = mma_store + simd_id * 256;
    simdgroup_store(Acc0, slot + 0,   8, ulong2(0, 0), false);
    simdgroup_store(Acc1, slot + 64,  8, ulong2(0, 0), false);
    simdgroup_store(Acc2, slot + 128, 8, ulong2(0, 0), false);
    simdgroup_store(Acc3, slot + 192, 8, ulong2(0, 0), false);
}
#endif

inline void maxsim_mma_dispatch_tile_dev_4x(
    threadgroup const float* Q_base,
    device const float*      D_base,
    uint                     dim,
    threadgroup float*       mma_store,
    uint                     simd_id)
{
    maxsim_mma_tile_8x8_float_dev_4x(Q_base, D_base, dim, mma_store, simd_id);
}

inline void maxsim_mma_dispatch_tile_dev_4x(
    threadgroup const half* Q_base,
    device const half*      D_base,
    uint                    dim,
    threadgroup float*      mma_store,
    uint                    simd_id)
{
    maxsim_mma_tile_8x8_half_dev_4x(Q_base, D_base, dim, mma_store, simd_id);
}

#if __METAL_VERSION__ >= 310
inline void maxsim_mma_dispatch_tile_dev_4x(
    threadgroup const bfloat* Q_base,
    device const bfloat*      D_base,
    uint                      dim,
    threadgroup float*        mma_store,
    uint                      simd_id)
{
    maxsim_mma_tile_8x8_bfloat_dev_4x(Q_base, D_base, dim, mma_store, simd_id);
}
#endif

// =============================================================================
// Fast kernel: one threadgroup per *pair*.
//
//   Q[q_id]            staged once in threadgroup memory (Lq * dim values)
//   D[d_id]            iterated in tiles of Td rows, staged in threadgroup mem
//   maxes[Lq]          running per-q_tok max in threadgroup memory
//
// Each threadgroup has SG_COUNT simdgroups; each simdgroup owns a disjoint
// slice of q_toks (no atomics). For each (q_tok, d_tok) pair the lanes of one
// simdgroup compute dot(q_row, d_row) cooperatively using vec<T,4> loads and
// simd_sum, then lane 0 updates maxes[q_tok]. After all d-tiles are processed
// the first simdgroup sums maxes[Lq] and writes scores[pair_idx] (single
// scalar store -- no atomics).
//
// Threadgroup memory layout (host-managed, sized for max_q_len over batch):
//   [ maxes  : max_q_len * float ]
//   [ Q tile : max_q_len * dim   * sizeof(T) ]
//   [ D tile : Td        * dim   * sizeof(T) ]
// =============================================================================

template <typename T>
inline void maxsim_pair_kernel_impl(
    device const T*        queries,
    device const int*      query_offsets,
    device const T*        documents,
    device const int*      document_offsets,
    device const int*      pair_query_ids,
    device const int*      pair_document_ids,
    device float*          scores,
    constant uint&         dim,
    constant uint&         max_q_len,
    threadgroup uchar*     shared_raw,
    uint                   pair_idx,
    uint                   tid,
    uint                   tg_size)
{
    constexpr uint SIMD_W   = 32u;   // simdgroup width on Apple GPUs

    const uint simd_lane = tid & (SIMD_W - 1u);
    const uint simd_id   = tid / SIMD_W;
    const uint num_simds = tg_size / SIMD_W;

    int q_id = pair_query_ids[pair_idx];
    int d_id = pair_document_ids[pair_idx];
    int q_start = query_offsets[q_id];
    int q_end   = query_offsets[q_id + 1];
    int d_start = document_offsets[d_id];
    int d_end   = document_offsets[d_id + 1];
    uint Lq = (uint)(q_end - q_start);
    uint Ld = (uint)(d_end - d_start);

    // Threadgroup memory layout. `maxes` is padded to a multiple of 4 floats
    // (16 bytes) so `Q_shared` starts at a 16-byte boundary -- a requirement
    // for `simdgroup_load` of half/bfloat 8x8 fragments. D is *not* staged
    // anymore: we `simdgroup_load` it directly from device memory in the MMA
    // loop, freeing the previous Td*dim*sizeof(T) byte slot for other use.
    const uint maxes_padded = (max_q_len + 3u) & ~3u;
    threadgroup float* maxes    = (threadgroup float*)shared_raw;
    threadgroup T*     Q_shared = (threadgroup T*)(maxes + maxes_padded);
    threadgroup float* mma_store =
        (threadgroup float*)((threadgroup uchar*)Q_shared +
                             (size_t)max_q_len * dim * sizeof(T));

    // Init maxes
    for (uint i = tid; i < Lq; i += tg_size) {
        maxes[i] = -INFINITY;
    }

    // Cooperatively load Q[q_id, 0..Lq) into Q_shared.
    {
        device const T* qsrc = queries + (size_t)q_start * dim;
        uint q_count = Lq * dim;
        for (uint i = tid; i < q_count; i += tg_size) {
            Q_shared[i] = qsrc[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Distribute q_toks across simdgroups (disjoint slices, no races).
    uint qs_per_sg = (Lq + num_simds - 1u) / num_simds;
    uint q_lo = simd_id * qs_per_sg;
    uint q_hi = min(q_lo + qs_per_sg, Lq);

    const bool use_mma = ((dim & 7u) == 0u);
    uint       dim4    = dim >> 2u;
    uint       tail0   = dim4 << 2u;

    device const T* D_base = documents + (size_t)d_start * dim;
    const uint db_end       = (Ld / 8u) * 8u;          // last db where any MMA fits
    const uint db_quad_end  = (db_end / 32u) * 32u;    // last db where 4-tile fits
    const uint db_pair_end  =
        db_quad_end + ((db_end - db_quad_end) / 16u) * 16u;  // ... then 2-tile

    uint q_cursor = q_lo;
    while (q_cursor < q_hi) {
        if (use_mma && q_cursor + 8u <= q_hi) {
            uint qb = q_cursor;
            threadgroup const T* Q_qb = Q_shared + (size_t)qb * dim;

            // 4-d-tile batched MMA: load A once per K-tile, accumulate into
            // four 8×8 fragments against D[db..db+8], D[db+8..db+16],
            // D[db+16..db+24], D[db+24..db+32]. Combined row-max over the
            // 8×32 scratch with the same 32-lane butterfly.
            for (uint db = 0; db < db_quad_end; db += 32u) {
                maxsim_mma_dispatch_tile_dev_4x(
                    Q_qb,
                    D_base + (size_t)db * dim,
                    dim,
                    mma_store,
                    simd_id);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                {
                    const uint row  = simd_lane >> 2;
                    const uint col0 = (simd_lane & 3u) << 1;
                    threadgroup const float* base =
                        mma_store + simd_id * 256 + row * 8;
                    float lane_max =
                        max(max(base[0   + col0], base[0   + col0 + 1]),
                            max(base[64  + col0], base[64  + col0 + 1]));
                    lane_max = max(lane_max,
                        max(max(base[128 + col0], base[128 + col0 + 1]),
                            max(base[192 + col0], base[192 + col0 + 1])));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 1u));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 2u));
                    if ((simd_lane & 3u) == 0u) {
                        maxes[qb + row] = max(maxes[qb + row], lane_max);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            // Then 2-tile if Ld/8 has 2 leftover.
            for (uint db = db_quad_end; db < db_pair_end; db += 16u) {
                maxsim_mma_dispatch_tile_dev_2x(
                    Q_qb,
                    D_base + (size_t)db * dim,
                    D_base + (size_t)(db + 8u) * dim,
                    dim,
                    mma_store,
                    simd_id);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                {
                    const uint row  = simd_lane >> 2;
                    const uint col0 = (simd_lane & 3u) << 1;
                    threadgroup const float* base =
                        mma_store + simd_id * 256 + row * 8;
                    float lane_max =
                        max(max(base[col0],      base[col0 + 1]),
                            max(base[64 + col0], base[64 + col0 + 1]));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 1u));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 2u));
                    if ((simd_lane & 3u) == 0u) {
                        maxes[qb + row] = max(maxes[qb + row], lane_max);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            // Then 1-tile for any remaining 8 d_toks.
            if (db_pair_end < db_end) {
                uint db = db_pair_end;
                maxsim_mma_dispatch_tile_dev(
                    Q_qb,
                    D_base + (size_t)db * dim,
                    dim,
                    mma_store,
                    simd_id);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                {
                    const uint row  = simd_lane >> 2;
                    const uint col0 = (simd_lane & 3u) << 1;
                    threadgroup const float* row_ptr =
                        mma_store + simd_id * 256 + row * 8;
                    float lane_max =
                        max(row_ptr[col0], row_ptr[col0 + 1]);
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 1u));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 2u));
                    if ((simd_lane & 3u) == 0u) {
                        maxes[qb + row] = max(maxes[qb + row], lane_max);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            // Scalar tail when Ld isn't a multiple of 8.
            for (uint ii = 0; ii < 8; ii++) {
                uint q_tok = qb + ii;
                threadgroup const T*        qrow  = Q_shared + (size_t)q_tok * dim;
                threadgroup const vec<T,4>* qrow4 =
                    (threadgroup const vec<T,4>*)qrow;
                float my_max = maxes[q_tok];
                for (uint d_tok = db_end; d_tok < Ld; ++d_tok) {
                    device const T*        drow  = D_base + (size_t)d_tok * dim;
                    device const vec<T,4>* drow4 =
                        (device const vec<T,4>*)drow;
                    float acc = 0.0f;
                    for (uint k = simd_lane; k < dim4; k += SIMD_W) {
                        vec<T,4> qv = qrow4[k];
                        vec<T,4> dv = drow4[k];
                        acc += float(qv.x) * float(dv.x)
                             + float(qv.y) * float(dv.y)
                             + float(qv.z) * float(dv.z)
                             + float(qv.w) * float(dv.w);
                    }
                    for (uint k = tail0 + simd_lane; k < dim; k += SIMD_W) {
                        acc += (float)qrow[k] * (float)drow[k];
                    }
                    acc = simd_sum(acc);
                    my_max = max(my_max, acc);
                }
                if (simd_lane == 0) {
                    maxes[q_tok] = my_max;
                }
            }
            q_cursor += 8u;
        } else {
            uint q_tok = q_cursor;
            threadgroup const T*        qrow  = Q_shared + (size_t)q_tok * dim;
            threadgroup const vec<T,4>* qrow4 =
                (threadgroup const vec<T,4>*)qrow;
            float my_max = maxes[q_tok];

            for (uint d_tok = 0; d_tok < Ld; ++d_tok) {
                device const T*        drow  = D_base + (size_t)d_tok * dim;
                device const vec<T,4>* drow4 =
                    (device const vec<T,4>*)drow;
                float acc = 0.0f;
                for (uint k = simd_lane; k < dim4; k += SIMD_W) {
                    vec<T,4> qv = qrow4[k];
                    vec<T,4> dv = drow4[k];
                    acc += float(qv.x) * float(dv.x)
                         + float(qv.y) * float(dv.y)
                         + float(qv.z) * float(dv.z)
                         + float(qv.w) * float(dv.w);
                }
                for (uint k = tail0 + simd_lane; k < dim; k += SIMD_W) {
                    acc += (float)qrow[k] * (float)drow[k];
                }
                acc = simd_sum(acc);
                my_max = max(my_max, acc);
            }
            if (simd_lane == 0) {
                maxes[q_tok] = my_max;
            }
            q_cursor += 1u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sum maxes -> scores[pair_idx]. Use simdgroup 0.
    if (simd_id == 0u) {
        float s = 0.0f;
        for (uint i = simd_lane; i < Lq; i += SIMD_W) {
            s += maxes[i];
        }
        s = simd_sum(s);
        if (simd_lane == 0u) {
            scores[pair_idx] = s;
        }
    }
}

// Stage-input attributes: same `uint3` shape rule as the naive kernel.
// simdgroup / lane indices are derived from `tid.x` to keep the input
// declarations homogeneous (avoids Metal 4 strict-mode mixing errors).
#define MAXSIM_PAIR_KERNEL_ENTRY(NAME, T)                                    \
    kernel void NAME(                                                        \
        device const T*       queries           [[buffer(0)]],               \
        device const int*     query_offsets     [[buffer(1)]],               \
        device const T*       documents         [[buffer(2)]],               \
        device const int*     document_offsets  [[buffer(3)]],               \
        device const int*     pair_query_ids    [[buffer(4)]],               \
        device const int*     pair_document_ids [[buffer(5)]],               \
        device float*         scores            [[buffer(6)]],               \
        constant uint&        dim               [[buffer(7)]],               \
        constant uint&        max_q_len         [[buffer(8)]],               \
        threadgroup uchar*    shared            [[threadgroup(0)]],          \
        uint3                 tgid              [[threadgroup_position_in_grid]], \
        uint3                 tid               [[thread_position_in_threadgroup]], \
        uint3                 tg_size           [[threads_per_threadgroup]]) \
    {                                                                        \
        maxsim_pair_kernel_impl<T>(queries, query_offsets, documents,        \
                                   document_offsets, pair_query_ids,         \
                                   pair_document_ids, scores, dim,           \
                                   max_q_len, shared, tgid.x, tid.x,         \
                                   tg_size.x);                                \
    }

MAXSIM_PAIR_KERNEL_ENTRY(maxsim_kernel_pair_fp32, float)
MAXSIM_PAIR_KERNEL_ENTRY(maxsim_kernel_pair_fp16, half)

#if __METAL_VERSION__ >= 310
MAXSIM_PAIR_KERNEL_ENTRY(maxsim_kernel_pair_bf16, bfloat)
#endif

// =============================================================================
// Padded variant: same per-pair design, but reads from row-padded queries
// and documents (no offsets / no pair-id buffers). Pair ordering is the
// canonical row-major (b, c) -> q_id = b, d_id = b * C + c, so the host can
// pass a single linear pair_idx in [0, B*C) and the kernel recovers q_id /
// d_id on its own.
// =============================================================================

template <typename T>
inline void maxsim_padded_kernel_impl(
    device const T*       queries,             // [num_q * Lq_max, dim]
    device const int*     query_lengths,       // [num_q]
    device const T*       documents,           // [num_pairs * Ld_max, dim]
    device const int*     doc_lengths,         // [num_pairs]
    device float*         scores,
    constant uint&        dim,
    constant uint&        Lq_max,
    constant uint&        Ld_max,
    constant uint&        num_candidates,
    threadgroup uchar*    shared_raw,
    uint                  pair_idx,
    uint                  tid,
    uint                  tg_size)
{
    constexpr uint SIMD_W   = 32u;

    const uint simd_lane = tid & (SIMD_W - 1u);
    const uint simd_id   = tid / SIMD_W;
    const uint num_simds = tg_size / SIMD_W;

    // Canonical pairing.
    const uint q_id    = pair_idx / num_candidates;
    const uint d_id    = pair_idx;
    const uint q_start = q_id * Lq_max;
    const uint d_start = d_id * Ld_max;
    const uint Lq      = (uint)query_lengths[q_id];
    const uint Ld      = (uint)doc_lengths[d_id];

    // See pair kernel: maxes is padded to 16-byte boundary so Q_shared is
    // safe for simdgroup_load. D is read directly from device.
    const uint maxes_padded = (Lq_max + 3u) & ~3u;
    threadgroup float* maxes    = (threadgroup float*)shared_raw;
    threadgroup T*     Q_shared = (threadgroup T*)(maxes + maxes_padded);
    threadgroup float* mma_store =
        (threadgroup float*)((threadgroup uchar*)Q_shared +
                             (size_t)Lq_max * dim * sizeof(T));

    for (uint i = tid; i < Lq; i += tg_size) {
        maxes[i] = -INFINITY;
    }

    {
        device const T* qsrc = queries + (size_t)q_start * dim;
        uint q_count = Lq * dim;
        for (uint i = tid; i < q_count; i += tg_size) {
            Q_shared[i] = qsrc[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qs_per_sg = (Lq + num_simds - 1u) / num_simds;
    uint q_lo = simd_id * qs_per_sg;
    uint q_hi = min(q_lo + qs_per_sg, Lq);

    const bool use_mma = ((dim & 7u) == 0u);
    uint       dim4    = dim >> 2u;
    uint       tail0   = dim4 << 2u;

    device const T* D_base = documents + (size_t)d_start * dim;
    const uint db_end       = (Ld / 8u) * 8u;
    const uint db_quad_end  = (db_end / 32u) * 32u;
    const uint db_pair_end  =
        db_quad_end + ((db_end - db_quad_end) / 16u) * 16u;

    uint q_cursor = q_lo;
    while (q_cursor < q_hi) {
        if (use_mma && q_cursor + 8u <= q_hi) {
            uint qb = q_cursor;
            threadgroup const T* Q_qb = Q_shared + (size_t)qb * dim;

            // 4-d-tile cascade (see pair kernel for design notes).
            for (uint db = 0; db < db_quad_end; db += 32u) {
                maxsim_mma_dispatch_tile_dev_4x(
                    Q_qb,
                    D_base + (size_t)db * dim,
                    dim,
                    mma_store,
                    simd_id);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                {
                    const uint row  = simd_lane >> 2;
                    const uint col0 = (simd_lane & 3u) << 1;
                    threadgroup const float* base =
                        mma_store + simd_id * 256 + row * 8;
                    float lane_max =
                        max(max(base[0   + col0], base[0   + col0 + 1]),
                            max(base[64  + col0], base[64  + col0 + 1]));
                    lane_max = max(lane_max,
                        max(max(base[128 + col0], base[128 + col0 + 1]),
                            max(base[192 + col0], base[192 + col0 + 1])));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 1u));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 2u));
                    if ((simd_lane & 3u) == 0u) {
                        maxes[qb + row] = max(maxes[qb + row], lane_max);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint db = db_quad_end; db < db_pair_end; db += 16u) {
                maxsim_mma_dispatch_tile_dev_2x(
                    Q_qb,
                    D_base + (size_t)db * dim,
                    D_base + (size_t)(db + 8u) * dim,
                    dim,
                    mma_store,
                    simd_id);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                {
                    const uint row  = simd_lane >> 2;
                    const uint col0 = (simd_lane & 3u) << 1;
                    threadgroup const float* base =
                        mma_store + simd_id * 256 + row * 8;
                    float lane_max =
                        max(max(base[col0],      base[col0 + 1]),
                            max(base[64 + col0], base[64 + col0 + 1]));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 1u));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 2u));
                    if ((simd_lane & 3u) == 0u) {
                        maxes[qb + row] = max(maxes[qb + row], lane_max);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (db_pair_end < db_end) {
                uint db = db_pair_end;
                maxsim_mma_dispatch_tile_dev(
                    Q_qb,
                    D_base + (size_t)db * dim,
                    dim,
                    mma_store,
                    simd_id);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                {
                    const uint row  = simd_lane >> 2;
                    const uint col0 = (simd_lane & 3u) << 1;
                    threadgroup const float* row_ptr =
                        mma_store + simd_id * 256 + row * 8;
                    float lane_max =
                        max(row_ptr[col0], row_ptr[col0 + 1]);
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 1u));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 2u));
                    if ((simd_lane & 3u) == 0u) {
                        maxes[qb + row] = max(maxes[qb + row], lane_max);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint ii = 0; ii < 8; ii++) {
                uint q_tok = qb + ii;
                threadgroup const T*        qrow  = Q_shared + (size_t)q_tok * dim;
                threadgroup const vec<T,4>* qrow4 =
                    (threadgroup const vec<T,4>*)qrow;
                float my_max = maxes[q_tok];
                for (uint d_tok = db_end; d_tok < Ld; ++d_tok) {
                    device const T*        drow  = D_base + (size_t)d_tok * dim;
                    device const vec<T,4>* drow4 =
                        (device const vec<T,4>*)drow;
                    float acc = 0.0f;
                    for (uint k = simd_lane; k < dim4; k += SIMD_W) {
                        vec<T,4> qv = qrow4[k];
                        vec<T,4> dv = drow4[k];
                        acc += float(qv.x) * float(dv.x)
                             + float(qv.y) * float(dv.y)
                             + float(qv.z) * float(dv.z)
                             + float(qv.w) * float(dv.w);
                    }
                    for (uint k = tail0 + simd_lane; k < dim; k += SIMD_W) {
                        acc += (float)qrow[k] * (float)drow[k];
                    }
                    acc = simd_sum(acc);
                    my_max = max(my_max, acc);
                }
                if (simd_lane == 0) {
                    maxes[q_tok] = my_max;
                }
            }
            q_cursor += 8u;
        } else {
            uint q_tok = q_cursor;
            threadgroup const T*        qrow  = Q_shared + (size_t)q_tok * dim;
            threadgroup const vec<T,4>* qrow4 =
                (threadgroup const vec<T,4>*)qrow;
            float my_max = maxes[q_tok];

            for (uint d_tok = 0; d_tok < Ld; ++d_tok) {
                device const T*        drow  = D_base + (size_t)d_tok * dim;
                device const vec<T,4>* drow4 =
                    (device const vec<T,4>*)drow;
                float acc = 0.0f;
                for (uint k = simd_lane; k < dim4; k += SIMD_W) {
                    vec<T,4> qv = qrow4[k];
                    vec<T,4> dv = drow4[k];
                    acc += float(qv.x) * float(dv.x)
                         + float(qv.y) * float(dv.y)
                         + float(qv.z) * float(dv.z)
                         + float(qv.w) * float(dv.w);
                }
                for (uint k = tail0 + simd_lane; k < dim; k += SIMD_W) {
                    acc += (float)qrow[k] * (float)drow[k];
                }
                acc = simd_sum(acc);
                my_max = max(my_max, acc);
            }
            if (simd_lane == 0) {
                maxes[q_tok] = my_max;
            }
            q_cursor += 1u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u) {
        float s = 0.0f;
        for (uint i = simd_lane; i < Lq; i += SIMD_W) {
            s += maxes[i];
        }
        s = simd_sum(s);
        if (simd_lane == 0u) {
            scores[pair_idx] = s;
        }
    }
}

#define MAXSIM_PADDED_KERNEL_ENTRY(NAME, T)                                  \
    kernel void NAME(                                                        \
        device const T*       queries        [[buffer(0)]],                  \
        device const int*     query_lengths  [[buffer(1)]],                  \
        device const T*       documents      [[buffer(2)]],                  \
        device const int*     doc_lengths    [[buffer(3)]],                  \
        device float*         scores         [[buffer(4)]],                  \
        constant uint&        dim            [[buffer(5)]],                  \
        constant uint&        Lq_max         [[buffer(6)]],                  \
        constant uint&        Ld_max         [[buffer(7)]],                  \
        constant uint&        num_candidates [[buffer(8)]],                  \
        threadgroup uchar*    shared         [[threadgroup(0)]],              \
        uint3                 tgid           [[threadgroup_position_in_grid]], \
        uint3                 tid            [[thread_position_in_threadgroup]], \
        uint3                 tg_size        [[threads_per_threadgroup]])    \
    {                                                                        \
        maxsim_padded_kernel_impl<T>(queries, query_lengths, documents,      \
                                     doc_lengths, scores, dim, Lq_max,       \
                                     Ld_max, num_candidates, shared,         \
                                     tgid.x, tid.x, tg_size.x);              \
    }

MAXSIM_PADDED_KERNEL_ENTRY(maxsim_kernel_padded_fp32, float)
MAXSIM_PADDED_KERNEL_ENTRY(maxsim_kernel_padded_fp16, half)

#if __METAL_VERSION__ >= 310
MAXSIM_PADDED_KERNEL_ENTRY(maxsim_kernel_padded_bf16, bfloat)
#endif

// =============================================================================
// K-pair packed padded kernel: pack K candidates (same q_id) per threadgroup.
// Q is staged ONCE and shared across all K pairs. Each pair has its own
// d_id and per-pair maxes slot. Work distribution: one simdgroup per
// (pair_k, q_block) task. Host sizes the threadgroup so num_simds == K *
// ceil(Lq_max/8); the last group in the launch may be partial (k_count <
// K) in which case some simdgroups idle.
// =============================================================================

template <typename T>
inline void maxsim_padded_kpair_kernel_impl(
    device const T*       queries,
    device const int*     query_lengths,
    device const T*       documents,
    device const int*     doc_lengths,
    device float*         scores,
    constant uint&        dim,
    constant uint&        Lq_max,
    constant uint&        Ld_max,
    constant uint&        num_candidates,
    constant uint&        K,
    threadgroup uchar*    shared_raw,
    uint                  tg_idx,
    uint                  tid,
    uint                  tg_size)
{
    constexpr uint SIMD_W = 32u;
    const uint simd_lane = tid & (SIMD_W - 1u);
    const uint simd_id   = tid / SIMD_W;

    const uint groups_per_q = (num_candidates + K - 1u) / K;
    const uint q_id  = tg_idx / groups_per_q;
    const uint c_lo  = (tg_idx % groups_per_q) * K;
    const uint k_count = min(K, num_candidates - c_lo);

    const uint q_start = q_id * Lq_max;
    const uint Lq      = (uint)query_lengths[q_id];

    const uint maxes_padded = (Lq_max + 3u) & ~3u;
    threadgroup float* maxes_per_pair = (threadgroup float*)shared_raw;
    threadgroup T*     Q_shared =
        (threadgroup T*)(maxes_per_pair + K * maxes_padded);
    threadgroup float* mma_store =
        (threadgroup float*)((threadgroup uchar*)Q_shared +
                             (size_t)Lq_max * dim * sizeof(T));

    for (uint i = tid; i < K * maxes_padded; i += tg_size) {
        maxes_per_pair[i] = -INFINITY;
    }

    {
        device const T* qsrc = queries + (size_t)q_start * dim;
        const uint q_count = Lq * dim;
        for (uint i = tid; i < q_count; i += tg_size) {
            Q_shared[i] = qsrc[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool use_mma = ((dim & 7u) == 0u);
    const uint dim4    = dim >> 2u;
    const uint tail0   = dim4 << 2u;

    const uint num_q_blocks = (Lq + 7u) / 8u;
    const uint total_tasks  = num_q_blocks * k_count;

    if (simd_id < total_tasks) {
        const uint pair_k  = simd_id / num_q_blocks;
        const uint q_block = simd_id % num_q_blocks;
        const uint qb      = q_block * 8u;

        // d_id is the global linear pair index (matches the K=1 kernel's
        // `pair_idx`): b * C + c. q_id is b; c_lo + pair_k is c.
        const uint d_id    = q_id * num_candidates + c_lo + pair_k;
        const uint d_start = d_id * Ld_max;
        const uint Ld      = (uint)doc_lengths[d_id];

        threadgroup float* my_maxes =
            maxes_per_pair + (size_t)pair_k * maxes_padded;
        device const T* D_base = documents + (size_t)d_start * dim;

        if (use_mma && qb + 8u <= Lq) {
            threadgroup const T* Q_qb = Q_shared + (size_t)qb * dim;
            const uint db_end = (Ld / 8u) * 8u;

            for (uint db = 0; db < db_end; db += 8u) {
                maxsim_mma_dispatch_tile_dev(
                    Q_qb, D_base + (size_t)db * dim, dim, mma_store, simd_id);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                {
                    const uint row  = simd_lane >> 2;
                    const uint col0 = (simd_lane & 3u) << 1;
                    threadgroup const float* row_ptr =
                        mma_store + simd_id * 256 + row * 8;
                    float lane_max =
                        max(row_ptr[col0], row_ptr[col0 + 1]);
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 1u));
                    lane_max = max(lane_max, simd_shuffle_xor(lane_max, 2u));
                    if ((simd_lane & 3u) == 0u) {
                        my_maxes[qb + row] =
                            max(my_maxes[qb + row], lane_max);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint ii = 0; ii < 8u; ii++) {
                uint q_tok = qb + ii;
                threadgroup const T*        qrow  = Q_shared + (size_t)q_tok * dim;
                threadgroup const vec<T,4>* qrow4 =
                    (threadgroup const vec<T,4>*)qrow;
                float my_max = my_maxes[q_tok];
                for (uint d_tok = db_end; d_tok < Ld; ++d_tok) {
                    device const T*        drow  = D_base + (size_t)d_tok * dim;
                    device const vec<T,4>* drow4 =
                        (device const vec<T,4>*)drow;
                    float acc = 0.0f;
                    for (uint k = simd_lane; k < dim4; k += SIMD_W) {
                        vec<T,4> qv = qrow4[k];
                        vec<T,4> dv = drow4[k];
                        acc += float(qv.x)*float(dv.x)
                             + float(qv.y)*float(dv.y)
                             + float(qv.z)*float(dv.z)
                             + float(qv.w)*float(dv.w);
                    }
                    for (uint k = tail0 + simd_lane; k < dim; k += SIMD_W) {
                        acc += (float)qrow[k] * (float)drow[k];
                    }
                    acc = simd_sum(acc);
                    my_max = max(my_max, acc);
                }
                if (simd_lane == 0u) {
                    my_maxes[q_tok] = my_max;
                }
            }
        } else {
            const uint q_end = min(qb + 8u, Lq);
            for (uint q_tok = qb; q_tok < q_end; q_tok++) {
                threadgroup const T*        qrow  = Q_shared + (size_t)q_tok * dim;
                threadgroup const vec<T,4>* qrow4 =
                    (threadgroup const vec<T,4>*)qrow;
                float my_max = my_maxes[q_tok];
                for (uint d_tok = 0; d_tok < Ld; d_tok++) {
                    device const T*        drow  = D_base + (size_t)d_tok * dim;
                    device const vec<T,4>* drow4 =
                        (device const vec<T,4>*)drow;
                    float acc = 0.0f;
                    for (uint k = simd_lane; k < dim4; k += SIMD_W) {
                        vec<T,4> qv = qrow4[k];
                        vec<T,4> dv = drow4[k];
                        acc += float(qv.x)*float(dv.x)
                             + float(qv.y)*float(dv.y)
                             + float(qv.z)*float(dv.z)
                             + float(qv.w)*float(dv.w);
                    }
                    for (uint k = tail0 + simd_lane; k < dim; k += SIMD_W) {
                        acc += (float)qrow[k] * (float)drow[k];
                    }
                    acc = simd_sum(acc);
                    my_max = max(my_max, acc);
                }
                if (simd_lane == 0u) {
                    my_maxes[q_tok] = my_max;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sum per-pair maxes -> scores. One simdgroup per active pair.
    if (simd_id < k_count) {
        const uint pair_k = simd_id;
        const uint d_id   = q_id * num_candidates + c_lo + pair_k;
        threadgroup const float* m =
            maxes_per_pair + (size_t)pair_k * maxes_padded;
        float s = 0.0f;
        for (uint i = simd_lane; i < Lq; i += SIMD_W) {
            s += m[i];
        }
        s = simd_sum(s);
        if (simd_lane == 0u) {
            scores[d_id] = s;
        }
    }
}

#define MAXSIM_PADDED_KPAIR_KERNEL_ENTRY(NAME, T)                            \
    kernel void NAME(                                                        \
        device const T*       queries        [[buffer(0)]],                  \
        device const int*     query_lengths  [[buffer(1)]],                  \
        device const T*       documents      [[buffer(2)]],                  \
        device const int*     doc_lengths    [[buffer(3)]],                  \
        device float*         scores         [[buffer(4)]],                  \
        constant uint&        dim            [[buffer(5)]],                  \
        constant uint&        Lq_max         [[buffer(6)]],                  \
        constant uint&        Ld_max         [[buffer(7)]],                  \
        constant uint&        num_candidates [[buffer(8)]],                  \
        constant uint&        K              [[buffer(9)]],                  \
        threadgroup uchar*    shared         [[threadgroup(0)]],              \
        uint3                 tgid           [[threadgroup_position_in_grid]], \
        uint3                 tid            [[thread_position_in_threadgroup]], \
        uint3                 tg_size        [[threads_per_threadgroup]])    \
    {                                                                        \
        maxsim_padded_kpair_kernel_impl<T>(queries, query_lengths, documents,\
                                           doc_lengths, scores, dim, Lq_max, \
                                           Ld_max, num_candidates, K,        \
                                           shared, tgid.x, tid.x,            \
                                           tg_size.x);                       \
    }

MAXSIM_PADDED_KPAIR_KERNEL_ENTRY(maxsim_kernel_padded_kpair_fp32, float)
MAXSIM_PADDED_KPAIR_KERNEL_ENTRY(maxsim_kernel_padded_kpair_fp16, half)
#if __METAL_VERSION__ >= 310
MAXSIM_PADDED_KPAIR_KERNEL_ENTRY(maxsim_kernel_padded_kpair_bf16, bfloat)
#endif
