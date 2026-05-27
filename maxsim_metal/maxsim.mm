#include <torch/torch.h>

#include <algorithm>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// ---------------------------------------------------------------------------
// Cache the Metal library + per-kernel pipeline state objects (PSOs). PSO
// creation involves shader compilation/loading and runs in the millisecond
// range; doing it on every call dwarfs the actual kernel work for small
// workloads. Keys are kernel-function names (the .metallib symbols).
// ---------------------------------------------------------------------------

static id<MTLDevice> get_metal_device() {
  static id<MTLDevice> device = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    device = MTLCreateSystemDefaultDevice();
  });
  return device;
}

static id<MTLLibrary> get_metal_library() {
  static id<MTLLibrary> library = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    NSError *error = nil;
    library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(get_metal_device(),
                                                         &error);
    if (!library) {
      NSLog(@"Failed to create Metal library: %@", error);
    }
  });
  return library;
}

static id<MTLComputePipelineState> get_pso(const char *kernel_name) {
  static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *cache =
      nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    cache = [[NSMutableDictionary alloc] init];
  });

  NSString *key = [NSString stringWithUTF8String:kernel_name];
  id<MTLComputePipelineState> pso;
  @synchronized(cache) {
    pso = cache[key];
    if (pso == nil) {
      id<MTLLibrary> library = get_metal_library();
      id<MTLFunction> func = [library newFunctionWithName:key];
      if (!func) return (id<MTLComputePipelineState>)nil;
      NSError *error = nil;
      pso = [get_metal_device()
          newComputePipelineStateWithFunction:func
                                        error:&error];
      if (pso == nil) {
        NSLog(@"Failed to build PSO for %s: %@", kernel_name, error);
        return (id<MTLComputePipelineState>)nil;
      }
      cache[key] = pso;
    }
  }
  return pso;
}

// Forward declarations (match torch-ext/torch_binding.h). Kept inline here so
// this translation unit does not depend on the torch-ext include path.
torch::Tensor maxsim_forward(torch::Tensor queries,
                             torch::Tensor query_offsets,
                             torch::Tensor documents,
                             torch::Tensor document_offsets,
                             torch::Tensor pair_query_ids,
                             torch::Tensor pair_document_ids,
                             int64_t max_q_len);
torch::Tensor maxsim_padded_forward(torch::Tensor queries,
                                    torch::Tensor query_lengths,
                                    torch::Tensor documents,
                                    torch::Tensor doc_lengths,
                                    int64_t Lq_max,
                                    int64_t Ld_max,
                                    int64_t num_candidates);

// 64 threads per threadgroup. Must be a power of two for the tree reduction
// inside maxsim_kernel_impl.
static constexpr NSUInteger kThreadgroupSize = 64;

// Fast pair kernel: one threadgroup per pair. Threadgroup width (in threads)
// is chosen by `pick_pair_tg_size` below based on the work-per-pair, between
// kPairMinThreadgroupSize (4 simdgroups) and kPairMaxThreadgroupSize.
static constexpr NSUInteger kPairMinThreadgroupSize = 128;
static constexpr NSUInteger kPairMaxThreadgroupSize = 1024;
static constexpr NSUInteger kSimdWidth = 32;
// `kPairDocTile` used to gate the threadgroup-memory budget for staged D
// tiles. After step 4 (drop D-staging), it is no longer used.
// Max threadgroup memory we allow for the *static* pair/padded sizing gate
// before we know the exact `tg_size` (MMA scratch scales with #simdgroups).
// Per-launch we also clamp against `[device maxThreadgroupMemoryLength]`.
static constexpr NSUInteger kPairMaxThreadgroupMemBytes = 32 * 1024;
// Upper bound on simdgroup MMA spill storage. The 4-d-tile batched path
// stores FOUR 8×8 fragments per simdgroup per iteration, so each simdgroup
// owns a 256-float slot in `mma_store` (1024 bytes).
static constexpr NSUInteger kMmaScratchWorstBytes = 32 * 256 * sizeof(float);

static NSUInteger mma_scratch_bytes_for_tg(NSUInteger tg_size) {
  NSUInteger num_simds = tg_size / kSimdWidth;
  return num_simds * 256 * sizeof(float);
}

// Pick a threadgroup size (in threads) that keeps work-per-simdgroup roughly
// constant across workloads. The kernel distributes q_toks across simdgroups
// (each owns ~Lq/num_simds query tokens) and each simdgroup walks all
// d_toks serially, so work-per-simdgroup ~= (Lq * Ld) / num_simds. We aim
// for ~4096 (q_tok, d_tok) pairs per simdgroup.
//
// When `mma_ok` is true (i.e. `dim % 8 == 0` so the 8×8 simdgroup_matrix
// path can fire), we additionally cap `num_simds` at `ceil(Lq/8)`. The
// reason: the MMA path requires `qs_per_sg >= 8`, otherwise that simdgroup
// silently falls back to the slower scalar `simd_sum` loop. Over-subscribing
// simdgroups starves the MMA path for no parallelism gain (each pair is
// already its own threadgroup; inter-pair parallelism fills the GPU).
static NSUInteger pick_pair_tg_size(int64_t Lq_max, int64_t Ld_max,
                                    NSUInteger max_threads, bool mma_ok) {
  const int64_t work = Lq_max * Ld_max;
  const int64_t target_per_simd = 4096;
  int64_t desired_simds = (work + target_per_simd - 1) / target_per_simd;
  if (desired_simds < (int64_t)(kPairMinThreadgroupSize / kSimdWidth)) {
    desired_simds = kPairMinThreadgroupSize / kSimdWidth;
  }
  if (mma_ok && Lq_max > 0) {
    const int64_t max_useful_simds = (Lq_max + 7) / 8;
    const int64_t floor_simds =
        (int64_t)(kPairMinThreadgroupSize / kSimdWidth);
    const int64_t capped =
        max_useful_simds > floor_simds ? max_useful_simds : floor_simds;
    if (desired_simds > capped) desired_simds = capped;
  }
  NSUInteger tg = (NSUInteger)desired_simds * kSimdWidth;
  if (tg > kPairMaxThreadgroupSize) tg = kPairMaxThreadgroupSize;
  if (tg > max_threads) tg = max_threads;
  // Round down to multiple of simd width to stay simdgroup-aligned.
  tg = (tg / kSimdWidth) * kSimdWidth;
  if (tg == 0) tg = kSimdWidth;
  return tg;
}

namespace {

// Validate offsets on CPU: must start at 0, be non-decreasing, end at the
// total token count, and have no zero-length segments. Also return the
// maximum segment length.
int compute_max_segment_len(const torch::Tensor &offsets_cpu_i32,
                            int64_t expected_total, const char *name) {
  TORCH_CHECK(offsets_cpu_i32.dim() == 1, name, " must be a 1-D tensor");
  TORCH_CHECK(offsets_cpu_i32.size(0) >= 2, name,
              " must have length >= 2 (need at least one segment)");
  const int32_t *p = offsets_cpu_i32.data_ptr<int32_t>();
  int64_t n = offsets_cpu_i32.size(0);
  TORCH_CHECK(p[0] == 0, name, "[0] must equal 0, got ", p[0]);
  TORCH_CHECK((int64_t)p[n - 1] == expected_total, name, "[-1] (", p[n - 1],
              ") must equal the corresponding total token count (",
              expected_total, ")");
  int max_len = 0;
  for (int64_t i = 0; i + 1 < n; ++i) {
    int diff = p[i + 1] - p[i];
    TORCH_CHECK(diff > 0, "empty segment in ", name, " at index ", i,
                " (offsets[", i, "]=", p[i], ", offsets[", i + 1,
                "]=", p[i + 1], ")");
    if (diff > max_len) max_len = diff;
  }
  return max_len;
}

void validate_pair_ids(const torch::Tensor &ids_cpu_i32, int64_t upper,
                       const char *name) {
  TORCH_CHECK(ids_cpu_i32.dim() == 1, name, " must be a 1-D tensor");
  const int32_t *p = ids_cpu_i32.data_ptr<int32_t>();
  int64_t n = ids_cpu_i32.size(0);
  for (int64_t i = 0; i < n; ++i) {
    TORCH_CHECK(p[i] >= 0 && (int64_t)p[i] < upper, name, "[", i, "] = ", p[i],
                " out of range [0, ", upper, ")");
  }
}

const char *kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_bf16";
    default:
      return nullptr;
  }
}

const char *pair_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_pair_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_pair_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_pair_bf16";
    default:
      return nullptr;
  }
}

const char *padded_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_padded_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_padded_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_padded_bf16";
    default:
      return nullptr;
  }
}

const char *padded_argmax_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_padded_argmax_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_padded_argmax_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_padded_argmax_bf16";
    default:
      return nullptr;
  }
}

const char *padded_backward_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_padded_backward_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_padded_backward_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_padded_backward_bf16";
    default:
      return nullptr;
  }
}

const char *packed_argmax_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_packed_argmax_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_packed_argmax_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_packed_argmax_bf16";
    default:
      return nullptr;
  }
}

const char *packed_backward_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_packed_backward_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_packed_backward_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_packed_backward_bf16";
    default:
      return nullptr;
  }
}

const char *contrastive_argmax_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_contrastive_argmax_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_contrastive_argmax_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_contrastive_argmax_bf16";
    default:
      return nullptr;
  }
}

const char *contrastive_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_contrastive_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_contrastive_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_contrastive_bf16";
    default:
      return nullptr;
  }
}

const char *contrastive_backward_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_contrastive_backward_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_contrastive_backward_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_contrastive_backward_bf16";
    default:
      return nullptr;
  }
}

const char *padded_kpair_kernel_name_for_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "maxsim_kernel_padded_kpair_fp32";
    case at::ScalarType::Half:
      return "maxsim_kernel_padded_kpair_fp16";
    case at::ScalarType::BFloat16:
      return "maxsim_kernel_padded_kpair_bf16";
    default:
      return nullptr;
  }
}

}  // namespace

torch::Tensor maxsim_forward(torch::Tensor queries,
                             torch::Tensor query_offsets,
                             torch::Tensor documents,
                             torch::Tensor document_offsets,
                             torch::Tensor pair_query_ids,
                             torch::Tensor pair_document_ids,
                             int64_t max_q_len) {
  // ---- Device / dtype / shape validation ------------------------------------
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
  TORCH_CHECK(documents.device() == queries.device(),
              "documents must be on the same device as queries");
  TORCH_CHECK(query_offsets.device() == queries.device(),
              "query_offsets must be on the same device as queries");
  TORCH_CHECK(document_offsets.device() == queries.device(),
              "document_offsets must be on the same device as queries");
  TORCH_CHECK(pair_query_ids.device() == queries.device(),
              "pair_query_ids must be on the same device as queries");
  TORCH_CHECK(pair_document_ids.device() == queries.device(),
              "pair_document_ids must be on the same device as queries");

  TORCH_CHECK(queries.dim() == 2,
              "queries must be 2-D [total_q_tokens, dim]; got ", queries.dim(),
              "-D");
  TORCH_CHECK(documents.dim() == 2,
              "documents must be 2-D [total_d_tokens, dim]; got ",
              documents.dim(), "-D");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim (", queries.size(1), ") must match documents.dim (",
              documents.size(1), ")");
  TORCH_CHECK(queries.size(1) > 0, "embedding dim must be > 0");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  const char *kname = kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

  // ---- Coerce shapes / dtypes ----------------------------------------------
  queries = queries.contiguous();
  documents = documents.contiguous();
  query_offsets = query_offsets.contiguous().to(torch::kInt32);
  document_offsets = document_offsets.contiguous().to(torch::kInt32);
  pair_query_ids = pair_query_ids.contiguous().to(torch::kInt32);
  pair_document_ids = pair_document_ids.contiguous().to(torch::kInt32);

  TORCH_CHECK(pair_query_ids.dim() == 1, "pair_query_ids must be 1-D");
  TORCH_CHECK(pair_document_ids.dim() == 1, "pair_document_ids must be 1-D");
  TORCH_CHECK(pair_query_ids.size(0) == pair_document_ids.size(0),
              "pair_query_ids and pair_document_ids must have the same length");

  const int64_t num_pairs = pair_query_ids.size(0);
  const int64_t dim = queries.size(1);

  auto scores = torch::zeros(
      {num_pairs},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));

  if (num_pairs == 0) {
    return scores;
  }

  // ---- Offset / pair-id validation on CPU (only when caller did not pre-
  //      compute max_q_len). This path forces 4 device->host syncs and is
  //      therefore only meant for first-time / debug usage; production
  //      callers should pass `max_q_len >= 0` and validate in Python.
  int max_q_len_i;
  if (max_q_len < 0) {
    auto qoff_cpu = query_offsets.to(torch::kCPU);
    auto doff_cpu = document_offsets.to(torch::kCPU);
    auto qids_cpu = pair_query_ids.to(torch::kCPU);
    auto dids_cpu = pair_document_ids.to(torch::kCPU);

    max_q_len_i =
        compute_max_segment_len(qoff_cpu, queries.size(0), "query_offsets");
    (void)compute_max_segment_len(doff_cpu, documents.size(0),
                                  "document_offsets");

    const int64_t num_queries = qoff_cpu.size(0) - 1;
    const int64_t num_documents = doff_cpu.size(0) - 1;
    validate_pair_ids(qids_cpu, num_queries, "pair_query_ids");
    validate_pair_ids(dids_cpu, num_documents, "pair_document_ids");
  } else {
    TORCH_CHECK(max_q_len > 0, "max_q_len must be > 0; got ", max_q_len);
    TORCH_CHECK(max_q_len <= queries.size(0),
                "max_q_len (", max_q_len, ") cannot exceed total query tokens (",
                queries.size(0), ")");
    max_q_len_i = (int)max_q_len;
  }
  const int max_q_len_used = max_q_len_i;

  // ---- Pick fast (pair) vs naive kernel ------------------------------------
  const size_t elem_size = (size_t)queries.element_size();
  const char *pair_kname = pair_kernel_name_for_dtype(queries.scalar_type());
  const size_t pair_q_bytes  = (size_t)max_q_len_used * (size_t)dim * elem_size;
  // D is no longer staged in threadgroup memory (`simdgroup_load` from device).
  // `pair_max_bytes` is rounded up to a multiple of 16 so the Q-tile region
  // starts 16-byte aligned -- required by `simdgroup_load` for half/bfloat
  // 8×8 fragments.
  const size_t pair_max_floats_padded =
      ((size_t)max_q_len_used + 3u) & ~(size_t)3u;
  const size_t pair_max_bytes = pair_max_floats_padded * sizeof(float);
  const size_t pair_prelim = pair_q_bytes + pair_max_bytes;
  NSUInteger dev_tg_max = [get_metal_device() maxThreadgroupMemoryLength];
  NSUInteger mem_budget = std::min(kPairMaxThreadgroupMemBytes, dev_tg_max);
  // Estimate the threadgroup size we'd use to size the MMA scratch correctly
  // (it scales with #simdgroups, so the worst-case 32-simd allowance can be a
  // significant overestimate for moderate workloads). `pick_pair_tg_size` is
  // monotonic in its `max_threads` argument so capping at kPairMaxThreadgroupSize
  // here is an upper bound on the value we'll choose post-PSO-load.
  const int64_t avg_d_len_est =
      num_pairs > 0
          ? (documents.size(0) / std::max<int64_t>(num_pairs, 1))
          : (int64_t)max_q_len_used;
  const bool packed_mma_ok = (dim % 8 == 0);
  const NSUInteger desired_pair_tg = pick_pair_tg_size(
      max_q_len_used, avg_d_len_est, kPairMaxThreadgroupSize, packed_mma_ok);
  const size_t pair_mma_scratch =
      (size_t)mma_scratch_bytes_for_tg(desired_pair_tg);
  const bool use_pair_kernel =
      pair_kname != nullptr &&
      pair_prelim + pair_mma_scratch <= mem_budget;
  const char *active_kname = use_pair_kernel ? pair_kname : kname;

  // ---- Metal dispatch -------------------------------------------------------
  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(active_kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", active_kname,
                " (kernel may not be available on this Metal version)");

    NSUInteger tg_size;
    if (use_pair_kernel) {
      // For the packed path we don't know per-doc max length without
      // another sync; use Ld_max ~= total / num_pairs as a rough proxy.
      const int64_t avg_d_len =
          num_pairs > 0
              ? (documents.size(0) / std::max<int64_t>(num_pairs, 1))
              : (int64_t)max_q_len_used;
      tg_size = pick_pair_tg_size(max_q_len_used, avg_d_len,
                                  pso.maxTotalThreadsPerThreadgroup,
                                  packed_mma_ok);
    } else {
      tg_size = kThreadgroupSize;
      if (tg_size > pso.maxTotalThreadsPerThreadgroup) {
        tg_size = pso.maxTotalThreadsPerThreadgroup;
      }
    }
    const NSUInteger threadgroup_mem_bytes =
        use_pair_kernel
            ? (NSUInteger)(pair_prelim + mma_scratch_bytes_for_tg(tg_size))
            : (NSUInteger)(tg_size * sizeof(float));
    TORCH_CHECK(
        threadgroup_mem_bytes <= dev_tg_max,
        "threadgroup memory (", threadgroup_mem_bytes,
        " bytes) exceeds device limit (", dev_tg_max, " bytes)");

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(query_offsets)
                  offset:query_offsets.storage_offset() *
                         query_offsets.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(document_offsets)
                  offset:document_offsets.storage_offset() *
                         document_offsets.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(pair_query_ids)
                  offset:pair_query_ids.storage_offset() *
                         pair_query_ids.element_size()
                 atIndex:4];
      [encoder setBuffer:getMTLBufferStorage(pair_document_ids)
                  offset:pair_document_ids.storage_offset() *
                         pair_document_ids.element_size()
                 atIndex:5];
      [encoder setBuffer:getMTLBufferStorage(scores)
                  offset:scores.storage_offset() * scores.element_size()
                 atIndex:6];

      uint dim_u = (uint)dim;
      [encoder setBytes:&dim_u length:sizeof(uint) atIndex:7];
      if (use_pair_kernel) {
        uint max_q_len_u = (uint)max_q_len_used;
        [encoder setBytes:&max_q_len_u length:sizeof(uint) atIndex:8];
      }

      [encoder setThreadgroupMemoryLength:threadgroup_mem_bytes atIndex:0];

      MTLSize grid = use_pair_kernel
                         ? MTLSizeMake((NSUInteger)num_pairs, 1, 1)
                         : MTLSizeMake((NSUInteger)num_pairs,
                                       (NSUInteger)max_q_len_used, 1);
      MTLSize tg = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return scores;
}

// =============================================================================
// Padded fast path: reads `[B, Lq_max, D]` / `[B, C, Ld_max, D]` directly via
// strides, no pack/gather, no CPU<->GPU syncs. Pair ordering is canonical
// row-major (b, c) -> (q_id=b, d_id=b*C+c).
// =============================================================================

torch::Tensor maxsim_padded_forward(torch::Tensor queries,
                                    torch::Tensor query_lengths,
                                    torch::Tensor documents,
                                    torch::Tensor doc_lengths,
                                    int64_t Lq_max,
                                    int64_t Ld_max,
                                    int64_t num_candidates) {
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
  TORCH_CHECK(documents.device() == queries.device(),
              "documents must be on the same device as queries");
  TORCH_CHECK(query_lengths.device() == queries.device(),
              "query_lengths must be on the same device as queries");
  TORCH_CHECK(doc_lengths.device() == queries.device(),
              "doc_lengths must be on the same device as queries");

  TORCH_CHECK(queries.dim() == 2,
              "queries must be 2-D [B*Lq_max, dim]; got ", queries.dim(), "-D");
  TORCH_CHECK(documents.dim() == 2,
              "documents must be 2-D [B*C*Ld_max, dim]; got ", documents.dim(),
              "-D");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim (", queries.size(1), ") must match documents.dim (",
              documents.size(1), ")");
  TORCH_CHECK(queries.size(1) > 0, "embedding dim must be > 0");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(Lq_max > 0, "Lq_max must be > 0; got ", Lq_max);
  TORCH_CHECK(Ld_max > 0, "Ld_max must be > 0; got ", Ld_max);
  TORCH_CHECK(num_candidates > 0,
              "num_candidates must be > 0; got ", num_candidates);

  const char *pad_kname =
      padded_kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(pad_kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

  queries = queries.contiguous();
  documents = documents.contiguous();
  query_lengths = query_lengths.contiguous().to(torch::kInt32);
  doc_lengths = doc_lengths.contiguous().to(torch::kInt32);

  const int64_t total_q_rows = queries.size(0);
  const int64_t total_d_rows = documents.size(0);
  TORCH_CHECK(total_q_rows % Lq_max == 0,
              "queries rows (", total_q_rows, ") must be a multiple of Lq_max (",
              Lq_max, ")");
  TORCH_CHECK(total_d_rows % Ld_max == 0,
              "documents rows (", total_d_rows,
              ") must be a multiple of Ld_max (", Ld_max, ")");
  const int64_t B = total_q_rows / Lq_max;
  const int64_t total_pairs = total_d_rows / Ld_max;
  TORCH_CHECK(total_pairs == B * num_candidates,
              "documents shape implies ", total_pairs,
              " pairs but expected B * num_candidates = ", B, " * ",
              num_candidates, " = ", B * num_candidates);
  TORCH_CHECK(query_lengths.dim() == 1 && query_lengths.size(0) == B,
              "query_lengths must be 1-D [B=", B, "]; got shape ",
              query_lengths.sizes());
  TORCH_CHECK(doc_lengths.dim() == 1 && doc_lengths.size(0) == total_pairs,
              "doc_lengths must be 1-D [B*C=", total_pairs, "]; got shape ",
              doc_lengths.sizes());

  const int64_t dim = queries.size(1);
  auto scores = torch::zeros(
      {total_pairs},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));
  if (total_pairs == 0) {
    return scores;
  }

  const size_t elem_size = (size_t)queries.element_size();
  const size_t pad_q_bytes  = (size_t)Lq_max * (size_t)dim * elem_size;
  // D not staged; pad maxes to 16-byte boundary (see packed forward).
  const size_t pad_max_floats_padded =
      ((size_t)Lq_max + 3u) & ~(size_t)3u;
  const size_t pad_max_bytes = pad_max_floats_padded * sizeof(float);
  NSUInteger dev_tg_max = [get_metal_device() maxThreadgroupMemoryLength];
  NSUInteger mem_budget = std::min(kPairMaxThreadgroupMemBytes, dev_tg_max);
  // Same logic as the packed forward: size the MMA scratch from the actual
  // workload, not the worst-case 32-simdgroup allowance.
  const bool padded_mma_ok = (dim % 8 == 0);
  // K-pair packing decision (experimental, currently disabled — it
  // regressed Heavy at K=8 and didn't help Small at K=2 because per-TG
  // launch overhead turns out to be well-amortized at K=1 once D-staging
  // is removed). Kept in the source for future re-evaluation.
  int64_t K = 1;

  if (K > 1) {
    // ---- K-pair packed dispatch -------------------------------------------
    const char *pad_kpair_kname =
        padded_kpair_kernel_name_for_dtype(queries.scalar_type());
    TORCH_CHECK(pad_kpair_kname != nullptr,
                "queries/documents dtype must be float32, float16, or bfloat16");

    const int64_t q_blocks  = (Lq_max + 7) / 8;
    const int64_t num_simds = q_blocks * K;             // simds per TG
    const NSUInteger kpair_tg_size = (NSUInteger)num_simds * kSimdWidth;
    const size_t kpair_maxes_bytes = (size_t)K * pad_max_bytes;
    const size_t kpair_mma_scratch =
        (size_t)mma_scratch_bytes_for_tg(kpair_tg_size);
    const size_t kpair_tg_bytes =
        kpair_maxes_bytes + pad_q_bytes + kpair_mma_scratch;
    TORCH_CHECK(
        kpair_tg_bytes <= dev_tg_max,
        "K-pair padded threadgroup memory (", kpair_tg_bytes,
        " bytes) exceeds device limit (", dev_tg_max, " bytes); reduce K");

    const int64_t groups_per_q = (num_candidates + K - 1) / K;
    const NSUInteger grid_x = (NSUInteger)(B * groups_per_q);

    @autoreleasepool {
      id<MTLComputePipelineState> pso = get_pso(pad_kpair_kname);
      TORCH_CHECK(pso, "Failed to load Metal function ", pad_kpair_kname);
      TORCH_CHECK(kpair_tg_size <= pso.maxTotalThreadsPerThreadgroup,
                  "K-pair TG size ", kpair_tg_size, " > PSO max ",
                  pso.maxTotalThreadsPerThreadgroup);
      id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
      dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pso];

        [encoder setBuffer:getMTLBufferStorage(queries)
                    offset:queries.storage_offset() * queries.element_size()
                   atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(query_lengths)
                    offset:query_lengths.storage_offset() *
                           query_lengths.element_size()
                   atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(documents)
                    offset:documents.storage_offset() * documents.element_size()
                   atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(doc_lengths)
                    offset:doc_lengths.storage_offset() *
                           doc_lengths.element_size()
                   atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(scores)
                    offset:scores.storage_offset() * scores.element_size()
                   atIndex:4];

        uint dim_u    = (uint)dim;
        uint Lq_max_u = (uint)Lq_max;
        uint Ld_max_u = (uint)Ld_max;
        uint C_u      = (uint)num_candidates;
        uint K_u      = (uint)K;
        [encoder setBytes:&dim_u    length:sizeof(uint) atIndex:5];
        [encoder setBytes:&Lq_max_u length:sizeof(uint) atIndex:6];
        [encoder setBytes:&Ld_max_u length:sizeof(uint) atIndex:7];
        [encoder setBytes:&C_u      length:sizeof(uint) atIndex:8];
        [encoder setBytes:&K_u      length:sizeof(uint) atIndex:9];

        [encoder setThreadgroupMemoryLength:(NSUInteger)kpair_tg_bytes
                                    atIndex:0];

        MTLSize grid = MTLSizeMake(grid_x, 1, 1);
        MTLSize tg   = MTLSizeMake(kpair_tg_size, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

        [encoder endEncoding];
        torch::mps::commit();
      });
    }
    return scores;
  }

  // ---- K=1 dispatch (existing per-pair kernel) ----------------------------
  const size_t pad_prelim = pad_q_bytes + pad_max_bytes;
  const NSUInteger desired_pad_tg = pick_pair_tg_size(
      Lq_max, Ld_max, kPairMaxThreadgroupSize, padded_mma_ok);
  const size_t pad_mma_scratch =
      (size_t)mma_scratch_bytes_for_tg(desired_pad_tg);
  TORCH_CHECK(
      pad_prelim + pad_mma_scratch <= mem_budget,
      "Lq_max * dim too large for the padded Metal kernel "
      "(threadgroup memory required >= ",
      pad_prelim + pad_mma_scratch, " bytes, budget = ", mem_budget,
      " bytes). Use the packed API for this shape.");

  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(pad_kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", pad_kname,
                " (kernel may not be available on this Metal version)");

    NSUInteger tg_size = pick_pair_tg_size(
        Lq_max, Ld_max, pso.maxTotalThreadsPerThreadgroup, padded_mma_ok);
    const NSUInteger padded_tg_bytes =
        (NSUInteger)(pad_prelim + mma_scratch_bytes_for_tg(tg_size));
    TORCH_CHECK(
        padded_tg_bytes <= dev_tg_max,
        "padded threadgroup memory (", padded_tg_bytes,
        " bytes) exceeds device limit (", dev_tg_max, " bytes)");
    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(query_lengths)
                  offset:query_lengths.storage_offset() *
                         query_lengths.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(doc_lengths)
                  offset:doc_lengths.storage_offset() *
                         doc_lengths.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(scores)
                  offset:scores.storage_offset() * scores.element_size()
                 atIndex:4];

      uint dim_u = (uint)dim;
      uint Lq_max_u = (uint)Lq_max;
      uint Ld_max_u = (uint)Ld_max;
      uint C_u = (uint)num_candidates;
      [encoder setBytes:&dim_u    length:sizeof(uint) atIndex:5];
      [encoder setBytes:&Lq_max_u length:sizeof(uint) atIndex:6];
      [encoder setBytes:&Ld_max_u length:sizeof(uint) atIndex:7];
      [encoder setBytes:&C_u      length:sizeof(uint) atIndex:8];

      [encoder setThreadgroupMemoryLength:padded_tg_bytes atIndex:0];

      MTLSize grid = MTLSizeMake((NSUInteger)total_pairs, 1, 1);
      MTLSize tg = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return scores;
}

// =============================================================================
// Padded forward + argmax (for backward / training): identical structure to
// `maxsim_padded_forward` but writes per-q-tok argmax positions to a second
// output tensor. Used as `save_for_backward` payload by a future
// `torch.autograd.Function`. PyTorch first-index-wins tiebreak.
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor> maxsim_padded_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor query_lengths,
    torch::Tensor documents,
    torch::Tensor doc_lengths,
    int64_t Lq_max,
    int64_t Ld_max,
    int64_t num_candidates) {
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
  TORCH_CHECK(documents.device() == queries.device(),
              "documents must be on the same device as queries");
  TORCH_CHECK(query_lengths.device() == queries.device(),
              "query_lengths must be on the same device as queries");
  TORCH_CHECK(doc_lengths.device() == queries.device(),
              "doc_lengths must be on the same device as queries");

  TORCH_CHECK(queries.dim() == 2,
              "queries must be 2-D [B*Lq_max, dim]; got ", queries.dim(), "-D");
  TORCH_CHECK(documents.dim() == 2,
              "documents must be 2-D [B*C*Ld_max, dim]; got ", documents.dim(),
              "-D");
  TORCH_CHECK(queries.size(1) == documents.size(1),
              "queries.dim must match documents.dim");
  TORCH_CHECK(queries.size(1) > 0, "embedding dim must be > 0");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(Lq_max > 0 && Ld_max > 0 && num_candidates > 0,
              "Lq_max/Ld_max/num_candidates must all be > 0");

  const char *pad_kname =
      padded_argmax_kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(pad_kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

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

  const int64_t dim = queries.size(1);
  auto scores = torch::zeros(
      {total_pairs},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));
  auto argmax = torch::zeros(
      {total_pairs, Lq_max},
      torch::TensorOptions().dtype(torch::kInt32).device(queries.device()));
  if (total_pairs == 0) {
    return std::make_tuple(scores, argmax);
  }

  const size_t elem_size = (size_t)queries.element_size();
  const size_t pad_q_bytes = (size_t)Lq_max * (size_t)dim * elem_size;
  const size_t pad_max_floats_padded =
      ((size_t)Lq_max + 3u) & ~(size_t)3u;
  const size_t pad_max_bytes = pad_max_floats_padded * sizeof(float);
  // Extra int32 argmax_buf alongside maxes in threadgroup memory.
  const size_t pad_arg_bytes = pad_max_floats_padded * sizeof(int);
  const size_t pad_prelim = pad_q_bytes + pad_max_bytes + pad_arg_bytes;
  NSUInteger dev_tg_max = [get_metal_device() maxThreadgroupMemoryLength];
  NSUInteger mem_budget = std::min(kPairMaxThreadgroupMemBytes, dev_tg_max);
  const bool padded_mma_ok = (dim % 8 == 0);
  const NSUInteger desired_pad_tg = pick_pair_tg_size(
      Lq_max, Ld_max, kPairMaxThreadgroupSize, padded_mma_ok);
  const size_t pad_mma_scratch =
      (size_t)mma_scratch_bytes_for_tg(desired_pad_tg);
  TORCH_CHECK(
      pad_prelim + pad_mma_scratch <= mem_budget,
      "Lq_max * dim too large for the argmax Metal kernel "
      "(threadgroup memory required >= ",
      pad_prelim + pad_mma_scratch, " bytes, budget = ", mem_budget,
      " bytes).");

  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(pad_kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", pad_kname);

    NSUInteger tg_size = pick_pair_tg_size(
        Lq_max, Ld_max, pso.maxTotalThreadsPerThreadgroup, padded_mma_ok);
    const NSUInteger padded_tg_bytes =
        (NSUInteger)(pad_prelim + mma_scratch_bytes_for_tg(tg_size));
    TORCH_CHECK(padded_tg_bytes <= dev_tg_max,
                "argmax padded threadgroup memory (", padded_tg_bytes,
                " bytes) exceeds device limit (", dev_tg_max, " bytes)");

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(query_lengths)
                  offset:query_lengths.storage_offset() *
                         query_lengths.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(doc_lengths)
                  offset:doc_lengths.storage_offset() *
                         doc_lengths.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(scores)
                  offset:scores.storage_offset() * scores.element_size()
                 atIndex:4];
      [encoder setBuffer:getMTLBufferStorage(argmax)
                  offset:argmax.storage_offset() * argmax.element_size()
                 atIndex:5];

      uint dim_u    = (uint)dim;
      uint Lq_max_u = (uint)Lq_max;
      uint Ld_max_u = (uint)Ld_max;
      uint C_u      = (uint)num_candidates;
      [encoder setBytes:&dim_u    length:sizeof(uint) atIndex:6];
      [encoder setBytes:&Lq_max_u length:sizeof(uint) atIndex:7];
      [encoder setBytes:&Ld_max_u length:sizeof(uint) atIndex:8];
      [encoder setBytes:&C_u      length:sizeof(uint) atIndex:9];

      [encoder setThreadgroupMemoryLength:padded_tg_bytes atIndex:0];

      MTLSize grid = MTLSizeMake((NSUInteger)total_pairs, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return std::make_tuple(scores, argmax);
}

// =============================================================================
// Padded backward. Routes incoming `dscore` to (dqueries, ddocuments) using
// the per-q-tok argmax positions saved by the forward pass. fp32 grad
// outputs irrespective of input dtype.
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
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
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

  const char *back_kname =
      padded_backward_kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(back_kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

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

  const int64_t dim = queries.size(1);

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto dqueries   = torch::zeros({total_q_rows, dim}, fp32_opts);
  auto ddocuments = torch::zeros({total_d_rows, dim}, fp32_opts);
  if (total_pairs == 0) return std::make_tuple(dqueries, ddocuments);

  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(back_kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", back_kname);

    // 4 simdgroups per block — splits q_toks 4 ways. lanes within a
    // simdgroup split the dim dimension.
    NSUInteger tg_size = 4 * kSimdWidth;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup) {
      tg_size = pso.maxTotalThreadsPerThreadgroup;
    }

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(query_lengths)
                  offset:query_lengths.storage_offset() *
                         query_lengths.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(doc_lengths)
                  offset:doc_lengths.storage_offset() *
                         doc_lengths.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(argmax)
                  offset:argmax.storage_offset() * argmax.element_size()
                 atIndex:4];
      [encoder setBuffer:getMTLBufferStorage(dscore)
                  offset:dscore.storage_offset() * dscore.element_size()
                 atIndex:5];
      [encoder setBuffer:getMTLBufferStorage(dqueries)
                  offset:dqueries.storage_offset() * dqueries.element_size()
                 atIndex:6];
      [encoder setBuffer:getMTLBufferStorage(ddocuments)
                  offset:ddocuments.storage_offset() * ddocuments.element_size()
                 atIndex:7];

      uint dim_u    = (uint)dim;
      uint Lq_max_u = (uint)Lq_max;
      uint Ld_max_u = (uint)Ld_max;
      uint C_u      = (uint)num_candidates;
      [encoder setBytes:&dim_u    length:sizeof(uint) atIndex:8];
      [encoder setBytes:&Lq_max_u length:sizeof(uint) atIndex:9];
      [encoder setBytes:&Ld_max_u length:sizeof(uint) atIndex:10];
      [encoder setBytes:&C_u      length:sizeof(uint) atIndex:11];

      MTLSize grid = MTLSizeMake((NSUInteger)total_pairs, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return std::make_tuple(dqueries, ddocuments);
}

// =============================================================================
// Contrastive (all-pairs) MaxSim — Metal host functions.
// API mirrors the CUDA side: queries[Nq, Lq, D] x packed_docs(document_offsets) ->
// scores[Nq, Nb] (+ optional argmax[Nq, Nb, Lq]).
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor>
maxsim_contrastive_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor document_offsets) {
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              document_offsets.device() == queries.device(),
              "all contrastive inputs must share a device");
  TORCH_CHECK(queries.dim() == 3, "queries must be [Nq, Lq, D]");
  TORCH_CHECK(documents.dim() == 2,
              "documents must be [total_d_tokens, D] (packed)");
  TORCH_CHECK(document_offsets.dim() == 1, "document_offsets must be 1-D [Nb+1]");
  TORCH_CHECK(queries.size(2) == documents.size(1),
              "queries.D must match documents.D");
  TORCH_CHECK(queries.size(2) > 0, "embedding dim must be > 0");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");

  const char *kname =
      contrastive_argmax_kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

  queries    = queries.contiguous();
  documents  = documents.contiguous();
  document_offsets = document_offsets.contiguous().to(torch::kInt32);

  const int64_t Nq  = queries.size(0);
  const int64_t Lq  = queries.size(1);
  const int64_t dim = queries.size(2);
  const int64_t Nb  = document_offsets.size(0) - 1;
  TORCH_CHECK(Nb >= 0, "document_offsets must have at least 1 element");
  TORCH_CHECK(Lq > 0, "Lq must be > 0");

  auto scores = torch::zeros(
      {Nq, Nb},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));
  auto argmax = torch::zeros(
      {Nq, Nb, Lq},
      torch::TensorOptions().dtype(torch::kInt32).device(queries.device()));
  const int64_t total_pairs = Nq * Nb;
  if (total_pairs == 0) {
    return std::make_tuple(scores, argmax);
  }

  // Estimate average Ld for tg-size picking. Read document_offsets[-1] to get
  // total_d_tokens cheaply (already on device — pull last entry to CPU).
  const int64_t total_d_tokens = documents.size(0);
  const int64_t avg_ld = Nb > 0 ? (total_d_tokens + Nb - 1) / Nb : 1;

  const size_t elem_size = (size_t)queries.element_size();
  const size_t q_bytes = (size_t)Lq * (size_t)dim * elem_size;
  const size_t maxes_padded = ((size_t)Lq + 3u) & ~(size_t)3u;
  const size_t maxes_bytes = maxes_padded * sizeof(float);
  const size_t arg_bytes   = maxes_padded * sizeof(int);
  const size_t prelim = maxes_bytes + arg_bytes + q_bytes;

  NSUInteger dev_tg_max = [get_metal_device() maxThreadgroupMemoryLength];
  NSUInteger mem_budget = std::min(kPairMaxThreadgroupMemBytes, dev_tg_max);
  const bool mma_ok = (dim % 8 == 0);
  const NSUInteger desired_tg = pick_pair_tg_size(
      Lq, avg_ld, kPairMaxThreadgroupSize, mma_ok);
  const size_t mma_scratch =
      (size_t)mma_scratch_bytes_for_tg(desired_tg);
  TORCH_CHECK(
      prelim + mma_scratch <= mem_budget,
      "Lq * dim too large for the contrastive Metal kernel "
      "(threadgroup memory required >= ",
      prelim + mma_scratch, " bytes, budget = ", mem_budget, " bytes).");

  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", kname);

    NSUInteger tg_size = pick_pair_tg_size(
        Lq, avg_ld, pso.maxTotalThreadsPerThreadgroup, mma_ok);
    const NSUInteger tg_bytes =
        (NSUInteger)(prelim + mma_scratch_bytes_for_tg(tg_size));
    TORCH_CHECK(tg_bytes <= dev_tg_max,
                "contrastive argmax threadgroup memory (", tg_bytes,
                " bytes) exceeds device limit (", dev_tg_max, " bytes)");

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(document_offsets)
                  offset:document_offsets.storage_offset() * document_offsets.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(scores)
                  offset:scores.storage_offset() * scores.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(argmax)
                  offset:argmax.storage_offset() * argmax.element_size()
                 atIndex:4];

      uint dim_u = (uint)dim;
      uint Lq_u  = (uint)Lq;
      uint Nb_u  = (uint)Nb;
      [encoder setBytes:&dim_u length:sizeof(uint) atIndex:5];
      [encoder setBytes:&Lq_u  length:sizeof(uint) atIndex:6];
      [encoder setBytes:&Nb_u  length:sizeof(uint) atIndex:7];

      [encoder setThreadgroupMemoryLength:tg_bytes atIndex:0];

      MTLSize grid = MTLSizeMake((NSUInteger)total_pairs, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return std::make_tuple(scores, argmax);
}

torch::Tensor maxsim_contrastive_forward(torch::Tensor queries,
                                         torch::Tensor documents,
                                         torch::Tensor document_offsets) {
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              document_offsets.device() == queries.device(),
              "all contrastive inputs must share a device");
  TORCH_CHECK(queries.dim() == 3, "queries must be [Nq, Lq, D]");
  TORCH_CHECK(documents.dim() == 2,
              "documents must be [total_d_tokens, D] (packed)");
  TORCH_CHECK(document_offsets.dim() == 1, "document_offsets must be 1-D [Nb+1]");
  TORCH_CHECK(queries.size(2) == documents.size(1),
              "queries.D must match documents.D");
  TORCH_CHECK(queries.size(2) > 0, "embedding dim must be > 0");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");

  const char *kname = contrastive_kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

  queries    = queries.contiguous();
  documents  = documents.contiguous();
  document_offsets = document_offsets.contiguous().to(torch::kInt32);

  const int64_t Nq  = queries.size(0);
  const int64_t Lq  = queries.size(1);
  const int64_t dim = queries.size(2);
  const int64_t Nb  = document_offsets.size(0) - 1;
  TORCH_CHECK(Nb >= 0, "document_offsets must have at least 1 element");
  TORCH_CHECK(Lq > 0, "Lq must be > 0");

  auto scores = torch::zeros(
      {Nq, Nb},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));
  const int64_t total_pairs = Nq * Nb;
  if (total_pairs == 0) {
    return scores;
  }

  const int64_t total_d_tokens = documents.size(0);
  const int64_t avg_ld = Nb > 0 ? (total_d_tokens + Nb - 1) / Nb : 1;

  const size_t elem_size = (size_t)queries.element_size();
  const size_t q_bytes = (size_t)Lq * (size_t)dim * elem_size;
  const size_t maxes_padded = ((size_t)Lq + 3u) & ~(size_t)3u;
  const size_t maxes_bytes = maxes_padded * sizeof(float);
  const size_t prelim = maxes_bytes + q_bytes;

  NSUInteger dev_tg_max = [get_metal_device() maxThreadgroupMemoryLength];
  NSUInteger mem_budget = std::min(kPairMaxThreadgroupMemBytes, dev_tg_max);
  const bool mma_ok = (dim % 8 == 0);
  const NSUInteger desired_tg = pick_pair_tg_size(
      Lq, avg_ld, kPairMaxThreadgroupSize, mma_ok);
  const size_t mma_scratch =
      (size_t)mma_scratch_bytes_for_tg(desired_tg);
  TORCH_CHECK(
      prelim + mma_scratch <= mem_budget,
      "Lq * dim too large for the contrastive Metal kernel "
      "(threadgroup memory required >= ",
      prelim + mma_scratch, " bytes, budget = ", mem_budget, " bytes).");

  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", kname);

    NSUInteger tg_size = pick_pair_tg_size(
        Lq, avg_ld, pso.maxTotalThreadsPerThreadgroup, mma_ok);
    const NSUInteger tg_bytes =
        (NSUInteger)(prelim + mma_scratch_bytes_for_tg(tg_size));
    TORCH_CHECK(tg_bytes <= dev_tg_max,
                "contrastive threadgroup memory (", tg_bytes,
                " bytes) exceeds device limit (", dev_tg_max, " bytes)");

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(document_offsets)
                  offset:document_offsets.storage_offset() * document_offsets.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(scores)
                  offset:scores.storage_offset() * scores.element_size()
                 atIndex:3];

      uint dim_u = (uint)dim;
      uint Lq_u  = (uint)Lq;
      uint Nb_u  = (uint)Nb;
      [encoder setBytes:&dim_u length:sizeof(uint) atIndex:4];
      [encoder setBytes:&Lq_u  length:sizeof(uint) atIndex:5];
      [encoder setBytes:&Nb_u  length:sizeof(uint) atIndex:6];

      [encoder setThreadgroupMemoryLength:tg_bytes atIndex:0];

      MTLSize grid = MTLSizeMake((NSUInteger)total_pairs, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return scores;
}

std::tuple<torch::Tensor, torch::Tensor> maxsim_contrastive_backward(
    torch::Tensor dscore,
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor argmax) {
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              document_offsets.device() == queries.device() &&
              argmax.device() == queries.device() &&
              dscore.device() == queries.device(),
              "all contrastive backward inputs must share a device");
  TORCH_CHECK(queries.dim() == 3, "queries must be [Nq, Lq, D]");
  TORCH_CHECK(documents.dim() == 2, "documents must be [total_d_tokens, D]");
  TORCH_CHECK(queries.scalar_type() == documents.scalar_type(),
              "queries and documents must have the same dtype");
  TORCH_CHECK(dscore.scalar_type() == at::ScalarType::Float,
              "dscore must be fp32");
  TORCH_CHECK(argmax.scalar_type() == at::ScalarType::Int,
              "argmax must be int32");

  const char *kname =
      contrastive_backward_kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

  queries    = queries.contiguous();
  documents  = documents.contiguous();
  document_offsets = document_offsets.contiguous().to(torch::kInt32);
  argmax     = argmax.contiguous();
  dscore     = dscore.contiguous();

  const int64_t Nq  = queries.size(0);
  const int64_t Lq  = queries.size(1);
  const int64_t dim = queries.size(2);
  const int64_t Nb  = document_offsets.size(0) - 1;
  const int64_t total_pairs = Nq * Nb;

  TORCH_CHECK(argmax.numel() == total_pairs * Lq,
              "argmax must have shape [Nq * Nb, Lq] (or equivalent)");
  TORCH_CHECK(dscore.numel() == total_pairs,
              "dscore must have shape [Nq, Nb] (or [Nq * Nb])");

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto dqueries   = torch::zeros_like(queries, fp32_opts);
  auto ddocuments = torch::zeros_like(documents, fp32_opts);
  if (total_pairs == 0) return std::make_tuple(dqueries, ddocuments);

  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", kname);

    NSUInteger tg_size = 4 * kSimdWidth;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup) {
      tg_size = pso.maxTotalThreadsPerThreadgroup;
    }

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(document_offsets)
                  offset:document_offsets.storage_offset() * document_offsets.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(argmax)
                  offset:argmax.storage_offset() * argmax.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(dscore)
                  offset:dscore.storage_offset() * dscore.element_size()
                 atIndex:4];
      [encoder setBuffer:getMTLBufferStorage(dqueries)
                  offset:dqueries.storage_offset() * dqueries.element_size()
                 atIndex:5];
      [encoder setBuffer:getMTLBufferStorage(ddocuments)
                  offset:ddocuments.storage_offset() * ddocuments.element_size()
                 atIndex:6];

      uint dim_u = (uint)dim;
      uint Lq_u  = (uint)Lq;
      uint Nb_u  = (uint)Nb;
      [encoder setBytes:&dim_u length:sizeof(uint) atIndex:7];
      [encoder setBytes:&Lq_u  length:sizeof(uint) atIndex:8];
      [encoder setBytes:&Nb_u  length:sizeof(uint) atIndex:9];

      MTLSize grid = MTLSizeMake((NSUInteger)total_pairs, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return std::make_tuple(dqueries, ddocuments);
}

// =============================================================================
// Packed forward + argmax (training-mode companion to `maxsim_forward`).
// Scalar kernel; WMMA argmax for packed is post-V2.
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor> maxsim_packed_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor query_offsets,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor pair_query_ids,
    torch::Tensor pair_document_ids,
    int64_t max_q_len) {
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
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

  const char *kname =
      packed_argmax_kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

  queries           = queries.contiguous();
  documents         = documents.contiguous();
  query_offsets     = query_offsets.contiguous().to(torch::kInt32);
  document_offsets  = document_offsets.contiguous().to(torch::kInt32);
  pair_query_ids    = pair_query_ids.contiguous().to(torch::kInt32);
  pair_document_ids = pair_document_ids.contiguous().to(torch::kInt32);

  const int64_t dim       = queries.size(1);
  const int64_t num_pairs = pair_query_ids.size(0);

  auto scores = torch::empty(
      {num_pairs},
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device()));
  auto argmax = torch::zeros(
      {num_pairs, max_q_len},
      torch::TensorOptions().dtype(torch::kInt32).device(queries.device()));
  if (num_pairs == 0) return std::make_tuple(scores, argmax);

  const size_t tg_bytes =
      (size_t)max_q_len * sizeof(float) + (size_t)max_q_len * sizeof(int);
  NSUInteger dev_tg_max = [get_metal_device() maxThreadgroupMemoryLength];
  TORCH_CHECK(tg_bytes <= dev_tg_max,
              "packed argmax threadgroup memory (", tg_bytes,
              " bytes) exceeds device limit (", dev_tg_max, ")");

  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", kname);

    NSUInteger tg_size = 4 * kSimdWidth;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup) {
      tg_size = pso.maxTotalThreadsPerThreadgroup;
    }

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(query_offsets)
                  offset:query_offsets.storage_offset() *
                         query_offsets.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(document_offsets)
                  offset:document_offsets.storage_offset() *
                         document_offsets.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(pair_query_ids)
                  offset:pair_query_ids.storage_offset() *
                         pair_query_ids.element_size()
                 atIndex:4];
      [encoder setBuffer:getMTLBufferStorage(pair_document_ids)
                  offset:pair_document_ids.storage_offset() *
                         pair_document_ids.element_size()
                 atIndex:5];
      [encoder setBuffer:getMTLBufferStorage(scores)
                  offset:scores.storage_offset() * scores.element_size()
                 atIndex:6];
      [encoder setBuffer:getMTLBufferStorage(argmax)
                  offset:argmax.storage_offset() * argmax.element_size()
                 atIndex:7];

      uint dim_u       = (uint)dim;
      uint max_q_len_u = (uint)max_q_len;
      [encoder setBytes:&dim_u       length:sizeof(uint) atIndex:8];
      [encoder setBytes:&max_q_len_u length:sizeof(uint) atIndex:9];

      [encoder setThreadgroupMemoryLength:(NSUInteger)tg_bytes atIndex:0];

      MTLSize grid = MTLSizeMake((NSUInteger)num_pairs, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return std::make_tuple(scores, argmax);
}

// =============================================================================
// Packed backward. Atomic-add accumulation into fp32 dq/dd.
// =============================================================================

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
  TORCH_CHECK(queries.device().is_mps(), "queries must be an MPS tensor");
  TORCH_CHECK(documents.device() == queries.device() &&
              dscore.device() == queries.device() &&
              argmax.device() == queries.device() &&
              query_offsets.device() == queries.device() &&
              document_offsets.device() == queries.device() &&
              pair_query_ids.device() == queries.device() &&
              pair_document_ids.device() == queries.device(),
              "all packed backward inputs must share a device");
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

  const char *kname =
      packed_backward_kernel_name_for_dtype(queries.scalar_type());
  TORCH_CHECK(kname != nullptr,
              "queries/documents dtype must be float32, float16, or bfloat16");

  queries           = queries.contiguous();
  documents         = documents.contiguous();
  query_offsets     = query_offsets.contiguous().to(torch::kInt32);
  document_offsets  = document_offsets.contiguous().to(torch::kInt32);
  pair_query_ids    = pair_query_ids.contiguous().to(torch::kInt32);
  pair_document_ids = pair_document_ids.contiguous().to(torch::kInt32);
  argmax            = argmax.contiguous();
  dscore            = dscore.contiguous();

  const int64_t dim       = queries.size(1);
  const int64_t num_pairs = pair_query_ids.size(0);

  TORCH_CHECK(argmax.numel() == num_pairs * max_q_len,
              "argmax must have shape [num_pairs, max_q_len]");
  TORCH_CHECK(dscore.numel() == num_pairs,
              "dscore must have shape [num_pairs]");

  auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
  auto dqueries   = torch::zeros_like(queries, fp32_opts);
  auto ddocuments = torch::zeros_like(documents, fp32_opts);
  if (num_pairs == 0) return std::make_tuple(dqueries, ddocuments);

  @autoreleasepool {
    id<MTLComputePipelineState> pso = get_pso(kname);
    TORCH_CHECK(pso, "Failed to load Metal function ", kname);

    NSUInteger tg_size = 4 * kSimdWidth;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup) {
      tg_size = pso.maxTotalThreadsPerThreadgroup;
    }

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];

      [encoder setBuffer:getMTLBufferStorage(queries)
                  offset:queries.storage_offset() * queries.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(query_offsets)
                  offset:query_offsets.storage_offset() *
                         query_offsets.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(documents)
                  offset:documents.storage_offset() * documents.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(document_offsets)
                  offset:document_offsets.storage_offset() *
                         document_offsets.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(pair_query_ids)
                  offset:pair_query_ids.storage_offset() *
                         pair_query_ids.element_size()
                 atIndex:4];
      [encoder setBuffer:getMTLBufferStorage(pair_document_ids)
                  offset:pair_document_ids.storage_offset() *
                         pair_document_ids.element_size()
                 atIndex:5];
      [encoder setBuffer:getMTLBufferStorage(argmax)
                  offset:argmax.storage_offset() * argmax.element_size()
                 atIndex:6];
      [encoder setBuffer:getMTLBufferStorage(dscore)
                  offset:dscore.storage_offset() * dscore.element_size()
                 atIndex:7];
      [encoder setBuffer:getMTLBufferStorage(dqueries)
                  offset:dqueries.storage_offset() * dqueries.element_size()
                 atIndex:8];
      [encoder setBuffer:getMTLBufferStorage(ddocuments)
                  offset:ddocuments.storage_offset() * ddocuments.element_size()
                 atIndex:9];

      uint dim_u       = (uint)dim;
      uint max_q_len_u = (uint)max_q_len;
      [encoder setBytes:&dim_u       length:sizeof(uint) atIndex:10];
      [encoder setBytes:&max_q_len_u length:sizeof(uint) atIndex:11];

      MTLSize grid = MTLSizeMake((NSUInteger)num_pairs, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      [encoder endEncoding];
      torch::mps::commit();
    });
  }

  return std::make_tuple(dqueries, ddocuments);
}
