/**
 * \file topk_v2.cuh
 * \brief TopK kernel for DeepSeek v4.
 * Adapted from
 * 1:
 *   https://github.com/vllm-project/vllm/blob/a8c6ee9b787d273916206a29b77feebadb80c368/csrc/persistent_topk.cuh
 * 2:
 *   https://github.com/flashinfer-ai/flashinfer/blob/c2b4db2b1a84448d802f0e6ac445243312bd6a4c/include/flashinfer/topk.cuh
 * DarkSharpness never took a detailed look at these 2 implementation, but his claude code did.
 * So we add credit to the reference implementations.
 */
#include <sgl_kernel/kvduo_cache_observation.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <iterator>

namespace {

namespace impl = device::topk;
using impl::TopKProblem;

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;
using Cluster = impl::TopKCluster<8>;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)
#define CLUSTER_TOPK_KERNEL TOPK_KERNEL __cluster_dims__(1, kClusterSize, 1)

constexpr uint32_t kClusterFloor = 65536;
constexpr uint32_t kClusterMaxBatch = 512;
constexpr uint32_t kNumPersistentClusters = 15 * kOccupancy;

/// Metadata tensor rows (each 8 B / 2 int32). Row 0 is the global plan result;
/// rows 1..N are the (batch_id, seq_len) of items routed to the cluster pool.
struct alignas(8) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;  // N = number of items routed to the cluster pool
};
struct alignas(8) PlanItem {
  uint32_t batch_id;
  uint32_t seq_len;
};
static_assert(sizeof(GlobalMetadata) == 2 * sizeof(int32_t) && sizeof(PlanItem) == sizeof(GlobalMetadata));

struct KVDuoFullParams {
  int32_t* output = nullptr;
  const int64_t* requests = nullptr;
  const int64_t* lookup = nullptr;
  const int64_t* mapping = nullptr;
  int64_t* touches = nullptr;
  int32_t* status = nullptr;
  int32_t* stats = nullptr;
  int32_t* growth = nullptr;
  const int32_t* observe = nullptr;
  const int32_t* real_requests = nullptr;
  const int64_t* clock_base = nullptr;
  float* prefill_ema = nullptr;
  int64_t* prefill_steps = nullptr;
  int64_t lookup_stride = 0;
  int64_t context = 0;
  int64_t request_count = 0;
  int64_t logical_size = 0;
  int64_t location_base = 0;
  int64_t layer = 0;
  int page_bits = 0;
};

struct TopKLaunchParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;      // optional raw (pre-transform) indices output; nullptr if unused
  int64_t* __restrict__ touch_clock;
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;
  KVDuoFullParams full;
  uint32_t cluster_floor;  // seq_len > this routes to the cluster path (batch-aware, host-set)

  SGL_DEVICE const GlobalMetadata& global() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE uint32_t cluster_threshold() const {
    return global().cluster_threshold;
  }
  SGL_DEVICE const PlanItem& item(uint32_t i) const {
    return metadata[1 + i];
  }
  SGL_DEVICE int32_t* get_output_ptr(uint32_t batch_id) const {
    return page_indices + batch_id * static_cast<int64_t>(topk);
  }
  template <bool RawOnly>
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        .in = scores + batch_id * score_stride,
        .out = page_indices + batch_id * k,
        .raw_out = raw_indices != nullptr ? raw_indices + batch_id * k : nullptr,
        .page_table = RawOnly ? nullptr : page_table + batch_id * page_table_stride,
        .topk = topk,
        .seq_len = seq_len,
        .page_bits = page_bits,
    };
  }
  template <bool RawOnly>
  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    return this->problem<RawOnly>(batch_id, static_cast<uint32_t>(seq_lens[batch_id]));
  }
};

SGL_DEVICE void resolve_full(const TopKLaunchParams& params, uint32_t row) {
  const auto full = params.full;
  int32_t* output = full.output + row * static_cast<int64_t>(params.topk);
  if (row >= static_cast<uint32_t>(full.real_requests[0])) {
    for (uint32_t i = threadIdx.x; i < params.topk; i += kBlockSize)
      output[i] = -1;
    return;
  }
  const int64_t rid = full.requests[row];
  if (rid < 0 || rid >= full.request_count) {
    if (threadIdx.x == 0) __assert_fail("KVDuo FULL TopK invalid request slot", __FILE__, __LINE__, __func__);
    return;
  }
  __shared__ int hits;
  if (threadIdx.x == 0) {
    hits = 0;
    full.status[rid] = 0;
    if (full.observe[0]) full.growth[rid] = 0;
  }
  __syncthreads();
  const int32_t* raw = params.get_output_ptr(row);
  const int64_t seq_len = params.seq_lens[row];
  const auto stamp = static_cast<unsigned long long>(full.clock_base[0] + full.layer + 1);
  int local_hits = 0;
  for (uint32_t i = threadIdx.x; i < params.topk; i += kBlockSize) {
    const int32_t token = raw[i];
    output[i] = -1;
    if (token < 0 || token >= seq_len) continue;
    const int64_t logical = token < full.context ? full.lookup[rid * full.lookup_stride + token] : -1;
    const int64_t location = logical >= 0 && logical < full.logical_size ? full.mapping[logical] : 0;
    if (location <= 0) {
      atomicExch(full.status + rid, 1);
      continue;
    }
    const unsigned peers =
        __match_any_sync(__activemask(), static_cast<unsigned long long>(location >> full.page_bits));
    if (threadIdx.x % 32 == __ffs(peers) - 1) {
      atomicMax(reinterpret_cast<unsigned long long*>(full.touches + logical), stamp);
    }
    output[i] = static_cast<int32_t>(location + full.location_base);
    ++local_hits;
  }
  for (int offset = 16; offset > 0; offset /= 2)
    local_hits += __shfl_down_sync(0xffffffffu, local_hits, offset);
  if (threadIdx.x % 32 == 0) atomicAdd(&hits, local_hits);
  __syncthreads();
  if (full.status[rid] != 0) {
    if (threadIdx.x == 0) __assert_fail("KVDuo FULL TopK missing resident KV", __FILE__, __LINE__, __func__);
    return;
  }
  if (threadIdx.x == 0) {
    atomicAdd(full.stats + rid * 2 + 1, hits);
    if (full.observe[0])
      sglang::kvduo_update_prefill_ema(0, 0, hits, full.prefill_ema + rid * 2, full.prefill_steps[rid]);
  }
}

/**
 * \brief Persistent cluster kernel for the long items. It will handle long inputs.
 * The short items are handled by the separate topk_kernel.
 */
template <bool kPDL, bool RawOnly>
CLUSTER_TOPK_KERNEL void topk_persistent_cluster_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  __shared__ impl::MaxSmem<Cluster::Smem> smem;
  const uint32_t num_cluster_items = params.global().num_cluster_items;
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
#pragma unroll 1
  for (uint32_t w = blockIdx.x; w < num_cluster_items; w += kNumPersistentClusters) {
    const auto it = params.item(w);
    const auto problem = params.problem<RawOnly>(it.batch_id, it.seq_len);
    Cluster::forward<false>(problem, &smem);
    __syncthreads();
  }
}

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      __builtin_assume(tx < kMaxTopK);
      f(tx, i);
    }
  }
}

template <bool kPDL, bool RawOnly>
SGL_DEVICE void trivial_transform(const TopKProblem& problem) {
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t) {
    const int32_t raw = tx < problem.seq_len ? static_cast<int32_t>(tx) : -1;
    if constexpr (RawOnly) {
      problem.out[tx] = raw;
    } else {
      problem.transform_output(tx, raw);
    }
  });
}

template <bool RawOnly>
SGL_DEVICE void problem_transform(TopKProblem& problem, int32_t* output_ptr) {
  if constexpr (RawOnly) {
    // Persistent clusters already published their raw selection to this bank.
    if (problem.out == output_ptr) return;
  }
  static_assert(kMaxTopK % kBlockSize == 0);
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
  int32_t source_index[kNumElems];
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { source_index[i] = problem.out[tx]; });
  problem.out = output_ptr;
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) {
    if constexpr (RawOnly) {
      problem.out[tx] = source_index[i];
    } else {
      problem.transform_output(tx, source_index[i]);
    }
  });
}

/**
 * \brief Main kernel for the short items and epilogue of long items.
 * \tparam kPDL whether to use PDL to synchronize with the cluster kernel (if any)
 * \tparam kLevel:
 * - Level 0: max_seq_len <= 8192           -> trivial + register<2>
 * - Level 1: max_seq_len <= 16384          -> trivial + register<4>
 * - Level 2: max_seq_len <= cluster_floor  -> trivial + register<4> + streaming
 * - Level 3: max_seq_len > cluster_floor   -> + epilogue process of cluster path
 */
template <bool kPDL, int kLevel, bool RawOnly, bool Full = false>
TOPK_KERNEL void topk_main_kernel(const __grid_constant__ TopKLaunchParams params) {
  if constexpr (!Full) device::enable_smem_spilling();
  if constexpr (RawOnly && !Full) {
    // The persistent route increments here, not in its preceding cluster pool.
    // The following non-PDL resolver waits for every TopK CTA to finish.
    if (blockIdx.x == 0 && threadIdx.x == 0 && params.touch_clock != nullptr) ++params.touch_clock[0];
  }
  auto problem = params.problem<RawOnly>(blockIdx.x);
  constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= problem.topk) {
    trivial_transform<kPDL, RawOnly>(problem);
    if constexpr (Full) resolve_full(params, blockIdx.x);
    return;
  }
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  constexpr bool kHandleCluster = (kLevel == 3);
  // non-trivial path: dispatch based on level and seq_len
  const auto cluster_threshold = kHandleCluster ? params.cluster_threshold() : kU32Max;
  if constexpr (kLevel == 0) {
    __builtin_assume(problem.seq_len <= kReg2MaxSeqLen);
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    __builtin_assume(problem.seq_len <= kReg4MaxSeqLen);
    Register4::forward<kPDL>(problem, &smem);  // max_seq_len <= 16384 guarantees seq <= 16384
  } else {
    static_assert(kLevel == 2 || kLevel == 3, "we only support level = 0,1,2,3 now");
    // if using cluster, we can delay the PDL wait
    constexpr bool kPDLEarly = kPDL && !kHandleCluster;
    constexpr bool kPDLFinal = kPDL && kHandleCluster;
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= cluster_threshold) {
      Streaming::forward<kPDLEarly>(problem, &smem);
    } else {  // cluster path do nothing here
      problem.out = params.get_output_ptr(blockIdx.x);
    }
    device::PDLWaitPrimary<kPDLFinal>();
  }

  // page-table transform pass (gathers kept out of the hot scatter loop),
  // then trigger the dependent kernel only after the full output is written.
  device::PDLTriggerSecondary<kPDL>();
  __syncthreads();
  problem_transform<RawOnly>(problem, params.get_output_ptr(blockIdx.x));
  if constexpr (Full) resolve_full(params, blockIdx.x);
}

template <bool kPDL, bool RawOnly, bool Full = false>
CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __grid_constant__ TopKLaunchParams params) {
  if constexpr (!Full) device::enable_smem_spilling();
  if constexpr (RawOnly && !Full) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && params.touch_clock != nullptr)
      ++params.touch_clock[0];
  }
  auto problem = params.problem<RawOnly>(blockIdx.x);
  __shared__ impl::MaxSmem<Streaming::Smem, Cluster::Smem> smem;
  if (problem.seq_len <= problem.topk) {
    trivial_transform<kPDL, RawOnly>(problem);
    if constexpr (Full) {
      if (blockIdx.y == blockIdx.x % kClusterSize) resolve_full(params, blockIdx.x);
    }
    return;
  }
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  // randomly elect one worker rank to avoid workload imbalance
  const auto worker_rank = blockIdx.x % kClusterSize;

  // for small batch, we will fuse in the cluster case
  if (problem.seq_len <= kReg4MaxSeqLen) {
    if (blockIdx.y == worker_rank) Register4::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= params.cluster_floor) {
    if (blockIdx.y == worker_rank) Streaming::forward<kPDL>(problem, &smem);
  } else {
    auto cluster = cooperative_groups::this_cluster();
    // The mapped alias stays in a copy: the elected rank reads the very same
    // bytes back through `topk_indices` below, and letting a shared::cluster
    // address reach the `problem.out` that problem_transform loads makes cicc
    // segfault on CUDA 13.x (issue #32830).
    auto peer_problem = problem;
    peer_problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
    Cluster::forward<kPDL>(peer_problem, &smem);  // write to peer's output shared memory
    cluster.sync();
  }

  device::PDLWaitPrimary<kPDL>();
  __syncthreads();
  if (blockIdx.y == worker_rank) {
    problem_transform<RawOnly>(problem, params.get_output_ptr(blockIdx.x));
    if constexpr (Full) resolve_full(params, blockIdx.x);
  }
}

// --- Plan: choose cluster_threshold from the seq_len distribution -----------
__global__ __launch_bounds__(kBlockSize, 1) void topk_plan(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,  // [0]=GlobalMetadata, [1+i]=PlanItem
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
  // Candidate (threshold T_j, cap_j) pairs, T strictly increasing. The plan lowers
  // cluster_threshold to T_j while #(items with seq_len > T_j) <= cap_j, so cap_j
  // bounds how many long items go to the persistent pool. The pool runs N items in
  // ceil(N / kNumPersistentClusters) waves; the longer the seq the more waves pay
  // off (streaming a single block over a long item is very slow), so cap_j is the
  // measured cluster-vs-streaming crossover (B200, occ2) and GROWS with T -- a flat
  // cap = pool size only fits the shortest (~98K, one-wave) bucket. (Plan is tunable.)
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
      {65536, 30},    // (65536,98304]:    ~1 pool wave, streams beyond 30
      {98304, 48},    // (98304,131072]
      {131072, 60},   // (131072,196608]
      {196608, 80},   // (196608,262144]
      {262144, 112},  // (262144,393216]
      {393216, 128},  // (393216,inf):     longest -- worth many pool waves; a top
                      // threshold here lets overloaded ~280-393K batches still stream
  };
  constexpr uint32_t kNumCandidates = std::size(kCandidates);
  static_assert(kCandidates[0].threshold == kClusterFloor);

  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;
  __shared__ uint32_t s_count;

  const auto tx = threadIdx.x;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  if (tx == 0) s_count = 0;
  __syncthreads();

  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) atomicAdd(&s_counts[count - 1], 1);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kCandidates[kNumCandidates - 1].threshold;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];  // # items with seq_len > kCandidates[j].threshold
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  const auto cluster_threshold = max(s_threshold, kClusterFloor);

  // Compact items with seq_len > threshold into metadata[1..N]: their batch ids
  // are the work list the persistent cluster pool fetches.
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl};
    }
  }
  __syncthreads();
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {.cluster_threshold = cluster_threshold, .num_cluster_items = s_count};
  }
}

struct TopKKernel {
  static void plan(  //
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView metadata,
      const uint32_t static_cluster_threshold) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({Bp1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto device = device_.unwrap();
    LaunchKernel(1, kBlockSize, device)(  //
        topk_plan,
        static_cast<const uint32_t*>(seq_lens.data_ptr()),
        static_cast<PlanItem*>(metadata.data_ptr()),
        batch_size,
        static_cluster_threshold);
  }

  static void transform_impl(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::TensorView metadata,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices,
      const bool raw_only,
      const tvm::ffi::Optional<tvm::ffi::TensorView> touch_clock,
      KVDuoFullParams full = {}) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, -1})  // page_table
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_table);
    TensorMatcher({B, K})  // page_indices
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);
    TensorMatcher({Bp1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device_).verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }
    RuntimeCheck(!raw_only || raw_indices_ptr != nullptr, "TopK raw-only mode requires raw indices");
    int64_t* clock_ptr = nullptr;
    if (touch_clock.has_value()) {
      RuntimeCheck(raw_only, "FULL clock requires raw-only TopK");
      TensorMatcher({1}).with_dtype<int64_t>().with_device(device_).verify(touch_clock.value());
      clock_ptr = static_cast<int64_t*>(touch_clock.value().data_ptr());
    }
    const auto raw_start = reinterpret_cast<uintptr_t>(raw_indices_ptr);
    const auto mapped_start = reinterpret_cast<uintptr_t>(page_indices.data_ptr());
    const auto output_bytes = B.unwrap() * K.unwrap() * sizeof(int32_t);
    RuntimeCheck(
        !raw_only || raw_start >= mapped_start + output_bytes || mapped_start >= raw_start + output_bytes,
        "TopK raw-only output must not alias mapped output");

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto max_seq_len = static_cast<uint32_t>(L.unwrap());
    const auto device = device_.unwrap();

    // The fused kernel runs one 8-block cluster per batch element, and B200 fits one
    // wave of exactly 15 such clusters (occ2). For batch <= 15 it stays latency-bound,
    // so the 8-way split beats streaming from a much lower seq (measured crossover
    // ~36-40K); batch 16 spills into a 2nd wave (+25%) and keeps the 64K floor.
    // The floor is chosen on the host per launch.
    constexpr uint32_t kClusterFloorSmall = 32768;
    constexpr uint32_t kSmallBatchLowFloor = 15;
    const auto params = TopKLaunchParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = raw_only ? raw_indices_ptr : static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_only ? nullptr : raw_indices_ptr,
        .touch_clock = clock_ptr,
        .metadata = static_cast<const PlanItem*>(metadata.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .topk = topk,
        .page_bits = page_bits,
        .full = full,
        .cluster_floor = (batch_size <= kSmallBatchLowFloor) ? kClusterFloorSmall : kClusterFloor,
    };

    const bool use_cluster = (max_seq_len > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
    constexpr bool kUsePDL = true;
    auto launch = [&]<bool RawOnly, bool Full>() {
      if (use_cluster) {
        if (batch_size <= kNumPersistentClusters) {
          LaunchKernel({batch_size, kClusterSize}, kBlockSize, device)
              .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
              .launch(topk_small_batch_kernel<kUsePDL, RawOnly, Full>, params);
        } else {
          const uint32_t num_clusters = std::min(batch_size, kNumPersistentClusters);
          LaunchKernel({num_clusters, kClusterSize}, kBlockSize, device)
              .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
              .launch(topk_persistent_cluster_kernel<kUsePDL, RawOnly>, params);
          LaunchKernel(batch_size, kBlockSize, device)
              .config({.use_pdl = kUsePDL})
              .launch(topk_main_kernel<kUsePDL, /*kLevel=*/3, RawOnly, Full>, params);
        }
      } else if (max_seq_len <= kReg2MaxSeqLen) {
        LaunchKernel(batch_size, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_main_kernel<kUsePDL, /*kLevel=*/0, RawOnly, Full>, params);
      } else if (max_seq_len <= kReg4MaxSeqLen) {
        LaunchKernel(batch_size, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_main_kernel<kUsePDL, /*kLevel=*/1, RawOnly, Full>, params);
      } else {
        LaunchKernel(batch_size, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_main_kernel<kUsePDL, /*kLevel=*/2, RawOnly, Full>, params);
      }
    };
    if (full.output != nullptr) {
      RuntimeCheck(raw_only && clock_ptr == nullptr, "FULL TopK requires reserved clocks and raw selection");
      launch.template operator()<true, true>();
    } else if (raw_only) {
      launch.template operator()<true, false>();
    } else {
      launch.template operator()<false, false>();
    }
  }
  static void transform(
      tvm::ffi::TensorView scores,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView page_indices,
      uint32_t page_size,
      tvm::ffi::TensorView metadata,
      tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices,
      bool raw_only,
      tvm::ffi::Optional<tvm::ffi::TensorView> clock) {
    transform_impl(scores, seq_lens, page_table, page_indices, page_size, metadata, raw_indices, raw_only, clock);
  }

  static void transform_full(
      tvm::ffi::TensorView scores,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView page_indices,
      uint32_t page_size,
      tvm::ffi::TensorView metadata,
      tvm::ffi::TensorView raw_indices,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView requests,
      tvm::ffi::TensorView lookup,
      tvm::ffi::TensorView mapping,
      tvm::ffi::TensorView touches,
      tvm::ffi::TensorView status,
      tvm::ffi::TensorView stats,
      tvm::ffi::TensorView growth,
      tvm::ffi::TensorView observe,
      tvm::ffi::TensorView real_requests,
      tvm::ffi::TensorView clock_base,
      tvm::ffi::TensorView prefill_ema,
      tvm::ffi::TensorView prefill_steps,
      int64_t layer,
      int64_t layer_stride,
      int64_t full_page_size) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto B = SymbolicSize{"batch"};
    auto K = SymbolicSize{"topk"};
    auto R = SymbolicSize{"request_count"};
    auto N = SymbolicSize{"logical_size"};
    auto C = SymbolicSize{"context"};
    auto S = SymbolicSize{"lookup_stride"};
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(raw_indices).verify(output);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device).verify(requests);
    TensorMatcher({R, C}).with_strides({S, 1}).with_dtype<int64_t>().with_device(device).verify(lookup);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(mapping).verify(touches);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device).verify(status).verify(growth);
    TensorMatcher({R, 2}).with_dtype<int32_t>().with_device(device).verify(stats);
    TensorMatcher({R, 2}).with_dtype<float>().with_device(device).verify(prefill_ema);
    TensorMatcher({R}).with_dtype<int64_t>().with_device(device).verify(prefill_steps);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(observe).verify(real_requests);
    TensorMatcher({1}).with_dtype<int64_t>().with_device(device).verify(clock_base);
    RuntimeCheck(
        full_page_size > 0 && std::has_single_bit(static_cast<uint64_t>(full_page_size)), "Invalid FULL page size");
    RuntimeCheck(layer >= 0 && layer_stride >= 0, "Invalid FULL layer geometry");
    const auto raw_begin = reinterpret_cast<uintptr_t>(raw_indices.data_ptr());
    const auto out_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
    const auto bytes = B.unwrap() * K.unwrap() * sizeof(int32_t);
    RuntimeCheck(
        raw_begin >= out_begin + bytes || out_begin >= raw_begin + bytes, "FULL output must not alias raw selection");
    KVDuoFullParams full{
        .output = static_cast<int32_t*>(output.data_ptr()),
        .requests = static_cast<const int64_t*>(requests.data_ptr()),
        .lookup = static_cast<const int64_t*>(lookup.data_ptr()),
        .mapping = static_cast<const int64_t*>(mapping.data_ptr()),
        .touches = static_cast<int64_t*>(touches.data_ptr()),
        .status = static_cast<int32_t*>(status.data_ptr()),
        .stats = static_cast<int32_t*>(stats.data_ptr()),
        .growth = static_cast<int32_t*>(growth.data_ptr()),
        .observe = static_cast<const int32_t*>(observe.data_ptr()),
        .real_requests = static_cast<const int32_t*>(real_requests.data_ptr()),
        .clock_base = static_cast<const int64_t*>(clock_base.data_ptr()),
        .prefill_ema = static_cast<float*>(prefill_ema.data_ptr()),
        .prefill_steps = static_cast<int64_t*>(prefill_steps.data_ptr()),
        .lookup_stride = S.unwrap(),
        .context = C.unwrap(),
        .request_count = R.unwrap(),
        .logical_size = N.unwrap(),
        .location_base = layer * layer_stride,
        .layer = layer,
        .page_bits = static_cast<int>(std::countr_zero(static_cast<uint64_t>(full_page_size))),
    };
    transform_impl(
        scores, seq_lens, page_table, page_indices, page_size, metadata, raw_indices, true, std::nullopt, full);
  }
};

}  // namespace
