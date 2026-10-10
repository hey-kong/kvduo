#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <type_traits>
#ifdef USE_ROCM
#include <hipcub/hipcub.hpp>
namespace merge_cub = hipcub;
#else
#include <cub/block/block_scan.cuh>
namespace merge_cub = cub;
#endif

namespace {

template <int CAPACITY>
using KVDuoMergeSlot = std::conditional_t<(CAPACITY > INT16_MAX), int32_t, int16_t>;

template <int BLOCK_SIZE>
__device__ int
merge_prefix(int flag, typename merge_cub::BlockScan<int, BLOCK_SIZE>::TempStorage& storage, int& carry) {
  int offset, total;
  merge_cub::BlockScan<int, BLOCK_SIZE>(storage).ExclusiveSum(flag, offset, total);
  const int base = carry;
  __syncthreads();
  if (threadIdx.x == 0) carry = base + total;
  __syncthreads();
  return base + offset;
}

template <int CAPACITY, int BLOCK_SIZE>
__global__ void kvduo_hot_merge_plan_kernel(
    const int32_t* tokens,
    const int32_t* locations,
    const KVDuoMergeSlot<CAPACITY>* lru,
    int32_t* new_tokens,
    int32_t* new_locations,
    KVDuoMergeSlot<CAPACITY>* new_lru,
    int64_t* sources,
    int64_t* destinations,
    int32_t* workspace,
    int32_t* summary,
    int capacity,
    int k,
    int target_pages,
    int hash_size,
    int64_t padded_capacity) {
  using Scan = merge_cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int counts[8], kept[8], page_rank[8];
  __shared__ int carry, survivor_start, survivors;
  const int tid = threadIdx.x;
  const int pages = capacity / k;
  const int target = target_pages * k;
  int* retained = workspace;
  int* valid_rank = retained + capacity;
  int* available = valid_rank + capacity;
  int* new_slot = available + capacity;
  int* hash = new_slot + capacity;
  if (tid == 0) {
    summary[0] = summary[1] = summary[2] = 0;
    carry = 0;
  }
  if (tid < 8) {
    counts[tid] = 0;
    page_rank[tid] = -1;
  }
  for (int slot = tid; slot < capacity; slot += BLOCK_SIZE)
    retained[slot] = 0;
  for (int slot = tid; slot < hash_size; slot += BLOCK_SIZE)
    hash[slot] = -1;
  for (int64_t slot = tid; slot < padded_capacity; slot += BLOCK_SIZE) {
    new_tokens[slot] = new_locations[slot] = -1;
  }
  for (int slot = tid; slot < CAPACITY; slot += BLOCK_SIZE)
    new_lru[slot] = -1;
  __syncthreads();

  for (int rank = tid; rank < capacity; rank += BLOCK_SIZE) {
    const int slot = lru[rank];
    if (slot < 0 || slot >= capacity || atomicAdd(retained + slot, 1) != 0) atomicOr(summary, 1);
    const int loc = locations[rank];
    if (loc < 0) {
      atomicOr(summary, 2);
    } else {
      unsigned key = static_cast<unsigned>(loc);
      key = (key ^ (key >> 16)) * 0x7feb352du;
      key = (key ^ (key >> 15)) * 0x846ca68bu;
      unsigned bucket = (key ^ (key >> 16)) & (hash_size - 1);
      while (true) {
        const int previous = atomicCAS(hash + bucket, -1, loc);
        if (previous == -1) break;
        if (previous == loc) {
          atomicOr(summary, 2);
          break;
        }
        bucket = (bucket + 1) & (hash_size - 1);
      }
    }
  }
  __syncthreads();
  if (summary[0] != 0) return;
  for (int slot = tid; slot < capacity; slot += BLOCK_SIZE)
    retained[slot] = 0;
  __syncthreads();

  for (int base = 0; base < capacity; base += BLOCK_SIZE) {
    const int rank = base + tid;
    const int valid = rank < capacity && tokens[lru[rank]] >= 0;
    const int offset = merge_prefix<BLOCK_SIZE>(valid, scan, carry);
    if (rank < capacity) valid_rank[rank] = offset;
  }
  if (tid == 0) {
    survivor_start = carry > target ? carry - target : 0;
    survivors = carry - survivor_start;
  }
  __syncthreads();
  for (int rank = tid; rank < capacity; rank += BLOCK_SIZE) {
    const int slot = lru[rank];
    if (tokens[slot] >= 0 && valid_rank[rank] >= survivor_start) {
      retained[slot] = 1;
      atomicAdd(counts + slot / k, 1);
    }
  }
  __syncthreads();
  if (tid == 0) {
    int mask = 0;
    for (int selection = 0; selection < target_pages; ++selection) {
      int best = -1;
      for (int page = 0; page < pages; ++page) {
        if (!(mask & (1 << page)) && (best < 0 || counts[page] > counts[best])) best = page;
      }
      mask |= 1 << best;
    }
    int index = 0;
    for (int page = 0; page < pages; ++page) {
      if (mask & (1 << page)) {
        kept[index] = page;
        page_rank[page] = index++;
      }
    }
    summary[2] = mask;
    carry = 0;
  }
  __syncthreads();

  for (int base = 0; base < target; base += BLOCK_SIZE) {
    const int slot = base + tid;
    const int old_slot = slot < target ? kept[slot / k] * k + slot % k : 0;
    const int empty = slot < target && !retained[old_slot];
    const int offset = merge_prefix<BLOCK_SIZE>(empty, scan, carry);
    if (slot < target) {
      new_locations[slot] = locations[old_slot];
      if (retained[old_slot]) {
        new_tokens[slot] = tokens[old_slot];
        new_slot[old_slot] = slot;
      } else {
        available[offset] = old_slot;
      }
    }
  }
  __syncthreads();
  if (tid == 0) carry = 0;
  __syncthreads();
  for (int base = 0; base < capacity; base += BLOCK_SIZE) {
    const int rank = base + tid;
    const int slot = rank < capacity ? lru[rank] : 0;
    const int move = rank < capacity && retained[slot] && page_rank[slot / k] < 0;
    const int offset = merge_prefix<BLOCK_SIZE>(move, scan, carry);
    if (move) {
      const int dst = available[offset];
      const int compacted = page_rank[dst / k] * k + dst % k;
      new_tokens[compacted] = tokens[slot];
      new_slot[slot] = compacted;
      sources[offset] = locations[slot];
      destinations[offset] = locations[dst];
    }
  }
  __syncthreads();
  if (tid == 0) {
    summary[1] = carry;
    carry = 0;
  }
  __syncthreads();
  for (int base = 0; base < target; base += BLOCK_SIZE) {
    const int slot = base + tid;
    const int empty = slot < target && new_tokens[slot] < 0;
    const int offset = merge_prefix<BLOCK_SIZE>(empty, scan, carry);
    if (empty) new_lru[offset] = slot;
  }
  for (int rank = tid; rank < capacity; rank += BLOCK_SIZE) {
    const int slot = lru[rank];
    if (retained[slot]) new_lru[target - survivors + valid_rank[rank] - survivor_start] = new_slot[slot];
  }
}

template <int CAPACITY, int BLOCK_SIZE>
void kvduo_hot_merge_plan(
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView locations,
    tvm::ffi::TensorView lru,
    tvm::ffi::TensorView new_tokens,
    tvm::ffi::TensorView new_locations,
    tvm::ffi::TensorView new_lru,
    tvm::ffi::TensorView sources,
    tvm::ffi::TensorView destinations,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView summary,
    int64_t capacity,
    int64_t k,
    int64_t target_pages,
    int64_t hash_size) {
  using namespace host;
  using SlotT = KVDuoMergeSlot<CAPACITY>;
  auto padded = SymbolicSize{"padded_capacity"};
  auto words = SymbolicSize{"workspace_words"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(tokens).verify(locations);
  TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(new_tokens).verify(new_locations);
  TensorMatcher({CAPACITY}).with_dtype<SlotT>().with_device(device).verify(lru).verify(new_lru);
  TensorMatcher({capacity}).with_dtype<int64_t>().with_device(device).verify(sources).verify(destinations);
  TensorMatcher({words}).with_dtype<int32_t>().with_device(device).verify(workspace);
  TensorMatcher({3}).with_dtype<int32_t>().with_device(device).verify(summary);
  RuntimeCheck(k > 0 && capacity > 0 && capacity <= CAPACITY, "Invalid HOT merge capacity");
  RuntimeCheck(capacity % k == 0 && capacity / k <= 8, "HOT merge requires at most eight whole pages");
  RuntimeCheck(target_pages > 0 && target_pages <= capacity / k, "Invalid HOT merge target");
  RuntimeCheck(padded.unwrap() >= CAPACITY, "Padded metadata cannot be smaller than LRU capacity");
  RuntimeCheck(hash_size >= 2 * capacity && (hash_size & (hash_size - 1)) == 0, "Invalid merge hash workspace");
  RuntimeCheck(words.unwrap() >= 4 * capacity + hash_size, "Insufficient merge workspace");
  LaunchKernel(1, BLOCK_SIZE, device.unwrap())(
      kvduo_hot_merge_plan_kernel<CAPACITY, BLOCK_SIZE>,
      static_cast<const int32_t*>(tokens.data_ptr()),
      static_cast<const int32_t*>(locations.data_ptr()),
      static_cast<const SlotT*>(lru.data_ptr()),
      static_cast<int32_t*>(new_tokens.data_ptr()),
      static_cast<int32_t*>(new_locations.data_ptr()),
      static_cast<SlotT*>(new_lru.data_ptr()),
      static_cast<int64_t*>(sources.data_ptr()),
      static_cast<int64_t*>(destinations.data_ptr()),
      static_cast<int32_t*>(workspace.data_ptr()),
      static_cast<int32_t*>(summary.data_ptr()),
      static_cast<int>(capacity),
      static_cast<int>(k),
      static_cast<int>(target_pages),
      static_cast<int>(hash_size),
      padded.unwrap());
}

}  // namespace
