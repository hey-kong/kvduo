#include "hisparse.cuh"
#include <climits>

namespace {
__device__ int freq_find(int token, const int* keys, const int* vals, int size) {
  int h = hash_slot(token, size);
  while (keys[h] != -1 && keys[h] != token)
    h = (h + 1) % size;
  return keys[h] == token ? vals[h] : -1;
}
__device__ void freq_add(unsigned* words, int slot, int count) {
  unsigned* ptr = words + slot / 16;
  const int shift = 2 * (slot % 16);
  unsigned old = atomicAdd(ptr, 0u), assumed;
  do {
    assumed = old;
    unsigned value = min(3, int((assumed >> shift) & 3) + count);
    old = atomicCAS(ptr, assumed, (assumed & ~(3u << shift)) | (value << shift));
  } while (old != assumed);
}

template <int K, bool Paged>
__global__ void frequency_resolve(
    const int* topk,
    int64_t top_stride,
    const int64_t* reqs,
    const int64_t* lengths,
    const int64_t* ptrs,
    const int* capacities,
    int requests,
    const int* real,
    const int64_t* lookup,
    int64_t lookup_stride,
    const int64_t* mapping,
    int64_t logical_size,
    int64_t* touches,
    const int64_t* versions,
    const int64_t* host_versions,
    const int64_t* host_locs,
    int64_t host_stride,
    const void* host_data,
    void* device_data,
    int64_t item_bytes,
    int64_t location_base,
    const int64_t* clock,
    const int* observe,
    int* output,
    int64_t output_stride,
    int* status,
    int* stats,
    int* growth,
    int* scratch) {
  constexpr int H = 2 * K;
  const int row = blockIdx.x, tid = threadIdx.x;
  int* out = output + row * output_stride;
  for (int i = tid; i < K; i += blockDim.x)
    out[i] = -1;
  if (row >= real[0]) return;
  const int64_t rid = reqs[row], length = lengths[row];
  if (rid < 0 || rid >= requests || length < 0 || length > lookup_stride || length > host_stride) {
    if (!tid) __assert_fail("KVDuo invalid request extent", __FILE__, __LINE__, __func__);
    return;
  }
  int* work = scratch + rid * (9 * K);
  int* keys = work;
  int* vals = keys + H;
  int* counts = vals + H;
  int* resolved = counts + H;
  int* misses = resolved + K;
  int* destinations = misses + K;
  const int* selected = topk + row * top_stride;
  int* tokens = reinterpret_cast<int*>(ptrs[rid * 4]);
  const int* locations = reinterpret_cast<const int*>(ptrs[rid * 4 + 1]);
  unsigned* frequency = reinterpret_cast<unsigned*>(ptrs[rid * 4 + 2]);
  auto visited = reinterpret_cast<uint8_t*>(ptrs[rid * 4 + 3]);
  const int capacity = capacities[rid];
  __shared__ int error, count, accesses, full_hits;
  if (!tid) {
    error = 0;
    count = 0;
    accesses = 0;
    full_hits = 0;
    status[rid] = 0;
    if (observe[0]) growth[rid] = 0;
  }
  for (int i = tid; i < H; i += blockDim.x) {
    keys[i] = -1;
    vals[i] = INT_MAX;
    counts[i] = 0;
  }
  for (int i = tid; i < K; i += blockDim.x)
    resolved[i] = -1;
  __syncthreads();
  for (int i = tid; i < K; i += blockDim.x) {
    const int token = selected[i];
    if (token < 0 || token >= length) continue;
    atomicAdd(&accesses, 1);
    const int64_t logical = lookup[rid * lookup_stride + token];
    if (logical < 0 || logical >= logical_size) {
      atomicExch(&error, 1);
      continue;
    }
    const int64_t loc = mapping[logical];
    if (loc > 0) {
      out[i] = int(loc + location_base);
      atomicMax(reinterpret_cast<unsigned long long*>(touches + logical), (unsigned long long)clock[0]);
      atomicAdd(&full_hits, 1);
    } else {
      if (host_locs[rid * host_stride + token] < 0 || versions[logical] != host_versions[logical]) {
        atomicExch(&error, 1);
        continue;
      }
      int h = hash_slot(token, H);
      while (true) {
        const int old = atomicCAS(keys + h, -1, token);
        if (old == -1 || old == token) {
          atomicMin(vals + h, i);
          atomicAdd(counts + h, 1);
          break;
        }
        h = (h + 1) % H;
      }
    }
  }
  __syncthreads();
  if (error) {
    if (!tid) {
      status[rid] = 1;
      __assert_fail("KVDuo missing valid KV backing", __FILE__, __LINE__, __func__);
    }
    return;
  }
  if (full_hits == accesses) {
    if (!tid) atomicAdd(stats + rid * 2 + 1, accesses);
    return;
  }
  if (capacity < K || !tokens || !locations || !frequency || !visited) {
    if (!tid) {
      status[rid] = 1;
      __assert_fail("KVDuo missing minimum HOT capacity", __FILE__, __LINE__, __func__);
    }
    return;
  }
  // One scan of actual capacity, with a top-k-sized hash table. No capacity
  // dependent shared memory, LRU permutation or compile-time capacity limit.
  for (int slot = tid; slot < capacity; slot += blockDim.x) {
    int token = tokens[slot], h = -1;
    if (token >= 0) {
      h = hash_slot(token, H);
      while (keys[h] != -1 && keys[h] != token)
        h = (h + 1) % H;
      if (keys[h] != token) h = -1;
    }
    visited[slot] = h >= 0;
    if (h >= 0) {
      resolved[vals[h]] = locations[slot];
      if (observe[0]) freq_add(frequency, slot, counts[h]);
    }
  }
  __syncthreads();
  if (!tid) {
    for (int i = 0; i < K; ++i) {
      int token = selected[i];
      if (token >= 0 && token < length && out[i] < 0 && freq_find(token, keys, vals, H) == i && resolved[i] < 0)
        misses[count++] = i;
    }
    int next = 0;
    // Empty slots first. Only if expansion could not yet happen, replace the
    // lowest frequency, ties by slot. Never evict this round's HOT hits.
    for (int priority = -1; priority <= 3 && next < count; ++priority) {
      for (int slot = 0; slot < capacity && next < count; ++slot) {
        if (visited[slot]) continue;
        const int token = tokens[slot];
        const unsigned shift = 2 * (slot % 16);
        const int f = (frequency[slot / 16] >> shift) & 3;
        if (priority == -1 ? token >= 0 : (token < 0 || f != priority)) continue;
        int i = misses[next];
        if (token >= 0 && observe[0]) growth[rid] = 1;
        tokens[slot] = selected[i];
        frequency[slot / 16] &= ~(3u << shift);
        visited[slot] = 1;
        resolved[i] = locations[slot];
        destinations[next++] = locations[slot];
      }
    }
    if (next != count) {
      error = 1;
      status[rid] = 1;
    }
    atomicAdd(stats + rid * 2, count);
    atomicAdd(stats + rid * 2 + 1, accesses);
  }
  __syncthreads();
  if (error) {
    if (!tid) __assert_fail("KVDuo insufficient HOT replacement capacity", __FILE__, __LINE__, __func__);
    return;
  }
  for (int i = tid; i < K; i += blockDim.x) {
    const int token = selected[i];
    if (token >= 0 && token < length && out[i] < 0) {
      int canonical = freq_find(token, keys, vals, H);
      out[i] = canonical >= 0 ? resolved[canonical] : -1;
    }
  }
  for (int m = tid / WARP_SIZE; m < count; m += blockDim.x / WARP_SIZE) {
    const int64_t src = host_locs[rid * host_stride + selected[misses[m]]], dst = destinations[m];
    if constexpr (Paged) {
      device::hisparse::transfer_item(device_data, const_cast<void*>(host_data), int(dst), int(src));
    } else {
      transfer_item_warp(
          tid % WARP_SIZE,
          static_cast<const char*>(host_data) + src * item_bytes,
          static_cast<char*>(device_data) + dst * item_bytes,
          item_bytes);
    }
  }
}
}  // namespace

template <int K, bool Paged, int Block>
void resolve_frequency(
    uint64_t topk,
    uint64_t requests,
    uint64_t lengths,
    uint64_t ptrs,
    uint64_t capacities,
    uint64_t real,
    uint64_t lookup,
    uint64_t mapping,
    uint64_t touches,
    uint64_t versions,
    uint64_t host_versions,
    uint64_t host_locs,
    uint64_t host_data,
    uint64_t backing,
    int64_t item_bytes,
    int64_t location_base,
    uint64_t clock,
    uint64_t observe,
    uint64_t output,
    uint64_t status,
    uint64_t stats,
    uint64_t growth,
    uint64_t scratch,
    int64_t batch,
    int64_t owners,
    int64_t top_stride,
    int64_t output_stride,
    int64_t lookup_stride,
    int64_t logical_size,
    int64_t host_stride,
    uint64_t stream) {
  using namespace host;
  LaunchKernel(batch, Block, reinterpret_cast<cudaStream_t>(stream))(
      frequency_resolve<K, Paged>,
      reinterpret_cast<int*>(topk),
      top_stride,
      reinterpret_cast<int64_t*>(requests),
      reinterpret_cast<int64_t*>(lengths),
      reinterpret_cast<int64_t*>(ptrs),
      reinterpret_cast<int*>(capacities),
      owners,
      reinterpret_cast<int*>(real),
      reinterpret_cast<int64_t*>(lookup),
      lookup_stride,
      reinterpret_cast<int64_t*>(mapping),
      logical_size,
      reinterpret_cast<int64_t*>(touches),
      reinterpret_cast<int64_t*>(versions),
      reinterpret_cast<int64_t*>(host_versions),
      reinterpret_cast<int64_t*>(host_locs),
      host_stride,
      reinterpret_cast<void*>(host_data),
      reinterpret_cast<void*>(backing),
      item_bytes,
      location_base,
      reinterpret_cast<int64_t*>(clock),
      reinterpret_cast<int*>(observe),
      reinterpret_cast<int*>(output),
      output_stride,
      reinterpret_cast<int*>(status),
      reinterpret_cast<int*>(stats),
      reinterpret_cast<int*>(growth),
      reinterpret_cast<int*>(scratch));
}

// Runtime-sized global workspace; merge never specializes on the page count.
__global__ void frequency_merge_plan(
    const int* tokens,
    const int* locations,
    const unsigned* frequency,
    int* new_tokens,
    int* new_locations,
    unsigned* new_frequency,
    int64_t* sources,
    int64_t* destinations,
    int* work,
    int* summary,
    int capacity,
    int k,
    int minimum) {
  const int tid = threadIdx.x, pages = capacity / k;
  int* counts = work;
  int* ranks = counts + pages;
  int* available = ranks + pages;
  __shared__ int target, moved;
  if (!tid) {
    target = 0;
    moved = 0;
  }
  for (int p = tid; p < pages; p += blockDim.x) {
    counts[p] = 0;
    ranks[p] = -1;
    summary[p + 2] = 0;
  }
  for (int s = tid; s < capacity; s += blockDim.x) {
    new_tokens[s] = -1;
    new_locations[s] = -1;
  }
  for (int w = tid; w < (capacity + 15) / 16; w += blockDim.x)
    new_frequency[w] = 0;
  __syncthreads();
  for (int s = tid; s < capacity; s += blockDim.x)
    if (tokens[s] >= 0 && ((frequency[s / 16] >> (2 * (s % 16))) & 3) == 3) atomicAdd(counts + s / k, 1);
  __syncthreads();
  if (!tid) {
    int total = 0;
    for (int p = 0; p < pages; ++p)
      total += counts[p];
    target = max(minimum, (total + k - 1) / k);
    for (int i = 0; i < target; ++i) {
      int best = -1;
      for (int p = 0; p < pages; ++p)
        if (!summary[p + 2] && (best < 0 || counts[p] > counts[best])) best = p;
      summary[best + 2] = 1;
    }
    int rank = 0, empty = 0;
    for (int p = 0; p < pages; ++p)
      if (summary[p + 2]) ranks[p] = rank++;
    for (int s = 0; s < capacity; ++s)
      if (ranks[s / k] >= 0 && !(tokens[s] >= 0 && ((frequency[s / 16] >> (2 * (s % 16))) & 3) == 3))
        available[empty++] = s;
    // Sources are exclusively on released pages and destinations on kept pages.
    for (int s = 0; s < capacity; ++s)
      if (ranks[s / k] < 0 && tokens[s] >= 0 && ((frequency[s / 16] >> (2 * (s % 16))) & 3) == 3) {
        const int dst = available[moved], out = ranks[dst / k] * k + dst % k;
        new_tokens[out] = tokens[s];
        new_frequency[out / 16] |= 1u << (2 * (out % 16));
        sources[moved] = locations[s];
        destinations[moved] = locations[dst];
        ++moved;
      }
    summary[0] = target;
    summary[1] = moved;
  }
  __syncthreads();
  for (int s = tid; s < capacity; s += blockDim.x)
    if (ranks[s / k] >= 0) {
      const int out = ranks[s / k] * k + s % k;
      new_locations[out] = locations[s];
      if (tokens[s] >= 0 && ((frequency[s / 16] >> (2 * (s % 16))) & 3) == 3) {
        new_tokens[out] = tokens[s];
        atomicOr(new_frequency + out / 16, 1u << (2 * (out % 16)));
      }
    }
}

template <bool Paged>
__global__ void frequency_merge_copy(
    void* backing, const int64_t* sources, const int64_t* destinations, const int* summary, int64_t bytes) {
  const int lane = threadIdx.x % 32, warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  for (int i = warp; i < summary[1]; i += gridDim.x * blockDim.x / 32) {
    if constexpr (Paged)
      device::hisparse::transfer_item(backing, backing, int(destinations[i]), int(sources[i]));
    else
      transfer_item_warp(
          lane,
          static_cast<const char*>(backing) + sources[i] * bytes,
          static_cast<char*>(backing) + destinations[i] * bytes,
          bytes);
  }
}

__global__ void frequency_merge_publish(
    int64_t* pointers,
    int* capacity,
    int* growth,
    int64_t tokens,
    int64_t locations,
    int64_t frequency,
    int64_t selected,
    const int* summary,
    int k) {
  if (threadIdx.x == 0) {
    pointers[0] = tokens;
    pointers[1] = locations;
    pointers[2] = frequency;
    pointers[3] = selected;
    capacity[0] = summary[0] * k;
    growth[0] = 0;
  }
}

template <bool Paged>
void merge_frequency(
    uint64_t tokens,
    uint64_t locations,
    uint64_t frequency,
    uint64_t new_tokens,
    uint64_t new_locations,
    uint64_t new_frequency,
    uint64_t sources,
    uint64_t destinations,
    uint64_t work,
    uint64_t summary,
    uint64_t backing,
    int64_t k,
    int64_t minimum,
    int64_t bytes,
    int capacity,
    uint64_t pointers,
    uint64_t published_capacity,
    uint64_t growth,
    uint64_t selected,
    uint64_t stream) {
  using namespace host;
  RuntimeCheck(
      k > 0 && capacity % k == 0 && minimum > 0 && minimum <= capacity / k, "Invalid frequency merge geometry");
  LaunchKernel(1, 256, reinterpret_cast<cudaStream_t>(stream))(
      frequency_merge_plan,
      reinterpret_cast<int*>(tokens),
      reinterpret_cast<int*>(locations),
      reinterpret_cast<unsigned*>(frequency),
      reinterpret_cast<int*>(new_tokens),
      reinterpret_cast<int*>(new_locations),
      reinterpret_cast<unsigned*>(new_frequency),
      reinterpret_cast<int64_t*>(sources),
      reinterpret_cast<int64_t*>(destinations),
      reinterpret_cast<int*>(work),
      reinterpret_cast<int*>(summary),
      capacity,
      int(k),
      int(minimum));
  LaunchKernel(std::min(128, (capacity + 7) / 8), 256, reinterpret_cast<cudaStream_t>(stream))(
      frequency_merge_copy<Paged>,
      reinterpret_cast<void*>(backing),
      reinterpret_cast<int64_t*>(sources),
      reinterpret_cast<int64_t*>(destinations),
      reinterpret_cast<int*>(summary),
      bytes);
  if (pointers)
    LaunchKernel(1, 32, reinterpret_cast<cudaStream_t>(stream))(
        frequency_merge_publish,
        reinterpret_cast<int64_t*>(pointers),
        reinterpret_cast<int*>(published_capacity),
        reinterpret_cast<int*>(growth),
        int64_t(new_tokens),
        int64_t(new_locations),
        int64_t(new_frequency),
        int64_t(selected),
        reinterpret_cast<int*>(summary),
        int(k));
}
