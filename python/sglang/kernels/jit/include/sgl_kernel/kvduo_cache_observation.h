#pragma once
#include <stdint.h>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define KVDUO_HOST_DEVICE __host__ __device__
#else
#define KVDUO_HOST_DEVICE
#endif
namespace sglang {
// Standalone HiSparse has neither a swap-status bank nor growth events. Gate
// their reads before dereferencing any KVDuo-only state in the shared resolver.
KVDUO_HOST_DEVICE inline bool kvduo_can_record_growth(
    bool enable_full_lookup, bool observe, const int32_t* growth_events, const int32_t* swap_status, int64_t rid) {
  return enable_full_lookup && observe && growth_events != nullptr && swap_status != nullptr && swap_status[rid] == 0;
}
KVDUO_HOST_DEVICE inline int kvduo_lru_segment(int capacity, int position, int k) {
  return (capacity - 1 - position) / k;
}
// Counts are taken at pre-update LRU ranks, from MRU segment 0 to LRU.
KVDUO_HOST_DEVICE inline void
kvduo_update_capacity_ema(const int* hits, int misses, int accesses, int pages, float* counts, int64_t& steps) {
  if (accesses <= 0 || pages <= 0) return;
  constexpr float alpha = 1.0f / 64.0f;
  for (int j = 0; j < 9; ++j) {
    const float value = j == 8 ? static_cast<float>(misses) : (j < pages ? static_cast<float>(hits[j]) : 0.0f);
    counts[j] = steps == 0 ? value : (1.0f - alpha) * counts[j] + alpha * value;
  }
  ++steps;
}
}  // namespace sglang
#undef KVDUO_HOST_DEVICE
