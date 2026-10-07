#pragma once

#include <stdint.h>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SGLANG_KVDUO_HOST_DEVICE __host__ __device__
#else
#define SGLANG_KVDUO_HOST_DEVICE
#endif

namespace sglang {

SGLANG_KVDUO_HOST_DEVICE inline int kvduo_hot_releasable_pages(int capacity, int slots_per_page, int top_k) {
  const int pages = capacity / slots_per_page;
  const int minimum = (top_k + slots_per_page - 1) / slots_per_page;
  const int half = (pages + 1) / 2;
  const int target = minimum > half ? minimum : half;
  return pages > target ? pages - target : 0;
}

// One valid resolver step, including a step with zero tail HOT hits. Capacity
// growth retains this state; shrink/merge and request reuse reset it externally.
SGLANG_KVDUO_HOST_DEVICE inline void kvduo_update_hot_tail_ema(
    int tail_hot_hits, int releasable_pages, float alpha, float& score, int64_t& observation_steps) {
  if (releasable_pages <= 0) {
    return;
  }
  const float loss_per_page = static_cast<float>(tail_hot_hits) / releasable_pages;
  score = observation_steps == 0 ? loss_per_page : (1.0f - alpha) * score + alpha * loss_per_page;
  ++observation_steps;
}

}  // namespace sglang

#undef SGLANG_KVDUO_HOST_DEVICE
