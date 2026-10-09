#pragma once
#include <stdint.h>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define KVDUO_HOST_DEVICE __host__ __device__
#else
#define KVDUO_HOST_DEVICE
#endif
namespace sglang {
KVDUO_HOST_DEVICE inline void kvduo_record_hot_hit(bool& visited, bool observe) {
  if (observe) visited = true;
}
KVDUO_HOST_DEVICE inline void kvduo_record_hot_insert(bool& visited) {
  visited = false;
}
// Actual extra-page reuse and actual H2D misses, including valid zero/zero steps.
KVDUO_HOST_DEVICE inline void
kvduo_update_prefill_ema(int extra_hits, int misses, int accesses, float* counts, int64_t& steps) {
  if (accesses <= 0) return;
  constexpr float alpha = 1.0f / 64.0f;
  const int values[2] = {extra_hits, misses};
  for (int i = 0; i < 2; ++i) {
    counts[i] = steps == 0 ? static_cast<float>(values[i]) : (1.0f - alpha) * counts[i] + alpha * values[i];
  }
  ++steps;
}
}  // namespace sglang
#undef KVDUO_HOST_DEVICE
