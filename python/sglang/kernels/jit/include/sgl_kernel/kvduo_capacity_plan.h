#pragma once
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define KVDUO_PLAN_HD __host__ __device__
#else
#define KVDUO_PLAN_HD
#endif

namespace sglang {

KVDUO_PLAN_HD inline int
kvduo_capacity_target(const float* hits, float misses, int64_t steps, int pages, int minimum, double rho) {
  if (minimum < 1 || minimum > pages || pages > 8) return -2;
  if (steps <= 64) return pages;
  if (!std::isfinite(misses) || misses < 0 || !std::isfinite(rho) || rho < 0) return pages;
  for (int j = 0; j < pages; ++j) {
    if (!std::isfinite(hits[j]) || hits[j] < 0) return pages;
  }
  for (int target = minimum; target < pages; ++target) {
    // The host policy sums float32 observations as Python doubles, in rank order.
    double loss = 0;
    for (int j = target; j < pages; ++j)
      loss += static_cast<double>(hits[j]);
    if (loss <= rho * static_cast<double>(misses)) return target;
  }
  return pages;
}

}  // namespace sglang
#undef KVDUO_PLAN_HD
