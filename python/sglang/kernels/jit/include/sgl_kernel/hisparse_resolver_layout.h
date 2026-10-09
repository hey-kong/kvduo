#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sglang {

// Shared and persistent-global scratch have exactly the same per-CTA layout.
template <int NUM_TOP_K, int HOT_BUFFER_SIZE, int WARP_SIZE>
struct HisparseResolverLayout {
  using SlotT = std::conditional_t<(HOT_BUFFER_SIZE > INT16_MAX), int32_t, int16_t>;
  static constexpr int HASH_SIZE = NUM_TOP_K * 2;
  static constexpr int NUM_BUFFER_CHUNKS = (HOT_BUFFER_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  // top_k_tokens, hit/miss offsets, evictable offsets, hash keys, four counters.
  static constexpr int TOTAL_INT32 = NUM_TOP_K + 2 * (NUM_BUFFER_CHUNKS + 1) + HASH_SIZE + 4;
  static constexpr int TOTAL_SLOTS = HOT_BUFFER_SIZE + HASH_SIZE;
  static constexpr size_t BYTES = TOTAL_INT32 * sizeof(int32_t) + TOTAL_SLOTS * sizeof(SlotT);
};

}  // namespace sglang
