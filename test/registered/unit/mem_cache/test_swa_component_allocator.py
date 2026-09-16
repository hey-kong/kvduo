from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import TreeComponent
from sglang.test.test_utils import CustomTestCase


class TestSWAComponentAllocatorValidation(CustomTestCase):
    def test_accepts_deepseek_v4_hisparse_wrapper(self):
        allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator.__new__(
            DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        params = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            sliding_window_size=4096,
        )

        with patch.object(TreeComponent, "__init__", return_value=None):
            component = SWAComponent(cache=object(), params=params)

        self.assertEqual(component.sliding_window_size, 4096)


if __name__ == "__main__":
    import unittest

    unittest.main()
