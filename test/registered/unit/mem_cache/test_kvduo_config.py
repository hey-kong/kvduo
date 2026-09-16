import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.sparsity import KVDuoConfig, parse_kvduo_config


class TestKVDuoConfig(unittest.TestCase):
    def test_defaults(self):
        config = parse_kvduo_config(SimpleNamespace(kvduo_config=None))
        self.assertEqual(config, KVDuoConfig())
        self.assertEqual(config.tail_protected_pages, 2)

    def test_tail_protected_pages_and_values(self):
        config = parse_kvduo_config(
            SimpleNamespace(
                kvduo_config=(
                    '{"top_k": 8, "tail_protected_pages": 3, '
                    '"host_to_device_ratio": 4, "swap_in_block_size": 256}'
                )
            )
        )
        self.assertEqual(config.top_k, 8)
        self.assertEqual(config.tail_protected_pages, 3)
        self.assertEqual(config.host_to_device_ratio, 4)
        self.assertEqual(config.swap_in_block_size, 256)

    def test_rejects_removed_n_alias(self):
        with self.assertRaisesRegex(ValueError, "Unknown kvduo_config"):
            parse_kvduo_config(SimpleNamespace(kvduo_config='{"N": 3}'))

    def test_rejects_removed_minimum_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown kvduo_config"):
            parse_kvduo_config(
                SimpleNamespace(
                    kvduo_config='{"top_k": 8, "min_device_buffer_size": 7}'
                )
            )

    def test_rejects_unknown_field(self):
        with self.assertRaisesRegex(ValueError, "Unknown kvduo_config"):
            parse_kvduo_config(SimpleNamespace(kvduo_config='{"unknown": 1}'))


if __name__ == "__main__":
    unittest.main()
