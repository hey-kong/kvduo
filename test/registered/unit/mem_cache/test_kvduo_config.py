import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.sparsity import KVDuoConfig, parse_kvduo_config


class TestKVDuoConfig(unittest.TestCase):
    def test_defaults(self):
        config = parse_kvduo_config(SimpleNamespace(kvduo_config=None))
        self.assertEqual(config, KVDuoConfig())
        self.assertEqual(config.tail_protected_pages, 2)
        self.assertEqual(config.full_reclaim_policy, "request")
        self.assertFalse(config.decode_emergency_reclaim)
        self.assertTrue(config.shared_hbm)
        self.assertEqual(config.hbm_borrow_chunk_mb, 128)

    def test_shared_hbm_explicit_disable(self):
        config = parse_kvduo_config(
            SimpleNamespace(kvduo_config='{"shared_hbm":false}')
        )
        self.assertFalse(config.shared_hbm)

    def test_shared_hbm_config(self):
        config = parse_kvduo_config(
            SimpleNamespace(
                kvduo_config='{"shared_hbm":true,"hbm_borrow_chunk_mb":256}'
            )
        )
        self.assertTrue(config.shared_hbm)
        self.assertEqual(config.hbm_borrow_chunk_mb, 256)
        for raw in (
            '{"shared_hbm":1}',
            '{"hbm_borrow_chunk_mb":0}',
            '{"hbm_borrow_chunk_mb":true}',
            '{"shared_hbm":true,"full_reclaim_policy":"page"}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_kvduo_config(SimpleNamespace(kvduo_config=raw))

    def test_decode_emergency_reclaim_requires_boolean(self):
        for raw, expected in (("true", True), ("false", False)):
            config = parse_kvduo_config(
                SimpleNamespace(kvduo_config=f'{{"decode_emergency_reclaim": {raw}}}')
            )
            self.assertEqual(config.decode_emergency_reclaim, expected)
        for raw in ("1", "0", "null", '"true"', "[]"):
            with (
                self.subTest(raw=raw),
                self.assertRaisesRegex(ValueError, "decode_emergency_reclaim"),
            ):
                parse_kvduo_config(
                    SimpleNamespace(
                        kvduo_config=f'{{"decode_emergency_reclaim": {raw}}}'
                    )
                )

    def test_full_reclaim_policy_values(self):
        for policy in ("request", "page"):
            with self.subTest(policy=policy):
                config = parse_kvduo_config(
                    SimpleNamespace(
                        kvduo_config=f'{{"full_reclaim_policy": "{policy}", "shared_hbm": false}}'
                    )
                )
                self.assertEqual(config.full_reclaim_policy, policy)

    def test_rejects_invalid_full_reclaim_policy(self):
        for value in ('"batch"', '"PAGE"', "true", "1", "null", "[]", "{}"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "full_reclaim_policy"),
            ):
                parse_kvduo_config(
                    SimpleNamespace(kvduo_config=f'{{"full_reclaim_policy": {value}}}')
                )

    def test_tail_protected_pages_and_values(self):
        config = parse_kvduo_config(
            SimpleNamespace(
                kvduo_config=(
                    '{"top_k": 8, "tail_protected_pages": 3, '
                    '"host_size": 4.5, "swap_in_block_size": 256}'
                )
            )
        )
        self.assertEqual(config.top_k, 8)
        self.assertEqual(config.tail_protected_pages, 3)
        self.assertEqual(config.host_size, 4.5)
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


class TestKVDuoPrefillConfig(unittest.TestCase):
    def test_removed_relative_miss_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown kvduo_config"):
            parse_kvduo_config(
                SimpleNamespace(kvduo_config='{"max_relative_miss_increase": 0.25}')
            )

    def test_invalid_relative_miss_values_and_removed_absolute_thresholds(self):
        for raw in [
            '{"max_relative_miss_increase": -0.1}',
            '{"max_relative_miss_increase": true}',
            '{"max_relative_miss_increase": "0.1"}',
            '{"max_relative_miss_increase": NaN}',
            '{"prefill_h_min": 0.8}',
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_kvduo_config(SimpleNamespace(kvduo_config=raw))


if __name__ == "__main__":
    unittest.main()
