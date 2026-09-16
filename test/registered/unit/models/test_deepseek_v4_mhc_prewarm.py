"""Unit tests for DeepSeek-V4 MHC kernel prewarming."""

import unittest

from sglang.kernels.ops.layernorm.mhc import (
    get_mhc_pre_token_count_representatives,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV4MHCPrewarm(CustomTestCase):
    def test_disabled_chunked_prefill_has_no_prewarm_buckets(self):
        self.assertEqual(get_mhc_pre_token_count_representatives(-1, 16384), ())

    def test_zero_token_budget_has_no_prewarm_buckets(self):
        self.assertEqual(get_mhc_pre_token_count_representatives(0, 16384), ())


if __name__ == "__main__":
    unittest.main()
