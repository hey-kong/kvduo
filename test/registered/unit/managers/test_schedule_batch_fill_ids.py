from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils.common import Range
from sglang.test.test_utils import CustomTestCase


class TestReqGetFillIds(CustomTestCase):
    def setUp(self):
        self.req = Req.__new__(Req)
        self.req.full_untruncated_fill_ids = array("q", [1, 2, 3, 4])

    def test_returns_full_input_before_extend_range_is_initialized(self):
        self.req.extend_range = None

        self.assertEqual(self.req.get_fill_ids(), array("q", [1, 2, 3, 4]))

    def test_honors_chunked_extend_range(self):
        self.req.extend_range = Range(0, 2)

        self.assertEqual(self.req.get_fill_ids(), array("q", [1, 2]))


if __name__ == "__main__":
    import unittest

    unittest.main()
