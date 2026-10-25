import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.lowpoly_pdf_to_svg.detect import PageDetector


class FoldScanCacheTests(unittest.TestCase):
    def test_mirror_page_switches_reuse_each_scan(self):
        detector = PageDetector()
        detector._masks = Mock(return_value=(np.zeros((20, 20), np.uint8),))
        left, right = SimpleNamespace(mm_per_px=1), SimpleNamespace(mm_per_px=1)
        for page in [left, right, left, right]:
            self.assertIsNone(detector.fold_kind(page, (1, 1), (15, 15)))
        self.assertEqual(detector._masks.call_count, 2)
