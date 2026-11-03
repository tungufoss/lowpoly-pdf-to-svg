import tempfile
import unittest
from pathlib import Path

from src.lowpoly_pdf_to_svg.patches import PatchMaker, pack
from tests.test_retab import REVIEW, pairs


class BackingPatches(unittest.TestCase):
    def test_patch_spans_the_seam_with_the_fold_inverted(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "m.svg"
            svg.write_text(REVIEW)
            maker = PatchMaker(svg, pairs(), {("7", frozenset(("piece-01-01", "piece-01-02"))): "2A"})
            found = maker.patches({"piece-01-01"}, short_mm=25)
        self.assertEqual(len(found), 1)
        patch = found[0]
        self.assertEqual((patch["label"], patch["kind"]), ("2A", "mountain"))   # the seam's valley, inverted
        self.assertAlmostEqual(patch["fold"].length, 20, delta=2)
        # On b's side the patch stays clear of b's diagonal fold.
        self.assertGreater(patch["shape"].area, 100)
        self.assertEqual(len(pack(found)), 1)

    def test_long_seams_get_no_patch(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "m.svg"
            svg.write_text(REVIEW)
            self.assertEqual(PatchMaker(svg, pairs(), {}).patches({"piece-01-01"}, short_mm=15), [])


if __name__ == "__main__":
    unittest.main()
