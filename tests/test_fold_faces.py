import math
import unittest
from src.lowpoly_pdf_to_svg.fold3d import FlatPiece


class FoldCornerTests(unittest.TestCase):
    def test_diagonal_retains_exact_corners_after_extension(self):
        for degrees in range(1, 180):
            angle = math.radians(degrees)
            def point(x, y):
                return (526.54 + x * math.cos(angle) - y * math.sin(angle),
                        256.74 + x * math.sin(angle) + y * math.cos(angle))
            outline = [point(0, 0), point(40, 0), point(40, 10),
                       point(30, 15), point(40, 20), point(40, 30), point(0, 30)]
            piece = FlatPiece('synthetic', 'rust', outline,
                              [(outline[0], outline[3], 'mountain')])
            piece.split_faces()
            with self.subTest(degrees=degrees):
                self.assertEqual(len(piece.faces), 2)
