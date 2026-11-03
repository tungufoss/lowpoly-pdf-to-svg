import math
import unittest

import numpy as np

from src.lowpoly_pdf_to_svg.headfit import FoldedMask, HeadFitter, HeadModel, HeadSize, SightFinder


def helmet(radius: float = 150.0, rings: int = 14, segments: int = 24) -> dict:
    """A sphere open below the equator's lower quarter (the neck), as fold3d.json would hold it."""
    vertices, faces = [], []
    lowest = int(rings * 0.8)                       # rings past this are left out: the neck opening
    for i in range(lowest + 1):
        theta = math.pi * i / rings
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            vertices.append([radius * math.sin(theta) * math.cos(phi),
                             radius * math.sin(theta) * math.sin(phi), radius * math.cos(theta)])
    for i in range(lowest):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            c, d = a + segments, b + segments
            quad = [a, c, d, b]
            faces.append(dict(id=f"piece-01-01/{len(faces)}", piece="piece-01-01", tab=False, v=quad,
                              flat=[[float(k), 0.0] for k in range(4)]))
    return dict(vertices=vertices, faces=faces, pieces={"piece-01-01": {"material": "red"}},
                colours={"red": "#c62828"})


class HeadFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = FoldedMask(helmet(), np.array([0, 0, 1.0]), np.array([0, -1.0, 0]))

    def test_measured_circumference_scales_the_preset(self):
        head = HeadSize.from_fit({"head": "male", "circumference_mm": 600})
        self.assertAlmostEqual(head.circumference, 600, delta=0.5)
        self.assertEqual(HeadSize.from_fit({"head": "child", "eye_distance_mm": 50}).eye_distance, 50)
        with self.assertRaises(ValueError):
            HeadSize.from_fit({"head": "giant"})

    def test_a_smaller_head_needs_a_smaller_mask(self):
        scales = {}
        for name in ("child", "male"):
            fitter = HeadFitter(self.mask, HeadModel(HeadSize.from_fit({"head": name})), clearance=10,
                                spacing=25)
            scales[name], pupils, margin = fitter.fit(low=0.8, high=2.5, tolerance=0.05)
            self.assertGreaterEqual(margin, 10)
        self.assertLess(scales["child"], scales["male"])

    def test_eyes_look_out_through_the_card(self):
        sight = SightFinder(self.mask, 1.0).look(np.array([30.0, 0.0, 20.0]))
        self.assertEqual(sight["through"], "card")
        self.assertAlmostEqual(sight["distance_mm"], math.sqrt(150 ** 2 - 30 ** 2 - 20 ** 2), delta=5)


if __name__ == "__main__":
    unittest.main()
