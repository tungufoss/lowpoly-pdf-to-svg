import unittest
import xml.etree.ElementTree as ET

from shapely.geometry import LineString, Polygon

from src.lowpoly_pdf_to_svg.export import Edge, GluePairs, SVG
from src.lowpoly_pdf_to_svg.retab import Retabber, TabStyle, _rings

# Piece a: a 20 mm square with a traced 5 mm tab on its right edge (fold x = 20).
# Piece b: a 20 mm square split by a diagonal fold; its left edge glues to a's tab.
REVIEW = f"""<svg xmlns="{SVG}">
<g id="piece-01-01" data-material="red">
 <g data-operation="cut"><path d="M 0,0 20,0 25,5 25,15 20,20 0,20 Z"/></g>
 <g data-operation="score"><g data-fold="valley"><path d="M 20,0 20,20"/></g></g>
 <g data-operation="draw"/>
</g>
<g id="piece-01-02" data-material="red">
 <g data-operation="cut"><path d="M 100,0 120,0 120,20 100,20 Z"/></g>
 <g data-operation="score"><g data-fold="mountain"><path d="M 100,0 120,20"/></g></g>
 <g data-operation="draw"/>
</g>
</svg>"""


def pairs():
    return GluePairs([Edge("7", "piece-01-01", 22, 10, "piece-01-02", "ok", "tab", (20, 0, 20, 20)),
                      Edge("7", "piece-01-02", 101, 10, "piece-01-01", "ok", "edge", (100, 20, 100, 0))])


def cut(root, piece):
    group = next(g for g in root.iter(f"{{{SVG}}}g") if g.get("id") == piece)
    path = next(group.iter(f"{{{SVG}}}path"))
    return Polygon(_rings(path.get("d"))[0])


class GeneratedTabs(unittest.TestCase):
    def test_tab_is_wider_and_stays_clear_of_the_partner_fold(self):
        root = ET.fromstring(REVIEW)
        stats = Retabber(TabStyle(depth_mm=7, flap_mm=0)).apply(root, pairs())
        self.assertEqual(stats["tabs generated"], 1)
        tab = cut(root, "piece-01-01").difference(Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]))
        self.assertGreater(tab.bounds[2], 25)            # wider than the traced 5 mm
        self.assertLessEqual(tab.bounds[2], 27.001)      # no deeper than 7 mm
        # b laid flat across the seam: its diagonal fold lands on (20, 0)-(40, 20); the tab keeps
        # the margin from it, so it never has to bend.
        self.assertGreaterEqual(tab.distance(LineString([(20, 0), (40, 20)])), 0.79)

    def test_flap_fills_the_face_but_not_past_the_fold(self):
        root = ET.fromstring(REVIEW)
        Retabber(TabStyle(depth_mm=7, flap_mm=20, flaps=["piece-01-02"])).apply(root, pairs())
        tab = cut(root, "piece-01-01").difference(Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]))
        self.assertGreater(tab.area, 7 * 20 * 0.6)       # bigger than a plain tab
        self.assertLess(tab.area, 200)                   # inside the 200 mm2 triangle it glues onto

    def test_small_pieces_get_plain_tabs_unless_listed(self):
        root = ET.fromstring(REVIEW)
        Retabber(TabStyle(depth_mm=7, flap_mm=20)).apply(root, pairs())
        tab = cut(root, "piece-01-01").difference(Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]))
        self.assertLessEqual(tab.bounds[2], 27.001)

    def test_tabs_false_keeps_the_traced_tabs(self):
        self.assertIsNone(TabStyle.from_config(False))
        self.assertEqual(TabStyle.from_config({"depth_mm": 8, "keep": ["green"]}).keep, ["green"])


if __name__ == "__main__":
    unittest.main()
