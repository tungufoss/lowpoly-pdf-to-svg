import unittest
from types import SimpleNamespace

from shapely.geometry import LineString, Polygon

from src.lowpoly_pdf_to_svg.retab import Glue, Retabber, TabStyle


class FakeShape:
    """A square piece whose glue edges are given directly as lines (document mm)."""
    def __init__(self):
        self.body = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    @staticmethod
    def line(row):
        ax, ay, bx, by = row.line
        return LineString([(ax, ay), (bx, by)])


def edge(piece, line):
    return SimpleNamespace(piece=piece, line=line)


def retabber(twins, outward=()):
    r = Retabber(TabStyle(outward=list(outward)))
    for a, b in twins:
        r.twin[a], r.twin[b] = b, a
    return r


class TabSideTests(unittest.TestCase):
    def test_centre_seam_alternates_even_when_pdf_put_all_tabs_on_one_side(self):
        r = retabber([("piece-01-01", "piece-01-02")])
        # Four glues along the left edge, all tabs on piece-01-01 in the PDF.
        glues = [Glue(str(n), edge("piece-01-01", (0, y, 0, y + 20)), edge("piece-01-02", (0, y, 0, y + 20)))
                 for n, y in enumerate((5, 30, 55, 80))]
        shapes = {"piece-01-01": FakeShape(), "piece-01-02": FakeShape()}
        r._choose_sides(glues, shapes)
        sides = [g.side for g in glues]
        self.assertEqual(sorted(sides), ["piece-01-01"] * 2 + ["piece-01-02"] * 2)
        self.assertTrue(all(a != b for a, b in zip(sides, sides[1:])))

    def test_outward_piece_owns_tabs_on_seams_with_unlisted_pieces(self):
        r = retabber([("piece-24-01", "piece-24-02")], outward=["24-01"])
        shapes = {p: FakeShape() for p in ("piece-24-01", "piece-24-02", "piece-14-01")}
        # The PDF put the tab on the lip (14-01); the tooth should get it.
        g = Glue("1", edge("piece-14-01", (0, 0, 10, 0)), edge("piece-24-02", (0, 0, 10, 0)))
        r._choose_sides([g], shapes)
        self.assertEqual(g.side, "piece-24-02")   # the twin of a listed tooth counts too

    def test_seam_between_two_outward_pieces_keeps_the_usual_rule(self):
        r = retabber([], outward=["24-*"])
        shapes = {p: FakeShape() for p in ("piece-24-01", "piece-24-03")}
        g = Glue("1", edge("piece-24-01", (0, 0, 10, 0)), edge("piece-24-03", (0, 0, 10, 0)))
        r._choose_sides([g], shapes)
        self.assertEqual(g.side, "piece-24-01")

    def test_outward_is_read_from_yaml(self):
        style = TabStyle.from_config({"outward": ["24-*"], "depth_mm": 7.5})
        self.assertEqual(style.outward, ["24-*"])


if __name__ == "__main__":
    unittest.main()
