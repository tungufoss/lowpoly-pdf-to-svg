import unittest
from collections import defaultdict

from src.lowpoly_pdf_to_svg.assembly import AssemblyOrder
from src.lowpoly_pdf_to_svg.export import PieceNumbers


def order_of(distance, links, groups, slack=0.0):
    """An AssemblyOrder over a hand-made seam graph (no fold3d.json needed)."""
    order = AssemblyOrder.__new__(AssemblyOrder)
    order.distance, order.groups, order.slack = distance, groups, slack
    order.links = defaultdict(set)
    for p, q in links:
        order.links[p].add(q)
        order.links[q].add(p)
    order.steps = order._sweep()
    return order


class AssemblyOrderTests(unittest.TestCase):
    def test_far_from_the_neck_goes_first_and_twins_share_a_step(self):
        # A column: top (far) -> middle pair -> neck rim.
        order = order_of({"top": 300, "mid-l": 200, "mid-r": 200, "rim": 0},
                         [("top", "mid-l"), ("top", "mid-r"), ("mid-l", "rim"), ("mid-r", "rim")],
                         [["top"], ["mid-l", "mid-r"], ["rim"]])
        self.assertEqual([s.pieces for s in order.steps], [["top"], ["mid-l", "mid-r"], ["rim"]])
        self.assertTrue(order.steps[0].starts_part)
        self.assertEqual(order.steps[2].onto, ["mid-l", "mid-r"])

    def test_two_far_branches_start_parts_and_are_joined(self):
        order = order_of({"horn-a": 300, "horn-b": 290, "head": 100, "rim": 0},
                         [("horn-a", "head"), ("horn-b", "head"), ("head", "rim")],
                         [["horn-a"], ["horn-b"], ["head"], ["rim"]])
        self.assertEqual([s.starts_part for s in order.steps], [True, True, False, False])
        self.assertEqual(order.steps[2].joins, 1)

    def test_slack_keeps_the_build_in_one_place(self):
        # b is a little nearer than c but glues onto a, the piece just placed.
        order = order_of({"a": 300, "b": 280, "c": 290, "rim": 0},
                         [("a", "b"), ("b", "rim"), ("c", "rim")], [["a"], ["b"], ["c"], ["rim"]], slack=25)
        self.assertEqual(order.order(), ["a", "b", "c", "rim"])


class BuildNumberTests(unittest.TestCase):
    def test_numbers_follow_the_build_order_and_twins_share_one(self):
        ids = ["piece-01-01", "piece-01-02", "piece-02-01", "piece-02-02"]
        twins = {"piece-01-01": "piece-02-01", "piece-02-01": "piece-01-01"}
        numbers = PieceNumbers(ids, twins, order=["piece-02-02", "piece-02-01", "piece-01-01"])
        self.assertEqual(numbers.numbers, {"piece-02-02": 1, "piece-02-01": 2, "piece-01-01": 2, "piece-01-02": 3})
        self.assertEqual(numbers.frames["piece-02-02"], "triangle")


if __name__ == "__main__":
    unittest.main()
