import unittest
import numpy as np
from unittest.mock import patch
from src.lowpoly_pdf_to_svg.export import Edge, GluePairs, LabelAlphabet
from src.lowpoly_pdf_to_svg.fold3d import FoldSolver, FlatPiece, Label, _look_alike, _apply_confirmed_edges


class ConfirmedJoinTests(unittest.TestCase):
    def test_reviewed_edge_assignment_overrides_label_position_guess(self):
        piece = FlatPiece('p', 'red', [(0, 0), (10, 0), (10, 10), (0, 10)], [],
                          [Label('47', np.array([5, 1]), 0, True)])
        piece.split_faces()
        piece.mark_labels()
        before = piece.labels[0].edge
        _apply_confirmed_edges(piece, [{'number': '47', 'line_mm': '10 0 10 10', 'side': 'edge'}])
        self.assertNotEqual(set(before), set(piece.labels[0].edge))
        self.assertTrue(all(piece.points[i][0] == 10 for i in piece.labels[0].edge))
        self.assertFalse(piece.labels[0].tab)
        with self.assertRaises(ValueError):
            _apply_confirmed_edges(piece, [{'number': '47', 'line_mm': '100 0 100 10', 'side': 'edge'}])

    def test_duplicate_number_join_preserves_other_pair_and_labels(self):
        edges = [Edge('269', 'a', 0, 0, 'b', 'ok', 'edge'),
                 Edge('269', 'b', 0, 0, 'a', 'ok', 'tab'),
                 Edge('269', 'c', 0, 0, '', 'extra copy', 'edge'),
                 Edge('269', 'd', 0, 0, '', 'extra copy', 'tab')]
        pairs = GluePairs(edges)
        pairs.assign(LabelAlphabet())
        original = pairs.pairs[0].label
        label = pairs.pairs[1].label
        pairs.confirm([{'number': '269', 'pieces': ['c', 'd'], 'label': label}])
        self.assertEqual(len(pairs.pairs), 2)
        self.assertEqual(pairs.pairs[0].label, original)
        self.assertEqual(pairs.pairs[1].label, label)
        self.assertEqual({e.piece for e in pairs.pairs[1].edges}, {'c', 'd'})
        self.assertTrue(all(e.status == 'ok' for e in pairs.pairs[1].edges))

    def test_confirmed_proposal_does_not_require_ocr_resemblance(self):
        with patch.object(FoldSolver, 'pinned_numbers', frozenset({'269/269'})):
            self.assertEqual(_look_alike('269/269'), 'yes')
            self.assertEqual(_look_alike('123/123'), 'no')
