import unittest
import copy
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from src.lowpoly_pdf_to_svg.export import Edge, GluePairs, LabelAlphabet, ExportWriter, Materials, SVG


def edge(number, piece, partner):
    return Edge(number, piece, 0, 0, partner, "ok", "edge")


class SelfJoinLabels(unittest.TestCase):
    def test_svg_and_csv_both_use_matching_single_letters(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures/export"
        tree = ET.parse(fixture / "sample-review.svg")
        pieces = [g for g in tree.getroot().iter(f"{{{SVG}}}g")
                  if g.get("id") == "piece-01-01"]
        text = next(t for t in pieces[0].iter(f"{{{SVG}}}text") if t.text == "558")
        pieces[0].append(copy.deepcopy(text))
        pairs = GluePairs([edge("558", "piece-01-01", "piece-01-01") for _ in range(2)])
        with tempfile.TemporaryDirectory() as directory:
            source, target, table = [Path(directory) / name for name in ("review.svg", "export.svg", "labels.csv")]
            tree.write(source)
            ExportWriter(pairs, Materials(fixture / "sample.yaml")).export(source, target)
            pairs.write(table)
            labels = [t for t in ET.parse(target).getroot().iter(f"{{{SVG}}}text")
                      if t.get("data-label") == "A"]
            self.assertEqual(len(labels), 2)
            self.assertTrue(all(t.text == "A" for t in labels))
            self.assertIn("A,558,ok,piece-01-01 piece-01-01", table.read_text())

    def test_self_joins_get_unique_letters_and_external_labels_stay_stable(self):
        pairs = GluePairs([
            edge("1", "piece-01-01", "piece-01-01"),
            edge("1", "piece-01-01", "piece-01-01"),
            edge("2", "piece-01-01", "piece-01-02"),
            edge("2", "piece-01-02", "piece-01-01"),
            edge("3", "piece-01-02", "piece-01-02"),
            edge("3", "piece-01-02", "piece-01-02"),
        ])
        pairs.assign(LabelAlphabet())
        self.assertEqual([p.label for p in pairs.pairs], ["A", "1B", "B"])

    def test_incomplete_or_ambiguous_pairs_keep_normal_labels(self):
        pairs = GluePairs([edge("1", "p", "p")] + [edge("2", "p", "p")] * 3)
        pairs.assign(LabelAlphabet())
        self.assertEqual([p.label for p in pairs.pairs], ["1A", "1B"])

    def test_exhaustion_fails_instead_of_reusing_letters(self):
        count = sum(c.isalpha() for c in LabelAlphabet.SYMBOLS) + 1
        pairs = GluePairs([edge(str(i), "p", "p") for i in range(count) for _ in range(2)])
        with self.assertRaisesRegex(ValueError, "self-join"):
            pairs.assign(LabelAlphabet())
