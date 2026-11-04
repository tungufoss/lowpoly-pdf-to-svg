import tempfile
import unittest
from pathlib import Path

from src.lowpoly_pdf_to_svg.config import Project
from src.lowpoly_pdf_to_svg.palette import Recolour, YamlMaterials, palette_json

YAML = """project: toy
source:
  pdf: toy.pdf
  page_size: letter
output:
  default_sheet: a4

palette:
  - id: blue
    name: Blue cardstock
    color: "#1565c0"

materials:
  # page 1 mixes colours
  - id: red
    name: Red cardstock
    pages: [1]          # the mouth
  - id: brown
    color: "#5d4037"
    pieces: [piece-01-03]   # rings
  - id: gold
    name: Gold cardstock
    pages: [2]

# the finished model
reference:
  parts: 1
"""


class PaletteTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "toy.yaml"
        self.path.write_text(YAML, encoding="utf-8")
        self.project = Project.load(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def recolour(self, changes):
        recolour = Recolour(self.project)
        text = YamlMaterials(YAML).write(recolour.apply(changes), {m.id: m for m in self.project.palette},
                                         {m.id: list(m.pieces) for m in self.project.materials})
        self.path.write_text(text, encoding="utf-8")
        return text, Recolour(Project.load(self.path))

    def test_palette_fills_names_and_colours(self):
        data = palette_json(self.project)
        by_id = {m["id"]: m for m in data["materials"]}
        self.assertEqual(by_id["brown"]["color"], "#5d4037")
        self.assertEqual(by_id["brown"]["name"], "brown")
        self.assertEqual(by_id["gold"]["color"], "#d4a017")      # named default
        self.assertTrue(by_id["blue"]["spare"])
        self.assertEqual(data["sheet_mm"], [210.0, 297.0])

    def test_recolour_moves_pieces_and_keeps_comments(self):
        text, after = self.recolour({"piece-01-02": "brown", "piece-02-01": "blue", "piece-01-03": "red"})
        self.assertEqual(after.material("piece-01-02"), "brown")
        self.assertEqual(after.material("piece-02-01"), "blue")
        self.assertEqual(after.material("piece-01-03"), "red")   # back on its page's colour
        self.assertEqual(after.material("piece-01-01"), "red")
        for kept in ("# page 1 mixes colours", "# the mouth", "# the finished model", "reference:"):
            self.assertIn(kept, text)
        self.assertIn('  - id: blue\n    name: Blue cardstock\n    color: "#1565c0"\n    pieces: [piece-02-01]\n', text)

    def test_unchanged_list_keeps_its_lines(self):
        text, _ = self.recolour({"piece-02-01": "blue"})
        self.assertIn("    pieces: [piece-01-03]   # rings\n", text)

    def test_unknown_colour_is_refused(self):
        with self.assertRaises(ValueError):
            Recolour(self.project).apply({"piece-01-01": "teal"})


if __name__ == "__main__":
    unittest.main()
