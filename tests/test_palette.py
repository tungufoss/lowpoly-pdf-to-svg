"""YAML palette and recolour.csv (lowpoly #88): a recolour is an edit, read by the trace and the export."""
import tempfile
import unittest
from pathlib import Path

from src.lowpoly_pdf_to_svg.config import Project
from src.lowpoly_pdf_to_svg.export import Materials

YAML = """project: model
source: {pdf: model.pdf, page_size: letter}
output: {default_sheet: a4}
palette:
  rust: "#B7410E"
  teal: "#00897b"
materials:
  - {id: red, name: Red cardstock, pages: [1], color: "#c62828"}
  - {id: rust, name: Orange cardstock, pages: [2], color: rust}
  - {id: green, name: Green cardstock, pages: [3], pieces: [piece-01-02]}
"""


class Palette(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.config = self.folder / "model.yaml"
        self.config.write_text(YAML)

    def test_material_colour_from_palette_name_or_hex(self):
        project = Project.load(self.config)
        colours = project.colours()
        self.assertEqual(colours["rust"], "#b7410e")
        self.assertEqual(colours["red"], "#c62828")
        self.assertEqual(colours["teal"], "#00897b")   # offered by the viewer before any piece uses it
        self.assertEqual(Materials(self.config).color("rust"), "#b7410e")

    def test_recolour_csv_wins_and_makes_palette_materials(self):
        (self.folder / "recolour.csv").write_text("piece,material\npiece-01-02,teal\npiece-02-01,red\n")
        project = Project.load(self.config)
        moved = project.piece_materials()
        self.assertEqual(moved["piece-01-02"].id, "teal")          # over the YAML's green
        self.assertEqual(moved["piece-02-01"].id, "red")
        self.assertEqual(moved["piece-01-02"].color, "#00897b")
        materials = Materials(self.config)
        self.assertEqual(materials.recolour["piece-01-02"], "teal")
        self.assertEqual(materials.color("teal"), "#00897b")
        self.assertEqual(materials.name("teal"), "Teal cardstock")

    def test_unknown_recolour_name_is_an_error(self):
        (self.folder / "recolour.csv").write_text("piece,material\npiece-01-02,mauve\n")
        with self.assertRaises(ValueError):
            Project.load(self.config)


if __name__ == "__main__":
    unittest.main()
