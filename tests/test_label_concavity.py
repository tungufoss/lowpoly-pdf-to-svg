import unittest
import xml.etree.ElementTree as ET
from src.lowpoly_pdf_to_svg.export import _inside, _place


class ConcaveLabelTests(unittest.TestCase):
    def test_label_center_cannot_fall_in_notch_between_inside_corners(self):
        outline = [(0, 0), (20, 0), (20, 20), (11, 20),
                   (11, 8), (9, 8), (9, 20), (0, 20)]
        self.assertTrue(all(_inside(p, outline) for p in [(7, 7), (13, 7), (7, 13), (13, 13)]))
        self.assertFalse(_inside((10, 10), outline))
        text = ET.Element('text', {'x': '10', 'y': '5'})
        _place(text, (2, 7.5, 18, 7.5), 0, 0, outline, tab=True, sizes=(5,))
        self.assertTrue(_inside((float(text.get('x')), float(text.get('y'))), outline))
