import re
import unittest
import xml.etree.ElementTree as ET
from src.lowpoly_pdf_to_svg.export import PieceNumbers, SVG


class PieceIdentifierTests(unittest.TestCase):
    def test_pair_numbers_and_circle_square_stay_stable(self):
        ids = [f'piece-01-{i:02d}' for i in range(1, 6)]
        twins = {ids[0]: ids[1], ids[1]: ids[0], ids[3]: ids[4], ids[4]: ids[3]}
        numbers = PieceNumbers(ids, twins)
        self.assertEqual([numbers.numbers[p] for p in ids], [1, 1, 2, 3, 3])
        self.assertEqual([numbers.frames[p] for p in ids],
                         ['circle', 'square', 'triangle', 'circle', 'square'])

    def test_unpaired_identifier_is_alphanumeric_inside_triangle(self):
        pid = 'piece-01-01'
        piece = ET.Element(f'{{{SVG}}}g', {'id': pid})
        draw = ET.SubElement(piece, f'{{{SVG}}}g', {'data-operation': 'draw'})
        PieceNumbers([pid]).add(piece, [(0, 0), (100, 0), (100, 70), (0, 70)])
        text = draw.find(f'{{{SVG}}}text')
        self.assertEqual(text.text, 'A1')
        path = next(p for p in draw.iter(f'{{{SVG}}}path') if p.get('data-mark') == 'triangle')
        self.assertEqual(path.get('data-piece-number'), 'A1')
        coords = [tuple(map(float, match)) for match in re.findall(r'([-\d.]+),([-\d.]+)', path.get('d'))]
        self.assertEqual(coords[0], coords[-1])
        self.assertEqual(len(coords), 4)
        self.assertTrue(all(0 <= x <= 100 and 0 <= y <= 70 for x, y in coords))
