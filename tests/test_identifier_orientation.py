import unittest,xml.etree.ElementTree as E,numpy as np
from src.lowpoly_pdf_to_svg.identifier_orientation import piece_angle,orient_identifiers
class IdentifierOrientationTest(unittest.TestCase):
 def model(self):return {'vertices':[[0,0,0],[0,10,0],[0,0,10]],'faces':[{'piece':'a','v':[0,1,2],'flat':[[0,0],[10,0],[0,10]]}]}
 def test_assembled_up_ignores_page_axes(self):self.assertAlmostEqual(piece_angle(self.model(),'a',[2,2]),90)
 def test_text_frame_rotated_together_and_repeat_is_stable(self):
  root=E.fromstring('<svg xmlns="http://www.w3.org/2000/svg"><g data-piece="a"><text x="0" y="0" data-piece-number="1">1</text><path data-piece-number="1" d="M 0,-2 2,0 0,2 -2,0 0,-2"/></g></svg>')
  orient_identifiers(root,self.model());path=list(root.iter('{http://www.w3.org/2000/svg}path'))[0];self.assertIn('2.000000,',path.get('d'));first=path.get('d');orient_identifiers(root,self.model());self.assertEqual(path.get('d'),first)
