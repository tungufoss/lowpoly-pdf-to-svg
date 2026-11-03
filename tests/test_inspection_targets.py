import copy
import unittest
from src.lowpoly_pdf_to_svg.inspection import inspection_targets

class InspectionTargetsTest(unittest.TestCase):
    def model(self):
        return {'vertices': [[0,0,0],[10,0,0],[0,18,0],[0,0,0],[10,0,0]],
                'faces': [{'piece':'a','v':[0,1,2],'flat':[[0,0],[10,0],[0,10]]},
                          {'piece':'b','v':[3,4,2],'flat':[[0,0],[10,0],[0,10]]}],
                'pieces': {'a':{'inspect':'a corner 5.0 mm off'},'b':{'inspect':''}},
                'glues':[{'number':'7','piece':'a','tab_piece':'b','v':[0,1,3,4],
                          'gap_mm':2,'status':'does not close'}]}
    def test_corner_locations_and_measurements_without_mutation(self):
        model=self.model(); before=copy.deepcopy(model);marks=inspection_targets(model)['a']
        corners=[m for m in marks if m['kind']=='corner']
        self.assertTrue(corners)
        for mark in corners:
            self.assertGreater(mark['mm'],1)
            self.assertEqual(mark['flat'],model['faces'][0]['flat'][mark['vertex']])
        self.assertEqual(model,before)
        self.assertNotIn('b',inspection_targets(model))
    def test_closed_edges_only_highlighted_when_held(self):
        model=self.model();model['glues'][0]['status']='closes'
        self.assertFalse(any(m['kind']=='edge'for m in inspection_targets(model)['a']))
        model['pieces']['a']['inspect']='moves held back, they would part glued edges: 7'
        self.assertEqual(inspection_targets(model)['a'][0]['vertices'],[0,1])
    def test_open_edge_and_accepted_piece(self):
        model=self.model();model['pieces']['a']['inspect']='glue does not close: 7 (2 mm)'
        self.assertEqual(inspection_targets(model)['a'][0]['kind'],'edge')
        model['pieces']['a']['inspect']=''
        self.assertEqual(inspection_targets(model),{})
