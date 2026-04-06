import sys
import numpy as np
from pytracking.evaluation.tracker import Tracker
import pytracking.evaluation.data as data
from pytracking.evaluation.data import Sequence

class MockSequence(Sequence):
    def __init__(self):
        super().__init__('MockSeq', [{'image': np.zeros((300, 300, 3), dtype=np.uint8)}]*100, 'mock', np.array([[10, 10, 30, 30]*100]).reshape(100,4))
    
    def init_info(self):
        return {'init_bbox': self.ground_truth_rect[0].tolist()}

class MockTracker(Tracker):
    def _read_image(self, image_file):
        return np.ones((300, 300, 3), dtype=np.uint8) * 128

import argparse
tracker = MockTracker('dimp', 'super_dimp', run_id=None)
try:
    tracker.run_sequence(MockSequence(), debug=False)
except Exception as e:
    import traceback
    traceback.print_exc()
