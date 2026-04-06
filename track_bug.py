import sys
import numpy as np
from pytracking.evaluation.tracker import Tracker
import pytracking.evaluation.data as data
from pytracking.evaluation.data import Sequence

class MockSequence(Sequence):
    def __init__(self):
        super().__init__('MockSeq', [{'image_path': 'dummy'}]*50, 'mock', np.array([[10, 10, 30, 30]*50]).reshape(50,4))
    
    def init_info(self):
        return {'init_bbox': self.ground_truth_rect[0].tolist()}

class MockTracker(Tracker):
    def _read_image(self, image_file):
        return np.ones((300, 300, 3), dtype=np.uint8) * 128

tracker = MockTracker('dimp', 'super_dimp', run_id=None)
tracker.run_sequence(MockSequence(), debug=0)
