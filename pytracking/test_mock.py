from pytracking.evaluation.tracker import Tracker
from pytracking.evaluation.data import Sequence
import numpy as np
class MockSequence:
    def __init__(self):
        self.name = 'MockSeq'
        self.dataset = 'mock'
        self.ground_truth_rect = np.array([[10, 10, 30, 30]]*100)
    def __len__(self):
        return 100
    def frames(self):
        for _ in range(100):
            yield np.zeros((300, 300, 3), dtype=np.uint8), None
tracker = Tracker('dimp', 'super_dimp', None)
tracker.run_sequence(MockSequence())

