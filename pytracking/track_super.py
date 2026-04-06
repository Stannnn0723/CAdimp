import sys
from pytracking.evaluation.environment import env_settings
from pytracking.evaluation.tracker import Tracker
from pytracking.evaluation.datasets import get_dataset

tracker = Tracker('dimp', 'super_dimp', run_id=None)
dataset = get_dataset('uav')
# Find Kongming_Lantern8
seq = None
for s in dataset:
    if s.name == 'Kongming_Lantern8':
        seq = s
        break
if seq is None:
    print("Sequence not found!")
    sys.exit(1)
tracker.run_sequence(seq, debug=0)
