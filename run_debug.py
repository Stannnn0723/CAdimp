import sys
import traceback
from pytracking.evaluation.tracker import Tracker
from pytracking.run_tracker import run_tracker

orig_track = Tracker._track_sequence

def patched_track(self, tracker, seq, init_info):
    try:
        return orig_track(self, tracker, seq, init_info)
    except Exception as e:
        print("====== TRACEBACK CAUGHT ======")
        traceback.print_exc()
        raise e

Tracker._track_sequence = patched_track

import argparse
# We just call run_tracker directly on the exact sequence
run_tracker('dimp', 'super_dimp', run_id=None, dataset_name='latot', sequence='Kongming_Lantern1', debug=0, threads=0)
