import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset, Tracker
from pytracking.analysis.plot_results import print_results, plot_results


def eval_small112():
    trackers = [Tracker('dimp', 'super_dimp', None, dataset_name='small112')]
    dataset = get_dataset('small112')

    report_name = 'small112_eval'

    print_results(trackers, dataset, report_name,
                  merge_results=False,
                  plot_types=('success', 'prec', 'norm_prec'))
    plot_results(trackers, dataset, report_name,
                 merge_results=False,
                 plot_types=('success', 'prec', 'norm_prec'))


if __name__ == '__main__':
    eval_small112()
