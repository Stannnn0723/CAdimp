from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def latot_test():
    trackers = trackerlist('latot', 'default', range(1))
    dataset = get_dataset('latot')
    return trackers, dataset


def super_dimp_latot_latot():
    """SuperDiMP with super_dimp_latot params (LaTOT fine-tuned net in param file or overridden by test script)."""
    trackers = trackerlist('dimp', 'super_dimp_latot', range(1))
    dataset = get_dataset('latot')
    return trackers, dataset


def small112_test():
    trackers = trackerlist('dimp', 'super_dimp', range(1))
    dataset = get_dataset('small112')
    return trackers, dataset


def small90_test():
    trackers = trackerlist('dimp', 'super_dimp', range(1))
    dataset = get_dataset('small90')
    return trackers, dataset


def small112_small90_test():
    trackers = trackerlist('dimp', 'super_dimp', range(1))
    dataset = get_dataset('small112', 'small90')
    return trackers, dataset
