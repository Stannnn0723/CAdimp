class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/lyx/project/pytracking-master/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.pregenerated_masks = ''
        self.dtb70_path = '/data/lyx/dataset/DTB70/DTB70/'
        self.latot_dir = '/data/lyx/dataset/LaTOT/LaTOT/'
        self.lasot_dir = '/data/lyx/dataset/LaSOT/LaSOTBenchmark/'
        self.got10k_dir = '/data/lyx/dataset/GOT-10k/GOT-10k/train/'
        self.trackingnet_dir = '/data/lyx/dataset/trackingnet/'
        self.coco_dir = '/data/lyx/dataset/coco/coco/'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_candidate_matching_dataset_path = ''
