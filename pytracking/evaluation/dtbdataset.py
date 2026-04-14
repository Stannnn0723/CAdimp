import os
import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class DTBDataset(BaseDataset):
    """ DTB70 dataset

    A drone tracking benchmark with 70 sequences.
    https://github.com/s大连理工大学/VisDrone2018-Tracking

    The dataset should be organized as:
    DTB70/
        Animal1/
            img/00001.jpg, 00002.jpg, ...
            groundtruth_rect.txt
        ...
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.dtb70_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_name = sequence_info['name']
        seq_base_path = os.path.join(self.base_path, sequence_name)
        img_folder = os.path.join(seq_base_path, 'img')
        
        if not os.path.exists(img_folder):
            raise FileNotFoundError(f"Image folder not found: {img_folder}")
        
        imgs = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        frames = [os.path.join(img_folder, img) for img in imgs]
        
        anno_path = os.path.join(seq_base_path, 'groundtruth_rect.txt')
        
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        try:
            ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        except Exception:
            try:
                ground_truth_rect = np.loadtxt(str(anno_path), delimiter=' ', dtype=np.float64)
            except Exception:
                ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        
        num_frames = len(frames)
        num_gt = ground_truth_rect.shape[0]
        
        if num_frames < num_gt:
            print(f"Warning: sequence {sequence_name} has {num_frames} frames but {num_gt} GT entries. Truncating GT.")
            ground_truth_rect = ground_truth_rect[:num_frames, :]
        elif num_frames > num_gt:
            print(f"Warning: sequence {sequence_name} has {num_frames} frames but {num_gt} GT entries. Truncating frames.")
            frames = frames[:num_gt]

        return Sequence(
            sequence_name,
            frames,
            'dtb70',
            ground_truth_rect,
            object_class=sequence_info.get('object_class', 'other')
        )

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Animal1", "object_class": "animal"},
            {"name": "Animal2", "object_class": "animal"},
            {"name": "Animal3", "object_class": "animal"},
            {"name": "Animal4", "object_class": "animal"},
            {"name": "Basketball", "object_class": "person"},
            {"name": "BMX2", "object_class": "person"},
            {"name": "BMX3", "object_class": "person"},
            {"name": "BMX4", "object_class": "person"},
            {"name": "BMX5", "object_class": "person"},
            {"name": "Car2", "object_class": "car"},
            {"name": "Car4", "object_class": "car"},
            {"name": "Car5", "object_class": "car"},
            {"name": "Car6", "object_class": "car"},
            {"name": "Car8", "object_class": "car"},
            {"name": "ChasingDrones", "object_class": "other"},
            {"name": "Girl1", "object_class": "person"},
            {"name": "Girl2", "object_class": "person"},
            {"name": "Gull1", "object_class": "bird"},
            {"name": "Gull2", "object_class": "bird"},
            {"name": "Horse1", "object_class": "animal"},
            {"name": "Horse2", "object_class": "animal"},
            {"name": "Kiting", "object_class": "person"},
            {"name": "ManRunning1", "object_class": "person"},
            {"name": "ManRunning2", "object_class": "person"},
            {"name": "Motor1", "object_class": "vehicle"},
            {"name": "Motor2", "object_class": "vehicle"},
            {"name": "MountainBike1", "object_class": "bicycle"},
            {"name": "MountainBike5", "object_class": "bicycle"},
            {"name": "MountainBike6", "object_class": "bicycle"},
            {"name": "Paragliding3", "object_class": "person"},
            {"name": "Paragliding5", "object_class": "person"},
            {"name": "RaceCar", "object_class": "car"},
            {"name": "RaceCar1", "object_class": "car"},
            {"name": "RcCar3", "object_class": "car"},
            {"name": "RcCar4", "object_class": "car"},
            {"name": "RcCar5", "object_class": "car"},
            {"name": "RcCar6", "object_class": "car"},
            {"name": "RcCar7", "object_class": "car"},
            {"name": "RcCar8", "object_class": "car"},
            {"name": "RcCar9", "object_class": "car"},
            {"name": "Sheep1", "object_class": "animal"},
            {"name": "Sheep2", "object_class": "animal"},
            {"name": "SkateBoarding4", "object_class": "person"},
            {"name": "Skiing1", "object_class": "person"},
            {"name": "Skiing2", "object_class": "person"},
            {"name": "SnowBoarding2", "object_class": "person"},
            {"name": "SnowBoarding4", "object_class": "person"},
            {"name": "SnowBoarding6", "object_class": "person"},
            {"name": "Soccer1", "object_class": "person"},
            {"name": "Soccer2", "object_class": "person"},
            {"name": "SpeedCar2", "object_class": "car"},
            {"name": "SpeedCar4", "object_class": "car"},
            {"name": "StreetBasketball1", "object_class": "person"},
            {"name": "StreetBasketball2", "object_class": "person"},
            {"name": "StreetBasketball3", "object_class": "person"},
            {"name": "SUP2", "object_class": "other"},
            {"name": "SUP4", "object_class": "other"},
            {"name": "SUP5", "object_class": "other"},
            {"name": "Surfing03", "object_class": "person"},
            {"name": "Surfing04", "object_class": "person"},
            {"name": "Surfing06", "object_class": "person"},
            {"name": "Surfing10", "object_class": "person"},
            {"name": "Surfing12", "object_class": "person"},
            {"name": "Vaulting", "object_class": "person"},
            {"name": "Wakeboarding1", "object_class": "person"},
            {"name": "Wakeboarding2", "object_class": "person"},
            {"name": "Walking", "object_class": "person"},
            {"name": "Yacht2", "object_class": "boat"},
            {"name": "Yacht4", "object_class": "boat"},
            {"name": "Zebra", "object_class": "animal"},
        ]
        return sequence_info_list