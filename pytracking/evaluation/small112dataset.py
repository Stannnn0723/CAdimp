import os
import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class Small112Dataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.small112_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info["path"]
        full_img_path = os.path.join(self.base_path, sequence_path)

        if not os.path.exists(full_img_path):
            raise FileNotFoundError(f"Image directory not found: {full_img_path}")

        imgs = [f for f in os.listdir(full_img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        imgs = sorted(imgs)
        frames = [os.path.join(full_img_path, img) for img in imgs]

        anno_path = os.path.join(self.base_path, sequence_info['anno_path'])

        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        try:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
        except Exception:
            try:
                ground_truth_rect = np.loadtxt(str(anno_path), delimiter=' ', dtype=np.float64)
            except Exception:
                ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)

        num_frames = len(frames)
        num_gt = ground_truth_rect.shape[0]

        if num_frames < num_gt:
            print(f"Warning: sequence {sequence_info['name']} has {num_frames} frames but {num_gt} GT entries. "
                  f"Truncating GT to match frames.")
            ground_truth_rect = ground_truth_rect[:num_frames, :]
        elif num_frames > num_gt:
            print(f"Warning: sequence {sequence_info['name']} has {num_frames} frames but only {num_gt} GT entries. "
                  f"Truncating frames to match GT.")
            frames = frames[:num_gt]

        return Sequence(
            sequence_info['name'], frames, 'small112',
            ground_truth_rect, object_class='unknown'
        )

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        all_sequences = []

        if not os.path.exists(self.base_path):
            print(f"Error: Base path does not exist: {self.base_path}")
            return []

        print(f"Scanning base path: {self.base_path}")

        try:
            entries = os.listdir(self.base_path)
        except Exception as e:
            print(f"Error listing directory {self.base_path}: {e}")
            return []

        sequence_dirs = [d for d in entries if os.path.isdir(os.path.join(self.base_path, d))]
        sequence_dirs = sorted(sequence_dirs)

        print(f"Found {len(sequence_dirs)} potential sequence directories")

        for seq_name in sequence_dirs:
            seq_path = os.path.join(self.base_path, seq_name)
            img_folder = os.path.join(seq_path, 'img')
            anno_file = os.path.join(seq_path, 'groundtruth_rect.txt')

            if not os.path.exists(img_folder):
                print(f"Warning: img folder not found for sequence {seq_name}, skipping")
                continue

            if not os.path.exists(anno_file):
                print(f"Warning: annotation file not found for sequence {seq_name}, skipping")
                continue

            try:
                files = os.listdir(img_folder)
                num_frames = len([f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))])
            except Exception as e:
                print(f"Warning: Error reading img folder for {seq_name}: {e}, skipping")
                continue

            if num_frames == 0:
                print(f"Warning: No image files found for sequence {seq_name}, skipping")
                continue

            sequence_info = {
                "name": seq_name,
                "path": os.path.join(seq_name, 'img'),
                "anno_path": os.path.join(seq_name, 'groundtruth_rect.txt'),
            }
            all_sequences.append(sequence_info)

        print(f"Successfully processed {len(all_sequences)} sequences")
        return all_sequences
