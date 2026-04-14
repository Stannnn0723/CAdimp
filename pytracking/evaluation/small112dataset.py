import os
import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class Small112Dataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.small112_path
        self.small90_path = getattr(self.env_settings, 'small90_path', '/data/lyx/dataset/small90/small90/')
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        base_path = sequence_info.get("base_path", self.base_path)
        sequence_path = sequence_info["path"]
        full_img_path = os.path.join(base_path, sequence_path)

        if not os.path.exists(full_img_path):
            raise FileNotFoundError(f"Image directory not found: {full_img_path}")

        imgs = [f for f in os.listdir(full_img_path) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP'))]
        imgs = sorted(imgs)
        frames = [os.path.join(full_img_path, img) for img in imgs]

        anno_path = os.path.join(base_path, sequence_info['anno_path'])

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

        # Use different alignment logic based on which dataset the sequence belongs to
        if "uav" in sequence_info['name'] and base_path == self.base_path: 
            # small112 jump-frame logic
            try:
                import re
                frame_indices = [int(re.search(r'\d+', os.path.basename(img)).group()) for img in frames]
                
                aligned_gt = []
                for idx in frame_indices:
                    gt_idx = idx - 1
                    if gt_idx < 0 or gt_idx >= num_gt:
                        aligned_gt.append(ground_truth_rect[0] if num_gt > 0 else [0,0,0,0])
                    else:
                        aligned_gt.append(ground_truth_rect[gt_idx])
                        
                ground_truth_rect = np.array(aligned_gt)
            except Exception as e:
                print(f"Warning: Failed to align GT via frame indices for {sequence_info['name']}: {e}")
                if num_frames < num_gt:
                    ground_truth_rect = ground_truth_rect[:num_frames, :]
                elif num_frames > num_gt:
                    frames = frames[:num_gt]
        else:
            # small90 standard sequential truncation logic
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

        def scan_dataset(base_path):
            if not os.path.exists(base_path):
                print(f"Error: Base path does not exist: {base_path}")
                return []

            print(f"Scanning base path: {base_path}")

            try:
                entries = os.listdir(base_path)
            except Exception as e:
                print(f"Error listing directory {base_path}: {e}")
                return []

            sequence_dirs = [d for d in entries if os.path.isdir(os.path.join(base_path, d))]
            sequence_dirs = sorted(sequence_dirs)
            
            scanned_seqs = []
            for seq_name in sequence_dirs:
                seq_path = os.path.join(base_path, seq_name)
                img_folder = os.path.join(seq_path, 'img')
                anno_file = os.path.join(seq_path, 'groundtruth_rect.txt')

                if not os.path.exists(img_folder) or not os.path.exists(anno_file):
                    continue

                try:
                    files = os.listdir(img_folder)
                    num_frames = len([f for f in files if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP'))])
                except Exception as e:
                    continue

                if num_frames == 0:
                    continue

                sequence_info = {
                    "name": seq_name,
                    "base_path": base_path,
                    "path": os.path.join(seq_name, 'img'),
                    "anno_path": os.path.join(seq_name, 'groundtruth_rect.txt'),
                }
                scanned_seqs.append(sequence_info)
            return scanned_seqs

        # Combine small112 and small90 sequences
        all_sequences.extend(scan_dataset(self.base_path))
        all_sequences.extend(scan_dataset(self.small90_path))

        print(f"Successfully processed {len(all_sequences)} sequences combined")
        return all_sequences
