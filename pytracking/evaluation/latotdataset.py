import os
import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pathlib import Path


class LaToTDataset(BaseDataset):

    def __init__(self, vos_mode=False, attribute=None, test_sequences_path="/data/lyx/dataset/LaTOT/test_sequences.txt"):
        super().__init__()
        self.base_path = self.env_settings.latot_path
        # 新增：接收测试序列文件路径参数
        self.test_sequences_path = test_sequences_path
        self.sequence_info_list = self._get_sequence_info_list()
        self.sr_scale = 2

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        # sequence_info["path"] 已经是相对于 base_path 的完整路径（例如：'sequence_name/img'）
        sequence_path = sequence_info["path"]
        full_img_path = os.path.join(self.base_path, sequence_path)
        
        if not os.path.exists(full_img_path):
            raise FileNotFoundError(f"Image directory not found: {full_img_path}")
        
        imgs = os.listdir(full_img_path)
        frames = list()
        for img in sorted(imgs):
            frames.append(os.path.join(full_img_path, img))

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']
        
        # anno_path 也是相对路径
        anno_path = os.path.join(self.base_path, sequence_info['anno_path'])
        
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        # Try to load annotation file robustly (comma or whitespace separated)
        try:
            # First try comma delimiter
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
        except Exception:
            # Fallback to whitespace delimiter
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        # ground_truth_rect = ground_truth_rect * self.sr_scale#对应2倍超分

        return Sequence(sequence_info['name'], frames, 'latot', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        # 读取所有序列信息
        all_sequences = []
        
        # 检查 base_path 是否存在
        if not os.path.exists(self.base_path):
            print(f"Error: Base path does not exist: {self.base_path}")
            return []
        
        print(f"Scanning base path: {self.base_path}")
        
        # 直接从 base_path 获取第一层目录（序列目录）
        try:
            entries = os.listdir(self.base_path)
        except Exception as e:
            print(f"Error listing directory {self.base_path}: {e}")
            return []
        
        sequence_dirs = [d for d in entries 
                        if os.path.isdir(os.path.join(self.base_path, d))]
        sequence_dirs = sorted(sequence_dirs)
        
        print(f"Found {len(sequence_dirs)} potential sequence directories")
        
        # 构建所有序列的信息列表
        for seq_name in sequence_dirs:
            seq_path = os.path.join(self.base_path, seq_name)
            img_folder = os.path.join(seq_path, 'img')
            anno_file = os.path.join(seq_path, seq_name + '.txt')
            
            # 检查必要的文件是否存在
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
                "path": seq_name + '/img',  # 相对于 base_path 的路径
                "startFrame": 1,
                "endFrame": num_frames,
                "nz": 6,
                "ext": 'jpg',
                "anno_path": os.path.join(seq_name, seq_name + '.txt'),  # 相对路径
                "object_class": 'person'
            }
            all_sequences.append(sequence_info)

        print(f"Successfully processed {len(all_sequences)} sequences")

        # 读取指定测试序列列表并过滤
        if self.test_sequences_path and os.path.exists(self.test_sequences_path):
            # 读取txt文件中的序列名称（每行一个序列名）
            with open(self.test_sequences_path, 'r') as f:
                test_sequence_names = set([line.strip() for line in f if line.strip()])
            print(f"Loaded {len(test_sequence_names)} test sequences from {self.test_sequences_path}")

            # 只保留在指定列表中的序列
            filtered_sequences = [seq for seq in all_sequences
                                  if seq["name"] in test_sequence_names]
            
            print(f"Filtered to {len(filtered_sequences)} sequences")
            
            # 打印未找到的序列（如果有）
            found_names = set([seq["name"] for seq in filtered_sequences])
            missing = test_sequence_names - found_names
            if missing:
                print(f"Warning: {len(missing)} sequences from test list not found in dataset")
                if len(missing) <= 10:  # 只打印前10个缺失的序列
                    print(f"Missing sequences: {sorted(list(missing))}")
                
            return filtered_sequences
        else:
            # 如果没有指定文件或文件不存在，返回所有序列
            print("Test sequences file not found, using all sequences")
            return all_sequences