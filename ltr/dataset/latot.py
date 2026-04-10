import torch
import os
import numpy as np
import random
from collections import OrderedDict
from lib.train.data import jpeg4py_loader  # 若没有可替换成 PIL.Image.open
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings

class LaToT(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, seed=42, split='train'):
        self.root = env_settings().latot_dir if root is None else root
        super().__init__('LaToT', self.root, image_loader)

        # 所有序列文件夹名
        all_sequences = sorted(os.listdir(self.root))

        # 读取测试序列文件
        test_file_path = os.path.join(self.root, 'test_sequences.txt')
        if os.path.exists(test_file_path):
            with open(test_file_path, 'r') as f:
                test_sequences = [line.strip() for line in f if line.strip()]
            test_sequences = set(test_sequences)
        else:
            test_sequences = set()

        # 根据 split 选择序列
        if split == 'train':
            self.sequence_list = [seq for seq in all_sequences if seq not in test_sequences]
        elif split == 'test':
            self.sequence_list = [seq for seq in all_sequences if seq in test_sequences]
        else:
            raise ValueError(f"split 必须是 'train' 或 'test'，当前是 {split}")

        # 采样数据
        if data_fraction is not None:
            random.seed(seed)
            sample_size = int(len(self.sequence_list) * data_fraction)
            self.sequence_list = random.sample(self.sequence_list, sample_size)

        # 预存每个序列的图像文件名
        self.seq_img_map = {}
        for seq_name in self.sequence_list:
            img_dir = os.path.join(self.root, seq_name, 'img')
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"图像目录不存在: {img_dir}")
            img_files = sorted(
                [f for f in os.listdir(img_dir)
                 if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            if not img_files:
                raise ValueError(f"序列 {seq_name} 的 img 目录中无有效图像文件")
            self.seq_img_map[seq_name] = img_files

    def get_name(self):
        return 'LaToT'  # 返回数据集名称，用于标识

    def _read_bb_anno(self, seq_path):
        """读取单个序列的边界框标注文件"""
        anno_path = os.path.join(seq_path, f"{os.path.basename(seq_path)}.txt")
        if not os.path.isfile(anno_path):
            raise FileNotFoundError(f"标注文件 {anno_path} 不存在")
        # 尝试加载标注，支持逗号、制表符、空格分隔
        try:
            gt = np.loadtxt(anno_path, delimiter=',', dtype=np.float32)
        except ValueError:
            try:
                gt = np.loadtxt(anno_path, delimiter='\t', dtype=np.float32)
            except ValueError:
                try:
                    gt = np.loadtxt(anno_path, dtype=np.float32)
                except Exception as e:
                    raise RuntimeError(f"加载标注文件 {anno_path} 失败: {str(e)}")
        # 处理单帧标注（维度为 1 时扩展为 2 维）
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)
        # 检查标注维度，需是 (帧数, 4) ，对应 x1,y1,w,h
        if gt.shape[1] != 4:
            raise ValueError(f"标注文件 {anno_path} 格式错误，需为 4 列（x1,y1,w,h），实际 {gt.shape[1]} 列")
        return torch.tensor(gt)

    def get_sequence_info(self, seq_name):
        """获取单个序列的边界框、有效性等信息"""
        seq_path = os.path.join(self.root, self.sequence_list[seq_name])
        bbox = self._read_bb_anno(seq_path)
        # 标记有效边界框（宽和高需大于 0，可根据实际需求调整阈值）
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()  # 简单认为可见性与有效性一致，可按需修改
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        """加载单个序列指定帧的图像"""
        frame_path = os.path.join(seq_path, 'img', sorted([p for p in os.listdir(os.path.join(seq_path, 'img')) if
                                                           os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])[frame_id])
        frame = self.image_loader(frame_path)
        return frame

    def get_frames(self, seq_name, frame_ids, anno=None):
        """获取指定序列、指定帧的图像和对应标注"""
        # 加载图像
        seq_path = os.path.join(self.root, self.sequence_list[seq_name])
        # frame_list = [self._get_frame(seq_name, f) for f in frame_ids]
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]

        # 加载标注（未传入时自动获取）
        if anno is None:
            anno = self.get_sequence_info(seq_name)

        # 提取指定帧的标注信息
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id].clone() for f_id in frame_ids]

        # 目标元信息（根据实际数据集，若没有则设为 None）
        object_meta = OrderedDict({
            'object_class_name': None,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta