import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import re
from scipy import signal

class AVFDataset(Dataset):
    def __init__(self, data_folder, segment_length=600, apply_filter=True):
        self.data_folder = data_folder
        self.segment_length = segment_length
        self.apply_filter = apply_filter
        self.file_paths = glob.glob(os.path.join(data_folder, "*.csv"))

        print(f"找到 {len(self.file_paths)} 个 CSV 文件")
        if len(self.file_paths) == 0:
            raise ValueError(f"数据集为空，请检查 {data_folder} 是否有 CSV 文件！")

        self.segments = []
        self.labels = []
        self.file_names = []

        for file_path in tqdm(self.file_paths, desc="加载 AVF 数据"):
            try:
                # 从文件名提取标签 (.csv前面的数字)
                file_name = os.path.basename(file_path)
                label_match = re.search(r'(\d+)\.csv$', file_name)
                if not label_match:
                    print(f"警告：无法从文件名解析标签: {file_path}")
                    continue
                label = int(label_match.group(1))
                if label not in [0, 1, 2]:
                    print(f"警告：标签不在[0,1,2]范围内: {file_path}, 标签为: {label}")
                    continue

                df = pd.read_csv(file_path)
                if 'PPG' not in df.columns:
                    print(f"警告：文件 {file_path} 没有 'PPG' 列，跳过！")
                    continue

                if len(df) < segment_length:
                    print(f"警告：文件 {file_path} 数据不足 {segment_length} 行，跳过！")
                    continue

                # 提取PPG信号
                segment = df['PPG'].values[:segment_length]
                
                # 检查信号质量
                if np.isnan(segment).any():
                    print(f"警告：文件 {file_path} 含有 NaN，跳过！")
                    continue
                
                if np.std(segment) < 1e-6:
                    print(f"警告：文件 {file_path} 信号无变化，跳过！")
                    continue
                
                # 应用带通滤波器去除噪声
                if self.apply_filter:
                    # 带通滤波器 (0.5-10 Hz)
                    fs = 50  # 采样频率50Hz
                    nyquist = 0.5 * fs
                    low = 0.5 / nyquist
                    high = 10.0 / nyquist
                    b, a = signal.butter(4, [low, high], btype='band')
                    segment = signal.filtfilt(b, a, segment)
                
                # 标准化
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-6)
                
                self.segments.append(segment)
                self.labels.append(label)
                self.file_names.append(file_name)

            except Exception as e:
                print(f"错误：加载 {file_path} 失败: {e}")

        print(f"最终加载了 {len(self.segments)} 条有效数据")
        
        # 打印每个类别的样本数量
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print("各类别样本数量:")
        for lbl, cnt in zip(unique_labels, counts):
            print(f"类别 {lbl}: {cnt}个样本")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        file_name = self.file_names[idx]
        
        # 不使用数据增强，直接返回原始信号
        return {
            'signal': torch.FloatTensor(segment),
            'label': torch.tensor(label, dtype=torch.long),
            'file_name': file_name
        }

# 设置随机种子函数
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 掩码生成器类（为跨文件依赖准备）
class MaskGenerator:
    def __init__(self, mask_ratio=0.20):
        self.mask_ratio = mask_ratio
        
    def generate_mask(self, length):
        num_masks = int(length * self.mask_ratio)
        mask = torch.randperm(length)[:num_masks]
        return mask 