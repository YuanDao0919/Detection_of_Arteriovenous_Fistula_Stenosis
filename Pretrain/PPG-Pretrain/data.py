import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


#--------------------------------掩码生成器--------------------------------


class MaskGenerator:
    """
    掩码生成器：负责在预训练过程中生成PPG信号的随机掩码
    用于掩码重构自监督学习任务
    """
    def __init__(self, mask_ratio=0.15, mask_length_mean=55, mask_length_std=2):
        """
        初始化掩码生成器
        
        参数:
            mask_ratio (float): 需要被掩码的信号比例
            mask_length_mean (int): 掩码段长度的均值
            mask_length_std (int): 掩码段长度的标准差
        """
        self.mask_ratio = mask_ratio
        self.mask_length_mean = mask_length_mean
        self.mask_length_std = mask_length_std
    
    def generate_mask(self, signal_length):
        """
        为给定长度的信号生成随机掩码
        
        参数:
            signal_length (int): 信号长度
            
        返回:
            torch.Tensor: 布尔型掩码 其中True表示被掩码的位置
        """
        # 计算需要掩码的总点数
        num_mask_points = int(signal_length * self.mask_ratio)
        mask = torch.zeros(signal_length, dtype=torch.bool)
        
        # 如果掩码点数太少，直接返回空掩码
        if num_mask_points <= 0:
            return mask
        
        # 生成连续的掩码段
        remaining_points = num_mask_points
        while remaining_points > 0:
            # 随机确定掩码段长度
            length = min(
                int(np.random.normal(self.mask_length_mean, self.mask_length_std)),
                remaining_points
            )
            length = max(1, length)  # 确保长度至少为1
            
            # 随机选择掩码段起始位置
            start = np.random.randint(0, signal_length - length + 1)
            
            # 应用掩码
            mask[start:start+length] = True
            
            remaining_points -= length
        
        return mask


#--------------------------------数据集--------------------------------



class PPGDataset(Dataset):
    """
    PPG数据集类：负责加载和处理PPG信号数据
    """
    def __init__(self, data_dir, signal_length=600):
        """
        初始化PPG数据集
        
        参数:
            data_dir (str): 包含CSV文件的数据目录路径
            signal_length (int): 每个PPG信号段的长度
        """
        self.data_dir = data_dir
        self.signal_length = signal_length
        self.csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        self.data = []
        
        # 加载所有CSV文件中的PPG信号段
        for file in self.csv_files:
        #     df = pd.read_csv(file, header=None)
        #     signals = df.iloc[:, :self.signal_length].values
        #     self.data.extend(signals)
        
            # 添加错误处理
            try:
                df = pd.read_csv(file, header=None)
                # 检查数据形状
                if df.shape[1] >= self.signal_length:
                    signals = df.iloc[:, :self.signal_length].values
                    self.data.extend(signals)
                else:
                    print(f"警告：文件 {file} 的列数不足 {self.signal_length}")
            except Exception as e:
                print(f"读取文件 {file} 时出错：{e}")

        self.data = np.array(self.data)
        print(f"加载了 {len(self.data)} 个PPG信号段，每个长度为 {self.signal_length}")
        
    def __len__(self):
        #返回数据集中的样本数量
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取指定索引的数据样本
        signal = self.data[idx]
        signal = torch.tensor(signal, dtype=torch.float32)
        return signal



#--------------------------------数据增强器--------------------------------


class DataAugmenter:
    """
    数据增强器类：用于PPG信号的增强，生成正样本对 ----高斯噪声，基线漂移，随机缩放----
    """

    def __init__(self):
        #初始化数据增强器
        pass
    





#--------------------------------高斯噪声--------------------------------






    def add_gaussian_noise(self, signal, sigma_range=(0.01, 0.05)):
        """
        添加高斯噪声
        
        参数:
            signal (ndarray): 输入信号
            sigma_range (tuple): 高斯噪声标准差的范围
            
        返回:
            ndarray: 添加噪声后的信号
        """
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        noise = np.random.normal(0, sigma, signal.shape)
        return signal + noise







#--------------------------------基线漂移--------------------------------







    
    # def add_baseline_wander(self, signal, length):
    #     """
    #     添加基线漂移
        
    #     参数:
    #         signal (ndarray): 输入信号
    #         length (int): 信号长度
            
    #     返回:
    #         ndarray: 添加基线漂移后的信号
    #     """
    #     t = np.arange(length)
    #     freq = np.random.uniform(0.05, 0.1)  # 漂移频率范围
    #     amp = np.random.uniform(0.02, 0.05)   # 轻微漂移幅度范围
    #     baseline = amp * np.sin(2 * np.pi * freq * t / length)
    #     return signal + baseline

    def add_baseline_wander(self, signal: np.ndarray,
                            fs: int = 50,
                            mode: str = 'mix') -> np.ndarray:
        """
        添加基线漂移（优化版）
        
        参数
        --------------------------------------------------------------
        signal : ndarray
            输入信号，shape (length,) 或 (batch, length)
        fs : int
            采样率，用于把频率转成采样点
        mode : str
            'sin'   : 原单频正弦（可复现论文）
            'mix'   : 扫频 + 随机游走（推荐，更真实）
            
        返回
        --------------------------------------------------------------
        out : ndarray
            与输入同 shape，添加漂移后的信号
        """
        signal = np.asarray(signal, dtype=np.float32)
        length = signal.shape[-1]
        t = np.arange(length, dtype=np.float32) / fs   # 秒时间轴
            
        if mode == 'sin':                     # === 原论文风格 ===
            freq = np.random.uniform(0.05, 0.1)          # Hz
            amp  = np.random.uniform(0.02, 0.05)         # 幅值
            baseline = amp * np.sin(2 * np.pi * freq * t)
            
        elif mode == 'mix':                   # === 真实漂移 ===
            # 1) 呼吸扫频 0.15–0.4 Hz，幅度 1–3 %
            f_breath = np.random.uniform(0.15, 0.4)
            f_dev    = np.random.uniform(0.02, 0.05)          # 慢扫频
            amp_b    = np.random.uniform(0.01, 0.03)
            breath = amp_b * np.sin(2 * np.pi * (f_breath + f_dev * np.sin(2 * np.pi * 0.05 * t)) * t)
            
            # 2) 随机游走（超低频漂移）
            walk = np.random.normal(0, 0.005, size=t.shape)
            walk = np.cumsum(walk)                      # 1/f 形态
            # 用 0.02 Hz 零相高通去掉直流&超低端，保形
            walk = self._butter_highpass(walk, 0.02, fs)
            
            baseline = breath + walk
            
        else:
            raise ValueError("mode must be 'sin' or 'mix'")
        
        # 支持批量维度
        if signal.ndim == 2:
            baseline = baseline[np.newaxis, :]   # (1, L) 广播
        
        return signal + baseline
    
    # ---- 辅助：零相高通 ----
    @staticmethod
    def _butter_highpass(data, cutoff, fs, order=2):
        from scipy.signal import butter, filtfilt
        b, a = butter(order, cutoff / (fs / 2), btype='high')
        return filtfilt(b, a, data).astype(np.float32)
    






#--------------------------------随机缩放--------------------------------







    def ryandom_scale(self, signal, scale_range=(0.9, 1.1)):
        """
        随机缩放信号幅度
        
        参数:
            signal (ndarray): 输入信号
            scale_range (tuple): 缩放因子的范围
            
        返回:
            ndarray: 缩放后的信号
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return signal * scale
    




    def augment(self, signal):
        """
        随机应用1-2种轻微增强方法
        
        参数:
            signal (ndarray): 输入信号
            
        返回:
            ndarray: 增强后的信号
        """
        augmented = signal.copy()  # ->创建输入信号的副本，避免修改原始数据
        
        num_augs = np.random.randint(1, 3) # ->随机选择1-2种增强方法

        augs = np.random.choice([0, 1, 2], size=num_augs, replace=False) # ->随机选择具体的增强方法
        
        # -> 0代表高斯噪声，1代表基线漂移，2代表随机缩放
        if 0 in augs:
            augmented = self.add_gaussian_noise(augmented)
        if 1 in augs:
            augmented = self.add_baseline_wander(augmented, len(signal))
        if 2 in augs:
            augmented = self.random_scale(augmented)
            
        return augmented
        # noinspection PyUnreachableCode
        '''
        
        对于PPG信号，当前的顺序可能更合理，因为：
        
            基线漂移通常反映生理状态（呼吸、运动）
            噪声通常是测量设备的固有特性
            幅度缩放可能反映个体差异或测量条件
            在真实场景中，个体差异（幅度缩放）应该影响整个测量结果，包括噪声和漂移。
                        
        '''