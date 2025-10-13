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

    def generate_mask_batch(self, batch_size: int, signal_length: int, device=None) -> torch.Tensor:
        """
        批量生成布尔掩码 [B, L]，向量化实现，支持 GPU。
        当前采用近似策略：先按 mask_ratio 计算每样本需要掩码的点数，再随机选取若干起点与段长，写入掩码张量。
        为避免 Python 循环，段数量固定为根据期望长度估算的上限，并用广播写入；不足的部分通过裁剪控制。
        """
        if device is None:
            device = torch.device('cpu')
        B, L = batch_size, signal_length
        dtype_bool = torch.bool

        total_mask_points = int(L * self.mask_ratio)
        if total_mask_points <= 0:
            return torch.zeros(B, L, device=device, dtype=dtype_bool)

        # 估算每段平均长度与段数上限（避免 while），长度>=1
        mean_len = max(1, int(self.mask_length_mean))
        # 至少 1 段，最多不超过信号长度
        max_segments = max(1, min(total_mask_points, L) // mean_len + 1)

        # 采样每个样本、每段的长度（截断在 [1, total_mask_points]）
        # 用正态近似，后续裁剪到合法范围
        lens = torch.normal(
            mean=float(self.mask_length_mean), std=float(self.mask_length_std),
            size=(B, max_segments), device=device
        ).clamp(min=1.0, max=float(total_mask_points)).round().to(torch.long)

        # 保证每样本总长度不超过 total_mask_points：按行做前缀和并截断
        cumsum = torch.cumsum(lens, dim=1)
        # 构造掩码，标记哪些段被保留（前缀和<=total_mask_points）
        keep = cumsum <= total_mask_points
        # 对超过的段长度置 0
        lens = lens * keep.to(lens.dtype)

        # 采样起点，确保 start+len<=L
        # 为避免非法起点，先将 len==0 的段起点置 0，不会生效
        max_start = torch.clamp(L - lens, min=0)
        # torch.randint 的 high 需要 >=1，处理全 0 的位置
        high = torch.maximum(max_start, torch.ones_like(max_start))
        # 逐元素向量化起点采样：rand*[0,high) 向下取整，避免 .item() 与 graph break
        starts = (torch.rand(B, max_segments, device=device, dtype=torch.float32) * high.to(torch.float32)).floor().to(torch.long)

        # 构造索引：为每段生成 [start, start+len) 的索引范围
        # 展开为 [B, S, L] 的布尔矩阵，指示该段覆盖的位置
        idx = torch.arange(L, device=device).view(1, 1, L)
        seg_mask = (idx >= starts.unsqueeze(-1)) & (idx < (starts + lens).unsqueeze(-1))  # [B,S,L]
        # 聚合所有段
        mask = seg_mask.any(dim=1)  # [B,L]
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

    def add_gaussian_noise_batch(self, signals: torch.Tensor, sigma_range=(0.01, 0.05)) -> torch.Tensor:
        """
        基于 torch 的批量高斯噪声增强，保持与输入相同 device 与 dtype。
        参数:
            signals: [B, L] 的张量
            sigma_range: (min, max)
        返回:
            [B, L] 的张量
        """
        if signals.ndim != 2:
            raise ValueError("signals must be 2D tensor [B, L]")
        device = signals.device
        dtype = signals.dtype
        batch_size, length = signals.shape
        sigma_min, sigma_max = sigma_range
        sigmas = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(sigma_min, sigma_max)
        noise = torch.randn(batch_size, length, device=device, dtype=dtype) * sigmas
        return signals + noise







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
            'sin'   : 原单频正弦（
            'mix'   : 扫频 + 随机游走
            
        返回
        --------------------------------------------------------------
        out : ndarray
            与输入同 shape，添加漂移后的信号
        """
        signal = np.asarray(signal, dtype=np.float32)
        length = signal.shape[-1]
        t = np.arange(length, dtype=np.float32) / fs   # 秒时间轴
            
        if mode == 'sin': 
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


    def add_baseline_wander_batch(self, signals: torch.Tensor,
                                  fs: int = 50,
                                  mode: str = 'mix') -> torch.Tensor:
        """
        基于 torch 的批量基线漂移（默认正弦版本，便于在 GPU 上高效运行）。
        参数:
            signals: [B, L]
            fs: 采样率
            mode: 目前支持 'sin'（更快）和 'mix'（更真实）。
        返回:
            [B, L]
        """
        if signals.ndim != 2:
            raise ValueError("signals must be 2D tensor [B, L]")
        device = signals.device
        dtype = signals.dtype
        batch_size, length = signals.shape
        t = torch.arange(length, device=device, dtype=dtype) / float(fs)
        if mode == 'sin':
            # 为每个样本采样频率与幅值
            freq = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.05, 0.1)
            amp = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.02, 0.05)
            baseline = amp * torch.sin(2 * torch.pi * freq * t)  # [B, L]
            return signals + baseline
        elif mode == 'mix':
            # 1) 呼吸扫频 0.15–0.4 Hz，幅度 1–3 %
            f_breath = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.15, 0.4)
            f_dev    = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.02, 0.05)
            amp_b    = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.01, 0.03)
            inner = 2 * torch.pi * 0.05 * t  # 0.05 Hz 调制
            breath_freq = f_breath + f_dev * torch.sin(inner)  # [B,1] + [1,L]广播为 [B,L]
            breath = amp_b * torch.sin(2 * torch.pi * breath_freq * t)  # [B,L]

            # 2) 随机游走 + 简单高通（IIR 一阶高通，fc≈0.02Hz）
            # 生成高斯噪声并累加为随机游走
            walk_noise = torch.randn(batch_size, length, device=device, dtype=dtype) * 0.005
            walk = torch.cumsum(walk_noise, dim=1)
            # 一阶高通滤波：y[n] = alpha*(y[n-1] + x[n] - x[n-1])
            fc = 0.02
            dt = 1.0 / float(fs)
            rc = 1.0 / (2.0 * torch.pi * fc)
            alpha = (rc / (rc + dt)).to(dtype)
            y = torch.zeros(batch_size, length, device=device, dtype=dtype)
            # 初始化
            y[:, 0] = 0.0
            for n in range(1, length):
                y[:, n] = alpha * (y[:, n - 1] + walk[:, n] - walk[:, n - 1])

            baseline = breath + y
            return signals + baseline
        else:
            raise ValueError("mode must be 'sin' or 'mix'")
    






#--------------------------------随机缩放--------------------------------







    def random_scale(self, signal, scale_range=(0.9, 1.1)):
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

    def random_scale_batch(self, signals: torch.Tensor, scale_range=(0.9, 1.1)) -> torch.Tensor:
        """
        基于 torch 的批量随机幅度缩放。
        参数:
            signals: [B, L]
            scale_range: (min, max)
        返回:
            [B, L]
        """
        if signals.ndim != 2:
            raise ValueError("signals must be 2D tensor [B, L]")
        device = signals.device
        dtype = signals.dtype
        batch_size = signals.shape[0]
        smin, smax = scale_range
        scales = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(smin, smax)
        return signals * scales
    




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
            augmented = self.add_baseline_wander(augmented, fs=50, mode='mix')
        if 2 in augs:
            augmented = self.random_scale(augmented)
            
        return augmented 
        '''

        对于PPG信号，当前的顺序可能更合理，因为：
        基线漂移通常反映生理状态（呼吸、运动）
        噪声通常是测量设备的固有特性
        幅度缩放可能反映个体差异或测量条件
        在真实场景中，个体差异（幅度缩放）应该影响整个测量结果，包括噪声和漂移。
                
        '''

    def augment_batch(self, signals: torch.Tensor,
                      use_gaussian_noise: bool = True,
                      use_baseline_wander: bool = True,
                      use_random_scale: bool = True,
                      fs: int = 50) -> torch.Tensor:
        """
        批量增强：对 [B, L] 张量进行 1-2 种轻微增强（逐样本随机）。
        为了速度，基线漂移采用 sin 模式，所有操作在 signals.device 上执行。
        返回新的张量，不修改原张量。
        """
        if signals.ndim != 2:
            raise ValueError("signals must be 2D tensor [B, L]")
        device = signals.device
        out = signals.clone()

        batch_size = signals.shape[0]
        # 随机决定每个样本应用多少种增强（1 或 2）
        num_augs = torch.randint(low=1, high=3, size=(batch_size,), device=device)

        # 为每个样本随机选择两种增强（可能会重复，稍后用 num_augs 控制）
        choices = torch.randint(low=0, high=3, size=(batch_size, 2), device=device)

        # 掩码：每个样本是否启用某增强
        apply_noise = (choices[:, 0] == 0) | ((choices[:, 1] == 0) & (num_augs == 2))
        apply_baseline = (choices[:, 0] == 1) | ((choices[:, 1] == 1) & (num_augs == 2))
        apply_scale = (choices[:, 0] == 2) | ((choices[:, 1] == 2) & (num_augs == 2))

        if use_gaussian_noise and apply_noise.any():
            out_noise = self.add_gaussian_noise_batch(out[apply_noise])
            out[apply_noise] = out_noise

        if use_baseline_wander and apply_baseline.any():
            out_bl = self.add_baseline_wander_batch(out[apply_baseline], fs=fs, mode='sin')
            out[apply_baseline] = out_bl

        if use_random_scale and apply_scale.any():
            out_sc = self.random_scale_batch(out[apply_scale])
            out[apply_scale] = out_sc

        return out

    def augment_pair_batch(self, signals: torch.Tensor,
                           use_gaussian_noise: bool = True,
                           use_baseline_wander: bool = True,
                           use_random_scale: bool = True,
                           fs: int = 50) -> tuple:
        """
        生成两组独立随机的数据增强视图（对比学习常用）。
        输入/输出均为 [B, L]，保持与输入相同的 device 与 dtype。
        """
        x1 = self.augment_batch(signals, use_gaussian_noise, use_baseline_wander, use_random_scale, fs)
        x2 = self.augment_batch(signals, use_gaussian_noise, use_baseline_wander, use_random_scale, fs)
        return x1, x2