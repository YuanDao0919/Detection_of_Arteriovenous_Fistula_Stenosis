import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import time
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 设置随机种子，确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
# 掩码策略：PPG信号的掩码生成
class MaskGenerator:
    def __init__(self, mask_ratio=0.15, mask_length_mean=15, mask_length_std=5):
        self.mask_ratio = mask_ratio
        self.mask_length_mean = mask_length_mean
        self.mask_length_std = mask_length_std
    
    def generate_mask(self, signal_length):
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
# 定义PPG数据集
class PPGDataset(Dataset):
    def __init__(self, data_dir, signal_length=600):
        self.data_dir = data_dir
        self.signal_length = signal_length
        self.csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        self.data = []
        
        # 加载所有CSV文件中的PPG信号段
        for file in self.csv_files:
            df = pd.read_csv(file, header=None)
            signals = df.iloc[:, :self.signal_length].values
            self.data.extend(signals)
        
        self.data = np.array(self.data)
        print(f"加载了 {len(self.data)} 个PPG信号段，每个长度为 {self.signal_length}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        signal = self.data[idx]
        signal = torch.tensor(signal, dtype=torch.float32)
        return signal

# 数据增强类，用于生成正样本对
class DataAugmenter:
    def __init__(self):
        pass
    
    def add_gaussian_noise(self, signal, sigma_range=(0.005, 0.02)):
        """添加高斯噪声 - 轻微噪声"""
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        noise = np.random.normal(0, sigma, signal.shape)
        return signal + noise
    
    def add_baseline_wander(self, signal, length):
        """添加基线漂移 - 轻微漂移"""
        t = np.arange(length)
        freq = np.random.uniform(0.05, 0.1)  # 漂移频率范围
        amp = np.random.uniform(0.02, 0.05)   # 轻微漂移幅度范围
        baseline = amp * np.sin(2 * np.pi * freq * t / length)
        return signal + baseline
    
    def random_scale(self, signal, scale_range=(0.9, 1.1)):
        """随机缩放信号幅度 - 轻微缩放"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return signal * scale
    
    def augment(self, signal):
        """随机应用1-2种轻微增强方法，不包括时间扭曲"""
        augmented = signal.copy()
        
        # 随机选择1-2种增强方法，不包括时间扭曲
        num_augs = np.random.randint(1, 3)
        augs = np.random.choice([0, 1, 2], size=num_augs, replace=False)
        
        if 0 in augs:
            augmented = self.add_gaussian_noise(augmented)
        if 1 in augs:
            augmented = self.add_baseline_wander(augmented, len(signal))
        if 2 in augs:
            augmented = self.random_scale(augmented)
            
        return augmented

# 编码器网络
class ResidualGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super(ResidualGRU, self).__init__()
        self.gru_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.gru_layers.append(nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=True
                ))
            else:
                self.gru_layers.append(nn.GRU(
                    input_size=hidden_size * 2,  # 双向GRU的输出大小
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout
                ))
    
    def forward(self, x):
        # 输入x的形状: [batch_size, seq_len, features]
        h = x
        for i, gru in enumerate(self.gru_layers):
            # 应用GRU层
            output, _ = gru(h)
            
            # 应用残差连接（从第二层开始）
            if i > 0:
                h = output + h
            else:
                h = output
                
        return h

class Encoder(nn.Module):
    def __init__(self, input_size=600, hidden_size=128, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        
        # 1x1卷积投影层，将通道数从1扩展到32
        self.projection_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # 多尺度卷积模块
        # 小尺度卷积 - 局部特征
        self.conv_small = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 中尺度卷积 - 中等范围特征
        self.conv_medium = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 大尺度卷积 - 全局特征
        self.conv_large = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 特征压缩层，将192维特征压缩到128维
        self.compression_conv = nn.Sequential(
            nn.Conv1d(64*3, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 残差GRU结构
        self.residual_gru = ResidualGRU(
            input_size=128,
            hidden_size=hidden_size//2,  # 因为是双向的，所以隐藏层大小减半
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 最终特征维度
        self.feature_dim = hidden_size
    
    def forward(self, x):
        # 输入x的形状: [batch_size, signal_length]
        
        # 转换为卷积需要的格式 [batch_size, channels, signal_length]
        x = x.unsqueeze(1)
        
        # 应用投影卷积
        x = self.projection_conv(x)  # [batch_size, 32, signal_length]
        
        # 应用多尺度卷积
        x_small = self.conv_small(x)
        x_medium = self.conv_medium(x)
        x_large = self.conv_large(x)
        
        # 拼接多尺度特征
        x_concat = torch.cat([x_small, x_medium, x_large], dim=1)  # [batch_size, 192, signal_length]
        
        # 应用特征压缩
        x = self.compression_conv(x_concat)  # [batch_size, 128, signal_length]
        
        # 转换为GRU需要的格式 [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch_size, signal_length, 128]
        
        # 应用残差GRU
        output = self.residual_gru(x)  # [batch_size, signal_length, hidden_size]
        
        return output

# 解码器网络 (用于重构任务)
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_length=600):
        super(Decoder, self).__init__()
        
        # GRU解码器
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        
        # 转置卷积映射到更高维度，准备多尺度解码
        self.expand_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, 192, kernel_size=1),
            nn.BatchNorm1d(192),
            nn.ReLU()
        )
        
        # 多尺度转置卷积
        self.upconv_small = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.upconv_medium = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.upconv_large = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # 合并通道
        self.merge_conv = nn.Sequential(
            nn.Conv1d(96, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # 最终输出层
        self.output_conv = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Tanh()  # 使用Tanh激活函数，规范化输出范围
        )
        
    def forward(self, x):
        # 输入x的形状: [batch_size, seq_len, features]
        
        # 应用GRU层
        x, _ = self.gru(x)  # [batch_size, seq_len, hidden_dim]
        
        # 转换为卷积需要的格式 [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, hidden_dim, seq_len]
        
        # 扩展维度，准备多尺度解码
        x = self.expand_conv(x)  # [batch_size, 192, seq_len]
        
        # 将特征分成三个通道，每个通道64维
        x_small, x_medium, x_large = torch.split(x, 64, dim=1)
        
        # 应用多尺度转置卷积
        x_small = self.upconv_small(x_small)    # [batch_size, 32, seq_len]
        x_medium = self.upconv_medium(x_medium)  # [batch_size, 32, seq_len]
        x_large = self.upconv_large(x_large)    # [batch_size, 32, seq_len]
        
        # 合并三个通道
        x_merged = torch.cat([x_small, x_medium, x_large], dim=1)  # [batch_size, 96, seq_len]
        
        # 合并通道
        x = self.merge_conv(x_merged)  # [batch_size, 32, seq_len]
        
        # 最终输出
        x = self.output_conv(x)  # [batch_size, 1, seq_len]
        
        # 转换为[batch_size, signal_length]格式
        x = x.squeeze(1)
        
        return x
# 交叉门控模块
class CrossGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(CrossGatingModule, self).__init__()
        
        # 特征变换层
        self.transform_contrastive = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.transform_reconstruction = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        # 门控生成层
        self.gate_contrastive = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        self.gate_reconstruction = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, contrastive_features, reconstruction_features):
        # 变换特征
        transformed_contrastive = self.transform_contrastive(contrastive_features)
        transformed_reconstruction = self.transform_reconstruction(reconstruction_features)
        
        # 从对方特征生成门控
        gate_for_contrastive = self.gate_reconstruction(reconstruction_features)
        gate_for_reconstruction = self.gate_contrastive(contrastive_features)
        
        # 应用交叉门控
        enhanced_contrastive = contrastive_features + transformed_contrastive * gate_for_contrastive
        enhanced_reconstruction = reconstruction_features + transformed_reconstruction * gate_for_reconstruction
        
        return enhanced_contrastive, enhanced_reconstruction
def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    """
    仅在 3 个尺度上计算对比损失:
      1) 原始尺度 (不池化)
      2) 下采样 1 次 (signal_length/2)
      3) 下采样 2 次 (signal_length/4)
    """
    device = z1.device
    total_loss = torch.tensor(0., device=device)
    d = 0  # 记录实际计算了几次损失

    # ------------ Scale 1: 原始尺度 ------------
    if alpha != 0:
        total_loss += alpha * instance_contrastive_loss(z1, z2)
    if (1 - alpha) != 0 and d >= temporal_unit:
        total_loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
    d += 1

    # ------------ Scale 2: 下采样 1 次 (1/2) ------------
    z1_half = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
    z2_half = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if alpha != 0:
        total_loss += alpha * instance_contrastive_loss(z1_half, z2_half)
    if (1 - alpha) != 0 and d >= temporal_unit:
        total_loss += (1 - alpha) * temporal_contrastive_loss(z1_half, z2_half)
    d += 1

    # ------------ Scale 3: 再次下采样 (1/4) ------------
    z1_quarter = F.max_pool1d(z1_half.transpose(1, 2), kernel_size=2).transpose(1, 2)
    z2_quarter = F.max_pool1d(z2_half.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if alpha != 0:
        total_loss += alpha * instance_contrastive_loss(z1_quarter, z2_quarter)
    if (1 - alpha) != 0 and d >= temporal_unit:
        total_loss += (1 - alpha) * temporal_contrastive_loss(z1_quarter, z2_quarter)
    d += 1

    # 将 3 个尺度的损失取平均
    return total_loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

# 掩码重构损失函数
def masked_reconstruction_loss(original, reconstructed, mask, reduction='mean'):
    # 只计算被掩码部分的重构损失
    mse_loss = F.mse_loss(
        reconstructed[mask], 
        original[mask], 
        reduction=reduction
    )
    return mse_loss

# 修改后的预训练模型，添加交叉门控机制
class PPGPretrainModel(nn.Module):
    def __init__(self, signal_length=600, feature_dim=256, mask_ratio=0.20):
        super(PPGPretrainModel, self).__init__()
        
        # 编码器 (用于对比学习和掩码重构共享)
        self.encoder = Encoder(input_size=signal_length, hidden_size=feature_dim//2)
        
        # 解码器 (用于掩码重构)
        self.decoder = Decoder(input_dim=self.encoder.feature_dim, hidden_dim=feature_dim, output_length=signal_length)
        
        # 掩码生成器
        self.mask_generator = MaskGenerator(mask_ratio=mask_ratio)
        
        # 交叉门控模块
        self.cross_gating = CrossGatingModule(self.encoder.feature_dim)
        
    def forward(self, x1, x2=None, mode="joint"):
        batch_size = x1.size(0)
        
        if x2 is not None and mode == "contrast_only":
            # 仅对比学习模式
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            return z1, z2
            
        elif x2 is None and mode == "recon_only":
            # 仅掩码重构模式
            # 生成掩码
            mask_batch = []
            masked_x1 = x1.clone()
            
            for i in range(batch_size):
                mask = self.mask_generator.generate_mask(x1.size(1))
                masked_x1[i, mask] = 0.0  # 掩蔽信号部分设为0
                mask_batch.append(mask)
            
            # 转换掩码列表为tensor
            mask_tensor = torch.stack(mask_batch)
            
            # 编码被掩码的信号
            z_masked = self.encoder(masked_x1)
            
            # 解码并重构原始信号
            reconstructed = self.decoder(z_masked)
            
            return masked_x1, reconstructed, mask_tensor, x1
            
        else:
            # 联合模式（默认）：应用交叉门控机制
            
            # 对比学习特征
            z1 = self.encoder(x1)
            z2 = self.encoder(x2) if x2 is not None else None
            
            # 掩码重构特征
            mask_batch = []
            masked_x1 = x1.clone()
            
            for i in range(batch_size):
                mask = self.mask_generator.generate_mask(x1.size(1))
                masked_x1[i, mask] = 0.0  # 掩蔽信号部分设为0
                mask_batch.append(mask)
            
            mask_tensor = torch.stack(mask_batch)
            z_masked = self.encoder(masked_x1)
            
            # 池化序列特征，用于交叉门控
            # 此处为简化计算，使用平均池化将时序特征压缩为单一特征向量
            z1_pooled = torch.mean(z1, dim=1)  # [batch_size, feature_dim]
            z_masked_pooled = torch.mean(z_masked, dim=1)  # [batch_size, feature_dim]
            
            # 应用交叉门控
            enhanced_z1_pooled, enhanced_z_masked_pooled = self.cross_gating(z1_pooled, z_masked_pooled)
            
            # 将增强后的特征扩展回序列维度
            enhanced_z1 = enhanced_z1_pooled.unsqueeze(1).expand(-1, z1.size(1), -1)  # [batch_size, seq_len, feature_dim]
            enhanced_z_masked = enhanced_z_masked_pooled.unsqueeze(1).expand(-1, z_masked.size(1), -1)
            
            # 如果有第二个对比样本，也应用相同的增强
            if z2 is not None:
                z2_pooled = torch.mean(z2, dim=1)
                _, enhanced_z2_pooled = self.cross_gating(z2_pooled, z_masked_pooled)
                enhanced_z2 = enhanced_z2_pooled.unsqueeze(1).expand(-1, z2.size(1), -1)
            else:
                enhanced_z2 = None
            
            # 解码并重构原始信号
            reconstructed = self.decoder(enhanced_z_masked)
            
            # 返回增强后的特征，用于对比学习和重构任务
            if z2 is not None:
                return enhanced_z1, enhanced_z2, masked_x1, reconstructed, mask_tensor, x1
            else:
                return enhanced_z1, None, masked_x1, reconstructed, mask_tensor, x1

# 可视化掩码和重构结果
def visualize_reconstruction(original, masked, reconstructed, mask, idx=0, save_path=None):
    plt.figure(figsize=(12, 8))
    
    # 绘制原始信号
    plt.subplot(3, 1, 1)
    plt.plot(original[idx].cpu().numpy())
    plt.title("Original PPG Signal")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # 绘制掩码信号
    plt.subplot(3, 1, 2)
    signal = masked[idx].cpu().numpy()
    mask_idx = mask[idx].cpu().numpy()
    plt.plot(signal)
    plt.scatter(np.where(mask_idx)[0], signal[mask_idx], color='red', s=10, alpha=0.7)
    plt.title("Masked PPG Signal (Red points are masked)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # 绘制重构信号和原始信号对比
    plt.subplot(3, 1, 3)
    plt.plot(original[idx].cpu().numpy(), label="Original")
    plt.plot(reconstructed[idx].detach().cpu().numpy(), label="Reconstructed", alpha=0.7)
    plt.title("Original vs Reconstructed Signal")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 修改训练函数以支持交叉门控
def train(model, train_loader, optimizer, device, epoch, lambda_contrast=1.0, lambda_recon=1.0, 
          visualize_interval=100, save_dir='./visualizations'):
    model.train()
    total_contrast_loss = 0
    total_recon_loss = 0
    total_loss = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据增强器
    augmenter = DataAugmenter()
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, signals in progress_bar:
        signals = signals.to(device)
        batch_size = signals.size(0)
        
        # 对每个信号生成两个不同的增强版本
        augmented_signals1 = []
        augmented_signals2 = []
        
        for i in range(batch_size):
            signal = signals[i].cpu().numpy()
            # 对同一个信号生成两个不同的增强版本
            aug1 = augmenter.augment(signal)
            aug2 = augmenter.augment(signal)
            augmented_signals1.append(torch.tensor(aug1, dtype=torch.float32))
            augmented_signals2.append(torch.tensor(aug2, dtype=torch.float32))
        
        # 转换为张量并移动到设备
        x1 = torch.stack(augmented_signals1).to(device)
        x2 = torch.stack(augmented_signals2).to(device)
        
        optimizer.zero_grad()
        
        # 前向传播，使用交叉门控联合模式
        enhanced_z1, enhanced_z2, masked_x, reconstructed, mask_tensor, original = model(x1, x2, mode="joint")
        
        # 对比学习损失 - 使用原有的层次对比损失
        contrast_loss = hierarchical_contrastive_loss(enhanced_z1, enhanced_z2, alpha=0.5, temporal_unit=0)
        
        # 掩码重构损失
        recon_loss = masked_reconstruction_loss(original, reconstructed, mask_tensor)
        
        # 总损失
        loss = lambda_contrast * contrast_loss + lambda_recon * recon_loss
        
        loss.backward()
        optimizer.step()
        
        total_contrast_loss += contrast_loss.item()
        total_recon_loss += recon_loss.item()
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_description(
            f"Epoch {epoch} | Contrast: {contrast_loss.item():.4f} | Recon: {recon_loss.item():.4f}"
        )
        
        # 每隔一定间隔可视化重构结果
        if (batch_idx + 1) % visualize_interval == 0:
            save_path = os.path.join(save_dir, f'epoch{epoch}_batch{batch_idx+1}.png')
            visualize_reconstruction(original, masked_x, reconstructed, mask_tensor, 
                                   idx=0, save_path=save_path)
    
    # 计算平均损失
    avg_contrast_loss = total_contrast_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_loss = total_loss / len(train_loader)
    
    return avg_contrast_loss, avg_recon_loss, avg_loss

# 修改验证函数以支持交叉门控
def validate(model, val_loader, device, lambda_contrast=1.0, lambda_recon=1.0):
    model.eval()
    total_contrast_loss = 0
    total_recon_loss = 0
    total_loss = 0
    
    # 创建数据增强器
    augmenter = DataAugmenter()
    
    with torch.no_grad():
        for signals in val_loader:
            signals = signals.to(device)
            batch_size = signals.size(0)
            
            # 对每个信号生成两个不同的增强版本
            augmented_signals1 = []
            augmented_signals2 = []
            
            for i in range(batch_size):
                signal = signals[i].cpu().numpy()
                # 对同一个信号生成两个不同的增强版本
                aug1 = augmenter.augment(signal)
                aug2 = augmenter.augment(signal)
                augmented_signals1.append(torch.tensor(aug1, dtype=torch.float32))
                augmented_signals2.append(torch.tensor(aug2, dtype=torch.float32))
            
            # 转换为张量并移动到设备
            x1 = torch.stack(augmented_signals1).to(device)
            x2 = torch.stack(augmented_signals2).to(device)
            
            # 前向传播，使用交叉门控联合模式
            enhanced_z1, enhanced_z2, masked_x, reconstructed, mask_tensor, original = model(x1, x2, mode="joint")
            
            # 对比学习损失 - 使用原有的层次对比损失
            contrast_loss = hierarchical_contrastive_loss(enhanced_z1, enhanced_z2, alpha=0.5, temporal_unit=0)
            
            # 掩码重构损失
            recon_loss = masked_reconstruction_loss(original, reconstructed, mask_tensor)
            
            # 总损失
            loss = lambda_contrast * contrast_loss + lambda_recon * recon_loss
            
            total_contrast_loss += contrast_loss.item()
            total_recon_loss += recon_loss.item()
            total_loss += loss.item()
    
    # 计算平均损失
    avg_contrast_loss = total_contrast_loss / len(val_loader)
    avg_recon_loss = total_recon_loss / len(val_loader)
    avg_loss = total_loss / len(val_loader)
    
    return avg_contrast_loss, avg_recon_loss, avg_loss

# 主函数
def main():
    # 配置参数
    data_dir = "/home/swucar/cyz/transferlearning/processed_datasets_avg_only_normalized"
    batch_size = 64
    epochs = 50
    learning_rate = 1e-3
    feature_dim = 256
    signal_length = 600
    mask_ratio = 0.30
    lambda_contrast = 1.0
    lambda_recon = 1.0
    save_dir = 'cyz/bloodpressure/duibibutongyuxunliancelue/ppg_pretrain_models'  # 模型保存路径
        # 添加新的配置信息

    
    # 创建时间戳，用于唯一标识此次训练
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_name = f"ppg_pretrain_{timestamp}"
    # 加载数据集
    dataset = PPGDataset(data_dir, signal_length=signal_length)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化模型
    model = PPGPretrainModel(signal_length=signal_length, feature_dim=feature_dim, mask_ratio=mask_ratio)
    model = model.to(device)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 创建保存模型和可视化结果的文件夹
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)
    
    # 保存训练配置
    config = {
        'data_dir': data_dir,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'feature_dim': feature_dim,
        'signal_length': signal_length,
        'mask_ratio': mask_ratio,
        'lambda_contrast': lambda_contrast,
        'lambda_recon': lambda_recon,
        'timestamp': timestamp,
        'use_cross_gating': True # 记录使用了交叉门控
    }
    
    with open(f"{save_dir}/{model_name}_config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # 训练循环
    best_val_loss = float('inf')
    training_history = {
        'train_contrast_loss': [],
        'train_recon_loss': [],
        'train_loss': [],
        'val_contrast_loss': [],
        'val_recon_loss': [],
        'val_loss': []
    }
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # 训练阶段
        train_contrast_loss, train_recon_loss, train_loss = train(
            model, train_loader, optimizer, device, epoch, 
            lambda_contrast, lambda_recon, 
            visualize_interval=100, save_dir='./visualizations'
        )
        
        # 验证阶段
        val_contrast_loss, val_recon_loss, val_loss = validate(
            model, val_loader, device, lambda_contrast, lambda_recon
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录训练历史
        training_history['train_contrast_loss'].append(train_contrast_loss)
        training_history['train_recon_loss'].append(train_recon_loss)
        training_history['train_loss'].append(train_loss)
        training_history['val_contrast_loss'].append(val_contrast_loss)
        training_history['val_recon_loss'].append(val_recon_loss)
        training_history['val_loss'].append(val_loss)
        
        # 打印损失信息
        print(f"Train - Contrast Loss: {train_contrast_loss:.4f}, Recon Loss: {train_recon_loss:.4f}, Total Loss: {train_loss:.4f}")
        print(f"Val   - Contrast Loss: {val_contrast_loss:.4f}, Recon Loss: {val_recon_loss:.4f}, Total Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存完整模型
            torch.save(model, f"{save_dir}/{model_name}_best_full.pth")
            # 保存模型权重和训练状态
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, f"{save_dir}/{model_name}_best.pth")
            print(f"Saved best model at epoch {epoch} with validation loss {val_loss:.4f}")
            
            # 单独保存编码器，方便下游任务使用
            torch.save(model.encoder.state_dict(), f"{save_dir}/{model_name}_encoder_best.pth")
        
        # 每隔几个epoch保存一次checkpoint
        if epoch % 5 == 0 or epoch == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, f"{save_dir}/{model_name}_epoch{epoch}.pth")
    
    # 保存最终模型
    torch.save(model, f"{save_dir}/{model_name}_final_full.pth")
    torch.save(model.state_dict(), f"{save_dir}/{model_name}_final.pth")
    
    # 单独保存编码器，方便下游任务使用
    torch.save(model.encoder.state_dict(), f"{save_dir}/{model_name}_encoder_final.pth")
    
    # 保存训练历史
    with open(f"{save_dir}/{model_name}_history.json", 'w') as f:
        json.dump(training_history, f, indent=4)
    
    print("训练完成！模型已保存至:", save_dir)
    print(f"最佳模型保存为: {model_name}_best.pth，最终模型保存为: {model_name}_final.pth")
    print(f"编码器单独保存为: {model_name}_encoder_best.pth 和 {model_name}_encoder_final.pth")
    
    # 可视化训练过程
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(training_history['train_contrast_loss'], label='Train')
    plt.plot(training_history['val_contrast_loss'], label='Validation')
    plt.title('Contrast Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(training_history['train_recon_loss'], label='Train')
    plt.plot(training_history['val_recon_loss'], label='Validation')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_training_history.png")
    plt.close()
    
    print("训练历史图已保存至:", f"{save_dir}/{model_name}_training_history.png")

if __name__ == "__main__":
    main()