import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from data import DataAugmenter

def set_seed(seed):
    """
    设置随机种子，确保结果可复现
    
    参数:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def visualize_reconstruction(original, masked, reconstructed, mask, idx=0, save_path=None):
    """
    可视化掩码和重构结果
    
    参数:
        original (Tensor): 原始信号
        masked (Tensor): 掩码后的信号
        reconstructed (Tensor): 重构的信号
        mask (Tensor): 掩码
        idx (int): 要可视化的批次中的样本索引
        save_path (str, optional): 图像保存路径
    """
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


def train(model, train_loader, optimizer, device, epoch, lambda_contrast=1.0, lambda_recon=1.0, 
          visualize_interval=100, save_dir='./visualizations'):
    """
    训练单个epoch
    
    参数:
        model (nn.Module): 模型
        train_loader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器
        device (device): 计算设备
        epoch (int): 当前epoch数
        lambda_contrast (float): 对比损失权重
        lambda_recon (float): 重构损失权重
        visualize_interval (int): 可视化间隔
        save_dir (str): 可视化结果保存目录
        
    返回:
        tuple: (对比损失, 重构损失, 总损失)
    """
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
        
        # 批量生成两组增强视图，避免CPU/GPU往返
        x1, x2 = augmenter.augment_pair_batch(signals)
        
        optimizer.zero_grad()

        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type if autocast_enabled else "cpu",
                             dtype=torch.bfloat16 if autocast_enabled else torch.float32,
                             enabled=autocast_enabled):
            # 前向传播，使用交叉门控联合模式
            enhanced_z1, enhanced_z2, masked_x, reconstructed, mask_tensor, original = model(x1, x2, mode="joint")

            # 对比学习损失 - 使用原有的层次对比损失
            from losses import hierarchical_contrastive_loss, masked_reconstruction_loss
            # 在 float32 下计算损失，避免混合精度下出现 NaN
            contrast_loss = hierarchical_contrastive_loss(
                enhanced_z1.float(), enhanced_z2.float(),
                alpha=0.7,                  # 减少时间对比权重
                temporal_unit=1,            # 从下采样尺度开始做时间对比
                instance_time_stride=2,     # 实例级按时间 stride 采样
                temporal_stride=4,          # 时间级使用更大 stride
                temperature=0.2,            # 温度放大，logits 更平缓
                normalize=True
            )

            # 掩码重构损失
            recon_loss = masked_reconstruction_loss(original.float(), reconstructed.float(), mask_tensor)

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


def validate(model, val_loader, device, lambda_contrast=1.0, lambda_recon=1.0):
    """
    验证模型
    
    参数:
        model (nn.Module): 模型
        val_loader (DataLoader): 验证数据加载器
        device (device): 计算设备
        lambda_contrast (float): 对比损失权重
        lambda_recon (float): 重构损失权重
        
    返回:
        tuple: (对比损失, 重构损失, 总损失)
    """
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
            
            # 验证阶段使用批量增强或直接原样输入以稳定评估
            # 这里直接使用原始信号两份，避免额外随机性与计算开销
            x1, x2 = signals, signals

            autocast_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type if autocast_enabled else "cpu",
                                 dtype=torch.bfloat16 if autocast_enabled else torch.float32,
                                 enabled=autocast_enabled):
                # 前向传播，使用交叉门控联合模式
                enhanced_z1, enhanced_z2, masked_x, reconstructed, mask_tensor, original = model(x1, x2, mode="joint")

                # 对比学习损失 - 使用原有的层次对比损失
                from losses import hierarchical_contrastive_loss, masked_reconstruction_loss
                contrast_loss = hierarchical_contrastive_loss(
                    enhanced_z1.float(), enhanced_z2.float(),
                    alpha=0.7,
                    temporal_unit=1,
                    instance_time_stride=2,
                    temporal_stride=4,
                    temperature=0.2,
                    normalize=True
                )

                # 掩码重构损失
                recon_loss = masked_reconstruction_loss(original.float(), reconstructed.float(), mask_tensor)

            print(f"ctr {contrast_loss.item():.4f}  recon {recon_loss.item():.4f}")

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