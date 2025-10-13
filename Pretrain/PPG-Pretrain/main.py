import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import warnings
from torch.optim import Adam

# 导入自定义模块
from data import PPGDataset
from models import PPGPretrainModel
from train import train, validate, set_seed, visualize_reconstruction

# 忽略UserWarning，避免不必要的警告干扰训练日志
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """
    主函数：PPG信号预训练的完整训练流程
    包括数据加载、模型初始化、训练循环、结果保存和可视化
    """
    # ================ 配置参数 ================
    # 数据路径：PPG信号CSV文件所在目录
    data_dir = "/Users/yuandao/YuanDao/AI与新医药/动静脉内瘘狭窄检测/代码/data/预训练的PPG数据集/处理后的公开PPG数据集"
    
    # 训练超参数设置
    batch_size = 64        # 批次大小，影响显存占用和训练速度
    epochs = 50            # 总训练轮数
    learning_rate = 1e-4   # 初始学习率
    
    # 模型结构参数
    feature_dim = 512      # 特征维度，影响模型容量和表示能力
    signal_length = 600    # PPG信号长度
    mask_ratio = 0.05      # 掩码比例，控制掩码重构任务的难度
    
    # 损失函数权重：平衡两个预训练任务
    lambda_contrast = 1.0  # 对比学习损失权重
    lambda_recon = 1.0     # 掩码重构损失权重
    
    # 模型保存路径
    save_dir = '/Users/yuandao/YuanDao/AI与新医药/动静脉内瘘狭窄检测/代码/pretrain/模型output'
    
    # 设置随机种子，确保结果可复现
    set_seed(42)
    
    # 创建时间戳，用于唯一标识此次训练，防止模型覆盖
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_name = f"ppg_pretrain_{timestamp}"
    
    # ================ 加载数据集 ================
    print(f"正在加载数据集，路径: {data_dir}...")
    dataset = PPGDataset(data_dir, signal_length=signal_length)
    
    # 划分训练集和验证集（90%用于训练，10%用于验证）
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器配置（兼容 CUDA / MPS / CPU）
    if torch.cuda.is_available():
        dl_cfg = dict(num_workers=min(8, os.cpu_count() or 4),
                      persistent_workers=True, prefetch_factor=4, pin_memory=True)
    elif torch.backends.mps.is_available():
        dl_cfg = dict(num_workers=min(8, os.cpu_count() or 4),
                      persistent_workers=True, prefetch_factor=2, pin_memory=False)
    else:
        dl_cfg = dict(num_workers=0, persistent_workers=False, prefetch_factor=2, pin_memory=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **dl_cfg
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **dl_cfg
    )
    
    print(f"数据集加载完成，训练集大小: {train_size}，验证集大小: {val_size}")
    
    # ================ 设置设备（优先 CUDA，其次 MPS） ================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用设备: NVIDIA GPU (CUDA)")
        use_compile = True
        use_autocast = True
        autocast_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: Apple Silicon GPU (MPS)")
        use_compile = False
        use_autocast = False
        autocast_dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")
        use_compile = False
        use_autocast = False
        autocast_dtype = torch.float32
    
    # ================ 初始化模型 ================
    print("初始化模型...")
    # 创建PPG预训练模型实例
    model = PPGPretrainModel(signal_length=signal_length, feature_dim=feature_dim, mask_ratio=mask_ratio)
    # 将模型转移到指定设备（GPU或CPU）
    model = model.to(device)
    if use_compile:
        try:
            model = torch.compile(model)
            print("已启用 torch.compile")
        except Exception:
            print("torch.compile 不可用或失败，已回退")
    
    # ================ 优化器设置 ================
    # 使用Adam优化器，自适应学习率优化算法
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器：ReduceLROnPlateau
    # 当验证损失不再下降时自动降低学习率
    # mode='min'：监控指标是越小越好（损失值）
    # factor=0.5：每次降低为当前学习率的一半
    # patience=5：容忍5个epoch验证损失没有改善
    # verbose=True：打印学习率变化信息
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # ================ 创建保存目录 ================
    # 确保模型保存目录和可视化目录存在
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)
    
    # ================ 保存训练配置 ================
    # 记录所有训练参数，便于复现和分析
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
        'use_cross_gating': True  # 记录使用了交叉门控
    }
    
    # 将配置保存为JSON文件
    with open(f"{save_dir}/{model_name}_config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # ================ 训练循环 ================
    print(f"开始训练，总共 {epochs} 个epoch...")
    best_val_loss = float('inf')  # 记录最佳验证损失，用于模型选择
    
    # 初始化训练历史记录字典，用于绘制损失曲线
    training_history = {
        'train_contrast_loss': [],  # 训练集对比损失
        'train_recon_loss': [],     # 训练集重构损失
        'train_loss': [],           # 训练集总损失
        'val_contrast_loss': [],    # 验证集对比损失
        'val_recon_loss': [],       # 验证集重构损失
        'val_loss': []              # 验证集总损失
    }
    
    # 主训练循环
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # 训练阶段：执行一个完整epoch的训练
        train_contrast_loss, train_recon_loss, train_loss = train(
            model, train_loader, optimizer, device, epoch, 
            lambda_contrast, lambda_recon, 
            visualize_interval=200,  # 减少I/O与绘图开销
            save_dir='./visualizations'
        )
        
        # 验证阶段：在验证集上评估模型性能
        val_contrast_loss, val_recon_loss, val_loss = validate(
            model, val_loader, device, lambda_contrast, lambda_recon
        )
        
        # 更新学习率：根据验证损失调整学习率
        # 如果验证损失连续patience个epoch没有改善，则降低学习率
        scheduler.step(val_loss)
        
        # 记录训练历史
        training_history['train_contrast_loss'].append(train_contrast_loss)
        training_history['train_recon_loss'].append(train_recon_loss)
        training_history['train_loss'].append(train_loss)
        training_history['val_contrast_loss'].append(val_contrast_loss)
        training_history['val_recon_loss'].append(val_recon_loss)
        training_history['val_loss'].append(val_loss)
        
        # 打印当前epoch的损失信息
        print(f"Train - Contrast Loss: {train_contrast_loss:.4f}, Recon Loss: {train_recon_loss:.4f}, Total Loss: {train_loss:.4f}")
        print(f"Val   - Contrast Loss: {val_contrast_loss:.4f}, Recon Loss: {val_recon_loss:.4f}, Total Loss: {val_loss:.4f}")
        
        # 保存最佳模型：当验证损失降低时
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 保存完整模型（包括结构和权重）
            torch.save(model, f"{save_dir}/{model_name}_best_full.pth")
            
            # 保存模型检查点（包含训练状态）
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),         # 模型权重
                'optimizer_state_dict': optimizer.state_dict(), # 优化器状态
                'scheduler_state_dict': scheduler.state_dict(), # 学习率调度器状态
                'val_loss': val_loss,                           # 当前验证损失
            }, f"{save_dir}/{model_name}_best.pth")
            
            print(f"已保存最佳模型，epoch {epoch}，验证损失 {val_loss:.4f}")
            
            # 单独保存编码器，方便下游任务使用
            torch.save(model.encoder.state_dict(), f"{save_dir}/{model_name}_encoder_best.pth")
        
        # 定期保存检查点：每5个epoch或最后一个epoch
        if epoch % 5 == 0 or epoch == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, f"{save_dir}/{model_name}_epoch{epoch}.pth")
    
    # ================ 保存最终模型 ================
    print("训练完成，保存最终模型...")
    # 保存完整的最终模型
    torch.save(model, f"{save_dir}/{model_name}_final_full.pth")
    # 保存最终模型权重
    torch.save(model.state_dict(), f"{save_dir}/{model_name}_final.pth")
    
    # 单独保存编码器，方便下游任务使用
    torch.save(model.encoder.state_dict(), f"{save_dir}/{model_name}_encoder_final.pth")
    
    # 保存完整训练历史记录为JSON文件
    with open(f"{save_dir}/{model_name}_history.json", 'w') as f:
        json.dump(training_history, f, indent=4)
    
    print("训练完成！模型已保存至:", save_dir)
    print(f"最佳模型保存为: {model_name}_best.pth，最终模型保存为: {model_name}_final.pth")
    print(f"编码器单独保存为: {model_name}_encoder_best.pth 和 {model_name}_encoder_final.pth")
    
    # ================ 可视化训练过程 ================
    print("正在绘制训练历史图...")
    plt.figure(figsize=(12, 8))
    
    # 绘制总损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制对比损失曲线
    plt.subplot(2, 2, 3)
    plt.plot(training_history['train_contrast_loss'], label='Train')
    plt.plot(training_history['val_contrast_loss'], label='Validation')
    plt.title('Contrast Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制重构损失曲线
    plt.subplot(2, 2, 4)
    plt.plot(training_history['train_recon_loss'], label='Train')
    plt.plot(training_history['val_recon_loss'], label='Validation')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_training_history.png")
    plt.close()
    
    print("训练历史图已保存至:", f"{save_dir}/{model_name}_training_history.png")


if __name__ == "__main__":
    main()