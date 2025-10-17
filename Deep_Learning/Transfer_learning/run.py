#!/usr/bin/env python
"""
AVF内瘘堵塞迁移学习主程序
使用预训练编码器进行迁移学习，分类非堵塞（0）、轻度堵塞（1）和重度堵塞（2）
"""

import os
import argparse
from train_eval import main as training_main

def parse_args():
    parser = argparse.ArgumentParser(description='AVF内瘘堵塞迁移学习')
    
    parser.add_argument('--data_folder', type=str, 
                        default="/Users/yuandao/YuanDao/AI与新医药/动静脉内瘘狭窄检测/代码/data/九院自采数据集/分段后数据（可直接输入模型）/2 pos",
                        help='数据文件夹路径，包含CSV文件')
    
    parser.add_argument('--pretrained_model', type=str, 
                        default="/Users/yuandao/YuanDao/AI与新医药/动静脉内瘘狭窄检测/代码/pretrain/模型output/ppg_pretrain_20251017_163254_encoder_best.pth",
                        help='预训练编码器模型路径')
    
    parser.add_argument('--save_dir', type=str, 
                        default="/Users/yuandao/YuanDao/AI与新医药/动静脉内瘘狭窄检测/代码/深度学习/Transfer_learning/avf_transfer_results",
                        help='保存结果的目录')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 验证参数
    if not os.path.exists(args.data_folder):
        raise ValueError(f"数据文件夹 {args.data_folder} 不存在!")
    
    if not os.path.exists(args.pretrained_model):
        raise ValueError(f"预训练模型 {args.pretrained_model} 不存在!")
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 将参数保存到日志文件
    with open(os.path.join(args.save_dir, "training_config.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # 调用训练主函数
    # 注意：由于需要修改原训练函数以接受命令行参数，这里暂时使用默认的训练函数
    # 未来可以扩展功能，允许从命令行设置更多参数
    training_main()
    
    print(f"\n训练和评估已完成！所有结果已保存至 {args.save_dir}")

if __name__ == "__main__":
    main() 