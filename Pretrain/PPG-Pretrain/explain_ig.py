import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients

from data import PPGDataset, MaskGenerator
from models import PPGPretrainModel
'''
主要参数
--model_path: 你的最佳模型或state_dict/checkpoint路径
--config_path: 若传入的是state_dict/checkpoint，需提供训练时保存的config.json（用于signal_length/feature_dim/mask_ratio）
--data_dir: CSV数据目录
--index: 选择解释的样本索引
--save_path: 图像保存路径
--baseline: IG基线，可选 zero 或 mean
--steps: IG积分步数（64–128较稳）
--seed: 固定掩码与随机性
--mask_ratio: 可选，覆盖模型默认mask_ratio（不传则读取模型里的)
'''

'''
1.新增 explain_ig.py，集成 Captum Integrated Gradients

2.实现负掩码MSE目标的前向封装，确保可复现

3.添加可视化绘图，叠加显著性到PPG并保存图片

4.提供CLI参数和加载checkpoint/配置的兼容逻辑
'''

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(model_path: str, device: torch.device, config_path: str = None):
    """
    优先加载保存的完整模型（full），否则根据state_dict和配置构建模型。
    """
    model = None
    try:
        obj = torch.load(model_path, map_location=device)
        if isinstance(obj, torch.nn.Module):
            model = obj
        else:
            # 可能是checkpoint，尝试从state_dict重建
            state_dict = obj.get('model_state_dict', None)
            if state_dict is None:
                raise RuntimeError("无可用的model或state_dict")
            # 配置优先从config_path读取，否则使用合理默认
            if config_path and os.path.isfile(config_path):
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                signal_length = int(cfg.get('signal_length', 600))
                feature_dim = int(cfg.get('feature_dim', 256))
                mask_ratio = float(cfg.get('mask_ratio', 0.2))
            else:
                signal_length = 600
                feature_dim = 256
                mask_ratio = 0.2
            model = PPGPretrainModel(signal_length=signal_length,
                                      feature_dim=feature_dim,
                                      mask_ratio=mask_ratio)
            model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        # 回退：尝试直接当state_dict加载
        if config_path and os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            signal_length = int(cfg.get('signal_length', 600))
            feature_dim = int(cfg.get('feature_dim', 256))
            mask_ratio = float(cfg.get('mask_ratio', 0.2))
        else:
            signal_length = 600
            feature_dim = 256
            mask_ratio = 0.2
        model = PPGPretrainModel(signal_length=signal_length,
                                  feature_dim=feature_dim,
                                  mask_ratio=mask_ratio)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model.to(device)


def build_mask(signal_length: int, mask_ratio: float, seed: int = 42):
    set_seed(seed)
    mg = MaskGenerator(mask_ratio=mask_ratio)
    mask = mg.generate_mask(signal_length)
    return mask


def make_target_fn(model: PPGPretrainModel,
                   mask_bool_1d: torch.Tensor,
                   device: torch.device):
    """
    返回可被 IntegratedGradients 调用的 forward_func：
    输入 x: [B, L]（B 支持批量，但IG默认我们用B=1），输出 [B] 的标量得分。
    得分定义为：-MSE(original[mask], reconstructed[mask])。
    注意：这里直接使用 model.encoder + model.decoder 的重构路径，保持与recon_only一致。
    """
    mask_bool_1d = mask_bool_1d.to(device)

    def forward_func(x: torch.Tensor):
        # x: [B, L]
        masked = x.masked_fill(mask_bool_1d.unsqueeze(0), 0.0)
        z = model.encoder(masked)
        recon = model.decoder(z)
        # 对被掩码位置计算 MSE
        loss = F.mse_loss(recon[:, mask_bool_1d], x[:, mask_bool_1d], reduction='mean')
        score = -loss
        # Captum 期望按 batch 返回 [B] 或标量。这里扩展为 [B]
        return score.repeat(x.shape[0])

    return forward_func


def build_baseline(x: torch.Tensor, kind: str = 'zero') -> torch.Tensor:
    if kind == 'zero':
        return torch.zeros_like(x)
    elif kind == 'mean':
        mean_val = x.mean(dim=1, keepdim=True)
        return torch.ones_like(x) * mean_val
    else:
        # 默认零基线
        return torch.zeros_like(x)


def visualize(signal: np.ndarray,
              attribution: np.ndarray,
              save_path: str,
              title: str = 'Integrated Gradients Attribution'):
    t = np.arange(signal.shape[0])
    attr = attribution
    attr_abs = np.abs(attr)
    # 归一化便于显示
    if attr_abs.max() > 0:
        attr_norm = attr / (attr_abs.max() + 1e-8)
    else:
        attr_norm = attr

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='PPG')
    plt.title('Original PPG')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, attr_norm, color='crimson', label='IG (signed, normalized)')
    plt.fill_between(t, 0, attr_norm, color='crimson', alpha=0.2)
    plt.title(title)
    plt.xlabel('Time')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Integrated Gradients for PPG reconstruction attribution')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径（full或state_dict/checkpoint）')
    parser.add_argument('--config_path', type=str, default=None, help='可选配置JSON（当model_path为state_dict/checkpoint时）')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录（CSV集合）')
    parser.add_argument('--index', type=int, default=0, help='选择的样本索引')
    parser.add_argument('--save_path', type=str, default='/Users/yuandao/YuanDao/AI与新医药/动静脉内瘘狭窄检测/代码/pretrain/PPG-Pretrain/explain-figs/ig_example.png', help='保存可视化的路径')
    parser.add_argument('--baseline', type=str, default='zero', choices=['zero', 'mean'], help='基线类型')
    parser.add_argument('--steps', type=int, default=64, help='IG积分步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（用于固定掩码）')
    parser.add_argument('--mask_ratio', type=float, default=None, help='覆盖模型默认的mask_ratio（可选）')

    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device()

    # 加载模型
    model = load_model(args.model_path, device, args.config_path)
    # 禁用可能的编译/混合精度（外部已eval，这里确保IG稳定）
    model.eval()

    # 加载数据
    # 注意：PPGDataset默认 signal_length=600，应与训练一致
    dataset = PPGDataset(args.data_dir, signal_length=600)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index超出范围: {args.index} / {len(dataset)}")
    x = dataset[args.index].unsqueeze(0).to(device)  # [1, L]

    # 构造固定掩码
    signal_length = x.shape[1]
    if args.mask_ratio is not None:
        mask_ratio = float(args.mask_ratio)
    else:
        # 从模型的掩码生成器读取；若无则使用0.2
        mask_ratio = getattr(getattr(model, 'mask_generator', None), 'mask_ratio', 0.2)
    mask_bool = build_mask(signal_length, mask_ratio, seed=args.seed)

    # 目标函数：-MSE(original[mask], reconstructed[mask])
    forward_func = make_target_fn(model, mask_bool, device)

    # 构建IG
    ig = IntegratedGradients(forward_func)

    # 基线
    baseline = build_baseline(x, args.baseline)

    # 计算归因
    attributions, delta = ig.attribute(x, baselines=baseline, n_steps=args.steps, return_convergence_delta=True)

    # 提取为numpy
    sig_np = x.detach().cpu().numpy()[0]
    attr_np = attributions.detach().cpu().numpy()[0]

    # 可视化
    title = f'IG (steps={args.steps}, baseline={args.baseline})'
    visualize(sig_np, attr_np, args.save_path, title)

    # 打印简单日志
    print(f"Saved IG visualization to: {args.save_path}")
    print(f"Convergence delta (mean): {delta.detach().cpu().mean().item():.6f}")


if __name__ == '__main__':
    main()


