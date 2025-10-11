import torch
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    """

   ----------------------------------多尺度层次对比损失-------------------------------------

    在3个尺度上计算对比损失:
      1) 原始尺度 (不池化)
      2) 下采样1次 (signal_length/2)
      3) 下采样2次 (signal_length/4)
    
    参数:
        z1 (Tensor): 第一个视图的特征表示
        z2 (Tensor): 第二个视图的特征表示
        alpha (float): 样本间与时间对比损失的平衡系数
        temporal_unit (int): 时间对比损失的起始尺度
        
    返回:
        Tensor: 层次对比损失
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

    # 将3个尺度的损失取平均
    return total_loss / d


def instance_contrastive_loss(z1, z2):
    """

    --------------------------------------------样本（实例）间对比损失---------------------------------------

    在两个视图之间的样本级别上计算对比损失
    
    参数:
        z1 (Tensor): 第一个视图的特征表示 [B, T, C]
        z2 (Tensor): 第二个视图的特征表示 [B, T, C]
        
    返回:
        Tensor: 样本间对比损失
    """
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
    """

    ------------------------------------------时间维度对比损失--------------------------------------------

    在序列的时间维度上计算对比损失
    
    参数:
        z1 (Tensor): 第一个视图的特征表示 [B, T, C]
        z2 (Tensor): 第二个视图的特征表示 [B, T, C]
        
    返回:
        Tensor: 时间维度对比损失
    """
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


def masked_reconstruction_loss(original, reconstructed, mask, reduction='mean'):
    """

    --------------------------------------掩码重构损失-------------------------------------------

    只计算被掩码部分的重构损失
    
    参数:
        original (Tensor): 原始信号
        reconstructed (Tensor): 重构信号
        mask (Tensor): 掩码，布尔型，True表示被掩码的位置
        reduction (str): 损失缩减方式，'mean'或'sum'
        
    返回:
        Tensor: 掩码重构损失
    """
    # 只计算被掩码部分的重构损失
    mse_loss = F.mse_loss(
        reconstructed[mask], 
        original[mask], 
        reduction=reduction
    )
    return mse_loss 