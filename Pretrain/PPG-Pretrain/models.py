import torch
import torch.nn as nn
import torch.nn.functional as F
from data import MaskGenerator

class ResidualGRU(nn.Module):
    """

    -----------------------------残差连接的GRU网络模块---------------------------------

    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        """
        初始化残差GRU
        
        参数:
            input_size (int): 输入特征大小
            hidden_size (int): 隐藏层大小
            num_layers (int): GRU层数
            dropout (float): Dropout比率
        """
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
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [batch_size, seq_len, features]
            
        返回:
            Tensor: 输出特征 [batch_size, seq_len, hidden_size*2]
        """
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
    """

    -----------------------------------PPG信号编码器网络--------------------------------------

    """
    def __init__(self, input_size=600, hidden_size=128, num_layers=2, dropout=0.1):
        """
        初始化编码器
        
        参数:
            input_size (int): 输入信号长度
            hidden_size (int): 隐藏层大小
            num_layers (int): GRU层数
            dropout (float): Dropout比率
        """
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
        """
        前向传播
        
        参数:
            x (Tensor): 输入信号 [batch_size, signal_length]
            
        返回:
            Tensor: 编码后的特征 [batch_size, signal_length, feature_dim]
        """
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


class Decoder(nn.Module):
    """

    ----------------------------------PPG信号解码器网络 (用于重构任务)------------------------------

    """
    def __init__(self, input_dim, hidden_dim=128, output_length=600):
        """
        初始化解码器
        
        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_length (int): 输出信号长度
        """
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
            # nn.Tanh()  # 使用Tanh激活函数，规范化输出范围
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入特征 [batch_size, seq_len, features]
            
        返回:
            Tensor: 重构的信号 [batch_size, signal_length]
        """
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


class CrossGatingModule(nn.Module):
    """

    ------------------------------交叉门控模块，增强对比学习和重构任务之间的特征交互---------------------------------

    """
    def __init__(self, feature_dim):
        """
        初始化交叉门控模块
        
        参数:
            feature_dim (int): 特征维度
        """
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
        """
        前向传播
        
        参数:
            contrastive_features (Tensor): 对比学习特征
            reconstruction_features (Tensor): 重构任务特征
            
        返回:
            tuple: (增强的对比学习特征, 增强的重构特征)
        """
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


class PPGPretrainModel(nn.Module):
    """

    -------------------------------！！！PPG预训练模型，结合对比学习和掩码重构任务！！！-------------------------------

    """
    def __init__(self, signal_length=600, feature_dim=256, mask_ratio=0.20):
        """
        初始化PPG预训练模型
        
        参数:
            signal_length (int): 信号长度
            feature_dim (int): 特征维度
            mask_ratio (float): 掩码比例
        """
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
        """
        前向传播
        
        参数:
            x1 (Tensor): 第一个输入信号 [batch_size, signal_length]
            x2 (Tensor, optional): 第二个输入信号 [batch_size, signal_length]
            mode (str): 运行模式 - "joint", "contrast_only", "recon_only"
            
        返回:
            tuple: 根据模式返回不同的输出组合
        """
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
            # 批量掩码生成（GPU 向量化）
            mask_tensor = self.mask_generator.generate_mask_batch(batch_size, x1.size(1), device=x1.device)
            masked_x1 = x1.masked_fill(mask_tensor, 0.0)
            # 批量掩码生成（GPU 向量化）
            mask_tensor = self.mask_generator.generate_mask_batch(batch_size, x1.size(1), device=x1.device)
            masked_x1 = x1.masked_fill(mask_tensor, 0.0)
            z_masked = self.encoder(masked_x1)
            
            # 池化得到门控生成所需的全局摘要，但对序列逐时刻做特征变换与门控，保留时间信息
            z1_pooled = torch.mean(z1, dim=1)            # [B, C]
            z_masked_pooled = torch.mean(z_masked, dim=1)  # [B, C]

            # 基于对方全局摘要生成门控向量（[B, C]）
            gate_for_contrastive = self.cross_gating.gate_reconstruction(z_masked_pooled)   # 用重构摘要门控对比
            gate_for_reconstruction = self.cross_gating.gate_contrastive(z1_pooled)         # 用对比摘要门控重构

            # 对序列逐时刻做可学习变换（Linear+LayerNorm 可对 [B, T, C] 直接作用）
            transformed_contrastive_seq = self.cross_gating.transform_contrastive(z1)       # [B, T, C]
            transformed_reconstruction_seq = self.cross_gating.transform_reconstruction(z_masked)  # [B, T, C]

            # 广播门控到时间维并进行逐时刻门控
            enhanced_z1 = z1 + transformed_contrastive_seq * gate_for_contrastive.unsqueeze(1)
            enhanced_z_masked = z_masked + transformed_reconstruction_seq * gate_for_reconstruction.unsqueeze(1)

            # 如果有第二个对比样本，也应用相同的门控策略
            if z2 is not None:
                transformed_z2 = self.cross_gating.transform_contrastive(z2)
                enhanced_z2 = z2 + transformed_z2 * gate_for_contrastive.unsqueeze(1)
            else:
                enhanced_z2 = None
            
            # 解码并重构原始信号
            reconstructed = self.decoder(enhanced_z_masked)
            
            # 返回增强后的特征，用于对比学习和重构任务
            if z2 is not None:
                return enhanced_z1, enhanced_z2, masked_x1, reconstructed, mask_tensor, x1
            else:
                return enhanced_z1, None, masked_x1, reconstructed, mask_tensor, x1 