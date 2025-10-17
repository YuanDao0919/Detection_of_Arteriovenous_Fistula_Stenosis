import torch
import torch.nn as nn

# 残差GRU模块
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

# 编码器模型
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

# Cross-Gating 模块实现
class CrossGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(CrossGatingModule, self).__init__()
        
        # Transformation layers for feature adaptation
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
        
        # Gate generation layers
        self.gate_contrastive = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        self.gate_reconstruction = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, contrastive_features, reconstruction_features):
        # Transform features
        transformed_contrastive = self.transform_contrastive(contrastive_features)
        transformed_reconstruction = self.transform_reconstruction(reconstruction_features)
        
        # Generate gates from the opposite branch
        gate_for_contrastive = self.gate_reconstruction(reconstruction_features)
        gate_for_reconstruction = self.gate_contrastive(contrastive_features)
        
        # Apply cross-gating
        enhanced_contrastive = contrastive_features + transformed_contrastive * gate_for_contrastive
        enhanced_reconstruction = reconstruction_features + transformed_reconstruction * gate_for_reconstruction
        
        return enhanced_contrastive, enhanced_reconstruction

# 添加Decoder类，为了完整的模型定义
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_length):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.decoder(x).squeeze(-1)

# 跨门控预训练模型
class PPGPretrainModelWithCrossGating(nn.Module):
    def __init__(self, signal_length=600, feature_dim=256, mask_ratio=0.20):
        super(PPGPretrainModelWithCrossGating, self).__init__()
        
        # 从data.py导入MaskGenerator
        from data import MaskGenerator
        
        # Shared encoder backbone
        self.encoder = Encoder(input_size=signal_length, hidden_size=feature_dim//2)
        
        # Specialized projection heads for different tasks
        self.contrastive_projection = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.reconstruction_projection = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Cross-gating module
        self.cross_gating = CrossGatingModule(feature_dim)
        
        # Decoder (for mask reconstruction)
        self.decoder = Decoder(input_dim=feature_dim, hidden_dim=feature_dim, output_length=signal_length)
        
        # Mask generator
        self.mask_generator = MaskGenerator(mask_ratio=mask_ratio)
        
        # Feature dimension
        self.feature_dim = feature_dim
    
    def forward(self, x1, x2=None, mode=None):
        batch_size = x1.size(0)
        
        if mode == "contrastive" or (x2 is not None and mode is None):
            # Contrastive learning mode
            # Encode both augmented versions
            z1_backbone = self.encoder(x1)
            z2_backbone = self.encoder(x2)
            
            # Reshape from [batch_size, seq_len, feature_dim] to [batch_size, feature_dim]
            # by taking the mean across the sequence dimension
            z1_pooled = torch.mean(z1_backbone, dim=1)
            z2_pooled = torch.mean(z2_backbone, dim=1)
            
            # Project features for contrastive learning
            z1_contrastive = self.contrastive_projection(z1_pooled)
            z2_contrastive = self.contrastive_projection(z2_pooled)
            
            return z1_contrastive, z2_contrastive, z1_backbone, z2_backbone
            
        elif mode == "reconstruction" or (x2 is None and mode is None):
            # Mask reconstruction mode
            # Generate masks for each sample in the batch
            mask_batch = []
            masked_x1 = x1.clone()
            
            for i in range(batch_size):
                mask = self.mask_generator.generate_mask(x1.size(1))
                masked_x1[i, mask] = 0.0  # Mask parts of the signal
                mask_batch.append(mask)
            
            # Convert mask list to tensor
            mask_tensor = torch.stack(mask_batch)
            
            # Encode the masked signal
            z_masked_backbone = self.encoder(masked_x1)
            
            # Project features for reconstruction
            z_masked_pooled = torch.mean(z_masked_backbone, dim=1)
            z_masked_recon = self.reconstruction_projection(z_masked_pooled)
            
            # Expand pooled features back to sequence length for decoding
            z_masked_expanded = z_masked_recon.unsqueeze(1).expand(-1, x1.size(1), -1)
            
            # Apply cross-gating if contrastive features are provided
            # This will be used in the full forward pass
            return masked_x1, z_masked_backbone, z_masked_recon, mask_tensor, x1
            
        else:
            # Full forward pass with cross-gating
            # First, compute contrastive features
            z1_contrastive, z2_contrastive, z1_backbone, z2_backbone = self(x1, x2, mode="contrastive")
            
            # Then, compute reconstruction features
            masked_x1, z_masked_backbone, z_masked_recon, mask_tensor, original = self(x1, mode="reconstruction")
            
            # Apply cross-gating between contrastive and reconstruction features
            # Using z1_contrastive for simplicity (could also use average of z1 and z2)
            z1_pooled = torch.mean(z1_backbone, dim=1)
            z_masked_pooled = torch.mean(z_masked_backbone, dim=1)
            
            enhanced_contrastive, enhanced_reconstruction = self.cross_gating(z1_pooled, z_masked_pooled)
            
            # Expand enhanced reconstruction features for decoding
            enhanced_reconstruction_expanded = enhanced_reconstruction.unsqueeze(1).expand(-1, x1.size(1), -1)
            
            # Decode the enhanced features to reconstruct the original signal
            reconstructed = self.decoder(enhanced_reconstruction_expanded)
            
            return z1_contrastive, z2_contrastive, masked_x1, reconstructed, mask_tensor, original

# 渐进式特征适应层
class ProgressiveAdaptationLayer(nn.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.3):
        super(ProgressiveAdaptationLayer, self).__init__()
        
        # 渐进式特征提取
        self.adaptation = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # 注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 8, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 渐进程度参数 (可训练)
        self.alpha = nn.Parameter(torch.tensor(0.5))
            
    def forward(self, x, training_progress=1.0):
        # 将输入从 [batch_size, seq_len, hidden_dim] 转换为 [batch_size, hidden_dim, seq_len]
        x = x.permute(0, 2, 1)  # 将维度调整为卷积层需要的格式
        
        # 应用自适应层
        adapted = self.adaptation(x)
        
        # 通道注意力
        attention = self.channel_attention(adapted)
        adapted = adapted * attention
        
        # 通过alpha参数和训练进度控制原始特征和适应特征的混合
        mix_ratio = torch.sigmoid(self.alpha) * training_progress
        output = (1 - mix_ratio) * x + mix_ratio * adapted
        
        return output

# 改进的内瘘分类器
class ImprovedAVFClassifier(nn.Module):
    def __init__(self, pretrained_encoder, hidden_dim=128, num_classes=3, dropout_rate=0.5):
        super(ImprovedAVFClassifier, self).__init__()
        
        # 预训练编码器
        self.encoder = pretrained_encoder
        
        # 创新：渐进式特征适应层 - 连接预训练特征和下游任务
        self.adaptation_layer = ProgressiveAdaptationLayer(hidden_dim, dropout_rate=0.3)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 辅助分类器 - 提高训练稳定性
        self.aux_classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 创新：任务特定特征调整
        self.task_specific_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        
        # 用于控制解冻过程的参数
        self.current_epoch = 0
        self.total_epochs = 1
        self.unfreezing_strategy = "progressive"  # progressive 或 all_at_once
    
    def update_epoch(self, current, total):
        """更新当前训练进度，用于渐进式解冻"""
        self.current_epoch = current
        self.total_epochs = total
    
    def unfreeze_encoder(self):
        """根据训练进度解冻编码器的不同部分"""
        
        if self.unfreezing_strategy == "all_at_once":
            # 一次性解冻整个编码器
            if self.current_epoch >= self.total_epochs * 0.3:  # 30%的训练后解冻
                for param in self.encoder.parameters():
                    param.requires_grad = True
                print("解冻整个编码器")
                
        elif self.unfreezing_strategy == "progressive":
            # 渐进式解冻，从后向前
            progress = self.current_epoch / self.total_epochs
            encoder_layers = list(self.encoder.children())
            
            if progress >= 0.3:  # 30%的训练后开始解冻
                # 计算要解冻的层数
                layers_to_unfreeze = int((progress - 0.3) / 0.7 * len(encoder_layers))
                
                # 至少解冻一层
                layers_to_unfreeze = max(1, layers_to_unfreeze)
                
                # 从后往前解冻
                for i in range(len(encoder_layers) - 1, len(encoder_layers) - layers_to_unfreeze - 1, -1):
                    if i >= 0:
                        for param in encoder_layers[i].parameters():
                            param.requires_grad = True
                print(f"渐进式解冻: 解冻最后 {layers_to_unfreeze} 层编码器")
    
    def forward(self, x, task_specific=True, training_progress=None):
        # 设置训练进度
        if training_progress is None:
            training_progress = self.current_epoch / max(1, self.total_epochs)
        
        # 任务特定处理
        if task_specific:
            task_x = self.task_specific_conv(x.unsqueeze(1))
            x = x + task_x.squeeze(1) * 0.1  # 轻微调整原始信号
        
        # 获取编码特征
        encoded = self.encoder(x)  # [batch_size, seq_len, hidden_dim]
        
        # 应用渐进式适应层
        adapted = self.adaptation_layer(encoded, training_progress)  # [batch_size, hidden_dim, seq_len]
        
        # 全局池化（注意adapted已经是[batch_size, hidden_dim, seq_len]格式）
        features = self.global_pool(adapted).flatten(1)
        
        # 主分类器输出
        main_out = self.classifier(features)
        
        # 辅助分类器输出
        aux_out = self.aux_classifier(features)
        
        return main_out, aux_out, features 