import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from einops import rearrange

import time
from thop import profile
from thop import clever_format


class GeometricPositionalEncoding(nn.Module):
    def __init__(self, num_regions, d_model, coords_dim=3):
        super().__init__()
        
        # 1. 尝试加载或初始化数据
        try:
            # 加载数据 (此时可能在 CPU)
            centroids = torch.load("/home/caojiaxiang/brain age/RBA/pths/region_centroids.pt")
            print("Loaded region centroids successfully.")
        except Exception as e:
            print(f"Warning: Could not load centroids ({e}), using random initialization.")
            centroids = torch.randn(num_regions, 3)

        # 2. 【关键步骤】使用 register_buffer
        # 第一个参数是名字（字符串），第二个参数是 tensor
        # 这样 self.region_centroids 会自动随 model.to(device) 移动
        self.register_buffer('region_centroids', centroids)
        
        self.proj = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, x):
        # x: [B, num_regions, d_model]
        
        # 此时 self.region_centroids 已经和 x 在同一个设备上了
        # 我们可以直接使用
        pos_embed = self.proj(self.region_centroids) # [num_regions, d_model]
        
        return x + pos_embed.unsqueeze(0)

def evaluate_model_complexity(model, x, region_masks):
    # ===== 1. 参数数量 =====
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.3f} M")

    # ===== 2. FLOPs计算 =====
    try:
        # thop需要forward的输入与输出都在CPU或GPU上
        flops, params = profile(model, inputs=(x, region_masks), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"FLOPs per forward: {flops}, Params: {params}")
    except Exception as e:
        print(f"[Warning] FLOPs计算失败: {e}")
    
    # ===== 3. 显存占用 =====
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x, region_masks)
    mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak GPU memory usage: {mem_allocated:.2f} MB")

    # ===== 4. 推理时间 =====
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):  # 连续10次，取平均
            _ = model(x, region_masks)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 10
    print(f"Avg forward time: {elapsed * 1000:.2f} ms")

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.conv(x)
    

# class conv_block(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.LeakyReLU(inplace=True), # LeakyReLU 往往比 ReLU 表现更好
#             nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch)
#         )
#         self.shortcut = nn.Sequential()
#         if in_ch != out_ch:
#             self.shortcut = nn.Sequential(
#                 nn.Conv3d(in_ch, out_ch, 1, bias=False),
#                 nn.BatchNorm3d(out_ch)
#             )
#         self.relu = nn.LeakyReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.conv(x) + self.shortcut(x))

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, size):
        super(up_conv, self).__init__()
        self.size = size
        self.up = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=False)
        )


    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='trilinear', align_corners=False)
        x = self.up(x)
        return x
    

class Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class MultiScaleMaskProcessor(nn.Module):
    """
    多尺度mask处理器
    将原始mask下采样到与特征图相同的尺寸
    """
    def __init__(self):
        super(MultiScaleMaskProcessor, self).__init__()
    
    def forward(self, region_masks, target_sizes):
        """
        region_masks: [batch, num_regions, 91, 109, 91]
        target_sizes: 目标尺寸列表，每个元素是(D, H, W)的元组
        """
        multi_scale_masks = []
        
        for size in target_sizes:
            # 使用最近邻插值保持mask的二进制特性
            scaled_mask = F.interpolate(
                region_masks, 
                size=size, 
                mode='nearest'  # 使用最近邻插值保持二值性
            )
            multi_scale_masks.append(scaled_mask)
        
        return multi_scale_masks

class MultiScaleRegionFeatureExtractor(nn.Module):
    def __init__(self, num_regions, feature_dims=[64, 32, 16], output_dim=64):
        super(MultiScaleRegionFeatureExtractor, self).__init__()
        self.num_regions = num_regions
        self.feature_dims = feature_dims
        self.output_dim = output_dim
    
        self.mask_processor = MultiScaleMaskProcessor()
        
        # 修改处理器：去掉AdaptiveAvgPool3d，因为einsum已经完成了池化
        # self.scale_processors = nn.ModuleList([
        #     nn.Sequential(
        #         nn.AdaptiveAvgPool1d(1),  # 全局平均池化
        #         nn.Flatten(),
        #         nn.Linear(dim, dim // 2),  # 直接使用线性层降维
        #         nn.ReLU(),
        #         nn.Dropout(0.1)
        #     ) for dim in feature_dims
        # ])
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                # 先用1D卷积提取局部模式
                nn.Conv1d(dim, dim//2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # 然后池化
                nn.Flatten(),
                nn.Linear(dim//2, dim//2),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for dim in feature_dims
        ])
        
        total_feature_dim = sum([dim // 2 for dim in feature_dims])
        self.region_fuser = nn.Sequential(
            nn.Linear(total_feature_dim, 256),
            nn.BatchNorm1d(256),  # 添加BN
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        # self.region_fuser = nn.Sequential(
        #     nn.Linear(total_feature_dim, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, output_dim),
        #     nn.ReLU()
        # )
        
    def forward(self, multi_scale_features, region_masks):
        batch_size = multi_scale_features[0].size(0)
        target_sizes = [feat.shape[2:] for feat in multi_scale_features]
        
        with torch.no_grad():  # mask处理不需要梯度
            multi_scale_masks = self.mask_processor(region_masks, target_sizes)
        
        all_scale_features = []
        
        for i,(feature_map, scaled_mask, processor) in enumerate(zip(
            multi_scale_features, multi_scale_masks, self.scale_processors)):

            # 使用einsum进行高效的内存计算
            # b: batch, r: regions, c: channels, d: depth, h: height, w: width
            # 计算每个脑区的加权特征和 [batch, regions, channels]
            B,C,D,H,W = feature_map.shape
            scaled_mask = scaled_mask.float()
            feature_map = feature_map.float()
            # feature_map = feature_map / (feature_map.abs().max(dim=(1,2,3,4), keepdim=True)[0] + 1e-8)
            # max_vals = torch.amax(feature_map.abs(), dim=(1,2,3,4), keepdim=True)
            # feature_map = feature_map / (max_vals + 1e-8)
            # feature_map = self.batch_norms[i](feature_map)
            # print(feature_map.min(), feature_map.max(), feature_map.mean())

            # region_features = torch.einsum('bcdhw,brdhw->brc', feature_map, scaled_mask) / (D*H*W)
            # 添加安全的归一化
            mask_sum = scaled_mask.sum(dim=(-3, -2, -1), keepdim=True)  # [B, R, 1, 1, 1]
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            region_features = torch.einsum('bcdhw,brdhw->brcd', feature_map, scaled_mask) / mask_sum.squeeze(-1)
            # region_features = torch.einsum('bcdhw,brdhw->brcd', feature_map, scaled_mask) / (H*W)
            # region_features = feature_map[:, None, :, :, :, :] * scaled_mask[:, :, None, :, :, :]
            # print(region_features)
            # region_features = region_features.to(torch.float16)
            # print(region_features.shape)

            B, R, C, D = region_features.shape
            # # 现在region_features已经是[batch, regions, channels]形状
            region_features = region_features.contiguous().view(B*R, C, D).contiguous()
            processed_features = processor(region_features)
            processed_features = processed_features.view(B,R,C//2).contiguous()
            all_scale_features.append(processed_features)
        
        # 拼接多尺度特征

        concatenated_features = torch.cat(all_scale_features, dim=2)
        B, R, C = concatenated_features.shape  # B=2, R=117, C=56
        x = concatenated_features.view(B*R, C)
        # 融合特征
        output_features = self.region_fuser(x)
        output_features = output_features.view(B, R, -1)
        return output_features


class AttU_Net_Shallow(nn.Module):
    """
    Shallow Attention U-Net with only 2 downsampling steps
    """
    def __init__(self, img_ch=1, output_ch=1,channel=16):
        super(AttU_Net_Shallow, self).__init__()

        n1 = channel
        filters = [n1, n1 * 2, n1 * 4]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

        self.Up2 = up_conv(filters[2], filters[1], size=(45, 54, 45))
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], size=(91, 109, 91))
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv1 = conv_block(filters[1], filters[0])

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)          # [batch, 16, 91, 109, 91]
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)         # [batch, 32, 45, 54, 45]
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)         # [batch, 64, 22, 27, 22]

        # Decoder with attention
        d2 = self.Up2(e3)           # [batch, 32, 45, 54, 45]
        x2 = self.Att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)      # [batch, 32, 45, 54, 45]

        d1 = self.Up1(d2)           # [batch, 16, 91, 109, 91]
        x1 = self.Att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)      # [batch, 16, 91, 109, 91]

        # 返回所有尺度的特征
        return [e1,e3, d2, d1]
    

class AttU_Net_Mid(nn.Module):
    """
    Medium Attention U-Net with 3 downsampling steps
    (增加了一层，现在有 3 层下采样)
    """
    def __init__(self, img_ch=1, output_ch=1, channel=16):
        super(AttU_Net_Mid, self).__init__()

        n1 = channel
        # 1. 修改 Filters：增加一级深度 [16, 32, 64, 128]
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2) # 新增

        # Encoder
        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3]) # 新增：瓶颈层 (Bottleneck)

        # Decoder - Stage 3 (新增的最深层)
        # size=(22, 27, 22) 是根据 (45,54,45) 再次下采样计算得出的
        self.Up3 = up_conv(filters[3], filters[2], size=(22, 27, 22)) 
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv3 = conv_block(filters[3], filters[2])

        # Decoder - Stage 2 (原有的，连接 input 变为来自 d3)
        self.Up2 = up_conv(filters[2], filters[1], size=(45, 54, 45))
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        # Decoder - Stage 1 (原有的，连接 input 变为来自 d2)
        self.Up1 = up_conv(filters[1], filters[0], size=(91, 109, 91))
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        # 如果你需要最后的输出层 conv1x1，可以在这里添加
        # self.Conv_1x1 = nn.Conv3d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # --- Encoder ---
        e1 = self.Conv1(x)          # [batch, 16, 91, 109, 91]
        
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)         # [batch, 32, 45, 54, 45]
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)         # [batch, 64, 22, 27, 22]

        e4 = self.Maxpool3(e3)      
        e4 = self.Conv4(e4)         # [batch, 128, 11, 13, 11] -> 新的瓶颈特征

        # --- Decoder with attention ---
        
        # 新增的深层解码
        d3 = self.Up3(e4)           # Upsample e4 to matches e3 size [22, 27, 22]
        x3 = self.Att3(g=d3, x=e3)  # Attention using e3
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)      # [batch, 64, 22, 27, 22]

        # 原有的解码层 (输入变为 d3)
        d2 = self.Up2(d3)           # Upsample d3 to matches e2 size [45, 54, 45]
        x2 = self.Att2(g=d2, x=e2)  # Attention using e2
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)      # [batch, 32, 45, 54, 45]

        d1 = self.Up1(d2)           # Upsample d2 to matches e1 size [91, 109, 91]
        x1 = self.Att1(g=d1, x=e1)  # Attention using e1
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)      # [batch, 16, 91, 109, 91]

        # 返回特征：包含 Encoder浅层, Encoder深层(瓶颈), Decoder深层, Decoder中层, Decoder浅层
        # 根据你的需求，通常 e4 是新的瓶颈层
        return [e1, e4, d3, d2, d1]

class BrainRegionTransformer(nn.Module):
    """
    专门处理脑区特征的Transformer
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, num_regions):
        super(BrainRegionTransformer, self).__init__()
        self.num_regions = num_regions
        self.d_model = d_model
        
        # 可学习的位置编码，因为脑区没有天然的顺序
        # self.positional_encoding = nn.Parameter(torch.randn(1, num_regions, d_model))
        # 使用更小的初始化范围
        # self.positional_encoding = nn.Parameter(torch.zeros(1, num_regions, d_model))
        # nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)  # 或者使用xavier初始化
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,  # 使用batch_first格式
            activation='gelu'
        )
        self.addPs = GeometricPositionalEncoding(117,d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(d_model)
        # 分类token [CLS]
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 回归头
        # 更简洁的回归头
        # self.regressor = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward // 2, 1)
        # )
        self.regressor = nn.Sequential(
            nn.Linear(2*d_model, 256),
            nn.LayerNorm(256),  # 关键改进：用LayerNorm替代BatchNorm
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, region_features, region_ages):
        batch_size = region_features.size(0)
        
        # 1. 输入归一化和位置编码
        region_features = self.input_norm(region_features)
        transformer_input = self.addPs(region_features)
        # transformer_input = region_features + self.positional_encoding
        
        # [可选优化：移除 CLS Token]
        # 如果使用 GAP，可以完全移除 [CLS] token，从而节省 1 个序列位置的计算和参数。
        # 如果决定保留 [CLS] token：
        
        # 添加[CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat((cls_tokens, transformer_input), dim=1)
        
        # 2. Transformer编码
        # transformer_output: [batch_size, num_regions + 1, d_model]
        transformer_output = self.transformer_encoder(transformer_input)
        
        # 3. 聚合策略 (新)：对所有脑区输出进行平均池化
        
        # 排除 [CLS] token (索引 0)
        region_outputs = transformer_output[:, 1:, :] # [batch_size, num_regions, d_model]
        
        # 全局平均池化 (GAP)
        gap_output = region_outputs.mean(dim=1)      # [batch_size, d_model]
        
        # [可选组合]：将 GAP 输出与 CLS token 输出结合，提供更全面的特征
        cls_output = transformer_output[:, 0, :]
        combined_output = torch.cat((cls_output, gap_output), dim=1)
        
        # 4. 回归预测
        output = self.regressor(combined_output) # 如果使用拼接
        # output = self.regressor(gap_output)      # 如果使用 GAP
        
        return output

class UNetWithBrainRegionTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, num_brain_regions,channel=16):
        super(UNetWithBrainRegionTransformer, self).__init__()
        self.AttU_Net = AttU_Net_Shallow(channel=channel)
        self.num_brain_regions = num_brain_regions
        
        # 多尺度脑区特征提取器，输出维度与Transformer的d_model对齐
        self.region_extractor = MultiScaleRegionFeatureExtractor(
            num_brain_regions, 
            feature_dims=[channel*4,channel*2,channel],
            output_dim=d_model
        )
        
        # 脑区特征Transformer
        self.region_transformer = BrainRegionTransformer(
            d_model, nhead, num_layers, dim_feedforward, dropout, num_brain_regions
        )
        # self.log_lambda_rba = nn.Parameter(torch.zeros(1))  # 对应 lambda_rba = exp(...)
        # self.log_lambda_consis = nn.Parameter(torch.zeros(1))
        # self.log_lambda_grad = nn.Parameter(torch.zeros(1))
        

        self.log_lambda_rba = nn.Parameter(torch.tensor(1.0))      
        self.log_lambda_consis = nn.Parameter(torch.tensor(-0.0))  
        self.log_lambda_grad = nn.Parameter(torch.tensor(0.0))   

        self.shared_age_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, region_masks, only_rba=False):
        """
        x: 输入MRI图像 [batch, 1, 91, 109, 91]
        region_masks: 脑图谱mask [batch, num_regions, 91, 109, 91]
        """
        # 获取U-Net所有尺度的特征
        # multi_scale_features = self.AttU_Net(x)
        multi_scale_featuress = self.AttU_Net(x)
        multi_scale_features = multi_scale_featuress[1:]
        self.saved_e1 = multi_scale_featuress[0]

        # 提取多尺度脑区特征 [batch_size, num_regions, d_model]
        region_features = self.region_extractor(multi_scale_features, region_masks)

        # batch_size, num_regions, feat_dim = region_features.shape
        # # 重塑为 [batch_size * num_regions, feat_dim]
        # flat_features = region_features.view(-1, feat_dim)
        # # 应用共享预测器 [batch_size * num_regions, 1]
        # flat_ages = self.shared_age_predictor(flat_features)
        # # 恢复形状 [batch_size, num_regions, 1]
        # region_ages = flat_ages.view(batch_size, num_regions, 1)
        region_ages = self.shared_age_predictor(region_features)
        if only_rba:
            return {'RBA':region_ages}
        
        # 通过Transformer处理脑区特征
        output = self.region_transformer(region_features,region_ages[:,1:,:])

        
        return {'BA':output,'RBA':region_ages}

# 简化版本，不使用[CLS] token，而是使用全局平均池化
class BrainRegionTransformerSimple(nn.Module):
    """
    简化的脑区特征Transformer，使用全局平均池化
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(BrainRegionTransformerSimple, self).__init__()
        self.d_model = d_model
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, region_features):
        """
        region_features: [batch_size, num_regions, feature_dim]
        """
        # Transformer编码
        transformer_output = self.transformer_encoder(region_features)
        
        # 全局平均池化
        global_features = torch.mean(transformer_output, dim=1)
        
        # 回归预测
        output = self.regressor(global_features)
        
        return output

if __name__ == '__main__':
    import nibabel as nib
    import numpy as np
    # 读取nii或nii.gz文件
    img = nib.load("/home/cjx/Im/ROI_MNI_V4.nii")
    data = img.get_fdata()        # 获取为 numpy 数组
    # 测试参数
    num_regions = 117  # AAL图谱有116个脑区
    d_model = 32
    nhead = 8
    num_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region_labels = [
        0, 2001, 2002, 2101, 2102, 2111, 2112, 2201, 2202, 2211, 2212, 2301,
        2302, 2311, 2312, 2321, 2322, 2331, 2332, 2401, 2402, 2501, 2502, 2601,
        2602, 2611, 2612, 2701, 2702, 3001, 3002, 4001, 4002, 4011, 4012, 4021,
        4022, 4101, 4102, 4111, 4112, 4201, 4202, 5001, 5002, 5011, 5012, 5021,
        5022, 5101, 5102, 5201, 5202, 5301, 5302, 5401, 5402, 6001, 6002, 6101,
        6102, 6201, 6202, 6211, 6212, 6221, 6222, 6301, 6302, 6401, 6402, 7001,
        7002, 7011, 7012, 7021, 7022, 7101, 7102, 8101, 8102, 8111, 8112, 8121,
        8122, 8201, 8202, 8211, 8212, 8301, 8302, 9001, 9002, 9011, 9012, 9021,
        9022, 9031, 9032, 9041, 9042, 9051, 9052, 9061, 9062, 9071, 9072, 9081,
        9082, 9100, 9110, 9120, 9130, 9140, 9150, 9160, 9170
    ]

    
    # 创建模型
    model = UNetWithBrainRegionTransformer(
        d_model, nhead, num_layers, dim_feedforward, dropout, num_regions
    ).to(device)

    
    # 测试输入
    batch_size = 1
    x = torch.randn((batch_size,1, 91, 109, 91)).to(device)
    # region_masks = torch.randn((batch_size, num_regions, 91, 109, 91)).to(device)  # binary masks
    num_regions = len(region_labels)
    # region_masks = np.zeros((num_regions, *data.shape), dtype=np.uint8)

    # # --- 4. 为每个脑区生成二值掩膜 ---
    # for i, label in enumerate(region_labels):
    #     region_mask = (data == label)
    #     region_masks[i] = region_mask.astype(np.uint8)
    # region_tensor = torch.from_numpy(region_masks)
    # torch.save(region_tensor, "/home/cjx/afterHW/ATUN/AAL_regions_116.pt")
    region_tensor = torch.load("/home/cjx/afterHW/ATUN/AAL_regions_116.pt")  # [116, 91, 109, 91]

    region_tensor = region_tensor.unsqueeze(0)  # [1, 116, 91, 109, 91]
    region_tensor = region_tensor.repeat(batch_size, 1, 1, 1, 1)  # [8, 116, 91, 109, 91]
    region_masks = region_tensor.to(device)
    outputs = model(x, region_masks)
    grads = torch.autograd.grad(
        outputs=outputs['RBA'],
        inputs=model.saved_e1,
        grad_outputs=torch.ones_like(outputs['RBA']),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    print(grads)

    # evaluate_model_complexity(model, x, region_masks)
    
    with torch.no_grad():
        outputs_noise = model(x, region_masks,only_rba=True)
        # rba_noise = outputs_noise['RBA'].detach().clone()
    

    # # 测试完整模型
    
    # print(f"Output shape: {output.shape}")
    