import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import DropPath


def sum_msblock_channels(cfg):
    return cfg[0] + cfg[2] + cfg[4] + cfg[5]


class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1),
                 activation_fn=nn.ReLU, use_batch_norm=True, use_bias=False):
        super(Unit3D, self).__init__()

        if isinstance(kernel_shape, int):
            ks = (kernel_shape, kernel_shape, kernel_shape)
        else:
            ks = kernel_shape
        padding = [k // 2 for k in ks]

        if use_batch_norm:
            use_bias = False

        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=output_channels, kernel_size=kernel_shape,
                                stride=stride, padding=padding, bias=use_bias)
        self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.1) if use_batch_norm else nn.Identity()
        self.activation_fn = activation_fn(inplace=True) if activation_fn is not None else nn.Identity()

    def forward(self, x):
        return self.activation_fn(self.bn(self.conv3d(x)))


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t, pad_h, pad_w = self.compute_pad(0, t), self.compute_pad(1, h), self.compute_pad(2, w)
        pad_t_f, pad_t_b = pad_t // 2, pad_t - (pad_t // 2)
        pad_h_f, pad_h_b = pad_h // 2, pad_h - (pad_h // 2)
        pad_w_f, pad_w_b = pad_w // 2, pad_w - (pad_w // 2)
        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        return super().forward(F.pad(x, pad))


class MSBlock(nn.Module):
    def __init__(self, in_channels, out_channels_cfg):
        super(MSBlock, self).__init__()
        self.b0 = Unit3D(in_channels, out_channels_cfg[0], [1, 1, 1])

        self.b1a = Unit3D(in_channels, out_channels_cfg[1], [1, 1, 1])
        self.b1b = Unit3D(out_channels_cfg[1], out_channels_cfg[2], [1, 3, 3])

        self.b2a = Unit3D(in_channels, out_channels_cfg[3], [1, 1, 1])
        self.b2b = Unit3D(out_channels_cfg[3], out_channels_cfg[4], [1, 3, 3])

        self.b3a = MaxPool3dSamePadding([3, 3, 3], (1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels, out_channels_cfg[5], [1, 1, 1])

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class MSBlockStage(nn.Module):
    def __init__(self, in_channels, msblock1_cfg, msblock2_cfg, pool_kernel_size, pool_stride):
        super().__init__()

        self.pool = MaxPool3dSamePadding(kernel_size=pool_kernel_size, stride=pool_stride, padding=0)
        self.msblock1 = MSBlock(in_channels, msblock1_cfg)
        self.msblock2 = MSBlock(sum_msblock_channels(msblock1_cfg), msblock2_cfg)

    def forward(self, x):
        x = self.pool(x)
        x = self.msblock1(x)
        x = self.msblock2(x)
        return x


class VisualStem(nn.Module):
    def __init__(self, in_channels, channels_cfg):
        super().__init__()
        self.stem = nn.Sequential(
            # conv1a
            Unit3D(in_channels, channels_cfg['c1a'], kernel_shape=[1, 7, 7], stride=(1, 2, 2)),
            # pool2a
            MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0),
            # conv2b
            Unit3D(channels_cfg['c1a'], channels_cfg['c2b'], kernel_shape=[1, 1, 1]),
            # conv2c
            Unit3D(channels_cfg['c2b'], channels_cfg['c2c'], kernel_shape=[1, 3, 3])
        )

        self.stage2 = MSBlockStage(
            in_channels=channels_cfg['c2c'], msblock1_cfg=channels_cfg['m3b'], msblock2_cfg=channels_cfg['m3c'],
            pool_kernel_size=[1, 3, 3], pool_stride=(1, 2, 2)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        return x


class ContextStream(nn.Module):
    def __init__(self, in_channels, context_dim, mid_channels_factor=0.5):
        super().__init__()
        mid_channels = int(in_channels * mid_channels_factor)
        self.conv_block = nn.Sequential(
            Unit3D(in_channels, mid_channels, kernel_shape=(1, 3, 3), stride=(1, 2, 2)),
            nn.GELU(),
            Unit3D(mid_channels, context_dim, kernel_shape=(1, 3, 3))
        )
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        x = self.conv_block(x)
        context_prior_tensor = self.pool(x)
        return context_prior_tensor.squeeze(-1).squeeze(-1).permute(0, 2, 1)


class AFM(nn.Module):
    def __init__(self, context_dim, target_channels):
        super().__init__()
        self.gamma_proj = nn.Linear(context_dim, target_channels)
        self.beta_proj = nn.Linear(context_dim, target_channels)

    def forward(self, feature_map, context_seq):
        context_vec = context_seq.mean(dim=1)
        gamma = self.gamma_proj(context_vec).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(context_vec).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * feature_map + beta


class CPE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, groups=channels, bias=True
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        # 利用卷积的 Zero-padding 效应注入绝对位置信息
        return x + self.proj(x)


class SplitScanMamba(nn.Module):
    def __init__(self, d_model, drop_path=0., **mamba_kwargs):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 2 for Split-Scan.")

        self.half_dim = d_model // 2

        # 实例化两个半维度的 Mamba
        # 参数量计算：2 * (d/2)^2 = 0.5 * d^2。
        # 相比原本的双向全维度 (2 * d^2)，参数量变成了原本的 1/4！
        self.forward_mamba = Mamba(d_model=self.half_dim, **mamba_kwargs)
        self.backward_mamba = Mamba(d_model=self.half_dim, **mamba_kwargs)

        # 通道混合层：让两组特征交换信息
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_proj = nn.Linear(d_model, d_model)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, L, C]
        residual = x

        # 1. 通道切分 (Split)
        x_f, x_b = torch.split(x, self.half_dim, dim=-1)

        # 2. 前向分支 (Group A)
        out_f = self.forward_mamba(x_f)

        # 3. 反向分支 (Group B)
        # 将序列在时间维度(dim=1)翻转，过 Mamba，再翻转回来
        x_b_rev = torch.flip(x_b, dims=[1])
        out_b = self.backward_mamba(x_b_rev)
        out_b = torch.flip(out_b, dims=[1])

        # torch.save(out_f.cpu().detach(), 'feat_f.pt')
        # torch.save(out_b.cpu().detach(), 'feat_b.pt')

        # 4. 拼接 (Concat)
        out = torch.cat([out_f, out_b], dim=-1)

        # 5. 融合 (Fuse)
        out = self.fusion_norm(out)
        out = self.fusion_proj(out)

        return residual + self.drop_path(out)


class S2Block(nn.Module):
    def __init__(self, feature_dim, patch_size=(2, 2), mamba_config=None, drop_path=None):
        super().__init__()

        if mamba_config is None:
            mamba_config = {'d_state': 16, 'd_conv': 4, 'expand': 2}
        self.mamba_dim = mamba_config.get('dim', 256)
        self.mamba_depth = mamba_config.get("depth", 2)

        if isinstance(drop_path, float):
            drop_path = [drop_path] * self.mamba_depth

        self.patch_size = patch_size
        patch_h, patch_w = patch_size
        self.patch_embedding_dim = feature_dim * patch_h * patch_w

        self.input_proj = nn.Linear(self.patch_embedding_dim, self.mamba_dim)
        self.output_proj = nn.Linear(self.mamba_dim, self.patch_embedding_dim)
        self.norm = nn.LayerNorm(self.mamba_dim)

        self.layers = nn.ModuleList([
            SplitScanMamba(
                d_model=self.mamba_dim,
                drop_path=drop_path[i],
                d_state=mamba_config.get('d_state'),
                d_conv=mamba_config.get('d_conv'),
                expand=mamba_config.get('expand')
            ) for i in range(self.mamba_depth)
        ])

        self.aggregator = nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim)

    def _scan_patch_interleaved(self, features):
        B, C, T, H, W = features.shape
        p_h, p_w = self.patch_size
        if not (H % p_h == 0 and W % p_w == 0):
            raise ValueError(f"Feature map dimensions ({H}, {W}) must be divisible by patch size.")
        num_patches = (H // p_h) * (W // p_w)
        x = features.view(B, C, T, H // p_h, p_h, W // p_w, p_w)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous().view(B, T, num_patches, self.patch_embedding_dim)
        interleaved_sequence = x.permute(0, 2, 1, 3).contiguous().view(B, num_patches * T, self.patch_embedding_dim)
        return interleaved_sequence, (B, C, T, H, W, num_patches)

    def _merge_patch_interleaved(self, processed_sequence, shape_info):
        B, C, T, H, W, num_patches = shape_info
        p_h, p_w = self.patch_size
        x = processed_sequence.view(B, num_patches, T, self.patch_embedding_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        num_patches_h, num_patches_w = H // p_h, W // p_w
        x = x.view(B, T, num_patches_h, num_patches_w, C, p_h, p_w)
        merged_features = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous().view(B, C, T, H, W)
        return merged_features

    def forward(self, features):
        residual = features

        interleaved_sequence, shape_info = self._scan_patch_interleaved(features)

        seq = self.input_proj(interleaved_sequence)
        seq = self.norm(seq)
        for layer in self.layers:
            seq = layer(seq)
        seq = self.output_proj(seq)

        merged_features = self._merge_patch_interleaved(seq, shape_info)

        aggregated_features = self.aggregator(merged_features)

        return residual + aggregated_features


class FocalStream(nn.Module):
    def __init__(self, in_channels, context_dim, channels_cfg, mamba_config, patch_size=(2, 2),
                 drop_path_rates=None):
        super().__init__()
        if drop_path_rates is None:
            dpr_s3 = [0.0] * 2
            dpr_s4 = [0.0] * 2
        else:
            dpr_s3, dpr_s4 = drop_path_rates

        s3_channels = channels_cfg['s3']
        s4_channels = channels_cfg['s4']

        self.afm1 = AFM(context_dim, in_channels)
        self.afm2 = AFM(context_dim, s3_channels)

        self.stage3 = MSBlockStage(in_channels=in_channels,
                                   msblock1_cfg=channels_cfg['m4b'],
                                   msblock2_cfg=channels_cfg['m4c'],
                                   pool_kernel_size=[1, 3, 3], pool_stride=(1, 2, 2))
        self.stage4 = MSBlockStage(in_channels=s3_channels,
                                   msblock1_cfg=channels_cfg['m5b'],
                                   msblock2_cfg=channels_cfg['m5c'],
                                   pool_kernel_size=[1, 2, 2], pool_stride=(1, 2, 2))

        self.cpe_s3 = CPE(s3_channels)
        self.cpe_s4 = CPE(s4_channels)

        self.s2block1 = S2Block(feature_dim=s3_channels, patch_size=patch_size,
                                mamba_config=mamba_config,
                                drop_path=dpr_s3)
        self.s2block2 = S2Block(feature_dim=s4_channels, patch_size=patch_size,
                                mamba_config=mamba_config,
                                drop_path=dpr_s4)

    def forward(self, x, context_prior_seq):
        x = self.afm1(x, context_prior_seq)
        x = self.stage3(x)
        x = self.cpe_s3(x)
        x = self.s2block1(x)

        x = self.afm2(x, context_prior_seq)
        x = self.stage4(x)
        x = self.cpe_s4(x)
        x = self.s2block2(x)

        return x


class S2Gest(nn.Module):
    def __init__(self, in_channels=1, num_classes=25, channels_config=None, mamba_config=None,
                 drop_path_rate=0.0, classifier_dropout=0.1):
        super().__init__()

        if channels_config is None or mamba_config is None:
            raise ValueError("S2Gest requires both 'channels_config' and 'mamba_config'")

        self.channels_cfg = channels_config
        self.mamba_cfg = mamba_config

        self.channels_cfg['s1'] = self.channels_cfg['c2c']
        self.channels_cfg['s2'] = sum_msblock_channels(self.channels_cfg['m3c'])
        self.channels_cfg['s3'] = sum_msblock_channels(self.channels_cfg['m4c'])
        self.channels_cfg['s4'] = sum_msblock_channels(self.channels_cfg['m5c'])

        self.visual_stem = VisualStem(in_channels=in_channels, channels_cfg=self.channels_cfg)

        visual_stem_out_channels = self.channels_cfg['s2']
        self.context_dim = self.mamba_cfg.get('dim', 256)
        self.context_stream = ContextStream(visual_stem_out_channels, self.context_dim)

        depth = self.mamba_cfg.get('depth', 2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth * 2)]
        dpr_s3 = dpr[0:depth]
        dpr_s4 = dpr[depth:]
        self.focal_stream = FocalStream(in_channels=visual_stem_out_channels,
                                        context_dim=self.context_dim,
                                        channels_cfg=self.channels_cfg,
                                        mamba_config=self.mamba_cfg,
                                        drop_path_rates=[dpr_s3, dpr_s4])

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.LayerNorm(self.channels_cfg['s4']),
            nn.Dropout(classifier_dropout),
            nn.Linear(self.channels_cfg['s4'], num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x_stem = self.visual_stem(x)
        z_ctx = self.context_stream(x_stem)
        focal_features = self.focal_stream(x_stem, z_ctx)
        logits = self.cls_head(focal_features)
        return logits


def GestureClassification(in_channels: int = 3, num_classes: int = 25, **kwargs):
    channels_config = {
        k: v for k, v in kwargs["model_config"].items()
        if not k.startswith('mamba_')
    }
    mamba_config = {
        k.replace('mamba_', ''): v
        for k, v in kwargs["model_config"].items()
        if k.startswith('mamba_')
    }
    print(mamba_config, channels_config)
    model = S2Gest(
        in_channels=in_channels,
        num_classes=num_classes,
        channels_config=channels_config,
        mamba_config=mamba_config,
        drop_path_rate=kwargs.get("drop_path_rate", 0.1),
        classifier_dropout=kwargs.get("classifier_dropout", 0.1)
    )
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "drop_path_rate": 0.3,
        "classifier_dropout": 0.1,
        "model_config": {
            "mamba_dim": 128,
            "mamba_depth": 2,
            "mamba_d_state": 16,
            "mamba_d_conv": 4,
            "mamba_expand": 2,

            "c1a": 32, "c2b": 32, "c2c": 96,
            "m3b": [32, 48, 64, 8, 16, 16],
            "m3c": [64, 64, 96, 16, 48, 32],
            "m4b": [96, 48, 104, 8, 24, 32],
            "m4c": [80, 56, 112, 12, 32, 32],
            "m5b": [128, 80, 160, 16, 64, 64],
            "m5c": [192, 96, 192, 24, 64, 64]
        }
    }
    in_channels = 1
    batch_size = 1
    num_frames = 40
    video_clip = torch.randn(batch_size, num_frames, in_channels, 192, 256).to(device)
    print(f"\nPreparing dummy input, shape: {video_clip.shape}")

    model = GestureClassification(in_channels=in_channels, num_classes=25, **config).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params / 1e6:.2f} M")

    try:
        output = model(video_clip)
        print(f"Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")