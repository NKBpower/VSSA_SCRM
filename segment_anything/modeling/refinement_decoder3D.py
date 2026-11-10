# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# 创新点二:两阶段粗到精解码器(Coarse-to-Fine Refinement Decoder)
# ============================================================
# 第一阶段:SAM-Med3D MaskDecoder3D 输出粗掩码 low-res logits(32^3);
# 第二阶段两种实现,由 --refine_type 选择:
#   light(旧轻量版):RefinementPromptEncoder + RefinementDecoder3D,
#       纯 3D 卷积(73k 参数),粗掩码特征 + 编码器特征融合;
#   sam(SAM 同构版,主线):RefinementPromptEncoder3D 继承 PromptEncoder3D、
#       RefinementMaskDecoder3D 继承 MaskDecoder3D,二阶段与一阶段结构同构。
#       一阶段接收交互点击负责"生成",二阶段只吃粗掩码负责"优化":
#       PE 唯一输入为粗掩码(无 points/boxes),Decoder 复用冻结编码器 8^3
#       embedding + 二阶段 dense prompt 走完整 TwoWayTransformer3D,
#       以可学习 α(初始 0)凸组合 refined=coarse+α*(refine-coarse)。
#       二阶段权重全部从头随机初始化,不拷贝一阶段。
#
# 设计约束:
#   - 纯 3D 卷积轻量结构(无注意力),参数量极小,符合 PEFT 范式;
#   - 卷积核采用轴向方向解耦先验:in-plane (1,3,3) + cross-plane (3,1,1),
#     数据已统一为 1.5mm 各向同性体素,两分支物理感受野等大,
#     拆分目的为轴向解耦与参数效率,与创新点一的 AnisotropicConvAdapter3D 呼应;
#   - α 初始 0:训练初期精修分支输出恒等粗掩码,随训练渐进激活,稳定优化;
#   - 二阶段提示以前一阶段结果为主:blend 初始 sigmoid(2.0)≈0.88,粗掩码权重占优;
#   - 训练损失统一为 Dice + 交叉熵(DiceCELoss),无额外附加损失项。

from typing import Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompt_encoder3D import PromptEncoder3D
from .mask_decoder3D import MaskDecoder3D


class AnisotropicRefineBlock(nn.Module):
    """方向解耦残差精修块。

    双分支深度可分离卷积:
      - in_plane:    kernel (1,3,3) 捕获 H-W 轴状面局部细节;
      - cross_plane: kernel (3,1,1) 建模 D 方向层间局部关联;
    数据已统一重采样为 1.5×1.5×1.5mm 各向同性体素,两分支物理感受野
    等大(1.5mm 输入网格上均为 4.5mm),拆分目的为轴向解耦(轴状面/
    层间统计特性不同)与参数效率(18 vs (3,3,3) 的 27 个深度卷积权重)。
    1x1 点卷积融合后做残差相加,保持优化稳定。
    """

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels
        self.norm = nn.GroupNorm(min(8, channels), channels)
        # 瓶颈降维(轻量)
        self.pw_in = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.in_plane = nn.Conv3d(
            hidden, hidden,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            groups=hidden,
        )
        self.cross_plane = nn.Conv3d(
            hidden, hidden,
            kernel_size=(3, 1, 1), padding=(1, 0, 0),
            groups=hidden,
        )
        self.fuse = nn.Conv3d(hidden * 2, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

        nn.init.kaiming_normal_(self.pw_in.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.fuse.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.pw_in(h)
        y_in = self.in_plane(h)
        y_cross = self.cross_plane(h)
        y = torch.cat([y_in, y_cross], dim=1)
        y = self.act(y)
        y = self.fuse(y)
        return residual + y


class RefinementPromptEncoder(nn.Module):
    """二阶段提示编码器:以前一阶段粗掩码为主,辅以编码器特征。

    核心思想:
      - 粗掩码编码(主):coarse_prob → mask_prompt,捕获前一阶段分割结果;
      - 编码器特征投影(辅):image_embedding → feat_prompt,提供空间上下文;
      - 可学习融合 blend(sigmoid):控制两者权重,初始 sigmoid(2.0)≈0.88,粗掩码为主。

    输出 dense_prompt 指导 RefinementDecoder3D 预测增量 δ。
    """

    def __init__(
        self,
        encoder_embed_dim: int = 384,
        prompt_dim: int = 64,
        mask_in_chans: int = 16,
    ):
        super().__init__()
        self.prompt_dim = prompt_dim

        # 粗掩码编码(主):coarse_prob → prompt_dim
        self.mask_downscaling = nn.Sequential(
            nn.Conv3d(1, mask_in_chans, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, mask_in_chans), mask_in_chans),
            nn.GELU(),
            nn.Conv3d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, mask_in_chans), mask_in_chans),
            nn.GELU(),
            nn.Conv3d(mask_in_chans, prompt_dim, kernel_size=1),
        )

        # 编码器特征投影(辅):image_embedding → prompt_dim
        self.feat_proj = nn.Sequential(
            nn.Conv3d(encoder_embed_dim, prompt_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, prompt_dim), prompt_dim),
            nn.GELU(),
        )

        # 可学习融合权重:blend_logit 初始 2.0 → sigmoid≈0.88,粗掩码为主
        self.blend_logit = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        coarse_logits: torch.Tensor,
        image_embedding: torch.Tensor,
        target_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coarse_logits: (B,1,D,H,W) 第一阶段粗掩码 logits(32^3)。
            image_embedding: (B,C,d,h,w) 编码器输出(8^3)。
            target_size: (D,H,W) 输出空间尺寸,与粗掩码对齐。
        Returns:
            dense_prompt: (B,prompt_dim,D,H,W) 融合后的提示嵌入。
            coarse_prob:  (B,1,D,H,W) 粗掩码概率(供训练侧门控/监控复用)。
        """
        coarse_prob = torch.sigmoid(coarse_logits)

        # 粗掩码提示(主)
        mask_prompt = self.mask_downscaling(coarse_prob)

        # 编码器特征提示(辅):上采样到粗掩码尺度
        feat_prompt = self.feat_proj(image_embedding)
        feat_prompt = F.interpolate(
            feat_prompt, size=target_size,
            mode='trilinear', align_corners=False,
        )

        # 融合:粗掩码为主(blend≈0.88),编码器特征为辅(1-blend≈0.12)
        blend = torch.sigmoid(self.blend_logit)
        dense_prompt = blend * mask_prompt + (1 - blend) * feat_prompt

        return dense_prompt, coarse_prob


class RefinementDecoder3D(nn.Module):
    """轻量两阶段精修解码器(创新点二)。

    消费 RefinementPromptEncoder 输出的 dense_prompt(粗掩码为主 + 编码器特征为辅),
    通过各向异性残差块预测增量 δ,以残差标量 α(初始 0)渐进融合到粗掩码。

    两种结构(由 use_pyramid 切换,输出契约完全一致,便于消融):
      - use_pyramid=False(默认): 3 个残差块在 32^3 平铺(原始结构)。
      - use_pyramid=True: 轻量 U 形金字塔 32^3→16^3→8^3→16^3→32^3,
        瓶颈层在编码器原生 8^3 尺度(24mm/token)直连 image_embedding,
        避免把粗粒度语义 trilinear 放大 4 倍后再消费;两级跳连保留 32/16^3 细节。
        新增参数名均带 enc/down/up/dec/bott/img 前缀,旧权重 strict=False 可加载,
        但金字塔权重会缺失并随机初始化——金字塔实验必须重新训练,不能加载旧 ckpt 直接推理。

    Args:
        encoder_embed_dim: 图像编码器输出通道数(vit_b_ori 经 neck 后为 384)。
        low_res_size: 粗掩码 low-res 空间尺寸(128^3 输入 / patch16 再 x4 上采样 → 32)。
        prompt_dim: RefinementPromptEncoder 输出通道数,也是 refine_blocks 输入通道。
        num_blocks: use_pyramid=False 时各向异性残差精修块数量。
        use_pyramid: 是否启用 U 形多尺度金字塔(消融开关,默认 False 保持旧行为)。
        use_alpha: 是否启用残差融合 α(默认 True)。False 为消融模式:α 恒为 1,
            输出 refined = coarse + delta(固定单位残差),模型中不注册 alpha 参数。
            注意不能直接返回 delta——小随机初始化的 head 起步输出≈空白掩码,
            会切断预训练粗掩码通路导致训练 Dice 恒 0 且短 schedule 内难以恢复。
    """

    def __init__(
        self,
        encoder_embed_dim: int = 384,
        low_res_size: int = 32,
        prompt_dim: int = 64,
        num_blocks: int = 3,
        use_pyramid: bool = False,
        use_alpha: bool = True,
    ):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.low_res_size = low_res_size
        self.use_pyramid = use_pyramid
        self.use_alpha = use_alpha

        if not use_pyramid:
            # refine_blocks 输入 = prompt_dim(来自 RefinementPromptEncoder 的 dense_prompt)
            self.refine_blocks = nn.Sequential(
                *[AnisotropicRefineBlock(prompt_dim) for _ in range(num_blocks)]
            )
        else:
            C = prompt_dim
            # ---- 编码臂(下采样;深度可分离 stride-2 卷积,通道混合交给后续残差块内 1x1) ----
            self.enc0 = AnisotropicRefineBlock(C)                 # 32^3
            self.down1 = nn.Conv3d(C, C, kernel_size=2, stride=2, groups=C, bias=False)  # →16^3
            self.enc1 = AnisotropicRefineBlock(C)                 # 16^3
            self.down2 = nn.Conv3d(C, C, kernel_size=2, stride=2, groups=C, bias=False)  # →8^3
            # ---- 瓶颈:8^3 原生尺度直连图像编码器特征(不经 4x trilinear 放大) ----
            self.img_proj = nn.Conv3d(encoder_embed_dim, C, kernel_size=1, bias=False)
            self.bott_fuse = nn.Conv3d(C * 2, C, kernel_size=1, bias=False)
            self.bott = AnisotropicRefineBlock(C)                 # 8^3
            # ---- 解码臂(上采样 + skip concat 融合) ----
            self.up1 = nn.ConvTranspose3d(C, C, kernel_size=2, stride=2, groups=C, bias=False)  # →16^3
            self.dec1_fuse = nn.Conv3d(C * 2, C, kernel_size=1, bias=False)
            self.dec1 = AnisotropicRefineBlock(C)                 # 16^3
            self.up2 = nn.ConvTranspose3d(C, C, kernel_size=2, stride=2, groups=C, bias=False)  # →32^3
            self.dec0_fuse = nn.Conv3d(C * 2, C, kernel_size=1, bias=False)
            self.dec0 = AnisotropicRefineBlock(C)                 # 32^3

            for m in (self.img_proj, self.bott_fuse,
                      self.dec1_fuse, self.dec0_fuse):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

        # 逐体素精修 logits 头
        self.head = nn.Conv3d(prompt_dim, 1, kernel_size=1)

        # 可学习残差标量 α(初始 0):先保护粗掩码先验,后渐进精修。
        # use_alpha=False(消融)时不注册参数,前向按 α≡1 的固定单位残差融合
        if use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.alpha = None

        self._init_head()

    def _init_head(self):
        # 启用 α 时:不能把 head 权重和 α 同时初始化为 0,否则
        #   ∂L/∂head = grad·α·x = 0 且 ∂L/∂α = grad·δ = 0,
        # 精修头与 α 构成永久零梯度死锁,第二阶段永不学习。
        # 这里给 head 小随机权重(δ 初始非零),α 仍为 0:
        # 起步输出恒等粗掩码(保护先验),同时 α 能收到非零梯度而渐进长大。
        # 禁用 α 时:head 直接输出精修 logits,小随机初始化同样合理
        nn.init.normal_(self.head.weight, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        dense_prompt: torch.Tensor,
        coarse_logits: torch.Tensor,
        image_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
          dense_prompt:   (B, prompt_dim, D, H, W) 来自 RefinementPromptEncoder,
                           以前一阶段粗掩码为主、编码器特征为辅融合而成。
          coarse_logits:  (B, 1, D, H, W) 第一阶段粗解码器的 low-res logits(32^3)。
          image_embedding:(B, encoder_embed_dim, d, h, w) 编码器 neck 输出(8^3)。
                           仅 use_pyramid=True 时必需,用于瓶颈层原生尺度直连;
                           平铺结构忽略该参数,旧调用方式不受影响。
        Returns:
          refined_logits: (B, 1, D, H, W) 精修后的 low-res logits。
        """
        if not self.use_pyramid:
            x = self.refine_blocks(dense_prompt)
        else:
            if image_embedding is None:
                raise ValueError(
                    "use_pyramid=True 时 forward 必须提供 image_embedding(8^3 编码器特征)")
            # 编码臂
            x0 = self.enc0(dense_prompt)                # 32^3
            x1 = self.enc1(self.down1(x0))             # 16^3
            x2 = self.down2(x1)                         # 8^3
            # 瓶颈:图像特征投影后对齐到 x2 网格(防御非 32 输入),concat 融合
            img_feat = self.img_proj(image_embedding)
            if img_feat.shape[-3:] != x2.shape[-3:]:
                img_feat = F.interpolate(
                    img_feat, size=x2.shape[-3:],
                    mode='trilinear', align_corners=False)
            b = self.bott(self.bott_fuse(
                torch.cat([x2, img_feat], dim=1)))      # 8^3
            # 解码臂(可学习转置卷积上采样 + 跳连)
            d1 = self.dec1(self.dec1_fuse(
                torch.cat([self.up1(b), x1], dim=1)))   # 16^3
            x = self.dec0(self.dec0_fuse(
                torch.cat([self.up2(d1), x0], dim=1)))  # 32^3

        delta = self.head(x)

        # α 残差渐进融合;alpha=None(消融)时退化为固定单位残差 α≡1:
        # refined = coarse + delta。必须保留 coarse 通路——head 为小随机初始化,
        # 若直接返回 delta,初始输出≈空白掩码(logits≈0),在训练 Dice 原始
        # logits>0.5 的口径下恒为 0,且短 schedule 内小精修头学不出完整掩码。
        # 固定残差下起步 refined≈coarse(预训练掩码),delta 从第一步即收满梯度
        if self.alpha is None:
            return coarse_logits + delta
        refined_logits = coarse_logits + self.alpha * delta
        return refined_logits


# ============================================================
# SAM 同构版二阶段(--refine_type sam,创新点二主线)
# ============================================================
class RefinementPromptEncoder3D(PromptEncoder3D):
    """二阶段提示编码器:与一阶段 PromptEncoder3D 结构完全一致。

    分工差异仅在输入:二阶段不接收 points/boxes,唯一交互信息是一阶段
    粗掩码 logits(32^3,经 mask_downscaling 两级 stride2 → 8^3 dense)。
    因此:
      - sparse embedding 恒为空(B, 0, embed_dim);
      - dense embedding 为粗掩码的编码(B, embed_dim, 8, 8, 8),
        与一阶段点击轮次中"带掩码输入"时的通路完全相同。
    权重从头随机初始化,不与一阶段共享/拷贝。
    """

    def forward(self, coarse_logits: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """以粗掩码为唯一 prompt,返回 (sparse_empty, dense)。

        coarse_logits=None 时走父类的 no_mask_embed 分支,以一张可学习的
        无条件占位嵌入代替掩码 prompt(R1-a 无提示对照:同结构、同参数、
        同前向次数,但没有病例相关的条件信号)。
        """
        return super().forward(points=None, boxes=None, masks=coarse_logits)


class RefinementMaskDecoder3D(MaskDecoder3D):
    """二阶段精修解码器:与一阶段 MaskDecoder3D 结构完全一致。

    消费冻结图像编码器的 8^3 image embedding + 二阶段 dense prompt,
    经完整 TwoWayTransformer3D(2 层双向注意力)输出 32^3 单掩码 logits,
    再以阶段残差标量 α(初始 0)与粗掩码凸组合:
        refined = coarse + α * (refine - coarse)
    起步 α=0 → 恒等粗掩码(保护预训练先验,初始 Dice 不塌);
    refine 为随机初始化解码器的输出,与 coarse 天然不同,
    α 从第一步即收到非零梯度,先苏醒后带动解码器权重更新。

    use_alpha=False(消融 --disable_refine_alpha)时无可学习 α,
    二阶段解码器以「零初始化残差头」起步:
        掩码生成超网络 output_hypernetworks_mlps 末层权重/偏置置零
        → 初始 refine delta ≡ 0 → refined = coarse + 0 ≡ coarse;
        二阶段从第一步即以满梯度学习对粗掩码的修正量(无 α 门控),
        与 light 版 refined=coarse+delta(α≡1) 的消融语义一致。
        注意:不能直接返回随机 decoder 的完整输出 refined=refine,
        那会使粗掩码通路完全消失、初始 Dice 恒为 0。
    iou_prediction_head 保留(结构对齐一阶段),输出不参与训练损失。
    权重从头随机初始化。
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_alpha: bool = True,
    ) -> None:
        super().__init__(
            transformer_dim=transformer_dim,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        )
        self.use_alpha = use_alpha
        if use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.alpha = None
            # 零初始化掩码超网络末层:单掩码 token 经此生成 1x1x1 卷积权重,
            # 末层置零 → hyper_in≡0 → refine_masks≡0,初始残差为零,
            # refined=coarse+0 恒等粗掩码;训练中该层学习输出修正量
            for hyper_mlp in self.output_hypernetworks_mlps:
                nn.init.zeros_(hyper_mlp.layers[-1].weight)
                nn.init.zeros_(hyper_mlp.layers[-1].bias)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        coarse_logits: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """与一阶段 forward 同参,额外以 coarse_logits 做 α 凸组合。

        固定 multimask_output=False(单掩码),训练/推理口径一致。
        返回 (refined_low_res (B,1,32,32,32), iou_pred)。
        """
        refine_masks, iou_pred = super().forward(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )
        if self.alpha is None:
            # 消融(零初始化单位残差):refine_masks 起步为零 delta,
            # refined=coarse+delta 保留粗掩码通路,delta 随训练学成修正量
            return coarse_logits + refine_masks, iou_pred
        refined_logits = coarse_logits + self.alpha * (refine_masks - coarse_logits)
        return refined_logits, iou_pred
