# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch

from .modeling import ImageEncoderViT3D, MaskDecoder3D, PromptEncoder3D, Sam3D


def build_sam3D_vit_h(checkpoint=None):
    return _build_sam3D(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam3D = build_sam3D_vit_h


def build_sam3D_vit_l(checkpoint=None):
    return _build_sam3D(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam3D_vit_b(checkpoint=None):
    return _build_sam3D(
        # encoder_embed_dim=768,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_sam3D_vit_b_ori(checkpoint=None):
    return _build_sam3D_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry3D = {
    "default": build_sam3D_vit_h,
    "vit_h": build_sam3D_vit_h,
    "vit_l": build_sam3D_vit_l,
    "vit_b": build_sam3D_vit_b,
    "vit_b_ori": build_sam3D_vit_b_ori,
    # Mamba 风格轻量适配器模型（创新点一）
    "vit_b_ori_mamba_adapter": lambda checkpoint=None: build_sam3D_with_mamba_adapter(
        model_type='vit_b_ori', checkpoint=checkpoint),
    "vit_b_mamba_adapter": lambda checkpoint=None: build_sam3D_with_mamba_adapter(
        model_type='vit_b', checkpoint=checkpoint),
    "vit_l_mamba_adapter": lambda checkpoint=None: build_sam3D_with_mamba_adapter(
        model_type='vit_l', checkpoint=checkpoint),
    "vit_h_mamba_adapter": lambda checkpoint=None: build_sam3D_with_mamba_adapter(
        model_type='vit_h', checkpoint=checkpoint),
}


def build_sam3D_with_mamba_adapter(
    model_type='vit_b_ori',
    checkpoint=None,
    mamba_start=None,
    ratio=8,
    d_state=16,
    d_conv=4,
    expand=2,
    scan_mode='bi_separate',
    scan_merge='mean',
    freeze_prompt_encoder=False,
    freeze_mask_decoder=False,
    use_refinement=False,
    low_res_size=32,
    use_pyramid=False,
    use_refine_alpha=True,
    refine_type='sam',
    use_adapter=True,
):
    """构建带 Mamba 风格适配器的 SAM-Med3D 模型（创新点一 + 可选创新点二）。

    详见 segment_anything.modeling.mamba_adapter3D.build_sam3D_with_mamba_adapter。
    use_refine_alpha 为二阶段精修融合 α 消融开关(默认 True);
    refine_type='sam' 为 SAM 同构二阶段(默认),'light' 为旧轻量卷积版。
    编码器侧为 12 个逐层可学习 gate(image_encoder.all_gate);
    scan_mode 为 SS3D 扫描方向消融(bi_separate/bi_shared/uni),
    scan_merge='mean' 为新训练默认,'sum' 仅验证旧 ckpt 时自动恢复。
    use_adapter=False 时退化为"无适配器纯冻结骨干"A-bl 消融 Baseline
    (不构建 all_gate 与适配器权重,键结构与原始冻结 Vit 完全一致)。
    """
    from .modeling.mamba_adapter3D import build_sam3D_with_mamba_adapter as _build
    return _build(
        model_type=model_type,
        checkpoint=checkpoint,
        mamba_start=mamba_start,
        ratio=ratio,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        scan_mode=scan_mode,
        scan_merge=scan_merge,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_mask_decoder=freeze_mask_decoder,
        use_refinement=use_refinement,
        low_res_size=low_res_size,
        use_pyramid=use_pyramid,
        use_refine_alpha=use_refine_alpha,
        refine_type=refine_type,
        use_adapter=use_adapter,
    )


def _build_sam3D(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 384
    image_size = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size,
                                  image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def _build_sam3D_ori(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 384
    image_size = 128
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size,
                                  image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam
