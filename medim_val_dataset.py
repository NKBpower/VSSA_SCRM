# -*- encoding: utf-8 -*-
"""数据集级批量验证(支持 PEFT 适配器 / 两阶段精修模型)。

与原版差异:
  - 通过 build_sam3D_with_mamba_adapter 构建 PEFT 模型结构,strict=False 加载本地检查点;
  - --use_refinement 加载创新点二的两阶段粗到精权重(推理时自动调用精修解码器);
  - 图像名与标签名需 base 一致(imagesVal / labelsVal 下同名 xxx.nii.gz)。

用法示例:
  python medim_val_dataset.py \
    --img_dir /data/xijing_dataset_sam3d/imagesVal \
    --gt_dir  /data/xijing_dataset_sam3d/labelsVal \
    --out_dir  ./pred_peft_refine \
    --ckpt_path work_dir/peft_refine_b2x1/sam_model_dice_best.pth \
    --use_refinement --num_clicks 11
"""
import argparse
import os
import os.path as osp
from glob import glob

import torch
from tqdm import tqdm

from segment_anything.build_sam3D import build_sam3D_with_mamba_adapter
from utils.infer_utils import validate_paired_img_gt


def build_peft_model(ckpt_path: str, use_refinement: bool, force_pyramid: bool = False):
    """构建 PEFT 模型并加载本地训练检查点(strict=False)。

    金字塔结构自动识别:检查点中出现 refinement_decoder.enc0.* 权重即为金字塔模型,
    无需手动记忆 --use_pyramid;force_pyramid 仅用于检查点缺失相关权重时的强制兜底。
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 从权重内容识别 U 形金字塔(平铺结构无 enc0/down1/bott 等键)
    ckpt_has_pyramid = any(
        k.startswith('refinement_decoder.enc0.') for k in state_dict.keys())
    use_pyramid = bool(force_pyramid or ckpt_has_pyramid)

    # 从权重内容识别 α 消融结构:
    #  编码器侧为逐层可学习门控 image_encoder.all_gate.<i>(12 个标量,
    #  初值 0),正常随 checkpoint 一起保存,无需开关识别;
    #  精修融合 α 仍需识别:
    #  - refinement_decoder.alpha 存在 → 带 α 结构;不存在 → --disable_refine_alpha。
    ckpt_keys = state_dict.keys()
    ckpt_has_refine_alpha = 'refinement_decoder.alpha' in ckpt_keys
    ckpt_gate_keys = [k for k in ckpt_keys if '.all_gate.' in k]

    # 旧版 checkpoint 中已删除的编码器门控键:这些键会被 strict=False 忽略,
    # 但适配器权重是在旧门控语义下训练的,加载到新结构结果不可比,必须重训。
    # 另检测"固定 0.01 残差时代"的 ckpt(2026-09-23 版,有适配器权重但无
    # all_gate):加载后 12 个 gate 保持初值 0 → 适配器全部被静音,
    # 等价于纯冻结编码器,验证结果无意义,必须用新代码重训。
    ckpt_legacy_gate = [k for k in ckpt_keys
                        if k in ('image_encoder.alpha_conv',
                                 'image_encoder.alpha_mamba')
                        or k.endswith('.adapter.gamma')
                        or '.adapter.axis_gate.' in k
                        or (k.startswith('image_encoder.blocks.')
                            and k.endswith('.alpha'))]
    ckpt_has_adapter = any('.adapter.' in k for k in ckpt_keys)
    ckpt_missing_gate = bool(ckpt_has_adapter) and not ckpt_gate_keys

    # 从权重内容识别二阶段精修结构(防止用错结构构建导致键失配):
    #  sam 同构版 = 完整 MaskDecoder3D 子类,含 mask_tokens/iou_token/transformer;
    #  light 轻量版 = 纯卷积分支,含 head./refine_blocks/blend_logit。
    #  无精修权重(use_refinement=False)时默认 sam 不影响,因为该模块不会构建
    if 'refinement_decoder.mask_tokens.weight' in ckpt_keys:
        ckpt_refine_type = 'sam'
    elif any('refinement_decoder.head.' in k or 'refine_blocks' in k
             for k in ckpt_keys):
        ckpt_refine_type = 'light'
    else:
        ckpt_refine_type = 'sam'

    # 识别编码器适配器层分配 mamba_start(浅 i<mamba_start 用卷积适配器
    # proj=Conv3d 5D;深 i>=mamba_start 用 Mamba 适配器 proj=Linear 2D)。
    # 训练侧 argparse 默认 mamba_start=0(全 Mamba),而构建器默认 depth//2=6,
    # 不恢复会导致 0-5 层结构建错、proj 形状 [96,768] vs [96,768,1,1,1] 失配。
    # 优先用 checkpoint 保存的训练 args;旧 ckpt 无 args 时按权重形状回退推断。
    # 识别 SS3D 扫描方向消融结构 scan_mode(决定 SSM 核拓扑:6 核键名
    # ssm_{d,h,w}_{fwd,bwd} vs 3 核键名 ssm_{d,h,w};两种键空间不相交,
    # 建错结构会在形状预检/missing 键审计暴露,但先从 ckpt args 正确恢复)。
    # 旧 ckpt 无 args/字段 → bi_separate + sum,严格复现训练时数值口径。
    ckpt_train_args = checkpoint.get('args', None)
    ckpt_scan_mode = getattr(ckpt_train_args, 'scan_mode', None)
    scan_mode_source = 'checkpoint args'
    if ckpt_scan_mode is None:
        # 无训练 args 的旧 ckpt:从权重键名推断拓扑
        if any('.adapter.ssm_d.' in k or '.adapter.ssm_h.' in k
               or '.adapter.ssm_w.' in k for k in state_dict.keys()):
            ckpt_scan_mode = 'bi_shared'  # 3 核键名无法区分 shared/uni,
            # 二者拓扑一致仅前向路由不同;无 args 的旧 ckpt 不可能来自
            # 这两种新模式(2026-09-25 才引入),此处仅为兜底
        else:
            ckpt_scan_mode = 'bi_separate'
        scan_mode_source = '权重键名推断(旧 ckpt 默认)'
    if ckpt_scan_mode not in ('bi_separate', 'bi_shared', 'uni'):
        ckpt_scan_mode = 'bi_separate'
        scan_mode_source += '(非法值回退)'
    ckpt_scan_merge = getattr(ckpt_train_args, 'scan_merge', None)
    if ckpt_scan_merge not in ('mean', 'sum'):
        ckpt_scan_merge = 'sum'  # 2026-09-25 前 ckpt 一律裸求和
    ckpt_mamba_start = getattr(ckpt_train_args, 'mamba_start', None)
    mamba_start_source = 'checkpoint args'
    if ckpt_mamba_start is None:
        mamba_layers = []
        for i in range(12):
            key = f'image_encoder.blocks.{i}.adapter.proj_in.weight'
            if key in state_dict:
                # 2D→Mamba(Linear)适配器;5D→卷积(Conv3d 1x1)适配器
                mamba_layers.append(i if state_dict[key].dim() == 2 else None)
        if any(i is not None for i in mamba_layers):
            mamba_indices = [i for i in mamba_layers if i is not None]
            ckpt_mamba_start = min(mamba_indices)
            mamba_start_source = '权重形状推断'
        else:
            ckpt_mamba_start = 6  # checkpoint 无适配器键(如非 PEFT 权重)时用构建器默认
            mamba_start_source = '默认 depth//2'
    # 无适配器 Baseline(A-bl):ckpt 不含 any adapter/all_gate 权重时退回纯冻结 Vit
    ckpt_use_adapter = getattr(ckpt_train_args, 'use_adapter', True)
    if not ckpt_use_adapter:
        try:
            adapter_key_exists = any(
                ('.adapter.' in k) or ('all_gate' in k) for k in state_dict)
        except TypeError:
            adapter_key_exists = False
        if adapter_key_exists:
            print(f"[Warning] ckpt args 标 use_adapter=False,但含适配器权重,"
                  f"仍按 use_adapter=False 构建(可能来自不一致训练,请核对)")
    else:
        # args 缺失/默认 True 时按权重键自动识别(向后兼容任意来源 ckpt)
        if ckpt_train_args is None:
            try:
                adapter_key_exists = any(
                    ('.adapter.' in k) or ('all_gate' in k) for k in state_dict)
            except TypeError:
                adapter_key_exists = False
            if not adapter_key_exists:
                ckpt_use_adapter = False

    # 适配器瓶颈压缩比 / SS3D d_state 恢复(Adapter 降维消融):
    # 结构超参以 checkpoint 保存的训练 args 为唯一真源;旧 ckpt 无 args 时
    # 从适配器权重形状反推——瓶颈实际宽度 inner_dim 才是结构量,ratio 只是
    # max(dim//ratio,8) 的生成参数,故形状反推可做到无损恢复。
    ckpt_ratio = getattr(ckpt_train_args, 'ratio', None)
    ratio_source = 'checkpoint args'
    ckpt_d_state = getattr(ckpt_train_args, 'd_state', None)
    d_state_source = 'checkpoint args'
    if not ckpt_use_adapter:
        # 无适配器模型不构建任何适配器,ratio/d_state 无结构意义,用构建器默认
        ckpt_ratio = 8
        ckpt_d_state = 16
        ratio_source = 'N/A(无适配器)'
        d_state_source = 'N/A(无适配器)'
    else:
        # 任取一层适配器的 proj_in:SS3D 为 Linear 权重 [inner,dim],
        # Conv 适配器为 Conv3d 权重 [inner,dim,1,1,1],两种情况都是
        # shape[0]=瓶颈宽度 inner_dim,shape[1]=基座通道 dim
        proj_key = next((k for k in state_dict
                         if k.startswith('image_encoder.blocks.')
                         and k.endswith('.adapter.proj_in.weight')), None)
        if proj_key is not None:
            _wshape = tuple(state_dict[proj_key].shape)
            inner_dim, base_dim = int(_wshape[0]), int(_wshape[1])
            if ckpt_ratio is None:
                # inner = max(dim//ratio,8):inner>8 时 ratio=dim//inner 精确可还原
                # (整除关系);inner==8 时 ratio≥dim/8 不可区分,但结构只取决于
                # inner_dim,取 dim//8 重建出的瓶颈仍是 8,结构完全等价
                guess_ratio = max(base_dim // max(inner_dim, 1), 1)
                if max(base_dim // guess_ratio, 8) == inner_dim:
                    ckpt_ratio = guess_ratio
                else:
                    ckpt_ratio = 8
                ratio_source = f'权重形状推断({proj_key} inner={inner_dim})'
            if ckpt_d_state is None:
                # SelectiveSSM.A_log 形状 [d_inner, d_state]
                # (d_inner=expand*inner);取 ssm_* 下的 A_log 读第 1 维
                a_log_key = next((k for k in state_dict
                                  if '.adapter.' in k and k.endswith('.A_log')),
                                 None)
                if a_log_key is not None:
                    ckpt_d_state = int(state_dict[a_log_key].shape[1])
                    d_state_source = f'权重形状推断({a_log_key})'
                else:
                    ckpt_d_state = 16
                    d_state_source = '默认 16(未找到 A_log 键)'
        else:
            # use_adapter=True 但没找到适配器键(极旧/异常 ckpt),回退默认值
            if ckpt_ratio is None:
                ckpt_ratio = 8
                ratio_source = '默认 8(未找到适配器 proj_in 键)'
            if ckpt_d_state is None:
                ckpt_d_state = 16
                d_state_source = '默认 16(未找到适配器 proj_in 键)'

    model = build_sam3D_with_mamba_adapter(
        model_type='vit_b_ori',
        checkpoint=None,
        use_refinement=use_refinement,
        use_pyramid=use_pyramid,
        use_refine_alpha=ckpt_has_refine_alpha,
        refine_type=ckpt_refine_type,
        mamba_start=ckpt_mamba_start,
        scan_mode=ckpt_scan_mode,
        scan_merge=ckpt_scan_merge,
        use_adapter=ckpt_use_adapter,
        ratio=ckpt_ratio,
        d_state=ckpt_d_state,
    )

    # 形状预检:strict=False 下形状失配会被静默拆成 missing+unexpected,
    # 对应适配器保持随机初始化(预测垃圾但不报错),必须在加载前显式拦截
    model_state = model.state_dict()
    shape_mismatch = [
        (k, tuple(state_dict[k].shape), tuple(model_state[k].shape))
        for k in state_dict.keys() & model_state.keys()
        if state_dict[k].shape != model_state[k].shape
    ]
    if shape_mismatch:
        details = '\n          '.join(
            f'{k}: ckpt{cs} vs model{ms}' for k, cs, ms in shape_mismatch[:8])
        raise RuntimeError(
            f"checkpoint 与模型结构存在 {len(shape_mismatch)} 个形状失配,已中止加载"
            f"(避免适配器随机初始化产生垃圾预测)。示例:\n          {details}\n"
            f"请检查该 checkpoint 训练时的结构参数(--mamba_start/--ratio/--d_state/"
            f"--refine_type/--use_pyramid/--disable_*_alpha)是否与自动识别结果一致。")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # 二阶段 sparse 透传是纯前向路由(不引入新参数),无法从权重键识别,
    # 改为读取 checkpoint 保存的训练 args 自动还原,保证训练/验证严格一致。
    # 旧 checkpoint 无 args 字段 → False(与历史行为完全一致)。
    ckpt_refine_use_sparse = bool(
        getattr(ckpt_train_args, 'refine_use_sparse', False))
    model.refine_use_sparse = bool(
        use_refinement and ckpt_refine_type == 'sam' and ckpt_refine_use_sparse)
    # R1 归因消融: 二阶段 mask prompt 来源同样是纯前向路由,从训练 args 还原,
    # 否则 R1-a/b 检查点会被当成主方法(真提示)评估,指标全部失效。
    # 旧 checkpoint 无该字段 → 'true'(与历史行为完全一致)。
    ckpt_refine_prompt_mode = str(
        getattr(ckpt_train_args, 'refine_prompt_mode', 'true'))
    model.refine_prompt_mode = (
        ckpt_refine_prompt_mode if use_refinement and ckpt_refine_type == 'sam'
        else 'true')

    refine_prefixes = ('refinement_decoder', 'refinement_prompt_encoder')
    refinement_missing = [k for k in missing
                          if k.startswith(refine_prefixes)]
    if use_refinement and refinement_missing:
        print(f"[Warning] 精修解码器/提示编码器参数缺失 {len(refinement_missing)} 个,"
              f"请确认检查点来自 --use_refinement 训练,且金字塔开关与训练时一致")
    refinement_unexpected = [k for k in unexpected
                             if k.startswith(refine_prefixes)]
    if not use_refinement and refinement_unexpected:
        print(f"[Warning] 检查点含 {len(refinement_unexpected)} 个精修模块权重"
              f"(refinement_decoder/refinement_prompt_encoder),但未指定 --use_refinement,"
              f"本次仅运行粗解码器,结果不可与精修模型对比,请确认是否漏加参数!")
    print(f"[Checkpoint] loaded: {ckpt_path}")
    print(f"[Checkpoint] mode: {'coarse+refine(两阶段)' if use_refinement else 'coarse-only(仅粗解码器)'}")
    if use_refinement:
        print(f"[Checkpoint] refine_type: {ckpt_refine_type} "
              f"({'SAM 同构 PE+Decoder' if ckpt_refine_type == 'sam' else '轻量卷积头'})")
        print(f"[Checkpoint] refine_use_sparse: "
              f"{'on(二阶段透传一阶段 points/boxes sparse)' if model.refine_use_sparse else 'off(二阶段 sparse 为空)'}")
    print(f"[Checkpoint] pyramid: {'on' if use_pyramid else 'off'} "
          f"(ckpt 检测: {'含金字塔权重' if ckpt_has_pyramid else '平铺结构'}"
          f"{', 命令行强制开启' if force_pyramid and not ckpt_has_pyramid else ''})")
    if ckpt_mamba_start <= 0:
        alloc_desc = '12 层全部为 Mamba 适配器'
    elif ckpt_mamba_start >= 12:
        alloc_desc = '12 层全部为卷积适配器'
    else:
        alloc_desc = (f'层 0~{ckpt_mamba_start - 1} 卷积适配器 / '
                      f'层 {ckpt_mamba_start}~11 Mamba 适配器')
    print(f"[Checkpoint] mamba_start: {ckpt_mamba_start} "
          f"(来源: {mamba_start_source};{alloc_desc})")
    if not ckpt_use_adapter:
        print("[Checkpoint] adapter: 无(use_adapter=False,A-bl 纯冻结骨干 Baseline)")
    else:
        print("[Checkpoint] adapter: 有(SS3D/卷积旁路适配器)")
        print(f"[Checkpoint] adapter_ratio: {ckpt_ratio} "
              f"(瓶颈 inner_dim={max(768 // ckpt_ratio, 8)}@768 通道,"
              f"来源: {ratio_source})")
        print(f"[Checkpoint] d_state: {ckpt_d_state} (来源: {d_state_source})")
    _scan_desc = {'bi_separate': '6 独立核(前/后向各一套)',
                  'bi_shared': '3 核(前/后向翻转共享)',
                  'uni': '3 核(仅前向单向)'}[ckpt_scan_mode]
    _scan_terms = 3 if ckpt_scan_mode == 'uni' else 6
    print(f"[Checkpoint] scan_mode: {ckpt_scan_mode} "
          f"({_scan_desc},{_scan_terms} 方向项),merge: {ckpt_scan_merge} "
          f"(来源: {scan_mode_source})")
    if ckpt_legacy_gate:
        print(f"[Warning] checkpoint 含 {len(ckpt_legacy_gate)} 个已删除的旧门控键"
              f"(alpha_conv/alpha_mamba/gamma/axis_gate/逐层α),已被忽略;"
              f"该 checkpoint 训练于旧门控结构,与当前逐层 gate 编码器语义不一致,"
              f"数值结果不可比,请用新代码重新训练后再验证!")
    elif ckpt_missing_gate:
        print(f"[Warning] checkpoint 含适配器权重但无 all_gate 门控键(训练于"
              f"固定 1e-2 残差时代);加载后 12 个 gate 保持初值 0 → 适配器"
              f"全部静音(等价纯冻结编码器),请用新代码重新训练后再验证!")
    if ckpt_gate_keys:
        gate_vals = [float(state_dict[k].reshape(-1)[0]) for k in sorted(
            ckpt_gate_keys, key=lambda s: int(s.rsplit('.', 1)[-1]))]
        gate_str = '[' + ', '.join(f'{v:+.4f}' for v in gate_vals) + ']'
        print(f"[Checkpoint] adapter_gate: {len(gate_vals)} 个逐层标量 {gate_str}")
    if use_refinement:
        print(f"[Checkpoint] refine_alpha: "
              f"{'on(渐进融合 α)' if ckpt_has_refine_alpha else 'off(固定单位残差 α≡1: refined=coarse+delta)'}")
    print(f"[Checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True,
                        help="验证集图像目录 (imagesVal)")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="验证集标签目录 (labelsVal)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="预测输出目录")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="PEFT 训练检查点,如 work_dir/peft_refine_b2x1/sam_model_dice_best.pth")
    parser.add_argument("--num_clicks", type=int, default=11,
                        help="交互点击次数(需与训练设定一致)")
    parser.add_argument("--use_refinement", action="store_true", default=False,
                        help="检查点是否含创新点二两阶段精修解码器")
    parser.add_argument("--use_pyramid", action="store_true", default=False,
                        help="强制按金字塔结构构建(默认自动:按检查点权重内容识别,一般无需指定)")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5],
                        help="推理重采样目标 spacing(mm),需与训练数据一致")
    parser.add_argument("--crop_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = build_peft_model(args.ckpt_path, args.use_refinement,
                             force_pyramid=args.use_pyramid)

    gt_fname_list = sorted(glob(osp.join(args.gt_dir, "*.nii.gz")))

    # R1-b 跨病例错配: 逐例推理时无法取"同批另一病例",改用上一例最后一轮的
    # 32^3 粗掩码作为本例的二阶段 mask prompt。与训练期 batch 内 shuffle 的
    # 语义一致(形状/数值分布同,病例内容错),且完全确定、可复现。
    use_foreign_prompt = getattr(model, 'refine_prompt_mode', 'true') == 'shuffle'
    if use_foreign_prompt:
        print("[R1-b] 跨病例错配推理: 每例使用上一例的粗掩码作为二阶段 mask prompt"
              "(首例无缓存,按空 prompt 处理)")
    prev_case_prompt = None

    for gt_fname in tqdm(gt_fname_list):
        case_name = osp.basename(gt_fname).replace(".nii.gz", "")
        img_path = osp.join(args.img_dir, f"{case_name}.nii.gz")
        gt_path = gt_fname
        out_path = osp.join(args.out_dir, f"{case_name}.nii.gz")
        if not osp.exists(img_path):
            print(f"[Warning] 图像不存在,跳过: {img_path}")
            continue
        last_low_res = validate_paired_img_gt(
            model, img_path, gt_path, out_path,
            num_clicks=args.num_clicks,
            crop_size=args.crop_size,
            target_spacing=tuple(args.spacing),
            r1_foreign_prompt=prev_case_prompt if use_foreign_prompt else None,
        )
        if use_foreign_prompt and last_low_res is not None:
            # 只传数值,不带计算图;下一例直接用它的内容(错误病例)作提示
            prev_case_prompt = last_low_res.detach().clone()


if __name__ == "__main__":
    main()
