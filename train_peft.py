# set up environment
"""PEFT 训练入口(Mamba 风格轻量适配器)
=====================================
基于 train.py 的最小修改版本,启用 SAM-Med3D + Mamba 适配器高效参数微调路径。

与 train.py 的差异:
  1. build_model 构建 vit_b_ori_mamba_adapter 模型(冻结骨干 + 旁路适配器 + α);
  2. set_optimizer 只收集 requires_grad=True 的参数,按模块分组设置不同学习率
     (适配器用 adapter_lr,借鉴 LoRA+ 思想给适配器更高 lr);
  3. init_checkpoint 默认 strict=False,兼容不含适配器参数的预训练权重;
  4. 训练开始前打印 PEFT 参数统计(可训练占比)。

用法:
    python train_peft.py --task_name "peft_b2x1" --checkpoint "ckpt/sam_med3d_turbo.pth"
"""
import argparse
import datetime
import logging
import os
import random
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from torch.backends import cudnn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.modeling.mamba_adapter3D import print_param_stats
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas

join = os.path.join

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='peft_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori_mamba_adapter')
parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d_turbo.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=True)

# PEFT 适配器超参
parser.add_argument('--mamba_start', type=int, default=0,
                    help='Mamba 适配器起始层索引,None 时自动取 depth//2')
parser.add_argument('--use_adapter', action=argparse.BooleanOptionalAction, default=True,
                    help='是否注入旁路适配器。--no-use_adapter 时为"无适配器纯冻结'
                         '骨干"A-bl 消融 Baseline(冻结全部编码器 Block3D/patch_embed'
                         '/pos_embed/neck,不构建 all_gate 与适配器权重,仅训练粗解码器'
                         '/prompt_encoder 等头部)')
parser.add_argument('--ratio', type=int, default=8, help='适配器瓶颈压缩比')
parser.add_argument('--d_state', type=int, default=16, help='SS3D 状态空间维度')
parser.add_argument('--d_conv', type=int, default=4, help='SS3D 因果卷积核大小')
parser.add_argument('--expand', type=int, default=2, help='SS3D 内部扩展系数')
parser.add_argument('--scan_mode', type=str, default='bi_separate',
                    choices=['bi_separate', 'bi_shared', 'uni'],
                    help='SS3D 扫描方向消融(仅深层 Mamba 适配器生效):'
                         'bi_separate=3 轴×前/后向 6 个独立 SSM 核(旧默认);'
                         'bi_shared=每轴 1 个核、翻转序列复用(VMamba 式,SSM 核'
                         '参数减半,小数据集下单核训练信号翻倍);'
                         'uni=每轴仅前向单向扫描(轴向起点有盲区,对照用)。'
                         '三种模式多分支统一按方向项数取平均(双向6项/单向3项),'
                         '输出尺度严格可比;结构不同,切换必须重新训练')
parser.add_argument('--freeze_prompt_encoder', action='store_true', default=False)
parser.add_argument('--freeze_mask_decoder', action='store_true', default=False,
                    help='是否冻结粗掩码解码器(默认 False:粗解码器与精修头联合训练;加上该 flag 才冻结)')
parser.add_argument('--full_finetune', action='store_true', default=False,
                    help='全微调实验:解冻图像编码器全部参数(12 个 Block3D/patch_embed/'
                         'pos_embed/neck),与适配器/两套解码器联合训练。开启后自动忽略 '
                         '--freeze_prompt_encoder/--freeze_mask_decoder;优化器中预训练'
                         '部分(骨干+PE+粗解码器)统一用 --lr(建议 1e-5 小 lr,远小于 '
                         'train.py 纯模型全微调的 8e-4,因本模型含随机初始化旁路),随机'
                         '初始化部分(adapter/精修头/gate/α)保持各自高 lr')
parser.add_argument('--grad_clip', type=float, default=0.0,
                    help='梯度裁剪 max_norm(0=关闭,全微调建议 1.0;在 GradScaler '
                         'unscale 后、optimizer.step 前裁剪)')
parser.add_argument('--prompt_lr', type=float, default=1e-5,
                    help='prompt_encoder 专用学习率(解冻时使用,通常远小于其它参数)')
parser.add_argument('--decoder_lr', type=float, default=1e-4,
                    help='粗掩码解码器 mask_decoder 解冻时的专用学习率')

# 创新点二:两阶段粗到精解码器
parser.add_argument('--use_refinement', action='store_true', default=False,
                    help='启用两阶段粗到精解码器(挂载轻量精修头);粗解码器是否冻结由 --freeze_mask_decoder 单独控制')
parser.add_argument('--low_res_size', type=int, default=32,
                    help='粗/精修掩码 low-res 空间尺寸(128^3 输入为 32)')
parser.add_argument('--refine_lr', type=float, default=5e-4,
                    help='精修解码器专用学习率')
parser.add_argument('--alpha_lr', type=float, default=3e-3,
                    help='二阶段精修融合 α 专用学习率(高 lr 加速苏醒,wd=0)')
parser.add_argument('--gate_lr', type=float, default=3e-3,
                    help='编码器逐层适配器门控 all_gate(12 个标量,初值 0)专用学习率;'
                         '高 lr + wd=0 加速各层适配器苏醒,初训勿与适配器权重同 lr')
parser.add_argument('--use_pyramid', action='store_true', default=False,
                    help='精修解码器启用 U 形多尺度金字塔(32→16→8→16→32,8^3 直连图像特征);'
                         '仅 --use_refinement 时生效,消融用,开启后需从头训练精修模块')

# α 消融开关:全模型唯一的可学习门控是二阶段精修融合 α(默认保留);
# --disable_refine_alpha 加上即去掉(改为固定单位残差)
parser.add_argument('--disable_refine_alpha', action='store_true', default=False,
                    help='消融:去掉精修融合 α,改为固定单位残差 refined=coarse+delta,'
                         '残差头零初始化故起步恒等粗掩码(初始 Dice 不塌),二阶段满梯度学修正量;'
                         '仅 --use_refinement 时有效。注意:store_true 请保持 default=False,'
                         '在命令行显式加 flag 才生效,勿改 default=True(会导致无法恢复带 α)')
parser.add_argument('--refine_type', type=str, default='sam', choices=['sam', 'light'],
                    help="二阶段精修结构:'sam'(默认)=与一阶段同构的 PromptEncoder3D+"
                         "MaskDecoder3D(~7.58M,从头随机训练,粗掩码为唯一二阶段 prompt);"
                         "'light'=旧纯卷积轻量版(~73k,支持 --use_pyramid)。仅 --use_refinement 时有效")
parser.add_argument('--detach_coarse', type=int, default=1,
                    help='喂给二阶段的粗掩码是否 stop-gradient(默认 1=开启,防两阶段共适应):'
                         '切断「精修损失→粗解码器」的两条梯度通路(α 凸组合的 (1-α) 通路 + '
                         '二阶段 PE 的 dense prompt 通路),粗阶段仅由自身 GT 损失监督、'
                         '精修阶段只能修正给定粗掩码而不能改造它;前向数值不变(α=0 仍恒等 coarse)。'
                         '设 0 恢复旧的联合反传(不推荐,初期精修损失 100% 回灌粗解码器)')
parser.add_argument('--coarse_loss_weight', type=float, default=1.0,
                    help='两阶段深监督中粗解码器自身 GT 损失的权重(默认 1.0,与精修等权;'
                         'detach_coarse=1 后两阶段职责独立,无需再靠降权防共适应;旧版为 0.5)')

# SPT(SAM-SPT, AAAI 2025)式二阶段优化策略:均为消融开关,默认关闭,
# 不改变现有训练行为;仅 --use_refinement --refine_type sam 时生效
parser.add_argument('--warmstart_refine', action='store_true', default=False,
                    help='消融:二阶段以一阶段预训练权重 warm-start(借鉴 SPT 的 draft decoder '
                         '复制初始化)——预训练权重加载后,把 prompt_encoder/mask_decoder 的'
                         '同名同形状权重复制到 refinement_prompt_encoder/refinement_decoder,'
                         '缓解短 schedule(~280 步)下 7.58M 随机解码器学不满;'
                         '仅全新训练(非 --resume)且预训练 checkpoint 加载成功后执行一次,'
                         'resume 自动跳过不覆盖已训练二阶段权重;α 在一阶段无同名键,不复制、'
                         '仍保持 0 初始化;若同时 --disable_refine_alpha,复制后会重新零初始化'
                         '超网络末层,保持 refined=coarse+0 起步恒等粗掩码')
parser.add_argument('--refine_prompt_mode', type=str, default='true',
                    choices=['true', 'none', 'shuffle'],
                    help='R1 归因消融: 二阶段 mask prompt(p_d)的来源。'
                         "'true'=本病例粗掩码(主方法); "
                         "'none'=p_d 置空,走二阶段 PE 的 no_mask_embed 占位(无提示对照); "
                         "'shuffle'=p_d 取同批另一病例的粗掩码(batch 内错配)。 "
                         '三者第二解码器结构/参数量/前向次数完全一致,'
                         '仅条件信号来源不同。默认 true(不改变现有训练行为)')
parser.add_argument('--refine_use_sparse', action='store_true', default=False,
                    help='消融:二阶段透传一阶段当前轮 points/boxes 的 sparse embedding'
                         '(与 SPT 两阶段共享 sparse 的设计一致),dense 仍只来自粗掩码;'
                         'mask-only 轮次(points=None)一阶段 sparse 本就是 Bx0xC 空张量,'
                         '行为自动对齐,无需特判。训练/验证必须一致:medim_val_dataset.py '
                         '从 checkpoint 保存的 args 自动识别本开关,无需也不要在验证命令手动指定')

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4, help='基础学习率(用于非适配器可训练参数)')
parser.add_argument('--adapter_lr', type=float, default=8e-4,
                    help='适配器学习率(借鉴 LoRA+ 思想,给适配器较高 lr)')
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--adapter_wd', type=float, default=None,
                    help='适配器/精修解码器等随机初始化新模块的专用权重衰减;'
                         'None(=不传)时跟随全局 --weight_decay。经验:从零初始化的 '
                         'adapter/refine 被强 wd(0.1)持续压小会削弱注入幅度,'
                         '建议解耦设 0.01~0(全微调/PEFT 保持一致口径)')
parser.add_argument('--port', type=int, default=12361)
parser.add_argument('--seed', type=int, default=2023,
                    help='随机种子(消融协议要求 3 种子 42/123/2024;默认 2023 '
                         '与历史训练完全一致,不传则不改变旧行为)')

args = parser.parse_args()

# 多扫描分支合并口径:新训练统一按方向项数取平均(三模式尺度可比)。
# 固化进 args 随 checkpoint 保存,验证侧据此还原;旧 ckpt 无此字段时
# medim_val_dataset 自动回退 'sum'(2026-09-25 前的裸求和历史行为)。
args.scan_merge = 'mean'

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def remap_pretrained_keys(state_dict, model):
    """将裸 SAM-Med3D 预训练权重重映射到适配器包装后的参数路径。

    ImageEncoderViT3DWithAdapter 把每个 Block3D 包进 MambaAdapterBlock,
    键路径由 ``image_encoder.blocks.<i>.<rest>`` 变为
    ``image_encoder.blocks.<i>.frozen_block.<rest>``。直接 strict=False
    加载会让 12 个编码器 block 的预训练权重全部静默失配(保持随机冻结),
    故加载前必须重映射。训练后保存的 checkpoint 已含 frozen_block 前缀,
    可直接匹配,不会触发本重映射(resume 安全)。

    Returns:
        (remapped_state_dict, num_remapped)
    """
    target_keys = model.state_dict()
    remapped = {}
    num_remapped = 0

    # 历次结构演进中,旧 checkpoint 里已删除的编码器门控/缩放键统一丢弃
    # (不进 unexpected 噪声):逐层 α、阶段共享 α、适配器 gamma、
    # AxisGate3D 三轴门控权重。这些模块已被逐层可学习 gate(all_gate)
    # 取代,旧 checkpoint 无法迁移,必须重新训练。
    def _is_legacy_gate_key(key: str) -> bool:
        return (
            key in ('image_encoder.alpha_conv', 'image_encoder.alpha_mamba')
            or (key.startswith('image_encoder.blocks.') and key.endswith('.alpha'))
            or ('.adapter.gamma' in key)
            or ('.adapter.axis_gate.' in key)
        )

    num_legacy_dropped = 0
    for key, value in state_dict.items():
        if _is_legacy_gate_key(key):
            num_legacy_dropped += 1
            continue
        if key in target_keys:
            remapped[key] = value
        elif key.startswith('image_encoder.blocks.'):
            parts = key.split('.')
            # ['image_encoder', 'blocks', '<i>', <rest>...]
            new_key = '.'.join(parts[:3] + ['frozen_block'] + parts[3:])
            if new_key in target_keys:
                remapped[new_key] = value
                num_remapped += 1
            else:
                remapped[key] = value
        else:
            remapped[key] = value
    if num_legacy_dropped:
        print(f"[Checkpoint] 已丢弃 {num_legacy_dropped} 个旧版编码器门控键"
              f"(逐层/阶段 α、gamma、axis_gate;新结构为普通残差相加,需重训)")
    return remapped, num_remapped


def build_model(args):
    """构建 PEFT 模型(frozen backbone + Mamba 适配器普通残差 + 二阶段精修 α)。

    使用 sam_model_registry3D 中的 mamba_adapter 变体,
    先构建随机权重模型(checkpoint=None),再由 init_checkpoint 加载预训练权重。
    """
    # 构建适配器模型时先不加载 checkpoint(由 init_checkpoint 统一加载)
    # 通过 build_sam3D_with_mamba_adapter 的参数透传适配器超参
    from segment_anything.build_sam3D import build_sam3D_with_mamba_adapter
    sam_model = build_sam3D_with_mamba_adapter(
        model_type='vit_b_ori',
        checkpoint=None,
        mamba_start=args.mamba_start,
        ratio=args.ratio,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        scan_mode=args.scan_mode,
        scan_merge=args.scan_merge,
        use_adapter=getattr(args, 'use_adapter', True),
        freeze_prompt_encoder=args.freeze_prompt_encoder,
        freeze_mask_decoder=args.freeze_mask_decoder,
        use_refinement=args.use_refinement,
        low_res_size=args.low_res_size,
        use_pyramid=args.use_pyramid,
        use_refine_alpha=not args.disable_refine_alpha,
        refine_type=args.refine_type,
    ).to(device)
    # 记录二阶段 sparse 透传开关,供前向分支与日志读取(纯前向路由,
    # 不引入任何新参数,checkpoint 键与旧模型完全一致)
    sam_model.refine_use_sparse = bool(getattr(args, 'refine_use_sparse', False))
    # R1 归因消融: 二阶段 mask prompt 来源(纯前向路由,同样不引入新参数,
    # 因此 R1-a/b/c 三个变体的权重键与参数量完全一致,可直接复用同一套
    # 优化器分组/checkpoint/warmstart 逻辑)
    sam_model.refine_prompt_mode = args.refine_prompt_mode
    if args.refine_prompt_mode != 'true':
        print(f"[R1] refine_prompt_mode={args.refine_prompt_mode}: "
              + ("二阶段 mask prompt 置空(no_mask_embed 占位)" if args.refine_prompt_mode == 'none'
                 else "二阶段 mask prompt 取同批错配病例(batch 内 shuffle)"))

    # SS3D 扫描消融结构打印(核数/方向项数/合并口径)
    # --no-use_adapter(A1 w/o 适配器)时无 SS3D 模块,改打印基线模式标识
    if not getattr(args, 'use_adapter', True):
        print("[A1] use_adapter=False: 无旁路适配器/无 all_gate,编码器纯冻结,"
              "仅训练 prompt_encoder + 解码器头部(w/o Adapter 消融)")
    else:
        _scan_cores = {'bi_separate': 6, 'bi_shared': 3, 'uni': 3}[args.scan_mode]
        _scan_terms = 3 if args.scan_mode == 'uni' else 6
        _scan_desc = {'bi_separate': '前/后向独立核',
                      'bi_shared': '前/后向共享同一核(翻转复用)',
                      'uni': '仅前向单向(反向盲区对照)'}[args.scan_mode]
        print(f"[SS3D] scan_mode={args.scan_mode}({_scan_cores} 个 SSM 核/"
              f"{_scan_terms} 个合并方向项,{_scan_desc}), merge={args.scan_merge}")

    # 全微调实验:统一解冻全部参数(覆盖编码器构建时对 frozen_block/
    # patch_embed/pos_embed/neck 的冻结,以及两个 freeze_* flag)。
    # 必须在 DDP 包装前完成。
    if getattr(args, 'full_finetune', False):
        if args.freeze_prompt_encoder or args.freeze_mask_decoder:
            print("[FullFT][WARNING] --full_finetune 与 --freeze_prompt_encoder/"
                  "--freeze_mask_decoder 互斥,后者被忽略,全部参数解冻")
        n_unfrozen = 0
        for p in sam_model.parameters():
            if not p.requires_grad:
                n_unfrozen += 1
            p.requires_grad = True
        total = sum(p.numel() for p in sam_model.parameters())
        print(f"[FullFT] 全微调模式:解冻 {n_unfrozen} 个原冻结参数张量,"
              f"全部 {total/1e6:.1f}M 参数参与训练(骨干 lr={args.lr},"
              f"随机旁路保持各自高 lr;gate 仍从 0 起步苏醒)")

    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def warmstart_refinement_from_stage1(model):
    """SPT 式二阶段 warm-start(--warmstart_refine,仅 refine_type='sam')。

    预训练权重加载完成后,把一阶段 prompt_encoder / mask_decoder 的参数按
    同名同形状复制到二阶段模块,使二阶段从「预训练 decoder 拿到粗掩码再预测
    一次」起步,而非 7.58M 完全随机——短 schedule(~280 optimizer step)下
    随机初始化的二阶段很难学满,warm-start 直接缩短这一差距。

    约束:
      - 只复制同名且同形状的键;refinement_decoder.alpha 在一阶段无对应键,
        自然跳过(α 保持 0 初始化,起步恒等 coarse 的苏醒动力学不变);
      - Module.load_state_dict 为参数原地 data 拷贝,不替换 Parameter 对象,
        此前已构建的优化器(param 引用与空 state)不受影响;
      - use_refine_alpha=False(α≡1 单位残差)时,复制会覆盖构造时对超网络
        末层的零初始化,需重新置零,保证 refine_masks 起步≡0、
        refined=coarse+0 仍恒等粗掩码(防止初始 Dice 塌掉);
      - 只允许在全新预训练初始化后调用;resume 训练 checkpoint 禁止调用,
        否则会覆盖已训练的二阶段权重。
    """
    if getattr(model, 'refine_type', None) != 'sam':
        print("[WarmStart] 二阶段非 sam 同构版(light),warm-start 不适用,跳过")
        return

    pair_specs = [
        ('prompt_encoder', 'refinement_prompt_encoder'),
        ('mask_decoder', 'refinement_decoder'),
    ]
    total_copied, total_skipped = 0, 0
    for src_name, dst_name in pair_specs:
        src_module = getattr(model, src_name)
        dst_module = getattr(model, dst_name)
        src_sd = src_module.state_dict()
        dst_sd = dst_module.state_dict()
        copy_sd = {}
        for key, dst_value in dst_sd.items():
            if key in src_sd and src_sd[key].shape == dst_value.shape:
                copy_sd[key] = src_sd[key].detach().clone().to(dst_value.device)
                total_copied += 1
            else:
                total_skipped += 1
        # strict=False:dst 独有键(如 refinement_decoder.alpha)保持原初始化
        dst_module.load_state_dict(copy_sd, strict=False)
        print(f"[WarmStart] {src_name} → {dst_name}: "
              f"复制 {len(copy_sd)}/{len(dst_sd)} 个同名同形状参数")

    # --disable_refine_alpha:复制来的预训练超网络末层会破坏构造时的零初始化,
    # 重新置零 → refine_masks 起步≡0,保住 refined=coarse+0 的恒等起点
    rd = model.refinement_decoder
    if getattr(rd, 'alpha', None) is None:
        for hyper_mlp in rd.output_hypernetworks_mlps:
            torch.nn.init.zeros_(hyper_mlp.layers[-1].weight)
            torch.nn.init.zeros_(hyper_mlp.layers[-1].bias)
        print("[WarmStart] use_refine_alpha=False: 已将掩码超网络末层重新零初始化,"
              "保持 refined=coarse+0 起步恒等粗掩码")

    print(f"[WarmStart] 完成:共复制 {total_copied} 个参数,跳过 {total_skipped} 个"
          f"(跳过项为 α 等二阶段独有参数,保持原初始化)")


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='label',
                          target_shape=(args.img_size, args.img_size,
                                        args.img_size)),  # crop only object region
            tio.RandomFlip(axes=(0, 1, 2)),
        ]),
        threshold=1000)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # train_dataloader = tio.SubjectsLoader(
    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader


class BaseTrainer:

    def __init__(self, model, dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.coarse_losses = []
        self.refine_losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if (args.resume):
            self.init_checkpoint(
                join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

        # 打印 PEFT 参数统计(仅主进程)
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            sam_model = self.model.module if self.args.multi_gpu else self.model
            print("\n" + "=" * 60)
            print("PEFT 模型参数统计")
            print("=" * 60)
            print_param_stats(sam_model)
            print("=" * 60 + "\n")

    def set_loss_fn(self):
        # 仅使用 Dice + 交叉熵组合损失(MONAI DiceCELoss 内部即两者加权和)
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def set_optimizer(self):
        """PEFT 优化器:只收集 requires_grad=True 的参数,按模块分组。

        分组策略(借鉴 LoRA+ 思想):
          - 适配器权重(名字含 'adapter'):adapter_lr;
          - 编码器逐层门控 all_gate(12 个标量,初值 0):gate_lr,wd=0;
          - 精修融合 α + light 版 blend_logit:alpha_lr,wd=0;
          - 其它可训练参数(如解冻的 prompt/decoder):使用 lr * 0.1。
        两类标量门控都必须高 lr、零权重衰减:初值为 0 时若 lr 过小会
        "苏醒过慢"(25epoch 仍贴近 0),且 wd 会持续把它们拉回 0。
        冻结参数不进入优化器,避免无谓的优化器状态显存开销。

        --full_finetune 时采用 discriminative lr(预训练 vs 随机新模块):
          - 预训练部分(frozen_block/patch_embed/pos_embed/neck +
            prompt_encoder + mask_decoder)合并为 backbone 组,统一 lr;
          - 随机初始化部分(adapter/refine/gate/α)保持上述各自高 lr。
        """
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        adapter_params = []
        gate_params = []
        prompt_params = []
        decoder_params = []
        refine_params = []
        alpha_params = []
        other_trainable_params = []

        for name, p in sam_model.named_parameters():
            if not p.requires_grad:
                continue
            # all_gate 已废弃(前向改为无门控残差相加):保留 Parameter 以兼容
            # 旧 checkpoint 键名,但不加入任何优化器组(否则无梯度会报错)
            if 'all_gate' in name:
                continue
            # 精修融合 α + light 版 blend_logit:高 lr 组、无权重衰减
            elif ('refinement_decoder' in name and 'alpha' in name) or \
               ('refinement_prompt_encoder' in name and 'blend_logit' in name):
                alpha_params.append(p)
            # 精修解码器 + 二阶段提示编码器(卷积参数):refine_lr 组
            elif 'refinement_decoder' in name or 'refinement_prompt_encoder' in name:
                refine_params.append(p)
            # 编码器适配器权重:名字含 'adapter'(SS3D/卷积适配器全部权重;
            # 逐层门控在 all_gate 中单独成组,已在上方分流)
            elif 'adapter' in name:
                adapter_params.append(p)
            # prompt_encoder 单独一组,使用极小的 prompt_lr
            elif 'prompt_encoder' in name:
                prompt_params.append(p)
            # 粗掩码解码器 mask_decoder 单独一组,使用 decoder_lr
            elif 'mask_decoder' in name:
                decoder_params.append(p)
            else:
                other_trainable_params.append(p)

        param_groups = []
        # 适配器权重组:专用 weight_decay(--adapter_wd,None 则跟随全局)
        if adapter_params:
            adapter_group = {
                'params': adapter_params,
                'lr': self.args.adapter_lr,
            }
            if self.args.adapter_wd is not None:
                adapter_group['weight_decay'] = self.args.adapter_wd
            param_groups.append(adapter_group)
        if gate_params:
            param_groups.append({
                'params': gate_params,
                'lr': self.args.gate_lr,
                'weight_decay': 0.0,  # 标量门控不应被权重衰减持续拉回 0
            })
        if alpha_params:
            param_groups.append({
                'params': alpha_params,
                'lr': self.args.alpha_lr,
                'weight_decay': 0.0,  # 标量不应被权重衰减持续拉回 0
            })
        if refine_params:
            refine_group = {
                'params': refine_params,
                'lr': self.args.refine_lr,
            }
            if self.args.adapter_wd is not None:
                refine_group['weight_decay'] = self.args.adapter_wd
            param_groups.append(refine_group)
        if getattr(self.args, 'full_finetune', False):
            # 全微调:全部预训练权重(编码器 frozen_block/patch_embed/pos_embed/
            # neck 落入 other + prompt_encoder + mask_decoder)合并为一个
            # backbone 组,统一小 lr;随机新模块仍在上面的高 lr 组
            backbone_params = other_trainable_params + prompt_params + decoder_params
            if backbone_params:
                param_groups.append({
                    'params': backbone_params,
                    'lr': self.args.lr,
                })
        else:
            if decoder_params:
                param_groups.append({
                    'params': decoder_params,
                    'lr': self.args.decoder_lr,
                })
            if prompt_params:
                param_groups.append({
                    'params': prompt_params,
                    'lr': self.args.prompt_lr,
                })
            if other_trainable_params:
                param_groups.append({
                    'params': other_trainable_params,
                    'lr': self.args.lr * 0.1,
                })

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay)

        # 打印参数分组信息
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            adapter_cnt = sum(p.numel() for p in adapter_params)
            gate_cnt = sum(p.numel() for p in gate_params)
            prompt_cnt = sum(p.numel() for p in prompt_params)
            decoder_cnt = sum(p.numel() for p in decoder_params)
            refine_cnt = sum(p.numel() for p in refine_params)
            alpha_cnt = sum(p.numel() for p in alpha_params)
            other_cnt = sum(p.numel() for p in other_trainable_params)
            print(f"\n[Optimizer] 适配器权重参数组: {adapter_cnt:,} (lr={self.args.adapter_lr})")
            if gate_params:
                print(f"[Optimizer] 编码器逐层 gate 参数组: {gate_cnt} 个标量 "
                      f"(image_encoder.all_gate,初值 0,lr={self.args.gate_lr},"
                      f"weight_decay=0)")
            if alpha_params:
                print(f"[Optimizer] 精修融合 α 参数组: {alpha_cnt} 个标量 "
                      f"(lr={self.args.alpha_lr}, weight_decay=0)")
            if refine_params:
                print(f"[Optimizer] 精修解码器参数组: {refine_cnt:,} (lr={self.args.refine_lr})")
            if getattr(self.args, 'full_finetune', False):
                backbone_cnt = other_cnt + prompt_cnt + decoder_cnt
                print(f"[Optimizer] 预训练 backbone 组(编码器+PE+粗解码器): "
                      f"{backbone_cnt:,} (统一 lr={self.args.lr}, "
                      f"其中编码器 {other_cnt:,}/PE {prompt_cnt:,}/粗解码器 {decoder_cnt:,})")
            else:
                if decoder_params:
                    print(f"[Optimizer] 粗解码器参数组: {decoder_cnt:,} (lr={self.args.decoder_lr})")
                if prompt_params:
                    print(f"[Optimizer] prompt_encoder 参数组: {prompt_cnt:,} (lr={self.args.prompt_lr})")
                if other_trainable_params:
                    print(f"[Optimizer] 其它可训练参数组: {other_cnt:,} (lr={self.args.lr * 0.1})")
            n_frozen = sum(p.numel() for p in sam_model.parameters() if not p.requires_grad)
            print(f"[Optimizer] 冻结参数: {n_frozen:,} (不分配优化器状态)\n")

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)

        if last_ckpt:
            # PEFT: 预训练权重不含适配器/精修参数,必须 strict=False;
            # 但裸预训练键 image_encoder.blocks.<i>.* 需重映射为
            # image_encoder.blocks.<i>.frozen_block.*,否则 12 个编码器
            # Block3D 的预训练权重会全部静默失配(保持随机冻结)
            target_model = self.model.module if self.args.multi_gpu else self.model
            state_dict, num_remapped = remap_pretrained_keys(
                last_ckpt['model_state_dict'], target_model)
            missing, unexpected = target_model.load_state_dict(
                state_dict, strict=False)
            # 关键诊断:重映射后冻结骨干的 block 权重必须全部命中
            frozen_missing = [k for k in missing if '.frozen_block.' in k]
            print(f"[Checkpoint] 预训练编码器 block 键重映射: {num_remapped} 个; "
                  f"frozen_block 未加载键: {len(frozen_missing)} 个 "
                  f"(必须为 0); unexpected: {len(unexpected)} 个")
            if frozen_missing:
                print(f"[Checkpoint][WARNING] 以下冻结骨干权重未加载(前5个): "
                      f"{frozen_missing[:5]}")
            if not self.args.resume:
                self.start_epoch = 0
                # SPT 式二阶段 warm-start:必须在预训练权重载入后执行
                # (此时 mask_decoder 才是 turbo 预训练值);resume 走 else 分支,
                # 训练 checkpoint 中的 refinement_* 已训练权重绝不被覆盖
                if getattr(self.args, 'warmstart_refine', False):
                    if self.args.use_refinement and \
                            hasattr(target_model, 'refinement_decoder'):
                        warmstart_refinement_from_stage1(target_model)
                    else:
                        print("[WarmStart][WARNING] --warmstart_refine 需配合 "
                              "--use_refinement 使用,当前模型无二阶段模块,跳过")
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                # .get 兼容不含双阶段记录的旧 checkpoint
                self.coarse_losses = last_ckpt.get('coarse_losses', [])
                self.refine_losses = last_ckpt.get('refine_losses', [])
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
                # sparse 透传是纯前向开关(权重键无差异),resume 时以 checkpoint
                # 记录的训练参数为准自动恢复,防止命令行漏带 --refine_use_sparse
                # 导致续训前后半程前向口径不一致
                ckpt_args = last_ckpt.get('args', None)
                ckpt_sparse = bool(getattr(ckpt_args, 'refine_use_sparse', False))
                if ckpt_sparse != bool(getattr(self.args, 'refine_use_sparse', False)):
                    print(f"[Checkpoint][WARNING] 命令行 refine_use_sparse="
                          f"{self.args.refine_use_sparse} 与 checkpoint 记录 "
                          f"{ckpt_sparse} 不一致,已自动以 checkpoint 为准恢复")
                self.args.refine_use_sparse = ckpt_sparse
                target_model.refine_use_sparse = ckpt_sparse
                # R1 变体同样是纯前向路由(权重键无差异),resume 时必须以
                # checkpoint 记录为准,否则续训会把 R1-a/b 悄悄变回真提示
                ckpt_prompt_mode = getattr(ckpt_args, 'refine_prompt_mode', 'true')
                if ckpt_prompt_mode != getattr(self.args, 'refine_prompt_mode', 'true'):
                    print(f"[Checkpoint][WARNING] 命令行 refine_prompt_mode="
                          f"{self.args.refine_prompt_mode} 与 checkpoint 记录 "
                          f"{ckpt_prompt_mode} 不一致,已自动以 checkpoint 为准恢复")
                self.args.refine_prompt_mode = ckpt_prompt_mode
                target_model.refine_prompt_mode = ckpt_prompt_mode
                # scan_mode 决定 SS3D 模块拓扑(6 核 vs 3 核,state_dict 键名
                # 不相交),不一致时 strict=False 会让一整组 SSM 核静默随机,
                # 优化器状态也按参数位置加载会错位,直接中止以防垃圾续训
                ckpt_scan_mode = getattr(ckpt_args, 'scan_mode', 'bi_separate')
                if ckpt_scan_mode != self.args.scan_mode:
                    raise RuntimeError(
                        f"--resume 失败:命令行 --scan_mode={self.args.scan_mode} "
                        f"与 checkpoint 记录 scan_mode={ckpt_scan_mode} 拓扑不同"
                        f"(SSM 核数/参数键名不相交)。续训必须去掉 --scan_mode "
                        f"(或传与该 checkpoint 一致的值);扫描消融请用新 "
                        f"task_name 从头训练。")
                # scan_merge 无参数差异但改变前向口径;旧 ckpt(无记录=sum)
                # 在新代码下续训将统一变成 mean,给明确提示
                ckpt_scan_merge = getattr(ckpt_args, 'scan_merge', 'sum')
                if ckpt_scan_merge != getattr(self.args, 'scan_merge', 'mean'):
                    print(f"[Checkpoint][WARNING] checkpoint 训练于 "
                          f"scan_merge={ckpt_scan_merge}(旧裸求和口径),当前代码"
                          f"按 {self.args.scan_merge} 续训,多分支合并尺度改变;"
                          f"如需严格复现旧实验请用旧代码,否则建议从头重训")
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch}) [strict=False]")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "coarse_losses": self.coarse_losses,
                "refine_losses": self.refine_losses,
                "dices": self.dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                "used_datas": img_datas,
            }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    @staticmethod
    def _r1_shuffle_batch(x: torch.Tensor) -> torch.Tensor:
        """R1-b 错配提示: 返回同批另一病例的粗掩码(无固定点错排)。

        形状/数值分布与原张量完全一致,仅病例内容被替换。要求每个病例都
        拿到别人的掩码(而不是可能恰好拿到自己),故用随机错排:先 randperm,
        再滚动消除固定点;batch=1 时无法错配,退回原张量(此时 R1-b 与 R1-c
        等价,结果不可作为错配对照,需在日志中注明)。
        """
        b = x.shape[0]
        if b < 2:
            return x
        idx = torch.randperm(b, device=x.device)
        arange = torch.arange(b, device=x.device)
        for _ in range(b):
            if not bool((idx == arange).any()):
                break
            idx = torch.roll(idx, 1)
        return x[idx]

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        # 第一阶段:冻结的 SAM-Med3D 粗掩码解码器 → 粗定位
        coarse_low_res, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 384, 8, 8, 8)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 第二阶段精修(创新点二),按 refine_type 走不同调用接口
        if self.args.use_refinement and hasattr(sam_model, 'refinement_decoder'):
            # stop-gradient 防共适应:二阶段只消费粗掩码的数值,精修损失不回灌粗解码器
            # (同时切断 α 凸组合 (1-α) 通路与二阶段 PE dense prompt 通路)
            coarse_for_refine = (coarse_low_res.detach()
                                 if self.args.detach_coarse else coarse_low_res)
            if getattr(sam_model, 'refine_type', 'light') == 'sam':
                # R1 归因消融: 只替换二阶段 mask prompt(p_d)的来源,
                # 解码器结构/参数/前向次数,以及 α 残差出口的 coarse_logits 均不变
                mode = getattr(sam_model, 'refine_prompt_mode', 'true')
                if mode == 'none':
                    prompt_src = None          # no_mask_embed 无条件占位
                elif mode == 'shuffle':
                    prompt_src = self._r1_shuffle_batch(coarse_for_refine)
                else:
                    prompt_src = coarse_for_refine
                # SAM 同构版:粗掩码是二阶段 dense prompt 的唯一来源,
                # 二阶段 decoder 复用 8^3 image embedding + 完整双向注意力
                sparse_ref, dense_ref = sam_model.refinement_prompt_encoder(
                    prompt_src)
                # SPT 式消融(--refine_use_sparse):dense 仍只来自粗掩码,
                # sparse 透传一阶段当前轮 points/boxes(两阶段共享同一交互信息,
                # 粗定位出错时二阶段可凭原始点击自救)。mask-only 轮 points=None,
                # 一阶段 sparse_embeddings 为 Bx0xC 空张量,与此处默认空 sparse
                # 完全等价,无需特判。
                if getattr(self.args, 'refine_use_sparse', False):
                    sparse_ref = sparse_embeddings
                low_res_masks, _ = sam_model.refinement_decoder(
                    image_embeddings=image_embedding.to(device),
                    image_pe=sam_model.refinement_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_ref,
                    dense_prompt_embeddings=dense_ref,
                    coarse_logits=coarse_for_refine,
                    multimask_output=False,
                )
            else:
                # 旧轻量版:二阶段提示编码器以粗掩码为主、辅以编码器特征
                dense_prompt, coarse_prob = sam_model.refinement_prompt_encoder(
                    coarse_for_refine,
                    image_embedding.to(device),
                    coarse_for_refine.shape[-3:],
                )
                # 精修解码器:消费 dense_prompt,残差渐进融合
                # (金字塔模式额外需要 8^3 image_embedding;平铺模式忽略该参数)
                low_res_masks = sam_model.refinement_decoder(
                    dense_prompt, coarse_for_refine,
                    image_embedding=image_embedding.to(device))
        else:
            low_res_masks = coarse_low_res

        prev_masks = F.interpolate(low_res_masks,
                                   size=gt3D.shape[-3:],
                                   mode='trilinear',
                                   align_corners=False)
        coarse_prev = F.interpolate(coarse_low_res,
                                    size=gt3D.shape[-3:],
                                    mode='trilinear',
                                    align_corners=False)
        return low_res_masks, prev_masks, coarse_prev

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)  # 点击坐标
        labels_co = torch.cat(batch_labels, dim=0).to(device)  # 点击标签

        self.click_points.append(points_co)
        self.click_labels.append(labels_co)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = labels_co
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        total_loss_tensor = 0          # 张量累加:用于反传(所有 click 损失之和)
        return_coarse_loss = 0         # 数值累加:粗阶段 raw loss,仅用于曲线
        return_refine_loss = 0         # 数值累加:精修阶段 raw loss,仅用于曲线
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.img_size // 4, args.img_size // 4,
                                            args.img_size // 4))
        random_insert = np.random.randint(2, 9)
        use_refine = self.args.use_refinement
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks, coarse_prev = self.batch_forward(
                    sam_model, image_embedding, gt3D, low_res_masks, points=None)
            else:
                low_res_masks, prev_masks, coarse_prev = self.batch_forward(
                    sam_model, image_embedding, gt3D, low_res_masks,
                    points=[points_input, labels_input])
            # 损失:Dice + 交叉熵(DiceCELoss)
            gt = gt3D.float()
            if use_refine:
                # 两阶段深监督:精修输出(权重 1.0) + 粗解码器自身 GT 损失
                # (权重 --coarse_loss_weight 默认 1.0)。
                # 防共适应靠 --detach_coarse(切断精修→粗的梯度通路),
                # 而非压低粗损失权重:两阶段各自独立受 GT 完整监督。
                # 分别记录两阶段 raw loss 供绘制双阶段曲线
                refine_raw = self.seg_loss(prev_masks, gt)
                coarse_raw = self.seg_loss(coarse_prev, gt)
                loss = refine_raw + self.args.coarse_loss_weight * coarse_raw
                return_refine_loss += refine_raw.item()
                return_coarse_loss += coarse_raw.item()
            else:
                loss = self.seg_loss(prev_masks, gt)
                return_refine_loss += loss.item()
                return_coarse_loss += loss.item()
            total_loss_tensor = total_loss_tensor + loss
            # 轮间接界 stop-gradient:本轮输出作为下一轮的 mask prompt 反馈,
            # 推理时它是不可导的"给定观测物"。不 detach 则下一轮粗损失会沿
            # 反馈链回传、跨轮次塑造上一轮精修模块(残余共适应通路)。
            # loss 已加入 total_loss_tensor,detach 重绑定不影响本轮反传图。
            if use_refine and self.args.detach_coarse:
                low_res_masks = low_res_masks.detach()
        return prev_masks, total_loss_tensor, return_coarse_loss, return_refine_loss

    def get_dice_score(self, prev_masks, gt3D):

        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_coarse_loss = 0
        epoch_refine_loss = 0
        epoch_iou = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        for step, data3D in enumerate(tbar):
            try:
                image3D, gt3D = data3D["image"], data3D["label"]
            except Exception as e:
                print(f"Error processing batch at step {step}: {e}")
            # import pdb; pdb.set_trace()
            my_context = self.model.no_sync if self.args.rank != - \
                1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                with torch.amp.autocast("cuda"):
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss, batch_coarse_loss, batch_refine_loss = self.interaction(
                        sam_model,
                        image_embedding,
                        gt3D,
                        num_clicks=11)

                epoch_loss += loss.item()
                epoch_coarse_loss += batch_coarse_loss
                epoch_refine_loss += batch_refine_loss
                epoch_dice += self.get_dice_score(prev_masks, gt3D)
                cur_loss = loss.item()

                loss /= self.args.accumulation_steps

                self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                # 可选梯度裁剪(全微调建议 --grad_clip 1.0):必须先
                # unscale_ 再裁剪,否则裁的是含 loss scale 的梯度
                clip = float(getattr(self.args, 'grad_clip', 0.0) or 0.0)
                if clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_params = [p for g in self.optimizer.param_groups
                                   for p in g['params'] if p.grad is not None]
                    grad_norm = torch.nn.utils.clip_grad_norm_(grad_params, clip)
                    if step % (self.args.accumulation_steps * 50) == 0:
                        print(f"  [GradClip] step={step} global_norm={float(grad_norm):.3f}")
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(epoch,
                                                 sam_model.state_dict(),
                                                 describe=f'{epoch}_step_dice:{print_dice}_best')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss

        epoch_loss /= step + 1
        epoch_coarse_loss /= step + 1
        epoch_refine_loss /= step + 1
        epoch_dice /= step + 1

        return (epoch_loss, epoch_coarse_loss, epoch_refine_loss,
                epoch_iou, epoch_dice, pred_list)

    def eval_epoch(self, epoch, num_clicks):
        return 0

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def plot_two_stage_loss(self, coarse_data, refine_data, total_data,
                            save_name='Two_Stage_Loss'):
        """在同一张图中绘制粗解码器、精修解码器两个阶段的训练损失曲线。

        Args:
            coarse_data: 每个 epoch 的粗阶段 raw DiceCE 损失
            refine_data: 每个 epoch 的精修阶段 raw DiceCE 损失
            total_data:  每个 epoch 的实际优化目标(refine + 0.5*coarse),浅色虚线参考
        """
        plt.figure(figsize=(8, 5))
        plt.plot(coarse_data, color='tab:blue', marker='o', markersize=3,
                 label='Coarse stage (mask decoder)')
        plt.plot(refine_data, color='tab:orange', marker='s', markersize=3,
                 label='Refine stage (refinement decoder)')
        plt.plot(total_data, color='gray', linestyle='--', linewidth=1,
                 label='Total optimized loss')
        plt.title('Two-Stage DiceCE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'), dpi=150)
        plt.close()

    def train(self):
        self.scaler = torch.amp.GradScaler("cuda")
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            (epoch_loss, epoch_coarse_loss, epoch_refine_loss,
             epoch_iou, epoch_dice, pred_list) = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.coarse_losses.append(epoch_coarse_loss)
                self.refine_losses.append(epoch_refine_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                if self.args.use_refinement:
                    print(f'EPOCH: {epoch}, Coarse loss: {epoch_coarse_loss}, '
                          f'Refine loss: {epoch_refine_loss}')
                    # 打印 α(两版通用)和 blend(仅 light 版)当前值,
                    # 验证残差标量与融合权重演化
                    sam = self.model.module if self.args.multi_gpu else self.model
                    rd = sam.refinement_decoder
                    rtype = getattr(sam, 'refine_type', 'light')
                    sparse_tag = 'on' if getattr(sam, 'refine_use_sparse', False) else 'off'
                    if rd.alpha is not None:
                        msg = (f'  [Refine] type={rtype}  alpha={rd.alpha.item():.6f}  '
                               f'sparse={sparse_tag}')
                        if rtype == 'light':
                            msg += f'  blend={torch.sigmoid(sam.refinement_prompt_encoder.blend_logit).item():.4f}'
                        print(msg)
                    else:
                        msg = (f'  [Refine] type={rtype}  '
                               f'alpha=disabled(--disable_refine_alpha)  sparse={sparse_tag}')
                        if rtype == 'light':
                            msg += f'  blend={torch.sigmoid(sam.refinement_prompt_encoder.blend_logit).item():.4f}'
                        print(msg)
                # 编码器逐层适配器 gate 的苏醒情况(12 个标量,浅→深):
                # 长期贴近 0 说明该层适配器基本未参与;正常训练各层应陆续长大
                # (--no-use_adapter 基线模式 all_gate=None,跳过打印)
                sam = self.model.module if self.args.multi_gpu else self.model
                enc = sam.image_encoder
                if enc.all_gate is not None:
                    with torch.no_grad():
                        gate_vals = [g.item() for g in enc.all_gate]
                    gate_str = '[' + ', '.join(f'{v:+.4f}' for v in gate_vals) + ']'
                    print(f'  [Encoder] gate[0..{len(gate_vals) - 1}]'
                          f'(卷积<{enc.mamba_start}/Mamba>={enc.mamba_start}): {gate_str}')
                else:
                    print('  [Encoder] gate: N/A(--no-use_adapter 无适配器基线,无 all_gate)')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, '
                            f'coarse_loss: {epoch_coarse_loss}, '
                            f'refine_loss: {epoch_refine_loss}, dice: {epoch_dice}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                # save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe='latest')

                # save train loss best checkpoint
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, state_dict, describe='loss_best')

                # save train dice best checkpoint
                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                if self.args.use_refinement:
                    # 两阶段训练:一张图内同时呈现粗/精修两条损失曲线
                    self.plot_two_stage_loss(
                        self.coarse_losses, self.refine_losses, self.losses,
                        save_name='Two_Stage_Loss')
                else:
                    self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Coarse stage loss: {self.coarse_losses}')
        logger.info(f'Refine stage loss: {self.refine_losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/random.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        # 种子由 --seed 控制(默认 2023,与历史训练一致;消融跑 42/123/2024)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(args.seed + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO if rank in [-1, 0] else logging.WARN,
                        filemode='w',
                        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://127.0.0.1:{args.port}',
                            world_size=world_size,
                            rank=rank)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
