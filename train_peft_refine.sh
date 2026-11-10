# PEFT 训练(单卡)—— 创新点一 + 创新点二
# Mamba 适配器(特征建模) + 两阶段粗到精解码器(输出净化)
#
# 用法: bash train_peft_refine.sh
#
# 冻结/训练说明:
#   冻结:图像编码器 ViT 骨干(patch embed / pos embed / neck / 全部 Block3D)
#   训练:适配器 + 粗掩码解码器 + 精修解码器 + prompt_encoder
#   (粗解码器默认解冻;如需冻结以严格复现"冻结粗头"两阶段,加 --freeze_mask_decoder)
#
# 门控设置(2026-09-24):
#   编码器侧每层 1 个可学习残差 gate(image_encoder.all_gate,共 12 个标量,
#   初值 0,F=frozen+gate·adapter,高 lr 3e-3 + wd=0 加速逐层苏醒);
#   二阶段精修融合 α(初值 0,默认开启,消融加 --disable_refine_alpha);
#   三轴扫描等权相加(无 AxisGate)。
#
# 训练损失仅 Dice + 交叉熵(DiceCELoss),无其它附加损失项
#
# 扫描方向消融(2026-09-25 新增,环境变量覆盖,三模式其余超参严格一致):
#   SCAN_MODE=bi_shared bash train_peft_refine.sh   # 3 核翻转共享
#   SCAN_MODE=uni bash train_peft_refine.sh         # 3 核仅前向(盲区对照)
#   SCAN_MODE=bi_separate bash train_peft_refine.sh # 6 核(新均值合并口径)
# 非 bi_separate 模式自动给 task_name 加后缀,不覆盖基线目录。

SCAN_MODE="${SCAN_MODE:-bi_separate}"
SCAN_TAG=""
if [ "$SCAN_MODE" != "bi_separate" ]; then
  SCAN_TAG="_${SCAN_MODE}"
fi

python train_peft.py \
  --batch_size 4 \
  --num_workers 4 \
  --task_name "peft_refine_ws_sparse_12gate_0924${SCAN_TAG}" \
  --checkpoint "/huangdong/code_gj/Code_gj/sam_med3d_turbo.pth" \
  --scan_mode "$SCAN_MODE" \
  --lr 1e-5 \
  --adapter_lr 1e-4 \
  --refine_lr 1e-4 \
  --gate_lr 3e-3 \
  --alpha_lr 3e-3 \
  --decoder_lr 1e-5 \
  --prompt_lr 1e-5 \
  --adapter_wd 0.01 \
  --grad_clip 1.0 \
  --use_refinement \
  --refine_type sam \
  --warmstart_refine \
  --refine_use_sparse \
  --num_epochs 25
