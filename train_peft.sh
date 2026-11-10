# PEFT 训练(单卡)—— Mamba 风格轻量适配器
# 用法: bash train_peft.sh
#
# 与 train.sh 的差异:
#   - 使用 train_peft.py(PEFT 训练入口)
#   --checkpoint 预训练权重只加载匹配的骨干参数(strict=False)
#   - 适配器参数随机初始化,训练中逐步更新

python train_peft.py \
  --batch_size 2 \
  --num_workers 4 \
  --task_name "peft_b2x1" \
  --checkpoint "ckpt/sam_med3d_turbo.pth" \
  --lr 8e-5 \
  --adapter_lr 8e-4 \
  --prompt_lr 1e-5 \
  --num_epochs 200
