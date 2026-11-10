# 数据集级批量验证(创新点一 + 二两阶段精修模型)
# 用法: bash val_on_dataset.sh
# 按实际路径修改 IMG_DIR / GT_DIR / OUT_DIR / CKPT
#
# 以下路径/参数均支持环境变量覆盖(默认值即上面原始硬编码值,直接 bash 执行与原行为完全一致):
#   IMG_DIR=... GT_DIR=... OUT_DIR=... CKPT=... \
#   NUM_CLICKS=11 SPACING="1.5 1.5 1.5" USE_REFINEMENT=1 bash val_on_dataset.sh
# 对照实验示例:
#   关闭精修(仅创新点一): USE_REFINEMENT=0 bash val_on_dataset.sh
#   换 checkpoint:       CKPT=work_dir/peft_refine_b2x1/sam_model_latest.pth bash val_on_dataset.sh
#   跨中心(tangdu):      IMG_DIR=/huangdong/code_gj/tangdu_dataset_sam3d/imagesVal \
#                        GT_DIR=/huangdong/code_gj/tangdu_dataset_sam3d/labelsVal \
#                        OUT_DIR=./pred/tangdu bash val_on_dataset.sh

# ---------------- 配置(可用环境变量覆盖,默认值与原脚本一致) ----------------
IMG_DIR="${IMG_DIR:-/huangdong/code_gj/xijing_dataset_sam3d/imagesVal}"
GT_DIR="${GT_DIR:-/huangdong/code_gj/xijing_dataset_sam3d/labelsVal}"
OUT_DIR="${OUT_DIR:-./pred/peft_refine_val}"
CKPT="${CKPT:-work_dir/peft_refine_b2x1/sam_model_dice_best.pth}"
NUM_CLICKS="${NUM_CLICKS:-11}"
SPACING="${SPACING:-1.5 1.5 1.5}"
USE_REFINEMENT="${USE_REFINEMENT:-1}"

# 精修开关: USE_REFINEMENT=1 传 --use_refinement; 0 则仅走粗解码器
REFINE_ARGS=""
if [ "$USE_REFINEMENT" = "1" ]; then
  REFINE_ARGS="--use_refinement"
fi

echo "========================================"
echo " 数据集级批量验证配置"
echo "  IMG_DIR:        $IMG_DIR"
echo "  GT_DIR:         $GT_DIR"
echo "  OUT_DIR:        $OUT_DIR"
echo "  CKPT:           $CKPT"
echo "  NUM_CLICKS:     $NUM_CLICKS"
echo "  SPACING:        $SPACING"
if [ "$USE_REFINEMENT" = "1" ]; then
  echo "  REFINEMENT:     on (coarse+refine 两阶段)"
else
  echo "  REFINEMENT:     off (coarse-only 仅粗解码器)"
fi
echo "========================================"

# ---------------- 第一步: 批量推理(命令与参数同原脚本) ----------------
python medim_val_dataset.py \
  --img_dir "$IMG_DIR" \
  --gt_dir  "$GT_DIR" \
  --out_dir  "$OUT_DIR" \
  --ckpt_path "$CKPT" \
  --num_clicks $NUM_CLICKS \
  --spacing $SPACING \
  $REFINE_ARGS

# ---------------- 第二步: 基础指标 DSC/NSD(原脚本保留) ----------------
python utils/compute_dataset_metrics.py \
  --gt_dir "$GT_DIR" \
  --pred_dir "$OUT_DIR"

# ---------------- 第三步: 全量指标(新增) ----------------
# 8 项指标 DSC/NSD/HD95/SO/SD/VD/Pre/Rec,每例明细落盘 JSON,与 baseline val_xijing.sh 对表口径一致
python utils/compute_metrics_full.py \
  --gt_dir "$GT_DIR" \
  --pred_dir "$OUT_DIR" \
  --classes 1 \
  --output_json "$OUT_DIR/metrics_full.json"
