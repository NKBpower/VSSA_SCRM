
IMG_DIR="${IMG_DIR:-}"
GT_DIR="${GT_DIR:-}"
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
