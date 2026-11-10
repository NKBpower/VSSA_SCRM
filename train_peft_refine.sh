SCAN_MODE="${SCAN_MODE:-bi_separate}"
SCAN_TAG=""
if [ "$SCAN_MODE" != "bi_separate" ]; then
  SCAN_TAG="_${SCAN_MODE}"
fi

python train_peft.py \
  --batch_size 4 \
  --num_workers 4 \
  --task_name "peft_refine_ws_sparse_12gate_0924${SCAN_TAG}" \
  --checkpoint "/sam_med3d_turbo.pth" \
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
