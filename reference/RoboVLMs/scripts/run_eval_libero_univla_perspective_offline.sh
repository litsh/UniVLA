#!/bin/bash
set -euo pipefail
export TORCH_HOME=/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/torch_cache
# Offline evaluation for a perspective-only checkpoint on a fixed dumped LIBERO rollout set.

ckpt_dir=${ckpt_dir:-"/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/gripper_WEIGHTED_VC=1.0---VS=0.1---AC=0.0---AS=0.0_V3/checkpoint-8000"}
DATASET_ROOT=${DATASET_ROOT:-"/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/libero/offline_perspective_evalset/libero_spatial-2026-03-22_10-28-38/spatial_eval_set"}
OUTPUT_DIR=${OUTPUT_DIR:-"/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/libero/offline_perspective_metrics/checkpoint-8000_V3/spatial"}
VISION_HUB=${VISION_HUB:-/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-VisionTokenizer}
VQ_HUB=${VQ_HUB:-/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-Stage1}
DEVICE=${DEVICE:-"cuda:0"}
QUALITATIVE_FPS=${QUALITATIVE_FPS:-1}

python eval/libero/evaluate_perspective_generation_offline.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --emu_hub "$ckpt_dir" \
  --vision_hub "$VISION_HUB" \
  --vq_hub "$VQ_HUB" \
  --device "$DEVICE" \
  --no_gripper \
  --qualitative_fps "$QUALITATIVE_FPS"
