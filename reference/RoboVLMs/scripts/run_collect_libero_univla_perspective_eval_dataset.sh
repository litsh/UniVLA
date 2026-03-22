#!/bin/bash
set -euo pipefail

# Collect a fixed offline evaluation set by rolling out an older joint image+action model.

ckpt_dir=${ckpt_dir:-"/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/UNIVLA_LIBERO_PERSPECTIVE_gripper_BS192_12k/checkpoint-8000"}
CACHE_ROOT=${CACHE_ROOT:-"/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/libero/offline_perspective_evalset"}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-"libero_goal"}
PERSPECTIVE_OBS_KEY=${PERSPECTIVE_OBS_KEY:-"robot0_eye_in_hand_image"}
MASTER_PORT=${MASTER_PORT:-29617}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-10}
CAMERA_RESOLUTION=${CAMERA_RESOLUTION:-200}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
DATASET_DIRNAME=${DATASET_DIRNAME:-"goal_eval_set"}
VISION_HUB=${VISION_HUB:-/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-VisionTokenizer}
VQ_HUB=${VQ_HUB:-/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-Stage1}

export MUJOCO_GL=osmesa
export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
export NUMBA_DISABLE_JIT=1

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

torchrun \
  --nnodes=1 \
  --nproc_per_node="$GPUS_PER_NODE" \
  --master_port="$MASTER_PORT" \
  eval/libero/evaluate_libero_emu_multi_gpu.py \
  --task_suite_name "$TASK_SUITE_NAME" \
  --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
  --num_steps_wait "$NUM_STEPS_WAIT" \
  --emu_hub "$ckpt_dir" \
  --cache_root "$CACHE_ROOT" \
  --vision_hub "$VISION_HUB" \
  --vq_hub "$VQ_HUB" \
  --with_perspective_gen \
  --cot_max_new_tokens 1024 \
  --no_gripper \
  --perspective_obs_key "$PERSPECTIVE_OBS_KEY" \
  --camera_resolution "$CAMERA_RESOLUTION" \
  --perspective_eval \
  --dump_perspective_eval_dataset \
  --dataset_dirname "$DATASET_DIRNAME"
