#!/bin/bash
set -euo pipefail

TEACHER_FORCE_MODE=${TEACHER_FORCE_MODE:-"perspective"}
ckpt_dir=${ckpt_dir:-"/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/logs/UNIVLA_LIBERO_PERSPECTIVE_BS192_12k/checkpoint-8000"}
CACHE_ROOT=${CACHE_ROOT:-"/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/logs/libero/UNIVLA_LIBERO_PERSPECTIVE_BS192_12k/checkpoint-8000/spatial_occluded_teacher_forcing"}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-"libero_spatial_occluded"}
MASTER_PORT=${MASTER_PORT:-29541}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-10}
PERSPECTIVE_OBS_KEY="robot0_eye_in_hand_image"
TEACHER_DATA_PATH=${TEACHER_DATA_PATH:-"/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/meta/libero_all_norm.pkl"}
TEACHER_IMAGE_KEY=${TEACHER_IMAGE_KEY:-"image"}
TEACHER_MIN_H=${TEACHER_MIN_H:-5}
TEACHER_MAX_H=${TEACHER_MAX_H:-10}
CAMERA_RESOLUTION=${CAMERA_RESOLUTION:-200}

# Examples:
#   CoT-VLA with training-set subgoal tokens:
#     bash reference/RoboVLMs/scripts/run_eval_libero_univla_multi_gpu_teacher_force.sh
#   Perspective-VLA with GT simulator images from birdview:
#     TEACHER_FORCE_MODE=perspective \
#     ckpt_dir=/path/to/UNIVLA_LIBERO_PERSPECTIVE_birdview/checkpoint-8000 \
#     CACHE_ROOT=/path/to/eval_cache \
#     TASK_SUITE_NAME=libero_spatial_occluded \
#     PERSPECTIVE_OBS_KEY=birdview_image \
#     bash reference/RoboVLMs/scripts/run_eval_libero_univla_multi_gpu_teacher_force.sh

NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
VISION_HUB=${VISION_HUB:-/inspire/hdd/global_user/chenfangke-253108540237/tsli/huggingface/Emu3-VisionTokenizer}
VQ_HUB=${VQ_HUB:-/inspire/hdd/global_user/chenfangke-253108540237/tsli/huggingface/Emu3-Stage1}

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
# export __GL_VND_DISPATCH_LIBRARY_NAME=nvidia

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

extra_args=(
  --teacher_force_mode "$TEACHER_FORCE_MODE"
  --teacher_data_path "$TEACHER_DATA_PATH"
  --teacher_image_key "$TEACHER_IMAGE_KEY"
  --teacher_min_h "$TEACHER_MIN_H"
  --teacher_max_h "$TEACHER_MAX_H"
  --perspective_obs_key "$PERSPECTIVE_OBS_KEY"
  --camera_resolution "$CAMERA_RESOLUTION"
  --no_gripper
)

if [[ "$TEACHER_FORCE_MODE" == "cot" ]]; then
  extra_args+=(--with_cot --cot_max_new_tokens 1024)
elif [[ "$TEACHER_FORCE_MODE" != "perspective" ]]; then
  echo "Unsupported TEACHER_FORCE_MODE: $TEACHER_FORCE_MODE" >&2
  exit 1
fi

torchrun \
  --nnodes=1 \
  --nproc_per_node="$GPUS_PER_NODE" \
  --master_port="$MASTER_PORT" \
  eval/libero/evaluate_libero_emu_multi_gpu_teacher_force.py \
  --task_suite_name "$TASK_SUITE_NAME" \
  --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
  --num_steps_wait "$NUM_STEPS_WAIT" \
  --emu_hub "$ckpt_dir" \
  --cache_root "$CACHE_ROOT" \
  --vision_hub "$VISION_HUB" \
  --vq_hub "$VQ_HUB" \
  "${extra_args[@]}"
