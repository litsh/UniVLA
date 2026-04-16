#!/bin/bash
# set -euo pipefai

# Perspective-VLA evaluation (predict target-view image tokens, then actions)
ckpt_dir="/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/gripper_WEIGHTED_VC=1.0---VS=0.2---AC=1.0---AS=0.2---Stage1=4000stepsAugmentedData--Stage2=AugmentedData/checkpoint-6000"
CACHE_ROOT="/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/libero/gripper_WEIGHTED_VC=1.0---VS=0.2---AC=1.0---AS=0.2---Stage1=4000stepsAugmentedData--Stage2=AugmentedData/checkpoint-6000/goal_occluded"
TASK_SUITE_NAME="libero_goal_occluded"
PERSPECTIVE_OBS_KEY="robot0_eye_in_hand_image"  # alias for LIBERO robot0_eye_in_hand_image
MASTER_PORT=29722
GPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_TRIALS_PER_TASK=10
CAMERA_RESOLUTION=${CAMERA_RESOLUTION:-200}

NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
VISION_HUB=${VISION_HUB:-/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-VisionTokenizer}
VQ_HUB=${VQ_HUB:-/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-Stage1}

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
  --camera_resolution "$CAMERA_RESOLUTION" 
  
  
  
#  \
#   --perspective_eval \
#   --debug 
#  --debug
