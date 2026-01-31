#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export ARNOLD_WORKER_GPU=1
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export RANK=0
export OMP_NUM_THREADS=16

# export MUJOCO_GL=osmesa
# export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.so
# export MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
# export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
# export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
# export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
# export NUMBA_DISABLE_JIT=1
# unset LD_LIBRARY_PATH

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"

# 1. Direct MuJoCo to use EGL
export MUJOCO_GL="egl"
# 2. Direct PyOpenGL to use EGL
export PYOPENGL_PLATFORM="egl"
# 3. (Optional) Force NVIDIA to be the vendor for GLVND
export __GL_VND_DISPATCH_LIBRARY_NAME=nvidia

ckpt_dir='/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/UNIVLA_LIBERO_CoTVLA_BS192_8k_gripper=False/checkpoint-8000'
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
# WITH_COT=${WITH_COT:-0}
# COT_ARGS=""
# if [ "$WITH_COT" = "1" ]; then
#     COT_ARGS="--with_cot"
#     if [ -n "$COT_MAX_NEW_TOKENS" ]; then
#         COT_ARGS+=" --cot_max_new_tokens ${COT_MAX_NEW_TOKENS}"
#     fi
# fi

python eval/libero/evaluate_libero_emu.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_10_occluded \
--cache_root /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/libero/UNIVLA_LIBERO_CoTVLA_BS192_8k_gripper=False/10_occluded \
--vision_hub /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-VisionTokenizer \
--vq_hub  /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-Stage1 \
--with_cot \
--cot_max_new_tokens 1024 \
# --debug
