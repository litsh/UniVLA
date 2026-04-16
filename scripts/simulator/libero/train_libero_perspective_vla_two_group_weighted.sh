WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

# Default weights for the 2-group mode:
# - all visual tokens are treated as one group
# - all action tokens are treated as one group
VISUAL_LOSS_WEIGHT=${VISUAL_LOSS_WEIGHT:-0.75}
ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-1.0}

DATAPATH=${DATAPATH:-'/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/data_storage/meta/libero_all_norm.pkl'}
ACTION_TOKENIZER_PATH="/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/pretrain/fast"
PERSPECTIVE_IMAGE_KEY="gripper_image"
PERSPECTIVE_VIEW_NAME="gripper"
PERSPECTIVE_USE_VANILLA_PREFIX=${PERSPECTIVE_USE_VANILLA_PREFIX:-False}
GROUP_LOSS_MODE=${GROUP_LOSS_MODE:-two_group}

EXP_NAME=${EXP_NAME:-"${PERSPECTIVE_VIEW_NAME}_WEIGHTED_MODE=two_group---V=${VISUAL_LOSS_WEIGHT}---A=${ACTION_LOSS_WEIGHT}---Stage1=4000stepsAugmentedData--Stage2=OriginalData"}

# Default weights for the current 4-group mode:
# - image content is downweighted because it has far more tokens
# - action content remains the main objective
# VISUAL_CONTENT_LOSS_WEIGHT=${VISUAL_CONTENT_LOSS_WEIGHT:-1.0}
# VISUAL_SPECIAL_LOSS_WEIGHT=${VISUAL_SPECIAL_LOSS_WEIGHT:-0.2}
# ACTION_CONTENT_LOSS_WEIGHT=${ACTION_CONTENT_LOSS_WEIGHT:-1.0}
# ACTION_SPECIAL_LOSS_WEIGHT=${ACTION_SPECIAL_LOSS_WEIGHT:-0.2}



if [[ "${GROUP_LOSS_MODE}" == "four_group" ]]; then
    GROUP_LOSS_ARGS=(
        --group_loss_mode "${GROUP_LOSS_MODE}"
        --use_group_loss_weighting True
        --visual_content_loss_weight "${VISUAL_CONTENT_LOSS_WEIGHT}"
        --visual_special_loss_weight "${VISUAL_SPECIAL_LOSS_WEIGHT}"
        --action_content_loss_weight "${ACTION_CONTENT_LOSS_WEIGHT}"
        --action_special_loss_weight "${ACTION_SPECIAL_LOSS_WEIGHT}"
    )
elif [[ "${GROUP_LOSS_MODE}" == "two_group" ]]; then
    GROUP_LOSS_ARGS=(
        --group_loss_mode "${GROUP_LOSS_MODE}"
        --use_group_loss_weighting True
        --visual_loss_weight "${VISUAL_LOSS_WEIGHT}"
        --action_loss_weight "${ACTION_LOSS_WEIGHT}"
    )
else
    echo "Unsupported GROUP_LOSS_MODE: ${GROUP_LOSS_MODE}. Expected one of: four_group, two_group." >&2
    exit 1
fi



global_batch_size=192
per_gpu_batch_size=3
grad_accumulation_steps=$((global_batch_size / NGPUS / per_gpu_batch_size))
export PYTHONPATH=$(pwd)
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} train/train_moe.py \
    --model_name_or_path /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/gripper_WEIGHTED_VC=1.0---VS=0.2---AC=0.0---AS=0.0--AugmentedData/checkpoint-4000 \
    --model_config_path /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3_H200.json \
    --output_dir "logs/${EXP_NAME}" \
    --learning_rate 4e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 6000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 50 \
    --frames 1 \
    --action_frames 10 \
    --max_position_embeddings 3200 \
    --seed 42 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --apply_loss_on_only_action False \
    --actions True \
    --actions_format "fast" \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    --per_device_train_batch_size ${per_gpu_batch_size} \
    --gradient_accumulation_steps ${grad_accumulation_steps} \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --eval_strategy no \
    --use_gripper False \
    --with_perspective True \
    --perspective_image_key "${PERSPECTIVE_IMAGE_KEY}" \
    --perspective_use_vanilla_prefix ${PERSPECTIVE_USE_VANILLA_PREFIX} \
    "${GROUP_LOSS_ARGS[@]}"
