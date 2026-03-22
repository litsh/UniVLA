WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

DATAPATH='/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/data_storage/meta/libero_all_norm.pkl'
ACTION_TOKENIZER_PATH="/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/pretrain/fast"
PERSPECTIVE_IMAGE_KEY="gripper_image"
PERSPECTIVE_VIEW_NAME="gripper"
EXP_NAME=${EXP_NAME:-"${PERSPECTIVE_VIEW_NAME}_WEIGHTED_Stage2_VC=0.75---VS=0.15---AC=1.0---AS=0.2"}

# Default group weights:
# - image content is downweighted because it has far more tokens
# - action content remains the main objective
# Override any of these from the shell when running experiments.
VISUAL_CONTENT_LOSS_WEIGHT=${VISUAL_CONTENT_LOSS_WEIGHT:-0.75}
VISUAL_SPECIAL_LOSS_WEIGHT=${VISUAL_SPECIAL_LOSS_WEIGHT:-0.15}
ACTION_CONTENT_LOSS_WEIGHT=${ACTION_CONTENT_LOSS_WEIGHT:-1.0}
ACTION_SPECIAL_LOSS_WEIGHT=${ACTION_SPECIAL_LOSS_WEIGHT:-0.2}

global_batch_size=192
per_gpu_batch_size=3
grad_accumulation_steps=$((global_batch_size / NGPUS / per_gpu_batch_size))
export PYTHONPATH=$(pwd)

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} train/train_moe.py \
    --model_name_or_path /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/gripper_WEIGHTED_VC=1.0---VS=0.1---AC=0.0---AS=0.0/checkpoint-2000 \
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
    --max_steps 4000 \
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
    --save_steps 1000 \
    --save_total_limit 3 \
    --eval_strategy no \
    --use_gripper False \
    --with_perspective True \
    --perspective_image_key "${PERSPECTIVE_IMAGE_KEY}" \
    --use_group_loss_weighting True \
    --visual_content_loss_weight ${VISUAL_CONTENT_LOSS_WEIGHT} \
    --visual_special_loss_weight ${VISUAL_SPECIAL_LOSS_WEIGHT} \
    --action_content_loss_weight ${ACTION_CONTENT_LOSS_WEIGHT} \
    --action_special_loss_weight ${ACTION_SPECIAL_LOSS_WEIGHT}
