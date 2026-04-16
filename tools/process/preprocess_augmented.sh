export OLD_RLDS=/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/openvla-oft/data_storage/original_libero_multiview
export NEW_RLDS=/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v
export STAGE1_POOL=/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/libero_augmented_v2_stage1_pool

export OUTPUT_DIR=/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/libero_augmented_v2_stage1_share035
export FAILURE_SHARE=0.35
python tools/process/materialize_libero_stage1_share.py \
    --dataset '/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/openvla-oft/data_storage/original_libero_multiview/libero_object/1.0.0|object|success|original|object_success' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_object_replay_failure/1.0.0|object|failure|original|object_failure_original' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_object_occluded_replay_failure/1.0.0|object|failure|occluded|object_failure_occluded' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_object_occluded_replay_success/1.0.0|object|success|occluded|object_success_occluded' \
    --dataset '/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/openvla-oft/data_storage/original_libero_multiview/libero_spatial/1.0.0|spatial|success|original|spatial_success' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_spatial_replay_failure/1.0.0|spatial|failure|original|spatial_failure_original' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_spatial_occluded_replay_success/1.0.0|spatial|success|occluded|spatial_success_occluded' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_spatial_occluded_replay_failure/1.0.0|spatial|failure|occluded|spatial_failure_occluded' \
    --dataset '/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/openvla-oft/data_storage/original_libero_multiview/libero_goal/1.0.0|goal|success|original|goal_success' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_goal_replay_failure/1.0.0|goal|failure|original|goal_failure_original' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_goal_occluded_replay_success/1.0.0|goal|success|occluded|goal_success_occluded' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero_goal_occluded_replay_failure/1.0.0|goal|failure|occluded|goal_failure_occluded' \
    --dataset '/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/openvla-oft/data_storage/original_libero_multiview/libero_10/1.0.0|10|success|original|libero10_success' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero10_replay_failure/1.0.0|10|failure|original|libero10_failure_original' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero10_occluded_replay_success/1.0.0|10|success|occluded|libero10_success_occluded' \
    --dataset '/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/data_storage/rlds_augmented_v2/libero10_occluded_replay_failure/1.0.0|10|failure|occluded|libero10_failure_occluded' \
    --pool_root "${STAGE1_POOL}" \
    --output_root "${OUTPUT_DIR}" \
    --target_failure_share "${FAILURE_SHARE}"


python tools/pickle_gen/pickle_generation_libero.py \
    --dataset_path "${OUTPUT_DIR}" \
    --output_path "${OUTPUT_DIR}/meta" \
    --normalizer_path "/inspire/hdd/global_user/chenfangke-253108540237/tsli/UniVLA/configs/normalizer_libero_augmented_v2_stage1_share035" \
    --output_filename libero_all_norm_stage1_share035.pkl