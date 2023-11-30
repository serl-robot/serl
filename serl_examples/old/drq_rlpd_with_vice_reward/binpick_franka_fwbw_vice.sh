export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES} && \
python binpick_franka_fwbw_vice.py \
    --env_name Franka-BinPick-v0 \
    --project_name new_franka_fwbw_pick_test \
    --exp_prefix=20demos_4dimstate_fwbw_pick_screw_vice_2wrists_fixbugutd4_099_eval \
    --seed 42 \
    --start_training 600 \
    --config configs/franka_vice_config.py \
    --entity dmc_hand \
    --mode online \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_interval 2000 \
    --ckpt_path /home/undergrad/20demos_fwbw_bin_pick_vice_ckpts \
    --save_video True \
    --fw_vice_goal_path /home/undergrad/code/jaxrl-franka/examples/pixels/bin_pick_fw_goal_images.npz \
    --bw_vice_goal_path /home/undergrad/code/jaxrl-franka/examples/pixels/bin_pick_bw_goal_images.npz
