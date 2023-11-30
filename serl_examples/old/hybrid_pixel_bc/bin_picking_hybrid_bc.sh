export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES} && \
python pixel_bc.py \
    --env_name Franka-BinPick-v0 \
    --exp_prefix=franka_bc \
    --seed 42 \
    --config configs/pixel_hybrid_bc_config.py \
    --entity dmc_hand \
    --mode online \
    --max_steps 5000 \
    --utd_ratio 1 \
    --batch_size 256 \
    --eval_interval 2000 \
    --ckpt_path /home/undergrad/franka_bin_pick_bc \
    --save_video True
