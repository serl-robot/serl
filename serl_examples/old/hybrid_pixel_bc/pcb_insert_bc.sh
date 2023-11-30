export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES} && \
python franka_bc.py \
    --env_name Franka-Robotiq-PCB-v0 \
    --exp_prefix=franka_pcb_bc \
    --seed 42 \
    --start_training 2000 \
    --config configs/pixel_bc_learner.py \
    --entity dmc_hand \
    --mode online \
    --utd_ratio 1 \
    --batch_size 256 \
    --eval_interval 2000 \
    --ckpt_path /home/undergrad/franka_pcb_bc \
    --save_video True