export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export EGL_DEVICE_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1) && \
python pcb_insertion_ddpg.py \
    --env_name Franka-Robotiq-PCB-v0 \
    --project_name franka_pcb_insertion \
    --exp_prefix=norand_ddpg_utd4_099_eval \
    --seed 42 \
    --start_training 300 \
    --config configs/pixel_ddpg_config.py \
    --entity dmc_hand \
    --mode online \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_interval 1000 \
    --ckpt_path /home/undergrad/norand_pcb_ddpg \
    --save_video True \
    --eval_mode
