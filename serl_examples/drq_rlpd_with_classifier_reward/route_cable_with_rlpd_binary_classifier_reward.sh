export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export EGL_DEVICE_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1) && \
python cable_route_franka_rlpd_classifier_reward_multithread.py \
    --env_name Franka-RouteCable-v0 \
    --exp_prefix=franka_cable_classifier_2wrists_fixbugutd4_099_eval \
    --seed 42 \
    --start_training 300 \
    --config configs/rlpd_classifier_reward_config.py \
    --entity dmc_hand \
    --mode online \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_interval 2000 \
    --ckpt_path /home/undergrad/franka_cable_ckpts \
    --save_video True
