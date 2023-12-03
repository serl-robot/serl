export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
python async_serl_drq_sim.py \
    --actor \
    --render \
    --env PandaPickCubeVision-v0 \
    --exp_name=serl_dev_sim_franka_test \
    --seed 0 \
    --training_starts 300 \
    --config configs/drq_config.py \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --debug
