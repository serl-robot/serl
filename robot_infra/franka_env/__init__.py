from gymnasium.envs.registration import register

register(
    id='FrankaRobotiq-Vision-v0',
    entry_point='franka_env.envs:FrankaRobotiq',
    max_episode_steps=100,
)