#!/usr/bin/env python3

# NOTE: this requires jaxrl_m to be installed:
#       https://github.com/rail-berkeley/jaxrl_minimal
# Requires mujoco_py and mujoco==2.2.2

import time
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax import linen as nn
from flax.core import frozen_dict
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl.agents import DrQLearner
from serl.wrappers.frame_stack import FrameStack
from serl.utils.timer_utils import Timer
from serl.evaluation import evaluate

from edgeml.trainer import TrainerServer, TrainerClient, TrainerTunnel
from edgeml.data.data_store import QueuedDataStore

from edgeml.utils.jaxrl_m_common import ReplayBufferDataStore
from edgeml.utils.jaxrl_m_common import make_trainer_config, make_wandb_logger

from franka_env.envs.wrappers import GripperCloseEnv, SpacemouseIntervention
from franka_env.envs.wrappers import FrankaSERLObsWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "HalfCheetah-v4", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("utd_ratio", 8, "UTD ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 100000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 10, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 0, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 1, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")

flags.DEFINE_boolean("debug", False, "Debug mode.")

def print_green(x): return print("\033[92m {}\033[00m" .format(x))

##############################################################################


def actor(agent: DrQLearner, data_store, env, sampling_rng, tunnel=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    NOTE: tunnel is used the transport layer for multi-threading
    """
    if tunnel:
        client = tunnel
    else:
        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(),
            data_store,
            wait_for_server=True,
        )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        # chex.assert_trees_all_equal_shapes(params, agent.state.params)
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    done = False

    # NOTE: either use client.update() or client.start_async_update()
    # client.start_async_update(interval=1)  # every 1 sec

    # training loop
    timer = Timer()
    running_return = 0.0
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            # next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)
            running_return += reward

            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
            )

            obs = next_obs
            if done or truncated:
                stats = {"train": info}
                client.request("send-stats", stats)
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if FLAGS.eval_period and step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

##############################################################################


def learner(rng, agent: DrQAgent, replay_buffer, replay_iterator, wandb_logger=None, tunnel=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    NOTE: tunnel is used the transport layer for multi-threading
    """
    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    if tunnel:
        tunnel.register_request_callback(stats_callback)
        server = tunnel
    else:
        server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
        server.register_data_store("actor_env", replay_buffer)
        server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(total=FLAGS.training_starts, initial=len(replay_buffer),
                     desc="Filling up replay buffer", position=0, leave=True)
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green('sent initial network to actor')

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # Train the networks
        for critic_step in range(FLAGS.utd_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )
                agent = jax.block_until_ready(agent)

        with timer.context("train"):
            batch = next(replay_iterator)
            agent, update_info = agent.update_high_utd(
                batch, utd_ratio=1
            )
            agent = jax.block_until_ready(agent)

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        update_steps += 1

##############################################################################


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env = gym.make(FLAGS.env, fake_env=FLAGS.learner)
    env = GripperCloseEnv(env)
    if FLAGS.actor:
        env = SpacemouseIntervention(env)
    env = FrankaSERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="serl_franka",
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                # 'batch_size': FLAGS.batch_size * FLAGS.utd_ratio,
                'batch_size': FLAGS.batch_size,
                'pack_obs_and_next_obs': True,
            },
            device=sharding.replicate(),
        )
        # learner loop
        print_green("starting learner loop")
        learner(sampling_rng, agent, replay_buffer, replay_iterator=replay_iterator, wandb_logger=wandb_logger, tunnel=None)

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng, tunnel=None)

    else:
        print_green("starting actor and learner loop with multi-threading")

        # In this example, the tunnel acts as the transport layer for the
        # trainerServer and trainerClient. Also, both actor and learner shares
        # the same replay buffer.
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        tunnel = TrainerTunnel()
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

        import threading
        # Start learner thread
        learner_thread = threading.Thread(
            target=learner,
            args=(agent, replay_buffer, wandb_logger, tunnel)
        )
        learner_thread.start()

        # Start actor in main process
        actor(agent, replay_buffer, env, sampling_rng, tunnel=tunnel)
        learner_thread.join()


if __name__ == "__main__":
    app.run(main)