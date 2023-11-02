import dmcgym
import gym
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
import numpy as np
from serl.agents import SACLearner, FrankaDRQClassifierLearner
from serl.data import ReplayBuffer, MemoryEfficientReplayBuffer
from serl.evaluation import evaluate
from serl.wrappers import wrap_gym
from serl.utils.commons import get_data, restore_checkpoint_
from franka.env_franka.franka_env.envs.wrappers import GripperCloseEnv, SpacemouseIntervention, TwoCameraFrankaWrapper, FourDoFWrapper, ResetFreeWrapper
from serl.wrappers.wandb_video import WANDBVideo
from serl.wrappers.frame_stack import FrameStack
import os
import threading
from queue import Queue
import time
import jax
from flax import jax_utils
from flax.core import frozen_dict
from datetime import datetime
from flax.training import checkpoints
from jax import numpy as jnp
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import pickle
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import copy

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "franka_cable_multicam", "wandb project name.")
flags.DEFINE_string('exp_prefix', 'exp_0', 'experiment prefix.')
flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_string("mode", "online", "wandb mode.")
flags.DEFINE_string("entity", "jianlan", "wandb entity.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 1, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e5), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("online_agent_update_interval", 100, "online agent update interval.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_string("ckpt_path", None, "Path to the checkpoints.")
flags.DEFINE_string('vice_goal_path', None, 'Path to vice goal file.')
flags.DEFINE_integer('vice_update_interval', 100, 'Vice update interval.')
flags.DEFINE_boolean('infinite_horizon', False, 'Is the env Infinite horizon.')
flags.DEFINE_boolean('eval_mode', False, 'is this a pure eval run')

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def learner_thread(agent: SACLearner, replay_buffer_iterator,
                   agent_queue: Queue, log_queue: Queue, train_queue: Queue, pixel_keys=('pixels',)):
    update_steps = 0
    for _ in tqdm.tqdm(range(FLAGS.max_steps), desc='learner thread', disable=not FLAGS.tqdm):
        # make sure there is always only two copies of the agent anywhere. This saves gpu memory
        # this also pauses the learner thread until the online agent consumes the latest update
        # this is optional, can be removed or changed
        while not agent_queue.empty():
            time.sleep(0.1)

        # block the training until an env step is taken, so one n utd update per env step
        train_step = train_queue.get()
        batch = next(replay_buffer_iterator)
        agent, update_info = agent.update(batch, FLAGS.utd_ratio, pixel_keys)

        # let the main thread log. In my tests, wandb only logs from the main thread.
        if (update_steps+1) % FLAGS.log_interval == 0:
            log_queue.put((update_info, train_step))

        # this number can be changed to control how often the online agent is updated
        if (update_steps+1) % FLAGS.online_agent_update_interval == 0:
            if agent_queue.empty():
                # agent queue is empty, put agent in queue @ {update_steps}
                agent_queue.put(agent)
            else:
                # agent queue is not empty, throw out the agent in queue, and put new agent in queue
                agent_queue.get()
                agent_queue.put(agent)

        update_steps += 1

def main(_):
    unique_identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project=FLAGS.project_name, mode=FLAGS.mode, entity=FLAGS.entity,
        group=FLAGS.exp_prefix,
        tags=f'{FLAGS.env_name}_{FLAGS.exp_prefix}',
        id=f'{FLAGS.exp_prefix}_{unique_identifier}_{FLAGS.seed}',
    )
    wandb.config.update(FLAGS)

    np.random.seed(FLAGS.seed) # in v4 env, env seed is deprecated, np.random.seed to control action sampling
    env = gym.make(FLAGS.env_name)

    # env = GripperCloseEnv(env)
    env = wrap_gym(env, rescale_actions=True, flatten_states=False)
    # env  = FourDoFWrapper(env)
    env_no_intervention = env
    env = SpacemouseIntervention(env, gripper_enabled=True)
    env = TwoCameraFrankaWrapper(env)
    pixel_keys = tuple([k for k in env.observation_space.spaces.keys() if 'state' not in k])
    env = FrameStack(env, 1, stacking_key=pixel_keys)
    env = ResetFreeWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    if FLAGS.save_video:
        env = WANDBVideo(env, pixel_keys=pixel_keys)
    eval_env = env_no_intervention

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, pixel_keys=pixel_keys, **kwargs
    )

    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps, pixel_keys=pixel_keys
    )
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            'batch_size': FLAGS.batch_size * FLAGS.utd_ratio,
    })
    replay_buffer.seed(FLAGS.seed)

    classifier = restore_checkpoint_(
        '/home/undergrad/code/jaxrl-franka/examples/pixels/franka_binary_classifier',
        agent.classifier,
        step = 100,
    )
    agent = agent.replace(classifier=classifier)

    # initializing multi-threading
    agent_queue = Queue()
    log_queue = Queue()
    train_queue = Queue()
    learn_thread = threading.Thread(target=learner_thread,
                        args=(agent, replay_buffer_iterator,
                              agent_queue, log_queue, train_queue, pixel_keys))

    observation, _ = env.reset()
    done = False
    transitions = [] # used for storing the replay buffer periodically


    # restore checkpoints and run eval
    # TODO: should make the path user specified
    if FLAGS.eval_mode:
        actor = restore_checkpoint_(
            '/home/undergrad/franka_cable_ckpts/franka_cable_classifier_2wrists_fixbugutd4_099_20230910_160303/actor_20000',
            # '/home/undergrad/ur_reset_free_ckpts/ur_cable_classifier_2wrists_fixbugutd4_099_20230910_000829/actor_16000',
            agent.actor,
            step = None,
        )
        agent = agent.replace(actor=actor)
        for i in range(100):
            eval_info, video = evaluate(
                agent,
                env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i % FLAGS.eval_interval == 0:
            # save checkpoints for each TrainState in agent, use Flax checkpointing
            checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                agent.actor,
                prefix='actor_',
                step=i,
                keep=1000,
            )
            checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                agent.critic,
                prefix='critic_',
                step=i,
                keep=1000,
            )
            checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                agent.target_critic,
                prefix='target_critic_',
                step=i,
                keep=1000,
            )
            checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                agent.temp,
                prefix='temp_',
                step=i,
                keep=1000,
            )

            try:
                with open(os.path.join(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                                        f'replay_buffer_{unique_identifier}_{i}.pkcl'), 'wb') as f:
                    pickle.dump(transitions, f)
                    transitions.clear()

            except Exception as e:
                print(f'save replay buffer failed at {i}', e)


        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        next_observation, reward, done, truncated, info = env.step(action)

        # over ride the reward with classifier reward
        # terminate the episode if classifier reward is 1
        tmp_obs = copy.deepcopy(next_observation)
        tmp_obs.pop('state')
        tmp_obs = jax.tree_map(lambda x: x.squeeze() / 255.0, tmp_obs)
        rew = agent.classify_reward(frozen_dict.freeze(tmp_obs))
        reward = int(rew >= 0.5) * 1.0
        done = done or reward == 1
        if reward == 1:
            info["episode"] = {"r": reward, "l": env.episode_lengths[0], "t": round(time.perf_counter() - env.t0, 6)}

        if not FLAGS.infinite_horizon:
            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
        else:
            mask = 1.0

        action = info['expert_action']
        if FLAGS.env_name == 'UR-Cable-v0':
            a = np.zeros(4)
            a[:3] = action[:3]
            a[-1] = action[-1]
            action = a.copy()
        transition = dict(observations=observation,
                        next_observations=next_observation,
                        actions=action,
                        rewards=reward,
                        masks=mask,
                        dones=done,)
        transitions.append(copy.deepcopy(transition))
        replay_buffer.insert(transition)

        observation = next_observation

        # each insert into the replay buffer, will trigger one train thread update with n utd
        train_queue.put(i)

        if not log_queue.empty():
            logs, log_step = log_queue.get()
            for k, v in logs.items():
                wandb.log({f'training/{k}': v}, step=i)
                wandb.log({f'training/update_step': log_step}, step=i)

        if done or truncated:
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i)

            if not agent_queue.empty():
                # print(f'update agent from training thread @ {i}')
                agent = agent_queue.get()

            observation, _ = env.reset()
            done = False

            if not agent_queue.empty():
                # print(f'update agent from training thread @ {i}')
                agent = agent_queue.get()

        if (i+1) == FLAGS.start_training:
            print('start learning thread')
            learn_thread.start()
            # block the main thread until the first agent update is ready
            # this avoids the main thread runs super fast ahead while waiting for jax to compile RL update in sim.
            # Most likely not an issue in real world.
            agent = agent_queue.get()

    learn_thread.join()

if __name__ == "__main__":
    app.run(main)
