import dmcgym
import gym
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
import numpy as np
from serl.agents import PixelDDPGLearner
from serl.data import ReplayBuffer, MemoryEfficientReplayBuffer
from serl.evaluation import evaluate
from serl.wrappers import wrap_gym
from franka.env_franka.franka_env.envs.wrappers import SpacemouseIntervention, TwoCameraFrankaWrapper, InsertionWrapper
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
flags.DEFINE_string("project_name", "franka_insertion_ddpg", "wandb project name.")
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
flags.DEFINE_boolean('infinite_horizon', False, 'Is the env Infinite horizon.')
flags.DEFINE_boolean('eval_mode', False, 'is this a pure eval run')

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def get_data(data, start, end):
    '''
    recursive helper function to extract data range from a dataset dict, which is used in the replay buffer

    :param data: the input data, can be a dataset dict or a numpy array from the replay buffer
    :param start: the start index of the data range
    :param end: the end index of the data range\
    :return: eventually returns the numpy array within range
    '''
    if type(data) == dict:
        return {k: get_data(v, start, end) for k,v in data.items()}
    return data[start:end]

def learner_thread(agent: PixelDDPGLearner, replay_buffer_iterator,
                   agent_queue: Queue, log_queue: Queue, train_queue: Queue, pixel_keys=('pixels',)):
    '''
    create a learner thread to update the agent, and put the updated agent in a queue for the main thread to consume

    :param agent: the agent to update, TrainState
    :param replay_buffer_iterator: the replay buffer iterator, which is used to sample the training batch
    :param agent_queue: the queue to put the updated agent in
    :param log_queue: the queue to put the log info in
    :param train_queue: the queue to get the train step from
    :param pixel_keys: the pixel keys in the observation space
    :return: None
    '''
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
    # initialize the wandb logger
    unique_identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project=FLAGS.project_name, mode=FLAGS.mode, entity=FLAGS.entity,
        group=FLAGS.exp_prefix,
        tags=f'{FLAGS.env_name}_{FLAGS.exp_prefix}',
        id=f'{FLAGS.exp_prefix}_{unique_identifier}_{FLAGS.seed}',
    )
    wandb.config.update(FLAGS)

    # initialize the random seed and the robot environment
    np.random.seed(FLAGS.seed) # in v4 env, env seed is deprecated, np.random.seed to control action sampling
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True, flatten_states=False)
    env_no_intervention = env
    env = SpacemouseIntervention(env)
    env = TwoCameraFrankaWrapper(env)
    env = InsertionWrapper(env)
    pixel_keys = tuple([k for k in env.observation_space.spaces.keys() if 'state' not in k])
    env = FrameStack(env, 1, stacking_key=pixel_keys)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    if FLAGS.save_video:
        env = WANDBVideo(env, pixel_keys=pixel_keys)
    eval_env = env_no_intervention

    def restore_checkpoint_(path, item, step):
        '''
        helper function to restore checkpoints from a path, checks if the path exists

        :param path: the path to the checkpoints folder
        :param item: the TrainState to restore
        :param step: the step to restore
        :return: the restored TrainState
        '''

        assert os.path.exists(path)
        return checkpoints.restore_checkpoint(path, item, step)

    # initialize the agent from user specified config
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, pixel_keys=pixel_keys, **kwargs
    )

    # initialize the replay buffer with env observation space and action space
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps, pixel_keys=pixel_keys
    )
    # initialize the replay buffer iterator
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            'batch_size': FLAGS.batch_size * FLAGS.utd_ratio,
            'pack_obs_and_next_obs': True,
    })
    replay_buffer.seed(FLAGS.seed)

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
    xy_s = [] # used for logging and plotting the Q values heatmap in the paper

    # restore checkpoints and run eval
    # TODO: should make the path user specified
    if FLAGS.eval_mode:
        import ipdb; ipdb.set_trace()
        actor = restore_checkpoint_(
            '/home/undergrad/norand_pcb_ddpg/norand_ddpg_utd4_099_20230915_001556/actor_22000',
            agent.actor,
            step=None,
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
        if i % FLAGS.eval_interval == 0: # save checkpoints and replay buffer periodically
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
            try:
                with open(os.path.join(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                                        f'replay_buffer_{unique_identifier}_{i}.pkcl'), 'wb') as f:
                    pickle.dump(transitions, f)
                    transitions.clear()
                with open(os.path.join(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                                        f'xy_{unique_identifier}_{i}.pkcl'), 'wb') as f:
                    pickle.dump(xy_s, f)
                    xy_s.clear()

            except Exception as e:
                print(f'save replay buffer failed at {i}', e)

        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        next_observation, reward, done, truncated, info = env.step(action)

        if not FLAGS.infinite_horizon:
            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
        else:
            mask = 1.0

        action = info['expert_action']
        transition = dict(observations=observation,
                        next_observations=next_observation,
                        actions=action,
                        rewards=reward,
                        masks=mask,
                        dones=done,)
        transitions.append(copy.deepcopy(transition))
        xy_s.append(info['xy'].copy()) # used for plotting later
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
