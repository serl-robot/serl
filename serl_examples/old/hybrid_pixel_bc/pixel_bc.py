import dmcgym
import gym
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
import numpy as np
from serl.agents import PixelBCLearner, PixelHybridBCLearner
from serl.data import ReplayBuffer, MemoryEfficientReplayBuffer
from serl.evaluation import evaluate
from serl.wrappers import wrap_gym
from robot_infra.env.wrappers import GripperCloseEnv, SpacemouseIntervention, TwoCameraFrankaWrapper, FourDoFWrapper, ResetFreeWrapper
from serl.wrappers.wandb_video import WANDBVideo
from serl.wrappers.frame_stack import FrameStack
from serl.utils.commons import restore_checkpoint_
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
flags.DEFINE_string("project_name", "franka_pcb_bc", "wandb project name.")
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
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_string("ckpt_path", None, "Path to the checkpoints.")
flags.DEFINE_boolean('eval_mode', False, 'is this a pure eval run')

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

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

    env = wrap_gym(env, rescale_actions=True, flatten_states=False)
    env_no_intervention = env
    env = SpacemouseIntervention(env, gripper_enabled=True)
    env = TwoCameraFrankaWrapper(env)
    pixel_keys = tuple([k for k in env.observation_space.spaces.keys() if 'state' not in k])
    env = FrameStack(env, 1, stacking_key=pixel_keys)
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
    with open('/home/undergrad/bin_pick_demos/'+
            f'fwbw_demos_2k.pkcl', 'rb') as f:
        dataset_dict = pickle.load(f)
        for d in tqdm.tqdm(dataset_dict):
            replay_buffer.insert(
                dict(
                    observations=d['observations'],
                    next_observations=d['next_observations'],
                    actions=d['actions'],
                    rewards=d['rewards'],
                    masks=d['masks'],
                    dones=d['dones'],               
                )
            )
    del dataset_dict
    print(f'buffer size: {len(replay_buffer)}')

    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            'batch_size': FLAGS.batch_size * FLAGS.utd_ratio,
    })
    replay_buffer.seed(FLAGS.seed)

    def ema(series, alpha=0.5):
        smoothed = np.zeros_like(series, dtype=float)
        smoothed[0] = series[0]
        for i in range(1, len(series)):
            smoothed[i] = alpha * series[i] + (1-alpha) * smoothed[i-1]
        return smoothed
    
    if FLAGS.eval_mode:
        actor = restore_checkpoint_(
            '/home/undergrad/franka_bin_pick_bc/20demos_franka_bc_20230913_132616/actor_10000',
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
        batch = next(replay_buffer_iterator)
        agent, info = agent.update(batch, utd_ratio=1, pixel_keys=pixel_keys)
        for k,v in info.items():
            wandb.log({f'training/{k}': v}, step=i)

    checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                agent.actor,
                prefix='actor_',
                step=FLAGS.max_steps,
                keep=1000,
            )

if __name__ == "__main__":
    app.run(main)
