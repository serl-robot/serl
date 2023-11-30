import dmcgym
import gym
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
import numpy as np
from serl.agents import SACLearner, VICELearner
from serl.data import ReplayBuffer, MemoryEfficientReplayBuffer
from serl.evaluation import evaluate
from serl.wrappers import wrap_gym
from serl.wrappers.wandb_video import WANDBVideo
from serl.wrappers.frame_stack import FrameStack
from serl.utils.commons import get_data, restore_checkpoint_, ema
from robot_infra.env.wrappers import GripperCloseEnv, SpacemouseIntervention, TwoCameraFrankaWrapper, FourDoFWrapper, ResetFreeWrapper
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
flags.DEFINE_string("project_name", "franka_bin_pick", "wandb project name.")
flags.DEFINE_string('exp_prefix', 'exp_0', 'experiment prefix.')
flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_string("mode", "online", "wandb mode.")
flags.DEFINE_string("entity", "use_your_own", "wandb entity.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 1, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e5), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("online_agent_update_interval", 100, "online agent update interval.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_string("ckpt_path", None, "Path to the checkpoints.")
flags.DEFINE_string('fw_vice_goal_path', None, 'Path to fw vice goal file.')
flags.DEFINE_string('bw_vice_goal_path', None, 'Path to bw vice goal file.')
flags.DEFINE_integer('vice_update_interval', 100, 'Vice update interval.')
flags.DEFINE_boolean('infinite_horizon', False, 'Is the env Infinite horizon.')
flags.DEFINE_boolean('eval_mode', False, 'is this a pure eval run')

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def learner_thread(agent: SACLearner, replay_buffer_iterator, vice_replay_buffer_iterator,
                   agent_queue: Queue, log_queue: Queue, train_queue: Queue, pixel_keys=('pixels',),
                   agent_idx=0):
    update_steps = 0
    for _ in tqdm.tqdm(range(FLAGS.max_steps), desc=f'learner thread_{agent_idx}', disable=not FLAGS.tqdm):
        # make sure there is always only two copies of the agent anywhere. This saves gpu memory
        # this also pauses the learner thread until the online agent consumes the latest update
        # this is optional, can be removed or changed
        while not agent_queue.empty():
            time.sleep(0.1)

        # block the training until an env step is taken, so one n utd update per env step
        train_step = train_queue.get()
        batch = next(replay_buffer_iterator)
        agent, update_info = agent.update(batch, FLAGS.utd_ratio, pixel_keys)

        if (update_steps+1) % FLAGS.vice_update_interval == 0:
            vice_batch = next(vice_replay_buffer_iterator)
            agent, vice_info = agent.update_classifier(vice_batch, pixel_keys)
            update_info.update(vice_info)

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
    # initialize wandb
    unique_identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project=FLAGS.project_name, mode=FLAGS.mode, entity=FLAGS.entity,
        group=FLAGS.exp_prefix,
        tags=f'{FLAGS.env_name}_{FLAGS.exp_prefix}',
        id=f'{FLAGS.exp_prefix}_{unique_identifier}_{FLAGS.seed}',
    )
    wandb.config.update(FLAGS)

    # initialize franka bin picking envs
    np.random.seed(FLAGS.seed) # in v4 env, env seed is deprecated, np.random.seed to control action sampling
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True, flatten_states=False)
    env_no_intervention = env
    env = SpacemouseIntervention(env, gripper_enabled=True)
    env = TwoCameraFrankaWrapper(env)
    pixel_keys = tuple([k for k in env.observation_space.spaces.keys() if 'state' not in k])
    env = RemoveGripperStateWrapper(env)
    env = FrameStack(env, 1, stacking_key=pixel_keys)
    env = ResetFreeWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    if FLAGS.save_video:
        env = WANDBVideo(env, pixel_keys=pixel_keys)
    eval_env = env_no_intervention

    # load VICE goal pools for each task
    vice_paths = [FLAGS.fw_vice_goal_path, FLAGS.bw_vice_goal_path]
    vice_goal_pools = []
    task_count = 0
    for vice_path in vice_paths:
        assert os.path.exists(vice_path)
        vice_goal_pool = jnp.load(vice_path)
        #expand the last dim to match the obs space dim
        vice_goal_pool = frozen_dict.freeze({k: v[..., None] for k,v in vice_goal_pool.items()})
        import imageio as imageio
        for k,v in vice_goal_pool.items():
            print(f'{k} goal pool shape: {v.shape}')
            for img_id in range(v.shape[0]):
                imageio.imsave(f'./goal_images/task_{task_count}_{k}_goal_image_{img_id}.png', v[img_id][..., 0])
        vice_goal_pool = jax.device_put(vice_goal_pool)
        vice_goal_pools.append(vice_goal_pool)
        task_count += 1

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agents = [globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space,
            vice_goal_pool=vice_goal_pool, pixel_keys=pixel_keys, **kwargs
    ) for vice_goal_pool in vice_goal_pools]

    agent_nums = len(agents)

    if FLAGS.eval_mode: # example: eval checkpoints, load from paths
        actor = restore_checkpoint_(
            f'/home/undergrad/20demos_fwbw_bin_pick_vice_ckpts/20demos_4dimstate_fwbw_pick_screw_vice_2wrists_fixbugutd4_099_20230914_103739/actor_0_26000',
            agents[0].actor,
            step=None,
        )
        agents[0] = agents[0].replace(actor=actor)
        actor = restore_checkpoint_(
            f'/home/undergrad/20demos_fwbw_bin_pick_vice_ckpts/20demos_4dimstate_fwbw_pick_screw_vice_2wrists_fixbugutd4_099_20230914_103739/actor_1_34000',
            agents[1].actor,
            step=None,
        )
        agents[1] = agents[1].replace(actor=actor)

        for i in range(100):
            for n in range(agent_nums):
                eval_info, video = evaluate(
                    agents[n],
                    env,
                    num_episodes=FLAGS.eval_episodes,
                    save_video=FLAGS.save_video,
                    name=f'eval_{n}_video',
                    reset_kwargs={'task_id': n},
                )

    # create replay buffers for each task
    replay_buffers = [MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps, pixel_keys=pixel_keys
    ) for _ in range(agent_nums)]

    for replay_buffer in replay_buffers:
        replay_buffer.seed(FLAGS.seed)

    # create replay buffer iterators for each task
    replay_buffer_iterators = [replay_buffer.get_iterator(
        sample_args={
            'batch_size': FLAGS.batch_size * FLAGS.utd_ratio,
            'pack_obs_and_next_obs': True,
            'demo_batch_size': FLAGS.batch_size // 2 * FLAGS.utd_ratio,
            'demo_size': len(replay_buffer),
    }) for replay_buffer in replay_buffers]

    # create vice sampler iterators for each task
    vice_replay_buffer_iterators = [replay_buffer.get_iterator(
        sample_args={
            'batch_size': FLAGS.batch_size,
            'pack_obs_and_next_obs': True,
    }) for replay_buffer in replay_buffers]

    def vice_plot(agent, step: int, images: OrderedDict[str, np.ndarray], rewards=None, id="train"):
        # a helper function to plot vice rewards and upload to wandb logger
        vice_obs = frozen_dict.freeze(images)
        new_agent, vice_rews = agent.vice_reward(vice_obs, pixel_keys)
        data_list=[*vice_rews.values()]
        labels=[*vice_rews.keys()]

        if rewards:
            data_list.append(np.array(rewards))
            labels.append('task_rews')
            data_list.append(ema(np.array(rewards)))
            labels.append('ema_task_rews')

        for data, label in zip(data_list, labels):
            fig = sns.lineplot(data=data, label=label)

        fig.set_ylim(-0.05,1.05)
        fig = fig.get_figure()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clf()
        wandb.log({id: wandb.Image(img)}, commit=False)
        return new_agent

    # initializing multi-threading
    agent_queues = [Queue() for _ in range(agent_nums)]
    log_queues = [Queue() for _ in range(agent_nums)]
    train_queues = [Queue() for _ in range(agent_nums)]
    learn_threads = [threading.Thread(target=learner_thread,
                        args=(agents[i], replay_buffer_iterators[i], vice_replay_buffer_iterators[i],
                              agent_queues[i], log_queues[i], train_queues[i], pixel_keys, i)) for i in range(agent_nums)]

    observation, _ = env.reset()
    done = False
    agent_idx = 0
    transitions = {f'agent_{n}': [] for n in range(agent_nums)} # for storing replay buffer for each task periodically

    for i in tqdm.tqdm(
        range(1, (FLAGS.max_steps + 1) * agent_nums), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i % FLAGS.eval_interval == 0:
            # save checkpoints for each TrainState in agent, use Flax checkpointing
            for n in range(agent_nums):
                checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                    agents[n].actor,
                    prefix=f'actor_{n}_',
                    step=i,
                    keep=2000,
                )
                checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                    agents[n].critic,
                    prefix=f'critic_{n}_',
                    step=i,
                    keep=2000,
                )
                checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                    agents[n].target_critic,
                    prefix=f'target_critic_{n}_',
                    step=i,
                    keep=2000,
                )
                checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                    agents[n].temp,
                    prefix=f'temp_{n}_',
                    step=i,
                    keep=2000,
                )
                for k,v in agents[n].vice_classifiers.items():
                    checkpoints.save_checkpoint(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                        v,
                        prefix=f'vice_classifier_{n}_{k}_',
                        step=i,
                        keep=2000,
                    )

                try:
                    with open(os.path.join(f'{FLAGS.ckpt_path}/{FLAGS.exp_prefix}_{unique_identifier}',
                                            f'replay_buffer_{n}_{unique_identifier}_{i}.pkcl'), 'wb') as f:
                        pickle.dump(transitions[f'agent_{n}'], f)
                        transitions[f'agent_{n}'].clear()
                except Exception as e:
                    print(f'save replay buffer {n} failed at {i}', e)


        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agents[agent_idx] = agents[agent_idx].sample_actions(observation)

        next_observation, reward, done, truncated, info = env.step(action)

        if not FLAGS.infinite_horizon:
            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
        else:
            mask = 1.0

        action = info['expert_action'] # critical: allows spacemouse to override action

        transition = dict(observations=observation,
                        next_observations=next_observation,
                        actions=action,
                        rewards=reward,
                        masks=mask,
                        dones=done,)
        transitions[f'agent_{agent_idx}'].append(copy.deepcopy(transition))
        replay_buffers[agent_idx].insert(transition)

        observation = next_observation

        # each insert into the replay buffer, will trigger one train thread update with n utd
        train_queues[agent_idx].put(i)

        if not log_queues[agent_idx].empty():
            logs, log_step = log_queues[agent_idx].get()
            for k, v in logs.items():
                wandb.log({f'training_{agent_idx}/{k}': v}, step=i)
                wandb.log({f'training_{agent_idx}/update_step': log_step}, step=i)

        if done or truncated:
            agents[agent_idx] = vice_plot(agents[agent_idx], i, env.get_video(), rewards=env.get_rewards(), id=f"train_{agent_idx}_vice")
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training_{agent_idx}/{decode[k]}": v}, step=i)

            if not agent_queues[agent_idx].empty():
                agents[agent_idx] = agent_queues[agent_idx].get()

            observation, _ = env.reset(task_id=(agent_idx + 1) % agent_nums)
            agent_idx = (agent_idx + 1) % agent_nums
            done = False

            if not agent_queues[agent_idx].empty():
                agents[agent_idx] = agent_queues[agent_idx].get()

        if (i+1) == FLAGS.start_training:
            print('start learning thread')
            for l in range(agent_nums):
                learn_threads[l].start()
                # block the main thread until the first agent update is ready
                # this avoids the main thread runs super fast ahead while waiting for jax to compile RL update in sim.
                # Most likely not an issue in real world.
                agents[l] = agent_queues[l].get()

    for learn_thread in learn_threads:
        learn_thread.join()

if __name__ == "__main__":
    app.run(main)
