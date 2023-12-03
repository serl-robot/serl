import time
import mujoco
import mujoco.viewer
import numpy as np
from mjenv import envs
from dm_control.viewer import user_input

env = envs.LiftCube(action_scale=(0.01, 1))
# env = envs.Op3Stand()
action_spec = env.action_spec()


def sample():
    a = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
    return a.astype(action_spec.dtype)


m = env.model
d = env.data

reset = False


def key_callback(keycode):
    if keycode == user_input.KEY_SPACE:
        global reset
        reset = True


env.reset()
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    start = time.time()
    while viewer.is_running():
        if reset:
            env.reset()
            reset = False
        else:
            step_start = time.time()
            env.step(sample())
            viewer.sync()
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
