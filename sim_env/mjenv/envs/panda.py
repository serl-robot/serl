import mujoco
from pathlib import Path
import numpy as np
import dm_env
from dm_env import specs
from mjenv import MujocoDmEnv, RenderingSpec
from mjenv.controllers import opspace

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])


class LiftCube(MujocoDmEnv):
    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 0.1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: RenderingSpec = RenderingSpec(),
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

    def reset(self) -> dm_env.TimeStep:
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2

        obs = self._compute_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs,
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        x, y, z, grasp = action

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp.
        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = self.time_limit_exceeded()
        discount = 1.0
        if terminated:
            step_type = dm_env.StepType.LAST
        else:
            step_type = dm_env.StepType.MID
        return dm_env.TimeStep(
            step_type=step_type,
            reward=rew,
            discount=discount,
            observation=obs,
        )

    def observation_spec(self):
        return {
            "panda/joint_pos": specs.Array(shape=(7,), dtype=np.float32),
            "panda/joint_vel": specs.Array(shape=(7,), dtype=np.float32),
            # "panda/joint_torque": specs.Array(shape=(21,), dtype=np.float32),
            # "panda/wrist_force": specs.Array(shape=(3,), dtype=np.float32),
            "block_pos": specs.Array(shape=(3,), dtype=np.float32),
        }

    def action_spec(self):
        return specs.BoundedArray(
            shape=(4,),
            dtype=np.float32,
            minimum=np.asarray([-1.0, -1.0, -1.0, -1.0]),
            maximum=np.asarray([1.0, 1.0, 1.0, 1.0]),
        )

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}

        joint_pos = np.stack(
            [self._data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
        ).ravel()
        obs["panda/joint_pos"] = joint_pos.astype(np.float32)

        joint_vel = np.stack(
            [self._data.sensor(f"panda/joint{i}_vel").data for i in range(1, 8)],
        ).ravel()
        obs["panda/joint_vel"] = joint_vel.astype(np.float32)

        # joint_torque = np.stack(
        # [self._data.sensor(f"panda/joint{i}_torque").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_torque"] = symlog(joint_torque.astype(np.float32))

        # wrist_force = self._data.sensor("panda/wrist_force").data.astype(np.float32)
        # obs["panda/wrist_force"] = symlog(wrist_force.astype(np.float32))

        block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        obs["block_pos"] = block_pos

        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        r_lift = np.clip(r_lift, 0.0, 1.0)
        rew = 0.3 * r_close + 0.7 * r_lift
        return rew
