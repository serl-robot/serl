import mujoco
from pathlib import Path
import numpy as np
import dm_env
from dm_env import specs
from dm_robotics.transformations import transformations as tr

from mjenv import MujocoDmEnv, RenderingSpec


_HERE = Path(__file__).parent

_XML_PATH = _HERE / "xmls" / "robotis_op3" / "scene.xml"
LIMITED_JNT_RANGE_XML_PATH = _HERE / "xmls" / "robotis_op3" / "scene_limited.xml"

_JOINT_NAMES = [
    "head_pan",
    "head_tilt",
    "l_sho_pitch",
    "l_sho_roll",
    "l_el",
    "r_sho_pitch",
    "r_sho_roll",
    "r_el",
    "l_hip_yaw",
    "l_hip_roll",
    "l_hip_pitch",
    "l_knee",
    "l_ank_pitch",
    "l_ank_roll",
    "r_hip_yaw",
    "r_hip_roll",
    "r_hip_pitch",
    "r_knee",
    "r_ank_pitch",
    "r_ank_roll",
]

_TARGET_Z = 0.279

_JOINT_WEIGHTS = np.asarray([1.0] * 20)
_JOINT_WEIGHTS[:8] = 0.0  # Zero out arm and head joints.


class Op3Stand(MujocoDmEnv):
    def __init__(
        self,
        restrict_joint_range: bool = False,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: RenderingSpec = RenderingSpec(),
    ):
        super().__init__(
            xml_path=LIMITED_JNT_RANGE_XML_PATH if restrict_joint_range else _XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

    def reset(self) -> dm_env.TimeStep:
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        init_scheme = self._random.choice(
            ["stand", "crouch", "fall"], p=[0.2, 0.2, 0.6]
        )
        if init_scheme == "stand":
            self._data.qpos[:] = self._model.key("stand").qpos
        elif init_scheme == "crouch":
            self._data.qpos[:] = self._model.key("crouch").qpos
        else:
            self._drop_from_height()

        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs,
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        self._data.ctrl[:] = action
        mujoco.mj_step(self._model, self._data, self._n_substeps)
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
        nq = self._model.nq - 7
        return {
            "op3/joint_pos": specs.Array(shape=(nq,), dtype=np.float32),
            "op3/joint_vel": specs.Array(shape=(nq,), dtype=np.float32),
            "op3/imu": specs.Array(shape=(4,), dtype=np.float32),
        }

    def action_spec(self):
        return specs.BoundedArray(
            shape=(self._model.nu,),
            dtype=np.float32,
            minimum=self._model.actuator_ctrlrange[:, 0],
            maximum=self._model.actuator_ctrlrange[:, 1],
        )

    # Helper methods.

    def _drop_from_height(self) -> None:
        # Randomize the joint positions of the robot.
        jnt_range = self._model.jnt_range[1:, 1:]
        qpos = self._random.uniform(*jnt_range.T, size=self._model.nu)
        self._data.qpos[7:] = qpos

        # Randomly initialize the position and orientation of the root body.
        root_pos = np.array([0, 0, self._random.uniform(0.4, 0.5)])
        euler = self._random.uniform(
            low=[-np.pi / 4, -np.pi / 4, -np.pi], high=[np.pi / 4, np.pi / 4, np.pi]
        )
        root_quat = tr.euler_to_quat(euler)
        self._data.qpos[:3] = root_pos
        self._data.qpos[3:7] = root_quat

        # Step for a few seconds to let the robot fall.
        while self._data.time < 2.0:
            mujoco.mj_step(self._model, self._data)
        self._data.time = 0.0

    def _compute_observation(self) -> dict:
        obs = {}

        joint_pos = np.stack(
            [self._data.sensor(f"{n}_pos").data for n in _JOINT_NAMES],
        ).ravel()
        obs["op3/joint_pos"] = joint_pos.astype(np.float32)

        joint_vel = np.stack(
            [self._data.sensor(f"{n}_vel").data for n in _JOINT_NAMES],
        ).ravel()
        obs["op3/joint_vel"] = joint_vel.astype(np.float32)

        quat = self._data.sensor("body_quat").data
        roll, pitch, _ = tr.quat_to_euler(quat)
        gyro = self._data.sensor("body_gyro").data
        dr, dp, _ = gyro
        obs["op3/imu"] = np.array([roll, pitch, dr, dp], dtype=np.float32)

        return obs

    def _compute_reward(self) -> float:
        r_height = self._compute_height_reward()
        r_roll = self._compute_roll_reward()
        rew = r_roll * (1.0 + r_height) / 2.0
        return rew

    def _compute_height_reward(self) -> float:
        root_h = self._data.sensor("body_pos").data[2]
        h_err = _TARGET_Z - root_h
        h_err /= _TARGET_Z
        h_err = np.clip(h_err, 0.0, 1.0)
        r_height = 1.0 - h_err
        return r_height

    def _compute_joint_position_reward(self) -> float:
        joint_pos = np.stack(
            [self._data.sensor(f"{n}_pos").data for n in _JOINT_NAMES],
        ).ravel()
        tar_pos = self._model.key("stand").qpos[7:]
        pose_diff = tar_pos - joint_pos
        pose_diff = (_JOINT_WEIGHTS * pose_diff) ** 2
        pose_err = np.sum(pose_diff)
        r_pose = np.exp(-0.6 * pose_err)
        return r_pose

    def _compute_roll_reward(self) -> float:
        quat = self._data.sensor("body_quat").data
        xmat = tr.quat_to_mat(quat)
        r_roll = (0.5 * xmat[0, 0] + 0.5) ** 2
        return r_roll
