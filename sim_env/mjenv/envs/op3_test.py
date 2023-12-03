"""Tests for op3.py."""

import numpy as np
from absl.testing import absltest
from pathlib import Path
import mujoco

from mjenv.envs import op3

_SEED = 12345
_NUM_EPISODES = 1
_NUM_STEPS_PER_EPISODE = 10

# Page 31 of https://arxiv.org/pdf/2304.13653.pdf
_RESTRICTED_RANGE = {
    "head_pan": (-0.79, 0.79),
    "head_tilt": (-0.63, -0.16),
    "l_ank_pitch": (-0.4, 1.8),
    "l_ank_roll": (-0.4, 0.4),
    "l_el": (-1.4, 0.2),
    "l_hip_pitch": (-1.6, 0.5),
    "l_hip_roll": (-0.4, -0.1),
    "l_hip_yaw": (-0.3, 0.3),
    "l_knee": (-0.2, 2.2),
    "l_sho_pitch": (-2.2, 2.2),
    "l_sho_roll": (-0.8, 1.6),
    "r_ank_pitch": (-1.8, 0.4),
    "r_ank_roll": (-0.4, 0.4),
    "r_el": (-0.2, 1.4),
    "r_hip_pitch": (-0.5, 1.6),
    "r_hip_roll": (0.1, 0.4),
    "r_hip_yaw": (-0.3, 0.3),
    "r_knee": (-2.2, 0.2),
    "r_sho_pitch": (-2.2, 2.2),
    "r_sho_roll": (-1.6, 0.8),
}

_HERE = Path(__file__).parent


class Op3Test(absltest.TestCase):
    def test_joint_ranges(self) -> None:
        restricted_xml_path = _HERE / "xmls" / "robotis_op3" / "op3_limited.xml"
        model = mujoco.MjModel.from_xml_path(restricted_xml_path.as_posix())
        for joint_name, (lower, upper) in _RESTRICTED_RANGE.items():
            joint_range = model.joint(joint_name).range
            self.assertAlmostEqual(joint_range[0], lower)
            self.assertAlmostEqual(joint_range[1], upper)

    def test_position_actuator_ctrlrange(self) -> None:
        restricted_xml_path = _HERE / "xmls" / "robotis_op3" / "op3_limited.xml"
        model = mujoco.MjModel.from_xml_path(restricted_xml_path.as_posix())
        for joint_name, (lower, upper) in _RESTRICTED_RANGE.items():
            ctrlrange = model.actuator(f"{joint_name}_act").ctrlrange
            self.assertAlmostEqual(ctrlrange[0], lower)
            self.assertAlmostEqual(ctrlrange[1], upper)


class Op3StandTest(absltest.TestCase):
    def _validate_observation(self, observation, observation_spec):
        self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
        for name, array_spec in observation_spec.items():
            array_spec.validate(observation[name])

    def test_task_runs(self) -> None:
        """Tests task loading and observation spec validity."""
        env = op3.Op3Stand()
        random_state = np.random.RandomState(_SEED)

        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        self.assertTrue(np.all(np.isfinite(action_spec.minimum)))
        self.assertTrue(np.all(np.isfinite(action_spec.maximum)))

        for _ in range(_NUM_EPISODES):
            timestep = env.reset()
            for _ in range(_NUM_STEPS_PER_EPISODE):
                self._validate_observation(timestep.observation, observation_spec)
                if timestep.first():
                    self.assertIsNone(timestep.reward)
                    self.assertIsNone(timestep.discount)
                action = random_state.uniform(
                    action_spec.minimum, action_spec.maximum, size=action_spec.shape
                ).astype(action_spec.dtype)
                timestep = env.step(action)


if __name__ == "__main__":
    absltest.main()
