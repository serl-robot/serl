"""Tests for panda.py."""

import numpy as np
from absl.testing import absltest

from mjenv.envs import panda

_SEED = 12345
_NUM_EPISODES = 1
_NUM_STEPS_PER_EPISODE = 10


class LiftCubeTest(absltest.TestCase):
    def _validate_observation(self, observation, observation_spec):
        self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
        for name, array_spec in observation_spec.items():
            array_spec.validate(observation[name])

    def test_task_runs(self) -> None:
        """Tests task loading and observation spec validity."""
        env = panda.LiftCube()
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
