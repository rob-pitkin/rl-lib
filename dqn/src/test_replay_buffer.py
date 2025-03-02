import unittest
import gymnasium as gym
from replay_buffer import ReplayBuffer
import numpy as np


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CartPole-v1")

    def testAddSample(self):
        r = ReplayBuffer(1)
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        sarst = (obs, action, reward, next_obs, terminated or truncated)
        self.assertEqual(len(r.buffer), 0)
        r.append(sarst)
        self.assertEqual(len(r.buffer), 1)
        self.assertEqual(r.buffer[-1], sarst)

    def testCapacity(self):
        r = ReplayBuffer(1)
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        exp1 = (obs, action, reward, next_obs, terminated or truncated)
        r.append(exp1)
        action2 = self.env.action_space.sample()
        next_obs2, reward2, terminated, truncated, _ = self.env.step(action)
        exp2 = (next_obs, action2, next_obs2, terminated or truncated)
        r.append(exp2)
        self.assertEqual(len(r.buffer), 1)
        self.assertEqual(r.buffer[0], exp2)

    def testSample(self):
        r = ReplayBuffer(1)
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        exp1 = (obs, action, reward, next_obs, terminated or truncated)
        r.append(exp1)
        sample = r.sample(1)[0]
        self.assertTrue(np.array_equal(exp1[0], sample[0]))
        self.assertEqual(exp1[1], sample[1])
        self.assertEqual(exp1[2], sample[2])
        self.assertTrue(np.array_equal(exp1[3], sample[3]))
        self.assertEqual(exp1[4], sample[4])


if __name__ == "__main__":
    unittest.main()
