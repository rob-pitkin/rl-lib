import gymnasium as gym
import torch
import policyGradientAgent
from policyGradientAgent import PolicyGradientAgent
import unittest


def testCartPole():
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()
    print(
        f"Observation space shape: {obs.shape}, action space shape: {env.action_space}"
    )

    episode_over = False
    while not episode_over:
        action = (
            env.action_space.sample()
        )  # sample a random action from the action space
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {obs}, action: {action}, reward: {reward}")

        episode_over = terminated or truncated


class TestPolicyGradientAgent(unittest.TestCase):
    def setUp(self):
        self.agent = PolicyGradientAgent()

    def testPolicyNetworkForward(self):
        env = gym.make("CartPole-v1")
        obs, info = env.reset()
        out = self.agent.policy.forward(torch.tensor(obs))
        print(f"Example forward from policy network: {out}")
        self.assertAlmostEqual(sum(out).item(), 1.0)

    def testCalculateReturn(self):
        rewards = [i for i in range(10)]
        expected_return = [45, 45, 44, 42, 39, 35, 30, 24, 17, 9]
        returns = self.agent.calculate_return(rewards, 1.0)
        for i, r in enumerate(returns):
            self.assertAlmostEqual(expected_return[i], r)

    def testCalculatePolicyLoss(self):
        log_probs = [0.9, 0.8, 0.7]
        returns = [3, 2, 1]
        baseline_values = [1, 1, 1]
        expected_loss = 0
        for i in range(3):
            expected_loss -= log_probs[i] * (returns[i] - baseline_values[i])
        policy_loss = self.agent.calculate_policy_loss(
            log_probs, returns, baseline_values
        )
        self.assertAlmostEqual(policy_loss.item(), expected_loss, places=5)

    def testCalculateBaselineLoss(self):
        returns = [3, 2, 1]
        baseline_values = [1, 1, 1]
        expected_loss = 0
        for i in range(3):
            expected_loss -= (baseline_values[i] - returns[i]) ** 2
        expected_loss /= 3
        baseline_loss = self.agent.calculate_baseline_loss(returns, baseline_values)
        self.assertAlmostEqual(baseline_loss.item(), expected_loss, places=5)


def main():
    # testCartPole()
    pass


if __name__ == "__main__":
    main()
    unittest.main()
