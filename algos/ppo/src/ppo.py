import torch
import gymnasium as gym
from torch.utils import data
from algos.utils import calculate_advantages_and_returns, ReplayBuffer
from network import PPONetwork, ValueNetwork
import numpy as np


class PPO:
    def __init__(
        self,
        env: gym.Env,
        net_arch: dict[str, list[int]],
        rollout_steps: int = 2048,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        seed: int | None = None,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
    ) -> None:
        """
        Initialize the Efficient PPO agent.

        Args:
            env (gym.Env): the gymnasium environment
            net_arch (dict[str, list[int]]): the network architecture for the policy and value networks
            rollout_steps (int): the number of steps to rollout for each iteration, default 2048
            lr (float): the learning rate for the policy and value networks, default 3e-4
            gamma (float): the discount factor hyperparameter of the agent, default 0.99
            gae_lambda (float): the lambda hyperparameter for GAE, default 0.95
            epsilon (float): the epsilon to use for gradient clipping in the PPO update, default 0.2
            epochs (int): the number of epochs to use for each iteration of PPO update, default 4
            batch_size (int): the batch size for the replay buffer to use for the PPO update, default 64
            seed (int | None): the seed to use for the random number generator, default None
            value_coeff (float): the coefficient for the value loss, default 0.5
            entropy_coeff (float): the coefficient for the entropy loss, default 0.01
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.env = env
        self.net_arch = net_arch
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        self.observation_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.action_dim = gym.spaces.utils.flatdim(env.action_space)

        self.policy_net = PPONetwork(
            self.observation_dim, self.action_dim, hidden_dims=self.net_arch["policy"]
        )
        self.value_net = ValueNetwork(
            self.observation_dim, hidden_dims=self.net_arch["value"]
        )

        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.lr,
        )

        self.rollout_buffer = ReplayBuffer(self.rollout_steps)

    def train(self, num_steps: int) -> None:
        """
        Train the PPO agent for a specified number of steps.

        Args:
            num_steps: The total number of training steps to perform.
        """
        t = 0
        state, _ = self.env.reset()

        while t < num_steps:
            if t % 20480 == 0:
                print(f"Training step {t}")

            self.rollout_buffer.clear()
            for step in range(self.rollout_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action_logits = self.policy_net(state_tensor)
                    dist = torch.distributions.Categorical(logits=action_logits)
                    action = dist.sample()

                next_state, reward, done, truncated, _ = self.env.step(action.item())

                self.rollout_buffer.append((state, action.item(), reward, done))

                state = next_state
                if done or truncated:
                    state, _ = self.env.reset()

            self._update_params(state)
            t += self.rollout_steps

    def _update_params(self, state) -> None:
        """
        Update the policy and value networks using the collected rollout data.

        Args:
            state: The most recent state of the environment (for creating the next state values)
        """
        batch = self.rollout_buffer.buffer

        (states, actions, rewards, dones) = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            all_state_values = self.value_net(
                torch.cat(
                    [states, torch.tensor(state, dtype=torch.float32).unsqueeze(0)],
                    dim=0,
                )
            )
            state_values = all_state_values[:-1]
            next_state_values = all_state_values[1:]

            # compute advantages and returns for updates
            advantages, returns = calculate_advantages_and_returns(
                rewards,
                state_values.detach(),
                next_state_values.detach(),
                dones,
                self.gamma,
                self.gae_lambda,
            )

            current_logits = self.policy_net(states)
            current_dist = torch.distributions.Categorical(logits=current_logits)
            current_log_probs = current_dist.log_prob(actions)

        dataset_size = self.rollout_steps

        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_log_probs = current_log_probs[batch_indices]

                # compute the ratio of the new policy to the old policy
                new_logits = self.policy_net(batch_states)
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = new_dist.log_prob(batch_actions)
                entropy = new_dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_log_probs)

                # compute the policy loss
                policy_loss = -torch.min(
                    ratio * batch_advantages.detach(),
                    torch.clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * batch_advantages.detach(),
                ).mean()  # Take mean across batch to get scalar loss

                # compute the value loss
                new_values = self.value_net(batch_states)
                value_loss = torch.nn.functional.mse_loss(
                    new_values, batch_returns.detach()
                )

                total_loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def eval_model(self, num_episodes: int) -> None:
        """
        Evaluates the model with human rendered episodes

        Args:
            num_episodes: number of evaluation episodes
        Returns:
            None
        """
        self.policy_net.eval()
        eval_env = gym.make(self.env.spec.id, render_mode="human")
        avg_reward = 0
        for _ in range(num_episodes):
            reward = 0
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action_logits = self.policy_net(torch.FloatTensor(obs).unsqueeze(0))
                action = torch.argmax(action_logits, dim=-1).item()
                obs, r, terminated, truncated, _ = eval_env.step(action)
                reward += r
                done = terminated or truncated
            avg_reward += reward
        eval_env.close()
        print(f"Average Reward: {avg_reward / num_episodes}")
        self.policy_net.train()

    def save_model(self, policy_filepath: str, value_filepath: str) -> None:
        """
        Saves the policy and value networks to respective filepaths

        Args:
            policy_filepath (str): path to save the policy network to
            value_filepath (str): path to save the value network to
        Returns:
            None
        """
        torch.save(self.policy_net, policy_filepath)
        torch.save(self.value_net, value_filepath)

    def load_model(self, policy_filepath: str, value_filepath: str) -> None:
        """
        Loads the policy and value networks from filepaths

        Args:
            policy_filepath (str): filepath of the policy_network to load from
            value_filepath (str): filepath of the value_network to load from
        Returns:
            None
        """
        self.policy_net = torch.load(policy_filepath, weights_only=False)
        self.value_net = torch.load(value_filepath, weights_only=False)
