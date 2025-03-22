import torch
import gymnasium as gym
from algos.utils import calculate_advantages_and_returns, ReplayBuffer


class PPO:
    """
    PPO agent class

    Attributes:
        policy_net (PPONetwork): the policy network used for picking actions (actor)
        value_net (ValueNetwork): the value network used for critiquing actions (critic)
        policy_optimizer (torch.optim): the actor optimizer used for SGD
        value_optimizer (torch.optim): the critic optimizer used for SGD
        buffer (ReplayBuffer): the replay buffer to store experiences
        epsilon (float): the epsilon to use for gradient clipping in the PPO update, default 0.2
        gamma (float): the discount factor hyperparameter of the agent
        lam (float): the lambda hyperparameter for GAE, default 0.95
        epochs (int): the number of epochs to use for each iteration of PPO update, default 10
        batch_size (int): the batch size for the replay buffer to use for the PPO update, default 64
        env_id (str): the gymnasium environment id
        env (gym.env): the gymnasium environment
        policy_lr (float): the learning rate of the policy_network, default 0.001
        value_lr (float): the learning rate of the value_network, default 0.001

    """

    def __init__(
        self,
        policy_net,
        value_net,
        buffer_capacity,
        env_id,
        lam=0.95,
        epsilon=0.2,
        gamma=0.99,
        epochs=10,
        batch_size=64,
        policy_lr=0.001,
        value_lr=0.001,
    ):
        """
        Args:
            policy_net (PPONetwork): the policy network used for picking actions (actor)
            value_net (ValueNetwork): the value network used for critiquing actions (critic)
            buffer_capacity (int): the capacity of the replay buffer
            env_id (str): the gymnasium environment id to use for training and eval
            lam (float): the lambda hyperparameter for GAE
            epsilon (float): the epsilon to use for gradient clipping in the PPO update, default 0.2
            gamma (float): the discount factor hyperparameter of the agent, default 0.99
            epochs (int): the number of epochs to use for each iteration of PPO update, default 10
            policy_lr (float): the learning rate of the policy_network, default 0.001
            value_lr (float): the learning rate of the value_network, default 0.001

        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.buffer = ReplayBuffer(buffer_capacity)
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.env_id = env_id
        self.env = gym.make(self.env_id)
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.policy_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=self.value_lr
        )
        if self.env.action_space.shape[0] != self.policy_net.output_dim:
            raise ValueError(
                f"Environment action space: {self.env.action_space.shape} incompatible with policy net output_dim: {self.policy_net.output_dim}"
            )
        if (
            self.env.observation_space.shape[0] != self.policy_net.input_dim
            or self.env.observation_space.shape[0] != self.value_net.input_dim
        ):
            raise ValueError(
                f"Environment observation space: {self.env.observation_space} incompatible with policy net input: {self.policy_net.input_dim} or value net input: {self.value_net.input_dim}"
            )

    def update_params(
        self,
        states: torch.tensor,
        actions: torch.tensor,
        prev_log_probs: torch.tensor,
        advantages: torch.tensor,
        returns: torch.tensor,
        epsilon: float = 0.2,
        epochs: float = 10,
    ) -> None:
        """ """
        for _ in range(epochs):
            # Compute the log probs of each taken action
            new_log_probs = torch.log(self.policy_net(states).gather(1, actions))
            # use log rules: log(a) - log(b) = log(a/b), exp(log(a/b)) = a/b
            prob_ratio = torch.exp(new_log_probs - prev_log_probs)
            # compute the PPO objective fn
            actor_loss = -torch.min(
                prob_ratio * advantages,
                torch.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages,
            )

            # Compute the state values and value loss
            state_values = self.value_net(states)
            critic_loss = torch.nn.functional.mse_loss(state_values, returns)

            # backprop the actor update
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            # backprop the critic update
            self.value_optimizer.zero_grad()
            critic_loss.backward()
            self.value_optimizer.step()

    def train(
        self,
        policy_save_path: str,
        value_save_path: str,
        num_episodes: int = 100,
        batch_size: int = 64,
    ):
        for _ in range(num_episodes):
            collect_rollout()
            if self.buffer.get_size() >= batch_size:
                batch = self.buffer.sample(batch_size)
                (
                    states,
                    actions,
                    rewards,
                    dones,
                    log_probs,
                    state_values,
                    next_state_values,
                ) = zip(*batch)
                advantages, returns = calculate_advantages_and_returns(
                    rewards, state_values, next_state_values, self.gamma, self.lam
                )

                # perform PPO update
                self.update_params(
                    states,
                    actions,
                    log_probs,
                    advantages,
                    returns,
                    self.epsilon,
                    self.epochs,
                )
                self.buffer.clear()
        self.save_model(policy_save_path, value_save_path)

    def collect_rollout(self):
        with torch.no_grad():
            obs, _ = self.env.reset()
            state = torch.tensor(obs)
            episode_done = False
            while not episode_done:
                # Pick the action from the policy
                action_probs = self.policy_net(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()

                # get the log prob of the action
                log_prob = torch.log_prob(action)

                # get the value of the state
                state_value = self.value_net(state)

                # take the next step in the env
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = torch.tensor(next_state)
                episode_done = terminated or truncated
                next_state_value = (
                    self.value_net(next_state) if not episode_done else 0.0
                )

                experience = (
                    state,
                    action,
                    reward,
                    done,
                    log_prob,
                    state_value,
                    next_state_value,
                )
                self.replay_buffer.append(experience)
                state = next_state

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

    def eval_model(self, num_episodes: int) -> None:
        """
        Evaluates the model with human rendered episodes

        Args:
            num_episodes: number of evaluation episodes
        Returns:
            None
        """
        self.policy_net.eval()
        eval_env = gym.make(self.env_id, render_mode="human")
        avg_reward = 0
        for i in range(num_episodes):
            reward = 0
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action = torch.argmax(self.policy_net(torch.FloatTensor(obs))).item()
                obs, r, terminated, truncated, _ = eval_env.step(action)
                reward += r
                done = terminated or truncated
            avg_reward += reward
        eval_env.close()
        print(f"Average Reward: {avg_reward / num_episodes}")
