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
        env_id,
        buffer_capacity,
        policy_net=None,
        value_net=None,
        cnn_net=None,
        lam=0.95,
        epsilon=0.2,
        gamma=0.99,
        epochs=10,
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
        self.policy_net, self.value_net, self.cnn_net = None, None, None
        if cnn_net:
            self.cnn_net = cnn_net
        else:
            self.policy_net = policy_net
            self.value_net = value_net
        self.buffer = ReplayBuffer(buffer_capacity)
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.env_id = env_id
        self.env = (
            gym.make(self.env_id)
            if not cnn_net
            else gym.make(self.env_id, continuous=False)
        )
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.policy_optimizer, self.value_optimizer, self.cnn_optimizer = (
            None,
            None,
            None,
        )
        if self.cnn_net:
            self.cnn_optimizer = torch.optim.Adam(
                self.cnn_net.parameters(), lr=self.policy_lr
            )
        else:
            self.policy_optimizer = torch.optim.Adam(
                self.policy_net.parameters(), lr=self.policy_lr
            )
            self.value_optimizer = torch.optim.Adam(
                self.value_net.parameters(), lr=self.value_lr
            )

    def update_params_2d(
        self,
        states: torch.tensor,
        actions: torch.tensor,
        prev_log_probs: torch.tensor,
        advantages: torch.tensor,
        returns: torch.tensor,
        epsilon: float = 0.2,
        epochs: float = 10,
        value_coeff: float = 0.5,
    ) -> None:
        """ """
        # state shape assertions
        torch._assert(
            states.dim() == 4,
            message=f"Expected states to be 4D tensors, got {states.shape}",
        )
        torch._assert(
            states.shape[3] == 3,
            message=f"Expected 3 input channels for each state, got {states.shape[1]}",
        )

        # action shape assertions
        torch._assert(
            actions.dim() == 2,
            f"Expected actions to be 2D tensors, got {actions.shape}",
        )

        for _ in range(epochs):
            # Compute a forward pass of the states, get the action_probs and state values
            # action_logits will be (N, action_space), state_values will be (N, 1)
            action_logits, state_values = self.cnn_net(states)
            # Compute the log probs of each taken action
            # gather is an indexing wrapper, takes the dim to "gather on" and the indices to select
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            # use log rules: log(a) - log(b) = log(a/b), exp(log(a/b)) = a/b
            prob_ratio = torch.exp(new_log_probs - prev_log_probs)
            # compute the PPO objective fn
            actor_loss = -torch.min(
                prob_ratio * advantages,
                torch.clip(prob_ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages,
            ).mean()  # Take mean across batch to get scalar loss

            # Compute the value loss
            critic_loss = torch.nn.functional.mse_loss(state_values, returns)

            # backprop the loss
            self.cnn_optimizer.zero_grad()
            total_loss = actor_loss + value_coeff * critic_loss
            total_loss.backward()
            self.cnn_optimizer.step()

    def update_params_1d(
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
            action_logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            # use log rules: log(a) - log(b) = log(a/b), exp(log(a/b)) = a/b
            prob_ratio = torch.exp(new_log_probs - prev_log_probs)
            # compute the PPO objective fn
            actor_loss = -torch.min(
                prob_ratio * advantages,
                torch.clip(prob_ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages,
            ).mean()  # Take mean across batch to get scalar loss

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
        policy_save_path: str = "",
        value_save_path: str = "",
        cnn_save_path: str = "",
        num_episodes: int = 100,
        batch_size: int = 64,
    ):
        for i in range(num_episodes):
            if i % 100 == 0:
                print(f"Episode {i} of {num_episodes}")
            if self.cnn_net:
                self.collect_rollout_2d()
            else:
                self.collect_rollout_1d()
            if self.buffer.get_size() >= self.buffer.capacity:
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
                # convert to tensors
                states = torch.stack([s.squeeze(0) for s in states])
                actions = torch.stack(actions)
                rewards = torch.stack(rewards)
                log_probs = torch.stack(log_probs)
                state_values = torch.stack(state_values)
                next_state_values = torch.stack(next_state_values)
                advantages, returns = calculate_advantages_and_returns(
                    rewards, state_values, next_state_values, self.gamma, self.lam
                )

                # perform PPO update
                if self.cnn_net:
                    self.update_params_2d(
                        states,
                        actions,
                        log_probs,
                        advantages,
                        returns,
                        self.epsilon,
                        self.epochs,
                    )
                else:
                    self.update_params_1d(
                        states,
                        actions,
                        log_probs,
                        advantages,
                        returns,
                        self.epsilon,
                        self.epochs,
                    )
                self.buffer.clear()
        if self.cnn_net:
            self.save_model_2d(cnn_save_path)
        else:
            self.save_model(policy_save_path, value_save_path)

    def collect_rollout_1d(self):
        with torch.no_grad():
            obs, _ = self.env.reset()
            state = torch.tensor(obs).unsqueeze(0)
            episode_done = False
            while not episode_done:
                # Pick the action from the policy
                action_logits = self.policy_net(state)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()

                # get the log prob of the action
                log_prob = dist.log_prob(action)

                # get the value of the state
                state_value = self.value_net(state).squeeze(0)

                # take the next step in the env
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
                next_state = torch.tensor(next_state).unsqueeze(0)
                episode_done = terminated or truncated
                next_state_value = (
                    self.value_net(next_state).squeeze(0)
                    if not episode_done
                    else torch.tensor([0.0], dtype=torch.float32)
                )

                experience = (
                    state,
                    action,
                    torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
                    episode_done,
                    log_prob,
                    state_value,
                    next_state_value,
                )
                self.buffer.append(experience)
                state = next_state

    def collect_rollout_2d(self):
        with torch.no_grad():
            obs, _ = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            episode_done = False
            while not episode_done:
                # Pick the action from the policy
                action_logits, state_value = self.cnn_net(state)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()

                # get the log prob of the action
                log_prob = dist.log_prob(action)

                # take the next step in the env
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                episode_done = terminated or truncated
                next_state_value = (
                    self.cnn_net(next_state)[1].squeeze(0)
                    if not episode_done
                    else torch.tensor([0.0], dtype=torch.float32)
                )

                experience = (
                    state,
                    action,
                    torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
                    episode_done,
                    log_prob,
                    state_value.squeeze(0),
                    next_state_value,
                )
                self.buffer.append(experience)
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

    def load_model_2d(self, cnn_filepath: str) -> None:
        """
        Loads the cnn from a filepath

        Args:
            cnn_filepath (str): filepath of the cnn to load from
        Returns:
            None
        """
        self.cnn_net = torch.load(cnn_filepath, weights_only=False)

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

    def save_model_2d(self, cnn_filepath: str) -> None:
        """
        Saves the cnn to a filepath

        Args:
            cnn_filepath (str): path to save the cnn to
        Returns:
            None
        """
        torch.save(self.cnn_net, cnn_filepath)

    def eval_model(self, num_episodes: int) -> None:
        """
        Evaluates the model with human rendered episodes

        Args:
            num_episodes: number of evaluation episodes
        Returns:
            None
        """
        if self.cnn_net:
            self.cnn_net.eval()
        else:
            self.policy_net.eval()
        eval_env = gym.make(self.env_id, render_mode="human")
        avg_reward = 0
        for _ in range(num_episodes):
            reward = 0
            obs, _ = eval_env.reset()
            done = False
            while not done:
                if self.cnn_net:
                    action_logits, _ = self.cnn_net(torch.FloatTensor(obs).unsqueeze(0))
                else:
                    action_logits = self.policy_net(torch.FloatTensor(obs).unsqueeze(0))
                action = torch.argmax(action_logits, dim=-1).item()
                obs, r, terminated, truncated, _ = eval_env.step(action)
                reward += r
                done = terminated or truncated
            avg_reward += reward
        eval_env.close()
        print(f"Average Reward: {avg_reward / num_episodes}")
