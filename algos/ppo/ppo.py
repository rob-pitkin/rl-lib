import torch
import gymnasium as gym
from algos.utils import calculate_advantages_and_returns


class PPO:
    """
    PPO agent class

    Attributes:
        policy_net (PPONetwork): the policy network used for picking actions (actor)
        value_net (ValueNetwork): the value network used for critiquing actions (critic)
        policy_optimizer (torch.optim): the actor optimizer used for SGD
        value_optimizer (torch.optim): the critic optimizer used for SGD
        buffer (ReplayBuffer): the replay buffer to store experiences
    """

    def __init__(self):
        """
        Args:
        """
        self.policy_net = None
        self.value_net = None
        self.policy_optimizer = None
        self.value_optimizer = None
        self.buffer = None
        self.epsilon = None
        self.gamma = None
        self.lam = None
        self.epochs = None
        self.batch_size = None

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
    
    def train(self, num_episodes:int = 100, batch_size: int = 64, save_path: str):
        for _ in range(num_episodes):
            collect_rollout()
            if self.buffer.get_size() == batch_size:
                batch = self.buffer.sample(batch_size)
                states, actions, rewards, dones, log_probs, state_values, next_state_values = zip(*batch)
                advantages, returns = calculate_advantages_and_returns(rewards, state_values, next_state_values, self.gamma, self.lam)

                # perform PPO update
                self.update_params(states, actions, log_probs, advantages, returns, self.epsilon, self.epochs)
                self.buffer.clear()

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
                next_state_value = self.value_net(next_state) if not episode_done else 0.0

                experience = (
                    state,
                    action,
                    reward,
                    done,
                    log_prob,
                    state_value,
                    next_state_value
                )
                self.replay_buffer.append(experience)
                state = next_state