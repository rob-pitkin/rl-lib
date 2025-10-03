import gymnasium
import torch
import numpy as np
from network import ActorNetwork, CriticNetwork
from algos.utils import ReplayBuffer, calculate_advantages_and_returns


class AdvantageActorCriticAgent:
    def __init__(
        self,
        env: gymnasium.Env,
        net_arch: dict[str, list[int]],
        update_frequency: int = 32,
        lr: float = 0.0007,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize the Advantage Actor-Critic agent.

        Args:
            env (gymnasium.Env): The environment to interact with.
            net_arch (dict[str, list[int]]): Network architecture for the actor and critic networks. Keys are 'actor' and 'critic'.
            update_frequency (int): Frequency of updating the actor and critic networks.
            lr (float): Learning rate for the actor and critic networks.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            seed (int): Seed for random number generation.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.env = env
        self.net_arch = net_arch
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_frequency = update_frequency
        self.observation_dim = gymnasium.spaces.utils.flatdim(env.observation_space)
        self.action_dim = gymnasium.spaces.utils.flatdim(env.action_space)
        self.actor = ActorNetwork(
            self.observation_dim,
            self.action_dim,
            self.net_arch["actor"] if "actor" in self.net_arch else [],
        )
        self.critic = CriticNetwork(
            self.observation_dim,
            self.net_arch["critic"] if "critic" in self.net_arch else [],
        )
        self.rollout_buffer = ReplayBuffer(self.update_frequency)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def train(self, num_steps: int) -> None:
        # set our current training step to 0
        t = 0
        # init the environment
        state, _ = self.env.reset()
        # while our current step is less than the number of training steps
        while t < num_steps:
            if t % 10000 == 0:
                print(f"Training step: {t}")
            # reset the rollout buffer
            self.rollout_buffer.clear()
            # start collecting experiences
            for step in range(self.update_frequency):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                # sample the action given the state (policy)
                with torch.no_grad():
                    action_logits = self.actor(state_tensor)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                # get the log probability for the future update
                log_prob = dist.log_prob(action).item()

                # take a step in the environment
                next_state, reward, done, truncated, _ = self.env.step(action.item())
                done = done or truncated

                # add the experience to the rollout buffer
                self.rollout_buffer.append(
                    (
                        state,
                        action.item(),
                        reward,
                        done,
                    )
                )

                # update the current state
                state = next_state

                # if the episode is done or truncated, reset the environment
                if done:
                    state, _ = self.env.reset()

            # update the actor and critic networks
            batch = self.rollout_buffer.buffer

            # parse out the components of the batch
            (
                states,
                actions,
                rewards,
                dones,
            ) = zip(*batch)

            # convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            all_state_values = self.critic(
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

            current_logits = self.actor(states)
            current_dist = torch.distributions.Categorical(logits=current_logits)
            current_log_probs = current_dist.log_prob(actions)

            # compute the actor loss
            actor_loss = -torch.mean(current_log_probs * advantages.detach())

            # compute the critic loss
            critic_loss = torch.nn.functional.mse_loss(state_values, returns.detach())

            # update the actor and critic networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            t += self.update_frequency

    def load_model(self, actor_filepath: str, critic_filepath: str) -> None:
        """
        Loads the actor and critic networks from filepaths

        Args:
            actor_filepath (str): filepath of the actor_network to load from
            critic_filepath (str): filepath of the critic_network to load from
        Returns:
            None
        """
        self.actor = torch.load(actor_filepath, weights_only=False)
        self.critic = torch.load(critic_filepath, weights_only=False)

    def save_model(self, actor_filepath: str, critic_filepath: str) -> None:
        """
        Saves the actor and critic networks to filepaths

        Args:
            actor_filepath (str): filepath of the actor_network to save to
            critic_filepath (str): filepath of the critic_network to save to
        Returns:
            None
        """
        torch.save(self.actor, actor_filepath)
        torch.save(self.critic, critic_filepath)

    def eval_model(self, num_episodes: int) -> None:
        """
        Evaluates the model with human rendered episodes

        Args:
            num_episodes: number of evaluation episodes
        Returns:
            None
        """
        self.actor.eval()
        self.critic.eval()
        eval_env = gymnasium.make(self.env.spec.id, render_mode="human")
        avg_reward = 0
        for _ in range(num_episodes):
            reward = 0
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action_logits = self.actor(torch.FloatTensor(obs).unsqueeze(0))
                action = torch.argmax(action_logits, dim=-1).item()
                obs, r, terminated, truncated, _ = eval_env.step(action)
                reward += r
                done = terminated or truncated
            avg_reward += reward
        eval_env.close()
        print(f"Average Reward: {avg_reward / num_episodes}")
        self.actor.train()
        self.critic.train()
