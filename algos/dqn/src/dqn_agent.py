import gymnasium as gym
from network import QNetwork, DuelingQNetwork
import random
import torch
from replay_buffer import ReplayBuffer


class DQN:
    """
    DQN Agent class

    Attributes:
        q (QNetwork): the q-network used for estimating the q-value function
        target (QNetwork): the target network used for calculating the loss, copies over parameters from the q-network periodically
        replay_buffer (ReplayBuffer): the replay buffer for experience replay
        env (gym.Env): the environment used for the dqn agent
        env_id (str): the environment id used to create the gym environment
        optimizer (torch.optim.Optimizer): the optimizer used for the q network
        gamma (float): the discount factor for the dqn agent
        epsilon (float): the threshold for e-greedy action selection
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        activation_fn: str,
        env_id: str,
        hidden_dims: list[int] = None,
        buffer_capacity: int = 10000,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        network_type: str = "regular",
    ):
        """
        Args:
            obs_dim (int): size of the observation space
            action_dim (int): size of the action space
            activation_fn (str): activation function used in the Q and target network, either 'relu' or 'tanh'
            env_id (str): gym environment id
            hidden_dims (list[int]): list of hidden layer sizes
            buffer_capacity (int): capacity of the replay buffer
            gamma (float): discount factor
            epsilon (float): threshold for e-greedy action selection
        """
        self.q = (
            DuelingQNetwork(obs_dim, action_dim, activation_fn, hidden_dims)
            if network_type == "dueling"
            else QNetwork(obs_dim, action_dim, activation_fn, hidden_dims)
        )
        self.target = (
            DuelingQNetwork(obs_dim, action_dim, activation_fn, hidden_dims)
            if network_type == "dueling"
            else QNetwork(obs_dim, action_dim, activation_fn, hidden_dims)
        )
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.gamma = gamma
        self.epsilon = epsilon

    def train(self, num_episodes: int, save_path: str = None) -> None:
        """
        Trains the DQN agent for 'num_episodes' episodes.

        Args:
            num_episodes (int): number of episodes to train the agent
            save_path (str): optional path to save the agent's parameters
        """
        for i in range(num_episodes):
            if i % 100 == 0:
                print(f"Episode {i} of {num_episodes}")
            obs, _ = self.env.reset()
            state = torch.FloatTensor(obs)
            episode_done = False
            while not episode_done:
                # Get action values from the Q-network
                q_values = self.q(state)
                greedy = random.random() > self.epsilon
                action = (
                    torch.argmax(q_values).item()
                    if greedy
                    else self.env.action_space.sample()
                )
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = torch.FloatTensor(next_state)
                episode_done = terminated or truncated
                experience = (
                    state,
                    action,
                    reward,
                    next_state,
                    episode_done,
                )
                self.replay_buffer.append(experience)
                state = next_state

                # check if we should sample from the replay buffer to update our network
                if self.replay_buffer.getSize() > 64:
                    batch = self.replay_buffer.sample(64)
                    (
                        batch_states,
                        batch_actions,
                        batch_rewards,
                        batch_next_states,
                        batch_dones,
                    ) = zip(*batch)

                    # stacking the states into a matrix
                    batch_states = torch.stack(batch_states)
                    # setting the actions and rewards to column vectors
                    batch_actions = torch.tensor(batch_actions, dtype=torch.long).view(
                        -1, 1
                    )
                    batch_rewards = torch.tensor(
                        batch_rewards, dtype=torch.float32
                    ).view(-1, 1)
                    # stacking the next_states into a matrix
                    batch_next_states = torch.stack(batch_next_states)
                    # setting the dones to column vectors
                    batch_dones = torch.tensor(batch_dones, dtype=torch.float32).view(
                        -1, 1
                    )

                    # Computing Q-learning targets
                    with torch.no_grad():
                        next_state_q_values = self.target(batch_next_states).max(
                            1, keepdim=True
                        )[0]
                        q_targets = (
                            batch_rewards
                            + (
                                1 - batch_dones
                            )  # if the episode is over, don't include the next reward
                            * self.gamma
                            * next_state_q_values
                        )

                    # select the q values for each action from the batch
                    # gather acts on the first dimension (columns of q values for each action)
                    # and for each column, selects the index of batch_actions[i]
                    q_values = self.q(batch_states).gather(1, batch_actions)
                    loss = torch.nn.MSELoss()(q_values, q_targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Every 100 episodes, copy over the params to the target network
            if i % 100 == 0:
                self.copyParams()

        if save_path:
            self.saveModel(save_path)

    def copyParams(self) -> None:
        """
        Copies the parameters from the q network to the target network

        Args:
            None
        Returns:
            None
        """
        self.target.load_state_dict(self.q.state_dict())

    def loadModel(self, filepath: str) -> None:
        """
        Loads the model from a filepath
        Copies the new params into the target network.

        Args:
            filepath (str): filepath of the model to load from
        Returns:
            None
        """
        self.q = torch.load(filepath, weights_only=False)
        self.copyParams()

    def saveModel(self, filepath: str) -> None:
        """
        Saves the entire model of the q network to a path

        Args:
            filepath (str): path to save the model to
        Returns:
            None
        """
        torch.save(self.q, filepath)

    def evalModel(self, num_episodes: int) -> None:
        """
        Evaluates the model with human rendered episodes

        Args:
            num_episodes: number of evaluation episodes
        Returns:
            None
        """
        self.q.eval()
        eval_env = gym.make(self.env_id, render_mode="human")
        avg_reward = 0
        for i in range(num_episodes):
            reward = 0
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action = torch.argmax(self.q(torch.FloatTensor(obs))).item()
                obs, r, terminated, truncated, _ = eval_env.step(action)
                reward += r
                done = terminated or truncated
            avg_reward += reward
        eval_env.close()
        print(f"Average Reward: {avg_reward / num_episodes}")
