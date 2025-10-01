import gymnasium
import numpy as np


class QLearningAgent:
    """
    Implementation of the Q-Learning algorithm for MDPs with epsilon-greedy exploration.
    Algorithm pseudocode from "Multi-agent Reinforcement Learning" by Albrecht, et al.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.95,
    ):
        """
        Initialize the Q-Learning agent.

        Args:
            num_states (int): The number of states in the environment.
            num_actions (int): The number of actions available in the environment.
            epsilon (float): The exploration rate.
            alpha (float): The learning rate.
            gamma (float): The discount factor.

        Returns:
            None
        """
        self.num_states: int = num_states
        self.num_actions: int = num_actions
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.q: np.ndarray = np.zeros((num_states, num_actions))

    def select_action(self, state: int) -> int:
        """
        Select an action using epsilon-greedy exploration.

        Args:
            state (int): The current state.

        Returns:
            int: The selected action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return int(np.argmax(self.q[state]))

    def update(
        self, state: int, action: int, reward: float, next_state: int, next_action: int
    ):
        """
        Updates Q-values using the Q-learning update rule.

        Args:
            state (int): Current state.
            action (int): Current action.
            reward (float): Reward received.
            next_state (int): Next state.
            next_action (int): Next action.

        Returns:
            None
        """
        prev_q = self.q[state, action]
        td_target = reward + self.gamma * np.max(self.q[next_state])
        self.q[state, action] += self.alpha * (td_target - prev_q)

    def train(self, env: gymnasium.Env, num_episodes: int):
        """
        Train a Q-learning agent for "num_episodes".

        Args:
            env (gymnasium.Env): Environment to train on.
            num_episodes (int): Number of episodes to train for.

        Returns:
            None
        """
        episode_rewards = np.array([])
        for i in range(num_episodes):
            state, _ = env.reset()
            action = self.select_action(state)
            done = False
            episode_reward = 0
            while not done:
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                next_action = self.select_action(next_state)
                self.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
            episode_rewards = np.append(episode_rewards, episode_reward)
            if i % 1000 == 0:
                print(f"Episode {i} completed, reward: {episode_reward}")
                self.print_q_table()
        print(
            f"Training completed after {num_episodes} episodes. Average reward: {np.mean(episode_rewards)}"
        )

    def print_q_table(self):
        """
        Print the Q-table.

        Returns:
            None
        """
        print("--------- Q-Table ---------")
        print(self.q)
        print("---------------------------")

    def save_q_table(self, filename: str):
        """
        Save the Q-table to a file.

        Args:
            filename (str): Name of the file to save the Q-table to.

        Returns:
            None
        """
        np.savetxt(filename, self.q)

    def load_q_table(self, filename: str):
        """
        Load the Q-table from a file.

        Args:
            filename (str): Name of the file to load the Q-table from.

        Returns:
            None
        """
        self.q = np.loadtxt(filename)
