import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium import Env


class PolicyNetwork(nn.Module):
    """
    A simple feedforward neural network for the policy

    Attributes:
        state_dim (int): the dimension of the state space
        hidden_dim (int): the dimension of the hidden layer
        hidden_layers (int): the number of hidden layers
        input_layer (nn.Linear): the first feedforward layer
        layers (List): the hidden layers (includes the input layer)
        output_layer (nn.Linear): the output layer
        action_dim (int): the dimension of the action space
        activation (nn.ReLU): the activation function
        softmax (nn.Softmax): the softmax function
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        hidden_layers: int = 0,
        action_dim: int = 2,
        activation: str = "relu",
    ):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        layers = []
        self.activation_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.input_layer = nn.Linear(self.state_dim, self.hidden_dim)
        layers.append(self.input_layer)
        layers.append(self.activation_fn)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self.activation_fn)
        self.output_layer = nn.Linear(self.hidden_dim, self.action_dim)
        layers.append(self.output_layer)
        self.fc = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the policy network

        Args:
            x (torch.tensor): the input state

        Returns:
            torch.Tensor: the output probabilities of the actions
        """
        x = self.fc(x)
        return self.softmax(x)

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        Selects an action given the current state

        Args:
            state (torch.tensor): the current state

        Returns:
            int: the action to take
            torch.Tensor: the log probability of the action
        """
        output_probs = self.forward(state)
        dist = torch.distributions.Categorical(output_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class BaselineNetwork(nn.Module):
    """
    A simple feedforward neural network for baseline estimation

    Attributes:
        state_dim (int): the dimension of the state space
        hidden_dim (int): the dimension of the hidden layer
        hidden_layers (int): the number of hidden layers
        input_layer (nn.Linear): the first feedforward layer
        layers (List): the hidden layers (includes the input layer)
        output_layer (nn.Linear): the output layer
        value_dim (int): the dimension of the value space
        activation (nn.ReLU): the activation function
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        hidden_layers: int = 0,
        value_dim: int = 1,
        activation: str = "relu",
    ):
        super(BaselineNetwork, self).__init__()
        self.state_dim = state_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        layers = []
        self.input_layer = nn.Linear(self.state_dim, self.hidden_dim)
        self.activation_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()
        layers.append(self.input_layer)
        layers.append(self.activation_fn)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self.activation_fn)
        self.output_layer = nn.Linear(self.hidden_dim, self.value_dim)
        layers.append(self.output_layer)
        self.fc = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the baseline network

        Args:
            x (torch.tensor): the input state

        Returns:
            torch.Tensor: the value of the state
        """
        x = self.fc(x)
        return x


class SharedNetwork(nn.Module):
    """
    A simple feedforward neural network for vanilla policy gradient with two heads,
    one for the policy and one for the baseline value function.

    Attributes:
        state_dim (int): the dimension of the state space
        hidden_dim (int): the dimension of the hidden layer
        hidden_layers (int): the number of hidden layers
        action_dim (int): the dimension of the action space
        value_dim (int): the dimension of the value space
        input_layer (nn.Linear): the first feedforward layer
        layers (List): the hidden layers (includes the input layer)
        output_layer (nn.Linear): the output layer
        policy_head (nn.Linear): the policy head
        value_head (nn.Linear): the value head
        activation (nn.ReLU): the activation function
        softmax (nn.Softmax): the softmax function
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        hidden_layers: int = 0,
        action_dim: int = 2,
        value_dim: int = 1,
        activation: str = "relu",
    ) -> None:
        super(SharedNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.value_dim = value_dim
        layers = []
        self.input_layer = nn.Linear(self.state_dim, self.hidden_dim)
        self.activation_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()
        layers.append(self.input_layer)
        layers.append(self.activation_fn)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self.activation_fn)
        self.fc = nn.Sequential(*layers)
        self.policy_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.value_head = nn.Linear(self.hidden_dim, self.value_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x) -> None:
        """
        Forward pass of the shared network

        Args:
            x (torch.tensor): the input state

        Returns:
            torch.Tensor: the output probabilities of the actions
            torch.Tensor: the value of the state
        """
        x = self.fc(x)
        action_probs = self.softmax(self.policy_head(x))
        state_value = self.value_head(x)
        return action_probs, state_value

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        Selects an action given the current state

        Args:
            state (torch.tensor): the current state

        Returns:
            int: the action to take
            torch.Tensor: the log probability of the action
        """
        output_probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(output_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class PolicyGradientAgent:
    """
    Vanilla policy gradient agent with a value function baseline estimation

    Attributes:
        policy (PolicyNetwork): the policy network
        baseline (BaselineNetwork): the baseline network
        learning_rate (float): the learning rate
        discount_factor (float): the discount factor
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.999,
        state_dim: int = 4,
        action_dim: int = 2,
        value_dim: int = 1,
        hidden_dim: int = 64,
        hidden_layers: int = 0,
        activation: str = "relu",
        shared: bool = False,
    ) -> None:
        self.shared = shared
        self.policy = None
        self.baseline = None
        if self.shared:
            self.network = SharedNetwork(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                action_dim=action_dim,
                value_dim=value_dim,
                activation=activation,
            )
        else:
            self.policy = PolicyNetwork(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                action_dim=action_dim,
                activation=activation,
            )
            self.baseline = BaselineNetwork(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                value_dim=value_dim,
                activation=activation,
            )
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor

    def calculate_return(
        self, rewards: list[torch.Tensor], discount_factor: float
    ) -> torch.Tensor:
        """
        Calculates the discounted returns from a list of rewards

        Args:
            rewards (List): The list of rewards from an episode
            discount_factor (float): The discount factor

        Returns:
            torch.Tensor: a tensor of discounted returns
        """
        returns = deque()
        running_return = 0
        for i, r in enumerate(reversed(rewards)):
            running_return = r + (discount_factor * running_return)
            returns.appendleft(running_return)
        return torch.tensor(returns, dtype=torch.float32)

    def calculate_policy_loss(
        self,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        baseline_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the policy loss

        Args:
            log_probs (torch.Tensor): the log probabilities of the actions
            returns (torch.Tensor): the returns from the episode
            baseline_values (torch.Tensor): the baseline values

        Returns:
            torch.Tensor: the policy loss
        """
        loss = -torch.sum(
            log_probs * (returns - baseline_values).detach()
        )  # detach the baseline values to prevent backpropagation through the baseline network
        return loss

    def calculate_baseline_loss(
        self, returns: torch.Tensor, baseline_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the baseline loss

        Args:
            returns (torch.Tensor): the returns from the episode
            baseline_values (torch.Tensor): the baseline values

        Returns:
            torch.Tensor: the baseline loss
        """
        loss_fn = nn.MSELoss()
        return loss_fn(baseline_values, returns)

    def train(self, env: Env, num_episodes: int, verbose: bool = False) -> None:
        """
        Trains the agent on an environment

        Args:
            env (gym.Env): the environment to train on
            num_episodes (int): the number of episodes to train for
            verbose (bool): whether to print progress

        Returns:
            None
        """
        optimizer = None
        policy_optim = None
        baseline_optim = None
        if self.shared:
            optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        else:
            policy_optim = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
            baseline_optim = optim.Adam(
                self.baseline.parameters(), lr=self.learning_rate
            )

        for i in range(num_episodes):
            # Initialize lists for tracking rewards, action probabilites, and baseline values
            rewards = []
            log_probs = []
            baseline_values = []

            # Reset the environment
            obs, info = env.reset()
            episode_over = False

            while not episode_over:
                # Forward passes of the policy and baseline network
                state = torch.tensor(obs, dtype=torch.float32)
                if self.shared:
                    action, log_prob = self.network.select_action(state)
                    baseline_value = self.network.forward(state)[1]
                else:
                    action, log_prob = self.policy.select_action(state)
                    baseline_value = self.baseline.forward(state)

                # Taking action a in the environment
                obs, reward, terminated, truncated, info = env.step(action)

                # recording the results
                rewards.append(reward)
                log_probs.append(log_prob)
                baseline_values.append(baseline_value)

                episode_over = terminated or truncated

            # Calculate the returns
            returns = self.calculate_return(rewards, self.discount_factor)

            # Normalize the returns
            returns = (returns - torch.mean(returns)) / (torch.std(returns) + 1e-8)

            # Convert the lists to tensors
            log_probs = torch.stack(log_probs).squeeze()
            baseline_values = torch.stack(baseline_values).squeeze()

            if self.shared:
                # Calculate the policy and baseline loss
                optimizer.zero_grad()
                policy_loss = self.calculate_policy_loss(
                    log_probs, returns, baseline_values
                )
                baseline_loss = self.calculate_baseline_loss(returns, baseline_values)
                loss = policy_loss + baseline_loss
                loss.backward()
                optimizer.step()
            else:
                # Calculate policy loss and backprop
                policy_optim.zero_grad()
                policy_loss = self.calculate_policy_loss(
                    log_probs, returns, baseline_values
                )
                policy_loss.backward()
                policy_optim.step()

                # Calculate baseline loss and backprop
                baseline_optim.zero_grad()
                baseline_loss = self.calculate_baseline_loss(returns, baseline_values)
                baseline_loss.backward()
                baseline_optim.step()

            # Print progress every 1000 episodes
            if verbose:
                if i % 1000 == 0:
                    print(
                        f"Episode {i}/{num_episodes}, Policy Loss: {policy_loss.item()}, Baseline Loss: {baseline_loss.item()}"
                    )

    def evaluate_agent(self, env: Env, num_episodes: int) -> None:
        """
        Evaluates the agent on an environment

        Args:
            env (gym.Env): the environment to evaluate on
            num_episodes (int): the number of episodes to evaluate for

        Returns:
            None
        """
        with torch.no_grad():
            total_rewards = 0
            for _ in range(num_episodes):
                obs, info = env.reset()
                episode_over = False
                while not episode_over:
                    state = torch.tensor(obs, dtype=torch.float32)
                    if self.shared:
                        action, _ = self.network.select_action(state)
                    else:
                        action, _ = self.policy.select_action(state)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_rewards += reward
                    episode_over = terminated or truncated
            print(
                f"Average reward after {num_episodes} episodes: {total_rewards / num_episodes}"
            )
