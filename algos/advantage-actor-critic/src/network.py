import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, net_arch: list[int]):
        """
        Actor network for Advantage Actor-Critic algorithm. Only works with discrete action spaces.
        Uses ReLU activation function. Outputs a probability distribution over actions via a softmax layer.

        Args:
            observation_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            net_arch (list[int]): List of integers representing the architecture of the network.
        """
        super(ActorNetwork, self).__init__()
        self.layers: list[nn.Module] = []
        if net_arch:
            # add the first linear layer from the obs space
            self.layers.append(nn.Linear(observation_dim, net_arch[0]))
            self.layers.append(nn.ReLU())
            # add the remaining linear layers from the net_arch list
            for i in range(len(net_arch) - 1):
                self.layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
                self.layers.append(nn.ReLU())
            # add the final linear layer from the last layer to the action space, don't apply a softmax, use raw logits
            self.layers.append(nn.Linear(net_arch[-1], action_dim))
        else:
            # defaults to a simple MLP with one hidden layer with 128 units
            self.layers.append(nn.Linear(observation_dim, 128))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(128, action_dim))
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the actor network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(x)


class CriticNetwork(nn.Module):
    def __init__(self, observation_dim: int, net_arch: list[int]):
        """
        Initialize the critic network. Uses ReLU activation function.

        Args:
            observation_dim (int): Dimension of the observation space.
            net_arch (list[int]): List of hidden layer sizes.
        """
        super(CriticNetwork, self).__init__()
        self.layers: list[nn.Module] = []
        if net_arch:
            # add the first linear layer from the obs space
            self.layers.append(nn.Linear(observation_dim, net_arch[0]))
            self.layers.append(nn.ReLU())
            # add the remaining linear layers from the net_arch list
            for i in range(len(net_arch) - 1):
                self.layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
                self.layers.append(nn.ReLU())
            # add the final linear layer from the last layer to the action space
            self.layers.append(nn.Linear(net_arch[-1], 1))
        else:
            # defaults to a simple MLP with one hidden layer with 128 units
            self.layers.append(nn.Linear(observation_dim, 128))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(128, 1))
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(x)
