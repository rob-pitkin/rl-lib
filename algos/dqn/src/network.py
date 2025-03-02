import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Feedforward MLP for the QNetwork (and target network)

    Attributes:
        input_dim (int): input size of the network, typically the size of the observation space
        output_dim (int): output size of the network, typically the size of the action space
        hidden_dims (List): list of hidden layer sizes from layer 0 to n
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_fn: str,
        hidden_dims: list[int] = None,
    ):
        super(QNetwork, self).__init__()
        if (
            input_dim < 1
            or (hidden_dims and any([h < 1 for h in hidden_dims]))
            or output_dim < 1
        ):
            raise ValueError("All dimension values must be >= 1")
        if activation_fn != "relu" and activation_fn != "tanh":
            raise ValueError("Supported activation functions: 'relu', 'tanh'")
        layer_sizes = (
            [input_dim] + hidden_dims + [output_dim]
            if hidden_dims
            else [input_dim, output_dim]
        )

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(nn.ReLU() if activation_fn == "relu" else nn.Tanh())
        self.layers.pop()
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the Q value network

        Args:
            x (torch.tensor): the input tensor to the network

        Returns:
            torch.tensor: a tensor with the values for each output
        """
        return self.fc(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q Network for the DQN agent
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_fn: str,
        hidden_dims: list[int] = None,
    ):
        super(DuelingQNetwork, self).__init__()
        if (
            input_dim < 1
            or (hidden_dims and any([h < 1 for h in hidden_dims]))
            or output_dim < 1
        ):
            raise ValueError("All dimension values must be >= 1")
        if activation_fn != "relu" and activation_fn != "tanh":
            raise ValueError("Supported activation functions: 'relu', 'tanh'")
        layer_sizes = [input_dim] + hidden_dims if hidden_dims else [input_dim]

        self.feature_layers = []
        for i in range(len(layer_sizes) - 1):
            self.feature_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.feature_layers.append(
                nn.ReLU() if activation_fn == "relu" else nn.Tanh()
            )

        self.fc = nn.Sequential(*self.feature_layers)

        # Add the value and advantage streams
        self.value_stream = nn.Linear(layer_sizes[-1], 1)
        self.advantage_stream = nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the Dueling Q value network

        Args:
            x (torch.tensor): the input tensor to the network

        Returns:
            torch.tensor: a tensor with the values for each output
        """
        features = self.fc(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean())
        return q_values
