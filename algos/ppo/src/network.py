import torch
import torch.nn as nn


class PPONetwork(nn.Module):
    """
    Feedforward MLP for the PPO Policy network (actor)

    Attributes:
        input_dim (int): input size of the network, typically the size of the observation space
        output_dim (int): output size of the network, typically the size of the action space
        activation_fn (str): activation function to use between hidden layers (can be "relu" or "tanh")
        hidden_dims (List): list of hidden layer sizes from layer 0 to n
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_fn: str,
        hidden_dims: list[int] = None,
    ):
        super(PPONetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.hidden_dims = hidden_dims
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
        self.layers.pop()  # pop off the last relu layer
        self.layers.append(
            nn.Softmax(dim=1)
        )  # add a softmax layer for action selection
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the PPO network

        Args:
            x (torch.tensor): the input tensor to the network

        Returns:
            torch.tensor: a tensor with the values for each output
        """
        return self.fc(x)


class ValueNetwork(nn.Module):
    """
    Feedforward MLP for the PPO Value network (critic)

    Attributes:
        input_dim (int): input size of the network, typically the size of the observation space
        activation_fn (str): activation function to use between hidden layers (can be "relu" or "tanh")
        hidden_dims (List): list of hidden layer sizes from layer 0 to n
    """

    def __init__(
        self, input_dim: int, activation_fn: str, hidden_dims: list[int] = None
    ):
        super(ValueNetwork, self).__init__()
        if input_dim < 1 or (hidden_dims and any([h < 1 for h in hidden_dims])):
            raise ValueError("All dimension values must be >= 1")
        if activation_fn != "relu" and activation_fn != "tanh":
            raise ValueError("Supported activation functions: 'relu', 'tanh'")
        layer_sizes = [input_dim] + hidden_dims + [1] if hidden_dims else [input_dim, 1]

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(nn.ReLU() if activation_fn == "relu" else nn.Tanh())
        self.layers.pop()  # pop off the last relu layer
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the PPO network

        Args:
            x (torch.tensor): the input tensor to the network

        Returns:
            torch.tensor: a tensor with the values for each output
        """
        return self.fc(x)
