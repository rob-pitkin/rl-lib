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
        activation_fn: str = "relu",
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
        self, input_dim: int, activation_fn: str = "relu", hidden_dims: list[int] = None
    ):
        super(ValueNetwork, self).__init__()
        if input_dim < 1 or (hidden_dims and any([h < 1 for h in hidden_dims])):
            raise ValueError("All dimension values must be >= 1")
        if activation_fn not in ("relu", "tanh"):
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


class PPOCNNetwork(nn.Module):
    """
    CNN feature extractor combined with a feedforward policy head (actor) and
    feedforward value head (critic) for discrete action spaces

    Attributes:
        input_dim (tuple[int,int]): 2D input size of the network,
            typically the size of the observation space
        output_dim (int): output size of the network, typically the size of the action space
        activation_fn (str): activation function to use between hidden layers
            (can be "relu" or "tanh")
        hidden_dims (list[tuple[int,int]]): list of hidden layer sizes from layer 0 to n
    """

    def __init__(
        self, input_shape=(3, 96, 96), action_space_dim=5, activation_fn="relu"
    ):
        super(PPOCNNetwork, self).__init__()
        self.input_shape = input_shape
        self.action_space_dim = action_space_dim
        if activation_fn not in ("relu", "tanh"):
            raise ValueError("Supported activation functions: 'relu', 'tanh'")
        self.activation_fn = (
            torch.functional.ReLU()
            if activation_fn == "relu"
            else torch.functional.Tanh()
        )
        # go from (3, 96, 96) to (16, 48, 48)
        self.conv1 = torch.nn.Conv2d(
            in_channels=self.input_shape[0],
            out_channels=16,
            kernel_size=8,
            stride=2,
            padding=3,
        )
        # go from (16, 48, 48) to (32, 24, 24)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        # stay at (32, 24, 24)
        self.conv3 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="same"
        )
        flattened_size = None

        # figure out the flattened input size
        with torch.no_grad():
            # create a dummy tensor via unpacking the input shape tuple
            # in the default case, unpacks to torch.zeros(1, 3, 96, 96)
            x = torch.zeros(1, *self.input_shape)
            x = self._get_conv_output(x)
            # x has shape (1, 32, 24, 24), reshape it to have (1, N), get N
            # torch.tensor.view(1, -1) reshapes the tensor to be a 1 x N since -1 means "the rest"
            flattened_size = x.view(1, -1).size(dim=1)

        if flattened_size is None:
            raise RuntimeError("Failed to get the flattened size of the conv net")

        self.fc = torch.nn.Linear(flattened_size, 256)

        # create policy and value heads for the agent, don't have any hidden layers
        self.policy_head = PPONetwork(input_dim=256, output_dim=self.action_space_dim)
        self.value_head = ValueNetwork(input_dim=256)

    def _get_conv_output(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        Forward pass of the PPO network

        Args:
            x (torch.tensor): the input tensor to the network, expects a (N, a, b, c) tensor
            where N is the batch size, a is the # of image channels, b is the image width,
            c is the image height

        Returns:
            tuple(torch.tensor, torch.tensor): the policy head output and the value head output
        """
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.fc(x)
        action_logits = self.policy_head(x)
        value = self.value_head(x)
        return action_logits, value
