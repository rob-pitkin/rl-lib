# vanilla-policy-gradient
 A PyTorch implementation of vanilla policy gradient with a value function baseline estimate. Code is based on [Lecture 3](https://www.dropbox.com/scl/fi/htn2r6ac807oluoxeihdt/l3-policy-gradient-and-advantage-estimation.pdf?rlkey=26hsbd5qvthb8ozq53vdfjrr4&e=1&dl=0) of Foundations of Deep RL by Pieter Abbeel.


## Installation
```conda env create --name <name> -f environment.yml```

```conda activate <name>```

## Usage
To train the agent and visualize the results, run the following command:
```python train.py```
or with custom arguments:
```python train.py --env CartPole-v1 --gamma 0.999 --learning_rate 0.001 --episodes 2000 --state_dim 4 --action_dim 2 --hidden_dim 64 --num_hidden_layers 0 --activation relu --shared False```

The Agent.py module can also be used independently.

### Command Line Arguments
- `--env`: Name of the environment (default: `CartPole-v1`)
- `--gamma`, `-g`: Discount factor (default: `0.999`)
- `--learning_rate`, `-lr`: Learning rate (default: `0.001`)
- `--episodes`, `-e`: Number of episodes to train (default: `2000`)
- `--state_dim`, `-sd`: Dimension of the state space (default: `4`)
- `--action_dim`, `-ad`: Dimension of the action space (default: `2`)
- `--hidden_dim`, `-hd`: Dimension of the hidden layer (default: `64`)
- `--num_hidden_layers`, `-hl`: Number of hidden layers (default: `0`)
- `--activation`, `-a`: Activation function (default: `relu`) (options: `relu`, `tanh`)
- `--shared`, `-s`: Use shared weights for policy and value function (default: `False`)

## Notes
- The code only runs on CPU for now. GPU support can be added by moving the model and tensors to the GPU.
- The networks only support discrete action spaces for now.