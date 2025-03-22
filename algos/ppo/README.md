# PPO
 
```ppo``` is an implementation of Proximal Policy Optimization from scratch with Python and PyTorch. The goal of this project is to create a simple implementation of PPO that runs on CPU and operates within discrete/enumerable action spaces (low dimension).

## Usage
To train the agent, run the following command:
```
python main.py
```

Use the main.py file as a template for how to train and evaluate the agent. You can modify the hyperparameters, the environment, and the neural network architecture to suit your needs.

## Installation
```
conda create -n env_name
conda activate env_name
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install pygame
conda install gymnasium
pip install box2d
```

## Motivation
The core motivation of this repo is to demonstrate implementing a state-of-the-art actor-critic DRL method. There's little to no emphasis on optimization, GPU usage, or customization since those concepts are mainly driven by the use-case and should ideally be tuned to maximize performance. That said, I try to enable different environments via the network.py file, which defines the policy and value network options available for users. The default ones are for discrete observation and action spaces (simplest case), but will/are extended to also handle 2D input spaces (images, for ex), which uses convolutional architectures for both policy and value networks.