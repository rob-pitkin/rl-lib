# dqn-scratch
 
```dqn-scratch``` is an implementation of Deep Q-Learning from scratch with Python and PyTorch. The goal of this project is to provide a simple, well-documented implementation of DQN that is easy to understand and modify.

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

## Implementation Steps
1. Define the Q-Network class [DONE]
2. Define the Replay Buffer class [DONE]
3. Define the Agent class [DONE]
4. Define the training loop [DONE]
5. Training and evaluation [DONE]
6. Define the main function [DONE]