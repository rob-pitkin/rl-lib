# RL-Lib

## About

RL-Lib is a collection of RL implementations such as algos, helper functions, and more. The purpose of this parent repo is to collect and organize reinforcement learning agents and methods I implement from scratch as self-interest projects or for flexible usage in other projects. While foundational methods and algorithms are the primary focus of this repo (ex: DQN, PPO, A2C/A3C (soon)), I intend to implement any interesting algorithm or method I come across as an understanding exercise. Feel free to use this repo for inspiration, self-projects, or anything else that comes to mind :)


## Installation/Requirements
The general requirements for algorithms in this repo are as follows:
1. PyTorch (for network creation, training, etc.)
2. Gymnasium (for RL environments)
3. PyGame (for environment visualization)
4. Box2D (for environment visualization)

Suggested installation instructions are as follows:
```
conda create -n <your_env_name>
conda activate <your_env_name>
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install pygame
conda install gymnasium
pip install box2d
```

## Running Algorithms
For ease of use and compatibility testing, scripts are provided in `./algos/scripts` for each algorithm to run its main file. Each script has the title format `run_<algorithm_name>.sh`. If a script isn't provided, navigate to the algorithm's folder in `./algos` to find more information on running it via the README.

```
# ex: running DQN, from the root of this repo
./algos/scripts/run_dqn.sh
```