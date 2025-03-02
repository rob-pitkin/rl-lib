from dqn_agent import DQN
import gymnasium as gym
import torch.optim


def main():
    dqn = DQN(8, 4, "relu", "LunarLander-v2", [32], network_type="dueling")
    dqn.train(5000, save_path="algos/dqn/src/dqn.pt")

    dqn.loadModel("algos/dqn/src/dqn.pt")
    dqn.evalModel(1)


if __name__ == "__main__":
    main()
