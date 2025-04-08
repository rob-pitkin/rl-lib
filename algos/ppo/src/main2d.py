from network import PPOCNNetwork
from ppo import PPO
import gymnasium as gym
import torch.optim


def main():
    policy_network_2d = PPOCNNetwork()
    save_and_load_path = "algos/ppo/src/ppo_cnn.pt"
    ppo_agent = PPO(
        env_id="CarRacing-v2",
        buffer_capacity=256,
        cnn_net=policy_network_2d,
    )
    ppo_agent.train(
        cnn_save_path=save_and_load_path,
        num_episodes=5000,
        batch_size=256,
    )

    ppo_agent.load_model_2d(save_and_load_path)
    ppo_agent.eval_model(1)


if __name__ == "__main__":
    main()
