from network import PPONetwork, ValueNetwork
from ppo import PPO
import gymnasium as gym
import torch.optim


def main():
    policy_network = PPONetwork(8, 4, "relu", [64])
    value_network = ValueNetwork(8, "relu", [64])
    ppo_agent = PPO(
        env_id="LunarLander-v3",
        buffer_capacity=256,
        policy_net=policy_network,
        value_net=value_network,
    )
    ppo_agent.train(
        policy_save_path="algos/ppo/src/ppo_policy.pt",
        value_save_path="algos/ppo/src/ppo_value.pt",
        num_episodes=5000,
        batch_size=256,
    )

    ppo_agent.load_model("algos/ppo/src/ppo_policy.pt", "algos/ppo/src/ppo_value.pt")
    ppo_agent.eval_model(1)


if __name__ == "__main__":
    main()
