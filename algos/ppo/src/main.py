from ppo import PPO
import gymnasium as gym


def main():
    env = gym.make("LunarLander-v3")
    ppo_agent = PPO(
        env,
        net_arch={"policy": [256, 256], "value": [256, 256]},
        rollout_steps=512,
    )
    # ppo_agent.load_model("algos/ppo/src/ppo_policy.pt", "algos/ppo/src/ppo_value.pt")
    ppo_agent.train(1000000)

    ppo_agent.save_model("algos/ppo/src/ppo_policy.pt", "algos/ppo/src/ppo_value.pt")

    ppo_agent.eval_model(1)


if __name__ == "__main__":
    main()
