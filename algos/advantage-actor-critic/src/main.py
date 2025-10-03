import gymnasium
from agent import AdvantageActorCriticAgent


def main():
    env = gymnasium.make("LunarLander-v3")
    agent = AdvantageActorCriticAgent(
        env,
        net_arch={},
    )
    agent.train(10000)
    env.close()


if __name__ == "__main__":
    main()
