import gymnasium
from agent import AdvantageActorCriticAgent


def main():
    env = gymnasium.make("LunarLander-v3")
    agent = AdvantageActorCriticAgent(env, net_arch={}, update_frequency=100)
    agent.train(
        1000000,
    )
    agent.save_model(
        "algos/advantage-actor-critic/src/a2c_actor.pt",
        "algos/advantage-actor-critic/src/a2c_critic.pt",
    )

    agent.load_model(
        "algos/advantage-actor-critic/src/a2c_actor.pt",
        "algos/advantage-actor-critic/src/a2c_critic.pt",
    )
    agent.eval_model(1)
    env.close()


if __name__ == "__main__":
    main()
