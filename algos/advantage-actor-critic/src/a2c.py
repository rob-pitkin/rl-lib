import gymnasium
from agent import AdvantageActorCriticAgent


def main():
    env = gymnasium.make("LunarLander-v3")
    agent = AdvantageActorCriticAgent(
        env,
        net_arch={"actor": [128, 128], "critic": [128, 128]},
        update_frequency=128,
        lr=3e-4,
    )
    agent.train(
        2000000,
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
