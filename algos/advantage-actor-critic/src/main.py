import gymnasium
from agent import AdvantageActorCriticAgent


def main():
    env = gymnasium.make("LunarLander-v3")
    print(
        gymnasium.spaces.utils.flatdim(env.observation_space),
        gymnasium.spaces.utils.flatdim(env.action_space),
    )
    obs_dim = gymnasium.spaces.utils.flatdim(env.observation_space)
    action_dim = gymnasium.spaces.utils.flatdim(env.action_space)
    agent = AdvantageActorCriticAgent(
        env,
        net_arch={},
    )
    agent.train(1)
    env.close()


if __name__ == "__main__":
    main()
