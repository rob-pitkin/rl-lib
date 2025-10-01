import gymnasium
from sarsa import Sarsa


def main():
    """
    Main function to run the SARSA algorithm.

    Returns:
        None
    """
    env = gymnasium.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        render_mode="rgb_array",
        is_slippery=True,
        success_rate=3.0 / 4.0,
        reward_schedule=(1, 0, 0),
    )
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    agent = Sarsa(num_states, num_actions, gamma=0.99, alpha=0.1, epsilon=0.1)
    agent.train(env, num_episodes=100000)
    agent.save_q_table("algos/sarsa/q_table.txt")


if __name__ == "__main__":
    main()
