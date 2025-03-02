from policyGradientAgent import PolicyGradientAgent
import gymnasium as gym
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train a policy gradient agent.")
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Gymnasium environment id"
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=2000, help="Number of training episodes"
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--gamma", "-g", type=float, default=0.999, help="Discount factor"
    )
    parser.add_argument(
        "--hidden_dim", "-hd", type=int, default=64, help="Hidden layer size"
    )
    parser.add_argument(
        "--state_dim", "-sd", type=int, default=4, help="Observation space dimension"
    )
    parser.add_argument(
        "--action_dim", "-ad", type=int, default=2, help="Action space dimension"
    )
    parser.add_argument(
        "--num_hidden_layers",
        "-hl",
        type=int,
        default=1,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--activation",
        "-a",
        type=str,
        default="relu",
        help="Activation function for hidden layers",
    )
    parser.add_argument("--shared", "-s", type=bool, help="Use shared weights")
    args = parser.parse_args()

    env = gym.make(args.env)
    num_episodes = args.episodes
    agent = PolicyGradientAgent(
        learning_rate=args.learning_rate,
        discount_factor=args.gamma,
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_layers=args.num_hidden_layers,
        activation=args.activation,
        shared=args.shared,
    )
    agent.train(env, num_episodes, True)
    show_env = gym.make(args.env, render_mode="human")
    agent.evaluate_agent(show_env, 3)


if __name__ == "__main__":
    main()
