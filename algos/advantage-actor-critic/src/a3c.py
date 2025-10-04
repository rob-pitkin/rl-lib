import gymnasium
from agent import AdvantageActorCriticAgent
import threading


def run_a3c() -> AdvantageActorCriticAgent:
    # config
    num_threads = 4
    num_steps = 1000000
    env_name = "LunarLander-v3"

    main_env = gymnasium.make(env_name)
    net_arch = {"actor": [128, 128], "critic": [128, 128]}

    agent = AdvantageActorCriticAgent(
        main_env, net_arch=net_arch, update_frequency=128, lr=3e-4, gae_lambda=0.95
    )

    lock = threading.Lock()

    threads = []
    for worker_id in range(num_threads):
        worker_env = gymnasium.make(env_name)
        thread = threading.Thread(
            target=agent.worker_train,
            args=(num_steps, worker_id, lock, worker_env),
            name=f"Worker-{worker_id}",
        )
        threads.append(thread)

    # start threads
    for thread in threads:
        thread.start()

    # wait for threads to finish
    for thread in threads:
        thread.join()

    print("Finished training")

    return agent


if __name__ == "__main__":
    agent = run_a3c()

    agent.save_model(
        "algos/advantage-actor-critic/src/a3c_actor.pt",
        "algos/advantage-actor-critic/src/a3c_critic.pt",
    )

    agent.eval_model(1)
