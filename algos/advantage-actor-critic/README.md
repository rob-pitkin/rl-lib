# Advantage Actor-Critic (A2C/A3C)

## Overview

This implementation provides both **Advantage Actor-Critic (A2C)** and **Asynchronous Advantage Actor-Critic (A3C)** algorithms. These are policy gradient methods that combine the benefits of value-based and policy-based approaches in reinforcement learning.

**Key Features:**
- **A2C**: Synchronous implementation that collects experiences and updates networks in batches
- **A3C**: Asynchronous implementation using multiple worker threads for parallel experience collection
- **Generalized Advantage Estimation (GAE)**: Configurable λ parameter for bias-variance tradeoff in advantage calculation
- **Discrete Action Spaces**: Currently supports discrete action environments
- **Flexible Network Architecture**: Customizable neural network architectures for both actor and critic networks

## Algorithm Details

### Advantage Actor-Critic (A2C)
A2C is a synchronous implementation where:
1. An actor network outputs action probabilities (policy)
2. A critic network estimates state values
3. Advantages are calculated using GAE to reduce variance
4. Both networks are updated simultaneously using collected experience batches

### Asynchronous Advantage Actor-Critic (A3C)
A3C extends A2C with asynchronous training:
1. Multiple worker threads collect experiences in parallel environments
2. Each worker maintains local copies of actor and critic networks
3. Workers periodically synchronize gradients with global networks
4. This improves sample efficiency and training stability

## Files Structure

```
advantage-actor-critic/
├── src/
│   ├── agent.py          # Main AdvantageActorCriticAgent class
│   ├── network.py        # ActorNetwork and CriticNetwork definitions
│   ├── a2c.py           # A2C training script
│   ├── a3c.py           # A3C training script
│   ├── a2c_actor.pt     # Trained A2C actor model
│   ├── a2c_critic.pt    # Trained A2C critic model
│   ├── a3c_actor.pt     # Trained A3C actor model
│   └── a3c_critic.pt    # Trained A3C critic model
└── README.md
```

## Usage

### A2C Training

```python
import gymnasium
from agent import AdvantageActorCriticAgent

# Create environment
env = gymnasium.make("LunarLander-v3")

# Initialize agent
agent = AdvantageActorCriticAgent(
    env,
    net_arch={"actor": [128, 128], "critic": [128, 128]},
    update_frequency=128,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95
)

# Train the agent
agent.train(2000000)

# Save trained models
agent.save_model("a2c_actor.pt", "a2c_critic.pt")

# Evaluate the agent
agent.eval_model(5)
```

### A3C Training

```python
import gymnasium
from agent import AdvantageActorCriticAgent
import threading

def run_a3c():
    num_threads = 4
    num_steps = 1000000
    env_name = "LunarLander-v3"

    main_env = gymnasium.make(env_name)
    agent = AdvantageActorCriticAgent(
        main_env,
        net_arch={"actor": [128, 128], "critic": [128, 128]},
        update_frequency=128,
        lr=3e-4,
        gae_lambda=0.95
    )

    lock = threading.Lock()
    threads = []

    # Create worker threads
    for worker_id in range(num_threads):
        worker_env = gymnasium.make(env_name)
        thread = threading.Thread(
            target=agent.worker_train,
            args=(num_steps, worker_id, lock, worker_env),
            name=f"Worker-{worker_id}"
        )
        threads.append(thread)

    # Start and join threads
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return agent
```

## Parameters

### AdvantageActorCriticAgent

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env` | gymnasium.Env | Required | The environment to interact with |
| `net_arch` | dict[str, list[int]] | Required | Network architecture for actor and critic |
| `update_frequency` | int | 32 | Number of steps before updating networks |
| `lr` | float | 0.0007 | Learning rate for both networks |
| `gamma` | float | 0.99 | Discount factor for future rewards |
| `gae_lambda` | float | 1.0 | Lambda parameter for GAE (1.0 = Monte Carlo, 0.0 = TD) |
| `seed` | int \| None | None | Random seed for reproducibility |

### Network Architecture

The `net_arch` parameter expects a dictionary with "actor" and "critic" keys:

```python
net_arch = {
    "actor": [128, 128],    # Two hidden layers with 128 units each
    "critic": [64, 64, 32]  # Three hidden layers with 64, 64, and 32 units
}
```

If empty lists are provided, defaults to a single 128-unit hidden layer.

## Dependencies

- **PyTorch**: Neural network implementation and optimization
- **Gymnasium**: Environment interface
- **NumPy**: Numerical computations
- **Threading**: For A3C parallel execution

## Key Methods

### Training
- `train(num_steps)`: Standard A2C training loop
- `worker_train(num_steps, worker_id, lock, env)`: A3C worker training function

### Model Management
- `save_model(actor_filepath, critic_filepath)`: Save trained models
- `load_model(actor_filepath, critic_filepath)`: Load pre-trained models
- `eval_model(num_episodes)`: Evaluate agent performance with rendering

## Running the Examples

### A2C
```bash
python src/a2c.py
```

### A3C
```bash
python src/a3c.py
```

Both scripts will train on the LunarLander-v3 environment and save the trained models.

## Notes

- Currently supports **discrete action spaces** only
- Uses **categorical distribution** for action sampling
- Implements **Generalized Advantage Estimation (GAE)** for variance reduction
- A3C uses **gradient accumulation** rather than parameter averaging for simplicity
- Models are saved as complete PyTorch objects (not just state dictionaries)
