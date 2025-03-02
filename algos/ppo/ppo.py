import torch


class PPO:
    """
    PPO agent class

    Attributes:
        policy_net (PPONetwork): the policy network used for picking actions
    """

    def update_params(
        self,
        optimizer: torch.optim,
        states: torch.tensor,
        actions: torch.tensor,
        prev_log_probs: torch.tensor,
        returns: torch.tensor,
        advantages: torch.tensor,
        epsilon: float = 0.2,
        epochs: float = 10,
    ) -> None:
        """ """
        for _ in range(epochs):
            # Compute the log probs of each taken action
            new_log_probs = torch.log(self.policy_net(states).gather(1, actions))
            # use log rules: log(a) - log(b) = log(a/b), exp(log(a/b)) = a/b
            prob_ratio = torch.exp(new_log_probs - prev_log_probs)
            # compute the PPO objective fn
            loss = -torch.min(
                prob_ratio * advantages,
                torch.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages,
            )

            # backprop the update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
