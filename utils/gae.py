import torch


def calculate_gae(
    rewards: torch.tensor,
    values: torch.tensor,
    next_values: torch.tensor,
    gamma: float,
    lam: float,
):
    """
    Computes the Generalized Advantage Estimation. Uses the following formula:
    A_t = SUM_{l=0}^{inf}(gamma * lambda)^l * d_{t+l} where d_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    is the TD error.

    Args:
        rewards (torch.tensor): (Nx1) tensor of rewards
        values (torch.tensor): (Nx1) tensor of values
        next_values (torch.tensor): (Nx1) tensor of values
        gamma (float): discount factor
        lam (float): lambda hyperparameter
    """
    td_errors = rewards + gamma * next_values - values
    adv = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(td_errors))):
        # in the t = N case, gae = td_errors[N]
        # in the t = N - 1 case, gae = gamma * lam * td_errors[N] + td_errors[N - 1]
        # in the t = N - 2 case, gae = gamma * lam * (gamma * lam * td_errors[N] + td_errors[N - 1]) + td_errors[N - 2]
        gae = gae * gamma * lam + td_errors[t]
        adv[t] = gae
    return adv
