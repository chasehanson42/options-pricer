"""Monte Carlo pricing for European options, with antithetic variates and a
control variate (terminal spot price, whose risk-neutral mean is known)."""
import math

import numpy as np


def price(S, K, T, r, sigma, option_type="call", n_paths=100_000, q=0.0,
          antithetic=True, control_variate=True, seed=None):
    """Returns (price_estimate, standard_error)."""
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * math.sqrt(T) * z
    ST = S * np.exp(drift + diffusion)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    disc = math.exp(-r * T)
    discounted = disc * payoff

    if control_variate:
        control = disc * ST
        control_mean = S * math.exp(-q * T)  # E[disc * ST] under risk-neutral measure
        var = np.var(control)
        beta = np.cov(discounted, control)[0, 1] / var if var > 0 else 0.0
        discounted = discounted - beta * (control - control_mean)

    n = len(discounted)
    estimate = discounted.mean()
    stderr = discounted.std(ddof=1) / math.sqrt(n)
    return float(estimate), float(stderr)
