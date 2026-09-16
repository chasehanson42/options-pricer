"""Cox-Ross-Rubinstein binomial tree pricing for European and American options."""
import math

import numpy as np


def price(S, K, T, r, sigma, n=200, option_type="call", exercise="european", q=0.0):
    """Price an option via an n-step CRR binomial tree.

    exercise: 'european' or 'american'.
    """
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        raise ValueError("Invalid inputs: risk-neutral probability outside (0, 1)")

    # Terminal spot prices, highest number of up-moves first.
    ups = np.arange(n, -1, -1)
    downs = np.arange(0, n + 1)
    terminal_prices = S * u ** ups * d ** downs

    if option_type == "call":
        values = np.maximum(terminal_prices - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - terminal_prices, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    for step in range(n - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if exercise == "american":
            ups = np.arange(step, -1, -1)
            downs = np.arange(0, step + 1)
            node_prices = S * u ** ups * d ** downs
            if option_type == "call":
                intrinsic = np.maximum(node_prices - K, 0.0)
            else:
                intrinsic = np.maximum(K - node_prices, 0.0)
            values = np.maximum(values, intrinsic)
        elif exercise != "european":
            raise ValueError("exercise must be 'european' or 'american'")

    return float(values[0])
