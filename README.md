# optpricer

A small options pricing library implementing three independent methods, so
they can be checked against each other:

- `optpricer.black_scholes` — closed-form European pricing + Greeks (delta,
  gamma, vega, theta, rho).
- `optpricer.binomial` — Cox-Ross-Rubinstein binomial tree, supports both
  European and American exercise.
- `optpricer.monte_carlo` — Monte Carlo simulation with antithetic variates
  and a control variate (terminal spot price) for variance reduction.

## Usage

```python
from optpricer import black_scholes, binomial, monte_carlo

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

black_scholes.price(S, K, T, r, sigma, "call")
black_scholes.greeks(S, K, T, r, sigma, "call")

binomial.price(S, K, T, r, sigma, n=500, option_type="put", exercise="american")

monte_carlo.price(S, K, T, r, sigma, option_type="call", n_paths=200_000)
```

## Run

```bash
pip install -r requirements.txt
pytest
python demo.py
```

`demo.py` prints a price comparison and writes `demo_output.png`, showing
Monte Carlo convergence (with/without variance reduction) and the Greeks
curves vs. spot price.
