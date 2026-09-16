"""Price a European call three ways, then visualize MC convergence and Greeks."""
import numpy as np
import matplotlib.pyplot as plt

from optpricer import black_scholes, binomial, monte_carlo

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

bs_price = black_scholes.price(S, K, T, r, sigma, "call")
bt_price = binomial.price(S, K, T, r, sigma, n=500, option_type="call")
mc_price, mc_stderr = monte_carlo.price(S, K, T, r, sigma, n_paths=200_000, seed=1)

print(f"Black-Scholes price:  {bs_price:.4f}")
print(f"Binomial tree price:  {bt_price:.4f}")
print(f"Monte Carlo price:    {mc_price:.4f}  (stderr {mc_stderr:.4f})")

# --- Convergence of MC estimate vs number of paths, with/without variance reduction ---
path_counts = np.unique(np.logspace(2, 5.3, 25).astype(int))
plain, plain_err, reduced, reduced_err = [], [], [], []
for n in path_counts:
    p, e = monte_carlo.price(S, K, T, r, sigma, n_paths=int(n),
                              antithetic=False, control_variate=False, seed=1)
    plain.append(p)
    plain_err.append(e)
    p2, e2 = monte_carlo.price(S, K, T, r, sigma, n_paths=int(n), seed=1)
    reduced.append(p2)
    reduced_err.append(e2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.errorbar(path_counts, plain, yerr=plain_err, label="Plain MC", capsize=2)
ax.errorbar(path_counts, reduced, yerr=reduced_err, label="Antithetic + control variate", capsize=2)
ax.axhline(bs_price, color="black", linestyle="--", label="Black-Scholes (exact)")
ax.set_xscale("log")
ax.set_xlabel("Number of paths")
ax.set_ylabel("Estimated price")
ax.set_title("Monte Carlo convergence")
ax.legend()

# --- Greeks vs spot price ---
spots = np.linspace(60, 140, 100)
greek_curves = {k: [] for k in ["delta", "gamma", "vega", "theta", "rho"]}
for s in spots:
    g = black_scholes.greeks(s, K, T, r, sigma, "call")
    for k in greek_curves:
        greek_curves[k].append(g[k])

ax = axes[1]
for k, vals in greek_curves.items():
    ax.plot(spots, vals, label=k)
ax.axvline(K, color="gray", linestyle=":", label="strike")
ax.set_xlabel("Spot price")
ax.set_title("Call option Greeks vs spot")
ax.legend()

plt.tight_layout()
plt.savefig("demo_output.png", dpi=150)
print("Saved plot to demo_output.png")
