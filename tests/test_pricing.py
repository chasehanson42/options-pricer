import math

import pytest

from optpricer import black_scholes, binomial, monte_carlo

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_converges_to_black_scholes(option_type):
    bs_price = black_scholes.price(S, K, T, r, sigma, option_type)
    bt_price = binomial.price(S, K, T, r, sigma, n=500, option_type=option_type)
    assert bt_price == pytest.approx(bs_price, abs=0.05)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_converges_to_black_scholes(option_type):
    bs_price = black_scholes.price(S, K, T, r, sigma, option_type)
    mc_price, stderr = monte_carlo.price(
        S, K, T, r, sigma, option_type=option_type, n_paths=200_000, seed=42
    )
    assert mc_price == pytest.approx(bs_price, abs=3 * stderr + 0.02)


def test_american_put_worth_at_least_as_much_as_european():
    euro = binomial.price(S, K, T, r, sigma, n=300, option_type="put", exercise="european")
    amer = binomial.price(S, K, T, r, sigma, n=300, option_type="put", exercise="american")
    assert amer >= euro - 1e-8


def test_american_call_equals_european_without_dividends():
    # No dividends -> early exercise of a call is never optimal.
    euro = binomial.price(S, K, T, r, sigma, n=300, option_type="call", exercise="european")
    amer = binomial.price(S, K, T, r, sigma, n=300, option_type="call", exercise="american")
    assert amer == pytest.approx(euro, abs=1e-6)


def test_put_call_parity_black_scholes():
    call = black_scholes.price(S, K, T, r, sigma, "call")
    put = black_scholes.price(S, K, T, r, sigma, "put")
    assert (call - put) == pytest.approx(S - K * math.exp(-r * T), abs=1e-8)


def test_call_delta_between_zero_and_one():
    g = black_scholes.greeks(S, K, T, r, sigma, "call")
    assert 0.0 <= g["delta"] <= 1.0


def test_variance_reduction_lowers_standard_error():
    _, se_plain = monte_carlo.price(
        S, K, T, r, sigma, n_paths=20_000, antithetic=False, control_variate=False, seed=7
    )
    _, se_reduced = monte_carlo.price(
        S, K, T, r, sigma, n_paths=20_000, antithetic=True, control_variate=True, seed=7
    )
    assert se_reduced < se_plain
