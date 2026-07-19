"""
Options Pricing: Binomial Tree vs Black-Scholes
================================================
Implements two option pricing models from scratch and compares them:

  - Black-Scholes: closed-form pricing for European options
  - Binomial Tree (Cox-Ross-Rubinstein): pricing for both European and
    American options, via backward induction

Also includes:
  - A convergence study: how the binomial tree price approaches the
    Black-Scholes price as the number of steps (tree depth) increases
  - A comparison across strikes, maturities, and volatility inputs
  - The early-exercise premium for American options

Usage
-----
    python options_pricing.py

Requirements
------------
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------------------------
# 1. Black-Scholes model
# ---------------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Closed-form Black-Scholes price for a European option.

    Parameters
    ----------
    S : float           current underlying price
    K : float           strike price
    T : float            time to expiry, in years
    r : float           risk-free rate (annualised, continuously compounded)
    sigma : float       volatility (annualised)
    option_type : str   "call" or "put"
    """
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if sigma <= 0:
        fwd = S * np.exp(r * T)
        intrinsic = max(fwd - K, 0.0) if option_type == "call" else max(K - fwd, 0.0)
        return intrinsic * np.exp(-r * T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# ---------------------------------------------------------------------------
# 2. Binomial tree (Cox-Ross-Rubinstein)
# ---------------------------------------------------------------------------
def binomial_tree(S, K, T, r, sigma, n_steps, option_type="call", american=False):
    """
    CRR binomial tree price for a European or American option.

    Parameters
    ----------
    n_steps : int       number of time steps in the tree
    american : bool     if True, allow early exercise at every node
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)          # risk-neutral probability
    disc = np.exp(-r * dt)

    # Terminal stock prices at maturity (vectorised)
    j = np.arange(n_steps + 1)
    ST = S * (u ** (n_steps - j)) * (d ** j)

    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        values = disc * (q * values[:-1] + (1 - q) * values[1:])

        if american:
            j = np.arange(step + 1)
            S_t = S * (u ** (step - j)) * (d ** j)
            intrinsic = (np.maximum(S_t - K, 0.0) if option_type == "call"
                         else np.maximum(K - S_t, 0.0))
            values = np.maximum(values, intrinsic)

    return values[0]


# ---------------------------------------------------------------------------
# 3. Convergence study
# ---------------------------------------------------------------------------
def convergence_study(S, K, T, r, sigma, option_type="call", max_steps=300, step_grid=None):
    """Binomial price vs Black-Scholes price as tree depth increases."""
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    if step_grid is None:
        step_grid = np.unique(np.linspace(2, max_steps, 60).astype(int))

    binom_prices = [binomial_tree(S, K, T, r, sigma, n, option_type, american=False)
                     for n in step_grid]
    errors = np.array(binom_prices) - bs_price
    return step_grid, np.array(binom_prices), errors, bs_price


# ---------------------------------------------------------------------------
# 4. Comparison across strikes / maturities / volatilities
# ---------------------------------------------------------------------------
def compare_across_inputs(S, r, n_steps=200):
    strikes = np.linspace(70, 130, 13)
    maturities = [0.25, 0.5, 1.0, 2.0]
    vols = [0.15, 0.25, 0.35]

    strike_results = {
        "european_call_bs": [black_scholes(S, K, 1.0, r, 0.25, "call") for K in strikes],
        "european_call_binom": [binomial_tree(S, K, 1.0, r, 0.25, n_steps, "call", False) for K in strikes],
        "american_put_binom": [binomial_tree(S, K, 1.0, r, 0.25, n_steps, "put", True) for K in strikes],
        "european_put_binom": [binomial_tree(S, K, 1.0, r, 0.25, n_steps, "put", False) for K in strikes],
    }

    maturity_results = {
        T: {
            "call_bs": black_scholes(S, S, T, r, 0.25, "call"),
            "call_binom": binomial_tree(S, S, T, r, 0.25, n_steps, "call", False),
        }
        for T in maturities
    }

    vol_results = {
        sigma: {
            "call_bs": black_scholes(S, S, 1.0, r, sigma, "call"),
            "put_american_binom": binomial_tree(S, S, 1.0, r, sigma, n_steps, "put", True),
            "put_european_binom": binomial_tree(S, S, 1.0, r, sigma, n_steps, "put", False),
        }
        for sigma in vols
    }

    return strikes, strike_results, maturity_results, vol_results


# ---------------------------------------------------------------------------
# 5. Reporting / plots
# ---------------------------------------------------------------------------
def print_convergence_report(step_grid, binom_prices, errors, bs_price, label):
    print(f"\n  {label}")
    print(f"  {'-' * 55}")
    print(f"    Black-Scholes benchmark price: {bs_price:.4f}\n")
    print(f"    {'Steps':>8} | {'Binomial Price':>15} | {'Abs. Error':>12}")
    for n, p, e in list(zip(step_grid, binom_prices, errors))[::8]:
        print(f"    {n:>8} | {p:>15.4f} | {abs(e):>12.5f}")


def print_early_exercise_premium(S, K, T, r, sigma, n_steps=200):
    euro_put = binomial_tree(S, K, T, r, sigma, n_steps, "put", american=False)
    amer_put = binomial_tree(S, K, T, r, sigma, n_steps, "put", american=True)
    premium = amer_put - euro_put
    print(f"\n  Early exercise premium (American - European put)")
    print(f"  {'-' * 55}")
    print(f"    S={S}, K={K}, T={T}, r={r:.2%}, sigma={sigma:.2%}")
    print(f"    European put : {euro_put:.4f}")
    print(f"    American put : {amer_put:.4f}")
    print(f"    Premium      : {premium:.4f}")


def plot_results(step_grid, call_prices, call_errors, put_step_grid, put_errors,
                  strikes, strike_results, path="options_pricing_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Convergence: price
    axes[0, 0].plot(step_grid, call_prices, label="Binomial (call)")
    axes[0, 0].axhline(call_prices[-1], color="gray", linestyle="--", label="BS benchmark")
    axes[0, 0].set_title("Binomial Price Convergence to Black-Scholes")
    axes[0, 0].set_xlabel("Number of tree steps")
    axes[0, 0].set_ylabel("Option price")
    axes[0, 0].legend()

    # Convergence: error, showing the classic oscillation at low depth
    axes[0, 1].plot(step_grid, call_errors, color="firebrick", label="Call error")
    axes[0, 1].plot(put_step_grid, put_errors, color="steelblue", label="Put error", alpha=0.7)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Binomial Pricing Error vs Black-Scholes")
    axes[0, 1].set_xlabel("Number of tree steps")
    axes[0, 1].set_ylabel("Price - BS price")
    axes[0, 1].legend()

    # Across strikes: European (BS vs binomial) and American put
    axes[1, 0].plot(strikes, strike_results["european_call_bs"], label="Call, Black-Scholes", marker="o", markersize=3)
    axes[1, 0].plot(strikes, strike_results["european_call_binom"], label="Call, Binomial", linestyle="--")
    axes[1, 0].set_title("European Call Price Across Strikes")
    axes[1, 0].set_xlabel("Strike")
    axes[1, 0].set_ylabel("Price")
    axes[1, 0].legend()

    # American vs European put across strikes (early exercise premium)
    premium = np.array(strike_results["american_put_binom"]) - np.array(strike_results["european_put_binom"])
    axes[1, 1].plot(strikes, premium, color="darkgreen")
    axes[1, 1].set_title("Early Exercise Premium (American - European Put)")
    axes[1, 1].set_xlabel("Strike")
    axes[1, 1].set_ylabel("Premium")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nSaved chart to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25

    print("=" * 60)
    print("  BLACK-SCHOLES vs BINOMIAL TREE — BASELINE PRICES")
    print("=" * 60)
    for opt in ["call", "put"]:
        bs = black_scholes(S, K, T, r, sigma, opt)
        binom_euro = binomial_tree(S, K, T, r, sigma, 200, opt, american=False)
        binom_amer = binomial_tree(S, K, T, r, sigma, 200, opt, american=True)
        print(f"\n  {opt.capitalize()} (S={S}, K={K}, T={T}, r={r:.2%}, sigma={sigma:.2%})")
        print(f"    Black-Scholes (European) : {bs:.4f}")
        print(f"    Binomial, 200 steps (Eur): {binom_euro:.4f}")
        print(f"    Binomial, 200 steps (Am) : {binom_amer:.4f}")

    print("\n" + "=" * 60)
    print("  CONVERGENCE STUDY")
    print("=" * 60)
    call_steps, call_prices, call_errors, call_bs = convergence_study(S, K, T, r, sigma, "call")
    put_steps, put_prices, put_errors, put_bs = convergence_study(S, K, T, r, sigma, "put")
    print_convergence_report(call_steps, call_prices, call_errors, call_bs, "Call option convergence")
    print_convergence_report(put_steps, put_prices, put_errors, put_bs, "Put option convergence")

    print("\n" + "=" * 60)
    print("  EARLY EXERCISE PREMIUM")
    print("=" * 60)
    print_early_exercise_premium(S, K, T, r, sigma)
    print_early_exercise_premium(S, 110, T, r, sigma)  # deeper in-the-money put

    print("\n" + "=" * 60)
    print("  COMPARISON ACROSS STRIKES / MATURITIES / VOLATILITIES")
    print("=" * 60)
    strikes, strike_results, maturity_results, vol_results = compare_across_inputs(S, r)

    print("\n  Across maturities (ATM call, sigma=0.25):")
    for T_, vals in maturity_results.items():
        print(f"    T={T_:>4} | BS: {vals['call_bs']:.4f} | Binomial: {vals['call_binom']:.4f}")

    print("\n  Across volatilities (ATM, T=1.0):")
    for sig, vals in vol_results.items():
        print(f"    sigma={sig:.2f} | Call BS: {vals['call_bs']:.4f} | "
              f"Am Put: {vals['put_american_binom']:.4f} | Eur Put: {vals['put_european_binom']:.4f}")

    print("\nGenerating plots...")
    plot_results(call_steps, call_prices, call_errors, put_steps, put_errors,
                 strikes, strike_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
