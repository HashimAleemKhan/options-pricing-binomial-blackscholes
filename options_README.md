# Options Pricing: Binomial Tree vs Black-Scholes

Implements the Black-Scholes model and a Cox-Ross-Rubinstein binomial tree
from scratch in Python, prices European and American options, and studies
how the binomial tree converges to the Black-Scholes benchmark as tree
depth increases.

## What it does

- **Black-Scholes**: closed-form European option pricing
- **Binomial tree (CRR)**: European and American option pricing via
  backward induction, with early exercise checked at every node
- **Convergence study**: binomial price vs Black-Scholes price as the
  number of steps grows, including the characteristic oscillation/error
  pattern at low tree depth
- **Early exercise premium**: American put price minus European put price,
  showing when early exercise actually adds value
- **Cross-sectional comparison**: pricing behaviour across a grid of
  strikes, maturities, and volatility inputs

## Usage

```bash
pip install -r requirements.txt
python options_pricing.py
```

Output includes a full console report (baseline prices, convergence table,
early exercise premium, cross-sectional comparison) plus a saved chart
(`options_pricing_results.png`) showing convergence, pricing error, the
strike curve, and the early-exercise premium curve.

## Notes / caveats

- Uses the standard Black-Scholes assumptions (constant volatility, no
  dividends, continuous trading, lognormal returns) — a real-world options
  book would need dividend adjustments, an implied-vol surface, etc.
- The binomial tree error does not shrink monotonically with more steps —
  it oscillates for even vs. odd step counts before converging, which the
  convergence chart makes visible.
