

"""
Portfolio Optimisation Tool
============================
Author: Sam Sousa

What this tool does:
  - Simulates 10,000 random portfolios across 5 asset classes
  - Plots the efficient frontier (best risk/return combinations)
  - Finds the maximum Sharpe ratio portfolio (optimal risk-adjusted return)
  - Finds the minimum variance portfolio (lowest possible risk)
  - Outputs allocation charts for both optimal portfolios

How to run:
  1. Install dependencies: pip install numpy pandas scipy matplotlib yfinance
  2. Run: python portfolio_optimisation.py

To use real market data instead of synthetic data:
  - Uncomment the yfinance block in Section 1
  - Replace TICKERS with your chosen assets
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # Use non-interactive backend (safe for all systems)
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# SECTION 1: DATA
# =============================================================================
# We use synthetic data here because the yfinance API requires internet access.
# In a real deployment, replace this block with:
#
import yfinance as yf
TICKERS = ["NVDA", "TSLA", "GLD", "JPM", "JD"]
raw = yf.download(TICKERS, period="3y", auto_adjust=True)["Close"]
returns = raw.pct_change().dropna()
returns.columns = ["Nvidia", "Tesla", "Gold", "JPMorgan", "JD"]
#
# The rest of the script is identical either way.



# =============================================================================
# SECTION 2: EXPECTED RETURNS AND COVARIANCE MATRIX
# =============================================================================
# mu_annual: the expected annual return for each asset
# cov_annual: how assets move together (covariance). If two assets are negatively
#             correlated, holding both reduces overall portfolio risk.

mu_annual  = returns.mean() * 252
cov_annual = returns.cov()  * 252

N_ASSETS    = len(mu_annual)
ASSET_NAMES = list(mu_annual.index)
RISK_FREE   = 0.045   # Approximate risk-free rate (e.g. short-dated UK gilts or US T-bills)

# =============================================================================
# SECTION 3: PORTFOLIO METRIC FUNCTIONS
# =============================================================================

def portfolio_return(weights, mu):
    """
    Expected return of the portfolio.
    weights: array of allocations to each asset (must sum to 1)
    mu: array of expected returns per asset
    """
    return np.dot(weights, mu)


def portfolio_volatility(weights, cov):
    """
    Risk of the portfolio, measured as annualised standard deviation.
    The covariance matrix captures how assets co-move.
    Diversification reduces volatility when assets are not perfectly correlated.
    """
    variance = weights @ cov @ weights   # matrix multiplication
    return np.sqrt(variance)


def sharpe_ratio(weights, mu, cov, rf=RISK_FREE):
    """
    Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
    Measures how much excess return you earn per unit of risk.
    A Sharpe above 1.0 is generally considered good.
    """
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    return (ret - rf) / vol

# =============================================================================
# SECTION 4: MONTE CARLO SIMULATION
# =============================================================================
# Generate 10,000 portfolios with random weights.
# This gives us a cloud of points in risk-return space.
# The efficient frontier sits on the upper-left edge of that cloud.

N_SIMULATIONS = 10_000
sim_returns = np.zeros(N_SIMULATIONS)
sim_vols    = np.zeros(N_SIMULATIONS)
sim_sharpes = np.zeros(N_SIMULATIONS)

for i in range(N_SIMULATIONS):
    # Dirichlet distribution gives random weights that always sum to 1
    w = np.random.dirichlet(np.ones(N_ASSETS))
    sim_returns[i] = portfolio_return(w, mu_annual)
    sim_vols[i]    = portfolio_volatility(w, cov_annual)
    sim_sharpes[i] = sharpe_ratio(w, mu_annual, cov_annual)

# =============================================================================
# SECTION 5: EFFICIENT FRONTIER
# =============================================================================
# For each target return level, find the minimum-risk portfolio that achieves it.
# Plotting these points traces the efficient frontier.
# Any portfolio below this line is suboptimal — same risk, less return.

def min_vol_for_return(target_return, mu, cov):
    """Minimise volatility subject to achieving a specific return."""
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},                      # weights sum to 1
        {"type": "eq", "fun": lambda w: portfolio_return(w, mu) - target_return}  # hit target return
    ]
    bounds = [(0, 1)] * N_ASSETS   # no short selling
    return minimize(
        portfolio_volatility,
        x0=np.ones(N_ASSETS) / N_ASSETS,   # start from equal weights
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

target_returns   = np.linspace(mu_annual.min(), mu_annual.max(), 60)
frontier_vols    = []
frontier_rets    = []
frontier_weights = []

for target in target_returns:
    res = min_vol_for_return(target, mu_annual, cov_annual)
    if res.success:
        frontier_vols.append(portfolio_volatility(res.x, cov_annual))
        frontier_rets.append(target)
        frontier_weights.append(res.x)

frontier_vols = np.array(frontier_vols)
frontier_rets = np.array(frontier_rets)

# =============================================================================
# SECTION 6: MAXIMUM SHARPE RATIO PORTFOLIO
# =============================================================================
# The tangency portfolio: the single point on the efficient frontier that
# maximises return per unit of risk. This is what most asset managers target
# when constructing a risk-adjusted optimal portfolio.

def neg_sharpe(weights, mu, cov, rf):
    """We minimise negative Sharpe to find the maximum."""
    return -sharpe_ratio(weights, mu, cov, rf)

sharpe_result = minimize(
    neg_sharpe,
    x0=np.ones(N_ASSETS) / N_ASSETS,
    args=(mu_annual, cov_annual, RISK_FREE),
    method="SLSQP",
    bounds=[(0, 1)] * N_ASSETS,
    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
)

opt_weights = sharpe_result.x
opt_return  = portfolio_return(opt_weights, mu_annual)
opt_vol     = portfolio_volatility(opt_weights, cov_annual)
opt_sharpe  = sharpe_ratio(opt_weights, mu_annual, cov_annual)

# =============================================================================
# SECTION 7: MINIMUM VARIANCE PORTFOLIO
# =============================================================================
# The leftmost point on the frontier. Pure risk minimisation with no return target.
# Used by ultra-conservative mandates or as a benchmark for risk budgeting.

minvar_result = minimize(
    portfolio_volatility,
    x0=np.ones(N_ASSETS) / N_ASSETS,
    args=(cov_annual,),
    method="SLSQP",
    bounds=[(0, 1)] * N_ASSETS,
    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
)

mv_weights = minvar_result.x
mv_return  = portfolio_return(mv_weights, mu_annual)
mv_vol     = portfolio_volatility(mv_weights, cov_annual)
mv_sharpe  = sharpe_ratio(mv_weights, mu_annual, cov_annual)

# =============================================================================
# SECTION 8: PRINT RESULTS
# =============================================================================

print("=" * 55)
print("MAXIMUM SHARPE RATIO PORTFOLIO")
print("=" * 55)
print(f"  Expected Return : {opt_return:.2%}")
print(f"  Volatility      : {opt_vol:.2%}")
print(f"  Sharpe Ratio    : {opt_sharpe:.3f}")
print()
print("  Allocations:")
for name, w in zip(ASSET_NAMES, opt_weights):
    print(f"    {name:<18} {w:.1%}")

print()
print("=" * 55)
print("MINIMUM VARIANCE PORTFOLIO")
print("=" * 55)
print(f"  Expected Return : {mv_return:.2%}")
print(f"  Volatility      : {mv_vol:.2%}")
print(f"  Sharpe Ratio    : {mv_sharpe:.3f}")
print()
print("  Allocations:")
for name, w in zip(ASSET_NAMES, mv_weights):
    print(f"    {name:<18} {w:.1%}")

# =============================================================================
# SECTION 9: PLOT
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

# Left panel: Efficient Frontier
ax1 = axes[0]
sc = ax1.scatter(sim_vols, sim_returns, c=sim_sharpes, cmap="plasma",
                 alpha=0.25, s=4, zorder=1)
ax1.plot(frontier_vols, frontier_rets, color="#00d4ff", linewidth=2.5,
         label="Efficient Frontier", zorder=3)
ax1.scatter(opt_vol, opt_return, color="#ffd700", s=160, zorder=5,
            label=f"Max Sharpe ({opt_sharpe:.2f})", edgecolors="white", linewidths=1)
ax1.scatter(mv_vol, mv_return, color="#00ff88", s=140, zorder=5,
            label="Min Variance", edgecolors="white", linewidths=1)
ax1.set_xlabel("Volatility (Annualised)", fontsize=11)
ax1.set_ylabel("Expected Return (Annualised)", fontsize=11)
ax1.set_title("Efficient Frontier", fontsize=13, fontweight="bold")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax1.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
cbar = fig.colorbar(sc, ax=ax1, pad=0.02)
cbar.set_label("Sharpe Ratio", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

# Right panel: Allocation Bar Chart
ax2 = axes[1]
colours = ["#4e9af1", "#f1c40f", "#e74c3c", "#2ecc71", "#9b59b6"]
x = np.arange(N_ASSETS)
width = 0.35

bars1 = ax2.bar(x - width/2, opt_weights, width, color=colours, alpha=0.9, label="Max Sharpe")
bars2 = ax2.bar(x + width/2, mv_weights,  width, color=colours, alpha=0.5,
                label="Min Variance", hatch="//", edgecolor="white", linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(ASSET_NAMES, rotation=20, ha="right", fontsize=9, color="white")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.set_ylabel("Portfolio Weight", fontsize=11)
ax2.set_title("Portfolio Allocations", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 0.9)
ax2.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)

for bar in bars1:
    h = bar.get_height()
    if h > 0.02:
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
                 ha="center", va="bottom", fontsize=8, color="white")
for bar in bars2:
    h = bar.get_height()
    if h > 0.02:
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
                 ha="center", va="bottom", fontsize=8, color="gray")

plt.suptitle("Portfolio Optimisation Tool", fontsize=15, fontweight="bold",
             color="white", y=1.01)
plt.tight_layout()
plt.savefig("portfolio_optimisation.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
print("\nChart saved as portfolio_optimisation.png")
