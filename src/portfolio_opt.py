# import necessary libraries
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt

# PyPortfolioOpt for stable optimization
try:
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models, expected_returns
    from pypfopt import objective_functions
    HAS_PFOPT = True
except Exception:
    HAS_PFOPT = False


# ------------------------------
# Utilities
# ------------------------------

def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from price dataframe (Adj Close preferred)."""
    # Assumes columns are ['TSLA','BND','SPY'] price series
    ret = price_df.sort_index().pct_change().dropna(how="all")
    return ret


def annualize_mean_return(daily_returns: pd.Series) -> float:
    """Annualize the mean daily return: mean * 252."""
    return daily_returns.mean() * 252.0


def annualize_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualize the covariance matrix: cov * 252."""
    return daily_returns.cov() * 252.0


def expected_return_from_tsla_forecast(tsla_forecast: pd.DataFrame) -> float:
    """
    Turn a TSLA future price forecast into an expected annual return.
    Uses start-to-end price growth over the forecast period.
    Expects a DataFrame with a 'mean' column (or first column as fallback).
    """
    # Accept flexible column names
    if "mean" in tsla_forecast.columns:
        price = tsla_forecast["mean"].copy()
    elif "ARIMA_Forecast" in tsla_forecast.columns:
        price = tsla_forecast["ARIMA_Forecast"].copy()
    elif "LSTM_Forecast" in tsla_forecast.columns:
        price = tsla_forecast["LSTM_Forecast"].copy()
    else:
        # fallback to first column
        price = tsla_forecast.iloc[:, 0].copy()

    # Compute daily return based on start-to-end growth
    n_days = len(price)
    if n_days < 2:
        raise ValueError("Forecast too short to compute returns")
    start_price = price.iloc[0]
    end_price = price.iloc[-1]
    daily_ret = (end_price / start_price) ** (1 / n_days) - 1

    # Annualize
    return daily_ret * 252


# ------------------------------
# Optimization Outputs
# ------------------------------

@dataclass
class PortfolioPoint:
    risk: float
    ret: float
    sharpe: float
    weights: dict


@dataclass
class FrontierResults:
    points: list            # list[PortfolioPoint]
    max_sharpe: PortfolioPoint
    min_vol: PortfolioPoint


# ------------------------------
# Optimization (with/without PyPortfolioOpt)
# ------------------------------

def _opt_with_pypfopt(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> FrontierResults:
    assets = list(mu.index)

    # Max Sharpe
    ef1 = EfficientFrontier(mu, cov)
    ef1.add_objective(objective_functions.L2_reg, gamma=0.001)  # tiny regularization
    w_ms = ef1.max_sharpe(risk_free_rate=rf)
    w_ms = ef1.clean_weights()
    ret_ms, risk_ms, sharpe_ms = ef1.portfolio_performance(risk_free_rate=rf)
    max_sharpe_pt = PortfolioPoint(risk=risk_ms, ret=ret_ms, sharpe=sharpe_ms, weights=w_ms)

    # Min volatility
    ef2 = EfficientFrontier(mu, cov)
    ef2.add_objective(objective_functions.L2_reg, gamma=0.001)
    w_mv = ef2.min_volatility()
    w_mv = ef2.clean_weights()
    ret_mv, risk_mv, sharpe_mv = ef2.portfolio_performance(risk_free_rate=rf)
    min_vol_pt = PortfolioPoint(risk=risk_mv, ret=ret_mv, sharpe=sharpe_mv, weights=w_mv)

    # Efficient frontier curve by sweeping target returns
    min_ret = min(ret_mv, ret_ms)
    max_ret = max(ret_mv, ret_ms)
    targets = np.linspace(min_ret, max_ret, 50)

    points = []
    for tr in targets:
        try:
            ef = EfficientFrontier(mu, cov)
            ef.add_objective(objective_functions.L2_reg, gamma=0.001)
            ef.efficient_return(target_return=tr)
            R, S, Sh = ef.portfolio_performance(risk_free_rate=rf)
            w = ef.clean_weights()
            points.append(PortfolioPoint(risk=S, ret=R, sharpe=Sh, weights=w))
        except Exception:
            # Some targets are infeasible
            continue

    return FrontierResults(points=points, max_sharpe=max_sharpe_pt, min_vol=min_vol_pt)


def _opt_with_numpy(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> FrontierResults:
    """
    Lightweight fallback if PyPortfolioOpt is not installed.
    Uses a coarse random search + normalization to approximate frontier.
    """
    rng = np.random.default_rng(42)
    assets = list(mu.index)
    n = len(assets)

    n_samples = 10000
    weights = rng.random((n_samples, n))
    weights = weights / weights.sum(axis=1, keepdims=True)

    rets = weights @ mu.values
    risks = np.sqrt(np.einsum("bi,ij,bj->b", weights, cov.values, weights))
    sharpes = (rets - rf) / np.where(risks == 0, np.nan, risks)

    # Build coarse "frontier": take the top portfolios by return buckets
    df = pd.DataFrame({
        "ret": rets, "risk": risks, "sharpe": sharpes
    })
    df["idx"] = np.arange(n_samples)

    # Choose ~60 frontier points by quantiles of return
    qs = np.linspace(df["ret"].min(), df["ret"].max(), 60)
    pts = []
    for q in qs:
        sub = df.iloc[(df["ret"] - q).abs().argsort()[:30]]  # nearest 30
        best = sub.sort_values("risk").iloc[:1]
        for _, row in best.iterrows():
            i = int(row["idx"])
            w = {assets[k]: weights[i, k] for k in range(n)}
            pts.append(PortfolioPoint(risk=row["risk"], ret=row["ret"], sharpe=row["sharpe"], weights=w))

    # Max Sharpe
    i_ms = np.nanargmax(sharpes)
    w_ms = {assets[k]: weights[i_ms, k] for k in range(n)}
    max_sharpe_pt = PortfolioPoint(risk=risks[i_ms], ret=rets[i_ms], sharpe=sharpes[i_ms], weights=w_ms)

    # Min Vol
    i_mv = np.nanargmin(risks)
    w_mv = {assets[k]: weights[i_mv, k] for k in range(n)}
    min_vol_pt = PortfolioPoint(risk=risks[i_mv], ret=rets[i_mv], sharpe=sharpes[i_mv], weights=w_mv)

    return FrontierResults(points=pts, max_sharpe=max_sharpe_pt, min_vol=min_vol_pt)


def optimize_portfolio(
    daily_returns: pd.DataFrame,
    tsla_expected_return_annual: float,
    rf: float = 0.0,
    use_pypfopt: bool = True
) -> FrontierResults:
    """
    Build expected returns vector & covariance, then compute:
      - Efficient frontier curve
      - Max Sharpe portfolio
      - Min Vol portfolio

    Parameters
    ----------
    daily_returns : DataFrame with columns ['TSLA','BND','SPY'] daily % returns
    tsla_expected_return_annual : float from your forecast (annualized)
    rf : risk-free rate (annualized), default 0.0
    use_pypfopt : prefer PyPortfolioOpt solver when available
    """
    cols = ["TSLA", "BND", "SPY"]
    dr = daily_returns[cols].dropna(how="any").copy()

    # Historical annualized mean for BND & SPY, model view for TSLA
    bnd_ann = annualize_mean_return(dr["BND"])
    spy_ann = annualize_mean_return(dr["SPY"])
    mu = pd.Series({"TSLA": tsla_expected_return_annual, "BND": bnd_ann, "SPY": spy_ann})
    cov = annualize_covariance(dr)

    if use_pypfopt and HAS_PFOPT:
        return _opt_with_pypfopt(mu, cov, rf=rf)
    else:
        return _opt_with_numpy(mu, cov, rf=rf)


# ------------------------------
# Plotting
# ------------------------------

def plot_efficient_frontier(frontier: FrontierResults, title="Efficient Frontier"):
    plt.figure(figsize=(9, 6))
    # Frontier cloud/curve
    if frontier.points:
        xs = [p.risk for p in frontier.points]
        ys = [p.ret for p in frontier.points]
        plt.scatter(xs, ys, s=20, alpha=0.6, label="Efficient frontier")

    # Max Sharpe
    ms = frontier.max_sharpe
    plt.scatter([ms.risk], [ms.ret], s=120, marker="*", label=f"Max Sharpe (SR={ms.sharpe:.2f})")

    # Min Vol
    mv = frontier.min_vol
    plt.scatter([mv.risk], [mv.ret], s=120, marker="D", label="Min Volatility")

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def format_weights(weights: dict, precision=4) -> dict:
    """Nicely round small numbers; keep only non-trivial weights."""
    clean = {k: round(float(v), precision) for k, v in weights.items() if abs(v) > 1e-4}
    # ensure they sum ~1 after rounding
    s = sum(clean.values())
    if 0.98 <= s <= 1.02:
        return clean
    return weights