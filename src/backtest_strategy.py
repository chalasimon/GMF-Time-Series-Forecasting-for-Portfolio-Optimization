import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Compute cumulative portfolio returns given daily returns and weights
def compute_portfolio_returns(daily_returns: pd.DataFrame, weights: dict) -> pd.Series:
    weights_series = pd.Series(weights).reindex(daily_returns.columns).fillna(0.0)
    portfolio_daily_returns = daily_returns @ weights_series
    cum_returns = (1 + portfolio_daily_returns).cumprod()
    return cum_returns, portfolio_daily_returns

# Annualized Sharpe Ratio
def annualized_sharpe(daily_returns: pd.Series, risk_free=0.0) -> float:
    mean_ret = daily_returns.mean() * 252
    std_ret  = daily_returns.std() * np.sqrt(252)
    return (mean_ret - risk_free) / std_ret if std_ret != 0 else np.nan

# Backtest function
def backtest_portfolio(daily_returns: pd.DataFrame,
                       strategy_weights: dict,
                       benchmark_weights: dict,
                       start_date: str,
                       end_date: str):
    bt_returns = daily_returns.loc[start_date:end_date]
    
    # Strategy
    strategy_cum, strategy_daily = compute_portfolio_returns(bt_returns, strategy_weights)
    
    # Benchmark
    benchmark_cum, benchmark_daily = compute_portfolio_returns(bt_returns, benchmark_weights)
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(strategy_cum, label="Model Portfolio")
    plt.plot(benchmark_cum, label="Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"Backtest: {start_date} to {end_date}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Performance metrics
    results = {
        "strategy_total_return": strategy_cum[-1] - 1,
        "benchmark_total_return": benchmark_cum[-1] - 1,
        "strategy_sharpe": annualized_sharpe(strategy_daily),
        "benchmark_sharpe": annualized_sharpe(benchmark_daily)
    }
    return results