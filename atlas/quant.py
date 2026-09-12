"""Deterministic, dependency-light quant research utilities.

This module is research infrastructure only. It does not fetch data or provide
financial advice. Callers must provide legally usable price data.
"""
from math import sqrt


def simple_returns(prices):
    prices = list(prices)
    return [(prices[i] / prices[i - 1]) - 1.0 for i in range(1, len(prices))]


def equity_curve(returns, positions=None, fee_bps=0.0):
    returns = list(returns)
    positions = [1.0] * len(returns) if positions is None else list(positions)
    if len(positions) != len(returns):
        raise ValueError("positions and returns must have equal length")
    equity = [1.0]
    for r, p in zip(returns, positions):
        cost = abs(p) * fee_bps / 10000.0
        equity.append(equity[-1] * (1.0 + p * r - cost))
    return equity


def metrics(returns, periods_per_year=252):
    returns = list(returns)
    if not returns:
        return {"cagr": 0.0, "volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    curve = equity_curve(returns)
    years = len(returns) / periods_per_year
    cagr = curve[-1] ** (1.0 / max(years, 1e-12)) - 1.0
    mean = sum(returns) / len(returns)
    var = sum((x - mean) ** 2 for x in returns) / max(len(returns) - 1, 1)
    vol = sqrt(var) * sqrt(periods_per_year)
    sharpe = (mean / sqrt(var)) * sqrt(periods_per_year) if var > 0 else 0.0
    peak = curve[0]
    max_dd = 0.0
    for value in curve:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1.0)
    return {"cagr": cagr, "volatility": vol, "sharpe": sharpe, "max_drawdown": max_dd}
