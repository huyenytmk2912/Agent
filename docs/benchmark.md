# Benchmark protocol

A stage is complete only after tests and benchmark artifacts are committed.

Required metrics:
- training and held-out loss/perplexity
- code exact match and unit-test pass rate
- math answer accuracy
- reasoning verifier pass rate
- tool-call validity and sandbox policy violations
- memory retrieval precision/recall
- quant backtest CAGR, volatility, Sharpe, max drawdown, turnover, and transaction-cost sensitivity

No metric may be claimed without a real run stored under runs/.