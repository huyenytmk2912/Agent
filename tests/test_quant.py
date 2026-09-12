from atlas.quant import equity_curve, metrics, simple_returns


def test_simple_returns_and_curve():
    r = simple_returns([100.0, 110.0, 99.0])
    assert len(r) == 2
    curve = equity_curve(r)
    assert curve[0] == 1.0
    assert curve[-1] > 0.0


def test_metrics_are_finite():
    out = metrics([0.01, -0.005, 0.002, 0.0])
    assert set(out) == {"cagr", "volatility", "sharpe", "max_drawdown"}
    assert all(value == value for value in out.values())
