import numpy as np
import pandas as pd
import pytest

import research as R


def make_returns(n_days=700, n_stocks=24, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(rng.normal(0, 0.01, (n_days, n_stocks)), index=idx, columns=[f"s{i}" for i in range(n_stocks)])


def test_weights_are_dollar_neutral_and_sized():
    sig = np.arange(20, dtype=float)
    w = R.long_short_weights(sig, 5)
    assert w.sum() == pytest.approx(0.0)
    assert np.abs(w).sum() == pytest.approx(1.0)
    assert (w > 0).sum() == 5 and (w < 0).sum() == 5
    assert w[-5:].min() > 0 and w[:5].max() < 0


def test_weights_flat_when_too_few_names():
    sig = np.array([1.0, np.nan, 2.0, np.nan])
    assert not R.long_short_weights(sig, 5).any()


def test_turnover_identical_zero_and_full_flip_two():
    w = R.long_short_weights(np.arange(20, dtype=float), 5)
    t = R.turnover(np.vstack([w, w, -w]))
    assert t[0] == pytest.approx(1.0)  # first trade from flat
    assert t[1] == pytest.approx(0.0)
    assert t[2] == pytest.approx(2.0)


def test_costs_reduce_net_monotonically():
    g = np.full(10, 0.01)
    t = np.full(10, 0.5)
    nets = [R.net_returns(g, t, b).mean() for b in (0, 5, 20)]
    assert nets[0] > nets[1] > nets[2]
    assert nets[0] == pytest.approx(0.01)


def test_signal_has_no_lookahead():
    ret = make_returns()
    base = R.momentum_signal(ret)
    t = 500
    changed = ret.copy()
    changed.iloc[t + 1 :] = 0.5  # rewrite the entire future
    alt = R.momentum_signal(changed)
    pd.testing.assert_series_equal(base.iloc[t], alt.iloc[t])


def test_holding_period_starts_after_rebalance_close():
    ret = pd.DataFrame({"a": [0.0, 0.10, 0.0, 0.0]})
    hold = R.holding_returns(ret, [0], step=2)
    assert hold[0, 0] == pytest.approx(0.10)  # day 1 is the first day earned
    hold2 = R.holding_returns(ret, [1], step=2)
    assert hold2[0, 0] == pytest.approx(0.0)  # day 1's return is NOT earned by a day-1 rebalance


def test_suspect_moves_are_masked_and_counted():
    close = pd.DataFrame({"a": [100.0, 101.0, 50.0, 50.5]})
    ret, n = R.daily_returns(close)
    assert n == 1 and np.isnan(ret["a"].iloc[2])


def test_newey_west_equals_plain_t_at_zero_lags():
    x = np.random.default_rng(3).normal(0.2, 1, 200)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert R.newey_west_t(x, lags=0) == pytest.approx(plain)


def test_newey_west_shrinks_t_for_positively_autocorrelated_series():
    rng = np.random.default_rng(4)
    e = rng.normal(size=500)
    x = np.zeros(500)
    for i in range(1, 500):
        x[i] = 0.8 * x[i - 1] + e[i]
    x += 0.3
    assert abs(R.newey_west_t(x, lags=8)) < abs(R.newey_west_t(x, lags=0))


def test_vectorised_matches_plain_loops():
    ret = make_returns()
    sig = R.momentum_signal(ret)
    pos = R.rebalance_dates(sig.index, 260)
    rows = sig.to_numpy()[pos]
    hold = R.holding_returns(ret, pos)
    w = np.vstack([R.long_short_weights(r, 5) for r in rows])
    net = R.net_returns((w * hold).sum(axis=1), R.turnover(w), 10)
    mean, sharpe = R.plain_loop_check(rows, hold, 5, 10)
    assert mean == pytest.approx(net.mean())
    assert sharpe == pytest.approx(R.ann_sharpe(net))


def test_permutation_detects_a_signal_with_foresight_and_not_noise():
    rng = np.random.default_rng(5)
    n_p, n_s = 60, 24
    hold = rng.normal(0, 0.05, (n_p, n_s))
    cheat = hold + rng.normal(0, 0.01, hold.shape)  # signal that sees the future
    noise = rng.normal(size=hold.shape)
    _, p_cheat, _ = R.permutation_pvalue(cheat, hold, 5, n_perm=200, seed=1)
    _, p_noise, _ = R.permutation_pvalue(noise, hold, 5, n_perm=200, seed=1)
    assert p_cheat < 0.02
    assert p_noise > 0.05


def test_permutation_is_deterministic_for_a_seed():
    rng = np.random.default_rng(6)
    hold = rng.normal(0, 0.05, (30, 24))
    sig = rng.normal(size=hold.shape)
    a = R.permutation_pvalue(sig, hold, 5, n_perm=50, seed=9)
    b = R.permutation_pvalue(sig, hold, 5, n_perm=50, seed=9)
    assert a[1] == b[1]


def test_capacity_scales_with_liquidity():
    w = np.vstack([R.long_short_weights(np.arange(20, dtype=float), 5), R.long_short_weights(np.arange(20, dtype=float)[::-1], 5)])
    adv_small = np.full(w.shape, 1e6)
    adv_big = np.full(w.shape, 1e8)
    assert R.capacity_aum(w, adv_big) == pytest.approx(100 * R.capacity_aum(w, adv_small))


def test_drawdown_of_known_path():
    x = np.array([0.10, -0.50, 0.20])
    assert R.max_drawdown(x) == pytest.approx(-0.50)
