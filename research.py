"""Net-of-cost signal research on Indian large-cap equities.

One pre-specified signal (12-1 month cross-sectional momentum), tested the way
a live process would have to defend it:

* point-in-time: weights at a rebalance use prices up to that close only,
  and earn the returns of the following holding period
* realistic costs: a one-way cost in basis points charged on turnover
* capacity: how much money the trades could move before they are too large a
  share of average daily traded value
* honest statistics: a permutation test against random portfolios with the
  same structure, a held-out final segment and a Newey-West t-statistic
* an independent plain-loop recomputation of the headline numbers

Everything is deterministic given the seed. Nothing here is tuned on the
holdout: the signal, window, portfolio size and rebalance period are fixed
up front.
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252 / 21  # monthly (21 trading day) holding periods


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_panel(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one CSV per stock (Date, Close, Volume, Series) into two panels."""
    closes, vols = {}, {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, usecols=lambda c: c in {"Date", "Close", "Volume", "Series"})
        if "Series" in df.columns:
            df = df[df["Series"] == "EQ"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates("Date").set_index("Date").sort_index()
        closes[name] = df["Close"]
        vols[name] = df["Volume"]
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(vols).sort_index()


def daily_returns(close: pd.DataFrame, max_abs_move: float = 0.25) -> tuple[pd.DataFrame, int]:
    """Simple daily returns with suspect moves masked.

    A one-day move beyond max_abs_move is far more likely to be an unadjusted
    split or a bad print than a real return, so it is set to NaN (treated as
    zero in compounding) and counted. Returns the frame and the count masked.
    """
    ret = close.pct_change(fill_method=None)
    bad = ret.abs() > max_abs_move
    return ret.mask(bad), int(bad.sum().sum())


# --------------------------------------------------------------------------
# Signal and portfolio
# --------------------------------------------------------------------------
def momentum_signal(ret: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 month momentum: cumulative log return from t-lookback to t-skip.

    Uses returns up to and including day t only, so it is known at the close
    of day t. NaN where fewer than 80% of the window's returns exist.
    """
    logr = np.log1p(ret)
    window = lookback - skip
    total = logr.shift(skip).rolling(window, min_periods=int(0.8 * window)).sum()
    return total


def rebalance_dates(index: pd.DatetimeIndex, start_pos: int, step: int = 21) -> list[int]:
    return list(range(start_pos, len(index) - step, step))


def long_short_weights(signal_row: np.ndarray, n_side: int) -> np.ndarray:
    """Dollar-neutral weights: +0.5 split over the top n_side, -0.5 over the bottom n_side."""
    w = np.zeros_like(signal_row, dtype=float)
    ok = np.where(~np.isnan(signal_row))[0]
    if len(ok) < 2 * n_side:
        return w
    order = ok[np.argsort(signal_row[ok], kind="mergesort")]
    w[order[:n_side]] = -0.5 / n_side
    w[order[-n_side:]] = 0.5 / n_side
    return w


def holding_returns(ret: pd.DataFrame, pos: list[int], step: int = 21) -> np.ndarray:
    """Compounded return of each stock over the period AFTER each rebalance close.

    Period i starts the day after pos[i] and lasts `step` days. NaN returns
    count as zero. Shape: (n_periods, n_stocks).
    """
    r = ret.fillna(0.0).to_numpy()
    out = np.zeros((len(pos), r.shape[1]))
    for i, p in enumerate(pos):
        out[i] = np.prod(1.0 + r[p + 1 : p + 1 + step], axis=0) - 1.0
    return out


def turnover(weights: np.ndarray) -> np.ndarray:
    """One-way-sum turnover per rebalance: sum |w_t - w_{t-1}|. First period trades from flat."""
    prev = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    return np.abs(weights - prev).sum(axis=1)


def net_returns(gross: np.ndarray, turn: np.ndarray, cost_bps: float) -> np.ndarray:
    """Subtract cost_bps (one way) on every unit of turnover."""
    return gross - turn * cost_bps / 1e4


def ann_sharpe(x: np.ndarray) -> float:
    s = np.std(x, ddof=1)
    return float(np.mean(x) / s * np.sqrt(PERIODS_PER_YEAR)) if s > 0 else float("nan")


def ann_return(x: np.ndarray) -> float:
    return float(np.mean(x) * PERIODS_PER_YEAR)


def max_drawdown(x: np.ndarray) -> float:
    curve = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def newey_west_t(x: np.ndarray, lags: int = 3) -> float:
    """t-statistic of the mean with Newey-West (Bartlett) standard error."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    var = np.dot(e, e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * np.dot(e[k:], e[:-k]) / n
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def permutation_pvalue(
    signal_rows: np.ndarray,
    hold: np.ndarray,
    n_side: int,
    n_perm: int = 2000,
    seed: int = 7,
) -> tuple[float, float, np.ndarray]:
    """Compare the real GROSS Sharpe to the Sharpe of random portfolios.

    Each permutation shuffles the cross-section of signal values at every
    rebalance independently (stocks keep their returns, the signal loses its
    information), rebuilds the portfolio and measures gross Sharpe. Gross is
    used on purpose: random portfolios churn far more than a momentum signal,
    so a net comparison would reward the signal for low turnover, not skill.
    The p-value is the share of shuffles doing at least as well as the real
    signal (add-one smoothing).
    """
    rng = np.random.default_rng(seed)

    def sharpe_of(sig: np.ndarray) -> float:
        w = np.vstack([long_short_weights(row, n_side) for row in sig])
        return ann_sharpe((w * hold).sum(axis=1))

    real = sharpe_of(signal_rows)
    null = np.empty(n_perm)
    for b in range(n_perm):
        shuffled = signal_rows.copy()
        for i in range(shuffled.shape[0]):
            ok = np.where(~np.isnan(shuffled[i]))[0]
            shuffled[i, ok] = rng.permutation(shuffled[i, ok])
        null[b] = sharpe_of(shuffled)
    p = (1 + int((null >= real).sum())) / (n_perm + 1)
    return real, p, null


def capacity_aum(
    weights: np.ndarray, adv_value: np.ndarray, max_participation: float = 0.10, q: float = 95.0
) -> float:
    """AUM (in rupees) at which the q-th percentile trade equals max_participation of daily value.

    trade_i = AUM * |dw_i|, participation_i = trade_i / ADV_i. Only trades that
    actually occur (dw != 0) are counted.
    """
    prev = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    dw = np.abs(weights - prev)
    mask = (dw > 0) & np.isfinite(adv_value) & (adv_value > 0)
    ratio = dw[mask] / adv_value[mask]
    return float(max_participation / np.percentile(ratio, q))


# --------------------------------------------------------------------------
# Independent recomputation (plain loops, no shared helpers)
# --------------------------------------------------------------------------
def plain_loop_check(
    signal_rows: np.ndarray, hold: np.ndarray, n_side: int, cost_bps: float
) -> tuple[float, float]:
    """Recompute mean net return per period and net Sharpe with nested loops only."""
    n_p, n_s = signal_rows.shape
    prev = [0.0] * n_s
    nets = []
    for i in range(n_p):
        vals = [(signal_rows[i][j], j) for j in range(n_s) if not np.isnan(signal_rows[i][j])]
        w = [0.0] * n_s
        if len(vals) >= 2 * n_side:
            vals.sort(key=lambda t: (t[0], t[1]))
            for _, j in vals[:n_side]:
                w[j] = -0.5 / n_side
            for _, j in vals[-n_side:]:
                w[j] = 0.5 / n_side
        gross = 0.0
        turn = 0.0
        for j in range(n_s):
            gross += w[j] * hold[i][j]
            turn += abs(w[j] - prev[j])
        nets.append(gross - turn * cost_bps / 1e4)
        prev = w
    m = sum(nets) / len(nets)
    var = sum((x - m) ** 2 for x in nets) / (len(nets) - 1)
    return m, m / var**0.5 * (PERIODS_PER_YEAR**0.5)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
@dataclass
class Results:
    summary: pd.DataFrame
    cost_table: pd.DataFrame
    perm_p: float
    null: np.ndarray
    real_sharpe: float
    capacity_cr: float
    n_masked: int
    n_periods: int
    n_names: int
    first: pd.Timestamp
    last: pd.Timestamp
    check: tuple[float, float]
    vec: tuple[float, float]
    net_curve: pd.Series
    gross_curve: pd.Series


def run(
    data_dir: str,
    n_side: int = 10,
    cost_bps: float = 10.0,
    holdout_frac: float = 0.4,
    n_perm: int = 2000,
    seed: int = 7,
    min_names: int = 40,
) -> Results:
    close, vol = load_panel(data_dir)
    ret, n_masked = daily_returns(close)
    sig = momentum_signal(ret)
    start = int(np.argmax(sig.notna().sum(axis=1).to_numpy() >= max(min_names, 2 * n_side)))
    pos = rebalance_dates(sig.index, start)
    signal_rows = sig.to_numpy()[pos]
    hold = holding_returns(ret, pos)
    weights = np.vstack([long_short_weights(r, n_side) for r in signal_rows])
    gross = (weights * hold).sum(axis=1)
    turn = turnover(weights)
    net = net_returns(gross, turn, cost_bps)
    dates = sig.index[pos]

    # held-out final segment: nothing above is tuned on it
    cut = int(len(net) * (1 - holdout_frac))
    seg = {"Full sample": slice(0, len(net)), "In-sample (first 60%)": slice(0, cut), "Holdout (last 40%)": slice(cut, len(net))}
    rows = []
    for name, s in seg.items():
        g, n = gross[s], net[s]
        rows.append(
            {
                "Segment": name,
                "Periods": len(n),
                "Gross return % a year": 100 * ann_return(g),
                "Net return % a year": 100 * ann_return(n),
                "Net Sharpe": ann_sharpe(n),
                "Net t (Newey-West)": newey_west_t(n),
                "Max drawdown %": 100 * max_drawdown(n),
                "Avg turnover per month": float(turn[s].mean()),
            }
        )
    summary = pd.DataFrame(rows).set_index("Segment")

    # cost sensitivity on the full sample
    ct = []
    for bps in (0, 5, 10, 20, 30, 50):
        n_ = net_returns(gross, turn, bps)
        ct.append({"One-way cost (bps)": bps, "Net return % a year": 100 * ann_return(n_), "Net Sharpe": ann_sharpe(n_)})
    cost_table = pd.DataFrame(ct).set_index("One-way cost (bps)")

    real, p, null = permutation_pvalue(signal_rows, hold, n_side, n_perm, seed)

    # capacity from trailing 20-day average traded value at each rebalance
    # recent 36 months only: older volume data is too thin to describe today's liquidity
    adv = (close * vol).rolling(20, min_periods=10).mean().to_numpy()[pos]
    cap_cr = capacity_aum(weights[-36:], adv[-36:]) / 1e7

    check = plain_loop_check(signal_rows, hold, n_side, cost_bps)
    vec = (float(net.mean()), ann_sharpe(net))

    return Results(
        summary, cost_table, p, null, real, cap_cr, n_masked, len(net), close.shape[1],
        dates[0], dates[-1], check, vec,
        pd.Series(np.cumprod(1 + net), index=dates), pd.Series(np.cumprod(1 + gross), index=dates),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, help="folder with one CSV per stock")
    ap.add_argument("--n-side", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    r = run(a.data_dir, a.n_side, a.cost_bps, n_perm=a.n_perm)
    os.makedirs(a.out, exist_ok=True)
    r.summary.round(3).to_csv(os.path.join(a.out, "summary.csv"))
    r.cost_table.round(3).to_csv(os.path.join(a.out, "cost_sensitivity.csv"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(r.gross_curve, label="Gross", color="#9aa5b1")
    ax[0].plot(r.net_curve, label=f"Net of {a.cost_bps:g} bps one way", color="#1f4e79")
    ax[0].set_title("Growth of 1 rupee, monthly long-short")
    ax[0].legend(frameon=False)
    ax[1].hist(r.null, bins=40, color="#c9d1d9")
    ax[1].axvline(r.real_sharpe, color="#c0392b")
    ax[1].set_title(f"Gross Sharpe vs {len(r.null)} random portfolios (p = {r.perm_p:.2f})")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "net_vs_gross.png"), dpi=150)

    print(f"{r.n_names} stocks, {r.n_periods} monthly periods, {r.first.date()} to {r.last.date()}")
    print(f"suspect daily moves masked (>25%): {r.n_masked}")
    print(r.summary.round(2).to_string())
    print()
    print(r.cost_table.round(2).to_string())
    print(f"\npermutation p-value (gross Sharpe {r.real_sharpe:.2f}): {r.perm_p:.3f}")
    print(f"capacity (last 36 months, 95th pct trade at 10% of daily value): about Rs {r.capacity_cr:,.0f} crore")
    print(f"plain-loop check: mean {r.check[0]:.6f} vs {r.vec[0]:.6f}, Sharpe {r.check[1]:.4f} vs {r.vec[1]:.4f}")


if __name__ == "__main__":
    main()
