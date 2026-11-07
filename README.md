# Net-of-Cost Signal Research

Does a simple momentum signal on Indian large caps survive realistic costs, and
can it be told apart from luck? One signal is fixed in advance and tested the
way a live research process would have to defend it.

**Headline result: it does not.** Gross return is 1.4% a year, falling to 0.8%
after 10 bps one-way costs, with a net Sharpe of 0.09 and a Newey-West t of
0.34. A permutation test cannot separate it from random portfolios (p = 0.30),
and the held-out final 40% of the sample loses money. The honest label is
**no evidence of a tradeable edge**. The value of this project is the process
that reached that answer, not the answer.

## What is tested

| Choice | Setting (fixed before looking at results) |
| --- | --- |
| Universe | 48 NSE large caps, daily bars, one CSV per stock |
| Signal | 12-1 month momentum: cumulative log return from t-252 to t-21 |
| Portfolio | Dollar-neutral long-short, top 10 and bottom 10, equal weight |
| Rebalance | Every 21 trading days, trading the day after the signal close |
| Costs | 10 bps one way on turnover (sensitivity from 0 to 50 bps) |
| Sample | Aug 2011 to Sep 2026, 178 monthly periods (starts once 40 names have a signal) |
| Holdout | Final 40% of periods, reported separately, never used to choose anything |

## Results

![gross vs net and permutation test](results/net_vs_gross.png)

| Segment | Periods | Gross % a year | Net % a year | Net Sharpe | Net t (Newey-West) | Max drawdown % |
| --- | --- | --- | --- | --- | --- | --- |
| Full sample | 178 | 1.36 | 0.79 | 0.09 | 0.34 | -23.1 |
| In-sample (first 60%) | 106 | 3.41 | 2.85 | 0.30 | 0.96 | -17.4 |
| Holdout (last 40%) | 72 | -1.66 | -2.24 | -0.26 | -0.63 | -21.5 |

Average turnover is 0.47 of gross exposure per month.

| One-way cost (bps) | 0 | 5 | 10 | 20 | 30 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| Net % a year | 1.36 | 1.07 | 0.79 | 0.23 | -0.34 | -1.47 |
| Net Sharpe | 0.15 | 0.12 | 0.09 | 0.02 | -0.04 | -0.16 |

Break-even cost is about 24 bps one way. **Capacity:** over the last 36 months,
the 95th percentile trade stays under 10% of a stock's average daily traded
value up to roughly Rs 238 crore of capital, using a 20-day average of price
times volume.

## How the checks work

* **No lookahead.** Weights at a rebalance use returns up to that close only
  and earn the following 21 days. A test rewrites the entire future and
  confirms the signal does not move.
* **Costs.** Turnover is `sum |w_t - w_(t-1)|`, charged at the stated bps.
* **Permutation test.** At every rebalance the cross-section of signal values
  is shuffled (stocks keep their returns, the signal loses its information),
  the portfolio is rebuilt and gross Sharpe measured, 2,000 times. Gross is
  used on purpose: random portfolios churn far more than momentum, so a net
  comparison would reward low turnover and call it skill.
* **Newey-West t-statistic** (3 lags) on monthly net returns, because
  consecutive holding periods are not independent.
* **Independent recomputation.** `plain_loop_check` recomputes the net mean
  and Sharpe with nested loops and no shared helpers. Both agree to six
  decimals, and a test enforces it.
* **Data cleaning.** One-day moves beyond 25% are treated as unadjusted splits
  or bad prints and masked (37 in this data).

## Limits, stated plainly

* **Survivorship bias.** The universe is today's large caps. Stocks that fell
  out of the index are missing, which flatters any long-only tilt and makes
  this result optimistic, not pessimistic.
* **Prices may not be fully adjusted for dividends.** Returns here are price
  returns.
* **Costs are an assumption.** 10 bps one way stands in for brokerage, taxes
  and impact. A short book in Indian cash equities is not freely available, so
  a real version would use futures.
* **Monthly periods ignore intra-month weight drift.** The turnover and return
  approximation is small at this frequency but not zero.
* **One signal, one window, one portfolio size.** Nothing was searched, so
  there is no multiple-testing problem, and also no claim beyond this setup.
* **Capacity is a rough participation rule**, not a market-impact model, and
  volume in the early years of the data is too thin to use.

## Run it

```bash
pip install -r requirements.txt
pytest -q                                  # 14 tests
python research.py --data-dir path/to/csvs --n-perm 2000
```

Each CSV needs `Date`, `Close` and `Volume` columns (and optionally `Series`,
where only `EQ` rows are kept). Outputs go to `results/`.

## Next steps

* Add point-in-time index membership to remove survivorship bias.
* Replace the flat cost with spread plus an impact term that grows with
  participation.
* Test a futures-based short leg and a sector-neutral version.
* Add a deflated Sharpe calculation if more than one signal is tried.
