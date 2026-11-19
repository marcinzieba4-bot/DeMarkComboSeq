# DeMark Combo + Sequential joint buy signal — tech sector, 2008 → Nov 2026

**Verdict: the joint Combo-13 + Sequential-13 buy signal shows no edge.** Tested on
point-in-time S&P 500 Information Technology members, it did slightly *worse* than
buying a random tech stock on the same day. Details in `results/report.txt`.

## Rules tested

| Item | Rule |
|---|---|
| Universe | S&P 500 members classified Information Technology **on the signal date** (membership history from `fja05680/sp500`; sector history hand-built in `data/tech_sector_map.csv`, including the 2018 and 2023 GICS reshuffles) |
| Buy setup | 9 consecutive closes below the close 4 bars earlier |
| Sequential 13 | from setup bar 9: 13 closes ≤ low two bars earlier; bar 13 also needs low ≤ close of bar 8 |
| Combo 13 (strict) | from setup bar 1: close ≤ low[-2], low ≤ low[-1], close < close[-1], close < previous count-bar close |
| Cancel / recycle | sell setup completes, or low above the setup's true high; new setup restarts the count if its range is 1–1.618× the old one |
| **Joint signal** | the second 13 prints within 10 bars of the first (one waits for the other), no sell setup in between |
| Entry | next day's open after the signal close |
| Costs | 10 bps per side |

Safeguards: signals are computed bar by bar (test in `src/test_demark.py` proves a
truncated history gives identical signals); signals use split-adjusted prices only,
returns use dividend-adjusted prices; the window of 10 bars was fixed before looking
at results and the 0/5/20 variants are all reported.

## Results

Joint signal, 246 events in 85 distinct months. "Excess" is versus the average
return of all tech members entered on the same day.

| Hold (days) | Mean return | Excess | t-stat (month-clustered) | Win rate |
|---|---|---|---|---|
| 5 | 0.32% | −0.19% | −0.5 | 55% |
| 10 | 0.26% | −0.62% | −1.3 | 53% |
| 20 | 0.93% | −0.21% | −0.3 | 55% |
| 40 | 1.91% | −0.78% | −0.8 | 52% |
| 60 | 3.46% | −1.77% | −1.7 | 59% |

- Excess is negative at every horizon, in both halves (2008–2016, 2017–2026), and under all three recycle rules.
- Same-bar joint signals (39 events) and 5- and 20-bar windows are no better.
- Sequential 13 alone *lags* peers by 1.55% over 60 days (t = −2.8): weak names kept lagging.
- As a portfolio (10 slots, idle cash earns nothing) the best joint variant made 2.6% a year against 15.8% for XLK buy-and-hold, because it is invested only 5–23% of the time and earns nothing extra while invested.
- Plain setup 9 and an RSI<30 control made more money simply by being invested more often; neither beats peers per trade.

## Study 2 — ideas from the earlier "joint 13" handoff (`src/study2.py`, `results/report2.txt`)

The handoff reported +4.15% per 21-day trade on S&P 500 longs (+6.66% when the market is
below its 200-day average). It used **today's** index members back to 2008 and data ending
10 Oct 2023. Re-tested here on point-in-time members through Nov 2026.

**Joint 13 buys, all sectors, 21-day hold**

| Group | Trades | Avg | Hit | Excess vs same-day peers (t) |
|---|---|---|---|---|
| Point-in-time members, all | 1,867 | +2.70% | 59% | +0.26% (0.6) |
| — market below 200-day | 927 | +5.33% | 65% | +1.06% (1.5) |
| — market above 200-day | 940 | +0.10% | 53% | −0.52% (−1.5) |
| — after 10 Oct 2023 (out of sample) | 356 | +2.38% | 58% | +0.05% (0.1) |
| Today's members (handoff's method), all | 1,948 | +3.45% | 61% | +0.68% (1.8) |
| Tech only, point-in-time | 245 | +1.02% | 54% | −0.29% (−0.4) |

- The regime effect is real and survives out of sample, but it is market timing: the signal does not pick better stocks than its peers.
- Using today's members adds about 0.75 points per trade and turns an insignificant excess into a "significant" one.
- Tech stocks remain the weak spot (the handoff also ranked Info Tech second-worst).

**Breadth trigger → buy XLK** (share of members with a joint buy 13 in the last 5 days; 21-day cooldown)

| Trigger | Events | XLK next 21d | Hit | XLK next 63d | Random-date 21d / 63d |
|---|---|---|---|---|---|
| ≥2% of all S&P members | 38 | +2.4% | 61% | +6.3% | +1.4% / +4.6% |
| ≥4% of all S&P members | 13 | +5.2% | 85% | +9.6% | same |
| ≥4% of tech members | 26 | +4.1% | 77% | +7.5% | same |
| ≥2% of all members, after Oct 2023 | 6 | +7.1% | 67% | +14.7% | +2.3% / +7.2% |

- The 4% triggers beat random dates in about 99% of resamples, but the events fall in roughly eight sell-off episodes, and three thresholds were tried.
- It failed in 2008: the worst 21-day result was −20% and the worst 63-day result −34% (2% trigger).
- Run as a stand-alone book (XLK for 21 days after a trigger, T-bills otherwise) it makes 4–6% a year against 15.8% for XLK buy-and-hold, because it is invested only 7–42% of the time. It is a timing overlay for adding exposure, not a replacement for holding.
- The ATR 2×/3× exit halves holding time and per-trade return and leaves Sharpe about unchanged; it did not materially cut drawdown here (−44% vs −47%).

## Limits

- **Survivorship gap.** 40 of 143 historical members (19% of member-days) have no free price history — nearly all acquired companies (Xilinx, EMC, Citrix, Red Hat…), one bankruptcy (SunEdison). Benchmarks use the same data, so the *excess* figures are comparable, but absolute returns are not exact.
- Sector history is reconstructed by hand; a few dates are approximate (Harris, Roper).
- The "DeMark" exit (stop below the 30-bar low, exit on sell setup 9) stops out 81% of trades — the stop sits too close to entry. It is reported, not tuned.
- Daily bars only; no sell side yet.

## Reproduce

```bash
pip install yfinance pandas numpy
python3 src/download.py
python3 src/test_demark.py
python3 src/backtest.py
python3 src/download_all.py
python3 src/download_etf.py   # study 6 also needs XLF XLE XLV XLI XLY XLP XLU XLB VNQ VOX and data/sector_map_all.csv
python3 src/study2.py
```

## Study 3 — every single tech signal as a trigger, buy and sell (`src/study3.py`, `results/report3.txt`)

Each joint 13 in a tech stock is traded three ways (stock, XLK, or the pair), for buy and for
sell signals. Risk statistics are in `results/s3_risk_stats.csv`; the chart is `results/s3_pnl.png`.
Buying XLK on a buy signal was the best of the six (3.0% a year, −23% drawdown), and every
variant of the sell signal lost money or made nothing. None came close to XLK buy-and-hold (15.8%).

## Study 4 — buy XLK when two of the 30 biggest tech stocks fire (`src/study4.py`, `results/report4.txt`)

Activation: a top-30 tech stock prints a joint buy 13 and a different top-30 stock did so on the
same day or within the previous 10 trading days. "Top 30" is ranked each day by trailing 250-day
dollar volume among that day's index members. Buy XLK at the next open, hold 21 days (restarted
by new activations), T-bills otherwise. 67 activation days form 19 episodes since 2008:
15 of 19 trades won (average +4.4%), but the Sep–Dec 2008 trade lost 28%. The book made 5.2% a
year invested 14% of the time, against 15.8% for XLK buy-and-hold. Dates are in
`results/s4_activations.csv`, risk statistics in `results/s4_risk_stats.csv`.

## Study 5 — trigger size, stop-loss, ban and 100-day filter (`src/study5.py`, `results/report5.txt`)

90 combinations of the study-4 XLK trigger: 1, 2 or 3 stocks; stop none/5%/8%/10%; ban after a
stop-out 0/21/63 days; XLK above/below its 100-day average or no filter. Full grid in
`results/s5_grid.csv`. Main findings: activations with XLK above its 100-day average lose money;
a 5% stop is too tight; an 8–10% stop only helps when paired with a ban, and then it roughly
halves the worst drawdown for about 1 point of yearly return. Each cell has only 13–35 trades and
the drawdown improvement comes almost entirely from 2008.

## Study 6 — every sector, combined into one book (`src/study6.py`, `results/report6.txt`)

The study-4/5 trigger is run on each of the 11 sectors (30 biggest point-in-time members,
sector ETF bought on activation) and the sleeves are combined. The plain two-stock trigger
with 1/11 of capital per sector made 4.4% a year with a −20% worst drawdown and 12% average
exposure; SPY made 10.9% with −52%. The stop, ban and 100-day filter that helped in tech made
most other sectors worse. Per-sector results: `results/s6_per_sector.csv`; joint risk
statistics: `results/s6_joint_risk.csv`; chart: `results/s6_joint_pnl.png`.
Sector labels for former members come from Yahoo's current profile and today's labels are
applied to the whole history (except tech, which uses the dated map).
