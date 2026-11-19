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
```
