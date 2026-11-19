"""Study 13: the all-sector ETF book (4.4% a year, -20% drawdown; study 6 version A) next to the
same strategy expressed with put spreads (study 10: sell at-the-money / buy 20-delta put, 30 days,
on the same 1/11 notional per active sector), and the two combined.  Adds recovery statistics.
Idle money earns the 13-week T-bill rate in every book (the put-spread book earns it on all capital).
Option prices are modelled, not quotes.
"""
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt, study6
from study12 import recovery_stats

ROOT = bt.ROOT
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format)


def main():
    e = pd.read_csv(ROOT / "results/s10_equity_curves.csv", index_col=0, parse_dates=True)
    cal = e.index
    rf = pd.read_csv(ROOT / "data/prices_etf/_IRX.csv", index_col=0, parse_dates=True)["Close"].reindex(cal).ffill().fillna(0) / 100 / 252
    r_etf = e["ETF book (base)"].pct_change().fillna(e["ETF book (base)"].iloc[0] - 1)
    r_ps = e["Spread ATM/20-delta 30d"].pct_change().fillna(e["Spread ATM/20-delta 30d"].iloc[0] - 1)
    spy = study6.etf_frame("SPY").c.reindex(cal)
    books = {"ETF book": (1 + r_etf).cumprod(), "Put-spread book": (1 + r_ps).cumprod(),
             "50/50 mix": (1 + 0.5 * r_etf + 0.5 * r_ps).cumprod(),
             "Both at full size": (1 + r_etf + r_ps - rf).cumprod(),
             "SPY buy & hold": spy / spy.iloc[0], "Money market only": (1 + rf).cumprod()}
    out, dds = {}, {}
    for k, eq in books.items():
        x = eq.pct_change().fillna(0) - rf
        d = study6.risk(eq)
        d.update({"Return above money market %/yr": x.mean() * 252 * 100,
                  "Sharpe above money market": x.mean() / x.std() * np.sqrt(252) if x.std() > 1e-9 else np.nan})
        rs, D = recovery_stats(eq)
        d.update(rs); out[k], dds[k] = d, D
    J = pd.DataFrame(out)
    print(J.to_string()); J.to_csv(ROOT / "results/s13_risk_stats.csv")
    for k in ("ETF book", "Put-spread book", "50/50 mix", "Both at full size", "SPY buy & hold"):
        print(f"\nfive deepest drawdowns - {k}")
        print(dds[k].sort_values("depth").head(5).to_string(index=False))
        dds[k].sort_values("depth").to_csv(ROOT / f"results/s13_drawdowns_{k.lower().replace(' ', '_').replace('/', '_').replace('&', 'and')}.csv", index=False)
    for k in ("ETF book", "Put-spread book", "50/50 mix"):
        D = dds[k]
        print(f"\n{k}: recovery time by depth of drawdown (calendar days, trough to new high)")
        for lo, hi in ((-2, -5), (-5, -10), (-10, -100)):
            g = D[(D.depth <= lo) & (D.depth > hi)]
            print(f"  {abs(lo)}-{abs(hi) if hi > -100 else '+'}%: n={len(g)}, median {g.recovery_days.median():.0f}, longest {g.recovery_days.max():.0f}, "
                  f"median days below high {g.under_water_days.median():.0f}")
    Y = pd.DataFrame({k: bt.yearly(v) * 100 for k, v in books.items()}); Y.index = Y.index.year
    print("\nyearly returns %:"); print(Y.T.round(1).to_string()); Y.to_csv(ROOT / "results/s13_yearly.csv")
    pd.DataFrame(books).to_csv(ROOT / "results/s13_equity_curves.csv")
    print("\ndaily correlation above cash, ETF vs put spreads:", round(np.corrcoef(r_etf - rf, r_ps - rf)[0, 1], 2))


if __name__ == "__main__":
    main()
