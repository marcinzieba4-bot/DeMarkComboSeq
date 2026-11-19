"""Study 4: buy XLK when at least two of the 30 biggest tech stocks print a joint buy 13
within 10 trading days of each other.

'30 biggest' is decided point-in-time: among stocks that are S&P 500 tech members that day,
the 30 with the highest average daily dollar volume over the previous 250 days (historical
market caps are not freely available; dollar volume is known on the day and tracks size).
GOOG is dropped so Alphabet counts once.
Activation = a day on which a top-30 stock fires and a different top-30 stock fired on the
same day or in the previous 10 trading days.  Buy XLK at the next open and hold `hold`
trading days; a new activation while invested restarts the clock.  Otherwise T-bills.
"""
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt
from study3 import signals, START, OOS, COST

ROOT = bt.ROOT
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format)


def main():
    data = signals()
    data.pop("GOOG", None)
    px, _, _ = bt.load()
    d = pd.read_csv(ROOT / "data/prices_etf/XLK.csv", index_col=0, parse_dates=True)
    f = d["Adj Close"] / d["Close"]
    xo, xc = d["Open"] * f, d["Adj Close"]
    cal = xc.index[xc.index >= "2007-01-01"]
    dv = pd.DataFrame({t: (px[t]["Close"] * px[t]["Volume"]).rolling(250, min_periods=120).mean()
                       for t in data}).reindex(cal)
    mem = pd.DataFrame({t: x.member.astype(bool) for t, x in data.items()}).reindex(cal).fillna(False).astype(bool)
    top = dv.where(mem).rank(axis=1, ascending=False) <= 30
    sig = pd.DataFrame({t: x.buy for t, x in data.items()}).reindex(cal).fillna(False).astype(bool) & top
    recent = sig.astype(int).rolling(11, min_periods=1).max()          # fired today or in last 10 days
    act = (sig.any(axis=1)) & (recent.sum(axis=1) >= 2)
    act = act[act.index >= START]
    days = act.index[act]
    pos = {dt: i for i, dt in enumerate(xc.index)}

    rows = []
    for dt in days:
        i = pos[dt]
        names = sig.columns[recent.loc[dt] > 0]
        r = {"activation": dt.date(), "stocks": " ".join(names)}
        for h in (21, 63):
            r[f"XLK {h}d %"] = (xc.iloc[i + h] / xo.iloc[i + 1] - 1) * 100 if i + h < len(xc) else np.nan
        lo = xc.iloc[i + 1:i + 22].min() / xo.iloc[i + 1] - 1 if i + 1 < len(xc) else np.nan
        r["worst dip in 21d %"] = lo * 100
        rows.append(r)
    A = pd.DataFrame(rows)
    A["new episode"] = np.r_[True, np.diff([pos[x] for x in days]) > 21]
    print(f"signals in top-30 stocks since 2008: {int(sig[sig.index >= START].values.sum())}; "
          f"activation days: {len(A)}; separate episodes (gap > 21 days): {int(A['new episode'].sum())}")
    print(A.to_string(index=False))
    A.to_csv(ROOT / "results/s4_activations.csv", index=False)

    c = xc[xc.index >= START]
    base = {h: (xc.shift(-h) / xo.shift(-1) - 1)[xc.index >= START] for h in (21, 63)}
    ep = A[A["new episode"]]
    print("\nforward XLK return, % (mean / hit rate / random-date mean / share of random samples at least as good)")
    for lab, tab in (("all activation days", A), ("first day of each episode", ep),
                     ("episodes after 10 Oct 2023", ep[pd.to_datetime(ep.activation) >= OOS])):
        for h in (21, 63):
            v = tab[f"XLK {h}d %"].dropna()
            b = base[h].dropna() * 100
            if lab.endswith("2023"):
                b = b[b.index >= OOS]
            p = (b.sample(len(v) * 4000, replace=True, random_state=2).values.reshape(4000, -1).mean(1) >= v.mean()).mean()
            print(f"  {lab:28s} {h:2d}d: n={len(v):2d} mean {v.mean():5.2f}  hit {100 * (v > 0).mean():3.0f}%  "
                  f"random {b.mean():4.2f}  p={p:.3f}  worst {v.min():6.2f}")

    irx = pd.read_csv(ROOT / "data/prices_etf/_IRX.csv", index_col=0, parse_dates=True)["Close"]
    cash = (irx.reindex(c.index).ffill().fillna(0) / 100 / 252).values
    out, curves = {}, {}
    for hold in (21, 63):
        inpos = np.zeros(len(c), bool)
        ret = np.zeros(len(c))
        left, trades, cur = 0, [], None
        ci = c.index
        for k in range(1, len(c)):
            dt_prev = ci[k - 1]
            was = left > 0
            if act.get(dt_prev, False):
                left = hold                                   # enter / restart at today's open
            if left > 0:
                o, cl, pc = xo[ci[k]], c.iloc[k], c.iloc[k - 1]
                if not was:                                    # fresh entry at the open
                    ret[k] = cl / o * (1 - COST) - 1 + cash[k] * 0
                    cur = dict(entry=ci[k], val=cl / o * (1 - COST))
                else:
                    ret[k] = cl / pc - 1
                    cur["val"] *= cl / pc
                inpos[k] = True
                left -= 1
                if left == 0:
                    ret[k] = (1 + ret[k]) * (1 - COST) - 1
                    cur["val"] *= 1 - COST
                    trades.append(dict(entry=cur["entry"].date(), exit=ci[k].date(), ret=cur["val"] - 1))
            else:
                ret[k] = cash[k]
        eq = pd.Series(np.cumprod(1 + ret), ci)
        T = pd.DataFrame(trades)
        s, yr = bt.stats(eq), bt.yearly(eq)
        w = T.ret > 0
        out[f"XLK after activation, hold {hold}d"] = {
            "Trades (episodes)": len(T), "Avg trade %": T.ret.mean() * 100, "Hit %": w.mean() * 100,
            "Profit factor": T.ret[w].sum() / -T.ret[~w].sum(), "Worst trade %": T.ret.min() * 100,
            "Best trade %": T.ret.max() * 100, "Time invested %": inpos.mean() * 100,
            "CAGR %": s["cagr"], "Vol %": s["vol"], "Sharpe": s["sharpe"], "Max DD %": s["maxdd"],
            "Calmar": s["cagr"] / -s["maxdd"], "Worst year %": yr.min() * 100, "Best year %": yr.max() * 100,
            "Return while invested, %/yr": ((1 + pd.Series(ret)[inpos]).prod() ** (252 / inpos.sum()) - 1) * 100,
            "CAGR 2008-Oct23 %": bt.stats(eq[eq.index < OOS])["cagr"],
            "CAGR since Oct23 %": bt.stats(eq[eq.index >= OOS])["cagr"], "End value of 1": s["total"]}
        curves[f"hold {hold}d"] = eq
        T.to_csv(ROOT / f"results/s4_trades_hold{hold}.csv", index=False)
        if hold == 21:
            print("\ntrades, hold 21d:"); print(T.assign(ret=T.ret * 100).to_string(index=False))
    bh = c / c.iloc[0]
    s, yr = bt.stats(bh), bt.yearly(bh)
    out["XLK buy & hold"] = {"Time invested %": 100, "CAGR %": s["cagr"], "Vol %": s["vol"], "Sharpe": s["sharpe"],
                             "Max DD %": s["maxdd"], "Calmar": s["cagr"] / -s["maxdd"], "Worst year %": yr.min() * 100,
                             "Best year %": yr.max() * 100, "Return while invested, %/yr": s["cagr"],
                             "CAGR 2008-Oct23 %": bt.stats(bh[bh.index < OOS])["cagr"],
                             "CAGR since Oct23 %": bt.stats(bh[bh.index >= OOS])["cagr"], "End value of 1": s["total"]}
    R = pd.DataFrame(out)
    print("\n=== RISK STATS (100% XLK while active, T-bills otherwise, 10 bps per side) ===")
    print(R.to_string())
    R.to_csv(ROOT / "results/s4_risk_stats.csv")
    pd.DataFrame(curves).assign(XLK=bh).to_csv(ROOT / "results/s4_equity_curves.csv")
    print("\ncurrent top-30 by dollar volume:", " ".join(top.columns[top.iloc[-1]]))


if __name__ == "__main__":
    main()
