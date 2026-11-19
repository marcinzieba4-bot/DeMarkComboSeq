"""Study 5: variations of the XLK trigger from study 4.

Grid (all combinations are run and reported, nothing is picked in advance):
  N stocks   : 1, 2 or 3 different top-30 tech stocks with a joint buy 13 inside 10 trading days
  stop-loss  : none, 5%, 8%, 10% below the XLK entry price (intraday; a gap through the level fills at the open)
  ban        : after a stop-out, no new entry for 0, 21 or 63 trading days
  MA filter  : none / only when XLK closes above its 100-day average / only when below
Common: buy XLK at next open, hold 21 trading days, a new valid activation restarts the clock,
T-bills otherwise, 10 bps per side.
"""
import numpy as np, pandas as pd, sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt
from study3 import signals, START, OOS, COST

ROOT, HOLD = bt.ROOT, 21
pd.set_option("display.width", 280, "display.max_columns", 60, "display.max_rows", 200, "display.float_format", "{:.1f}".format)


def setup():
    data = signals(); data.pop("GOOG", None)
    px, _, _ = bt.load()
    d = pd.read_csv(ROOT / "data/prices_etf/XLK.csv", index_col=0, parse_dates=True)
    f = d["Adj Close"] / d["Close"]
    x = pd.DataFrame({"o": d["Open"] * f, "l": d["Low"] * f, "c": d["Adj Close"]})
    x["above"] = x.c > x.c.rolling(100).mean()
    cal = x.index[x.index >= "2007-01-01"]
    dv = pd.DataFrame({t: (px[t]["Close"] * px[t]["Volume"]).rolling(250, min_periods=120).mean() for t in data}).reindex(cal)
    mem = pd.DataFrame({t: v.member.astype(bool) for t, v in data.items()}).reindex(cal).fillna(False).astype(bool)
    top = dv.where(mem).rank(axis=1, ascending=False) <= 30
    sig = pd.DataFrame({t: v.buy for t, v in data.items()}).reindex(cal).fillna(False).astype(bool) & top
    recent = sig.astype(int).rolling(11, min_periods=1).max().sum(axis=1)
    x = x[x.index >= START]
    acts = {n: ((sig.any(axis=1)) & (recent >= n)).reindex(x.index).fillna(False).values for n in (1, 2, 3)}
    irx = pd.read_csv(ROOT / "data/prices_etf/_IRX.csv", index_col=0, parse_dates=True)["Close"]
    cash = (irx.reindex(x.index).ffill().fillna(0) / 100 / 252).values
    return x, acts, cash


def sim(x, act, cash, stop, ban, ma):
    o, l, c, above = x.o.values, x.l.values, x.c.values, x.above.values
    ok = act & (above if ma == "above" else ~above if ma == "below" else True)
    n = len(x)
    ret, inv = np.zeros(n), np.zeros(n, bool)
    inpos, left, entry, ban_until, val, e_k = False, 0, 0.0, -1, 1.0, 0
    trades = []
    for k in range(1, n):
        ref = c[k - 1]
        fresh = False
        if ok[k - 1]:
            if inpos:
                left = HOLD
            elif k > ban_until:
                inpos, left, entry, ref, fresh, val, e_k = True, HOLD, o[k], o[k], True, 1 - COST, k
        if not inpos:
            ret[k] = cash[k]; continue
        inv[k] = True
        px, done, why = c[k], False, ""
        if stop:
            lvl = entry * (1 - stop)
            if not fresh and o[k] <= lvl:
                px, done, why = o[k], True, "stop"
            elif l[k] <= lvl:
                px, done, why = lvl, True, "stop"
        left -= 1
        if not done and left == 0:
            done, why = True, "time"
        r = px / ref * ((1 - COST) if fresh else 1) * ((1 - COST) if done else 1) - 1
        ret[k] = r
        val *= px / ref * ((1 - COST) if done else 1)
        if done:
            trades.append((x.index[e_k], x.index[k], val - 1, why))
            inpos = False
            if why == "stop":
                ban_until = k + ban
    eq = pd.Series(np.cumprod(1 + ret), x.index)
    T = pd.DataFrame(trades, columns=["entry", "exit", "ret", "why"])
    return eq, T, inv


def row(eq, T, inv):
    s = bt.stats(eq)
    w = T.ret > 0
    return {"trades": len(T), "stops": int((T.why == "stop").sum()), "hit%": w.mean() * 100, "avg%": T.ret.mean() * 100,
            "worst%": T.ret.min() * 100, "in mkt%": inv * 100, "CAGR": s["cagr"], "Sharpe": s["sharpe"],
            "maxDD": s["maxdd"], "Calmar": s["cagr"] / -s["maxdd"], "worst yr": bt.yearly(eq).min() * 100,
            "CAGR->Oct23": bt.stats(eq[eq.index < OOS])["cagr"], "CAGR Oct23->": bt.stats(eq[eq.index >= OOS])["cagr"]}


def main():
    x, acts, cash = setup()
    rows, curves = [], {}
    for n, ma, (stop, ban) in itertools.product((1, 2, 3), ("none", "above", "below"),
                                                [(0, 0)] + [(s, b) for s in (0.05, 0.08, 0.10) for b in (0, 21, 63)]):
        eq, T, inv = sim(x, acts[n], cash, stop, ban, ma)
        lab = dict(N=n, MA100=ma, stop=f"{stop:.0%}" if stop else "none", ban=ban if stop else "-")
        rows.append({**lab, **row(eq, T, inv.mean())})
        curves[f"N{n}|{ma}|{lab['stop']}|{lab['ban']}"] = eq
        if (n, ma, stop, ban) in ((1, "none", 0, 0), (3, "none", 0, 0), (2, "none", 0.08, 63)):
            T.assign(ret=T.ret * 100).to_csv(ROOT / f"results/s5_trades_N{n}_{lab['stop']}_{lab['ban']}.csv", index=False)
    G = pd.DataFrame(rows)
    G.to_csv(ROOT / "results/s5_grid.csv", index=False)
    pd.DataFrame(curves).to_csv(ROOT / "results/s5_equity_curves.csv")
    for n in (1, 2, 3):
        print(f"\n===== N = {n} stock(s): activation days {int(acts[n].sum())} =====")
        print(G[G.N == n].drop(columns="N").to_string(index=False))
    print("\n=== average effect of each choice across the whole grid ===")
    for k in ("N", "MA100", "stop", "ban"):
        print(G.groupby(k)[["trades", "hit%", "avg%", "CAGR", "Sharpe", "maxDD", "worst yr", "CAGR Oct23->"]].mean().to_string(), "\n")
    print("top 8 by Sharpe (of 90 tested - read as a ranking, not as expected performance):")
    print(G.sort_values("Sharpe", ascending=False).head(8).to_string(index=False))
    print("activation dates N=3:", ", ".join(str(d.date()) for d in x.index[acts[3]]))


if __name__ == "__main__":
    main()
