"""Study 8: trade WITH the move while a Sequential countdown is young, then fade it via XLK.

Tech stocks (point-in-time S&P 500 tech members), 2008 -> Nov 2026.
Down leg:  a stock prints bar K (4 or 5) of a Sequential BUY countdown (it is falling)
           -> short the stock at the next open.
           When the joint buy 13 fires (Sequential + Combo within 10 bars) -> keep the short and
           buy XLK in equal size at the next open; hold the pair HEDGE days, then close both.
Up leg:    mirror image.  Bar K of a Sequential SELL countdown -> buy the stock; at the joint
           sell 13 -> short XLK against it for HEDGE days, then close both.
Other exits of the stock position (at the next open): the countdown is cancelled; the Sequential
13 printed but no joint signal followed within 10 bars; MAXH bars passed.
Book: 20 slots of 5% of equity, compounding, idle cash in T-bills, 10 bps per side per leg,
0.5%/yr borrow fee on shorts.
"""
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt
from demark import td_buy_signals, joint_signal

ROOT = bt.ROOT
START, OOS, HEDGE, MAXH, SLOTS, COST, BORROW = "2008-01-01", "2023-10-11", 21, 90, 20, 0.001, 0.005
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format)


def load():
    px, member, _ = bt.load()
    px.pop("GOOG", None)
    out = {}
    for t, d in px.items():
        d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
        o, h, l, c = (d[k].values.astype(float) for k in ["Open", "High", "Low", "Close"])
        f = d["Adj Close"].values / c
        x = pd.DataFrame({"o": o * f, "c": c * f, "member": member[t].reindex(d.index).values.astype(bool)}, index=d.index)
        for side, s in (("down", td_buy_signals(o, h, l, c)), ("up", td_buy_signals(-o, -l, -h, -c))):
            x[f"{side}_inc"], x[f"{side}_cancel"], x[f"{side}_13"] = s["seq_inc"], s["seq_cancel"], s["seq13"]
            x[f"{side}_joint"] = joint_signal(s["seq13"], s["combo13"], s["sell9"], 10)
        out[t] = x
    return out


def make_trades(data, xlk, side, K):
    d = -1 if side == "down" else 1           # direction of the stock position
    rows = []
    for t, x in data.items():
        idx, o, c = x.index, x.o.values, x.c.values
        inc, cancel, s13, joint = (x[f"{side}_{k}"].values for k in ("inc", "cancel", "13", "joint"))
        n, busy = len(x), -1
        for i in np.flatnonzero((inc == K) & x.member.values & (idx >= START)):
            if i <= busy or i + 2 >= n:
                continue
            e, t13, why, hs = i + 1, None, None, None
            for j in range(e, n - 1):
                if joint[j]:
                    why, hs = "joint", j + 1; break
                if cancel[j]:
                    why = "cancelled"; break
                if s13[j]:
                    t13 = j
                if t13 is not None and j - t13 >= 10:
                    why = "13 without combo"; break
                if j - e + 1 >= MAXH:
                    why = "max hold"; break
            if why is None:
                continue                                            # still open at the end of data
            if why == "joint":
                q = min(hs + HEDGE - 1, n - 1)
                if idx[hs] not in xlk.index or idx[q] not in xlk.index:
                    continue
                p_sig = o[hs]                                       # stock price when the hedge goes on
                r1 = d * (p_sig / o[e] - 1)
                r_stock2 = d * (c[q] / p_sig - 1) * (p_sig / o[e])  # still per unit of entry notional
                r_xlk = -d * (xlk.c[idx[q]] / xlk.o[idx[hs]] - 1)
                x_ph1 = d * (xlk.o[idx[hs]] / xlk.o[idx[e]] - 1) if idx[e] in xlk.index else np.nan
                exit_px, exit_dt, legs = c[q], idx[q], 2
            else:
                q = j + 1
                r1, r_stock2, r_xlk = d * (o[q] / o[e] - 1), 0.0, 0.0
                x_ph1 = d * (xlk.o[idx[q]] / xlk.o[idx[e]] - 1) if idx[e] in xlk.index and idx[q] in xlk.index else np.nan
                exit_px, exit_dt, legs, hs = o[q], idx[q], 1, None
            days = q - e + 1
            short_days = days if d < 0 else (q - hs + 1 if hs else 0)
            cost = 2 * COST * legs + BORROW * short_days / 252
            if abs(r1 + r_stock2) > 0.9:
                continue
            busy = q
            rows.append(dict(ticker=t, entry=idx[e], hedge=idx[hs] if hs else pd.NaT, exit=exit_dt, why=why, bars=days,
                             r_trend=r1, r_stock_hedged=r_stock2, r_xlk=r_xlk, ret=r1 + r_stock2 + r_xlk - cost,
                             xlk_same_trend=x_ph1, e_px=o[e], exit_px=exit_px, d=d))
    return pd.DataFrame(rows).sort_values(["entry", "ticker"]).reset_index(drop=True)


def book(tr, data, xlk, cal, cashr):
    px = {t: data[t].c.reindex(cal).ffill() for t in tr.ticker.unique()}
    xc = xlk.c.reindex(cal).ffill()
    by_entry = {dt: g for dt, g in tr.groupby("entry")}
    cash, pos, eq, gross, net, taken = 1.0, [], [], [], [], []
    for k, dt in enumerate(cal):
        cash *= 1 + cashr[k]
        for p in [p for p in pos if p["exit"] == dt]:
            cash += p["size"] * (1 + p["ret"]); pos.remove(p)
        if dt in by_entry:
            equity = cash + sum(p["size"] * (1 + p["m"]) for p in pos)
            for j, t in by_entry[dt].iterrows():
                size = equity / SLOTS
                if len(pos) < SLOTS and cash >= size * 0.999:
                    cash -= size; taken.append(j)
                    pos.append(dict(t, size=size, m=0.0, xh=xlk.o.get(t.hedge) if pd.notna(t.hedge) else None))
        g = nt = 0.0
        for p in pos:
            m = p["d"] * (px[p["ticker"]].at[dt] / p["e_px"] - 1)
            g += p["size"]; nt += p["d"] * p["size"]
            if p["xh"] is not None and dt >= p["hedge"]:
                m += -p["d"] * (xc.at[dt] / p["xh"] - 1)
                g += p["size"]; nt -= p["d"] * p["size"]
            p["m"] = m
        e = cash + sum(p["size"] * (1 + p["m"]) for p in pos)
        eq.append(e); gross.append(g / e); net.append(nt / e)
    return pd.Series(eq, cal), np.array(gross), np.array(net), tr.loc[taken]


def stats_row(eq, T, gross, net, spy_r):
    s, yr = bt.stats(eq), bt.yearly(eq)
    w = T.ret > 0
    j = T[T.why == "joint"]
    r = eq.pct_change().dropna()
    return {"Trades": len(T), "Avg trade %": T.ret.mean() * 100, "Hit %": w.mean() * 100,
            "Profit factor": T.ret[w].sum() / -T.ret[~w].sum(), "Worst trade %": T.ret.min() * 100, "Avg bars held": T.bars.mean(),
            "Reached joint 13 %": len(j) / len(T) * 100, "Avg trend phase %": T.r_trend.mean() * 100,
            "Trend phase, XLK same days %": T.xlk_same_trend.mean() * 100,
            "Hedge phase: stock leg %": j.r_stock_hedged.mean() * 100, "Hedge phase: XLK leg %": j.r_xlk.mean() * 100,
            "CAGR %": s["cagr"], "Vol %": s["vol"], "Sharpe": s["sharpe"], "Max DD %": s["maxdd"], "Calmar": s["cagr"] / -s["maxdd"],
            "Worst year %": yr.min() * 100, "Best year %": yr.max() * 100, "Losing years": int((yr < 0).sum()),
            "Avg gross expo %": gross.mean() * 100, "Avg net expo %": net.mean() * 100,
            "Beta to XLK": np.cov(r, spy_r.reindex(r.index).fillna(0))[0, 1] / spy_r.var(),
            "CAGR 2008-Oct23 %": bt.stats(eq[eq.index < OOS])["cagr"], "CAGR since Oct23 %": bt.stats(eq[eq.index >= OOS])["cagr"],
            "End value of 1": s["total"]}


def main():
    data = load()
    d = pd.read_csv(ROOT / "data/prices_etf/XLK.csv", index_col=0, parse_dates=True)
    f = d["Adj Close"] / d["Close"]
    xlk = pd.DataFrame({"o": d["Open"] * f, "c": d["Adj Close"]})
    cal = xlk.index[xlk.index >= START]
    irx = pd.read_csv(ROOT / "data/prices_etf/_IRX.csv", index_col=0, parse_dates=True)["Close"]
    cashr = (irx.reindex(cal).ffill().fillna(0) / 100 / 252).values
    xr = xlk.c.reindex(cal).pct_change().dropna()
    out, curves = {}, {}
    for K in (4, 5):
        legs = {}
        for side, lab in (("down", "short stock at buy-countdown"), ("up", "buy stock at sell-countdown")):
            tr = make_trades(data, xlk, side, K)
            legs[side] = tr
            eq, g, nt, T = book(tr, data, xlk, cal, cashr)
            name = f"bar {K}: {lab}"
            out[name], curves[name] = stats_row(eq, T, g, nt, xr), eq
            print(f"{name}: exits {tr.why.value_counts().to_dict()}")
            tr.drop(columns=["e_px", "exit_px"]).to_csv(ROOT / f"results/s8_trades_{side}_bar{K}.csv", index=False)
            if K == 4:
                by = tr.groupby(tr.entry.dt.year).agg(n=("ret", "size"), avg=("ret", lambda v: v.mean() * 100))
                print(by.T.round(1).to_string())
        both = pd.concat(legs.values()).sort_values(["entry", "ticker"]).reset_index(drop=True)
        eq, g, nt, T = book(both, data, xlk, cal, cashr)
        out[f"bar {K}: both legs"], curves[f"bar {K}: both legs"] = stats_row(eq, T, g, nt, xr), eq
    J = pd.DataFrame(out)
    print("\n=== RISK STATS (20 slots of 5%) ==="); print(J.to_string())
    bh = bt.stats(xlk.c.reindex(cal))
    print(f"XLK buy & hold: CAGR {bh['cagr']:.1f}%, max DD {bh['maxdd']:.0f}%, Sharpe {bh['sharpe']:.2f}")
    J.to_csv(ROOT / "results/s8_risk_stats.csv")
    pd.DataFrame(curves).assign(**{"T-bills": np.cumprod(1 + cashr)}).to_csv(ROOT / "results/s8_equity_curves.csv")


if __name__ == "__main__":
    main()
