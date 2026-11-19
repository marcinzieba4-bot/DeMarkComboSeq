"""Study 3: every single joint 13 in a tech stock is a trigger (no breadth threshold).

For each BUY signal (Sequential buy 13 + Combo buy 13 within 10 bars) three trades are compared:
  1. stock      : buy the stock that fired
  2. XLK        : buy XLK instead
  3. pair       : buy XLK, short the stock
For each SELL signal the mirror image:
  1. stock      : short the stock
  2. XLK        : short XLK
  3. pair       : short XLK, buy the stock
Entry at next open, exit at the close of day 21.  All three variants take exactly the same
signals with the same sizes, so only the instrument differs.
Book: 10 slots of 10% of equity each (pair = 10% long + 10% short), compounding, idle cash in
T-bills, 10 bps per side per leg, 0.5%/yr borrow fee on short legs.
"""
import numpy as np, pandas as pd, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from demark import td_buy_signals, joint_signal
import backtest as bt

ROOT = bt.ROOT
START, OOS, H, SLOTS, COST, BORROW = "2008-01-01", "2023-10-11", 21, 10, 0.001, 0.005
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format)


def signals():
    px, member, _ = bt.load()
    data = {}
    for t, d in px.items():
        d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
        o, h, l, c = (d[k].values.astype(float) for k in ["Open", "High", "Low", "Close"])
        b = td_buy_signals(o, h, l, c)
        s = td_buy_signals(-o, -l, -h, -c)          # mirror image = sell side
        f = d["Adj Close"].values / c
        x = pd.DataFrame({"o": o * f, "c": c * f}, index=d.index)
        x["buy"] = joint_signal(b["seq13"], b["combo13"], b["sell9"], 10)
        x["sell"] = joint_signal(s["seq13"], s["combo13"], s["sell9"], 10)
        x["member"] = member[t].reindex(d.index).values
        data[t] = x
    return data


def trade_list(data, side, xlk):
    """One row per signal taken (one open trade per stock per side)."""
    rows = []
    for t, x in data.items():
        idx = x.index
        ok = (x[side] & x.member.astype(bool)).values & (idx >= START)
        busy = -1
        for i in np.flatnonzero(ok):
            e, q = i + 1, i + H
            if i <= busy or q >= len(x) or idx[e] not in xlk.index or idx[q] not in xlk.index:
                continue
            rs = x.c.values[q] / x.o.values[e] - 1
            if abs(rs) > 0.9:
                continue
            busy = q
            rows.append(dict(ticker=t, signal=idx[i], entry=idx[e], exit=idx[q], r_stock=rs,
                             r_xlk=xlk.c[idx[q]] / xlk.o[idx[e]] - 1))
    return pd.DataFrame(rows).sort_values(["entry", "ticker"]).reset_index(drop=True)


def legs(side, variant):
    """(stock weight, XLK weight) per unit of slot capital."""
    sgn = 1 if side == "buy" else -1
    return {"stock": (sgn, 0), "XLK": (0, sgn), "pair": (-sgn, sgn)}[variant]


def trade_ret(tr, side, variant):
    ws, wx = legs(side, variant)
    n_legs = abs(ws) + abs(wx)
    shorts = (ws < 0) + (wx < 0)
    return ws * tr.r_stock + wx * tr.r_xlk - 2 * COST * n_legs - shorts * BORROW * H / 252


def book(tr, data, xlk, cal, side, variant, cashr):
    ws, wx = legs(side, variant)
    cost_in = COST * (abs(ws) + abs(wx))
    px = {t: data[t].c.reindex(cal).ffill() for t in tr.ticker.unique()}
    xc = xlk.c.reindex(cal).ffill()
    by_entry = {d: g for d, g in tr.groupby("entry")}
    cash, pos, eq, gross = 1.0, [], [], []
    taken = []
    for k, d in enumerate(cal):
        cash *= 1 + cashr[k]
        for p in [p for p in pos if p["exit"] == d]:
            cash += p["size"] * (1 + p["pnl"] - cost_in); pos.remove(p)
        if d in by_entry:
            equity = cash + sum(p["size"] * (1 + p["pnl"]) for p in pos)
            for j, t in by_entry[d].iterrows():
                size = equity / SLOTS
                if len(pos) < SLOTS and cash >= size * 0.999:
                    cash -= size; taken.append(j)
                    pos.append(dict(ticker=t.ticker, exit=t.exit, size=size, pnl=-cost_in, days=0,
                                    es=data[t.ticker].o[t.entry], ex=xlk.o[t.entry]))
        for p in pos:
            p["days"] += 1
            shorts = (ws < 0) + (wx < 0)
            p["pnl"] = ws * (px[p["ticker"]].at[d] / p["es"] - 1) + wx * (xc.at[d] / p["ex"] - 1) \
                - cost_in - shorts * BORROW * p["days"] / 252
        inv = sum(p["size"] * (1 + p["pnl"]) for p in pos)
        eq.append(cash + inv)
        gross.append(sum(p["size"] for p in pos) * (abs(ws) + abs(wx)) / (cash + inv))
    return pd.Series(eq, cal), np.mean(gross), tr.loc[taken]


def risk(eq, tr_ret, gross):
    s = bt.stats(eq)
    r = eq.pct_change().dropna()
    yr = eq.resample("YE").last().pct_change().dropna()
    dd = eq / eq.cummax() - 1
    under = (dd < 0).astype(int)
    longest = under.groupby((under == 0).cumsum()).sum().max() / 252
    w = tr_ret > 0
    o = bt.stats(eq[eq.index >= OOS])
    return {"Trades": len(tr_ret), "Avg trade %": tr_ret.mean() * 100, "Hit %": w.mean() * 100,
            "Profit factor": tr_ret[w].sum() / -tr_ret[~w].sum(), "Worst trade %": tr_ret.min() * 100,
            "CAGR %": s["cagr"], "Vol %": s["vol"], "Sharpe": s["sharpe"], "Max DD %": s["maxdd"],
            "Calmar": s["cagr"] / -s["maxdd"], "Longest DD yrs": longest,
            "Worst year %": yr.min() * 100, "Best year %": yr.max() * 100,
            "Worst day %": r.min() * 100, "Avg gross expo %": gross * 100, "End value of 1": s["total"],
            "CAGR 2008-Oct23 %": bt.stats(eq[eq.index < OOS])["cagr"], "CAGR since Oct23 %": o["cagr"]}


def main():
    data = signals()
    d = pd.read_csv(ROOT / "data/prices_etf/XLK.csv", index_col=0, parse_dates=True)
    f = d["Adj Close"] / d["Close"]
    xlk = pd.DataFrame({"o": d["Open"] * f, "c": d["Adj Close"]})
    cal = xlk.index[xlk.index >= START]
    irx = pd.read_csv(ROOT / "data/prices_etf/_IRX.csv", index_col=0, parse_dates=True)["Close"]
    cashr = (irx.reindex(cal).ffill().fillna(0) / 100 / 252).values
    rows, curves = {}, {}
    for side in ("buy", "sell"):
        tr = trade_list(data, side, xlk)
        print(f"{side} signals: {len(tr)} in {tr.signal.dt.to_period('M').nunique()} months; "
              f"stock 21d {tr.r_stock.mean() * 100:.2f}%, XLK same window {tr.r_xlk.mean() * 100:.2f}%")
        by = tr.groupby(tr.signal.dt.year).agg(n=("r_stock", "size"), stock=("r_stock", lambda v: v.mean() * 100),
                                               xlk=("r_xlk", lambda v: v.mean() * 100))
        by.to_csv(ROOT / f"results/s3_{side}_by_year.csv")
        print(by.T.to_string())
        for variant in ("stock", "XLK", "pair"):
            eq, gross, taken = book(tr, data, xlk, cal, side, variant, cashr)
            name = f"{side.upper()} signal | " + {
                ("buy", "stock"): "1 buy the stock", ("buy", "XLK"): "2 buy XLK", ("buy", "pair"): "3 buy XLK, short stock",
                ("sell", "stock"): "1 short the stock", ("sell", "XLK"): "2 short XLK",
                ("sell", "pair"): "3 short XLK, buy stock"}[(side, variant)]
            rows[name] = risk(eq, trade_ret(taken, side, variant), gross)
            curves[name] = eq
        tr.to_csv(ROOT / f"results/s3_trades_{side}.csv", index=False)
    bh = xlk.c.reindex(cal) / xlk.c.reindex(cal).iloc[0]
    s = bt.stats(bh)
    tb = pd.Series(np.cumprod(1 + cashr), cal)
    out = pd.DataFrame(rows)
    print(out.to_string())
    print(f"XLK buy & hold: CAGR {s['cagr']:.1f}%, vol {s['vol']:.1f}%, Sharpe {s['sharpe']:.2f}, max DD {s['maxdd']:.0f}%; "
          f"T-bills only: CAGR {bt.stats(tb)['cagr']:.2f}%")
    out.to_csv(ROOT / "results/s3_risk_stats.csv")
    pd.DataFrame(curves).assign(**{"T-bills": tb}).to_csv(ROOT / "results/s3_equity_curves.csv")


if __name__ == "__main__":
    main()
