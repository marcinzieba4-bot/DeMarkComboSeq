"""Second study: ideas from the earlier 'joint 13' handoff, re-tested with
point-in-time index membership and an untouched out-of-sample period.

  1. regime filter  : buy joint 13s only when the market is below its 200-day average
  2. breadth trigger: many members firing at once -> buy the sector ETF
  3. ATR exit       : stop 2x ATR(14), target 3x ATR(14), max 21 days
  4. survivorship   : same test on today's members (as the handoff did) vs point-in-time
The handoff's data ended 10 Oct 2023, so everything after that date is out of sample.
"""
import numpy as np, pandas as pd, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from demark import td_buy_signals, joint_signal
import backtest as bt

ROOT = bt.ROOT
START, OOS = "2008-01-01", "2023-10-11"
H = 21
COST = 0.001
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format)


def frame(d, member):
    o, h, l, c = (d[k].values.astype(float) for k in ["Open", "High", "Low", "Close"])
    s = td_buy_signals(o, h, l, c)
    f = d["Adj Close"].values / c
    x = pd.DataFrame({"o": o * f, "h": h * f, "l": l * f, "c": c * f}, index=d.index)
    pc = np.r_[np.nan, x.c.values[:-1]]
    tr = np.maximum(x.h - x.l, np.maximum(abs(x.h - pc), abs(x.l - pc)))
    x["atr"] = tr.rolling(14).mean()
    x["member"] = member
    x["joint"] = joint_signal(s["seq13"], s["combo13"], s["sell9"], 10)
    x["ret1"] = x.c.pct_change()
    x["bad"] = x.ret1.abs().rolling(H + 2).max().shift(-(H + 1)) > 0.6   # bad tick / spin-off guard
    return x


def load_all():
    mem = pd.read_csv(ROOT / "data/sp500_ticker_start_end.csv", parse_dates=["start_date", "end_date"])
    cur = set(pd.read_csv(ROOT / "data/sp500_current.csv").Symbol)
    data, n_all = {}, 0
    for t, rows in mem.groupby("ticker"):
        if not (rows.end_date.isna() | (rows.end_date >= START)).any():
            continue
        n_all += 1
        f = ROOT / f"data/prices_all/{t}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
        m = pd.Series(False, index=d.index)
        for _, r in rows.iterrows():
            b = r.end_date if pd.notna(r.end_date) else pd.Timestamp("2100-01-01")
            m[(d.index >= r.start_date) & (d.index < b)] = True
        x = frame(d, m.values)
        x["current"] = t in cur
        data[t] = x
    return data, n_all


def load_tech():
    px, member, _ = bt.load()
    return {t: frame(d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)],
                     member[t].reindex(d.index).values) for t, d in px.items()}


def events(data, pit=True):
    """member-days with 21-day forward total return (entry next open, exit close of day 21)."""
    rows = []
    for t, x in data.items():
        r = pd.DataFrame({"r": x.c.shift(-H) / x.o.shift(-1) - 1, "joint": x.joint, "bad": x.bad,
                          "member": x.member, "ticker": t}, index=x.index)
        keep = (x.index >= START) & (x.member.values if pit else x.get("current", True))
        rows.append(r[keep])
    a = pd.concat(rows).rename_axis("date").reset_index()
    a = a[~a.bad.fillna(False)]
    a["x"] = a.r - a.groupby("date").r.transform("mean")
    return a


def line(ev, label):
    if len(ev) == 0:
        return dict(group=label, n=0)
    w = ev.r > 0
    return dict(group=label, n=len(ev), months=ev.date.dt.to_period("M").nunique(),
                avg=ev.r.mean() * 100, med=ev.r.median() * 100, hit=w.mean() * 100,
                pf=ev.r[w].sum() / -ev.r[~w].sum(), excess=ev.x.mean() * 100,
                t_excess=bt.cluster_t(ev.rename(columns={"x": "xx"}), "xx"))


def regime_series():
    spy = pd.read_csv(ROOT / "data/prices_etf/SPY.csv", index_col=0, parse_dates=True)["Close"]
    return (spy < spy.rolling(200).mean()).rename("bear")


def study_events(a, name, bear):
    a = a.join(bear, on="date")
    ev = a[a.joint].dropna(subset=["r"])
    rows = [line(a.dropna(subset=["r"]), "ALL member-days (unconditional)"),
            line(ev, "joint 13 buys"),
            line(ev[ev.bear], "  market below 200d"),
            line(ev[~ev.bear], "  market above 200d"),
            line(a[a.bear].dropna(subset=["r"]), "ALL member-days, market below 200d"),
            line(ev[ev.date < OOS], "  in-sample  (to 10 Oct 2023)"),
            line(ev[ev.date >= OOS], "  OUT-OF-SAMPLE (after)"),
            line(ev[(ev.date >= OOS) & ev.bear], "  OUT-OF-SAMPLE, below 200d"),
            line(a[a.date >= OOS].dropna(subset=["r"]), "ALL member-days, out-of-sample")]
    out = pd.DataFrame(rows).set_index("group")
    print(f"\n=== {name}: 21-day return after joint 13 buy (entry next open) ===")
    print(out.to_string())
    out.to_csv(ROOT / f"results/s2_events_{name.split()[0].lower()}.csv")
    return a


# ------------------------------------------------------------------ breadth trigger
def breadth(a, etf, name):
    """Share of members with a joint buy 13 in the trailing 5 days.  Trigger = first day the
    share reaches the threshold (then 21-day cooldown).  Buy the ETF at next open."""
    d = pd.read_csv(ROOT / f"data/prices_etf/{etf}.csv", index_col=0, parse_dates=True)
    f = d["Adj Close"] / d["Close"]
    o, c = d["Open"] * f, d["Adj Close"]
    cnt = a.groupby("date").agg(sig=("joint", "sum"), n=("joint", "size"))
    share = (cnt.sig.rolling(5).sum() / cnt.n).reindex(c.index).fillna(0)
    fw = {h: (c.shift(-h) / o.shift(-1) - 1) for h in (21, 63)}
    base = {h: fw[h][c.index >= START] for h in fw}
    rows = []
    for thr in (0.01, 0.02, 0.04):
        trig, last = [], None
        for dt in share.index[share.index >= START]:
            if share[dt] >= thr and (last is None or (c.index.get_loc(dt) - last) > 21):
                trig.append(dt); last = c.index.get_loc(dt)
        trig = pd.DatetimeIndex(trig)
        for lab, tt in (("all", trig), ("out-of-sample", trig[trig >= OOS])):
            r = {"etf": etf, "threshold": f"{thr:.0%}", "period": lab, "n": len(tt)}
            for h in fw:
                v = fw[h].reindex(tt).dropna()
                b = base[h] if lab == "all" else base[h][base[h].index >= OOS]
                r[f"avg{h}"] = v.mean() * 100; r[f"hit{h}"] = (v > 0).mean() * 100
                r[f"base{h}"] = b.mean() * 100; r[f"worst{h}"] = v.min() * 100 if len(v) else np.nan
                # how unusual vs random dates
                r[f"pct{h}"] = (b.sample(len(v) * 3000, replace=True, random_state=1).values
                                .reshape(3000, -1).mean(1) >= v.mean()).mean() if len(v) else np.nan
            rows.append(r)
            if lab == "all" and thr == 0.02:
                print(f"{name}->{etf} trigger dates at 2%:", ", ".join(str(x.date()) for x in tt))
    return pd.DataFrame(rows), share


def etf_book(share, etf, thr, hold=21):
    """In the ETF for `hold` days after each trigger day (extended by new triggers), else T-bills."""
    d = pd.read_csv(ROOT / f"data/prices_etf/{etf}.csv", index_col=0, parse_dates=True)
    irx = pd.read_csv(ROOT / "data/prices_etf/_IRX.csv", index_col=0, parse_dates=True)["Close"]
    c = d["Adj Close"][d.index >= START]
    r = c.pct_change().fillna(0)
    cash = (irx.reindex(c.index).ffill().fillna(0) / 100 / 252)
    sig = (share.reindex(c.index).fillna(0) >= thr)
    inpos = sig.shift(1, fill_value=False).rolling(hold, min_periods=1).max().astype(bool)
    turn = inpos.astype(int).diff().abs().fillna(0)
    eq = (1 + np.where(inpos, r, cash) - turn * COST).cumprod()
    return pd.Series(eq, c.index), inpos.mean()


# ------------------------------------------------------------------ stock book
def trades(data, bear, exit_rule, regime):
    out = []
    for t, x in data.items():
        idx, o, h, l, c, atr = x.index, x.o.values, x.h.values, x.l.values, x.c.values, x.atr.values
        ok = (x.joint & x.member).values & (idx >= START)
        if regime:
            ok &= bear.reindex(idx).fillna(False).values
        busy = -1
        for i in np.flatnonzero(ok):
            if i <= busy or i + 1 >= len(x) or not np.isfinite(atr[i]):
                continue
            e = i + 1
            last = min(e + H - 1, len(x) - 1)
            q, px, why = last, c[last], "time"
            if exit_rule == "atr":
                stop, tgt = o[e] - 2 * atr[i], o[e] + 3 * atr[i]
                for j in range(e, last + 1):
                    if j > e and o[j] <= stop:
                        q, px, why = j, o[j], "stop"; break
                    if l[j] <= stop:
                        q, px, why = j, stop, "stop"; break
                    if j > e and o[j] >= tgt:
                        q, px, why = j, o[j], "target"; break
                    if h[j] >= tgt:
                        q, px, why = j, tgt, "target"; break
            ret = px / o[e] * (1 - COST) / (1 + COST) - 1
            if abs(ret) > 0.9:
                continue
            busy = q
            out.append(dict(ticker=t, signal=idx[i], entry=idx[e], exit=idx[q], ret=ret,
                            bars=q - e + 1, why=why, dd60=0.0))
    return pd.DataFrame(out).sort_values(["entry", "ticker"]).reset_index(drop=True)


def book(tr, data, cal, slots):
    """Fixed number of equal slots, compounding, idle cash in T-bills."""
    irx = pd.read_csv(ROOT / "data/prices_etf/_IRX.csv", index_col=0, parse_dates=True)["Close"]
    cashr = (irx.reindex(cal).ffill().fillna(0) / 100 / 252).values
    close = {t: data[t].c.reindex(cal).ffill() for t in tr.ticker.unique()}
    by_entry = {d: g for d, g in tr.groupby("entry")}
    cash, pos, eq, ex, taken = 1.0, [], [], [], 0
    for k, d in enumerate(cal):
        cash *= 1 + cashr[k]
        for p in [p for p in pos if p["exit"] == d]:
            cash += p["cost"] * (1 + p["ret"]); pos.remove(p)
        if d in by_entry:
            equity = cash + sum(p["cost"] * p["m"] for p in pos)
            for _, t in by_entry[d].iterrows():
                size = equity / slots
                if len(pos) < slots and cash >= size * 0.999 and t.exit > d:
                    cash -= size; taken += 1
                    pos.append(dict(t, cost=size, m=1.0, e=data[t.ticker].o.get(t.entry)))
        for p in pos:
            p["m"] = close[p["ticker"]].at[d] / p["e"]
        inv = sum(p["cost"] * p["m"] for p in pos)
        eq.append(cash + inv); ex.append(inv / (cash + inv))
    return pd.Series(eq, cal), np.mean(ex), taken


def main():
    bear = regime_series()
    print(f"market below its 200-day average on {bear[bear.index >= START].mean():.0%} of days since 2008")
    allu, n_all = load_all()
    print(f"all-sector universe: {len(allu)} of {n_all} tickers have price data")
    a_pit = study_events(events(allu, True), "ALL-SECTORS point-in-time members", bear)
    study_events(events({t: x for t, x in allu.items() if x.current.iloc[0]}, False),
                 "TODAYS members back to 2008 (survivorship-biased, as in the handoff)", bear)
    tech = load_tech()
    a_tech = study_events(events(tech, True), "TECH point-in-time members", bear)

    print("\n=== BREADTH TRIGGER: share of members with a joint buy 13 in last 5 days >= threshold -> buy ETF next open ===")
    print("pct = share of random-date samples doing at least as well (small = unusual)")
    b1, share_all = breadth(a_pit, "SPY", "all-sector breadth")
    b2, _ = breadth(a_pit, "XLK", "all-sector breadth")
    b3, share_tech = breadth(a_tech, "XLK", "tech breadth")
    b3["etf"] = "XLK (tech breadth)"
    bb = pd.concat([b1, b2, b3])
    print(bb.to_string(index=False))
    bb.to_csv(ROOT / "results/s2_breadth.csv", index=False)

    print("\n=== PORTFOLIOS 2008 -> Nov 2026 (10 bps/side, idle cash in T-bills) ===")
    cal = pd.read_csv(ROOT / "data/prices_etf/XLK.csv", index_col=0, parse_dates=True).index
    cal = cal[cal >= START]
    rows, curves = [], {}

    def add(name, eq, expo, n=np.nan, tr=None):
        s = bt.stats(eq)
        s2 = bt.stats(eq[eq.index >= OOS])
        r = dict(strategy=name, trades=n, expo=expo * 100, **s, oos_cagr=s2["cagr"], oos_maxdd=s2["maxdd"])
        if tr is not None and len(tr):
            r.update(avg_trade=tr.ret.mean() * 100, hit=(tr.ret > 0).mean() * 100, bars=tr.bars.mean())
        rows.append(r); curves[name] = eq

    for uni, data, slots in (("tech", tech, 10), ("all sectors", allu, 20)):
        for regime in (False, True):
            for rule in ("time", "atr"):
                tr = trades(data, bear, rule, regime)
                eq, expo, taken = book(tr, data, cal, slots)
                add(f"stocks {uni} | {'below 200d' if regime else 'any regime'} | {'ATR 2x/3x' if rule == 'atr' else 'hold 21d'}",
                    eq, expo, taken, tr)
                if uni == "all sectors" and regime and rule == "atr":
                    tr.to_csv(ROOT / "results/s2_trades_all_bear_atr.csv", index=False)
    for thr in (0.01, 0.02, 0.04):
        for sh, lab in ((share_tech, "tech breadth"), (share_all, "all-sector breadth")):
            eq, expo = etf_book(sh, "XLK", thr)
            add(f"XLK when {lab} >= {thr:.0%}, 21d", eq, expo)
    for etf in ("XLK", "SPY"):
        c = pd.read_csv(ROOT / f"data/prices_etf/{etf}.csv", index_col=0, parse_dates=True)["Adj Close"].reindex(cal)
        add(f"{etf} buy & hold", c / c.iloc[0], 1.0)
    out = pd.DataFrame(rows).set_index("strategy")
    print(out.to_string())
    out.to_csv(ROOT / "results/s2_portfolios.csv")
    pd.DataFrame(curves).to_csv(ROOT / "results/s2_equity_curves.csv")


if __name__ == "__main__":
    main()
