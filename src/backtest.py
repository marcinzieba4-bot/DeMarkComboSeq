"""Long-only test of joint TD Combo + TD Sequential buy-13 signals,
point-in-time S&P 500 Information Technology members, 2008 -> today.

No-lookahead conventions
  * signal is evaluated on the close of day t using data <= t
  * stock must be an index/sector member on day t
  * entry at the OPEN of day t+1; exits decided on a close are executed at next open
  * signals use split-adjusted OHLC (invariant to later dividends);
    P&L uses dividend-adjusted (total return) prices
"""
import numpy as np, pandas as pd, pathlib, sys, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from demark import td_buy_signals, joint_signal

ROOT = pathlib.Path(__file__).resolve().parents[1]
START, SPLIT = "2008-01-01", "2017-01-01"
HORIZONS = [5, 10, 20, 40, 60]
COST = 0.001          # 10 bps per side (commission + slippage)
WINDOW = 10           # baseline: second 13 within 10 bars of the first
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- data
def load():
    smap = pd.read_csv(ROOT / "data/tech_sector_map.csv")
    mem = pd.read_csv(ROOT / "data/sp500_ticker_start_end.csv", parse_dates=["start_date", "end_date"])
    px, member = {}, {}
    last = None
    cov = []
    for _, r in smap.iterrows():
        rows = mem[mem.ticker == r.ticker]
        f = ROOT / f"data/prices/{r.yahoo}.csv"
        have = isinstance(r.yahoo, str) and f.exists()
        if have and r.yahoo not in px:
            d = pd.read_csv(f, index_col=0, parse_dates=True)
            if len(d) < 300:
                have = False
            else:
                px[r.yahoo] = d
                member[r.yahoo] = pd.Series(False, index=d.index)
        elif have and r.yahoo in px:
            pass
        for _, m in rows.iterrows():
            a = max(m.start_date, pd.Timestamp(r.it_from) if isinstance(r.it_from, str) else m.start_date)
            b = m.end_date if pd.notna(m.end_date) else pd.Timestamp("2100-01-01")
            if isinstance(r.it_to, str):
                b = min(b, pd.Timestamp(r.it_to))
            a = max(a, pd.Timestamp(START))
            if b <= a:
                continue
            cov.append((r.ticker, a, min(b, pd.Timestamp.today()), have and r.yahoo in px))
            if have and r.yahoo in px:
                idx = member[r.yahoo].index
                member[r.yahoo][(idx >= a) & (idx < b)] = True
    cov = pd.DataFrame(cov, columns=["ticker", "a", "b", "have"])
    cov["days"] = (cov.b - cov.a).dt.days
    return px, member, cov


def build(px, member, recycle="perl"):
    """Per-ticker frame with signals and helper columns."""
    out = {}
    for t, d in px.items():
        o, h, l, c = (d[k].values.astype(float) for k in ["Open", "High", "Low", "Close"])
        s = td_buy_signals(o, h, l, c, recycle=recycle)
        f = d["Adj Close"].values / c
        x = pd.DataFrame(index=d.index)
        x["o"], x["c"] = o * f, c * f          # total-return prices for P&L
        x["raw_c"], x["raw_l"] = c, l
        x["member"] = member[t].values
        for k in ["setup9", "seq13", "combo13", "sell9"]:
            x[k] = s[k]
        for w in [0, 5, 10, 20]:
            x[f"joint{w}"] = joint_signal(s["seq13"], s["combo13"], s["sell9"], w)
        # naive oversold control: RSI(14) crosses below 30
        dlt = pd.Series(c).diff()
        up = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = (100 - 100 / (1 + up / dn)).values
        x["rsi30"] = (rsi < 30) & (np.r_[np.nan, rsi[:-1]] >= 30)
        x["dd60"] = c / pd.Series(h).rolling(60, min_periods=20).max().values - 1
        # stop reference: lowest low of last 30 bars (known at t)
        x["low30"] = pd.Series(l).rolling(30, min_periods=10).min().values
        out[t] = x
    return out


# ---------------------------------------------------------------- event study
def fwd_table(data):
    """For every member-day: forward total return, entry next open -> close t+h."""
    rows = []
    for t, x in data.items():
        o1 = x.o.shift(-1)
        r = pd.DataFrame({f"r{h}": x.c.shift(-h) / o1 - 1 for h in HORIZONS})
        r["ticker"] = t
        r = pd.concat([r, x[["member", "setup9", "seq13", "combo13", "joint0", "joint5",
                             "joint10", "joint20", "rsi30", "dd60"]]], axis=1)
        rows.append(r[(x.member.values) & (x.index >= START)])
    a = pd.concat(rows).rename_axis("date").reset_index()
    # date-matched benchmark: mean forward return of ALL members on the same date
    for h in HORIZONS:
        a[f"x{h}"] = a[f"r{h}"] - a.groupby("date")[f"r{h}"].transform("mean")
    return a


def cluster_t(ev, col):
    """t-stat of the mean with clustering by calendar month of the signal
    (signals bunch in sell-offs, so naive t-stats are badly overstated)."""
    v = ev[[col, "date"]].dropna()
    if len(v) < 5:
        return np.nan
    m = v[col].mean()
    g = (v[col] - m).groupby(v.date.dt.to_period("M")).sum()
    se = np.sqrt((g ** 2).sum()) / len(v)
    return m / se


def event_study(a, sig, mask=None):
    ev = a[a[sig]] if mask is None else a[a[sig] & mask]
    res = {"signal": sig, "n": len(ev), "months": ev.date.dt.to_period("M").nunique()}
    for h in HORIZONS:
        res[f"raw{h}"] = ev[f"r{h}"].mean() * 100
        res[f"exc{h}"] = ev[f"x{h}"].mean() * 100
        res[f"t{h}"] = cluster_t(ev, f"x{h}")
        res[f"win{h}"] = (ev[f"r{h}"] > 0).mean() * 100
    return res


# ---------------------------------------------------------------- trade simulation
def make_trades(data, sig, exit_rule):
    """exit_rule: ("time", n) or ("demark", max_hold).
    demark: exit at next open after the first of
            (a) a TD sell setup 9 completes            [exit into strength]
            (b) close < lowest low of the 30 bars up to the signal bar [stop]
            (c) max_hold bars elapsed."""
    trades = []
    for t, x in data.items():
        idx = x.index
        o, c = x.o.values, x.c.values
        raw_c = x.raw_c.values
        sigs = np.flatnonzero((x[sig] & x.member).values & (idx >= START))
        busy_until = -1
        for i in sigs:
            if i <= busy_until or i + 1 >= len(x):
                continue                      # one position per stock at a time
            e = i + 1
            if exit_rule[0] == "time":
                q = min(e + exit_rule[1] - 1, len(x) - 1)
                px_out, why = c[q], "time"
            else:
                stop = x.low30.values[i]
                q, px_out, why = None, None, None
                for j in range(e, len(x)):
                    last = j == len(x) - 1
                    if raw_c[j] < stop:
                        why = "stop"
                    elif x.sell9.values[j]:
                        why = "sell9"
                    elif j - e + 1 >= exit_rule[1]:
                        why = "time"
                    if why or last:
                        if last:
                            q, px_out, why = j, c[j], why or "open"
                        else:
                            q, px_out = j + 1, o[j + 1]
                        break
            busy_until = q
            trades.append(dict(ticker=t, signal=idx[i], entry=idx[e], exit=idx[q],
                               ret=px_out / o[e] * (1 - COST) / (1 + COST) - 1,
                               bars=q - e + 1, why=why, dd60=x.dd60.values[i]))
    return pd.DataFrame(trades).sort_values(["entry", "dd60"]).reset_index(drop=True)


def portfolio(trades, data, cal, slots=10):
    """Capital-constrained daily mark-to-market.  Each new position gets
    equity/slots (if a slot and cash are free).  Same-day conflicts are
    resolved by deeper 60-day drawdown at signal time (known in advance).
    Idle cash earns nothing."""
    close = pd.DataFrame({t: x.c for t, x in data.items()}).reindex(cal).ffill()
    by_entry = {d: g for d, g in trades.groupby("entry")}
    cash, pos, eq, taken, expo = 1.0, [], [], [], []
    for d in cal:
        # exits at today's price (open or close, both already inside trade return)
        for p in [p for p in pos if p["exit"] == d]:
            cash += p["cost"] * (1 + p["ret"])
            pos.remove(p)
        if d in by_entry:
            equity_now = cash + sum(p["cost"] * p["mtm"] for p in pos)
            for _, tr in by_entry[d].iterrows():
                size = equity_now / slots
                if len(pos) < slots and cash >= size * 0.999 and tr.exit > d:
                    cash -= size
                    pos.append(dict(tr, cost=size, mtm=1.0, entry_px=None))
                    taken.append(tr.name)
        for p in pos:
            x = data[p["ticker"]]
            e_px = x.o.get(p["entry"])
            p["mtm"] = close.at[d, p["ticker"]] / e_px
        inv = sum(p["cost"] * p["mtm"] for p in pos)
        eq.append(cash + inv)
        expo.append(inv / (cash + inv))
    return pd.Series(eq, cal), pd.Series(expo, cal), trades.loc[taken]


def stats(eq):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) * 100 - 100,
                vol=r.std() * np.sqrt(252) * 100,
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                maxdd=(eq / eq.cummax() - 1).min() * 100,
                total=eq.iloc[-1] / eq.iloc[0])


def trade_stats(tr):
    if len(tr) == 0:
        return dict(n=0)
    w = tr.ret > 0
    return dict(n=len(tr), win=w.mean() * 100, avg=tr.ret.mean() * 100, med=tr.ret.median() * 100,
                avg_win=tr.ret[w].mean() * 100, avg_loss=tr.ret[~w].mean() * 100,
                pf=tr.ret[w].sum() / -tr.ret[~w].sum() if (~w).any() else np.inf,
                worst=tr.ret.min() * 100, bars=tr.bars.mean())


# ---------------------------------------------------------------- main
def main():
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.2f}".format)
    px, member, cov = load()
    res = ROOT / "results"
    print("=== UNIVERSE COVERAGE (point-in-time S&P 500 Info Tech, from 2008) ===")
    tot, got = cov.days.sum(), cov[cov.have].days.sum()
    print(f"member-days covered by price data: {got / tot:.1%}  "
          f"({cov[cov.have].ticker.nunique()} of {cov.ticker.nunique()} tickers)")
    miss = cov[~cov.have].groupby("ticker").days.sum().sort_values(ascending=False)
    print("missing (no free price history):", ", ".join(miss.index))
    cov["yr"] = cov.a.dt.year
    yrs = []
    for y in range(2008, 2027):
        d = pd.Timestamp(f"{y}-06-30")
        c = cov[(cov.a <= d) & (cov.b > d)]
        yrs.append((y, len(c), int(c.have.sum())))
    print("members / with data on 30 Jun:", " ".join(f"{y}:{h}/{n}" for y, n, h in yrs))

    data = build(px, member)
    a = fwd_table(data)
    a.to_pickle(res / "member_days.pkl")
    print(f"\nmember-days in sample: {len(a):,}; unconditional mean fwd return (%):",
          {h: round(a[f'r{h}'].mean() * 100, 2) for h in HORIZONS})

    print("\n=== EVENT STUDY: entry next open, exit close after h bars. raw = mean % return, "
          "exc = excess vs all members same day, t = month-clustered t-stat of excess ===")
    sigs = ["setup9", "seq13", "combo13", "joint0", "joint5", "joint10", "joint20", "rsi30"]
    rows = [event_study(a, s) for s in sigs]
    es = pd.DataFrame(rows).set_index("signal")
    cols = ["n", "months"] + [f"{k}{h}" for h in HORIZONS for k in ("raw", "exc", "t", "win")]
    print(es[cols].to_string())
    es.to_csv(res / "event_study.csv")

    print(f"\n=== STABILITY of joint{WINDOW}: two halves, and by year ===")
    halves = [("2008-2016", a.date < SPLIT), ("2017-2026", a.date >= SPLIT)]
    hs = pd.DataFrame([dict(event_study(a, f"joint{WINDOW}", m), period=p) for p, m in halves]).set_index("period")
    print(hs[cols].to_string())
    ev = a[a[f"joint{WINDOW}"]]
    by = ev.groupby(ev.date.dt.year).agg(n=("r20", "size"), raw20=("r20", lambda v: v.mean() * 100),
                                         exc20=("x20", lambda v: v.mean() * 100),
                                         raw60=("r60", lambda v: v.mean() * 100),
                                         exc60=("x60", lambda v: v.mean() * 100))
    print(by.T.to_string())
    by.to_csv(res / "joint_by_year.csv")

    print("\n=== RECYCLE-RULE SENSITIVITY (joint10, excess %) ===")
    for rc in ["perl", "always", "never"]:
        e = event_study(fwd_table(build(px, member, rc)), "joint10")
        print(rc, "n=%d" % e["n"], " ".join(f"exc{h}={e[f'exc{h}']:.2f}(t={e[f't{h}']:.1f})" for h in HORIZONS))

    print("\n=== TRADE SIMULATION (10 bps/side, one position per stock) ===")
    xlk = pd.read_csv(ROOT / "data/prices/XLK.csv", index_col=0, parse_dates=True)["Adj Close"]
    cal = xlk.index[xlk.index >= START]
    bench = {}
    xlk = xlk.reindex(cal)
    bench["XLK buy&hold"] = stats(xlk)
    rets = pd.DataFrame({t: x.c.pct_change().where(x.member.shift(1, fill_value=False)) for t, x in data.items()}).reindex(cal)
    ew = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    bench["Equal-weight PIT tech universe (same data)"] = stats(ew)
    rules = [("time", 10), ("time", 20), ("time", 40), ("time", 60), ("demark", 60)]
    out, curves = [], {}
    for sig in ["joint10", "joint0", "seq13", "combo13", "setup9", "rsi30"]:
        for rule in rules:
            tr = make_trades(data, sig, rule)
            eq, expo, taken = portfolio(tr, data, cal)
            name = f"{sig} | {rule[0]}{rule[1]}"
            curves[name] = eq
            out.append(dict(strategy=name, **{f"tr_{k}": v for k, v in trade_stats(tr).items()},
                            taken=len(taken), expo=expo.mean() * 100, **stats(eq)))
            if sig == "joint10" and rule == ("demark", 60):
                tr.to_csv(res / "trades_joint10_demark60.csv", index=False)
                print("exit reasons joint10|demark60:", tr.why.value_counts().to_dict())
    o = pd.DataFrame(out).set_index("strategy")
    print(o.to_string())
    o.to_csv(res / "strategies.csv")
    print(pd.DataFrame(bench).T.to_string())
    pd.DataFrame(bench).T.to_csv(res / "benchmarks.csv")
    pd.DataFrame(curves).assign(XLK=xlk / xlk.iloc[0], EW_universe=ew).to_csv(res / "equity_curves.csv")

    # return per unit of time invested vs benchmark
    print("\n=== RANDOM-ENTRY PLACEBO for joint10|time20: same number of trades, random member-days ===")
    n = int(a["joint10"].sum())
    pool = a.dropna(subset=["r20"])
    sims = np.array([pool.r20.sample(n, random_state=int(s)).mean() for s in RNG.integers(0, 1e9, 2000)]) * 100
    actual = a[a.joint10].r20.mean() * 100
    print(f"actual mean r20 = {actual:.2f}%; placebo mean = {sims.mean():.2f}%, 5-95% = "
          f"[{np.percentile(sims, 5):.2f}, {np.percentile(sims, 95):.2f}], share of placebos >= actual = {(sims >= actual).mean():.3f}")
    # date-preserving placebo: random member stock on the SAME signal dates
    ev = a[a.joint10].dropna(subset=["r20"])
    grp = {d: g.r20.values for d, g in pool[pool.date.isin(ev.date)].groupby("date")}
    sims2 = np.array([np.mean([RNG.choice(grp[d]) for d in ev.date]) for _ in range(2000)]) * 100
    print(f"same-dates placebo (random member on the signal dates): mean = {sims2.mean():.2f}%, "
          f"5-95% = [{np.percentile(sims2, 5):.2f}, {np.percentile(sims2, 95):.2f}], share >= actual = {(sims2 >= actual).mean():.3f}")


if __name__ == "__main__":
    main()
