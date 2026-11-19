"""Study 6: the study-5 trigger run on every sector, then combined into one book.

Per sector: the 30 biggest point-in-time members (trailing 250-day dollar volume; sectors with
fewer than 30 members use all of them).  When N different stocks print a joint buy 13 inside
10 trading days, buy that sector's ETF at the next open and hold 21 days.
Sector labels: today's GICS sector for current members, Yahoo's sector for former members;
tech uses the hand-built dated map from study 1 and those stock-days are excluded elsewhere.
ETFs: SPDR sector funds; VNQ and VOX stand in for real estate and communication services
because XLRE (2015) and XLC (2018) are too young.

Configurations (fixed before running, taken from study 5):
  A  2 stocks, no stop, no filter
  B  2 stocks, ETF below its 100-day average, 8% stop, 21-day ban after a stop
  C  1 stock,  ETF below its 100-day average, 8% stop, 63-day ban after a stop
Joint books:
  equal   : each sector sleeve has 1/11 of capital (cash when its sector is inactive)
  shared  : active sectors share the capital equally, max 25% per sector; rest in T-bills
"""
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt
from demark import td_buy_signals, joint_signal
import study5
from study3 import START, OOS

ROOT = bt.ROOT
ETF = {"Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV", "Industrials": "XLI",
       "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE", "Utilities": "XLU",
       "Materials": "XLB", "Real Estate": "VNQ", "Communication Services": "VOX"}
CFG = {"A: 2 stocks, plain": (2, 0, 0, "none"),
       "B: 2 stocks, below 100d, 8% stop, 21d ban": (2, 0.08, 21, "below"),
       "C: 1 stock, below 100d, 8% stop, 63d ban": (1, 0.08, 63, "below")}
pd.set_option("display.width", 280, "display.max_columns", 60, "display.float_format", "{:.1f}".format)


def etf_frame(tk):
    d = pd.read_csv(ROOT / f"data/prices_etf/{tk}.csv", index_col=0, parse_dates=True)
    f = d["Adj Close"] / d["Close"]
    x = pd.DataFrame({"o": d["Open"] * f, "l": d["Low"] * f, "c": d["Adj Close"]})
    x["above"] = x.c > x.c.rolling(100).mean()
    return x


def sector_acts(cal):
    """activation arrays {sector: {N: bool array on cal}} for the non-tech sectors."""
    smap = pd.read_csv(ROOT / "data/sector_map_all.csv").dropna(subset=["sector"]).set_index("ticker").sector
    tech = pd.read_csv(ROOT / "data/tech_sector_map.csv").set_index("ticker")
    mem = pd.read_csv(ROOT / "data/sp500_ticker_start_end.csv", parse_dates=["start_date", "end_date"])
    full = pd.date_range("2006-01-01", cal[-1], freq="B")
    sig, dv, member, sec = {}, {}, {}, {}
    for t, rows in mem.groupby("ticker"):
        f = ROOT / f"data/prices_all/{t}.csv"
        if t not in smap.index or not f.exists():
            continue
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
        m = pd.Series(False, index=d.index)
        for _, r in rows.iterrows():
            m[(d.index >= r.start_date) & (d.index < (r.end_date if pd.notna(r.end_date) else pd.Timestamp("2100-01-01")))] = True
        s = smap[t]
        if t in tech.index:                       # tech stock-days belong to the tech sleeve only
            fr, to = tech.at[t, "it_from"], tech.at[t, "it_to"]
            if isinstance(to, str):
                m &= d.index >= to
            elif isinstance(fr, str):
                m &= d.index < fr
            else:
                continue
            if s == "Information Technology":
                continue
        elif s == "Information Technology":
            continue
        o, h, l, c = (d[k].values.astype(float) for k in ["Open", "High", "Low", "Close"])
        b = td_buy_signals(o, h, l, c)
        sig[t] = pd.Series(joint_signal(b["seq13"], b["combo13"], b["sell9"], 10), d.index)
        dv[t] = (d["Close"] * d["Volume"]).rolling(250, min_periods=120).mean()
        member[t], sec[t] = m, s
    out, info = {}, {}
    for s in ETF:
        if s == "Information Technology":
            continue
        tk = [t for t in sig if sec[t] == s]
        M = pd.DataFrame({t: member[t] for t in tk}).reindex(full).fillna(False).astype(bool)
        D = pd.DataFrame({t: dv[t] for t in tk}).reindex(full)
        top = D.where(M).rank(axis=1, ascending=False) <= 30
        S = pd.DataFrame({t: sig[t] for t in tk}).reindex(full).fillna(False).astype(bool) & top
        S = S[S.index.isin(pd.read_csv(ROOT / "data/prices_etf/SPY.csv", index_col=0, parse_dates=True).index)]
        recent = S.astype(int).rolling(11, min_periods=1).max().sum(axis=1)
        out[s] = {n: ((S.any(axis=1)) & (recent >= n)).reindex(cal).fillna(False).values for n in (1, 2, 3)}
        info[s] = (len(tk), int(S[S.index >= START].values.sum()))
    return out, info


def risk(eq, expo=None):
    s, yr = bt.stats(eq), bt.yearly(eq)
    dd = eq / eq.cummax() - 1
    r = {"CAGR %": s["cagr"], "Vol %": s["vol"], "Sharpe": s["sharpe"], "Max DD %": s["maxdd"],
         "Calmar": s["cagr"] / -s["maxdd"], "Worst year %": yr.min() * 100, "Best year %": yr.max() * 100,
         "Losing years": int((yr < 0).sum()), "Worst day %": eq.pct_change().min() * 100,
         "CAGR 2008-Oct23 %": bt.stats(eq[eq.index < OOS])["cagr"], "CAGR since Oct23 %": bt.stats(eq[eq.index >= OOS])["cagr"],
         "End value of 1": s["total"]}
    if expo is not None:
        r["Avg exposure %"], r["Peak exposure %"] = expo.mean() * 100, expo.max() * 100
    return r


def main():
    xk, acts_tech, cash = study5.setup()
    cal = xk.index
    acts, info = sector_acts(cal)
    acts["Information Technology"] = acts_tech
    info["Information Technology"] = (np.nan, np.nan)
    cash_s = pd.Series(cash, cal)
    joint, curves, per_sector = {}, {}, []
    for name, (n, stop, ban, ma) in CFG.items():
        R, I = {}, {}
        for s, tk in ETF.items():
            x = etf_frame(tk).reindex(cal)
            eq, T, inv = study5.sim(x, acts[s][n], cash, stop, ban, ma)
            R[s], I[s] = eq.pct_change().fillna(eq.iloc[0] - 1), pd.Series(inv, cal)
            w = T.ret > 0
            bh = bt.stats(x.c / x.c.iloc[0])
            per_sector.append({"config": name[0], "sector": s, "ETF": tk, "trades": len(T), "stops": int((T.why == "stop").sum()),
                               "hit%": w.mean() * 100 if len(T) else np.nan, "avg%": T.ret.mean() * 100,
                               "worst%": T.ret.min() * 100, "in mkt%": inv.mean() * 100, "CAGR": bt.stats(eq)["cagr"],
                               "maxDD": bt.stats(eq)["maxdd"], "ETF B&H CAGR": bh["cagr"], "ETF B&H maxDD": bh["maxdd"]})
            if name[0] == "B":
                T.assign(sector=s, ret=T.ret * 100).to_csv(ROOT / f"results/s6_trades_B_{tk}.csv", index=False)
        R, I = pd.DataFrame(R), pd.DataFrame(I)
        eq_equal = (1 + R.mean(axis=1)).cumprod()
        nact = I.sum(axis=1)
        w = I.mul(np.minimum(0.25, 1 / nact.replace(0, np.nan)).fillna(0), axis=0)
        ex = R.sub(cash_s, axis=0).where(I.astype(bool), 0)          # sleeve return above cash while invested
        eq_shared = (1 + cash_s + (w * ex).sum(axis=1)).cumprod()
        joint[f"{name[0]} equal 1/11"] = risk(eq_equal, I.mean(axis=1))
        joint[f"{name[0]} shared, max 25%"] = risk(eq_shared, w.sum(axis=1))
        curves[f"{name[0]} equal 1/11"], curves[f"{name[0]} shared, max 25%"] = eq_equal, eq_shared
        if name[0] == "B":
            yrs = pd.DataFrame({"B equal": bt.yearly(eq_equal) * 100, "B shared": bt.yearly(eq_shared) * 100})
            active = nact
    spy = pd.read_csv(ROOT / "data/prices_etf/SPY.csv", index_col=0, parse_dates=True)["Adj Close"].reindex(cal)
    ew = (1 + pd.DataFrame({s: etf_frame(tk).c.reindex(cal).pct_change() for s, tk in ETF.items()}).mean(axis=1).fillna(0)).cumprod()
    joint["SPY buy & hold"] = risk(spy / spy.iloc[0])
    joint["11 sector ETFs equal-weight"] = risk(ew)
    curves["SPY buy & hold"], curves["11 sector ETFs equal-weight"] = spy / spy.iloc[0], ew
    P = pd.DataFrame(per_sector)
    print("configs:", CFG)
    for c in "ABC":
        print(f"\n=== per sector, config {c} ===")
        print(P[P.config == c].drop(columns="config").to_string(index=False))
    J = pd.DataFrame(joint)
    print("\n=== JOINT BOOK risk stats, 2008 -> Nov 2026 ==="); print(J.round(2).to_string())
    yrs["SPY"] = bt.yearly(spy / spy.iloc[0]) * 100
    yrs.index = yrs.index.year
    print("\nyearly returns %, config B:"); print(yrs.T.to_string())
    print("\nconfig B: days with k sectors active:", active.value_counts().sort_index().to_dict())
    P.to_csv(ROOT / "results/s6_per_sector.csv", index=False); J.to_csv(ROOT / "results/s6_joint_risk.csv")
    yrs.to_csv(ROOT / "results/s6_yearly.csv"); pd.DataFrame(curves).to_csv(ROOT / "results/s6_equity_curves.csv")


if __name__ == "__main__":
    main()
