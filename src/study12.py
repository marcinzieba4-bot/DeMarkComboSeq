"""Study 12: XLK only.  ETF book, put-spread book, and a 50/50 mix, with recovery statistics.

Trigger: two of the 30 biggest tech stocks print a joint buy 13 within 10 trading days (study 4).
  ETF book        : 100% of capital in XLK from the next open, 21 trading days (restarted by new
                    activations); otherwise in a money-market account (13-week T-bill rate).
  Put-spread book : at the same moments sell an at-the-money XLK put and buy a 20-delta put, 30 days
                    to expiry, on a notional equal to 100% of capital; bought back after 21 trading
                    days (or expired) and re-sold while the trigger is still active.  ALL capital sits
                    in the money-market account the whole time (it is the collateral).
  50/50           : half the capital in each book, rebalanced daily.
Option prices are modelled (see study 10): Black-Scholes, vol = VIX x (XLK / SPY 60-day realised
vol), at-the-money 0.95x, 20-delta put 1.10x; costs 3% / 6% of premium per side.  ETF: 10 bps per side.
"""
import numpy as np, pandas as pd, sys, pathlib
from math import sqrt, exp
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt, study5, study6
from study9 import SEG
from study10 import put, LEG

ROOT, DTE = bt.ROOT, 30
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format)


def drawdowns(eq):
    """every spell below a previous high: peak, trough, recovery date, depth, lengths in calendar days"""
    hi = eq.cummax()
    under = eq < hi * (1 - 1e-12)
    rows, start = [], None
    for i, u in enumerate(under.values):
        if u and start is None:
            start = i - 1
        if (not u or i == len(eq) - 1) and start is not None:
            seg = eq.iloc[start:i + 1]
            tr = seg.idxmin()
            rec = eq.index[i] if not u else pd.NaT
            rows.append(dict(peak=eq.index[start].date(), trough=tr.date(), recovered=rec.date() if pd.notna(rec) else "not yet",
                             depth=(seg.min() / seg.iloc[0] - 1) * 100, fall_days=(tr - eq.index[start]).days,
                             recovery_days=(rec - tr).days if pd.notna(rec) else np.nan,
                             under_water_days=((rec if pd.notna(rec) else eq.index[-1]) - eq.index[start]).days))
            start = None
    return pd.DataFrame(rows)


def recovery_stats(eq):
    D = drawdowns(eq)
    big = D[D.depth <= -5]
    hi = eq.cummax()
    return {"Time below a previous high %": (eq < hi * (1 - 1e-12)).mean() * 100,
            "Longest spell below the high, years": D.under_water_days.max() / 365.25,
            "Longest fall, peak to trough, days": D.fall_days.max(),
            "Longest recovery, trough to new high, days": D.recovery_days.max(),
            "Drawdowns deeper than 5%": len(big),
            "Avg recovery of those, days": big.recovery_days.mean(), "Median recovery of those, days": big.recovery_days.median(),
            "Avg spell below high of those, days": big.under_water_days.mean(),
            "Currently below the high by %": (eq.iloc[-1] / hi.iloc[-1] - 1) * 100}, D


def main():
    xk, acts, cash = study5.setup()
    cal = xk.index
    rf_d = pd.Series(cash, cal)
    eq_etf, T, inv = study5.sim(xk, acts[2], cash, 0, 0, "none")
    ld = lambda n: pd.read_csv(ROOT / f"data/prices_etf/{n}.csv", index_col=0, parse_dates=True)["Close"]
    rf = ld("_IRX").reindex(cal).ffill().fillna(0).values / 100
    vo, vc = ld("_VIX").shift(1).reindex(cal).ffill().values / 100, ld("_VIX").reindex(cal).ffill().values / 100
    x_full = study6.etf_frame("XLK")
    ratio = (x_full.c.pct_change().rolling(60).std() / study6.etf_frame("SPY").c.pct_change().rolling(60).std()
             ).clip(0.7, 2.0).shift(1).reindex(cal).ffill().values
    o, c = xk.o.values, xk.c.values
    days = (cal - cal[0]).days.values
    starts = {}
    for _, t in T.iterrows():
        a, b = cal.get_loc(t.entry), cal.get_loc(t.exit)
        for k in range(a, b + 1, SEG):
            starts[k] = min(k + SEG - 1, b)
    legs = [(-1, "atm"), (1, "20d")]
    cashv, pos, eq, res = 1.0, None, [], []
    for k in range(len(cal)):
        cashv *= 1 + rf[k] / 252                              # all capital earns the money-market rate
        if k in starts:
            S, Tm, n, L, net = o[k], DTE / 365, (cashv) / o[k], [], 0.0
            for sign, kind in legs:
                d1, vm, tc = LEG[kind]
                v = vo[k] * ratio[k] * vm
                K = S if d1 is None else S * exp(-d1 * v * sqrt(Tm) + (rf[k] + v * v / 2) * Tm)
                net += -sign * n * put(S, K, Tm, v, rf[k]) * (1 - tc if sign < 0 else 1 + tc)
                L.append((sign, K, vm, tc))
            cashv += net
            pos = dict(b=starts[k], n=n, legs=L, exp=days[k] + DTE, credit=net, notional=n * S, entry=cal[k], width=(L[0][1] - L[1][1]) / S)
        liab = 0.0
        if pos:
            Trem = (pos["exp"] - days[k]) / 365
            mark = close = 0.0
            for sign, K, vm, tc in pos["legs"]:
                val = pos["n"] * put(c[k], K, Trem, vc[k] * ratio[k] * vm, rf[k])
                mark += sign * val
                close += sign * val * ((1 + tc if sign < 0 else 1 - tc) if Trem > 0 else 1)
            if Trem <= 0 or k >= pos["b"]:
                cashv += close
                res.append(dict(sold=pos["entry"].date(), closed=cal[k].date(), credit=pos["credit"] / pos["notional"] * 100,
                                width=pos["width"] * 100, result=(pos["credit"] + close) / pos["notional"] * 100))
                pos = None
            else:
                liab = mark
        eq.append(cashv + liab)
    eq_ps = pd.Series(eq, cal)
    P = pd.DataFrame(res)
    r_etf, r_ps = eq_etf.pct_change().fillna(eq_etf.iloc[0] - 1), eq_ps.pct_change().fillna(eq_ps.iloc[0] - 1)
    eq_mix = (1 + 0.5 * r_etf + 0.5 * r_ps).cumprod()
    bh = xk.c / xk.c.iloc[0]
    mm = (1 + rf_d).cumprod()
    books = {"XLK ETF book": eq_etf, "XLK put-spread book": eq_ps, "50/50 mix": eq_mix, "XLK buy & hold": bh, "Money market only": mm}
    out, dds = {}, {}
    for k, e in books.items():
        r = e.pct_change().fillna(0)
        xs = r - rf_d
        d = study6.risk(e)
        d.update({"Return above money market %/yr": xs.mean() * 252 * 100,
                  "Sharpe above money market": xs.mean() / xs.std() * np.sqrt(252) if xs.std() > 1e-9 else np.nan})
        rs, D = recovery_stats(e)
        d.update(rs); out[k], dds[k] = d, D
    J = pd.DataFrame(out)
    print(J.to_string()); J.to_csv(ROOT / "results/s12_risk_stats.csv")
    inv = pd.Series(inv, cal)
    idle = (mm.iloc[-1]) ** (252 / len(cal)) - 1
    print(f"\nactive {inv.mean():.0%} of days; money-market rate averaged {idle * 100:.2f}%/yr; interest earned: "
          f"ETF book {((rf_d[~inv.values]).sum()) / (len(cal) / 252) * 100:.2f}%/yr (idle days only), put-spread book {rf_d.sum() / (len(cal) / 252) * 100:.2f}%/yr (all days)")
    w = T.ret > 0
    print(f"ETF trades: {len(T)}, hit {w.mean():.0%}, avg {T.ret.mean() * 100:.2f}%, worst {T.ret.min() * 100:.1f}%")
    print(f"put spreads: {len(P)}, hit {(P.result > 0).mean():.0%}, avg credit {P.credit.mean():.2f}% of notional, avg width {P.width.mean():.1f}%, "
          f"avg result {P.result.mean():.2f}%, worst {P.result.min():.2f}%, best {P.result.max():.2f}%")
    P.to_csv(ROOT / "results/s12_put_spreads.csv", index=False)
    for k in ("XLK ETF book", "XLK put-spread book", "50/50 mix", "XLK buy & hold"):
        print(f"\nfive deepest drawdowns - {k}")
        print(dds[k].sort_values("depth").head(5).to_string(index=False))
        dds[k].to_csv(ROOT / f"results/s12_drawdowns_{k.split()[0].replace('/', '_')}_{k.split()[1][:3]}.csv", index=False)
    Y = pd.DataFrame({k: bt.yearly(v) * 100 for k, v in books.items()}); Y.index = Y.index.year
    print("\nyearly returns %:"); print(Y.T.round(1).to_string()); Y.to_csv(ROOT / "results/s12_yearly.csv")
    pd.DataFrame(books).to_csv(ROOT / "results/s12_equity_curves.csv")


if __name__ == "__main__":
    main()
