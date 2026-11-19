"""Study 9: the all-sector book (study 6, version A: 1/11 per sector, 21-day hold) expressed with
call options on the sector ETFs instead of the ETFs themselves.

There is no free history of sector-ETF option prices, so calls are PRICED WITH A MODEL:
  Black-Scholes, rate = 13-week T-bill, no dividends (total-return prices are used),
  implied vol = VIX (30-day calls) or VIX3M (60-day calls)
                x (sector ETF 60-day realised vol / SPY 60-day realised vol, clipped 0.7-2.0),
  20-delta calls use 95% of that vol (assumed call skew).
  Cost: 3% of premium per side for at-the-money calls, 6% for 20-delta calls.
On an activation the call is bought at the next open.  It is sold after 21 trading days (or left
to expire if that comes first); if the sector is still active, a new call is bought.
Sizing per active sector, as option notional (shares x price) relative to equity:
  "1x"            : 1/11 of equity, the same notional as the ETF version
  "delta-matched" : 2/11 for at-the-money calls, 5/11 for 20-delta calls (starting delta about 1/11)
Premium is paid from cash; the rest of the book is in T-bills.
"""
import numpy as np, pandas as pd, sys, pathlib
from math import log, sqrt, exp, erf
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt, study5, study6

ROOT, SEG = bt.ROOT, 21
pd.set_option("display.width", 280, "display.max_columns", 60, "display.float_format", "{:.2f}".format)
N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
Z20 = 0.841621   # N^-1(0.80)


def call(S, K, T, v, r):
    if T <= 0 or v <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + v * v / 2) * T) / (v * sqrt(T))
    return S * N(d1) - K * exp(-r * T) * N(d1 - v * sqrt(T))


def main():
    xk, acts_tech, cash = study5.setup()
    cal = xk.index
    acts, _ = study6.sector_acts(cal)
    acts["Information Technology"] = acts_tech
    rf = pd.read_csv(ROOT / "data/prices_etf/_IRX.csv", index_col=0, parse_dates=True)["Close"].reindex(cal).ffill().fillna(0).values / 100
    load = lambda n: pd.read_csv(ROOT / f"data/prices_etf/{n}.csv", index_col=0, parse_dates=True)["Close"]
    vix = {30: load("_VIX").shift(1).reindex(cal).ffill().values / 100,          # yesterday's close: known at the open
           60: load("_VIX3M").shift(1).reindex(cal).ffill().values / 100}
    vix_c = {30: load("_VIX").reindex(cal).ffill().values / 100, 60: load("_VIX3M").reindex(cal).ffill().values / 100}
    spy_rv = study6.etf_frame("SPY").c.pct_change().rolling(60).std()
    E, segs, base_R, base_I = {}, [], {}, {}
    for s, tk in study6.ETF.items():
        x = study6.etf_frame(tk)
        ratio = (x.c.pct_change().rolling(60).std() / spy_rv).clip(0.7, 2.0).shift(1).reindex(cal).ffill().values
        x = x.reindex(cal)
        E[s] = dict(o=x.o.values, c=x.c.values, ratio=ratio)
        eq, T, inv = study5.sim(x, acts[s][2], cash, 0, 0, "none")
        base_R[s], base_I[s] = eq.pct_change().fillna(eq.iloc[0] - 1), inv
        for _, t in T.iterrows():
            a, b = cal.get_loc(t.entry), cal.get_loc(t.exit)
            k = a
            while k <= b:
                segs.append((s, k, min(k + SEG - 1, b)))
                k += SEG
    base = (1 + pd.DataFrame(base_R).mean(axis=1)).cumprod()
    starts = {}
    for s, a, b in segs:
        starts.setdefault(a, []).append((s, b))
    days = (cal - cal[0]).days.values

    def run(dte, delta20, mult):
        skew, tc = (0.95, 0.06) if delta20 else (1.0, 0.03)
        cashv, pos, eq, prem_paid, seg_ret, ivs = 1.0, [], [], np.zeros(len(cal)), [], []
        equity = 1.0
        for k in range(len(cal)):
            cashv *= 1 + rf[k] / 252
            for s, b in starts.get(k, []):
                e = E[s]
                v = vix[dte][k] * e["ratio"][k] * skew
                T = dte / 365
                S = e["o"][k]
                K = S * exp(Z20 * v * sqrt(T) + (rf[k] + v * v / 2) * T) if delta20 else S
                p = call(S, K, T, v, rf[k])
                n = mult * equity / 11 / S
                cost = n * p * (1 + tc)
                cost = min(cost, max(cashv, 0))                      # never spend more than the cash on hand
                n = cost / (p * (1 + tc))
                cashv -= cost
                prem_paid[k] += cost / equity
                pos.append(dict(s=s, b=b, K=K, n=n, exp=days[k] + dte, cost=cost, val=n * p, skew=skew))
                ivs.append(v)
            tot = 0.0
            for p in pos[:]:
                e = E[p["s"]]
                Trem = (p["exp"] - days[k]) / 365
                v = vix_c[dte][k] * e["ratio"][k] * p["skew"]
                val = p["n"] * call(e["c"][k], p["K"], Trem, v, rf[k])
                if Trem <= 0 or k >= p["b"]:
                    proceeds = val * (1 - tc) if Trem > 0 else val
                    cashv += proceeds
                    seg_ret.append(proceeds / p["cost"] - 1)
                    pos.remove(p)
                else:
                    tot += val
            equity = cashv + tot
            eq.append(equity)
        eq = pd.Series(eq, cal)
        sr = np.array(seg_ret)
        r = study6.risk(eq)
        r.update({"Options bought": len(sr), "Avg result per option, % of premium": sr.mean() * 100,
                  "Options that lost money %": (sr < 0).mean() * 100, "Options that lost >90% %": (sr < -0.9).mean() * 100,
                  "Median result, % of premium": np.median(sr) * 100, "Best option, x premium": sr.max() + 1,
                  "Premium spent, % of equity per year": prem_paid.sum() / (len(cal) / 252) * 100,
                  "Most premium at risk in one day %": pd.Series(prem_paid).rolling(21).sum().max() * 100,
                  "Avg implied vol at purchase %": np.mean(ivs) * 100})
        return eq, r

    out, curves = {}, {}
    r0 = study6.risk(base)
    out["ETF book (base)"], curves["ETF book (base)"] = r0, base
    for name, dte, d20, mult in (("ATM 30d, 1x", 30, False, 1), ("ATM 30d, delta-matched 2x", 30, False, 2),
                                 ("ATM 60d, 1x", 60, False, 1), ("ATM 60d, delta-matched 2x", 60, False, 2),
                                 ("20-delta 30d, 1x", 30, True, 1), ("20-delta 30d, delta-matched 5x", 30, True, 5),
                                 ("20-delta 60d, 1x", 60, True, 1), ("20-delta 60d, delta-matched 5x", 60, True, 5)):
        eq, r = run(dte, d20, mult)
        out[name], curves[name] = r, eq
    J = pd.DataFrame(out)
    print(J.to_string())
    J.to_csv(ROOT / "results/s9_risk_stats.csv")
    Y = pd.DataFrame({k: bt.yearly(v) * 100 for k, v in curves.items()}); Y.index = Y.index.year
    print("\nyearly returns %:"); print(Y.T.round(1).to_string())
    Y.to_csv(ROOT / "results/s9_yearly.csv")
    pd.DataFrame(curves).to_csv(ROOT / "results/s9_equity_curves.csv")


if __name__ == "__main__":
    main()
