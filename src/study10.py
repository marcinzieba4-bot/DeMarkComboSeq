"""Study 10: sell puts / put spreads on the sector ETF at each activation of the all-sector book
(study 6, version A).  Same model pricing as study 9 (no free option price history):
  Black-Scholes, rate = T-bill, vol = VIX (30d) or VIX3M (60d) x sector/SPY realised-vol ratio, then
  at-the-money 0.95x, 20-delta put 1.10x, 10-delta put 1.22x   (assumed put skew; VIX itself sits
  a little above at-the-money vol, so the at-the-money premium collected is haircut)
  cost: 3% of premium per side at the money, 6% on out-of-the-money legs.
Sold at the next open after an activation, bought back after 21 trading days or left to expire;
re-sold if the sector is still active.  Notional per active sector = mult x equity/11, cash-secured
(the whole book stays in T-bills as collateral).
"""
import numpy as np, pandas as pd, sys, pathlib
from math import log, sqrt, exp
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt, study5, study6
from study9 import call, SEG

ROOT = bt.ROOT
pd.set_option("display.width", 300, "display.max_columns", 60, "display.float_format", "{:.2f}".format)
LEG = {"atm": (None, 0.95, 0.03), "20d": (0.841621, 1.10, 0.06), "10d": (1.281552, 1.22, 0.06)}   # d1, vol mult, cost


def put(S, K, T, v, r):
    return call(S, K, T, v, r) - S + K * exp(-r * max(T, 0))


def main():
    xk, acts_tech, cash = study5.setup()
    cal = xk.index
    acts, _ = study6.sector_acts(cal)
    acts["Information Technology"] = acts_tech
    ld = lambda n: pd.read_csv(ROOT / f"data/prices_etf/{n}.csv", index_col=0, parse_dates=True)["Close"]
    rf = ld("_IRX").reindex(cal).ffill().fillna(0).values / 100
    vix_o = {30: ld("_VIX").shift(1).reindex(cal).ffill().values / 100, 60: ld("_VIX3M").shift(1).reindex(cal).ffill().values / 100}
    vix_c = {30: ld("_VIX").reindex(cal).ffill().values / 100, 60: ld("_VIX3M").reindex(cal).ffill().values / 100}
    spy_rv = study6.etf_frame("SPY").c.pct_change().rolling(60).std()
    E, starts, base_R = {}, {}, {}
    for s, tk in study6.ETF.items():
        x = study6.etf_frame(tk)
        ratio = (x.c.pct_change().rolling(60).std() / spy_rv).clip(0.7, 2.0).shift(1).reindex(cal).ffill().values
        x = x.reindex(cal)
        E[s] = dict(o=x.o.values, c=x.c.values, ratio=ratio)
        eq, T, inv = study5.sim(x, acts[s][2], cash, 0, 0, "none")
        base_R[s] = eq.pct_change().fillna(eq.iloc[0] - 1)
        for _, t in T.iterrows():
            a, b = cal.get_loc(t.entry), cal.get_loc(t.exit)
            for k in range(a, b + 1, SEG):
                starts.setdefault(k, []).append((s, min(k + SEG - 1, b)))
    base = (1 + pd.DataFrame(base_R).mean(axis=1)).cumprod()
    days = (cal - cal[0]).days.values

    def run(dte, legs, mult):
        """legs: list of (sign, kind); sign -1 = sold put, +1 = bought put."""
        cashv, pos, eq, equity = 1.0, [], [], 1.0
        res, prem, ivs, notion = [], 0.0, [], np.zeros(len(cal))
        for k in range(len(cal)):
            cashv *= 1 + rf[k] / 252
            for s, b in starts.get(k, []):
                e, T, S = E[s], dte / 365, E[s]["o"][k]
                n = mult * equity / 11 / S
                L, net = [], 0.0
                for sign, kind in legs:
                    d1, vm, tc = LEG[kind]
                    v = vix_o[dte][k] * e["ratio"][k] * vm
                    K = S if d1 is None else S * exp(-d1 * v * sqrt(T) + (rf[k] + v * v / 2) * T)
                    p = put(S, K, T, v, rf[k])
                    net += -sign * n * p * (1 - tc if sign < 0 else 1 + tc)      # cash in for sold legs, out for bought
                    L.append((sign, K, vm, tc))
                cashv += net
                prem += net / equity
                ivs.append(vix_o[dte][k] * e["ratio"][k] * 0.95)
                pos.append(dict(s=s, b=b, n=n, legs=L, exp=days[k] + dte, credit=net, notional=n * S))
            liab = 0.0
            for p in pos[:]:
                e, Trem = E[p["s"]], (p["exp"] - days[k]) / 365
                mark = close = 0.0
                for sign, K, vm, tc in p["legs"]:
                    val = p["n"] * put(e["c"][k], K, Trem, vix_c[dte][k] * e["ratio"][k] * vm, rf[k])
                    mark += sign * val
                    close += sign * val * ((1 + tc if sign < 0 else 1 - tc) if Trem > 0 else 1)
                notion[k] += p["notional"] / equity
                if Trem <= 0 or k >= p["b"]:
                    cashv += close
                    res.append(((p["credit"] + close) / p["notional"], p["credit"] / p["notional"]))
                    pos.remove(p)
                else:
                    liab += mark
            equity = cashv + liab
            eq.append(equity)
        eq = pd.Series(eq, cal)
        R = np.array(res)
        r = study6.risk(eq)
        r.update({"Positions": len(R), "Avg result, % of notional": R[:, 0].mean() * 100, "Avg credit, % of notional": R[:, 1].mean() * 100,
                  "Kept of credit %": R[:, 0].sum() / R[:, 1].sum() * 100, "Winning positions %": (R[:, 0] > 0).mean() * 100,
                  "Worst position, % of notional": R[:, 0].min() * 100, "Avg loss when losing, % of notional": R[R[:, 0] < 0, 0].mean() * 100,
                  "Credit collected, % of equity per year": prem / (len(cal) / 252) * 100,
                  "Avg notional % equity": notion.mean() * 100, "Peak notional % equity": notion.max() * 100,
                  "Avg ATM vol at sale %": np.mean(ivs) * 100})
        return eq, r

    out, curves = {"ETF book (base)": study6.risk(base)}, {"ETF book (base)": base}
    for name, dte, legs, mult in (
            ("Sell ATM put 30d", 30, [(-1, "atm")], 1), ("Sell ATM put 60d", 60, [(-1, "atm")], 1),
            ("Sell 20-delta put 30d", 30, [(-1, "20d")], 1), ("Sell 20-delta put 60d", 60, [(-1, "20d")], 1),
            ("Spread ATM/20-delta 30d", 30, [(-1, "atm"), (1, "20d")], 1), ("Spread ATM/20-delta 60d", 60, [(-1, "atm"), (1, "20d")], 1),
            ("Spread 20/10-delta 30d", 30, [(-1, "20d"), (1, "10d")], 1),
            ("Spread ATM/20-delta 30d, 2x", 30, [(-1, "atm"), (1, "20d")], 2), ("Sell ATM put 30d, 2x", 30, [(-1, "atm")], 2)):
        eq, r = run(dte, legs, mult)
        out[name], curves[name] = r, eq
    J = pd.DataFrame(out)
    print(J.to_string()); J.to_csv(ROOT / "results/s10_risk_stats.csv")
    Y = pd.DataFrame({k: bt.yearly(v) * 100 for k, v in curves.items()}); Y.index = Y.index.year
    print("\nyearly returns %:"); print(Y.T.round(1).to_string()); Y.to_csv(ROOT / "results/s10_yearly.csv")
    pd.DataFrame(curves).to_csv(ROOT / "results/s10_equity_curves.csv")


if __name__ == "__main__":
    main()
