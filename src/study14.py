"""Study 14: trade-by-trade log of the all-sector book, ETF version and put-spread version
(sell at-the-money / buy 20-delta put, 30 days, same notional), with each put spread matched to
the ETF trade it belongs to.  Writes results/s14_trades_all.csv and prints 2008-2009.
"""
import numpy as np, pandas as pd, sys, pathlib
from math import sqrt, exp
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt, study5, study6
from study9 import SEG
from study10 import put, LEG

ROOT, DTE = bt.ROOT, 30
pd.set_option("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.1f}".format)


def main():
    xk, acts_tech, cash = study5.setup()
    cal = xk.index
    acts, _ = study6.sector_acts(cal)
    acts["Information Technology"] = acts_tech
    ld = lambda n: pd.read_csv(ROOT / f"data/prices_etf/{n}.csv", index_col=0, parse_dates=True)["Close"]
    rf = ld("_IRX").reindex(cal).ffill().fillna(0).values / 100
    vo, vc = ld("_VIX").shift(1).reindex(cal).ffill().values / 100, ld("_VIX").reindex(cal).ffill().values / 100
    spy_rv = study6.etf_frame("SPY").c.pct_change().rolling(60).std()
    days = (cal - cal[0]).days.values
    rows = []
    for s, tk in study6.ETF.items():
        x = study6.etf_frame(tk)
        ratio = (x.c.pct_change().rolling(60).std() / spy_rv).clip(0.7, 2.0).shift(1).reindex(cal).ffill().values
        x = x.reindex(cal)
        o, c = x.o.values, x.c.values
        eq, T, inv = study5.sim(x, acts[s][2], cash, 0, 0, "none")
        for _, t in T.iterrows():
            a, b = cal.get_loc(t.entry), cal.get_loc(t.exit)
            n_act = int(acts[s][2][max(a - 1, 0):b].sum())
            tot_credit = tot_res = 0.0
            k, nsp, worst = a, 0, 0.0
            while k <= b:
                e = min(k + SEG - 1, b)
                S, Tm, L, credit = o[k], DTE / 365, [], 0.0
                for sign, kind in [(-1, "atm"), (1, "20d")]:
                    d1, vm, tc = LEG[kind]
                    v = vo[k] * ratio[k] * vm
                    K = S if d1 is None else S * exp(-d1 * v * sqrt(Tm) + (rf[k] + v * v / 2) * Tm)
                    credit += -sign * put(S, K, Tm, v, rf[k]) * (1 - tc if sign < 0 else 1 + tc)
                    L.append((sign, K, vm, tc))
                j = e                                         # close at the segment end, or at expiry if earlier
                while days[j] - days[k] > DTE:
                    j -= 1
                Trem = (days[k] + DTE - days[j]) / 365
                close = sum(sign * put(c[j], K, Trem, vc[j] * ratio[j] * vm, rf[j]) * ((1 + tc if sign < 0 else 1 - tc) if Trem > 0 else 1)
                            for sign, K, vm, tc in L)
                res = (credit + close) / S * 100
                tot_credit += credit / S * 100; tot_res += res; worst = min(worst, res); nsp += 1
                k += SEG
            rows.append(dict(ETF=tk, sector=s, entry=t.entry.date(), exit=t.exit.date(), trading_days=b - a + 1, activations=n_act,
                             etf_ret=t.ret * 100, etf_contrib=t.ret * 100 / 11, spreads=nsp, credit=tot_credit, spread_ret=tot_res,
                             spread_contrib=tot_res / 11, vix_at_entry=vo[a] * 100))
    A = pd.DataFrame(rows).sort_values(["entry", "ETF"]).reset_index(drop=True)
    A.to_csv(ROOT / "results/s14_trades_all.csv", index=False)
    W = A[(pd.to_datetime(A.entry) >= "2008-01-01") & (pd.to_datetime(A.entry) <= "2009-12-31")]
    W.to_csv(ROOT / "results/s14_trades_2008_2009.csv", index=False)
    print(W.drop(columns="sector").to_string(index=False))
    for y in (2008, 2009):
        g = W[pd.to_datetime(W.entry).dt.year == y]
        print(f"{y}: {len(g)} trades, ETF winners {(g.etf_ret > 0).sum()}, avg ETF {g.etf_ret.mean():.1f}%, sum of ETF contributions {g.etf_contrib.sum():.1f} pts; "
              f"spreads winners {(g.spread_ret > 0).sum()}, avg {g.spread_ret.mean():.1f}%, sum of contributions {g.spread_contrib.sum():.1f} pts")
    print("all years:", len(A), "trades")


if __name__ == "__main__":
    main()
