"""Study 7: the all-sector book (study 6, version A, 1/11 per sector) with 2x leverage on the
sector ETFs and an unleveraged SPY short as the second leg.

Per active sector, as a share of equity:   long sector ETF = L/11,   short SPY = h/11.
Cash and interest (daily):
  free cash   = 1 - (L - h) * N/11      (short proceeds offset the purchase of the longs)
  earns T-bill when positive, pays T-bill + 1.0% when negative
  short leg   : pays SPY's total return (dividends included), and the proceeds earn the
                short rebate = T-bill - 0.25% borrow fee.  That rebate is the repo rate
                implied by the short; it is reported per year.
Costs: 10 bps per side on every leg, sized by its notional.
"""
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt, study5, study6
from study3 import OOS

ROOT, COST, FEE, SPREAD = bt.ROOT, 0.001, 0.0025, 0.01
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format)


def main():
    xk, acts_tech, cash = study5.setup()
    cal = xk.index
    acts, _ = study6.sector_acts(cal)
    acts["Information Technology"] = acts_tech
    rf = pd.Series(cash, cal)
    spy = study6.etf_frame("SPY").reindex(cal)
    R, I = {}, {}
    for s, tk in study6.ETF.items():
        x = study6.etf_frame(tk).reindex(cal)
        eq, T, inv = study5.sim(x, acts[s][2], cash, 0, 0, "none")
        R[s], I[s] = eq.pct_change().fillna(eq.iloc[0] - 1), pd.Series(inv, cal)
    R, I = pd.DataFrame(R), pd.DataFrame(I).astype(bool)
    fresh = I & ~I.shift(1, fill_value=False)
    last = I & ~I.shift(-1, fill_value=False)
    spy_cc, spy_oc = spy.c.pct_change().fillna(0), spy.c / spy.o - 1
    # return of a SPY short opened/closed together with each sector sleeve (per unit notional)
    S = pd.DataFrame({s: -np.where(fresh[s], spy_oc, spy_cc) - COST * fresh[s] - COST * last[s] for s in I}, index=cal).where(I, 0)
    Lret = R.where(I, 0)                               # sector ETF return while invested, costs included
    N = I.sum(axis=1)

    def book(L, h):
        w = 1 / 11
        free = 1 - (L - h) * N * w
        interest = np.where(free >= 0, free * rf, free * (rf + SPREAD / 252))
        rebate = h * N * w * (rf - FEE / 252)
        legs_long, legs_short = L * w * Lret.sum(axis=1), h * w * S.sum(axis=1)
        r = legs_long + legs_short + interest + (rebate - h * N * w * rf)   # rebate replaces T-bill on proceeds
        return (1 + r).cumprod(), dict(long=legs_long, short=legs_short, interest=pd.Series(interest, cal),
                                       fee=-(h * N * w * FEE / 252), gross=(L + h) * N * w, net=(L - h) * N * w, free=free)

    books = {"1x long (study 6 base)": (1, 0), "2x long": (2, 0), "2x long + 1x short SPY": (2, 1), "1x long + 1x short SPY": (1, 1)}
    out, curves, parts = {}, {}, {}
    for name, (L, h) in books.items():
        eq, p = book(L, h)
        r = study6.risk(eq)
        yrs = len(cal) / 252
        r.update({"Avg gross exposure %": p["gross"].mean() * 100, "Peak gross exposure %": p["gross"].max() * 100,
                  "Avg net exposure %": p["net"].mean() * 100, "Peak net exposure %": p["net"].max() * 100,
                  "Lowest free cash %": p["free"].min() * 100,
                  "From sector longs, %/yr": p["long"].sum() / yrs * 100, "From SPY short, %/yr": p["short"].sum() / yrs * 100,
                  "From interest, %/yr": p["interest"].sum() / yrs * 100, "Borrow fee on short, %/yr": np.sum(p["fee"]) / yrs * 100,
                  "Beta to SPY": np.cov(eq.pct_change().dropna(), spy_cc.iloc[1:])[0, 1] / spy_cc.iloc[1:].var(),
                  "Corr to SPY": np.corrcoef(eq.pct_change().dropna(), spy_cc.iloc[1:])[0, 1]})
        out[name], curves[name], parts[name] = r, eq, p
    J = pd.DataFrame(out)
    print("=== RISK STATS 2008 -> Nov 2026 ==="); print(J.to_string())
    J.to_csv(ROOT / "results/s7_risk_stats.csv")

    # implied repo rate on the short
    short_on = N > 0
    ann = rf * 252 * 100
    rep = pd.DataFrame({"days short open": short_on.groupby(cal.year).sum(),
                        "avg short size % equity": (N / 11 * 100).where(short_on).groupby(cal.year).mean(),
                        "T-bill % (when short)": ann.where(short_on).groupby(cal.year).mean(),
                        "implied repo / rebate %": (ann - FEE * 100).where(short_on).groupby(cal.year).mean(),
                        "rebate earned, % of equity": ((N / 11) * (rf - FEE / 252)).groupby(cal.year).sum() * 100,
                        "SPY dividends paid, % of equity": np.nan})
    div = (spy.c.pct_change() - pd.read_csv(ROOT / "data/prices_etf/SPY.csv", index_col=0, parse_dates=True)["Close"].reindex(cal).pct_change()).fillna(0)
    rep["SPY dividends paid, % of equity"] = ((N / 11) * div).groupby(cal.year).sum() * 100
    wavg = ((ann - FEE * 100) * N).sum() / N.sum()
    print("\n=== IMPLIED REPO RATE ON THE SPY SHORT (rebate = T-bill - 0.25% fee) ===")
    print(rep.round(2).to_string())
    print(f"notional-weighted average implied repo rate over all short days: {wavg:.2f}% a year "
          f"(T-bill {wavg + FEE * 100:.2f}% less 0.25% fee); short open on {short_on.mean():.0%} of days")
    print(f"total rebate earned {rep.iloc[:, 4].sum():.2f}% of equity over the period; SPY dividends paid on the short {rep.iloc[:, 5].sum():.2f}%")
    rep.to_csv(ROOT / "results/s7_implied_repo.csv")
    Y = pd.DataFrame({k: bt.yearly(v) * 100 for k, v in curves.items()}); Y.index = Y.index.year
    Y["SPY"] = (bt.yearly(spy.c / spy.c.iloc[0]) * 100).values
    print("\nyearly returns %:"); print(Y.T.round(1).to_string())
    Y.to_csv(ROOT / "results/s7_yearly.csv")
    pd.DataFrame(curves).assign(SPY=spy.c / spy.c.iloc[0]).to_csv(ROOT / "results/s7_equity_curves.csv")


if __name__ == "__main__":
    main()
