"""Study 11: (1) buy SPY instead of the sector ETFs when sectors fire, (2) run the ETF book and the
put-spread book together.

SPY versions (two-stock trigger per sector, 21-day hold, as in study 6 version A):
  SPY scaled      : each active sector puts 1/11 of equity into SPY (same size and timing as the ETF book)
  SPY all-in      : 100% in SPY while at least one sector is active, T-bills otherwise
  SPY all-in >=3  : 100% in SPY only after a day on which 3 or more sectors were active
Combined books add the put-spread book of study 10 (sell at-the-money / buy 20-delta put, 30 days,
model-priced) on top of the ETF or SPY book: both run on the same capital, so the combined daily
return is ETF return + (spread return - T-bill).  "Half + half" runs each at half size.
"""
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import backtest as bt, study5, study6

ROOT = bt.ROOT
pd.set_option("display.width", 300, "display.max_columns", 60, "display.float_format", "{:.2f}".format)


def main():
    xk, acts_tech, cash = study5.setup()
    cal = xk.index
    acts, _ = study6.sector_acts(cal)
    acts["Information Technology"] = acts_tech
    rf = pd.Series(cash, cal)
    spy = study6.etf_frame("SPY").reindex(cal)
    R_etf, R_spy, I = {}, {}, {}
    for s, tk in study6.ETF.items():
        for dst, x in ((R_etf, study6.etf_frame(tk).reindex(cal)), (R_spy, spy)):
            eq, T, inv = study5.sim(x, acts[s][2], cash, 0, 0, "none")
            dst[s] = eq.pct_change().fillna(eq.iloc[0] - 1)
        I[s] = inv
    I = pd.DataFrame(I, index=cal)
    n_act = I.sum(axis=1)
    etf = pd.DataFrame(R_etf).mean(axis=1)
    spy_scaled = pd.DataFrame(R_spy).mean(axis=1)
    any_act = np.any([acts[s][2] for s in acts], axis=0)
    eq_any, T_any, inv_any = study5.sim(spy, any_act, cash, 0, 0, "none")
    three = (n_act >= 3).values                               # known at the close -> acted on next open
    eq_3, T_3, inv_3 = study5.sim(spy, three, cash, 0, 0, "none")
    opt = pd.read_csv(ROOT / "results/s10_equity_curves.csv", index_col=0, parse_dates=True)
    spread = opt["Spread ATM/20-delta 30d"].pct_change().fillna(0) - rf
    atm = opt["Sell ATM put 30d"].pct_change().fillna(0) - rf

    books = {"Sector ETFs (base)": etf, "SPY scaled (1/11 per active sector)": spy_scaled,
             "SPY all-in, any sector active": eq_any.pct_change().fillna(0),
             "SPY all-in, 3+ sectors active": eq_3.pct_change().fillna(0),
             "Put spreads only": spread + rf,
             "Sector ETFs + put spreads": etf + spread,
             "SPY scaled + put spreads": spy_scaled + spread,
             "Half ETFs + half put spreads": 0.5 * etf + 0.5 * (spread + rf),
             "Sector ETFs + sold ATM puts": etf + atm}
    expo = {"Sector ETFs (base)": n_act / 11, "SPY scaled (1/11 per active sector)": n_act / 11,
            "SPY all-in, any sector active": pd.Series(inv_any, cal).astype(float),
            "SPY all-in, 3+ sectors active": pd.Series(inv_3, cal).astype(float)}
    spy_r = spy.c.pct_change().fillna(0)
    out, curves = {}, {}
    for k, r in books.items():
        eq = (1 + r).cumprod()
        x = r - rf
        d = study6.risk(eq)
        d.update({"Return above T-bills %/yr": x.mean() * 252 * 100, "Sharpe above T-bills": x.mean() / x.std() * np.sqrt(252),
                  "Beta to SPY": np.cov(r, spy_r)[0, 1] / spy_r.var(), "Time invested %": (expo[k] > 0).mean() * 100 if k in expo else np.nan,
                  "Avg ETF/SPY exposure %": expo[k].mean() * 100 if k in expo else np.nan})
        out[k], curves[k] = d, eq
    bh = spy.c / spy.c.iloc[0]
    d = study6.risk(bh); x = spy_r - rf
    d.update({"Return above T-bills %/yr": x.mean() * 252 * 100, "Sharpe above T-bills": x.mean() / x.std() * np.sqrt(252), "Beta to SPY": 1.0,
              "Time invested %": 100.0, "Avg ETF/SPY exposure %": 100.0})
    out["SPY buy & hold"], curves["SPY buy & hold"] = d, bh
    J = pd.DataFrame(out)
    print(J.to_string()); J.to_csv(ROOT / "results/s11_risk_stats.csv")
    for nm, T in (("any sector", T_any), ("3+ sectors", T_3)):
        w = T.ret > 0
        print(f"\nSPY all-in ({nm}): {len(T)} trades, hit {w.mean():.0%}, avg {T.ret.mean() * 100:.2f}%, worst {T.ret.min() * 100:.1f}%, "
              f"best {T.ret.max() * 100:.1f}%, avg length {((T.exit - T.entry).dt.days).mean():.0f} calendar days")
    print("\nper sector, average 21-day-style trade: sector ETF vs SPY over the same days (% per day invested, annualised)")
    inv_days = I.sum()
    cmp = pd.DataFrame({"days invested": inv_days,
                        "sector ETF %/yr while invested": [(pd.Series(R_etf[s]) - rf)[I[s].values].mean() * 252 * 100 for s in I],
                        "SPY %/yr on the same days": [(pd.Series(R_spy[s]) - rf)[I[s].values].mean() * 252 * 100 for s in I]})
    cmp["ETF minus SPY"] = cmp.iloc[:, 1] - cmp.iloc[:, 2]
    print(cmp.round(1).to_string())
    cmp.to_csv(ROOT / "results/s11_sector_vs_spy.csv")
    Y = pd.DataFrame({k: bt.yearly(v) * 100 for k, v in curves.items()}); Y.index = Y.index.year
    print("\nyearly returns %:"); print(Y.T.round(1).to_string()); Y.to_csv(ROOT / "results/s11_yearly.csv")
    pd.DataFrame(curves).to_csv(ROOT / "results/s11_equity_curves.csv")
    print("\ncorrelation of daily returns above cash, ETF book vs put-spread book:", round(np.corrcoef(etf - rf, spread)[0, 1], 2))


if __name__ == "__main__":
    main()
