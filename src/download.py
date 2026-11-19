"""Download daily OHLCV from Yahoo Finance for the tech universe + benchmark.

Stores split-adjusted OHLC (auto_adjust=False) plus Adj Close (total return).
Signals use split-adjusted OHLC only; returns use dividend-adjusted prices.
"""
import pandas as pd, yfinance as yf, pathlib, sys

root = pathlib.Path(__file__).resolve().parents[1]
m = pd.read_csv(root / "data" / "tech_sector_map.csv")
tickers = sorted(set(m.yahoo.dropna())) + ["XLK", "SPY"]
out = root / "data" / "prices"
out.mkdir(exist_ok=True)
df = yf.download(tickers, start="2006-01-01", auto_adjust=False,
                 group_by="ticker", progress=False, threads=4)
for t in tickers:
    d = df[t].dropna(subset=["Close"])
    if len(d) == 0:
        print("NO DATA", t, file=sys.stderr); continue
    d.to_csv(out / f"{t}.csv")
    print(t, d.index[0].date(), d.index[-1].date(), len(d))
