"""Download every ticker that was an S&P 500 member at any time since 2008."""
import pandas as pd, yfinance as yf, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
m = pd.read_csv(root / "data/sp500_ticker_start_end.csv", parse_dates=["start_date", "end_date"])
m = m[m.end_date.isna() | (m.end_date >= "2008-01-01")]
tick = sorted(set(m.ticker))
out = root / "data/prices_all"; out.mkdir(exist_ok=True)
ok = 0
for i in range(0, len(tick), 150):
    chunk = tick[i:i + 150]
    df = yf.download([t.replace(".", "-") for t in chunk], start="2006-01-01", auto_adjust=False,
                     group_by="ticker", progress=False, threads=4)
    for t in chunk:
        try:
            d = df[t.replace(".", "-")].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(d) >= 300:
            d.to_csv(out / f"{t}.csv"); ok += 1
print("tickers", len(tick), "with data", ok)
