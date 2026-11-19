"""ETFs, indices and the 13-week T-bill yield used by study2."""
import yfinance as yf, pathlib
p = pathlib.Path(__file__).resolve().parents[1] / "data/prices_etf"; p.mkdir(exist_ok=True)
for x in ["XLK", "SPY", "^IRX"]:
    d = yf.download(x, start="1998-01-01", auto_adjust=False, progress=False)
    d.columns = d.columns.get_level_values(0)
    d.to_csv(p / f"{x.replace('^', '_')}.csv")
