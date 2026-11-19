"""XLK books: growth of 1 (top) and distance below the previous high (bottom)."""
import pandas as pd, matplotlib, pathlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
eq = pd.read_csv(ROOT / "results/s12_equity_curves.csv", index_col=0, parse_dates=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S = [("XLK ETF book", "ETF book", "#2a78d6"), ("XLK put-spread book", "Put-spread book", "#eb6834"), ("50/50 mix", "50/50 mix", "#1baf7a")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True, facecolor=SURF, gridspec_kw={"height_ratios": [3, 2]})
a1.plot(eq.index, eq["Money market only"], color="#9a9993", lw=1.2, ls=(0, (4, 3)), label="Money market only")
ends = []
for col, lab, c in S:
    a1.plot(eq.index, eq[col], color=c, lw=2, label=lab)
    ends.append((eq[col].iloc[-1], lab, c))
    dd = (eq[col] / eq[col].cummax() - 1) * 100
    a2.plot(eq.index, dd, color=c, lw=1.6)
    a2.annotate(f"{lab}  {dd.min():.0f}%", (dd.idxmin(), dd.min()), xytext=(10, -2), textcoords="offset points", color=INK, fontsize=9, va="center")
ends.append((eq["Money market only"].iloc[-1], "Money market", "#9a9993"))
prev = -9
for y, lab, c in sorted(ends):
    yy = max(y, prev + 0.1)
    a1.plot([eq.index[-1]], [y], "o", ms=5, color=c, mec=SURF, mew=1.5, zorder=5)
    a1.annotate(f"{lab}  {y:.2f}", (eq.index[-1], yy), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    prev = yy
a1.axhline(1, color=INK2, lw=0.8); a2.axhline(0, color=INK2, lw=0.8)
a1.set_title("Growth of 1 (after costs, idle cash in the money market)", loc="left", color=INK, fontsize=11, fontweight="bold")
a2.set_title("Distance below the previous high, %", loc="left", color=INK, fontsize=11, fontweight="bold")
for ax in (a1, a2):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
a1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
a2.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=1500))
a2.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2008, 2027, 3)])); a2.set_xticklabels(range(2008, 2027, 3))
fig.suptitle("XLK trigger: ETF book, put-spread book and a 50/50 mix, 2008 – Nov 2026", x=0.012, ha="left", color=INK, fontsize=14, fontweight="bold")
fig.text(0.012, 0.935, "Trigger: two of the 30 biggest tech stocks print a joint buy 13 within 10 days. Option prices are modelled. XLK buy-and-hold: 16.05",
         color=INK2, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.925))
fig.savefig(ROOT / "results/s12_pnl.png", dpi=200, facecolor=SURF)
