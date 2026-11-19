"""Joint all-sector book vs SPY (log scale, growth of 1)."""
import pandas as pd, matplotlib, pathlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = pathlib.Path(__file__).resolve().parents[1]
eq = pd.read_csv(ROOT / "results/s6_equity_curves.csv", index_col=0, parse_dates=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S = [("SPY buy & hold", "SPY buy & hold", "#9a9993", 1.4),
     ("A shared, max 25%", "Plain trigger, active sectors share capital", "#eb6834", 2),
     ("A equal 1/11", "Plain trigger, 1/11 per sector", "#2a78d6", 2),
     ("B shared, max 25%", "With filter + stop + ban, shared capital", "#1baf7a", 2)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(12, 6), facecolor=SURF)
ax.set_facecolor(SURF)
ends = []
for col, lab, c, lw in S:
    ax.plot(eq.index, eq[col], color=c, lw=lw, label=lab)
    ends.append((eq[col].iloc[-1], lab, c))
ax.set_yscale("log")
prev = 0
for y, lab, c in sorted(ends):
    yy = max(y, prev * 1.09)
    ax.plot([eq.index[-1]], [y], "o", ms=5, color=c, mec=SURF, mew=1.5, zorder=5)
    ax.annotate(f"{lab}  {y:.2f}", (eq.index[-1], yy), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    prev = yy
ax.axhline(1, color=INK2, lw=0.8)
ax.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 5, 7]))
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
ax.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=2500))
ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2008, 2027, 3)])); ax.set_xticklabels(range(2008, 2027, 3))
ax.set_ylabel("Growth of 1, log scale (after costs)", color=INK2)
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
fig.suptitle("Joint 13 trigger on all 11 sectors, combined into one book, 2008 – Nov 2026", x=0.012, ha="left", color=INK, fontsize=14, fontweight="bold")
fig.text(0.012, 0.915, "Two top-30 stocks of a sector fire within 10 days -> buy that sector's ETF for 21 days; T-bills otherwise; 10 bps per side.", color=INK2, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(ROOT / "results/s6_joint_pnl.png", dpi=200, facecolor=SURF)
