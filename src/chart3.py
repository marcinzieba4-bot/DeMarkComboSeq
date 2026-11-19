"""PnL chart for study 3: growth of 1 for the three ways of trading each signal."""
import pandas as pd, matplotlib, pathlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
eq = pd.read_csv(ROOT / "results/s3_equity_curves.csv", index_col=0, parse_dates=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"stock": "#2a78d6", "XLK": "#eb6834", "pair": "#1baf7a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, facecolor=SURF)
for ax, side, title in zip(axes, ("BUY", "SELL"), ("Buy signals (joint buy 13)", "Sell signals (joint sell 13)")):
    ax.set_facecolor(SURF)
    cols = [c for c in eq.columns if c.startswith(side)]
    ends = []
    ax.plot(eq.index, eq["T-bills"], color="#9a9993", lw=1.2, ls=(0, (4, 3)), label="T-bills only (idle-cash baseline)")
    for c, key in zip(cols, ("stock", "XLK", "pair")):
        lab = c.split("| ")[1][2:]
        lab = lab[0].upper() + lab[1:]
        ax.plot(eq.index, eq[c], color=COL[key], lw=2, label=lab)
        ends.append((eq[c].iloc[-1], lab, COL[key]))
    # direct labels at line ends, nudged apart
    ends.sort()
    y_prev = -9
    for y, lab, col in ends:
        yy = max(y, y_prev + 0.085)
        ax.plot([eq.index[-1]], [y], "o", ms=5, color=col, mec=SURF, mew=1.5, zorder=5)
        ax.annotate(f"{lab}  {y:.2f}", (eq.index[-1], yy), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
        y_prev = yy
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=2300))
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2008, 2027, 3)]))
    ax.set_xticklabels(range(2008, 2027, 3))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
axes[0].set_ylabel("Growth of 1 (after costs)", color=INK2)
fig.suptitle("Tech joint 13 signals, 2008 – Nov 2026: three ways to trade each signal",
             x=0.012, ha="left", color=INK, fontsize=14, fontweight="bold")
fig.text(0.012, 0.915, "10 slots of 10% each, 21-day hold, entry next open, 10 bps per side per leg, idle cash in T-bills. "
         "XLK buy-and-hold over the same period: 16.05", color=INK2, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(ROOT / "results/s3_pnl.png", dpi=200, facecolor=SURF)
