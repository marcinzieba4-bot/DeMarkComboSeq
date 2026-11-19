import numpy as np, pandas as pd, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from demark import td_buy_signals, joint_signal

# 1. causality: signals on a truncated series must equal the prefix of signals on the full series
p = pathlib.Path(__file__).resolve().parents[1] / "data" / "prices"
for t in ["AAPL", "INTC", "MU", "XRX"]:
    d = pd.read_csv(p / f"{t}.csv")
    o, h, l, c = (d[k].values for k in ["Open", "High", "Low", "Close"])
    full = td_buy_signals(o, h, l, c)
    for cut in [1000, 2500, 4000]:
        part = td_buy_signals(o[:cut], h[:cut], l[:cut], c[:cut])
        for k in full:
            assert (full[k][:cut] == part[k]).all(), (t, cut, k)
    j = joint_signal(full["seq13"], full["combo13"], full["sell9"], 10)
    print(t, "setup9", full["setup9"].sum(), "seq13", full["seq13"].sum(),
          "combo13", full["combo13"].sum(), "joint10", j.sum())

# 2. synthetic steady decline: first close below close[-4] is bar 7, so setup 9 = bar 15
n = 60
c = np.r_[100, 101, 102, 103, 104, 104 - np.arange(1, n - 4) * 1.0]
h, l, o = c + 0.2, c - 0.2, c
s = td_buy_signals(o, h, l, c)
i9 = np.flatnonzero(s["setup9"])[0]
print("setup9 bar", i9, "seq13 bar", np.flatnonzero(s["seq13"])[:2], "combo13 bar", np.flatnonzero(s["combo13"])[:2])
assert i9 == 15 and np.flatnonzero(s["seq13"])[0] == i9 + 12
assert np.flatnonzero(s["combo13"])[0] == i9 + 4   # 9 retro counts incl. bar 9, then 4 more
print("OK")
