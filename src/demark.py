"""TD Setup, TD Sequential countdown and TD Combo countdown (buy side + sell setup).

Strictly causal: the state at bar i depends only on bars <= i.  The Combo
countdown is counted retroactively from bar 1 of the setup, but only *after*
the setup has completed (bar 9), so nothing is ever known before it happens.

Rules implemented (Perl, "DeMark Indicators", standard settings):
  Buy setup      : 9 consecutive closes < close 4 bars earlier (after a bar that
                   was not, i.e. a bearish price flip).
  Sequential     : from setup bar 9 on, count bars with close <= low[-2]
                   (non-consecutive).  Bar 13 additionally needs
                   low <= close of countdown bar 8, else it is deferred.
  Combo (strict) : from setup bar 1 on, count bars with
                   close <= low[-2], low <= low[-1], close < close[-1],
                   close < close of previous Combo count bar.
  Cancel         : a TD sell setup completes, or a bar's low is above the
                   highest true high of the buy setup (TDST resistance).
  Recycle        : a new buy setup completing while a countdown is running
                   replaces it (countdown restarts) according to `recycle`:
                   "perl"   - only if new setup range is >= old and <= 1.618x old
                   "always" - always restart
                   "never"  - ignore new setups until countdown ends
"""
import numpy as np


def td_buy_signals(o, h, l, c, recycle="perl"):
    n = len(c)
    bs = np.zeros(n, int)      # buy setup count
    ss = np.zeros(n, int)      # sell setup count
    setup9 = np.zeros(n, bool)
    seq13 = np.zeros(n, bool)
    combo13 = np.zeros(n, bool)
    sell9 = np.zeros(n, bool)
    seq_inc = np.zeros(n, int)     # Sequential countdown number printed on this bar (0 = none)
    seq_cancel = np.zeros(n, bool) # a running Sequential countdown was cancelled on this bar

    seq = None    # dict(count, bar8_close, tdst, rng)
    combo = None  # dict(count, last_close, tdst, rng)
    low_ref = None  # (lowest low since setup start, true range of that bar)

    for i in range(n):
        if i >= 4:
            bs[i] = bs[i - 1] + 1 if c[i] < c[i - 4] else 0
            ss[i] = ss[i - 1] + 1 if c[i] > c[i - 4] else 0
        new_setup = bs[i] == 9
        sell9[i] = ss[i] == 9

        # ---- cancellations (checked before counting this bar)
        if sell9[i]:
            if seq is not None:
                seq_cancel[i] = True
            seq = combo = None
        for st in (seq, combo):
            if st is not None and l[i] > st["tdst"]:
                st["dead"] = True
        if seq is not None and seq.get("dead"):
            seq = None
            seq_cancel[i] = True
        if combo is not None and combo.get("dead"):
            combo = None

        started_seq = started_combo = False
        if new_setup:
            setup9[i] = True
            a = i - 8
            th = max(h[a:i + 1].max(), c[a - 1])       # highest true high
            rng = h[a:i + 1].max() - l[a:i + 1].min()

            def accept(st):
                if st is None:
                    return True
                if recycle == "always":
                    return True
                if recycle == "never":
                    return False
                return st["rng"] <= rng <= 1.618 * st["rng"]

            if accept(seq):
                seq = dict(count=0, bar8=np.nan, tdst=th, rng=rng)
                started_seq = True
            if accept(combo):
                combo = dict(count=0, last=np.inf, tdst=th, rng=rng)
                started_combo = True
                # retroactive count over setup bars 1..8 (bar 9 handled below)
                for j in range(a, i):
                    if j >= 2 and c[j] <= l[j - 2] and l[j] <= l[j - 1] \
                            and c[j] < c[j - 1] and c[j] < combo["last"]:
                        combo["count"] += 1
                        combo["last"] = c[j]

        # ---- Sequential countdown on bar i
        if seq is not None and i >= 2 and c[i] <= l[i - 2]:
            if seq["count"] < 12:
                seq["count"] += 1
                seq_inc[i] = seq["count"]
                if seq["count"] == 8:
                    seq["bar8"] = c[i]
            elif l[i] <= seq["bar8"]:
                seq13[i] = True
                seq_inc[i] = 13
                seq = None

        # ---- Combo countdown on bar i
        if combo is not None and i >= 2 and c[i] <= l[i - 2] and l[i] <= l[i - 1] \
                and c[i] < c[i - 1] and c[i] < combo["last"]:
            combo["count"] += 1
            combo["last"] = c[i]
            if combo["count"] == 13:
                combo13[i] = True
                combo = None
    return dict(bs=bs, ss=ss, setup9=setup9, sell9=sell9, seq13=seq13, combo13=combo13,
                seq_inc=seq_inc, seq_cancel=seq_cancel)


def joint_signal(seq13, combo13, sell9, window):
    """Fire on the bar where the *second* 13 prints, if the other 13 printed
    within the previous `window` bars (0 = same bar) and no sell setup
    completed in between.  Each 13 is used at most once."""
    n = len(seq13)
    out = np.zeros(n, bool)
    last_s = last_c = -10**9
    for i in range(n):
        if sell9[i]:
            last_s = last_c = -10**9
        if seq13[i]:
            last_s = i
        if combo13[i]:
            last_c = i
        if (seq13[i] or combo13[i]) and abs(last_s - last_c) <= window \
                and min(last_s, last_c) > -10**8:
            out[i] = True
            last_s = last_c = -10**9
    return out
