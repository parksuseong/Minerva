"""131번 차트: (a) 11/03~11/07 깃발·ML 점수와 급락, (b) ML 상위 1% 확인 구간 신호 표본.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <t131 폴더>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

KST = timezone(timedelta(hours=9))
INK, GRID, RED, GREEN, BLUE = "#333333", "#e6e6e6", "#d73027", "#1a9850", "#4575b4"


def ts(y, m, d, hh=0):
    return datetime(y, m, d, hh, tzinfo=timezone.utc).timestamp() * 1000


def main() -> None:
    root = Path(sys.argv[1])
    z = np.load(root / "series.npz")
    r = json.loads((root / "result.json").read_text())
    t, c, y, sc = z["t"], z["c"], z["y"], z["score"]
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    a, b = np.searchsorted(t, ts(2026, 11, 3)), len(t)
    prv = (t >= ts(2025, 1, 1)) & (t < ts(2026, 1, 1)) & np.isfinite(sc)
    th1, th5 = np.quantile(sc[prv], 0.99), np.quantile(sc[prv], 0.95)
    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [4, 1.2, 1.6]})
    p = ax[0]
    p.plot(x[a:b], c[a:b], color=INK, lw=0.8)
    cr = np.flatnonzero(y[a:b]) + a
    p.scatter(x[cr], c[cr], s=8, color=RED, label="bar followed by ≥3% drop within 4h (label)")
    s0, s1 = ts(2026, 11, 5, 13), ts(2026, 11, 5, 15)
    p.axvspan(mdates.date2num(datetime.fromtimestamp(s0 / 1000, KST)),
              mdates.date2num(datetime.fromtimestamp(s1 / 1000, KST)), color=RED, alpha=0.12, label="user's crash 13–15 UTC")
    p.legend(loc="upper left", frameon=False, fontsize=9)
    p.set_title("BTCUSDT futures 15m, 11/03–11/07 (KST): test 131 precursor flags and ML crash score (case is hindsight, not evidence)",
                loc="left", color=INK, fontsize=11)
    f = ax[1]
    for k, (nm, col) in enumerate((("P", BLUE), ("R4", "#fdae61"), ("H1", RED), ("H2", "#a50026"), ("H3", "#542788"))):
        on = z[nm][a:b].astype(bool)
        f.scatter(x[a:b][on], np.full(on.sum(), k), s=6, color=col)
    f.set_yticks(range(5))
    f.set_yticklabels(["squeeze pump P", "4h rollover", "H1", "H2", "H3"])
    m = ax[2]
    m.plot(x[a:b], sc[a:b], color=INK, lw=0.8)
    m.axhline(th1, color=RED, ls="--", lw=0.8, label=f"top 1% threshold {th1:.2f}")
    m.axhline(th5, color="#fdae61", ls="--", lw=0.8, label=f"top 5% threshold {th5:.2f}")
    m.set_ylabel("ML crash score")
    m.legend(loc="upper left", frameon=False, fontsize=9)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    fig.tight_layout()
    fig.savefig(root / "t131_case.png", dpi=110)
    plt.close(fig)
    # (b) ML 상위 1% 확인 구간 신호 표본: 적중 4개, 빗나감 4개
    fires = np.array(r["ml"]["fire_idx_hold_top1"], int)
    hits = [i for i in fires if y[i]][-4:]
    miss = [i for i in fires if not y[i]][-4:]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7))
    for row, (lst, lab, col) in enumerate(((hits, "hit", RED), (miss, "miss", INK))):
        for k in range(4):
            a_ = ax[row, k]
            if k >= len(lst):
                a_.axis("off")
                continue
            i = lst[k]
            s, e = max(0, i - 48), min(len(t), i + 24)
            a_.plot(x[s:e], c[s:e], color=INK, lw=0.8)
            a_.scatter(x[i], c[i], marker="^", s=80, color=RED, zorder=5)
            a_.axhline(c[i] * 0.97, color=RED, ls=":", lw=0.8)
            a_.axvspan(x[i], x[min(i + 16, len(x) - 1)], color=col, alpha=0.08)
            a_.set_title(f"{lab}: {datetime.fromtimestamp(t[i] / 1000, KST):%Y-%m-%d %H:%M} score {sc[i]:.2f}", fontsize=9, color=INK)
            a_.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            a_.tick_params(labelsize=7)
            a_.grid(color=GRID, lw=0.5)
    fig.suptitle("Test 131 ML top-1% crash flags in holdout (2024~): triangle = flag (short), dotted = −3%, shade = next 4h",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(root / "t131_ml_samples.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
