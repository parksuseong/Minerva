"""146번 차트: 코인별 TH1 롱 vs 단순 보유 자산(로그), 비율 막대.

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python docs/backtests/raw/th1_alts_chart.py <결과 폴더>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from th1_alts import SYMS

INK, GRID, TH, BH = "#333333", "#e6e6e6", "#1a9850", "#999999"


def main() -> None:
    out = Path(sys.argv[1])
    rep = json.loads((out / "result.json").read_text())
    s = np.load(out / "series.npz")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, sym in zip(axes.ravel(), SYMS):
        r = rep[sym]
        t0 = datetime.strptime(r["span"][0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        for name, col, lab in (("long", TH, "TH1 long"), ("BH", BH, "buy & hold")):
            t, e = s[f"{sym}|{name}|t"], s[f"{sym}|{name}|e"]
            k = t >= t0
            x = mdates.date2num([datetime.fromtimestamp(v / 1000, timezone.utc) for v in t[k]])
            ax.plot(x, e[k] / e[k][0], color=col, lw=1.1 if name == "long" else 0.8, label=lab)
        ax.set_yscale("log")
        a, b = r["long"]["all"], r["BH"]["all"]
        ax.set_title(f"{sym}  TH1 {a.get('cagr')}%/{a.get('mdd')}% (ratio {a.get('ratio')})\n"
                     f"hold {b.get('cagr')}%/{b.get('mdd')}% (ratio {b.get('ratio')}), null p {r['null']['p']}",
                     fontsize=9, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Test 146: TH1 (long-only daily CCI20>0 & RSI14>50, ATR risk 2%, 1% stop) on 8 unseen alt futures, "
                 "1x equity, daily, log scale", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "t146_equity.png", dpi=105)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(SYMS))
    th = [rep[z]["long"]["all"].get("ratio") or 0 for z in SYMS]
    bh = [rep[z]["BH"]["all"].get("ratio") or 0 for z in SYMS]
    sh = [rep[z]["short"]["all"].get("ratio") or 0 for z in SYMS]
    ax.bar(xs - 0.27, th, 0.27, color=TH, label="TH1 long")
    ax.bar(xs, sh, 0.27, color="#d73027", label="mirror short (ref)")
    ax.bar(xs + 0.27, bh, 0.27, color=BH, label="buy & hold")
    ax.axhline(0.5, color=INK, lw=0.8, ls="--")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(xs, [z.replace("USDT", "") for z in SYMS])
    ax.set_ylabel("CAGR / |max DD|", color=INK)
    c = rep["criteria"]
    ax.set_title(f"Ratio by coin  |  median TH1 {c['median_ratio']}, CAGR>0 in {c['n_cagr_pos']}/8, beats hold in "
                 f"{c['n_beat_bh']}/8, null p<0.05 in {c['n_null_p_lt_0.05']}/8  |  pass={c['pass']}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.6, axis="y")
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "t146_ratios.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
