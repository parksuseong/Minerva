"""단서 모음 차트: 가장 나은 안 최근 30일 매매(chart_d30.png), 구간별 자산 곡선(equity.png).

사용: python3 leads_ens_chart.py OUT [안 이름]  (이름 없으면 고른 안)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import matplotlib.pyplot as plt
from r5_data import load
from rr_asym_chart import equity, trades_chart


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    allr = json.loads((out / "result.json").read_text())
    res = allr["res"]
    name = sys.argv[2] if len(sys.argv) > 2 else allr["final"]
    tj = json.loads((out / "trades.json").read_text())[name]
    b = load(DW.R5)
    r30 = res[name]["d30"]
    trades_chart(b, tj["d30"], f"{name}: 1m fills, fees+funding - last 30 days: {r30['n']} trades, "
                 f"win {100 * (r30['win'] or 0):.0f}%, 5x/100% {r30['x5_f1.0']['ret'] * 100:+.1f}%, "
                 f"5x/40% {r30['x5_f0.4']['ret'] * 100:+.1f}%. Labels = 5x net (non-stop exits)",
                 out / "chart_d30.png")  # fmt: skip
    fig, ax = plt.subplots(figsize=(12, 6))
    for w, col in (("fresh", "#7b3294"), ("design", "#2c7bb6"), ("val", "#1a9641"), ("year", "#d7191c")):
        tr = tj[w]
        if not tr:
            continue
        for frac, ls in ((1.0, "-"), (0.4, "--")):
            xs, ys = equity(tr, frac)
            wr = 100 * (res[name][w]["win"] or 0)
            ax.plot(xs, ys, ls=ls, color=col, lw=1, label=f"{w} {int(frac * 100)}% margin (win {wr:.0f}%)")
    ax.set_yscale("log")
    ax.axhline(1, color="#333", lw=0.6)
    ax.set_title(f"{name}: equity at 5x by window (log)", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close()
    print("ok", name)


if __name__ == "__main__":
    main()
