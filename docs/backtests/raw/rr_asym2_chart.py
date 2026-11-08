"""손익비 비대칭 2차 차트: 최근 30일 매매, 구간별 자산 곡선, 진단(+X가 −Y보다 먼저) vs 무작위·손익분기."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import matplotlib.pyplot as plt
import numpy as np
from r5_data import load
from rr_asym_chart import equity, trades_chart


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b = load(DW.R5)
    res = json.loads((out / "result.json").read_text())
    tj = json.loads((out / "trades.json").read_text())["final"]
    fin = res["final"]
    r30 = res["res"][fin]["d30"]
    trades_chart(b, tj["d30"], f"{fin}: wider stop, 1m fills, fees+funding - last 30 days: {r30['n']} trades, "
                 f"win {100 * (r30['win'] or 0):.0f}%, 5x/100% {r30['x5_f1.0']['ret'] * 100:+.1f}%. "
                 "Labels = 5x net (non-stop exits)", out / "chart_d30.png")  # fmt: skip

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    for w, col in (("fresh", "#7b3294"), ("design", "#2c7bb6"), ("val", "#1a9641"), ("year", "#d7191c")):
        tr = tj[w]
        if not tr:
            continue
        for frac, ls in ((1.0, "-"), (0.4, "--")):
            xs, ys = equity(tr, frac)
            ax[0].plot(xs, ys, ls=ls, color=col, lw=1, label=f"{w} {int(frac * 100)}% margin")
    ax[0].set_yscale("log")
    ax[0].axhline(1, color="#333", lw=0.6)
    ax[0].set_title(f"{fin}: equity at 5x by window (log)", loc="left", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.2)
    D = res["diag"]
    names = list(D)
    keys = ("0.03|0.01", "0.05|0.01", "0.05|0.02")
    wdt = 0.8 / (2 * len(keys))
    for j, kk in enumerate(keys):
        x_, y_ = map(float, kk.split("|"))
        xx = np.arange(len(names)) + (2 * j) * wdt - 0.4
        lab = f"+{x_ * 100:g}% before -{y_ * 100:g}%"
        ax[1].bar(xx, [D[n]["sig"][kk] * 100 for n in names], wdt, color=plt.cm.tab10(j), label=f"signal {lab}")
        ax[1].bar(xx + wdt, [D[n]["rnd"][kk] * 100 for n in names], wdt, color=plt.cm.tab10(j), alpha=0.35,
                  label=f"random {lab}")  # fmt: skip
        be = (y_ + 0.0012) / (x_ + y_ + 0.0004) * 100
        ax[1].axhline(be, color=plt.cm.tab10(j), lw=0.8, ls="--")
    ax[1].set_xticks(range(len(names)), names, rotation=20, fontsize=8)
    ax[1].set_title("P(+X before -Y), 2020-03~2025-10; dashed = fee breakeven", loc="left", fontsize=10)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close()
    print("ok")


if __name__ == "__main__":
    main()
