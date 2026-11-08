"""BTC 더 파기 차트: 고른 안마다 최근 30일 매매(가격 + Stoch·CCI·RSI)와 구간별 5배 자산 곡선."""

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
    names = sys.argv[2:]
    b = load(DW.R5)
    res = json.loads((out / "result.json").read_text())
    tj = json.loads((out / "trades.json").read_text())
    fig, axs = plt.subplots(1, len(names), figsize=(9 * len(names), 6), squeeze=False)
    for j, fn in enumerate(names):
        tag = fn.replace("|", "_")
        r30 = res["res"][fn]["d30"]
        trades_chart(b, tj[fn]["d30"], f"{fn}: 1m fills, fees+funding - last 30 days: {r30['n']} trades, "
                     f"win {100 * (r30['win'] or 0):.0f}%, 5x/100% {r30['x5_f1.0']['ret'] * 100:+.1f}%. "
                     "Labels = 5x net (non-stop exits)", out / f"chart_d30_{tag}.png")  # fmt: skip
        ax = axs[0][j]
        for w, col in (("fresh", "#7b3294"), ("design", "#2c7bb6"), ("val", "#1a9641"), ("year", "#d7191c")):
            tr = tj[fn].get(w) or []
            if not tr:
                continue
            for frac, ls in ((1.0, "-"), (0.4, "--")):
                xs, ys = equity(tr, frac)
                ax.plot(xs, ys, ls=ls, color=col, lw=1, label=f"{w} {int(frac * 100)}% margin")
        ax.set_yscale("log")
        ax.axhline(1, color="#333", lw=0.6)
        ax.set_title(f"{fn}: equity at 5x by window (log)", loc="left", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close()
    print("ok")


if __name__ == "__main__":
    main()
