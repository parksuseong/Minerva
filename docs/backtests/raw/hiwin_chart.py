"""고승률 차트: 최근 30일 매매(NRg|E4, R1|E1), 구간별 자산 곡선."""

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

SHOW = (("NRg|E4", "chart_d30.png"), ("R1|E1", "chart_d30_R1E1.png"))


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b = load(DW.R5)
    res = json.loads((out / "result.json").read_text())["res"]
    tj = json.loads((out / "trades.json").read_text())
    for name, fn in SHOW:
        r30 = res[name]["d30"]
        trades_chart(b, tj[name]["d30"], f"{name}: 1m fills, fees+funding - last 30 days: {r30['n']} trades, "
                     f"win {100 * (r30['win'] or 0):.0f}%, 5x/100% {r30['x5_f1.0']['ret'] * 100:+.1f}%. "
                     "Labels = 5x net (non-stop exits)", out / fn)  # fmt: skip
    fig, ax = plt.subplots(1, len(SHOW), figsize=(18, 6))
    for a, (name, _) in zip(ax, SHOW):
        for w, col in (("fresh", "#7b3294"), ("design", "#2c7bb6"), ("val", "#1a9641"), ("year", "#d7191c")):
            tr = tj[name][w]
            if not tr:
                continue
            for frac, ls in ((1.0, "-"), (0.4, "--")):
                xs, ys = equity(tr, frac)
                wr = 100 * (res[name][w]["win"] or 0)
                a.plot(xs, ys, ls=ls, color=col, lw=1, label=f"{w} {int(frac * 100)}% margin (win {wr:.0f}%)")
        a.set_yscale("log")
        a.axhline(1, color="#333", lw=0.6)
        a.set_title(f"{name}: equity at 5x by window (log)", loc="left", fontsize=10)
        a.legend(fontsize=7)
        a.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close()
    print("ok")


if __name__ == "__main__":
    main()
