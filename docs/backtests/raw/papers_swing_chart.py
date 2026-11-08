"""스윙 논문 규칙 차트: 최근 30일 매매(가격 + Stoch·CCI·RSI), 구간별 5배 자산 곡선.

실행: python3 papers_swing_chart.py <OUT> <규칙1> [<규칙2>]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import matplotlib.pyplot as plt
from r5_data import load
from rr_asym_chart import equity, trades_chart

COLS = (("fresh", "#7b3294"), ("design", "#2c7bb6"), ("val", "#1a9641"), ("year", "#d7191c"))


def kdt(v):
    return datetime.fromtimestamp(v / 1000, timezone.utc)


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    names = sys.argv[2:4]
    b = load(DW.R5)
    res = json.loads((out / "result.json").read_text())
    tj = json.loads((out / "trades.json").read_text())
    cv = json.loads((out / "curves.json").read_text())
    for i, nm in enumerate(names):
        r30 = res["res"][nm]["d30"]
        fn = "chart_d30.png" if i == 0 else f"chart_d30_{nm}.png"
        trades_chart(b, tj[nm]["d30"], f"{nm}: daily rule, futures 1m fills, fees+funding - last 30 days: "
                     f"{r30['n']} trades, 5x/100% {r30['x5_f1.0']['ret'] * 100:+.1f}%, 5x/40% "
                     f"{r30['x5_f0.4']['ret'] * 100:+.1f}%. Labels = 5x per trade (1x x5)", out / fn)  # fmt: skip
    fig, ax = plt.subplots(1, len(names), figsize=(9 * len(names), 6), squeeze=False)
    for i, nm in enumerate(names):
        a = ax[0][i]
        for w, col in COLS:
            for f, ls in ((1.0, "-"), (0.4, "--")):
                if nm in cv:
                    pts = cv[nm].get(f"{w}|{f}", [])
                    xs, ys = [kdt(t) for t, _ in pts], [max(e, 1e-6) for _, e in pts]
                else:
                    xs, ys = equity(tj[nm][w], f)
                if xs:
                    a.plot(xs, ys, ls=ls, color=col, lw=1, label=f"{w} {int(f * 100)}% margin")
        a.set_yscale("log")
        a.axhline(1, color="#333", lw=0.6)
        p = res["rnd"][nm].get("pool", {}).get("p")
        a.set_title(f"{nm}: equity at 5x by window (log), random p (2020-03~2025-10) = {p}", loc="left",
                    fontsize=10)  # fmt: skip
        a.legend(fontsize=7)
        a.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close()
    print("ok")


if __name__ == "__main__":
    main()
