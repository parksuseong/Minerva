"""진입 위치·횟수 시험 차트: 최근 30일 매매(P2), 구간별 수익·낙폭 막대, 예시 하루 진입 위치 그림."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import ql_upgrade as U
from ql_upgrade_chart import kdt, trades_chart
from r5_data import load


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b = load(DW.R5)
    res = json.loads((out / "result.json").read_text())
    ph = json.loads((out / "posthoc.json").read_text())
    pt = json.loads((out / "posthoc_trades.json").read_text())
    tr = pt["P2"]["d30"]
    trades_chart(b, tr, "P2 (current best, unchanged): NY first-hour breakout, limit at the 15m close, exit by NY 16:00 - "
                 f"last 30 days: {len(tr)} trades, labels = 5x net per trade (100% margin)", out / "chart_d30.png")  # fmt: skip

    rows = [("P2 (current best)", ph["P2"]["res"]), ("A2 split 50/50 entry", res["A"]["A2"]),
            ("A3b 50% retrace entry", res["A"]["A3b"]), ("R2 P2 + same-day re-entry (post-hoc)", ph["R2"]["res"]),
            ("pre-registered final (+ range-length union)", res["final"])]  # fmt: skip
    wins = ("fresh", "design", "val", "year", "m6", "d30")
    lab = [
        "2020-03~21-10\n(untouched)",
        "2022-01~24-06",
        "2024-07~25-10",
        "last year",
        "6 months",
        "30 days",
    ]
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    wdt = 0.16
    cols = ["#1a9850", "#2c7bb6", "#7b3294", "#fdae61", "#d73027"]
    for j, (nm, r) in enumerate(rows):
        v = [r[w]["x5_f1.0"]["ret"] * 100 for w in wins]
        dd = [r[w]["x5_f1.0"]["mdd"] * 100 for w in wins]
        xx = np.arange(len(wins)) + (j - 2) * wdt
        ax[0].bar(xx, np.sign(v) * np.log10(1 + np.abs(v)), wdt, color=cols[j], label=nm)
        ax[1].bar(xx, dd, wdt, color=cols[j], label=nm)
    for z in ax:
        z.set_xticks(range(len(wins)), lab, fontsize=8)
        z.grid(alpha=0.2, axis="y")
    ax[0].set_yticks(
        [-2, -1, 0, 1, 2, 3, 4], ["-100%", "-10%", "0", "+10%", "+100%", "+1,000%", "+10,000%"]
    )
    ax[0].axhline(0, color="#333", lw=0.6)
    ax[0].set_title("5x, 100% margin: return by window (log scale)", loc="left", fontsize=10)
    ax[1].set_title("max drawdown % (5x, 100% margin)", loc="left", fontsize=10)
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "windows.png", dpi=100)
    plt.close()

    # 예시 하루: 최근 6개월 P2 거래 중 마지막 이긴 날
    m6 = [x for x in pt["P2"]["m6"] if x["r"] > 0]
    x = m6[-1]
    m = U.load_1m(int(b["t"][-1]) + U.M15)
    a = int(np.searchsorted(m["t"], x["st"]))
    z = a + 60
    H, L = float(m["h"][a:z].max()), float(m["l"][a:z].min())
    rng = H - L
    j = int(x["j"])
    d = x["d"]
    cj = float(m["c"][j])
    edge = H if d == 1 else L
    lv = {"A0 breakout 15m close (current)": cj, "A1 range edge retest": edge,
          "A3b 50% retrace": cj - d * 0.5 * abs(cj - edge),
          "A4 inside-range limit": (L + 0.25 * rng) if d == 1 else (H - 0.25 * rng),
          "stop (other side of range)": L if d == 1 else H, "take-profit (A0)": cj + d * rng}  # fmt: skip
    k0, k1 = a - 30, min(int(x["x"]) + 30, len(m["t"]) - 1)
    X = md.date2num([kdt(v) for v in m["t"][k0:k1]])
    fig, axx = plt.subplots(figsize=(15, 6))
    axx.plot(X, m["c"][k0:k1], color="#555", lw=0.8)
    axx.axvspan(
        md.date2num(kdt(m["t"][a])),
        md.date2num(kdt(m["t"][z])),
        color="#ccc",
        alpha=0.4,
        label="first hour range",
    )
    cols = ["#1a9850", "#2c7bb6", "#7b3294", "#fdae61", "#d73027", "#1a9850"]
    for (nm, y), c in zip(lv.items(), cols):
        axx.axhline(
            y,
            color=c,
            lw=1.2,
            ls="--" if "stop" in nm or "take" in nm else "-",
            label=f"{nm}: {y:,.0f}",
        )
    axx.scatter(md.date2num(kdt(x["t_in"])), x["ep"], marker="o" if d == 1 else "v", s=90, color="#1a9850" if d == 1 else "#d73027",
                zorder=5, edgecolor="white")  # fmt: skip
    axx.scatter(md.date2num(kdt(x["t_out"])), x["xp"], marker="o" if d == 1 else "^", s=90, color="#d73027" if d == 1 else "#1a9850",
                zorder=5, edgecolor="white")  # fmt: skip
    axx.xaxis.set_major_formatter(md.DateFormatter("%m-%d %H:%M"))
    axx.set_title(f"Entry locations on {kdt(x['st']):%Y-%m-%d} (KST) - {'long' if d == 1 else 'short'}, 1m closes; "
                  f"P2 result {5 * x['r'] * 100:+.1f}% at 5x ({x['why']})", loc="left", fontsize=10)  # fmt: skip
    axx.legend(fontsize=8, loc="best")
    axx.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "entry_levels.png", dpi=100)
    plt.close()
    print("ok")


if __name__ == "__main__":
    main()
