"""테이커 흐름 따라가기 차트: 최근 30일 매매(가격 + Stoch·CCI·RSI + 테이커 z)와 구간별 5배 자산 곡선.

실행: python3 taker_follow_chart.py <out> <이름1> [이름2 ...]  (첫 이름이 chart_d30.png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import deep as DP
import matplotlib.dates as md
import matplotlib.pyplot as plt
import nonprice_data as ND
import numpy as np
import ql_upgrade as U
from btc_fractal import cci, rsi, stoch
from matplotlib.lines import Line2D
from ql_upgrade_chart import G, R, kdt
from r5_data import load
from rr_asym_chart import equity


def trades_chart(b, tr, z, th, title, path) -> None:
    t = b["t"]
    a = int(np.searchsorted(t, t[-1] - 30 * U.DAY))
    X = md.date2num([kdt(v) for v in t[a:]])
    k, cc, rr = stoch(b["h"], b["l"], b["c"], 5), cci(b["h"], b["l"], b["c"], 20), rsi(b["c"], 14)
    fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1.2]})
    ax[0].plot(X, b["c"][a:], color="#555", lw=0.8)
    for x in tr:
        long = x["d"] == 1
        ax[0].scatter(md.date2num(kdt(x["t_in"])), x["ep"], marker="o" if long else "v", s=60,
                      color=G if long else R, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].scatter(md.date2num(kdt(x["t_out"])), x["xp"], marker="o" if long else "^", s=60,
                      color=R if long else G, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].annotate(f"{5 * x['r'] * 100:+.1f}%", (md.date2num(kdt(x["t_out"])), x["xp"]), fontsize=7,
                       color=G if x["r"] > 0 else R, xytext=(3, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(title, loc="left", fontsize=10)
    panels = ((k, "Stoch 5", 20, 80), (cc, "CCI 20", -100, 100), (rr, "RSI 14", 30, 70), (z, "taker z 4h", -th, th))
    for i_, (y, nm, lo, hi) in enumerate(panels, 1):
        ax[i_].plot(X, y[a:], lw=0.6, color="#2c7bb6" if i_ < 4 else "#7b3294")
        ax[i_].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[i_].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[i_].set_ylabel(nm)
    ax[4].axhline(0, lw=0.5, color="#333")
    ax[4].xaxis.set_major_formatter(md.DateFormatter("%m-%d"))
    for zz in ax:
        zz.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    names = sys.argv[2:]
    b = load(DW.R5)
    res = json.loads((out / "result.json").read_text())
    tj = json.loads((out / "trades.json").read_text())
    th = res["th"]
    v, tb = b["v"].astype(float), b["tbv"].astype(float)
    with np.errstate(all="ignore"):
        zs = {nm: ND.roll_z(DP.rmean(tb, N) / DP.rmean(v, N)) for nm, N in (("T2h", 8), ("T4h", 16), ("T8h", 32))}
    for j, fn in enumerate(names):
        feat = res["spec"][fn][0]
        z = zs.get(feat, zs["T4h"])
        r30 = res["res"][fn]["d30"]
        fname = "chart_d30.png" if j == 0 else f"chart_d30_{fn.replace('|', '_')}.png"
        trades_chart(b, tj[fn]["d30"], z, th.get(feat, th["T4h"]),
                     f"{fn}: 1m fills, fees+funding - last 30 days: {r30['n']} trades, "
                     f"win {100 * (r30['win'] or 0):.0f}%, 5x/100% {r30['x5_f1.0']['ret'] * 100:+.1f}%. "
                     f"Labels = 5x net. Bottom: taker-buy-share z ({feat}), dashed = threshold", out / fname)  # fmt: skip
    fig, axs = plt.subplots(1, len(names), figsize=(9 * len(names), 6), squeeze=False)
    for j, fn in enumerate(names):
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
