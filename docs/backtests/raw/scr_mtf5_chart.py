"""SCR 다섯 프레임 일치 시험 차트: 최근 30일 매매(+ 다섯 프레임 방향 띠)와 자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 scr_mtf5_chart.py --out <결과 폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scr_mtf5 as X
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

KST = timezone(timedelta(hours=9))
G, R = "#1a9850", "#d73027"


def kdt(ms):
    return datetime.fromtimestamp(ms / 1000, KST).replace(tzinfo=None)


def chart_d30(out: Path, res: dict, tr: list) -> None:
    parts = sorted(X.DIR1M.glob("*.npz"))[-3:]
    p = [np.load(f) for f in parts]
    m = {k: np.concatenate([q[k] for q in p]) for k in ("t", "o", "h", "l", "c")}
    F = {n: X.frame(m, X.FRAMES[n]) for n in X.FIVE}
    end = int(m["t"][-1]) + X.MIN
    a = end - 30 * X.DAY
    f15 = F["15m"]
    s15 = f15["te"] >= a
    xs = mdates.date2num([kdt(v) for v in f15["te"][s15]])
    fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True,
                           gridspec_kw={"height_ratios": [4, 1.1, 1, 1, 1]})  # fmt: skip
    ax[0].plot(xs, f15["c"][s15], color="#555", lw=0.8)
    for x in tr:
        long = x["d"] == 1
        ax[0].scatter(mdates.date2num(kdt(x["t_in"])), x["ep"], marker="o" if long else "v", s=70,
                      color=G if long else R, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].scatter(mdates.date2num(kdt(x["t_out"])), x["xp"], marker="o" if long else "^", s=70,
                      color=R if long else G, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].annotate(f"{X.LEV * x['r'] * 100:+.1f}%", (mdates.date2num(kdt(x["t_out"])), x["xp"]),
                       fontsize=7, color=G if x["r"] > 0 else R, xytext=(3, 6),
                       textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    fin = res["final"]
    s = res["grid"].get(fin) or res["extras"].get(fin)
    d30 = s["d30"]
    ax[0].set_title(
        f"SCR 5-frame alignment {fin} - BTCUSDT perp, last 30 days ({kdt(a):%m-%d} ~ {kdt(end):%m-%d %H:%M} KST)\n"
        f"{d30['n']} trades (L {d30['nl']} / S {d30['ns']}), 5x 100% margin {d30['x5_f1.0']['ret'] * 100:+.1f}% "
        f"(max DD {d30['x5_f1.0']['mdd'] * 100:.0f}%), 40% margin {d30['x5_f0.4']['ret'] * 100:+.1f}%; "
        "fees + funding, 1m fills. Labels = 5x net per trade",
        fontsize=10, loc="left")  # fmt: skip
    # 다섯 프레임 방향 띠: 15분 마감마다 각 프레임의 마감된 마지막 봉 방향
    tc = f15["te"][s15]
    D = np.stack([F[n]["dir"][np.maximum(X.at(F[n], tc), 0)] for n in X.FIVE])
    cmap = ListedColormap([R, "#bbbbbb", G])
    ax[1].imshow(D + 1, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                 extent=[xs[0], xs[-1], len(X.FIVE) - 0.5, -0.5])  # fmt: skip
    ax[1].set_yticks(range(len(X.FIVE)))
    ax[1].set_yticklabels(X.FIVE, fontsize=8)
    ax[1].set_ylabel("direction")
    for i, (key, nm, lo, hi) in enumerate((("k", "Stoch 5", 20, 80), ("cci", "CCI 20", -100, 100),
                                           ("rsi", "RSI 14", 30, 70)), 2):  # fmt: skip
        ax[i].plot(xs, f15[key][s15], lw=0.6, color="#2c7bb6")
        ax[i].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[i].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[i].set_ylabel(f"{nm} (15m)")
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for z in ax:
        z.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "chart_d30.png", dpi=100)
    plt.close(fig)


def chart_equity(out: Path, res: dict, pool: list) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    eq, xs, ys = 1.0, [], []
    for x in pool:
        eq *= 1 + 0.4 * max(X.LEV * x["r"], -1.0)
        xs.append(kdt(x["t_out"]))
        ys.append(eq)
    if xs:
        ax[0].plot(xs, ys, color="#2c7bb6", lw=1.4)
    ax[0].set_yscale("log")
    ax[0].axhline(1, color="#999", lw=0.6)
    ax[0].set_title(f"{res['final']} equity 2020-03~2025-10, 5x, 40% margin (log)", loc="left", fontsize=10)
    wins = ("fresh", "design", "val", "year", "m6", "d30")
    lab = ["2020-21\n(untouched)", "2022-24.6\n(design)", "2024.7-25.10", "last year", "6 months", "30 days"]
    fin = res["final"]
    s = res["grid"].get(fin) or res["extras"].get(fin)
    p2 = res.get("p2") or {}
    xx = np.arange(len(wins))
    v1 = [s[w]["x5_f1.0"]["ret"] * 100 for w in wins]
    v4 = [s[w]["x5_f0.4"]["ret"] * 100 for w in wins]
    vp = [p2.get(w, {}).get("x5_f1.0", {}).get("ret", np.nan) * 100 for w in wins]
    ax[1].bar(xx - 0.27, v4, 0.27, color="#74add1", label=f"{fin} 40% margin")
    ax[1].bar(xx, v1, 0.27, color="#2c7bb6", label=f"{fin} 100% margin")
    ax[1].bar(xx + 0.27, vp, 0.27, color="#bbbbbb", label="P2 (NY breakout) 100%")
    ax[1].axhline(0, color="#555", lw=0.6)
    ax[1].set_yscale("symlog", linthresh=50)
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(lab, fontsize=8)
    ax[1].set_ylabel("5x return %")
    ax[1].legend(fontsize=8)
    for z in ax:
        z.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).expanduser()
    res = json.loads((out / "result.json").read_text())
    trs = json.loads((out / "trades.json").read_text())
    fin = res["final"]
    chart_d30(out, res, trs[fin]["d30"])
    chart_equity(out, res, trs["_pool_final"])
    print("charts", out)


if __name__ == "__main__":
    main()
