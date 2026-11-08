"""daily wave 차트: 최근 30일 매매(가격 + Stoch·CCI·RSI)와 구간별 5배 자산 곡선(테이커 vs 메이커).

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 daily_wave_chart.py --out <폴더> 진입|문
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from r5_data import load

G, R = "#1a9850", "#d73027"
FEE_LAB = {"FA": "taker in/out (market)", "FB": "maker limit entry + taker exit",
           "FC": "maker entry + maker TP"}  # fmt: skip


def dt(t, i):
    return datetime.fromtimestamp(int(t[i]) / 1000, DW.KST).replace(tzinfo=None)


def trades_chart(b, F, C, en, g, fee, a, z, path):
    t = b["t"]
    tr, cnt = DW.simulate(b, F, C[en], g, fee, a, z)
    s = DW.summ(tr, a, z, b, cnt)
    X = mdates.date2num([dt(t, i) for i in range(a, z)])
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                           gridspec_kw={"height_ratios": [4, 1, 1, 1]})  # fmt: skip
    ax[0].plot(X, b["c"][a:z], color="#555", lw=0.8)
    for x in tr:
        lg = x["d"] == 1
        xe, xx = mdates.date2num(dt(t, x["e"])), mdates.date2num(dt(t, x["x"]))
        ax[0].scatter(xe, x["ep"], marker="o" if lg else "v", s=60, color=G if lg else R,
                      zorder=5, edgecolor="white")  # fmt: skip
        ax[0].scatter(xx, x["xp"], marker="o" if lg else "^", s=60, color=R if lg else G,
                      zorder=5, edgecolor="white")  # fmt: skip
        ax[0].annotate(f"{DW.LEV * x['r'] * 100:+.1f}%", (xx, x["xp"]), fontsize=6,
                       color=G if x["r"] > 0 else R, xytext=(3, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(
        f"{en}|{g} - {FEE_LAB[fee]} - BTCUSDT perp 15m, last 30 days {dt(t, a):%m-%d} ~ {dt(t, z - 1):%m-%d %H:%M} KST\n"
        f"{s['n']} trades ({s['per_day']:.2f}/day), win {(s['win'] or 0):.0%}, 5x {s['x5']['ret']:+.1%} "
        f"(max drawdown {s['x5']['mdd']:.0%}), 1x {s['x1']['ret']:+.1%}. Labels = 5x net per trade (fees + funding)",
        loc="left", fontsize=10)  # fmt: skip
    for k, (y, nm, lo, hi) in enumerate(((F["stoch"], "Stoch 5", 20, 80), (F["cci"], "CCI 20", -100, 100),
                                         (F["rsi"], "RSI 14", 30, 70)), 1):  # fmt: skip
        ax[k].plot(X, y[a:z], lw=0.6, color="#2c7bb6")
        ax[k].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[k].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[k].set_ylabel(nm)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for q in ax:
        q.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def equity_chart(b, F, C, en, g, W, path):
    t = b["t"]
    wins = ("fresh", "design", "val", "year")
    lab = {"fresh": "2020-03~2021-10 (untouched)", "design": "2022-01~2024-06",
           "val": "2024-07~2025-10", "year": "2025-11~ (last year)"}  # fmt: skip
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    for k, w in enumerate(wins):
        a, z = W[w]
        for fee, col in (("FA", "#d73027"), ("FB", "#2c7bb6"), ("FC", "#1a9850")):
            tr, _ = DW.simulate(b, F, C[en], g, fee, a, z)
            eq, xs, ys = 1.0, [dt(t, a)], [0.0]
            for x in tr:
                eq *= max(1 + DW.LEV * x["r"], 0)
                xs.append(dt(t, x["x"]))
                ys.append((eq - 1) * 100)
            ax[k].step(xs, ys, where="post", color=col, lw=1.3, label=f"{FEE_LAB[fee]}: {ys[-1]:+.0f}%")
        ax[k].axhline(0, color="#999", lw=0.6)
        ax[k].set_title(lab[w], fontsize=10)
        ax[k].legend(fontsize=7, loc="upper left")
        ax[k].tick_params(axis="x", labelsize=7, rotation=30)
        ax[k].grid(alpha=0.2)
    ax[0].set_ylabel("5x return %")
    fig.suptitle(f"{en}|{g} - 5x equity by window, same signals, three fee regimes (taker 0.05%, maker 0.02%)",
                 fontsize=11)  # fmt: skip
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("items", nargs="+", help="진입|문 (예: S-NY|GT)")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    b = load(DW.R5)
    F = DW.features(b)
    C = DW.all_cands(b, F)
    W = DW.windows(b["t"])
    for item in args.items:
        en, g = item.split("|")
        tag = f"{en}_{g}"
        for fee in ("FA", "FB"):
            trades_chart(b, F, C, en, g, fee, *W["d30"], out / f"chart_{tag}_{fee}_d30.png")
        equity_chart(b, F, C, en, g, W, out / f"equity_{tag}.png")
        print("wrote", tag, flush=True)
    _ = json, np


if __name__ == "__main__":
    main()
