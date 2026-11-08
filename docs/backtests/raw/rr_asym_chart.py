"""손익비 비대칭 차트: 최근 30일 매매(가격 + Stoch·CCI·RSI), 자산 곡선(고른 안), MFE/MAE 진단 막대."""

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
from btc_fractal import cci, rsi, stoch
from matplotlib.lines import Line2D
from ql_upgrade_chart import G, R, kdt
from r5_data import load


def trades_chart(b, tr, title, path):
    t = b["t"]
    a = int(np.searchsorted(t, t[-1] - 30 * U.DAY))
    X = md.date2num([kdt(v) for v in t[a:]])
    k, cc, rr = stoch(b["h"], b["l"], b["c"], 5), cci(b["h"], b["l"], b["c"], 20), rsi(b["c"], 14)
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, b["c"][a:], color="#555", lw=0.8)
    for x in tr:
        long = x["d"] == 1
        ax[0].scatter(md.date2num(kdt(x["t_in"])), x["ep"], marker="o" if long else "v", s=60,
                      color=G if long else R, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].scatter(md.date2num(kdt(x["t_out"])), x["xp"], marker="o" if long else "^", s=60,
                      color=R if long else G, zorder=5, edgecolor="white")  # fmt: skip
        if x["why"] != "stop":
            ax[0].annotate(f"{5 * x['r'] * 100:+.1f}%", (md.date2num(kdt(x["t_out"])), x["xp"]), fontsize=7,
                           color=G if x["r"] > 0 else R, xytext=(3, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(title, loc="left", fontsize=10)
    panels = ((k, "Stoch 5", 20, 80), (cc, "CCI 20", -100, 100), (rr, "RSI 14", 30, 70))
    for i_, (y, nm, lo, hi) in enumerate(panels, 1):
        ax[i_].plot(X, y[a:], lw=0.6, color="#2c7bb6")
        ax[i_].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[i_].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[i_].set_ylabel(nm)
    ax[3].xaxis.set_major_formatter(md.DateFormatter("%m-%d"))
    for z in ax:
        z.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def equity(tr, frac):
    eq, xs, ys = 1.0, [], []
    for x in tr:
        if 5 * x["worst"] <= -0.95:
            eq = 0.0
        else:
            eq *= 1 + frac * max(5 * x["r"], -1.0)
        xs.append(kdt(x["t_out"]))
        ys.append(max(eq, 1e-6))
    return xs, ys


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b = load(DW.R5)
    res = json.loads((out / "result.json").read_text())
    tj = json.loads((out / "trades.json").read_text())["final"]
    fin = res["final"]
    r30 = res["res"][fin]["d30"]
    trades_chart(b, tj["d30"], f"{fin}: short stop / long take-profit, 1m fills, fees+funding - last 30 days: "
                 f"{r30['n']} trades, win {100 * (r30['win'] or 0):.0f}%, 5x/100% {r30['x5_f1.0']['ret'] * 100:+.1f}%. "
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
    keys = ("0.01|0.003", "0.02|0.005", "0.03|0.005")
    wdt = 0.8 / (2 * len(keys))
    for j, kk in enumerate(keys):
        sig = [D[n]["sig"][kk] * 100 for n in names]
        rnd = [D[n]["rnd"][kk] * 100 for n in names]
        xx = np.arange(len(names)) + (2 * j) * wdt - 0.4
        x_, y_ = kk.split("|")
        lab = f"+{float(x_) * 100:g}% before -{float(y_) * 100:g}%"
        ax[1].bar(xx, sig, wdt, color=plt.cm.tab10(j), label=f"signal {lab}")
        ax[1].bar(xx + wdt, rnd, wdt, color=plt.cm.tab10(j), alpha=0.35, label=f"random {lab}")
        rw = float(y_) / (float(x_) + float(y_)) * 100
        ax[1].axhline(rw, color=plt.cm.tab10(j), lw=0.6, ls=":")
    ax[1].set_xticks(range(len(names)), names, rotation=30, fontsize=8)
    ax[1].set_title("P(+X before -Y within 48h), 2020-03~2025-10; dotted = random-walk Y/(X+Y)",
                    loc="left", fontsize=10)  # fmt: skip
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close()
    print("ok")


if __name__ == "__main__":
    main()
