"""SCR 15분봉 진입 시점 변형 차트: 최근 30일 매매(가격 + Stoch·CCI·RSI)와 최근 1년·2020~ 자산 곡선.

롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 scr15_timing_chart.py --out <폴더> 변형...
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scr15_timing as T
from matplotlib.lines import Line2D

G, R = "#1a9850", "#d73027"
LEV = 5
DESC = {"B0": "enter next open", "A2": "confirm close beyond signal bar within 2",
        "A4": "confirm close beyond signal bar within 4", "A8": "confirm close beyond signal bar within 8",
        "P4": "Stoch 50 pullback re-cross within 4", "P8": "Stoch 50 pullback re-cross within 8",
        "P16": "Stoch 50 pullback re-cross within 16", "S1": "1% stop", "C8": "8-bar swing stop (0.2-2%)",
        "C16": "16-bar swing stop (0.2-2%)"}  # fmt: skip


def dt(b, i):
    return datetime.fromtimestamp(int(b["t"][i]) / 1000, T.KST).replace(tzinfo=None)


def trade_chart(b, F, name, a, z, path: Path) -> None:
    v = next(x for x in T.VARIANTS if T.vname(x) == name)
    tr, cnt = T.sim(b, F, v, a, z)
    s = T.summ(tr, cnt)
    X = mdates.date2num([dt(b, i) for i in range(a, z)])
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, b["c"][a:z], color="#555", lw=0.8)
    for x in tr:
        long = x["d"] == 1
        xe = mdates.date2num(dt(b, x["e"]))
        ax[0].scatter(xe, x["ep"], marker="o" if long else "v", s=70, color=G if long else R, zorder=5,
                      edgecolor="white")  # fmt: skip
        if x["why"] != "end":
            xx = mdates.date2num(dt(b, min(x["x"], z - 1)))
            ax[0].scatter(xx, x["xp"], marker="o" if long else "^", s=70, color=R if long else G, zorder=5,
                          edgecolor="white")  # fmt: skip
            ax[0].annotate(f"{LEV * x['r'] * 100:+.1f}%", (xx, x["xp"]), fontsize=7, xytext=(3, 6),
                           textcoords="offset points", color=G if x["r"] > 0 else R)  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    e, st = v[0], v[1]
    ax[0].set_title(
        f"SCR 3-level 15m, {name}: {DESC[e]}, {DESC[st]} - BTCUSDT perp, last 30 days "
        f"({dt(b, a):%m-%d} ~ {dt(b, z - 1):%m-%d %H:%M} KST)\n"
        f"{s['n']} trades, win {s['wins']}/{s['n']}, 5x {s['x5']['ret']:+.1%} (max drawdown {s['x5']['mdd']:.0%}), "
        f"1x {s['x1']['ret']:+.1%}; fees 0.05%/side + funding, exit on opposite signal. Labels = 5x net per trade",
        fontsize=10, loc="left")  # fmt: skip
    for k, (y, nm, lo, hi) in enumerate(((F["k"], "Stoch 5", 20, 80), (F["cci"], "CCI 20", -100, 100),
                                          (F["rsi"], "RSI 14", 30, 70)), 1):  # fmt: skip
        ax[k].plot(X, y[a:z], lw=0.6, color="#2c7bb6")
        for lv in (lo, hi):
            ax[k].axhline(lv, ls="--", lw=0.6, color="#888")
        ax[k].set_ylabel(nm)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for z_ in ax:
        z_.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def equity_chart(b, F, names, W, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#555555", "#2c7bb6", "#e08214", "#7b3294"]
    for k, (w, title) in enumerate((("year", "last 1 year (2025-11 ~, already seen)"),
                                    ("all", "2020-03 ~ now (fresh + 2022-25 + last year)"))):  # fmt: skip
        a, z = (W["fresh"][0], W["year"][1]) if w == "all" else W[w]
        for n, col in zip(names, colors):
            v = next(x for x in T.VARIANTS if T.vname(x) == n)
            tr, _ = T.sim(b, F, v, a, z)
            eq, xs = [1.0], [dt(b, a)]
            for x in tr:
                eq.append(eq[-1] * max(1 + LEV * x["r"], 0))
                xs.append(dt(b, min(x["x"], z - 1)))
            ax[k].plot(xs, eq, color=col, lw=1.2, label=f"{n} ({eq[-1] - 1:+.0%})")
        ax[k].set_yscale("log")
        ax[k].set_title(f"5x equity, {title}", fontsize=10, loc="left")
        ax[k].legend(fontsize=8)
        ax[k].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("names", nargs="+")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b = T.load()
    F = T.features(b)
    W = T.windows(b["t"])
    for n in a_.names:
        p = out / f"chart_{n.replace('+', '_')}_d30.png"
        trade_chart(b, F, n, *W["d30"], p)
        print(p)
    equity_chart(b, F, a_.names, W, out / "equity.png")
    print(out / "equity.png", np.round(0, 0))


if __name__ == "__main__":
    main()
