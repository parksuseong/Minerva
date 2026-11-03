"""손절 폭 시험의 최근 30일 거래 차트 (문 없음, 손절 3%). 가격과 Stoch·CCI·RSI 패널.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <출력 png>
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from btc_fractal import load5, resample, scr3
from btc_scr_stopwidth import COST, trade

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
STOP = 0.03
DAYS = 30


def main() -> None:
    out = Path(sys.argv[1])
    d = resample(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), 6)
    n = len(d["c"])
    on_l, on_s, s, cc, r = scr3(d)
    end = d["t"][-1] + 1_800_000
    lo = int(np.searchsorted(d["t"], end - DAYS * 86_400_000))
    view = max(0, lo - 48)
    sigs = sorted([(int(i), sd) for sd, on in ((1, on_l), (-1, on_s)) for i in np.flatnonzero(on)
                   if i >= lo and i < n - 2])
    busy, tr = -1, []
    for i, side in sigs:
        if i <= busy:
            continue
        res = trade(d, i, side, on_s if side > 0 else on_l, STOP)
        if res is None:
            tr.append((i + 1, None, side, None))
            break
        g, x, _ = res
        tr.append((i + 1, x, side, g))
        busy = x
    x = mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in d["t"]])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                           gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[view:], d["c"][view:], color=INK, lw=0.8)
    wins = done = 0
    for e, q, side, g in tr:
        px_in = d["o"][e]
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=110, zorder=5, edgecolor="white",
                  color=BUY if side > 0 else SELL)
        if q is None:
            p.annotate("open", (x[e], px_in), xytext=(6, 10), textcoords="offset points",
                       fontsize=10, color=INK)
            continue
        net = g - 2 * COST
        done += 1
        wins += net > 0.001
        px_out = px_in * (1 + g * side)
        p.plot([x[e], x[q]], [px_in, px_out], color="#777777", lw=1.2,
               ls="-" if net > 0.001 else ":")
        p.scatter(x[q], px_out, marker=mk, s=110, zorder=5, edgecolor="white",
                  color=SELL if side > 0 else BUY)
        p.annotate(f"{net * 100:+.2f}%", (x[q], px_out), xytext=(6, 8) if net > 0.001 else (-10, -22), textcoords="offset points",
                   fontsize=11, fontweight="bold", color=BUY if net > 0.001 else SELL)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], color="#777777", ls="-", label="win (net > +0.1%)"),
        plt.Line2D([], [], color="#777777", ls=":", label="loss"),
    ]
    p.legend(handles=handles, loc="upper left", ncol=6, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT 30m, last {DAYS} days: SCR 3-level entry, exit on opposite SCR signal, "
                f"stop {STOP * 100:g}% (same trades with 5%, 7%, 10%, no stop)  |  "
                f"{wins}/{done} wins, net of 0.1% round-trip cost (KST)", loc="left", color=INK,
                fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", s, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 14", r, (30, 70)))):
        a_.plot(x[view:], v[view:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
        for e, q, *_ in tr:
            a_.axvline(x[e], color="#cccccc", lw=0.6)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    for e, q, side, g in tr:
        print(datetime.fromtimestamp(d["t"][e] / 1000, KST).strftime("%m-%d %H:%M"),
              "L" if side > 0 else "S", d["o"][e],
              None if q is None else datetime.fromtimestamp(d["t"][q] / 1000, KST).strftime("%m-%d %H:%M"),
              None if g is None else round((g - 2 * COST) * 100, 2))


if __name__ == "__main__":
    main()
