"""원래 SCR 3단(Stoch5 20/80, CCI20 ±100, RSI14 30/70, 3봉 안 돌파) 최근 N일 매매 차트.

진입: 신호 봉 다음 시가 시장가. 청산: 1% 손절 또는 반대 SCR 3단 신호 다음 시가. 롱·숏 거울, 포지션 하나.
실행: PYTHONPATH=<numpy·matplotlib>:src:docs/backtests/raw python <이 파일> <봉(5분봉 개수)> <일수> <출력 png>
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
from btc_scr_stopwidth import trade

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
COST = 0.0005
STOP = 0.01


def main() -> None:
    k, days, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    d = resample(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), k)
    n = len(d["c"])
    on_l, on_s, s, cc, r = scr3(d)
    bar = k * 300_000
    lo = int(np.searchsorted(d["t"], d["t"][-1] + bar - days * 86_400_000))
    view = max(0, lo - 8)
    sigs = sorted((int(i), sd) for sd, on in ((1, on_l), (-1, on_s)) for i in np.flatnonzero(on) if lo <= i < n - 2)
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
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[view:], d["c"][view:], color=INK, lw=0.8)
    wins = done = 0
    total = 1.0
    rows = []
    for e, q, side, g in tr:
        px_in = d["o"][e]
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=110, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        if q is None:
            now = (d["c"][-1] / px_in - 1) * side - 2 * COST
            p.annotate(f"open {now * 100:+.2f}%", (x[e], px_in), xytext=(6, 10), textcoords="offset points",
                       fontsize=10, color=INK)
            rows.append((e, None, side, now))
            continue
        net = g - 2 * COST
        done += 1
        wins += net > 0
        total *= 1 + net
        px_out = px_in * (1 + g * side)
        p.plot([x[e], x[q]], [px_in, px_out], color="#777777", lw=1.2, ls="-" if net > 0 else ":")
        p.scatter(x[q], px_out, marker=mk, s=110, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.annotate(f"{net * 100:+.2f}%", (x[q], px_out), xytext=(6, 8) if net > 0 else (-10, -20),
                   textcoords="offset points", fontsize=10, fontweight="bold", color=BUY if net > 0 else SELL)
        rows.append((e, q, side, net))
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
    ]
    p.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    frame = {3: "15m", 6: "30m", 12: "1h"}.get(k, f"{k * 5}m")
    p.set_title(f"BTCUSDT futures {frame}, last {days} days: original SCR 3-level (Stoch5 20/80, CCI20 ±100, "
                f"RSI14 30/70), exit opposite signal or 1% stop  |  {wins}/{done} closed wins, "
                f"total {(total - 1) * 100:+.2f}% (1x, fees 0.05%/side) (KST)", loc="left", color=INK, fontsize=11)
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
        for e, *_ in tr:
            a_.axvline(x[e], color="#cccccc", lw=0.6)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    for e, q, side, net in rows:
        t_in = datetime.fromtimestamp(d["t"][e] / 1000, KST).strftime("%m-%d %H:%M")
        t_out = "open" if q is None else datetime.fromtimestamp(d["t"][q] / 1000, KST).strftime("%m-%d %H:%M")
        print(t_in, "L" if side > 0 else "S", round(d["o"][e]), t_out, round(net * 100, 2))
    print("closed", done, "wins", wins, "total", round((total - 1) * 100, 2))


if __name__ == "__main__":
    main()
