"""대기 진입·빠른 청산 시험의 최근 30일 거래 차트. 30분봉 가격과 Stoch·CCI·RSI 패널.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <result.json> <칸 이름> <출력 png>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from btc_fractal import load5, resample, scr3

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
COST = 0.0005
DAYS = 30
NAMES = {
    "T5m|P0|F0|SL3%": "wait for 5m SCR, exit on 30m opposite SCR, stop 3%",
    "T5m|P0|F1|SL3%": "wait for 5m SCR, fast exit on 5m opposite SCR when in profit, stop 3%",
}


def main() -> None:
    res, key, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    trades = json.loads(res.read_text())[key]["trades"]
    d = resample(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), 6)
    _, _, s, cc, r = scr3(d)
    end = d["t"][-1] + 1_800_000
    lo = int(np.searchsorted(d["t"], end - DAYS * 86_400_000))
    view = max(0, lo - 48)
    tr = [t for t in trades if t[0] >= d["t"][lo]]
    xt = lambda ms: mdates.date2num(datetime.fromtimestamp(ms / 1000, KST))  # noqa: E731
    x = np.array([xt(t) for t in d["t"]])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                           gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[view:], d["c"][view:], color=INK, lw=0.8)
    wins = 0
    for t_in, g, side, t_out, px in tr:
        net = g - 2 * COST
        wins += net > 0.001
        px_out = px * (1 + g * side)
        mk = "o" if side > 0 else "^"
        p.plot([xt(t_in), xt(t_out)], [px, px_out], color="#777777", lw=1.2,
               ls="-" if net > 0.001 else ":")
        p.scatter(xt(t_in), px, marker=mk, s=110, zorder=5, edgecolor="white",
                  color=BUY if side > 0 else SELL)
        p.scatter(xt(t_out), px_out, marker=mk, s=110, zorder=5, edgecolor="white",
                  color=SELL if side > 0 else BUY)
        p.annotate(f"{net * 100:+.2f}%", (xt(t_out), px_out),
                   xytext=(6, 8) if net > 0.001 else (-10, -22), textcoords="offset points",
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
    p.set_title(f"BTCUSDT 30m, last {DAYS} days: 30m SCR 3-level arms, {NAMES.get(key, key)}  |  "
                f"{wins}/{len(tr)} wins, net of 0.1% round-trip cost (KST)", loc="left",
                color=INK, fontsize=11)
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
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
