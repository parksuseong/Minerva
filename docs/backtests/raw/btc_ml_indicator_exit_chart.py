"""119번 거래 차트: 최근 N일, 현물 15분봉 + Stoch·CCI·RSI 패널.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <trades.json> <변형> <일수> <png>
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

from btc_fractal import cci, rsi, stoch

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
PRED = Path("~/.minerva/research/btc-2026-11-04/ml15/pred.npz").expanduser()


def main() -> None:
    src, var, days, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    z = np.load(PRED)
    t, o, h, l, c = z["t"], z["o"], z["h"], z["l"], z["c"]  # noqa: E741
    tr = json.loads(src.read_text())[var]
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    tr = [x for x in tr if x[0] >= t[lo]]
    k, cc, r = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    idx = {float(v): i for i, v in enumerate(t)}
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.7)
    wins = 0
    for te, g, side, tx, _ in tr:
        e, q = idx[te], idx[tx]
        net = g
        wins += net > 0
        px_in = c[e]
        px_out = px_in * (1 + g * side)
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=80, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(x[q], px_out, marker=mk, s=80, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([x[e], x[q]], [px_in, px_out], color="#777777", lw=1.0, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[q], px_out), xytext=(4, 6) if net > 0 else (-8, -15),
                   textcoords="offset points", fontsize=7, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
    ]
    p.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT spot 15m, last {days} days to 11-03: test 119 {var} (ML entry, indicator exit, 1% stop)  |  "
                f"{wins}/{len(tr)} wins, 1x net of fees (approx. prices, KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 14", r, (30, 70)))):
        a_.plot(x[lo:], v[lo:], color=INK, lw=0.5)
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
    print(var, len(tr), wins)


if __name__ == "__main__":
    main()
