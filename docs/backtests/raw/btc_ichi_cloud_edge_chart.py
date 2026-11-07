"""138번 칸 하나의 최근 N일 매매 차트(구름 포함).

실행: PYTHONPATH=<numpy·matplotlib>:src:docs/backtests/raw python <이 파일> <trades.json> "<칸 키>" <일수> <출력 png>
"""

from __future__ import annotations

import ast
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
from btc_ichi_cloud_edge import COST, FRAMES
from btc_scr_macd_ichi import mid, shift
from btc_signal_scan import agg, load

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"


def main() -> None:
    src, key_s, days, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    key = ast.literal_eval(key_s)
    fn, (n1, n2, n3) = key[0], key[1]
    tr = json.loads(src.read_text())[key_s]
    d5 = load(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    f = agg(d5, FRAMES[fn], 0)
    h, l, c, t = f["h"], f["l"], f["c"], f["t"]  # noqa: E741
    ten, kij = mid(h, l, n1), mid(h, l, n2)
    sa, sb = shift((ten + kij) / 2, n2), shift(mid(h, l, n3), n2)
    s, cc, r = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    t_end = t[-1]
    lo = int(np.searchsorted(t, t_end - days * 86_400_000))
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    x5 = lambda ms: mdates.date2num(datetime.fromtimestamp(ms / 1000, KST))  # noqa: E731
    fig, ax = plt.subplots(
        4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]}
    )
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.8)
    p.fill_between(x[lo:], sa[lo:], sb[lo:], where=sa[lo:] >= sb[lo:], color=BUY, alpha=0.15, lw=0)
    p.fill_between(x[lo:], sa[lo:], sb[lo:], where=sa[lo:] < sb[lo:], color=SELL, alpha=0.15, lw=0)
    p.plot(x[lo:], sa[lo:], color=BUY, lw=0.5)
    p.plot(x[lo:], sb[lo:], color=SELL, lw=0.5)
    wins = n = 0
    for e5, side, g, xe, _d, te, _why in tr:
        if te < t[lo]:
            continue
        px_in = d5["o"][e5]
        px_out = px_in * (1 + g * side)
        net = g - 2 * COST
        n += 1
        wins += net > 0
        mk = "o" if side > 0 else "^"
        xi, xo = x5(d5["t"][e5]), x5(d5["t"][xe])
        p.scatter(
            xi,
            px_in,
            marker=mk,
            s=100,
            zorder=5,
            edgecolor="white",
            color=BUY if side > 0 else SELL,
        )
        p.scatter(
            xo,
            px_out,
            marker=mk,
            s=100,
            zorder=5,
            edgecolor="white",
            color=SELL if side > 0 else BUY,
        )
        p.plot([xi, xo], [px_in, px_out], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(
            f"{net * 100:+.2f}%",
            (xo, px_out),
            xytext=(4, 6) if net > 0 else (-8, -16),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=BUY if net > 0 else SELL,
        )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], color=BUY, lw=4, alpha=0.3, label="cloud (span A ≥ B)"),
        plt.Line2D([], [], color=SELL, lw=4, alpha=0.3, label="cloud (span A < B)"),
    ]
    p.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9)
    p.set_title(
        f"BTCUSDT futures {fn}, last {days} days: test 138 {key[2]} {key[3]} {key[4]}, Ichimoku "
        f"{(n1, n2, n3)}  |  {wins}/{n} wins, net of 0.1% + funding (KST)",
        loc="left",
        color=INK,
        fontsize=11,
    )
    for a_, (name, v, lines) in zip(
        ax[1:], (("Stoch 5", s, (20, 80)), ("CCI 20", cc, (-100, 100)), ("RSI 14", r, (30, 70)))
    ):
        a_.plot(x[lo:], v[lo:], color=INK, lw=0.6)
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
    print(key_s, n, wins)


if __name__ == "__main__":
    main()
