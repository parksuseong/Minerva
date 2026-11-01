"""등록 보고용 차트. 가격·이평과 Stoch·CCI·RSI 패널, 마지막 3개월 거래를 그린다.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <출력 폴더>
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

import btc_scr_ma as B
from minerva.strategy.indicators import cci, rsi, stoch_fast_k

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
MA_COLORS = ("#1f6fb4", "#e08a00")
TOPS = [
    ("R", ("X", "S", 20, 60), "M", "RSI 30/70 + SMA20 vs SMA60, exit on SMA60 cross"),
    ("R", ("X", "S", 20, 50), "M", "RSI 30/70 + SMA20 vs SMA50, exit on SMA50 cross"),
]


def draw(d, sig, m, flt, ex, title, out, bars_all):
    n = len(d["c"])
    a = n - B.YEAR
    tr = B.run(d, sig, B.masks(flt, d["c"], m), ex, a, n)
    lo = n - B.YEAR // 4
    k = tr[0] >= lo
    ts = [datetime.fromtimestamp(t / 1000, KST) for t in d["t"]]
    x = mdates.date2num(ts)
    fig, ax = plt.subplots(
        4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]}
    )
    p = ax[0]
    p.plot(x[lo:], d["c"][lo:], color=INK, lw=0.8, label="close")
    for col, per in zip(MA_COLORS, flt[2:]):
        p.plot(x[lo:], m[flt[1], per][lo:], color=col, lw=1.6, label=f"SMA {per}")
    for e, q, s, g in zip(tr[0][k], tr[1][k], tr[2][k], tr[3][k]):
        win = g - 2 * B.COSTS[0] > B.WIN
        px_in, px_out = d["o"][e], d["o"][e] * (1 + float(g) * int(s))
        p.plot([x[e], x[q]], [px_in, px_out], color="#777777", lw=1, ls="-" if win else ":")
        mk = "o" if s > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=70, zorder=5, edgecolor="white",
                  color=BUY if s > 0 else SELL)
        p.scatter(x[q], px_out, marker=mk, s=70, zorder=5, edgecolor="white",
                  color=SELL if s > 0 else BUY)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], color="#777777", ls="-", label="win (net > +0.1%)"),
        plt.Line2D([], [], color="#777777", ls=":", label="loss"),
    ]
    p.legend(handles=p.get_legend_handles_labels()[0] + handles, loc="upper left", ncol=5,
             frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT 30m, last 3 months, {title} (KST)", loc="left", color=INK)
    series = (
        ("Stoch 5", stoch_fast_k(bars_all), (20, 80)),
        ("CCI 20", cci(bars_all, 20), (-100, 100)),
        ("RSI 14", rsi(d["c"].tolist()), (30, 70)),
    )
    for a_, (name, v, lines) in zip(ax[1:], series):
        y = np.array([np.nan if w is None else w for w in v], dtype=float)
        a_.plot(x[lo:], y[lo:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    d = B.load(Path("~/.minerva/data/btc/30m.csv").expanduser())
    sigs = B.raw_signals(d)
    m = B.all_mas(d["c"])
    bars = [
        B.Bar(datetime.fromtimestamp(t / 1000, timezone.utc).date(), o, h, l, c, 0.0)
        for t, o, h, l, c in zip(d["t"], d["o"], d["h"], d["l"], d["c"])
    ]
    for i, (sn, flt, ex, title) in enumerate(TOPS, 1):
        draw(d, sigs[sn], m, flt, ex, title, out / f"top{i}.png", bars)


if __name__ == "__main__":
    main()
