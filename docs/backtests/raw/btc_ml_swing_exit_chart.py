"""120번 거래 차트: 최근 N일, 현물 15분봉 + 확정 스윙 라벨 + 손절·익절 선 + Stoch·CCI·RSI 패널.

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
from btc_fut_rsi_swing import bd_signals

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
PRED = Path("~/.minerva/research/btc-2026-11-04/ml15/pred.npz").expanduser()


def main() -> None:
    src, var, days, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    z = np.load(PRED)
    t, h, l, c = z["t"], z["h"], z["l"], z["c"]  # noqa: E741
    tr = json.loads(src.read_text())[var]
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    tr = [x for x in tr if x[0] >= t[lo]]
    k, cc, r7 = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 7)
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    idx = {float(v): i for i, v in enumerate(t)}
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.7)
    # 확정 스윙: 극점 위치에 라벨, 확정 봉까지 점선
    prev = {1: None, -1: None}
    for i, side, px, ext in bd_signals(h, l, r7):
        lab = ("HL" if prev[1] is not None and px > prev[1] else "LL") if side > 0 else \
              ("LH" if prev[-1] is not None and px < prev[-1] else "HH")
        prev[side] = px
        if ext < lo:
            continue
        col = BUY if side > 0 else SELL
        p.annotate(lab, (x[ext], px), xytext=(0, -14 if side > 0 else 8), textcoords="offset points",
                   fontsize=7, color=col, ha="center", fontweight="bold")
        p.plot([x[ext], x[i]], [px, px], color=col, lw=0.6, ls=":", alpha=0.7)
    wins = 0
    for te, g, side, tx, win, stop, tgt, kind in tr:
        e, q = idx[te], idx[tx]
        wins += g > 0
        px_in = c[e]
        px_out = px_in * (1 + g * side)
        if stop is not None:
            p.plot([x[e], x[q]], [stop, stop], color=SELL, lw=0.9, alpha=0.6)
        if tgt is not None:
            p.plot([x[e], x[q]], [tgt, tgt], color=BUY, lw=0.9, alpha=0.6)
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=80, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(x[q], px_out, marker=mk, s=80, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([x[e], x[q]], [px_in, px_out], color="#777777", lw=1.0, ls="-" if g > 0 else ":")
        p.annotate(f"{g * 100:+.2f}%{'' if kind == 'S' else ' ' + kind}", (x[q], px_out),
                   xytext=(4, 6) if g > 0 else (-8, -15), textcoords="offset points", fontsize=7,
                   fontweight="bold", color=BUY if g > 0 else SELL)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], color=SELL, alpha=0.6, label="stop (swing)"),
        plt.Line2D([], [], color=BUY, alpha=0.6, label="target (swing)"),
    ]
    p.legend(handles=handles, loc="upper left", ncol=6, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT spot 15m, last {days} days to 11-03: test 120 {var} (ML entry, RSI7 swing stop/target; "
                f"'E0' = fallback ±1.5%)  |  {wins}/{len(tr)} wins, 1x net of fees (KST)",
                loc="left", color=INK, fontsize=10)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 7 (swing)", r7, (30, 70)))):
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
