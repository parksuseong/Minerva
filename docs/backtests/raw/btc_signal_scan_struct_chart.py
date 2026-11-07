"""126번 칸 하나의 확인 구간 최근 거래 차트: 지그재그(확정 피벗), 손절·목표 선, 진입·청산, 그 봉의 Stoch·CCI·RSI.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <result.json> <칸 키 a/b/c/d/e> <일수> <png>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import btc_signal_scan as SS
import btc_swing_patterns as S
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from btc_fractal import cci, rsi, stoch

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"


def main() -> None:
    res, key, days, out = Path(sys.argv[1]), sys.argv[2].split("/"), int(sys.argv[3]), Path(sys.argv[4])
    row = next(r for r in json.loads(res.read_text())["rows"] if r["key"] == key)
    d5 = SS.load(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    b = SS.agg(d5, 3, 0)
    f = SS.agg(d5, SS.FRAMES[key[1]], 0)
    t15 = b["t"]
    lo_t = t15[-1] - days * 86_400_000
    lo = int(np.searchsorted(t15, lo_t))
    x15 = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t15])
    fl = int(np.searchsorted(f["t"], lo_t))
    xf = mdates.date2num([datetime.fromtimestamp((v + f["span"]) / 1000, KST) for v in f["t"]])
    th = float(key[2]) if key[2] != "-" else 0.02
    piv = [p for p in S.zigzag(f["h"], f["l"], th) if p[0] >= fl]
    tr = [z for z in row["trades_hold"] if z[0] >= lo]
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x15[lo:], b["c"][lo:], color=INK, lw=0.6)
    if piv:
        p.plot([xf[q[0]] - (xf[1] - xf[0]) for q in piv], [q[1] for q in piv], color="#4575b4", lw=1.0,
               label=f"zigzag {th:g} ({key[1]}, confirmed pivots)")
        for q in piv:
            p.annotate("H" if q[2] > 0 else "L", (xf[q[0]] - (xf[1] - xf[0]), q[1]), fontsize=7, color="#4575b4",
                       xytext=(0, 5 if q[2] > 0 else -10), textcoords="offset points", ha="center")
    wins = 0
    for e, side, g, x, sd, px, stop, tgt in tr:
        net = g - 0.001
        wins += net > 0
        mk = "o" if side > 0 else "^"
        xp = px * (1 + g * side)
        p.scatter(x15[e], px, marker=mk, s=80, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(x15[min(x, len(x15) - 1)], xp, marker=mk, s=80, zorder=5, edgecolor="white",
                  color=SELL if side > 0 else BUY)
        xe = x15[min(x, len(x15) - 1)]
        p.plot([x15[e], xe], [stop, stop], color=SELL, lw=0.8, ls="--")
        T = tgt if (tgt is not None and key[4] == "Y1") else px + (px - stop) * (1 if key[4] == "Y2" else 2)
        p.plot([x15[e], xe], [T, T], color=BUY, lw=0.8, ls="--")
        p.annotate(f"{net / sd:+.2f}R", (xe, xp), xytext=(4, 6), textcoords="offset points", fontsize=8,
                   fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], color=SELL, ls="--", label="stop"),
        plt.Line2D([], [], color=BUY, ls="--", label="target"),
        plt.Line2D([], [], color="#4575b4", label="zigzag"),
    ]
    p.legend(handles=handles, loc="upper left", ncol=7, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures, last {days} days: test 126 {'/'.join(key)}  |  {wins}/{len(tr)} wins, "
                f"net of 0.1% round trip + funding (KST)", loc="left", color=INK, fontsize=11)
    k, cc, r = stoch(f["h"], f["l"], f["c"]), cci(f["h"], f["l"], f["c"]), rsi(f["c"])
    for a_, (name, v, lines) in zip(ax[1:], ((f"Stoch ({key[1]})", k, (20, 80)), (f"CCI ({key[1]})", cc, (-100, 100)),
                                             (f"RSI 14 ({key[1]})", r, (30, 70)))):
        a_.plot(xf[fl:], v[fl:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd_ in ("top", "right"):
            a_.spines[sd_].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(key, len(tr), wins)


if __name__ == "__main__":
    main()
