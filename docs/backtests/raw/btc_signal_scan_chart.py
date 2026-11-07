"""125번 칸의 최근 거래 차트. 가격(15분봉) + 그 봉의 확정 지그재그 + 거래, Stoch·CCI·RSI 패널.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <result.json> <칸 번호> <일수> <png>
칸 번호는 result.json rows의 순서다.
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

import btc_signal_scan as X
import btc_swing_patterns as S
from btc_fractal import cci, rsi, stoch

KST = timezone(timedelta(hours=9))
BUY, SELL, INK, GRID = "#1a9850", "#d73027", "#333333", "#e6e6e6"


def main() -> None:
    res, k, days, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    row = json.loads(res.read_text())["rows"][k]
    name, frame, size, gate, xn = row["key"]
    d5 = X.load(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    b = X.agg(d5, 3, 0)
    t = b["t"]
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    tr = [z for z in row["trades_hold"] if z[0] >= lo]
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    f = X.agg(d5, X.FRAMES[frame], 0)
    fx = mdates.date2num([datetime.fromtimestamp((v + f["span"]) / 1000, KST) for v in f["t"]])
    th = float(size) if size != "-" else 0.02
    piv = S.zigzag(f["h"], f["l"], th)
    flo = int(np.searchsorted(fx, x[lo]))
    pv = [p for p in piv if p[0] >= flo and p[3] < len(fx)]
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], b["c"][lo:], color=INK, lw=0.7)
    if pv:
        p.plot([fx[q[0]] for q in pv], [q[1] for q in pv], color="#4575b4", lw=1.0, alpha=0.8,
               label=f"{frame} zigzag {th:g} (confirmed later)")
    wins = 0
    for e, side, g, q in tr:
        px_in = b["o"][e]
        net = g - 2 * X.COST
        wins += net > 0
        px_out = px_in * (1 + g * side)
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=90, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        qq = min(q, len(x) - 1)
        p.scatter(x[qq], px_out, marker=mk, s=90, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([x[e], x[qq]], [px_in, px_out], color="#777777", lw=1.0, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[qq], px_out), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
               plt.Line2D([], [], color="#4575b4", label=f"{frame} zigzag {th:g}")]
    p.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9)
    h = row["hold"]
    p.set_title(f"BTCUSDT futures, last {days} days: test 125 cell {name}/{frame}/{size}/{gate}/{xn}  |  "
                f"{wins}/{len(tr)} wins here; holdout 2024~ n={h['n']} win {h['win']}% ev {h['ev']:+.3f}%/trade (KST)",
                loc="left", color=INK, fontsize=10)
    fl = int(np.searchsorted(fx, x[lo]))
    for a_, (nm, v, lines) in zip(ax[1:], ((f"Stoch 5 ({frame})", stoch(f["h"], f["l"], f["c"]), (20, 80)),
                                           (f"CCI 20 ({frame})", cci(f["h"], f["l"], f["c"]), (-100, 100)),
                                           (f"RSI 14 ({frame})", rsi(f["c"]), (30, 70)))):
        a_.plot(fx[fl:], v[fl:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(nm, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(name, frame, size, gate, xn, len(tr), wins)


if __name__ == "__main__":
    main()
