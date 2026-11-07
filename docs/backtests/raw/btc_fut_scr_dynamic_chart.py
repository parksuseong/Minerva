"""115번 SCR Dynamic StochOnly 최근 거래 차트. 가격 + Stoch·CCI·RSI 패널.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <result.json> <변형> <일수> <출력 png>
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

from btc_fractal import cci, load5, rsi, stoch
from btc_fut_blog_combo import ema
from btc_scr_learned import bars

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
COST = 0.0005


def main() -> None:
    res, var, days, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    d = bars(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), 0)
    h, l, c, t = d["h"], d["l"], d["c"], d["t"]
    k, cc, r, e50 = stoch(h, l, c, 5), cci(h, l, c, 9), rsi(c, 14), ema(c, 50)
    tr = json.loads(res.read_text())[var]["trades"]
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    view = max(0, lo - 16)
    tr = [z for z in tr if z[0] >= lo]
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[view:], c[view:], color=INK, lw=0.8)
    p.plot(x[view:], e50[view:], color="#4575b4", lw=0.8, label="EMA 50")
    wins = 0
    for i, side, g, q, _ in tr:
        e = i + 1
        px_in = d["o"][e]
        net = g - 2 * COST
        wins += net > 0
        px_out = px_in * (1 + g * side)
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=90, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(x[q], px_out, marker=mk, s=90, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([x[e], x[q]], [px_in, px_out], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[q], px_out), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], color="#4575b4", label="EMA 50"),
    ]
    p.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9)
    ex = "opposite signal or 1% stop" if var.endswith("X1") else "+1% target / -1% stop / 96 bars"
    p.set_title(f"BTCUSDT futures 15m, last {days} days: SCR Dynamic StochOnly ({var}), exit {ex}  |  "
                f"{wins}/{len(tr)} wins, net of 0.1% round trip + funding (KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k, (20, 80)), ("CCI 9", cc, (-100, 100)),
                                             ("RSI 14", r, (30, 70)))):
        a_.plot(x[view:], v[view:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(var, len(tr), "trades", wins, "wins")


if __name__ == "__main__":
    main()
