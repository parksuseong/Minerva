"""선물 가격 재체결 거래 차트(최근 30일). 실행: python <이 파일> <trades.json> <키> <출력 png>"""

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
from btc_scr_learned import bars

KST = timezone(timedelta(hours=9))
BUY, SELL, INK, GRID = "#1a9850", "#d73027", "#333333", "#e6e6e6"


def main() -> None:
    src, key, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    tr = json.loads(src.read_text())[key]
    d = bars(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), 0)
    end = max(x[3] for x in tr)
    keep = (d["t"] >= end - 31 * 86_400_000) & (d["t"] <= end + 86_400_000)
    t, h, l, c = d["t"][keep], d["h"][keep], d["l"][keep], d["c"][keep]
    tr = [x for x in tr if x[0] >= end - 30 * 86_400_000]
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    pos = {v: i for i, v in enumerate(t)}
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x, c, color=INK, lw=0.7)
    wins = 0
    for tf, net, side, tx, win in tr:
        i, j = pos.get(tf), pos.get(tx)
        if i is None or j is None:
            continue
        wins += win
        # 진입은 지정가(봉 안 체결), 청산은 ±1.5% 수준
        px = (c[i - 1] if i else c[i]) * (1 - 0.001 * side)
        pout = px * (1 + 0.015 * side) if win else px * (1 - 0.015 * side)
        mk = "o" if side > 0 else "^"
        p.scatter(x[i], px, marker=mk, s=80, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(x[j], pout, marker=mk, s=80, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([x[i], x[j]], [px, pout], color="#777777", lw=1.0, ls="-" if win else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[j], pout), xytext=(4, 6) if win else (-8, -16),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if win else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last 30 days: {key} (limit entry, +/-1.5% exit) | "
                f"{wins}/{len(tr)} wins, 1x net of fees (KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", stoch(h, l, c, 5), (20, 80)),
                                             ("CCI 20", cci(h, l, c), (-100, 100)),
                                             ("RSI 14", rsi(c), (30, 70)))):
        a_.plot(x, v, color=INK, lw=0.5)
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
    print(key, len(tr), wins)


if __name__ == "__main__":
    main()
