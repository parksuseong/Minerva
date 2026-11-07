"""123번 차트: 확정 스윙 라벨, 거래, 추적 손절, 일봉 방향 음영, Stoch·CCI·RSI.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <result.json> <P15|P1h> <일수> <png>
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

import btc_fut_daily_regime as R
import btc_swing_trend_trail as T
from btc_fractal import cci, load5, rsi, stoch
from btc_scr_learned import bars

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"


def main() -> None:
    res, var, days, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    d5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    d = bars(d5, 0)
    t, c, h, l = d["t"], d["c"], d["h"], d["l"]
    reg, _ = R.regime(d5, d, 0)
    frame = 1 if var == "P15" else 4
    piv = T.pivots(d, frame)
    tr = [z for z in json.loads(res.read_text())[var]["trades"]]
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    tr = [z for z in tr if z[0] >= lo]
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    k, cc, r = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.7)
    for i in range(lo, len(t) - 1):
        if reg[i]:
            p.axvspan(x[i], x[i + 1], color=BUY if reg[i] > 0 else SELL, alpha=0.06, lw=0)
    for jj, kind, px, hi, low in piv:
        if jj < lo:
            continue
        lab = ("HL" if hi else "LL") if kind == 1 else ("LH" if low else "HH")
        col = BUY if kind == 1 else SELL
        p.annotate(lab, (x[jj], px), xytext=(0, -14 if kind == 1 else 8), textcoords="offset points",
                   fontsize=7, color=col, ha="center")
    wins = 0
    for i, side, g, q, _ in tr:
        e = i + 1
        px_in = d["o"][e]
        net = g - 0.001
        wins += net > 0
        px_out = px_in * (1 + g * side)
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=90, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(x[q], px_out, marker=mk, s=90, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([x[e], x[q]], [px_in, px_out], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[q], px_out), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if net > 0 else SELL)
        # 추적 손절 계단: 진입 손절에서 새 HL/LH마다 갱신
        stops = []
        for jj, kind, pv, hi, low in piv:
            if e - 1 <= jj < q and ((side > 0 and kind == 1 and hi) or (side < 0 and kind == -1 and low)):
                stops.append((jj, pv * (1 - 0.001 * side)))
        if stops:
            first = stops[0][1]
            lim = px_in * (1 - 0.01 * side)
            cur = max(first, lim) if side > 0 else min(first, lim)
            xs, ys = [x[e]], [cur]
            for jj, sp in stops[1:]:
                cur = max(cur, sp) if side > 0 else min(cur, sp)
                xs += [x[jj + 1], x[jj + 1]]
                ys += [ys[-1], cur]
            xs.append(x[q])
            ys.append(ys[-1])
            p.plot(xs, ys, color="#999999", lw=0.9, ls="--")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], color="#999999", ls="--", label="trailing stop"),
    ]
    p.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last {days} days: test 123 {var} (confirmed RSI7 swings, "
                f"shade = daily direction)  |  {wins}/{len(tr)} wins, net of 0.1% + funding (KST)",
                loc="left", color=INK, fontsize=11)
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
