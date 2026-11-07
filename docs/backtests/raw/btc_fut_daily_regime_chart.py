"""R1(일봉 방향 보유, 1% 손절) 최근 거래 차트, 진행 중인 포지션 포함.

실행: PYTHONPATH=<numpy·matplotlib 경로>:src:docs/backtests/raw python <이 파일> <result.json> <일수> <출력 png> [변형, 기본 R1]
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
from btc_fractal import cci, load5, resample, rsi, stoch
from btc_scr_learned import bars

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
COST = 0.0005


def main() -> None:
    res, days, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    var = sys.argv[4] if len(sys.argv) > 4 else "R1"
    d5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    d = bars(d5, 0)
    t, c = d["t"], d["c"]
    reg, newday = R.regime(d5, d, 0)
    tr = [z for z in json.loads(res.read_text())[var]["trades"]]
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    tr = [z for z in tr if z[0] >= lo]
    # run()은 끝나지 않은 다리에서 멈춘다. 마지막 청산 뒤 첫 방향 봉이 진행 중인 진입이다.
    last_x = tr[-1][3] if tr else lo
    nz = [i for i in range(last_x - 1, len(t) - 2) if reg[i]]
    opened = (nz[0], int(reg[nz[0]])) if nz and var == "R1" else None  # 다른 변형은 방향이 R1과 달라 생략
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    u = resample(d, 96, base_ms=900_000)
    ux = mdates.date2num([datetime.fromtimestamp((v + 86_400_000) / 1000, KST) for v in u["t"]])
    k, cc, r = stoch(u["h"], u["l"], u["c"], 5), cci(u["h"], u["l"], u["c"], 20), rsi(u["c"], 14)
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.7)
    for i in range(lo, len(t) - 1):
        if reg[i]:
            p.axvspan(x[i], x[i + 1], color=BUY if reg[i] > 0 else SELL, alpha=0.06, lw=0)
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
    note = ""
    if opened:
        i, side = opened
        e = i + 1
        px_in = d["o"][e]
        now = (c[-1] / px_in - 1) * side - 2 * COST
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=110, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.axhline(px_in * (1 - 0.01 * side), color="#999999", lw=0.8, ls="--")
        p.annotate(f"open {'long' if side > 0 else 'short'} {px_in:,.0f}, now {now * 100:+.2f}%",
                   (x[e], px_in), xytext=(6, 10), textcoords="offset points", fontsize=10, color=INK)
        note = f" | open {'long' if side > 0 else 'short'} since " + \
            datetime.fromtimestamp(t[e] / 1000, KST).strftime("%m-%d %H:%M")
    p.set_title(f"BTCUSDT futures 15m, last {days} days: {var} (shade = daily direction), 1% stop  |  "
                f"{wins}/{len(tr)} closed wins, net of 0.1% round trip + funding{note} (KST)",
                loc="left", color=INK, fontsize=11)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], color="#999999", ls="--", label="stop"),
    ]
    p.legend(handles=handles, loc="upper right", ncol=5, frameon=False, fontsize=9)
    ulo = int(np.searchsorted(ux, x[lo]))
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5 (1D)", k, (20, 80)), ("CCI 20 (1D)", cc, (-100, 100)),
                                             ("RSI 14 (1D)", r, (30, 70)))):
        a_.plot(ux[ulo:], v[ulo:], color=INK, lw=0.9, marker=".")
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.set_xlim(x[lo], x[-1])
        a_.grid(color=GRID, lw=0.6)
        a_.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(len(tr), wins, opened)


if __name__ == "__main__":
    main()
