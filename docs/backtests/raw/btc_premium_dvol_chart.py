"""136번 차트: (a) 2024~ 가격과 CP·KP·DVOL, (b) 한 칸의 최근 N일 매매(+ CP 패널, Stoch·CCI·RSI).

실행: PYTHONPATH=<numpy·matplotlib>:src:docs/backtests/raw python <이 파일> <결과 폴더> <칸> <일수>
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

from btc_fractal import cci, load5, resample, rsi, stoch

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
H = 3_600_000
COST = 0.0005


def dnum(ms):
    return mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in ms])


def overview(res: Path) -> None:
    s = np.load(res / "series.npz")
    tc = s["tc"]
    lo = int(np.searchsorted(tc, datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
    x = dnum(tc[lo:])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    ax[0].plot(x, s["close"][lo:], color=INK, lw=0.6)
    ax[0].set_ylabel("BTCUSDT perp", color=INK)
    ax[0].set_title("2024~: Coinbase premium (CP), Kimchi premium (KP), Deribit DVOL vs 30d realised vol (KST)",
                    loc="left", color=INK, fontsize=11)
    ax[1].plot(x, s["cp"][lo:] * 100, color="#4575b4", lw=0.5)
    ax[1].axhline(0, color="#999999", lw=0.8)
    ax[1].set_ylabel("CP %", color=INK)
    ax[2].plot(x, s["kp"][lo:] * 100, color="#d73027", lw=0.5)
    ax[2].axhline(0, color="#999999", lw=0.8)
    ax[2].set_ylabel("KP %", color=INK)
    ax[3].plot(x, s["dv"][lo:], color="#7b3294", lw=0.6, label="DVOL")
    ax[3].plot(x, (s["dv"] - s["spread"])[lo:], color="#999999", lw=0.6, label="RV 30d")
    ax[3].legend(loc="upper right", frameon=False, fontsize=9)
    ax[3].set_ylabel("vol %", color=INK)
    for a in ax:
        a.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a.spines[sd].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    fig.tight_layout()
    fig.savefig(res / "t136_overview.png", dpi=110)
    plt.close(fig)


def trades(res: Path, cell: str, days: int) -> None:
    rep = json.loads((res / "result.json").read_text())
    tr = rep["cells"][cell]["hold_trades"]
    s = np.load(res / "series.npz")
    f5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    d = resample(f5, 12)
    t = d["t"]
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    tr = [z for z in tr if z[0] >= t[lo]]
    x = dnum(t)
    st, cc, r = stoch(d["h"], d["l"], d["c"]), cci(d["h"], d["l"], d["c"]), rsi(d["c"])
    j = np.searchsorted(s["tc"], t + H)
    ok = (j < len(s["tc"])) & (s["tc"][np.minimum(j, len(s["tc"]) - 1)] == t + H)
    cp = np.full(len(t), np.nan)
    cp[ok] = s["cp"][j[ok]]
    fig, ax = plt.subplots(5, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], d["c"][lo:], color=INK, lw=0.8)
    wins = 0
    for t0, side, g, t1, why in tr:
        net = g - 2 * COST
        wins += net > 0
        e = int(np.searchsorted(t, t0, side="right") - 1)
        q = int(np.searchsorted(t, t1, side="right") - 1)
        e5 = int(np.searchsorted(f5["t"], t0))
        px = f5["o"][e5]
        pout = px * (1 + g * side)
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px, marker=mk, s=90, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(x[q], pout, marker=mk, s=90, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([x[e], x[q]], [px, pout], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[q], pout), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT perp 1h, last {days} days: test 136 {cell} (Coinbase premium), {wins}/{len(tr)} wins, "
                "net of 0.1% round trip + funding, entry 1h close +5 min (KST)", loc="left", color=INK, fontsize=11)
    ax[1].plot(x[lo:], cp[lo:] * 100, color="#4575b4", lw=0.7)
    ax[1].axhline(0, color="#999999", lw=0.8)
    ax[1].set_ylabel("CP %", color=INK)
    for a_, (name, v, lines) in zip(ax[2:], (("Stoch 5", st, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 14", r, (30, 70)))):
        a_.plot(x[lo:], v[lo:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a in ax:
        a.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a.spines[sd].set_visible(False)
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(res / f"t136_{cell}_{days}d.png", dpi=110)
    plt.close(fig)
    print(cell, len(tr), "trades", wins, "wins")


if __name__ == "__main__":
    res = Path(sys.argv[1])
    overview(res)
    trades(res, sys.argv[2], int(sys.argv[3]))
