"""124번 차트: (a) 묶음 평가액(로그)과 낙폭, (b) 최근 30일 레그별 거래.

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python <이 파일> <t124 폴더>
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
from btc_scr_learned import bars

KST = timezone(timedelta(hours=9))
BUY, SELL, INK, GRID = "#1a9850", "#d73027", "#333333", "#e6e6e6"
COLORS = {"C1": "#999999", "C2": "#4575b4", "C3": "#d73027", "C4": "#1a9850"}
LEG_MK = {"R1": 120, "BS": 70, "AS4": 40}


def equity(folder: Path) -> None:
    z = np.load(folder / "series.npz")
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in z["dt"]])
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, col in COLORS.items():
        eq = np.cumprod(1 + z[f"c_{k}"])
        dd = eq / np.maximum.accumulate(eq) - 1
        ax[0].plot(x, eq, color=col, lw=1.2, label=k)
        ax[1].plot(x, dd * 100, color=col, lw=0.9)
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False, ncol=4, loc="upper left",
                 title="C1 R1 1x | C2 R1+BS2x | C3 R1+AS4 2x | C4 1/3 each")
    ax[0].set_title("BTCUSDT futures 2022-01 ~ 2026-11-03: daily mark-to-market equity, monthly rebalance",
                    loc="left", color=INK)
    ax[1].axhline(-25, color="#999999", ls="--", lw=0.8)
    ax[1].set_ylabel("drawdown %")
    for a in ax:
        a.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a.spines[sd].set_visible(False)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(folder / "t124_equity.png", dpi=110)
    plt.close(fig)


def recent(folder: Path, days: int = 30) -> None:
    d = bars(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), 0)
    tr = json.loads((folder / "trades.json").read_text())
    end = max(v[-1][3] for v in tr.values() if v)
    lo = int(np.searchsorted(d["t"], end - days * 86_400_000))
    hi = int(np.searchsorted(d["t"], end)) + 8
    t, c = d["t"][lo:hi], d["c"][lo:hi]
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    xs = lambda ms: mdates.date2num(datetime.fromtimestamp(ms / 1000, KST))  # noqa: E731
    k, cc, r = stoch(d["h"], d["l"], d["c"]), cci(d["h"], d["l"], d["c"]), rsi(d["c"])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x, c, color=INK, lw=0.7)
    for leg, v in tr.items():
        for e, px, s, xt, ex in v:
            if e < t[0]:
                continue
            net = (ex / px - 1) * s
            mk = "o" if s > 0 else "^"
            p.scatter(xs(e), px, marker=mk, s=LEG_MK[leg], zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
            p.scatter(xs(xt), ex, marker=mk, s=LEG_MK[leg], zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
            p.plot([xs(e), xs(xt)], [px, ex], color="#777777", lw=0.9, ls="-" if net > 0 else ":")
            p.annotate(f"{leg} {net * 100:+.1f}%", (xs(xt), ex), xytext=(3, 5), textcoords="offset points",
                       fontsize=7, color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last {days} days to the last closed trade: R1 (big marker), BS, AS4 (small) trades, "
                "price move % at 1x before fees (KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 14", r, (30, 70)))):
        a_.plot(x, v[lo:hi], color=INK, lw=0.5)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(folder / "t124_recent.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    f = Path(sys.argv[1])
    equity(f)
    recent(f)
