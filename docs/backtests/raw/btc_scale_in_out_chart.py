"""127번 차트: (1) 규칙별 기준 vs 고른 변형의 일별 자산·낙폭, (2) 고른 변형의 최근 30일 분할 진입·청산.

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_scale_in_out_chart.py \
      <t127 폴더> <변형 키, 예: BS-SO>
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
from btc_ml_futures_reprice import futures_on
from btc_scr_learned import bars
from btc_strategy_combo import RES

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
PICK = {"R1": "R1-PB", "BS": "BS-SO", "AS4": "AS4-SO"}


def kst(ms):
    return mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in np.atleast_1d(ms)])


def equity_chart(folder: Path) -> Path:
    z = np.load(folder / "series.npz")
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for j, (rule, pick) in enumerate(PICK.items()):
        for key, col in ((f"{rule}-base", "#888888"), (pick, "#d73027")):
            dt, dr = z[f"{key}__dt"], z[f"{key}__dr"]
            m = dt >= datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
            eq = np.cumprod(1 + dr[m])
            dd = eq / np.maximum.accumulate(eq) - 1
            x = kst(dt[m])
            ax[0, j].plot(x, eq, color=col, lw=1.1, label=key)
            ax[1, j].plot(x, dd * 100, color=col, lw=0.9)
        ax[0, j].set_yscale("log")
        ax[0, j].set_title(f"{rule}: baseline vs {pick} (1x, daily MTM, from 2022)", loc="left", color=INK, fontsize=11)
        ax[0, j].legend(frameon=False, fontsize=9, loc="upper left")
        ax[1, j].set_ylabel("drawdown %", color=INK)
        for a_ in ax[:, j]:
            a_.grid(color=GRID, lw=0.6)
            for sd in ("top", "right"):
                a_.spines[sd].set_visible(False)
        ax[1, j].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    fig.tight_layout()
    p = folder / "t127_equity.png"
    fig.savefig(p, dpi=105)
    plt.close(fig)
    return p


def trade_chart(folder: Path, key: str) -> Path:
    groups = json.loads((folder / "groups.json").read_text())[key]
    if key.startswith("R1"):
        d = bars(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), 0)
        A = {k: d[k] for k in ("o", "h", "l", "c", "t")}
    else:
        zz = np.load(RES / "ml15/pred.npz")
        A, _ = futures_on(zz["t"], {k: zz[k] for k in "ohlc"})
        A["t"] = zz["t"]
    t, c = A["t"], A["c"]
    last = max(g[3] for g in groups)
    lo = int(np.searchsorted(t, t[last] - 30 * 86_400_000))
    gs = [g for g in groups if g[0] >= lo]
    x = kst(t)
    k, cc, r = stoch(A["h"], A["l"], c, 5), cci(A["h"], A["l"], c, 20), rsi(c, 14)
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:last + 2], c[lo:last + 2], color=INK, lw=0.7)
    wins = 0
    for e, s, segs, end in gs:
        net = sum(w * (s * (xp / ep - 1)) for _, ep, w, _, xp in segs)
        wins += net > 0
        mk = "o" if s > 0 else "^"
        for eb, ep, w, xb, xp in segs:
            p.scatter(x[eb], ep, marker=mk, s=40 + 160 * w, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
            p.scatter(x[xb], xp, marker=mk, s=40 + 160 * w, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
            p.plot([x[eb], x[xb]], [ep, xp], color="#888888", lw=0.8, ls=":")
        p.annotate(f"{net * 100:+.2f}%", (x[end], segs[-1][4]), xytext=(4, 6 if net > 0 else -14),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open (size = part)"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last 30 days to last closed trade: {key} partial entries/exits  |  "
                f"{wins}/{len(gs)} groups won, gross (KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 14", r, (30, 70)))):
        a_.plot(x[lo:last + 2], v[lo:last + 2], color=INK, lw=0.5)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    out = folder / f"t127_{key}.png"
    fig.savefig(out, dpi=105)
    plt.close(fig)
    print(key, len(gs), "groups", wins, "won")
    return out


if __name__ == "__main__":
    f = Path(sys.argv[1])
    print(equity_chart(f))
    print(trade_chart(f, sys.argv[2]))
