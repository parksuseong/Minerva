"""128번 차트: (1) 자산 곡선·낙폭 (모형 bc) (2) 한 변형의 최근 30일 거래.

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_r1_as4_entry_chart.py \
    <결과 폴더> <변형 키, 예: K2|bc|mean>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import btc_fut_daily_regime as R
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from btc_fractal import cci, load5, rsi, stoch
from btc_ml_futures_reprice import futures_on
from btc_strategy_combo import RES

KST = timezone(timedelta(hours=9))
BUY, SELL, INK, GRID = "#1a9850", "#d73027", "#333333", "#e6e6e6"
COLORS = {"R1": "#999999", "AS4": "#4575b4", "K1": "#fdae61", "K2": "#d73027", "K3": "#1a9850"}


def equity(out: Path) -> None:
    z = np.load(out / "series.npz")
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for v, col in COLORS.items():
        k = f"{v}|bc|mean"
        dt, dr = z[f"{k}__dt"], z[f"{k}__dr"]
        m = dt >= datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
        dt, dr = dt[m], dr[m]
        eq = np.cumprod(1 + dr)
        dd = eq / np.maximum.accumulate(eq) - 1
        x = mdates.date2num([datetime.fromtimestamp(t / 1000, timezone.utc) for t in dt])
        ax[0].plot(x, eq, color=col, lw=1.2, label=v)
        ax[1].plot(x, dd * 100, color=col, lw=0.9)
    ax[0].set_yscale("log")
    ax[0].legend(loc="upper left", frameon=False, ncol=5)
    ax[0].set_title("BTCUSDT futures 2023-01 ~ 2026-11-03, 1x, daily MTM, realistic fills "
                    "(5-min order delay + limit 0.05% through), 5-seed mean", loc="left", color=INK)
    ax[1].set_ylabel("drawdown %")
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(out / "t128_equity.png", dpi=110)
    plt.close(fig)


def trades(out: Path, key: str, days: int = 30) -> None:
    z = np.load(RES / "ml15/pred.npz")
    tp = z["t"]
    P, _ = futures_on(tp, {k: z[k] for k in "ohlc"})
    P["t"] = tp
    f5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    f5 = {k: v[f5["t"] <= tp[-1] + 900_000] for k, v in f5.items()}
    reg, _ = R.regime(f5, P, 0)
    g = json.loads((out / "groups.json").read_text())[key]
    lo = int(np.searchsorted(tp, tp[-1] - days * 86_400_000))
    g = [x for x in g if x[0] >= lo]
    h, l, c = P["h"], P["l"], P["c"]  # noqa: E741
    k5, c20, r14 = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    x = mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in tp])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.7)
    for i in range(lo, len(tp) - 1):
        if reg[i]:
            p.axvspan(x[i], x[i + 1], color=BUY if reg[i] > 0 else SELL, alpha=0.06, lw=0)
    fee = 0.0007
    wins = 0
    for e, s, ep, xb, xp in g:
        net = s * (xp / ep - 1) - fee
        wins += net > 0
        mk = "o" if s > 0 else "^"
        p.scatter(x[e], ep, marker=mk, s=100, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
        p.scatter(x[xb], xp, marker=mk, s=100, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
        p.plot([x[e], x[xb]], [ep, xp], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[xb], xp), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=9, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last {days} days to 11/03: test 128 {key} (shade = R1 daily direction)  |  "
                f"{wins}/{len(g)} wins, approx. net of fees (KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k5, (20, 80)), ("CCI 20", c20, (-100, 100)),
                                             ("RSI 14", r14, (30, 70)))):
        a_.plot(x[lo:], v[lo:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out / f"t128_{key.split('|')[0]}.png", dpi=110)
    plt.close(fig)
    print(key, len(g), wins)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    equity(o)
    trades(o, sys.argv[2])
