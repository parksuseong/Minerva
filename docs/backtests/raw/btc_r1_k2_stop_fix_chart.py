"""129번 차트: (1) 규칙별 기준 vs 고른 변형의 자산 곡선·낙폭 (2) 고른 변형의 최근 30일 거래와 첫 손절선.

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_r1_k2_stop_fix_chart.py \
    <결과 폴더> <R1 변형> <K2 변형> <거래 차트 키, 예: K2|V3|eq|sh0|mean>
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


def equity(out: Path, r1v: str, k2v: str) -> None:
    z = np.load(out / "series.npz")
    fig, ax = plt.subplots(2, 2, figsize=(16, 8), sharex="col", gridspec_kw={"height_ratios": [3, 1]})
    pairs = (("R1", "R1|base|eq|sh0", f"R1|{r1v}|eq|sh0"), ("K2", "K2|base|eq|sh0|mean", f"K2|{k2v}|eq|sh0|mean"))
    for col, (name, kb, kv) in enumerate(pairs):
        for k, c_, lab in ((kb, "#999999", f"{name} base (1% stop)"), (kv, "#d73027", kv.split("|")[1])):
            dt, dr = z[f"{k}__dt"], z[f"{k}__dr"]
            t0 = datetime(2020 if name == "R1" else 2023, 3 if name == "R1" else 1, 1, tzinfo=timezone.utc)
            m = dt >= t0.timestamp() * 1000
            eq = np.cumprod(1 + dr[m])
            dd = eq / np.maximum.accumulate(eq) - 1
            x = mdates.date2num([datetime.fromtimestamp(t / 1000, timezone.utc) for t in dt[m]])
            ax[0, col].plot(x, eq, color=c_, lw=1.2, label=lab)
            ax[1, col].plot(x, dd * 100, color=c_, lw=0.9)
        ax[0, col].set_yscale("log")
        ax[0, col].legend(loc="upper left", frameon=False)
        ax[0, col].set_title(f"{name}: equal risk (1% of equity per trade), realistic fills, daily MTM",
                             loc="left", color=INK, fontsize=10)
        ax[1, col].set_ylabel("drawdown %")
        ax[1, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for a_ in ax.flat:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "t129_equity.png", dpi=110)
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
    fee = 0.0007 if key.startswith("K2") else 0.001
    wins = 0
    for e, s, ep, xb, xp, w in g:
        net = s * (xp / ep - 1) - fee
        wins += net > 0
        sd = 0.01 / w if w < 3 else None
        if sd:
            p.plot([x[e], x[xb]], [ep * (1 - s * sd)] * 2, color="#999999", lw=0.9, ls="--")
        mk = "o" if s > 0 else "^"
        p.scatter(x[e], ep, marker=mk, s=100, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
        p.scatter(x[xb], xp, marker=mk, s=100, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
        p.plot([x[e], x[xb]], [ep, xp], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[xb], xp), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=9, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
               plt.Line2D([], [], color="#999999", ls="--", label="initial stop")]
    p.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last {days} days to 11/03: test 129 {key} (shade = R1 daily direction)  |  "
                f"{wins}/{len(g)} wins, price move net of fees at 1x (KST)", loc="left", color=INK, fontsize=11)
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
    fig.savefig(out / f"t129_{key.split('|')[0]}_{key.split('|')[1]}.png", dpi=110)
    plt.close(fig)
    print(key, len(g), wins)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    equity(o, sys.argv[2], sys.argv[3])
    for k in sys.argv[4:]:
        trades(o, k)
