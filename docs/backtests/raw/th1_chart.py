"""145번 TH1 차트: (a) 100만원 시작 자산 곡선 vs 보유, (b) 최근 90일 매매.

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python docs/backtests/raw/th1_chart.py <t145 폴더>
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
from btc_fractal import cci, resample, rsi, stoch
from trend_hold_cross_market import BAR, Mkt

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
C15, C20, CBH = "#4575b4", "#d95f02", "#999999"
KRW = 1_000_000


def kx(ms):
    return mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in np.atleast_1d(ms)])


def equity(folder: Path) -> None:
    z = np.load(folder / "series.npz")
    fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for key, col, lab in (("BH", CBH, "BTC buy & hold"), ("r1.5", C15, "TH1 risk 1.5%"), ("r2", C20, "TH1 risk 2.0%")):
        t, e = z[f"M1|{key}|t"], z[f"M1|{key}|e"]
        t0 = z["M1|r2|t"][0]
        m = t >= t0
        t, e = t[m], e[m]
        v = e / e[0] * KRW
        ax[0].plot(kx(t), v, color=col, lw=1.4 if key != "BH" else 1.0, label=f"{lab}  final {v[-1]:,.0f} KRW")
        ax[1].plot(kx(t), (v / np.maximum.accumulate(v) - 1) * 100, color=col, lw=1.0)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("KRW (log)", color=INK)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y / 1e6:,.1f}M"))
    ax[0].legend(loc="upper left", frameon=False, fontsize=10)
    ax[0].set_title("BTCUSDT futures 2020-03 ~ 2026-11, start 1,000,000 KRW: TH1 (daily direction long-only, ATR sizing, 1% stop) "
                    "vs buy & hold (fees, funding, stop slippage included)", loc="left", color=INK, fontsize=11)
    ax[1].set_ylabel("drawdown %", color=INK)
    ax[1].axhline(-25, color="#999999", lw=0.8, ls="--")
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(folder / "t145_equity_krw.png", dpi=110)
    plt.close(fig)


def recent(folder: Path, days: int = 90) -> None:
    m = Mkt("M1")
    d = m.d
    t, c = d["t"], d["c"]
    reg, _ = R.regime(m.d5, d, 0)
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    tr = json.loads((folder / "trades_M1_r2.json").read_text())
    tr = [z for z in tr if z["x"] >= lo]
    u = resample(m.d5, 288)
    ux = kx(u["t"] + 86_400_000)
    s, cc, r = stoch(u["h"], u["l"], u["c"], 5), cci(u["h"], u["l"], u["c"]), rsi(u["c"])
    x = kx(t)
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.7)
    for i in range(lo, len(t) - 1):
        if reg[i] == 1:
            p.axvspan(x[i], x[i + 1], color=BUY, alpha=0.07, lw=0)
    wins = done = 0
    for z in tr:
        xi, xo = kx(z["t_in"])[0], kx(z["t_out"])[0]
        p.scatter(xi, z["px"], marker="o", s=100, zorder=5, edgecolor="white", color=BUY)
        if z["open"]:
            p.annotate(f"open {z['ret'] * 100:+.2f}%", (xi, z["px"]), xytext=(6, 10), textcoords="offset points",
                       fontsize=10, color=INK)
            continue
        done += 1
        wins += z["ret"] > 0
        p.scatter(xo, z["xp"], marker="o", s=100, zorder=5, edgecolor="white", color=SELL)
        p.plot([xi, xo], [z["px"], z["xp"]], color="#777777", lw=1.1, ls="-" if z["ret"] > 0 else ":")
        p.annotate(f"{z['ret'] * 100:+.2f}%", (xo, z["xp"]), xytext=(4, 6) if z["ret"] > 0 else (-8, -16),
                   textcoords="offset points", fontsize=9, fontweight="bold", color=BUY if z["ret"] > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Rectangle((0, 0), 1, 1, color=BUY, alpha=0.15, label="daily direction long")]
    p.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures, last {days} days: TH1 risk 2.0%  |  {wins}/{done} closed wins, % of equity (KST)",
                loc="left", color=INK, fontsize=11)
    ulo = int(np.searchsorted(ux, x[lo]))
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5 (1D)", s, (20, 80)), ("CCI 20 (1D)", cc, (-100, 0, 100)),
                                             ("RSI 14 (1D)", r, (30, 50, 70)))):
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
    fig.savefig(folder / "t145_recent90.png", dpi=110)
    plt.close(fig)
    print("recent trades", len(tr), "closed", done, "wins", wins, "last bar", BAR and datetime.fromtimestamp(t[-1] / 1000, KST))


def main() -> None:
    folder = Path(sys.argv[1])
    for fam in ("NanumGothic", "Noto Sans CJK KR", "UnDotum"):
        if any(fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    equity(folder)
    recent(folder)


if __name__ == "__main__":
    main()
