"""144번 차트: 시장별 자산 곡선(로그)과 낙폭, BTC 최근 60일 매매.

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python <이 파일> <결과 폴더> <변형>
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
from trend_hold_cross_market import MARKETS, Mkt

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
COLORS = {"BH": "#999999", "T0": "#333333", "T1": "#8c564b", "T2": "#1f77b4", "T3": "#2ca02c",
          "T4": "#ff7f0e", "T5": "#9467bd"}
NAMES = {"M1": "BTC futures 2020~", "M2": "BTC spot 2017-10~2019", "M3": "ETH futures 2020~"}


def equity(res: Path) -> None:
    z = np.load(res / "series.npz")
    fig, ax = plt.subplots(2, 3, figsize=(18, 9), gridspec_kw={"height_ratios": [3, 1]})
    for c, mk in enumerate(MARKETS):
        for v in ("BH", "T0", "T1", "T2", "T3", "T4", "T5"):
            t, e = z[f"{mk}|{v}|t"], z[f"{mk}|{v}|e"]
            x = mdates.date2num([datetime.fromtimestamp(a / 1000, timezone.utc) for a in t])
            e = e / e[0]
            lw = 1.6 if v in ("T3", "T0", "BH") else 1.0
            ax[0, c].plot(x, e, color=COLORS[v], lw=lw, label=v, ls="--" if v == "BH" else "-")
            ax[1, c].plot(x, (e / np.maximum.accumulate(e) - 1) * 100, color=COLORS[v], lw=0.8)
        ax[0, c].set_yscale("log")
        ax[0, c].set_title(NAMES[mk], loc="left", color=INK)
        ax[1, c].set_ylabel("drawdown %")
        for a_ in ax[:, c]:
            a_.grid(color=GRID, lw=0.6)
            a_.xaxis.set_major_locator(mdates.YearLocator())
            a_.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            for sd in ("top", "right"):
                a_.spines[sd].set_visible(False)
    ax[0, 0].legend(ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Test 144: daily trend hold T0–T5 vs buy & hold (1x equity, daily MTM, fees + funding). "
                 "T0=R1, T1 long-only, T2/T3 ATR-sized, T4/T5 ATR-sized no stop", color=INK)
    fig.tight_layout()
    fig.savefig(res / "t144_equity.png", dpi=100)
    plt.close(fig)


def recent(res: Path, var: str, days: int = 60) -> None:
    tr = json.loads((res / "trades.json").read_text())[f"M1|{var}"]
    m = Mkt("M1")
    d = m.d
    reg, _ = R.regime(m.d5, d, 0)
    t, c = d["t"], d["c"]
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    x = mdates.date2num([datetime.fromtimestamp(a / 1000, KST) for a in t])
    u = resample(m.d5, 288)
    ux = mdates.date2num([datetime.fromtimestamp((a + 86_400_000) / 1000, KST) for a in u["t"]])
    k, cc, r = stoch(u["h"], u["l"], u["c"], 5), cci(u["h"], u["l"], u["c"], 20), rsi(u["c"])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.7)
    for i in range(lo, len(t) - 1):
        if reg[i]:
            p.axvspan(x[i], x[i + 1], color=BUY if reg[i] > 0 else SELL, alpha=0.06, lw=0)
    shown = [z for z in tr if z[3] >= lo]
    wins = 0
    for e, s, g, xb, stopped, w, _ in shown:
        px_in = d["o"][e]
        px_out = d["o"][min(xb, len(t) - 1)] if not stopped else px_in * (1 - 0.01 * s)
        mk = "o" if s > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=100, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
        p.scatter(x[xb], px_out, marker=mk, s=100, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
        p.plot([x[e], x[xb]], [px_in, px_out], color="#777777", lw=1.1, ls="-" if g > 0 else ":")
        wins += g > 0
        p.annotate(f"{g * 100:+.2f}%", (x[xb], px_out), xytext=(4, 6) if g > 0 else (-8, -16),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if g > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures, last {days} days: {var} (shade = daily direction)  |  {wins}/{len(shown)} trades "
                f"positive, % of equity incl. fees + funding (KST)", loc="left", color=INK, fontsize=11)
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
    fig.savefig(res / f"t144_{var}_recent.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    res = Path(sys.argv[1])
    equity(res)
    recent(res, sys.argv[2])
