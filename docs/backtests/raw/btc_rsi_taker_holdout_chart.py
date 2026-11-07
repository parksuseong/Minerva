"""141번 차트: 현물 표본 밖 자산 곡선(RT2-F, RT2-R)과 표본 밖 마지막 60일 매매.

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python <이 파일> <결과 폴더>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import btc_indicator_library as L
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from btc_fractal import cci, rsi, stoch

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"


def main() -> None:
    d = Path(sys.argv[1])
    rep = json.loads((d / "result.json").read_text())["spot_holdout"]
    npz = np.load(d / "spot.npz")
    t15, taker = npz["t15"], npz["taker"]
    xt = lambda ms: mdates.date2num(datetime.fromtimestamp(ms / 1000, KST))  # noqa: E731
    # (a) 자산 곡선
    fig, ax = plt.subplots(figsize=(14, 5))
    for v, col in (("F", "#4575b4"), ("R", "#fdae61")):
        tr = rep[v]["trades"]
        if not tr:
            continue
        eq = np.cumprod(1 + np.array([z[2] for z in tr]))
        ax.plot([xt(t15[z[0]]) for z in tr], eq, color=col, lw=1.4,
                label=f"RT2-{v}: {len(tr)} trades, {(eq[-1] - 1) * 100:+.1f}% (1x)")
    ax.axhline(1, color="#999999", lw=0.8)
    ax.set_title("BTCUSDT spot 2017-09 ~ 2019-12 (holdout): 4h RSI slope + 4h taker ratio (RT2), ±1.5%, "
                 "fees 0.05%/side, 5-min delay", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False)
    ax.grid(color=GRID, lw=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(d / "t141_equity.png", dpi=110)
    plt.close(fig)
    # (b) 마지막 60일 매매 (RT2-F)
    sp = L.load_fut5(Path("~/.minerva/data/btc/5m.csv").expanduser())
    d15 = L.resample(sp, 3)
    assert np.array_equal(d15["t"], t15)
    end = L.ts(2020)
    lo = int(np.searchsorted(t15, end - 60 * L.DAY))
    hi = int(np.searchsorted(t15, end))
    x = np.array([xt(v) for v in t15[lo:hi]])
    h, l_, c = d15["h"], d15["l"], d15["c"]
    s, cc, r = stoch(h, l_, c, 5), cci(h, l_, c, 20), rsi(c, 14)
    fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1]})
    p = ax[0]
    p.plot(x, c[lo:hi], color=INK, lw=0.7)
    t5 = sp["t"]
    o5 = sp["o"]
    wins = n = 0
    for i, side, net, xtm in rep["F"]["trades"]:
        if not lo <= i < hi:
            continue
        e = int(np.searchsorted(t5, t15[i] + L.B15 + L.M5))
        px = o5[e]
        k = int(np.searchsorted(t5, xtm))
        out_px = px * (1 + (net + 2 * L.FEE) * side)
        mk = "o" if side > 0 else "^"
        p.scatter(xt(t5[e]), px, marker=mk, s=90, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(xt(t5[k]), out_px, marker=mk, s=90, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([xt(t5[e]), xt(t5[k])], [px, out_px], color="#777777", lw=1.0, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (xt(t5[k]), out_px), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if net > 0 else SELL)
        n += 1
        wins += net > 0
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT spot 15m, last 60 days of holdout (2019-11/12): RT2-F, ±1.5%  |  {wins}/{n} wins (KST)",
                loc="left", color=INK, fontsize=11)
    panels = (("Stoch 5", s, (20, 80)), ("CCI 20", cc, (-100, 100)), ("RSI 14", r, (30, 70)),
              ("4h taker ratio", taker, (npz["F_tak_lo"][0], npz["F_tak_hi"][0])))
    for a_, (name, v, lines) in zip(ax[1:], panels):
        a_.plot(x, v[lo:hi], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(d / "t141_trades.png", dpi=110)
    plt.close(fig)
    print("charts done", n, wins)


if __name__ == "__main__":
    main()
