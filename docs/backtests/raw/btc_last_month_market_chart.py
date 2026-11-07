"""최근 한 달 시장가 가정 차트: 자산 곡선 1장 + 기법별 매매 차트(2개씩 묶어 3장, R1 1장).

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python <이 파일> <결과 폴더>
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
from btc_last_month_market import FEE, FUT5, equity
from btc_scr_learned import bars

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
NAMES = ("A", "B", "AS", "BS", "AS4", "K2", "R1")
COLORS = {
    "A": "#4575b4",
    "B": "#91bfdb",
    "AS": "#fc8d59",
    "BS": "#fee090",
    "AS4": "#d73027",
    "K2": "#1a9850",
    "R1": "#762a83",
}


def main() -> None:
    res = Path(sys.argv[1])
    rep = json.loads((res / "result.json").read_text())
    trades = json.loads((res / "trades.json").read_text())
    d5 = load5(FUT5)
    d = bars(d5, 0)
    t, c = d["t"], d["c"]
    t0 = datetime.fromisoformat(rep["start_kst"]).timestamp() * 1000
    a = int(np.searchsorted(t, t0))
    x = mdates.date2num([datetime.fromtimestamp((v + 900_000) / 1000, KST) for v in t])
    reg, _ = R.regime(d5, d, 0)

    # (a) 자산 곡선
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(
        x[a:],
        c[a:] / c[a - 1],
        color="#999999",
        lw=1.2,
        ls="--",
        label=f"BTC buy & hold {rep['buy_hold_pct']:+.1f}%",
    )
    for k in NAMES:
        e = equity(d, trades[k], a, 1.0, True)
        ax.plot(
            x[a:],
            e,
            color=COLORS[k],
            lw=1.4,
            label=f"{k} {rep[k]['cost']['ret1']:+.2f}% ({rep[k]['cost']['n']} trades)",
        )
    ax.axhline(1, color="#bbbbbb", lw=0.8)
    ax.set_title(
        "BTCUSDT futures, 2026-10-07 ~ 11-07 KST: market fill at intended price, 1x, "
        "fees 0.05%/side + funding, 15m mark-to-market",
        loc="left",
        color=INK,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)
    ax.grid(color=GRID, lw=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    fig.tight_layout()
    fig.savefig(res / "lastmonth_equity.png", dpi=110)
    plt.close(fig)

    # (b) 매매 차트
    k_, cc_, r_ = stoch(d["h"], d["l"], c, 5), cci(d["h"], d["l"], c, 20), rsi(c, 14)
    groups = (("A", "B"), ("AS", "BS"), ("AS4", "K2"), ("R1",))
    for gi, grp in enumerate(groups, 1):
        rows = len(grp) + 3
        fig, ax = plt.subplots(
            rows,
            1,
            figsize=(16, 4.2 * len(grp) + 5),
            sharex=True,
            gridspec_kw={"height_ratios": [4] * len(grp) + [1, 1, 1]},
        )
        for pi, name in enumerate(grp):
            p = ax[pi]
            p.plot(x[a:], c[a:], color=INK, lw=0.7)
            if name in ("K2", "R1"):
                for i in range(a, len(t) - 1):
                    if reg[i]:
                        p.axvspan(
                            x[i], x[i + 1], color=BUY if reg[i] > 0 else SELL, alpha=0.05, lw=0
                        )
            wins = 0
            for z in trades[name]:
                i, q, s = z["i"], z["x"], z["s"]
                net = z["net"]
                wins += net > 0
                mk = "o" if s > 0 else "^"
                p.scatter(
                    x[i],
                    z["px"],
                    marker=mk,
                    s=80,
                    zorder=5,
                    edgecolor="white",
                    color=BUY if s > 0 else SELL,
                )
                if z["open"]:
                    p.annotate(
                        f"open {net * 100:+.2f}%",
                        (x[i], z["px"]),
                        xytext=(6, 8),
                        textcoords="offset points",
                        fontsize=8,
                        color=INK,
                    )
                    continue
                p.scatter(
                    x[q],
                    z["xp"],
                    marker=mk,
                    s=80,
                    zorder=5,
                    edgecolor="white",
                    color=SELL if s > 0 else BUY,
                )
                p.plot(
                    [x[i], x[q]],
                    [z["px"], z["xp"]],
                    color="#777777",
                    lw=1.0,
                    ls="-" if net > 0 else ":",
                )
                p.annotate(
                    f"{net * 100:+.2f}%",
                    (x[q], z["xp"]),
                    xytext=(4, 6) if net > 0 else (-8, -14),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color=BUY if net > 0 else SELL,
                )
            cs = rep[name]["cost"]
            p.set_title(
                f"{name}: {cs['n']} trades (L {cs['long']} / S {cs['short']}), win {cs['win']}%, "
                f"1x {cs['ret1']:+.2f}% / 3x {cs['ret3']:+.2f}%, MDD 1x {cs['mdd1']}%  "
                f"(net of 0.1% round trip + funding)",
                loc="left",
                color=INK,
                fontsize=10,
            )
            p.grid(color=GRID, lw=0.6)
            if pi == 0:
                p.legend(
                    handles=[
                        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
                        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
                        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
                        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
                    ],
                    loc="upper right",
                    ncol=4,
                    frameon=False,
                    fontsize=9,
                )
        for a_, (nm, v, lines) in zip(
            ax[len(grp) :],
            (("Stoch 5", k_, (20, 80)), ("CCI 20", cc_, (-100, 100)), ("RSI 14", r_, (30, 70))),
        ):
            a_.plot(x[a:], v[a:], color=INK, lw=0.5)
            for ln in lines:
                a_.axhline(ln, color="#999999", lw=0.8, ls="--")
            a_.set_ylabel(nm, color=INK)
            a_.grid(color=GRID, lw=0.6)
        for a_ in ax:
            for sd in ("top", "right"):
                a_.spines[sd].set_visible(False)
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        fig.tight_layout()
        fig.savefig(res / f"lastmonth_trades{gi}.png", dpi=100)
        plt.close(fig)
    print("ok", FEE)


if __name__ == "__main__":
    main()
