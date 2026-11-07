"""최근 7일 SCR 최적화 후보 차트. 실행: python <이 파일> <result.json> <출력 폴더>"""

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

from btc_fractal import load5, resample
from btc_scr_week_opt import FRAMES, ms, signals

KST = timezone(timedelta(hours=9))
BUY, SELL, INK, GRID = "#1a9850", "#d73027", "#333333", "#e6e6e6"


def main() -> None:
    rep = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    d5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    keep = d5["t"] >= ms("2023-09-01 00:00")
    d5 = {k: v[keep] for k, v in d5.items()}
    cache = {}
    for x in rep["picks"]:
        fn = x["frame"]
        if fn not in cache:
            cache[fn] = resample(d5, FRAMES[fn])
        d = cache[fn]
        _, _, s, cc, r = signals(d, x["entry"])
        lo = int(np.searchsorted(d["t"], rep["window7"][0])) - 8
        xs = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in d["t"]])
        fig, ax = plt.subplots(4, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
        p = ax[0]
        p.plot(xs[lo:], d["c"][lo:], color=INK, lw=0.8)
        for g, _, _, sd, i, xi, px, xp in x["trades7"]:
            mk = "o" if sd > 0 else "^"
            p.scatter(xs[i], px, marker=mk, s=110, zorder=5, edgecolor="white", color=BUY if sd > 0 else SELL)
            p.scatter(xs[xi], xp, marker=mk, s=110, zorder=5, edgecolor="white", color=SELL if sd > 0 else BUY)
            p.plot([xs[i], xs[xi]], [px, xp], color="#777777", lw=1.1, ls="-" if g > 0 else ":")
            p.annotate(f"{g * 100:+.2f}%", (xs[xi], xp), xytext=(5, 7) if g > 0 else (-8, -16),
                       textcoords="offset points", fontsize=10, fontweight="bold", color=BUY if g > 0 else SELL)
        w, y = x["w7"], x["y1"]
        p.set_title(f"#{x['rank']} BTCUSDT futures {x['frame']} | entry {x['entry']} | gate {x['gate']} | exit {x['exit']}"
                    f" | last 7d: {w['n']} trades, win {w['win']}%, total {w['ret']:+.2f}% (1x)"
                    f" | prior 12m: {y.get('n')} trades, avg {y.get('avg')}%/trade", loc="left", fontsize=10, color=INK)
        handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
                   plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
                   plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
                   plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
        p.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
        cn = "CCI 9" if x["entry"] == "SCR3c9" else "CCI 20"
        for a_, (nm, v, lines) in zip(ax[1:], (("Stoch 5", s, (20, 80)), (cn, cc, (-100, 100)), ("RSI 14", r, (30, 70)))):
            a_.plot(xs[lo:], v[lo:], color=INK, lw=0.7)
            for ln in lines:
                a_.axhline(ln, color="#999999", lw=0.8, ls="--")
            a_.set_ylabel(nm, color=INK)
        for a_ in ax:
            a_.grid(color=GRID, lw=0.6)
            for sd_ in ("top", "right"):
                a_.spines[sd_].set_visible(False)
        ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
        fig.tight_layout()
        fig.savefig(out / f"scrweek_{x['rank']}.png", dpi=105)
        plt.close(fig)
    fig, a = plt.subplots(figsize=(12, 5))
    k = np.arange(len(rep["picks"]))
    a.bar(k - 0.2, [x["w7"]["avg"] for x in rep["picks"]], 0.4, color="#4575b4", label="last 7 days")
    a.bar(k + 0.2, [x["y1"].get("avg", 0) for x in rep["picks"]], 0.4, color="#f46d43", label="prior 12 months")
    a.axhline(0, color=INK, lw=0.8)
    a.set_xticks(k)
    a.set_xticklabels([f"#{x['rank']}\n{x['frame']} {x['entry']}\n{x['exit']}" for x in rep["picks"]], fontsize=8)
    a.set_ylabel("avg net % per trade (fees 0.1% RT + funding)")
    a.set_title(f"SCR rules optimised on the last 7 days: {rep['combos']} combos tried, "
                f"{rep['w7_positive']} positive in 7 days, only {rep['w7_pos_and_y1_pos']} of those positive "
                f"over the prior 12 months", loc="left", fontsize=10)
    a.legend(frameon=False)
    a.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(out / "scrweek_summary.png", dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()
