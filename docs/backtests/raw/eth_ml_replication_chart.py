"""143번 차트: ETH vs BTC 진입 우위 막대, ETH 최근 30일 매매.

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python <이 파일> <t143 폴더> <btc 133 레드팀 c133_stage1.json> <규칙>
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

from btc_fractal import cci, rsi, stoch

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"


def bars(res: dict, btc: dict, out: Path) -> None:
    names = [("AS-x", "AS"), ("BS-x", "BS"), ("AS4-x", "AS4"), ("K2-x", "K2")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for a_, per in zip(ax, ("2023", "2024~")):
        x = np.arange(len(names))
        eth = [res["edge"][e][per].get("edge", np.nan) for e, _ in names]
        bt = [btc.get(f"{b}|{per}", {}).get("edgeB", np.nan) for _, b in names]
        a_.bar(x - 0.2, bt, 0.4, color="#999999", label="BTC (133, same fill model)")
        a_.bar(x + 0.2, eth, 0.4, color="#4575b4", label="ETH (143)")
        for xi, v in zip(x + 0.2, eth):
            if np.isfinite(v):
                a_.annotate(f"{v:+.1f}", (xi, v), xytext=(0, 3 if v >= 0 else -12), textcoords="offset points",
                            ha="center", fontsize=9, color=INK)
        a_.axhline(0, color=INK, lw=0.8)
        a_.set_xticks(x)
        a_.set_xticklabels([e for e, _ in names])
        a_.set_title(f"{per}: entry edge, P(+1.5% before -1.5%) minus random, %p", loc="left", color=INK)
        a_.grid(color=GRID, lw=0.6, axis="y")
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def trades(z, groups: list, name: str, out: Path, days: int = 30) -> None:
    t, o, h, l, c = z["t"], z["o"], z["h"], z["l"], z["c"]  # noqa: E741
    lo = int(np.searchsorted(t, t[-1] - days * 86_400_000))
    view = max(0, lo - 16)
    s, cc, r = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[view:], c[view:], color=INK, lw=0.8)
    wins = n = 0
    for e, side, segs, q in groups:
        if e < lo:
            continue
        sg = segs[0]
        px_in, px_out = sg[1], sg[4]
        net = side * (px_out / px_in - 1) - sg[5] - sg[6]
        n += 1
        wins += net > 0
        q = min(int(q), len(x) - 1)
        mk = "o" if side > 0 else "^"
        p.scatter(x[e], px_in, marker=mk, s=100, zorder=5, edgecolor="white", color=BUY if side > 0 else SELL)
        p.scatter(x[q], px_out, marker=mk, s=100, zorder=5, edgecolor="white", color=SELL if side > 0 else BUY)
        p.plot([x[e], x[q]], [px_in, px_out], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[q], px_out), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=9, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
    ]
    p.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"ETHUSDT futures 15m, last {days} days: {name} (BTC ML pipeline unchanged, 5 XGB seeds), "
                f"realistic fills  |  {wins}/{n} wins, net of fees (KST)", loc="left", color=INK, fontsize=11)
    for a_, (nm, v, lines) in zip(ax[1:], (("Stoch 5", s, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                           ("RSI 14", r, (30, 70)))):
        a_.plot(x[view:], v[view:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(nm, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    d, btcf, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    res = json.loads((d / "result.json").read_text())
    btc = json.loads(btcf.read_text())
    bars(res, btc, d / "t143_edge.png")
    z = np.load(d / "pred.npz")
    g = json.loads((d / "groups.json").read_text())[name]
    trades(z, g, name, d / f"t143_{name}_30d.png")


if __name__ == "__main__":
    main()
