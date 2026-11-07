"""135번 차트. (a) 문 켜짐 음영 (b) 최고 칸 vs 문 없는 판 자산·낙폭 (c) 최고 칸 최근 30일 매매.

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python <이 파일> <결과 폴더> <규칙|문>
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
DAY = 86_400_000


def kx(ms):
    return mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in ms])


def main() -> None:
    d, best = Path(sys.argv[1]), sys.argv[2]
    rule, gname = best.split("|")
    z = np.load(d / "series.npz")
    t, c = z["t15"], z["c"]
    i0 = int(np.searchsorted(t, datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
    x = kx(t)
    # (a)
    fig, ax = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    for a_, g in zip(ax, ("G1", "G2", "G3", "G4")):
        on = z[g]
        a_.plot(x[i0::4], c[i0::4], color=INK, lw=0.6)
        edges = np.flatnonzero(np.diff(np.r_[0, on[i0:].astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            a_.axvspan(x[i0 + s0], x[min(i0 + s1, len(x) - 1)], color=BUY, alpha=0.12, lw=0)
        a_.set_ylabel(f"{g} on {on[i0:].mean() * 100:.0f}%", color=INK)
        a_.set_yscale("log")
        a_.grid(color=GRID, lw=0.6)
    ax[0].set_title("BTCUSDT futures 2023~: entry gates on (green). G1 daily ER10≥0.30, G2 4h CI14<50, "
                    "G3 4h RV pct≥50, G4 G1&G2", loc="left", color=INK)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(d / "t135_gates.png", dpi=100)
    plt.close(fig)
    # (b)
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for key, col in ((f"{rule}|none", "#999999"), (best, "#d73027")):
        s = z[key.replace("|", "__")]
        dt, dr = s[:, 0], s[:, 1]
        m = dt >= t[i0]
        eq = np.cumprod(1 + dr[m])
        pk = np.maximum.accumulate(eq)
        ax[0].plot(kx(dt[m]), eq, color=col, lw=1.2, label=key)
        ax[1].plot(kx(dt[m]), (eq / pk - 1) * 100, color=col, lw=1.0)
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False)
    ax[0].set_title(f"{best} vs {rule} ungated: 1x daily MTM, realistic fills, futures + funding", loc="left",
                    color=INK)
    ax[1].set_ylabel("drawdown %")
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(d / "t135_equity.png", dpi=100)
    plt.close(fig)
    # (c)
    groups = json.loads((d / "groups.json").read_text())[best]
    # 지표는 선물 15분봉 고가·저가로 다시 계산한다
    from btc_fractal import load5
    from btc_scr_learned import bars
    f = bars(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), 0)
    j = np.searchsorted(f["t"], t)
    j = np.minimum(j, len(f["t"]) - 1)
    hh, ll, cc = f["h"][j], f["l"][j], f["c"][j]
    k5, c20, r14 = stoch(hh, ll, cc), cci(hh, ll, cc), rsi(cc)
    lo = int(np.searchsorted(t, t[-1] - 30 * DAY))
    tr = [g for g in groups if g[0] >= lo]
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], cc[lo:], color=INK, lw=0.8)
    on = z[gname]
    edges = np.flatnonzero(np.diff(np.r_[0, on[lo:].astype(int), 0]))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        for a_ in ax:
            a_.axvspan(x[lo + s0], x[min(lo + s1, len(x) - 1)], color=BUY, alpha=0.07, lw=0)
    wins = 0
    for e, s, px, b, xp in tr:
        net = (xp / px - 1) * s - 0.0007
        wins += net > 0
        mk = "o" if s > 0 else "^"
        p.scatter(x[e], px, marker=mk, s=100, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
        p.scatter(x[b], xp, marker=mk, s=100, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
        p.plot([x[e], x[b]], [px, xp], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[b], xp), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=9, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
    ]
    p.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last 30 days to predictions end: {best} (green shade = gate on), "
                f"±1.5% exit  |  {wins}/{len(tr)} wins, net approx (KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k5, (20, 80)), ("CCI 20", c20, (-100, 100)),
                                             ("RSI 14", r14, (30, 70)))):
        a_.plot(x[lo:], v[lo:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(d / "t135_trades.png", dpi=105)
    plt.close(fig)
    print(len(tr), wins)


if __name__ == "__main__":
    main()
