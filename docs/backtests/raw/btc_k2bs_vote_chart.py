"""134번 차트: 변형별 자산 곡선, 가장 좋은 변형의 최근 30일 매매(검문·투표 수).

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python <이 파일> <134 결과 폴더>
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
from btc_ml_futures_reprice import futures_on
from btc_strategy_combo import RES

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"
DAY = 86_400_000
COLS = {"E-X0": "#999999", "E-X1": "#3182bd", "E-X2": "#d73027", "E-X3": "#1a9850", "K2-X0": "#fdae6b",
        "BS-X0": "#c6dbef"}


def main() -> None:
    d = Path(sys.argv[1])
    rep = json.loads((d / "result.json").read_text())
    z = np.load(d / "series.npz")
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
    for var, col in COLS.items():
        dt, dr = z[f"{var}__dt"], z[f"{var}__dr"]
        m = dt >= t0
        eq = np.cumprod(1 + dr[m])
        xx = mdates.date2num([datetime.fromtimestamp(t / 1000, timezone.utc) for t in dt[m]])
        r24 = rep["res"][f"{var}|bc"]["2024~ 1x"]
        ax[0].plot(xx, eq, color=col, lw=1.3 if var.startswith("E") else 0.9,
                   label=f"{var} (2024~ CAGR {r24.get('cagr')}%, MDD {r24.get('mdd')}%)")
        ax[1].plot(xx, (eq / np.maximum.accumulate(eq) - 1) * 100, color=col, lw=0.8)
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Test 134: K2+BS entries (E), exits X0 ±1.5% / X1 vote, hold stop +1% / X2 vote, 2% steps / "
                    "X3 K2 by R1 exit; 1x, daily MTM, realistic fills", loc="left", color=INK, fontsize=10)
    ax[1].set_ylabel("drawdown %")
    for a_ in ax:
        a_.grid(color=GRID)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(d / "t134_equity.png", dpi=110)
    plt.close(fig)
    best = max(("E-X1", "E-X2", "E-X3"), key=lambda v: rep["res"][f"{v}|bc"]["2024~ 1x"].get("ratio") or -9)
    logs = json.loads((d / "logs.json").read_text())[best]
    zz = np.load(RES / "ml15/pred.npz")
    tp = zz["t"]
    P, _ = futures_on(tp, {k: zz[k] for k in "ohlc"})
    end = tp[-1]
    lo = int(np.searchsorted(tp, end - 30 * DAY))
    xs = mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in tp])
    k_, cc, r = stoch(P["h"], P["l"], P["c"], 5), cci(P["h"], P["l"], P["c"]), rsi(P["c"])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(xs[lo:], P["c"][lo:], color=INK, lw=0.7)
    wins = cnt = 0
    for x_ in logs:
        if x_["e"] < lo:
            continue
        cnt += 1
        e, s, px, xb, xp = x_["e"], x_["s"], x_["px"], x_["xb"], x_["xp"]
        net = s * (xp / px - 1) - 0.0007
        wins += net > 0
        mk = "o" if s > 0 else "^"
        p.scatter(xs[e], px, marker=mk, s=100, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
        p.annotate(x_["src"], (xs[e], px), xytext=(-6, 10), textcoords="offset points", fontsize=8, color=INK)
        p.scatter(xs[xb], xp, marker=mk, s=100, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
        p.plot([xs[e], xs[xb]], [px, xp], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}% {x_['why']}", (xs[xb], xp), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=BUY if net > 0 else SELL)
        for b, lv, ours, nz, hold in x_["cps"]:
            p.scatter(xs[b], lv, marker="D", s=30, color="#ff7f00" if hold else "#555555", zorder=6)
            p.annotate(f"{ours}/{nz} {'hold' if hold else 'sell'}", (xs[b], lv), xytext=(4, -12),
                       textcoords="offset points", fontsize=7, color="#ff7f00" if hold else "#555555")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close"),
        plt.Line2D([], [], marker="D", ls="", color="#ff7f00", label="checkpoint: hold (votes ours/nonzero)"),
        plt.Line2D([], [], marker="D", ls="", color="#555555", label="checkpoint: sell"),
    ]
    p.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=8)
    p.set_title(f"BTCUSDT futures 15m, last 30 days to {datetime.fromtimestamp(end / 1000, KST):%m-%d}: "
                f"K2+BS entry, {best} exit  |  {wins}/{cnt} wins (KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k_, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 14", r, (30, 70)))):
        a_.plot(xs[lo:], v[lo:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(d / f"t134_{best}_30d.png", dpi=110)
    plt.close(fig)
    print(best, cnt, wins)


if __name__ == "__main__":
    main()
