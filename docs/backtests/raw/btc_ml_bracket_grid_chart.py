"""130번 차트: 규칙별 TP×SL 격자의 2024~ CAGR/|MDD| 열지도, 고른 칸의 최근 30일 거래.

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_bracket_grid_chart.py \
    <result.json> <groups.json> <칸 키, 예: B|TP2/SL1.5> <출력 폴더>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from btc_fractal import cci, rsi, stoch  # noqa: E402
from btc_ml_bracket_grid import RULES, SLS, TPS, cell  # noqa: E402
from btc_ml_futures_reprice import futures_on  # noqa: E402
from btc_strategy_combo import RES  # noqa: E402

KST = timezone(timedelta(hours=9))
BUY, SELL, INK, GRID = "#1a9850", "#d73027", "#333333", "#e6e6e6"


def heatmap(rep: dict, out: Path) -> None:
    fig, axs = plt.subplots(1, len(RULES), figsize=(4 * len(RULES), 4), sharey=True)
    for ax, r in zip(axs, RULES):
        m = np.full((len(TPS), len(SLS)), np.nan)
        for a, tp in enumerate(TPS):
            for b, sl in enumerate(SLS):
                v = rep[f"{r}|{cell(tp, sl)}"]["2024~ 1x"].get("ratio")
                m[a, b] = np.nan if v is None else v
        im = ax.imshow(m, cmap="RdYlGn", vmin=-1, vmax=1, origin="lower")
        for a in range(len(TPS)):
            for b in range(len(SLS)):
                key = f"{r}|{cell(TPS[a], SLS[b])}"
                mark = "*" if key in rep.get("passed", []) else ""
                base = "□" if (TPS[a], SLS[b]) == (0.015, "1.5") else ""
                txt = "nan" if np.isnan(m[a, b]) else f"{m[a, b]:.2f}"
                ax.text(b, a, f"{txt}{mark}{base}", ha="center", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(SLS)), [f"SL {s}" for s in SLS])
        ax.set_yticks(range(len(TPS)), [f"TP {t * 100:g}%" for t in TPS])
        ax.set_title(r, color=INK)
    fig.colorbar(im, ax=axs, shrink=0.8, label="2024~ CAGR/|MDD| (1x)")
    fig.suptitle("Test 130: ML bracket grid, realistic fills (5-min delay + 0.05% through). □ baseline, * pass",
                 color=INK)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def trades_chart(groups: list, key: str, out: Path) -> None:
    z = np.load(RES / "ml15/pred.npz")
    t = z["t"]
    P, _ = futures_on(t, {q: z[q] for q in "ohlc"})  # 체결과 같은 선물 가격
    h, l, c = P["h"], P["l"], P["c"]  # noqa: E741
    end = t[-1]
    lo = int(np.searchsorted(t, end - 30 * 86_400_000))
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t])
    k, cc, r = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], c[lo:], color=INK, lw=0.7)
    wins = n = 0
    for e, s, px, b, xp in groups:
        if e < lo:
            continue
        net = s * (xp / px - 1) - 0.0007
        n += 1
        wins += net > 0
        mk = "o" if s > 0 else "^"
        p.scatter(x[e], px, marker=mk, s=80, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
        p.scatter(x[b], xp, marker=mk, s=80, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
        p.plot([x[e], x[b]], [px, xp], color="#777777", lw=1.0, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[b], xp), xytext=(4, 6) if net > 0 else (-8, -14),
                   textcoords="offset points", fontsize=8, color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last 30 days to predictions end: {key}  |  {wins}/{n} wins "
                "(approx. net of fees, futures prices, KST)", loc="left", color=INK, fontsize=11)
    for a_, (name, v, lines) in zip(ax[1:], (("Stoch 5", k, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 14", r, (30, 70)))):
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
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(key, n, "trades", wins, "wins")


def main() -> None:
    rep = json.loads(Path(sys.argv[1]).read_text())
    groups = json.loads(Path(sys.argv[2]).read_text())
    key, outdir = sys.argv[3], Path(sys.argv[4])
    heatmap(rep, outdir / "t130_heatmap.png")
    trades_chart(groups[key], key, outdir / f"t130_{key.replace('|', '_').replace('/', '_')}.png")


if __name__ == "__main__":
    main()
