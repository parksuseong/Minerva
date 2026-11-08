"""상위 프레임 후보 차트: 최근 6개월·30일 매매(가격 + Stoch·CCI·RSI)와 전 구간 5배 자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 htf_live_chart.py --out <폴더> 이름 ...
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from btc_fractal import cci, rsi, stoch
from htf_live import DAY, LEV, M15, R5, WIN, all_windows, build, load, ms, run_variant
from matplotlib.lines import Line2D

KST = timezone(timedelta(hours=9))
G, R = "#1a9850", "#d73027"


def kdt(t: int) -> datetime:
    return datetime.fromtimestamp(int(t) / 1000, KST).replace(tzinfo=None)


def trade_chart(b, tr, key, a, z, title, path, step) -> None:
    t, c = b["t"], b["c"]
    sl = slice(a, z, step)
    X = mdates.date2num([kdt(v) for v in t[sl]])
    h4, l4, c4 = b["h"], b["l"], c
    k, cc, rr = stoch(h4, l4, c4, 5), cci(h4, l4, c4, 20), rsi(c4, 14)
    fig, ax = plt.subplots(4, 1, figsize=(15, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, c[sl], color="#555", lw=0.8)
    sub = [x for x in tr if a <= x["e"] < z]
    for x in sub:
        long_ = x["d"] == 1
        ax[0].scatter(mdates.date2num(kdt(t[x["e"]])), x["ep"], marker="o" if long_ else "v", s=80,
                      color=G if long_ else R, zorder=5, edgecolor="white")  # fmt: skip
        xi = min(x["x"], len(t) - 1)
        ax[0].scatter(mdates.date2num(kdt(t[xi])), x["xp"], marker="o" if long_ else "^", s=80,
                      color=R if long_ else G, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].annotate(f"{LEV * x['r'] * 100:+.1f}%", (mdates.date2num(kdt(t[xi])), x["xp"]), fontsize=8,
                       color=G if x["r"] > 0 else R, xytext=(3, 6), textcoords="offset points")  # fmt: skip
    r = np.array([x["r"] for x in sub]) if sub else np.zeros(0)
    eq = np.prod(1 + LEV * r) - 1 if len(r) else 0.0
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper left", fontsize=8)  # fmt: skip
    ax[0].set_title(f"{key} - {title} ({kdt(t[a]):%Y-%m-%d} ~ {kdt(t[z - 1]):%m-%d %H:%M} KST)\n"
                    f"{len(sub)} trades, win {int((r > 0).sum())}/{len(sub)}, 5x {eq:+.1%}; "
                    "fees 0.05%/side + funding, entry at next 15m open after the signal bar closes. "
                    "Labels = 5x net per trade", fontsize=10, loc="left")  # fmt: skip
    for i, (y, nm, lo_, hi_) in enumerate(((k, "Stoch 5 (15m)", 20, 80), (cc, "CCI 20 (15m)", -100, 100),
                                           (rr, "RSI 14 (15m)", 30, 70)), 1):  # fmt: skip
        ax[i].plot(X, y[sl], lw=0.6, color="#2c7bb6")
        ax[i].axhline(lo_, ls="--", lw=0.6, color="#888")
        ax[i].axhline(hi_, ls="--", lw=0.6, color="#888")
        ax[i].set_ylabel(nm, fontsize=8)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for q in ax:
        q.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def equity_chart(b, trs: dict, path) -> None:
    t = b["t"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for (key, tr), col in zip(trs.items(), ("#2c7bb6", "#fdae61", "#888")):
        eq, xs, ys = 1.0, [kdt(ms("2020-03-01"))], [0.0]
        for x in tr:
            eq *= 1 + LEV * x["r"]
            xs.append(kdt(t[min(x["x"], len(t) - 1)]))
            ys.append(eq)
        ys[0] = 1.0
        ax.step(xs, ys, where="post", color=col, lw=1.6, label=f"{key}: x{eq:.2f}")
    for w, (a_, z_) in WIN.items():
        ax.axvline(kdt(ms(a_)), color="#bbb", lw=0.8, ls="--")
        ax.text(kdt(ms(a_)), ax.get_ylim()[1] if False else 1.0, f" {w}", fontsize=8, color="#666")
    ax.set_yscale("log")
    ax.set_ylabel("5x equity (log, start = 1)")
    ax.set_title("5x equity 2020-03 ~ 2026-11 (windows: fresh 2020-03, design 2022-01, val 2024-07, "
                 "seen 2025-11)", loc="left", fontsize=10)  # fmt: skip
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("keys", nargs="+")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b = load(R5)
    S = build(b)
    W = all_windows(b["t"])
    a0 = W["fresh"][0]
    trs = {}
    for key in a_.keys:
        name, g, ex = key.split("|")
        tr = run_variant(b, S, name, g, ex, a0, len(b["t"]))[0]
        trs[key] = tr
        tag = key.replace("|", "_")
        trade_chart(b, tr, key, *W["m6"], "last 6 months", out / f"chart_{tag}_m6.png", 4)
        trade_chart(b, tr, key, *W["d30"], "last 30 days", out / f"chart_{tag}_d30.png", 1)
    equity_chart(b, trs, out / "equity.png")
    print("ok", out, flush=True)


if __name__ == "__main__":
    main()
