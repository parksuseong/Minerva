"""토너먼트 차트: 1라운드 순위표, 최종 조합 최근 30일 매매(Stoch·CCI·RSI 칸), 최종 vs P2 자산.

롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 tournament_chart.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tournament as TN
from btc_fractal import cci, rsi, stoch
from matplotlib.lines import Line2D

KST = timezone(timedelta(hours=9))
GREEN, RED = "#1a9850", "#d73027"
WCOL = {"fresh": "#4575b4", "design": "#bbbbbb", "val": "#fdae61", "year": "#d73027"}
P2_TR = Path("~/.minerva/research/ql-entry-freq-2026-11-08/posthoc_trades.json").expanduser()


def kst(ms):
    return datetime.fromtimestamp(ms / 1000, KST).replace(tzinfo=None)


def leaderboard(R, surv, path):
    names = sorted(R["r1"], key=lambda n: R["r1"][n]["design"]["m40"]["ret"])
    fig, ax = plt.subplots(figsize=(11, 13))
    y = np.arange(len(names))
    for j, w in enumerate(("fresh", "design", "val", "year")):
        v = [np.clip(R["r1"][n][w]["m40"]["ret"] * 100, -100, 400) for n in names]
        ax.barh(y + (j - 1.5) * 0.2, v, height=0.2, color=WCOL[w], label=w)
    ax.set_yticks(y)
    ax.set_yticklabels([("★ " if n in surv else "") + n for n in names], fontsize=8)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("5x, 40% margin return % (clipped -100..+400)")
    ax.set_title("Technique tournament round 1 (canonical params) - BTCUSDT, fees+funding, ★ = survived design screen",
                 loc="left", fontsize=10)  # fmt: skip
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def d30_chart(b, tgt, W, title, path):
    a, z = W["d30"]
    a0 = max(a - 200, 0)
    t = b["t"][a0:z]
    o, h, lo, c = (b[k][a0:z] for k in ("o", "h", "l", "c"))
    k, cc, rr = stoch(h, lo, c, 5), cci(h, lo, c, 20), rsi(c, 14)
    s = a - a0
    X = mdates.date2num([kst(v) for v in t[s:]])
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, c[s:], color="#555", lw=0.8)
    T = np.nan_to_num(tgt[a:z])
    prev = 0.0
    for i in range(len(T)):
        cur = T[i]
        if np.sign(cur) != np.sign(prev):
            xi, px = X[i], o[s + i]
            if prev != 0:  # 청산
                lng = prev > 0
                ax[0].scatter(xi, px, marker="o" if lng else "^", s=60, color=RED if lng else GREEN, zorder=5,
                              edgecolor="white")  # fmt: skip
            if cur != 0:  # 진입
                lng = cur > 0
                ax[0].scatter(xi, px, marker="o" if lng else "v", s=60, color=GREEN if lng else RED, zorder=5,
                              edgecolor="white")  # fmt: skip
        prev = cur
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=GREEN, label="long open"),
                          Line2D([], [], marker="o", ls="", color=RED, label="long close"),
                          Line2D([], [], marker="v", ls="", color=RED, label="short open"),
                          Line2D([], [], marker="^", ls="", color=GREEN, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(title, loc="left", fontsize=10)
    for j, (v, nm, lo_, hi_) in enumerate(((k, "Stoch 5", 20, 80), (cc, "CCI 20", -100, 100), (rr, "RSI 14", 30, 70)), 1):
        ax[j].plot(X, v[s:], lw=0.6, color="#2c7bb6")
        ax[j].axhline(lo_, ls="--", lw=0.6, color="#888")
        ax[j].axhline(hi_, ls="--", lw=0.6, color="#888")
        ax[j].set_ylabel(nm)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for q in ax:
        q.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def equity_chart(b, tgt, kw, W, path, label):
    a, z = W["year"]
    fig, ax = plt.subplots(figsize=(14, 6))
    X = mdates.date2num([kst(v) for v in b["t"][a:z]])
    for f, col in ((0.4, "#2c7bb6"), (1.0, "#d73027")):
        r = TN.run(b, tgt, a, z, TN.LEV * f, keep_curve=True, **kw)
        cv = np.array(r["curve"])
        ax.plot(X[: len(cv)], (cv - 1) * 100, color=col, lw=1.4, label=f"{label} 5x, {int(f * 100)}% margin: {r['ret'] * 100:+.0f}%")
    if P2_TR.exists():
        tr = json.loads(P2_TR.read_text())["P2"]["year"]
        eq, xs, ys = 1.0, [X[0]], [0.0]
        for x in tr:
            eq *= 1 + 5 * x["size"] * x["r"]
            xs.append(mdates.date2num(kst(x["t_out"])))
            ys.append((eq - 1) * 100)
        ax.step(xs, ys, where="post", color="#888", lw=1.2, label=f"P2 (approx, gate size only) 5x 100%: {ys[-1]:+.0f}%")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_ylabel("return %")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Last year (2025-11 ~): final tournament combo vs P2", loc="left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).expanduser()
    R = json.loads((out / "result.json").read_text())
    leaderboard(R, set(R["survivors"]), out / "leaderboard.png")
    b = TN.load()
    z = np.load(out / "final_tgt.npz")
    tgt, scale = z["tgt"], z["scale"]
    n = len(tgt)  # 계산 뒤에 붙은 봉은 잘라 신호와 맞춘다
    for k in ("t", "o", "h", "l", "c", "fund", "zP"):
        b[k] = b[k][:n]
    W = TN.windows(b["t"])
    kw = {"scale": scale, "dd_throttle": bool(R["final"].get("dd_throttle"))}
    fin = R["final"]
    fr = R["final_res"][fin["final"]]["d30"]
    title = (f"Final combo {fin['combo']} -> {fin['final']} (survivors {', '.join(R['survivors'])}), last 30 days, "
             f"5x 100% {fr['m100']['ret'] * 100:+.1f}% / 40% {fr['m40']['ret'] * 100:+.1f}%, entries {fr['m40']['entries']}")  # fmt: skip
    d30_chart(b, tgt * np.nan_to_num(scale, nan=1.0), W, title, out / "chart_d30.png")
    equity_chart(b, tgt, kw, W, out / "equity.png", "final")


if __name__ == "__main__":
    main()
