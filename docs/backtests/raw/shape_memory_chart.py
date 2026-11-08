"""모양 기억 차트: 최근 30일 매매, 지금 모양과 닮은 과거 10개, 구간별 자산 곡선.

실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 shape_memory_chart.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scr15_timing as X
import shape_memory as SM
from btc_fractal import cci, rsi, stoch
from matplotlib.lines import Line2D

G, R = "#1a9850", "#d73027"


def dt(ms) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, X.KST).replace(tzinfo=None)


def load_rule(out: Path, key: str, b: dict):
    name, ks, qs, fee = key.split("|")
    d = np.load(out / f"votes_{name}.npz")
    a0 = int(np.searchsorted(d["t"], b["t"][0]))
    m = min(len(d["t"]) - a0, len(b["t"]))
    v = {k: np.full(len(b["t"]), np.nan, np.float32) for k in d.files if k != "t"}
    for k in v:
        v[k][:m] = d[k][a0:a0 + m]
    L, S = SM.signals(v, int(ks[1:]), int(qs[1:]) / 100, slice(None))
    return v, L, S, int(ks[1:]), fee == "M"


def chart_trades(out: Path, key: str, b, F, A, W) -> Path:
    v, L, S, k, maker = load_rule(out, key, b)
    a, z = W["d30"]
    tr, _ = SM.sim(b, F, A, L, S, a, z, maker=maker)
    s = SM.summ(tr, {}, 30)
    h, l, c = b["h"], b["l"], b["c"]
    X_ = mdates.date2num([dt(x) for x in b["t"][a:z]])
    fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1]})
    ax[0].plot(X_, c[a:z], color="#555", lw=0.8)
    for x in tr:
        lg = x["d"] == 1
        xe, xx = mdates.date2num(dt(b["t"][x["e"]])), mdates.date2num(dt(b["t"][x["x"]]))
        ax[0].scatter(xe, x["ep"], marker="o" if lg else "v", s=60, color=G if lg else R, zorder=5, edgecolor="white")
        ax[0].scatter(xx, x["xp"], marker="o" if lg else "^", s=60, color=R if lg else G, zorder=5, edgecolor="white")
        ax[0].annotate(f"{5 * x['r'] * 100:+.1f}%", (xx, x["xp"]), fontsize=6, color=G if x["r"] > 0 else R,
                       xytext=(2, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(
        f"Shape memory {key} - BTCUSDT perp 15m, last 30 days {dt(b['t'][a]):%m-%d} ~ {dt(b['t'][z - 1]):%m-%d %H:%M} KST\n"
        f"{s['n']} trades ({s['per_day']:.1f}/day), win {s['win'] or 0:.0%}, 5x {s['x5']['ret']:+.1%} "
        f"(max DD {s['x5']['mdd']:.0%}), 1x {s['x1']['ret']:+.1%}. TP +1%, stop 1xATR, 2h time exit. "
        "Labels = 5x net per trade (fees + funding)", loc="left", fontsize=10)  # fmt: skip
    kk, cc, rr = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    for i_, (y, nm, lo, hi) in enumerate(((kk, "Stoch 5", 20, 80), (cc, "CCI 20", -100, 100), (rr, "RSI 14", 30, 70)), 1):
        ax[i_].plot(X_, y[a:z], lw=0.6, color="#2c7bb6")
        for lv in (lo, hi):
            ax[i_].axhline(lv, ls="--", lw=0.6, color="#888")
        ax[i_].set_ylabel(nm)
    q = int(key.split("|")[2][1:]) / 100
    ax[4].plot(X_, v[f"p{k}"][a:z], lw=0.6, color="#6a3d9a")
    ax[4].axhline(q, ls="--", lw=0.6, color=G)
    ax[4].axhline(1 - q, ls="--", lw=0.6, color=R)
    ax[4].set_ylabel("neighbours up %")
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for z_ in ax:
        z_.grid(alpha=0.2)
    plt.tight_layout()
    p = out / "chart_trades_d30.png"
    plt.savefig(p, dpi=100)
    plt.close()
    return p


def chart_matches(out: Path, name: str) -> Path:
    _, s, _ = SM.load_all()
    A = SM.atr(s["h"], s["l"], s["c"])
    V = SM.shape_vecs(name, s, A)
    n_path = {"N16": 16, "N32": 32, "N64": 64}.get(name, 32)
    c = s["c"]
    i = len(c) - 1
    lib = np.arange(n_path, i - SM.H - 1, SM.LIB_STRIDE)
    lib = lib[np.linalg.norm(V[lib], axis=1) > 0.5]
    sims = V[lib] @ V[i]
    order = lib[np.argsort(-sims)]
    picks: list = []
    for j in order:  # 서로 겹치지 않는 10개
        if all(abs(j - p_) > n_path for p_ in picks):
            picks.append(j)
        if len(picks) == 10:
            break
    fig, axs = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    xs = np.arange(-n_path + 1, SM.H + 1)
    ups = 0
    for ax, j in zip(axs.ravel(), picks):
        seg = c[j - n_path + 1:j + SM.H + 1] / c[j] * 100 - 100
        now = c[i - n_path + 1:i + 1] / c[i] * 100 - 100
        fut = seg[n_path - 1:]
        up = fut[-1] > 0
        ups += up
        ax.plot(xs[:n_path], now, color="#999", lw=1.0, label="now")
        ax.plot(xs[:n_path], seg[:n_path], color="#2c7bb6", lw=1.2, label="past match")
        ax.plot(xs[n_path - 1:], fut, color=G if up else R, lw=2.0, label="what followed (2h)")
        ax.axvline(0, color="#888", lw=0.6, ls="--")
        ax.set_title(f"{dt(s['t'][j]):%Y-%m-%d %H:%M}  corr {float(V[j] @ V[i]):.2f}  next {fut[-1]:+.2f}%", fontsize=8)
        ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Now ({dt(s['t'][i]):%m-%d %H:%M} KST, last {n_path} x 15m bars, % from last close) vs 10 most similar "
                 f"past shapes ({name}). After them: {ups} up / {10 - ups} down", fontsize=11)  # fmt: skip
    plt.tight_layout()
    p = out / "chart_matches_now.png"
    plt.savefig(p, dpi=100)
    plt.close()
    return p


def chart_equity(out: Path, keys: list, b, F, A, W) -> Path:
    wins = ("fresh", "mid", "y1", "m6")
    lab = {"fresh": "2020-03~2021-10", "mid": "2022-01~2025-10", "y1": "2025-11~now", "m6": "last 6 months"}
    fig, axs = plt.subplots(1, len(wins), figsize=(18, 4.5))
    for ax, w in zip(axs, wins):
        a, z = W[w]
        for key, col in zip(keys, ("#2c7bb6", "#fdae61", "#888")):
            _, L, S, _, maker = load_rule(out, key, b)
            tr, _ = SM.sim(b, F, A, L, S, a, z, maker=maker)
            eq, xs, ys = 1.0, [dt(b["t"][a])], [0.0]
            for x in tr:
                eq = 0.0 if 5 * x["worst"] <= -0.95 else eq * (1 + 5 * x["r"])
                xs.append(dt(b["t"][x["x"]]))
                ys.append((eq - 1) * 100)
            ax.step(xs, ys, where="post", color=col, lw=1.3, label=f"{key}: {ys[-1]:+.0f}%")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(lab[w], fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    axs[0].set_ylabel("5x return %")
    fig.suptitle("Shape memory - 5x equity by window (fees + funding; memory only from the past)", fontsize=11)
    plt.tight_layout()
    p = out / "equity.png"
    plt.savefig(p, dpi=100)
    plt.close()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).expanduser()
    r = json.loads((out / "result.json").read_text())
    b = X.load()
    F = X.features(b)
    A = SM.atr(b["h"], b["l"], b["c"])
    W = SM.windows(b["t"])
    keys = list(dict.fromkeys(k for k in (r["best"], r["best_busy"]) if k))
    print(chart_trades(out, r["best_busy"] or r["best"], b, F, A, W))
    print(chart_equity(out, keys, b, F, A, W))
    print(chart_matches(out, (r["best_busy"] or r["best"]).split("|")[0]))


if __name__ == "__main__":
    main()
