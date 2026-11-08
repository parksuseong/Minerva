"""모양 기억 2 차트: 큰 봉·시간 늘림 닮은 모양, 보정 곡선, 최근 30일 매매, 자산 곡선.

실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 shape_memory2_chart.py --out <폴더>
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
import shape_memory2 as S2
from btc_fractal import cci, rsi, stoch
from matplotlib.lines import Line2D
from pine_eval import lev_path

G, R = "#1a9850", "#d73027"


def dt(ms) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, X.KST).replace(tzinfo=None)


def top_matches(V, i, lib, ready, n_path, k=10):
    ok = lib[ready < i]
    sims = V[ok] @ V[i]
    picks: list = []
    for j in ok[np.argsort(-sims)]:
        if all(abs(j - p_) > n_path for p_ in picks):
            picks.append(j)
        if len(picks) == k:
            break
    return picks


def chart_matches(out: Path) -> Path:
    _, s, _ = SM.load_all()
    fig, axs = plt.subplots(2, 2, figsize=(17, 10))
    for ax, (fr, (fac, hz)) in zip(axs.ravel()[:3], S2.FRAMES.items()):
        F_ = S2.frame_bars(s, fac)
        c = F_["c"]
        V = SM.path_vecs(c, 32)
        i = len(c) - 1
        lib = np.arange(32, len(c) - hz)
        picks = top_matches(V, i, lib, lib + hz, 32)
        xs = np.arange(-31, hz + 1)
        ups = 0
        for j in picks:
            seg = c[j - 31:j + hz + 1] / c[j] * 100 - 100
            up = seg[-1] > 0
            ups += up
            ax.plot(xs[:32], seg[:32], color="#2c7bb6", lw=0.7, alpha=0.6)
            ax.plot(xs[31:], seg[31:], color=G if up else R, lw=1.2, alpha=0.8)
        now = c[i - 31:i + 1] / c[i] * 100 - 100
        ax.plot(xs[:32], now, color="black", lw=2.2)
        ax.axvline(0, color="#888", ls="--", lw=0.7)
        ax.set_title(f"{fr} bars: now ({dt(F_['t'][i]):%m-%d %H:%M} KST bar) vs 10 closest past shapes\n"
                     f"what followed next {hz} bars: {ups} up / {len(picks) - ups} down", fontsize=10, loc="left")  # fmt: skip
        ax.set_ylabel("% vs last close")
        ax.grid(alpha=0.2)
    ax = axs.ravel()[3]
    logc = np.log(s["c"]).astype(np.float64)
    i = len(logc) - 1
    qv = S2.tw_vecs(logc, np.array([i]), 96)[0]
    best: list = []
    for W in S2.TW_SCALES:
        h = W // 4
        idx = np.arange(W - 1, i - h - 1, S2.TW_LIB_STRIDE)
        V = S2.tw_vecs(logc, idx, W)
        sims = V @ qv
        top = np.argsort(-sims)[:300]
        best += [(float(sims[t]), int(idx[t]), W, V[t]) for t in top]
    best.sort(key=lambda x: -x[0])
    cand = best[:200]
    d = S2.dtw_dist(qv[None].astype(np.float32), np.stack([x[3] for x in cand])[None].astype(np.float32))[0]
    order = np.argsort(d)
    picks: list = []
    for o in order:
        _, j, W, _v = cand[o]
        if all(abs(j - p_[0]) > W for p_ in picks):
            picks.append((j, W))
        if len(picks) == 10:
            break
    ups = 0
    xs_p = np.linspace(-1, 0, S2.TW_PTS)
    for j, W in picks:
        h = W // 4
        path = logc[j - W + 1:j + 1]
        fut = logc[j:j + h + 1]
        rng = path.max() - path.min()
        y = (np.interp(np.linspace(0, W - 1, S2.TW_PTS), np.arange(W), path) - logc[j]) / rng
        yf = (fut - logc[j]) / rng
        up = yf[-1] > 0
        ups += up
        ax.plot(xs_p, y, color="#2c7bb6", lw=0.7, alpha=0.6)
        ax.plot(np.linspace(0, 0.25, len(yf)), yf, color=G if up else R, lw=1.2, alpha=0.8)
        ax.annotate(f"{W * 15 / 60:.0f}h", (0.25, yf[-1]), fontsize=7, color="#555")
    path = logc[i - 95:i + 1]
    rng = path.max() - path.min()
    ax.plot(xs_p, (np.interp(np.linspace(0, 95, S2.TW_PTS), np.arange(96), path) - logc[i]) / rng, color="black", lw=2.2)
    ax.axvline(0, color="#888", ls="--", lw=0.7)
    ax.set_title(f"time-warp (DTW): last 1 day vs 10 closest shapes of ANY length (4h~5d, label = length)\n"
                 f"x = share of the shape's own length; what followed (1/4 of length): {ups} up / {len(picks) - ups} down",
                 fontsize=10, loc="left")  # fmt: skip
    ax.set_ylabel("move / shape range")
    ax.grid(alpha=0.2)
    fig.legend(handles=[Line2D([], [], color="black", lw=2, label="now"),
                        Line2D([], [], color="#2c7bb6", label="past match"),
                        Line2D([], [], color=G, label="followed: up"), Line2D([], [], color=R, label="followed: down")],
               loc="lower center", ncol=4, fontsize=9)  # fmt: skip
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    p = out / "chart_matches_frames.png"
    plt.savefig(p, dpi=95)
    plt.close()
    return p


def chart_calibration(out: Path, cal: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0.2, 0.8], [0.2, 0.8], color="#999", ls="--", lw=1, label="perfect (says 70% → 70% go up)")
    cols = ["#999999", "#2c7bb6", "#1a9850", "#d73027", "#6a3d9a", "#ff7f00", "#a6761d", "#e7298a"]
    for col, (name, d) in zip(cols, cal.items()):
        pts = [(r["pred"], r["up"]) for r in d["bins"] if r["n"] >= 30 and r["pred"] is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=col, lw=1.4,
                    label=f"{name}: hit {d['hit_confident'] or 0:.1%} when ≥60/40 (n={d['n_confident']})")  # fmt: skip
    ax.set_xlabel("neighbours say: chance of going up")
    ax.set_ylabel("actually went up (2020-03 ~ 2025-10)")
    ax.set_title("Does the shape memory know the direction? (flat line = no)", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    p = out / "calibration.png"
    plt.savefig(p, dpi=100)
    plt.close()
    return p


def chart_trades(out: Path, key: str, b, F, A, W, save) -> Path:
    name, qs, fee = key.split("|")
    L, S, Hs = S2.rule_signals(save, b, name, int(qs[1:]) / 100)
    a, z = W["d30"]
    tr, _ = S2.sim2(b, F, A, L, S, Hs, a, z, maker=fee == "M")
    s = SM.summ(tr, {}, 30)
    h, l, c = b["h"], b["l"], b["c"]
    X_ = mdates.date2num([dt(x) for x in b["t"][a:z]])
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
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
        f"Shape memory 2 {key} - BTCUSDT perp, last 30 days {dt(b['t'][a]):%m-%d} ~ {dt(b['t'][z - 1]):%m-%d %H:%M} KST\n"
        f"{s['n']} trades ({s['per_day']:.1f}/day), win {s['win'] or 0:.0%}, 5x {s['x5']['ret']:+.1%} "
        f"(max DD {s['x5']['mdd']:.0%}), 1x {s['x1']['ret']:+.1%}. TP = stop = ATR15 x sqrt(horizon), horizon exit. "
        "Labels = 5x net per trade (fees + funding)", loc="left", fontsize=10)  # fmt: skip
    kk, cc, rr = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    for i_, (y, nm, lo, hi) in enumerate(((kk, "Stoch 5", 20, 80), (cc, "CCI 20", -100, 100), (rr, "RSI 14", 30, 70)), 1):
        ax[i_].plot(X_, y[a:z], lw=0.6, color="#2c7bb6")
        for lv in (lo, hi):
            ax[i_].axhline(lv, ls="--", lw=0.6, color="#888")
        ax[i_].set_ylabel(nm)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for z_ in ax:
        z_.grid(alpha=0.2)
    plt.tight_layout()
    p = out / f"chart_trades_d30_{name}.png"
    plt.savefig(p, dpi=95)
    plt.close()
    return p


def chart_equity(out: Path, keys: list, b, F, A, W, save) -> Path:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    cols = ["#2c7bb6", "#d73027", "#1a9850", "#6a3d9a"]
    for ax, w, lab in zip(axs, ("fresh", "mid", "y1"), ("2020-03~2021-10", "2022-01~2025-10", "last year 2025-11~")):
        a, z = W[w]
        for col, key in zip(cols, keys):
            name, qs, fee = key.split("|")
            L, S, Hs = S2.rule_signals(save, b, name, int(qs[1:]) / 100)
            tr, _ = S2.sim2(b, F, A, L, S, Hs, a, z, maker=fee == "M")
            eq, xs, ys = 1.0, [dt(b["t"][a])], [0.0]
            for x in tr:
                eq = max(eq * (1 + 5 * x["r"]), 0)
                xs.append(dt(b["t"][x["x"]]))
                ys.append((eq - 1) * 100)
            ax.step(xs, ys, where="post", color=col, lw=1.3, label=f"{key}: {ys[-1]:+.0f}% ({len(tr)} tr)")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(lab, loc="left", fontsize=10)
        ax.set_ylabel("5x return %")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    p = out / "equity.png"
    plt.savefig(p, dpi=95)
    plt.close()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).expanduser()
    res = json.loads((out / "result.json").read_text())
    b = X.load()
    F = X.features(b)
    A = SM.atr(b["h"], b["l"], b["c"])
    W = SM.windows(b["t"])
    save = dict(np.load(out / "votes2.npz"))
    keys = list(dict.fromkeys(k for k in (res["best"], res["best_busy"], res["best_tw"], res["best_hf"]) if k))
    print(chart_calibration(out, res["calibration"]))
    print(chart_equity(out, keys, b, F, A, W, save))
    print(chart_trades(out, res["best_busy"] or res["best"], b, F, A, W, save))
    print(chart_matches(out))
    _ = lev_path


if __name__ == "__main__":
    main()
