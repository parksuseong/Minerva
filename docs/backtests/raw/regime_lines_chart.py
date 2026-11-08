"""큰 구간 선 차트: 구간 선(상승 초록·하락 빨강·횡보 회색), 지금 다리 순서 vs 닮은 과거 10개, 최근 30일 매매, 자산 곡선·보정.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 regime_lines_chart.py --out <폴더> 변형1 변형2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import regime_lines as RL
from matplotlib.lines import Line2D

G, R, S = "#1a9850", "#d73027", "#9e9e9e"
COL = {1: G, -1: R, 0: S}


def kst(ms):
    return datetime.fromtimestamp(int(ms) / 1000, DW.KST).replace(tzinfo=None)


def lines_chart(s, path):
    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    X = mdates.date2num([kst(v) for v in s["t"]])
    for ax, name in zip(axs, ("Z8", "L3")):
        E = RL.events(s, name)
        ax.plot(X, s["c"], color="#ccc", lw=0.6)
        P = E["P"]
        for e in range(1, len(P)):
            ax.plot([X[P[e - 1]], X[P[e]]], [s["c"][P[e - 1]], s["c"][P[e]]], color=COL[int(E["lab"][e])], lw=1.6)
        ax.plot([X[P[-1]], X[-1]], [s["c"][P[-1]], s["c"][-1]], color="#2c7bb6", lw=1.6, ls="--")
        c = RL.count_regimes(s)[name]
        ax.set_yscale("log")
        ax.set_title(f"{name}: {c['legs']} legs since 2017-09 - up {c['count_share']['up']:.0%} / down "
                     f"{c['count_share']['down']:.0%} / sideways {c['count_share']['side']:.0%} by count "
                     f"(time: {c['time_share']['up']:.0%} / {c['time_share']['down']:.0%} / {c['time_share']['side']:.0%}), "
                     f"median leg {c['median_days']:.1f} days", loc="left", fontsize=10)  # fmt: skip
        ax.grid(alpha=0.2)
    axs[0].legend(handles=[Line2D([], [], color=G, label="up"), Line2D([], [], color=R, label="down"),
                           Line2D([], [], color=S, label="sideways"),
                           Line2D([], [], color="#2c7bb6", ls="--", label="in progress")], loc="upper left")  # fmt: skip
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(path, dpi=95)
    plt.close(fig)


def seq_xy(E, x, e, k, nxt: bool):
    """다리 순서를 같은 간격(봉 수 무시)으로, 높이는 k다리 평균 높이 단위로."""
    P = E["P"]
    pts = [x[P[j]] for j in range(e - k, e + 1)] + [x[E["C"][e]]]
    sc = np.mean(np.abs(np.diff(pts[:-1])))
    y = (np.array(pts) - pts[0]) / sc
    xs = list(range(k + 1)) + [k + 0.5]
    out_y = None
    if nxt:
        m_ = e + 1
        while m_ < len(P) and P[m_] <= E["C"][e]:
            m_ += 1
        if m_ < len(P):
            out_y = (x[P[m_]] - pts[0]) / sc
    return xs, y, out_y


def matches_chart(s, name, k, sim, path):
    Rr = RL.predict(s, name, k, sim)
    E, pr = Rr["E"], Rr["pr"]
    x = s["x"]
    e = len(E["P"]) - 1
    while e > 0 and pr["nb"][e] is None:
        e -= 1
    fig, axs = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    qx, qy, _ = seq_xy(E, x, e, k, False)
    nb = pr["nb"][e][:10]
    for ax, j in zip(axs.ravel(), nb):
        jx, jy, jo = seq_xy(E, x, j, k, True)
        ax.plot(qx, qy, color="#2c7bb6", lw=2.2, label="now")
        ax.plot(jx, jy, color="#555", lw=1.2, ls="-", label="past match")
        if jo is not None:
            ax.plot([jx[-1], jx[-1] + 1], [jy[-1], jo], color=G if jo > jy[-1] else R, lw=2, ls="--")
        ax.set_title(f"{kst(s['t'][E['C'][j]]):%Y-%m-%d} -> {np.expm1(E['g'][j]):+.1%}", fontsize=9)
        ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{name}|k{k}|{sim}: current leg sequence (blue, confirmed {kst(s['t'][E['C'][e]]):%m-%d %H:%M} KST) vs 10 most similar past "
                 f"sequences (gray) and what came next (dashed: green up / red down). Up share {pr['p'][e]:.0%}, "
                 f"median next {np.expm1(pr['med'][e]):+.1%}.\nX = leg number (bar count ignored), Y = mean leg height units",
                 fontsize=10, x=0.01, ha="left")  # fmt: skip
    plt.tight_layout()
    plt.savefig(path, dpi=95)
    plt.close(fig)


def variant_trades(b, F, C, s, vn, a, z):
    kind, name, kk, last = vn.split("|")
    k = int(kk[1:])
    sim = last if kind == "R" else "E"
    Rr = RL.predict(s, name, k, sim)
    cands, gate = RL.regime_cands(b, s, Rr)
    if kind == "R":
        return DW.simulate(b, F, cands, "G0", "FA", a, z)
    F["gate"]["RG"] = gate
    return DW.simulate(b, F, C[last], "RG", "FA", a, z)


def trades_chart(b, F, C, s, vn, a, z, path):
    t = b["t"]
    tr, cnt = variant_trades(b, F, C, s, vn, a, z)
    sm = DW.summ(tr, a, z, b, cnt)
    X = mdates.date2num([kst(t[i]) for i in range(a, z)])
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, b["c"][a:z], color="#555", lw=0.8)
    for x in tr:
        lg = x["d"] == 1
        xe, xx = mdates.date2num(kst(t[x["e"]])), mdates.date2num(kst(t[x["x"]]))
        ax[0].scatter(xe, x["ep"], marker="o" if lg else "v", s=60, color=G if lg else R, zorder=5, edgecolor="white")
        ax[0].scatter(xx, x["xp"], marker="o" if lg else "^", s=60, color=R if lg else G, zorder=5, edgecolor="white")
        ax[0].annotate(f"{DW.LEV * x['r'] * 100:+.1f}%", (xx, x["xp"]), fontsize=7,
                       color=G if x["r"] > 0 else R, xytext=(3, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")], loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(f"{vn} - BTCUSDT perp 15m, last 30 days {kst(t[a]):%m-%d} ~ {kst(t[z - 1]):%m-%d %H:%M} KST, market orders\n"
                    f"{sm['n']} trades ({sm['per_day']:.2f}/day), win {(sm['win'] or 0):.0%}, 5x {sm['x5']['ret']:+.1%} "
                    f"(max drawdown {sm['x5']['mdd']:.0%}). Labels = 5x net per trade (fees + funding)", loc="left", fontsize=10)  # fmt: skip
    for k, (y, nm, lo, hi) in enumerate(((F["stoch"], "Stoch 5", 20, 80), (F["cci"], "CCI 20", -100, 100),
                                         (F["rsi"], "RSI 14", 30, 70)), 1):  # fmt: skip
        ax[k].plot(X, y[a:z], lw=0.6, color="#2c7bb6")
        ax[k].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[k].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[k].set_ylabel(nm)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for q in ax:
        q.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def equity_chart(b, F, C, s, names, W, res, calib, path):
    t = b["t"]
    fig, ax = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    a, z = W["fresh"][0], len(t)
    for vn, col in zip(names, ("#2c7bb6", "#fdae61")):
        tr, _ = variant_trades(b, F, C, s, vn, a, z)
        eq, xs, ys = 1.0, [kst(t[a])], [0.0]
        for x in tr:
            eq *= 1 + DW.LEV * x["r"]
            eq = max(eq, 0)
            xs.append(kst(t[x["x"]]))
            ys.append(np.log10(max(eq, 1e-4)))
        r = res[vn]
        lab = ", ".join(f"{w} {r[w]['x5']['ret']:+.0%}" for w in RL.MAIN)
        ax[0].step(xs, ys, where="post", color=col, lw=1.5, label=f"{vn}: {lab}")
    for w in ("design", "val", "year"):
        ax[0].axvline(kst(t[W[w][0]]), color="#999", lw=0.6, ls=":")
    ax[0].set_ylabel("5x equity, log10 (0 = start)")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("Equity 2020-03 ~ now, 5x, market orders, fees + funding (dotted = window starts)", loc="left", fontsize=10)
    ax[0].grid(alpha=0.2)
    cal = np.array([(p, y) for vn, p, y in calib if vn.split("|")[0].startswith("L")])
    calz = np.array([(p, y) for vn, p, y in calib if vn.split("|")[0].startswith("Z")])
    for arr, lab, col in ((cal, "line-fit segments (L)", "#2c7bb6"), (calz, "zigzag (Z)", "#fdae61")):
        if len(arr):
            bins = np.linspace(0, 1, 6)
            idx = np.clip(np.digitize(arr[:, 0], bins) - 1, 0, 4)
            mx = [arr[idx == i, 0].mean() for i in range(5) if (idx == i).sum() >= 20]
            my = [arr[idx == i, 1].mean() for i in range(5) if (idx == i).sum() >= 20]
            ax[1].plot(mx, my, "o-", color=col, label=lab)
    ax[1].plot([0, 1], [0, 1], color="#999", ls="--", lw=0.8)
    ax[1].set_xlabel("matches' up share (predicted)")
    ax[1].set_ylabel("actually went up")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Calibration 2020-03 ~ 2025-10 (events)", loc="left", fontsize=10)
    ax[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=95)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("names", nargs=2)
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b, s = RL.load_all(DW.R5)
    F = DW.features(b)
    C = DW.all_cands(b, F)
    W = RL.windows(b["t"])
    res = json.loads((out / "result.json").read_text())
    lines_chart(s, out / "chart_regime_lines.png")
    matches_chart(s, "L1.5", 3, "E", out / "chart_matches_now.png")
    a, z = W["d30"]
    trades_chart(b, F, C, s, a_.names[0], a, z, out / "chart_trades_d30.png")
    equity_chart(b, F, C, s, a_.names, W, res["res"], res["calib"], out / "equity.png")
    print("ok")


if __name__ == "__main__":
    main()
