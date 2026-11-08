"""큰 구간 선 2 차트: 최근 30일 매매(목표가 선·정리 이유), 거래 하나의 p(t), 2020-03 ~ 지금 자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익, 글자는 청산 이유.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 regime_lines2_chart.py --out <폴더> 변형1 변형2
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
import regime_lines2 as G2
from matplotlib.lines import Line2D

G, R = "#1a9850", "#d73027"
WHY = {"tp": "TP", "bail": "bail p<0.3", "time": "time", "struct": "leg ended", "stop": "stop 6%", "end": "open"}


def kst(ms):
    return datetime.fromtimestamp(int(ms) / 1000, DW.KST).replace(tzinfo=None)


def build(b, F, C, s, vn):
    kind, name, tp = vn.split("|")
    S = G2.ev_stats(s, name)
    if kind in ("RB", "RN", "RU"):
        D = G2.decisions(s, S, tp, "U" if kind == "RU" else "E")
        return lambda a, z: G2.simulate(b, F, s, S, D, a, z, bail=kind != "RN")
    D = G2.decisions(s, S, "EV", "E")
    key = f"G2c_{name}"
    F["gate"][key] = G2.gate_array(b, s, S, D, len(b["t"]))
    return lambda a, z: DW.simulate(b, F, C[tp], key, "FA", a, z)[0]


def trades_chart(b, F, tr, vn, a, z, path):
    t = b["t"]
    sm = G2.summ(tr, a, z, b)
    X = mdates.date2num([kst(t[i]) for i in range(a, z)])
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, b["c"][a:z], color="#555", lw=0.8)
    ax[0].set_ylim(b["l"][a:z].min() * 0.985, b["h"][a:z].max() * 1.015)
    ax[0].set_autoscaley_on(False)
    for x in tr:
        lg = x["d"] == 1
        xe, xx = mdates.date2num(kst(t[x["e"]])), mdates.date2num(kst(t[min(x["x"], z - 1)]))
        if "tp" in x:
            ax[0].plot([xe, xx], [x["tp"], x["tp"]], color=G if lg else R, ls=":", lw=1.2)
            if not (ax[0].get_ylim()[0] < x["tp"] < ax[0].get_ylim()[1]):
                ax[0].annotate(f"TP {x['tp']:,.0f} (off chart)", (xe, x["ep"]), fontsize=7, xytext=(3, -12),
                               textcoords="offset points", color="#555")
        ax[0].scatter(xe, x["ep"], marker="o" if lg else "v", s=70, color=G if lg else R, zorder=5, edgecolor="white")
        ax[0].scatter(xx, x["xp"], marker="o" if lg else "^", s=70, color=R if lg else G, zorder=5, edgecolor="white")
        ax[0].annotate(f"{DW.LEV * x['r'] * 100:+.1f}% {WHY.get(x['why'], x['why'])}", (xx, x["xp"]), fontsize=7,
                       color=G if x["r"] > 0 else R, xytext=(3, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close"),
                          Line2D([], [], ls=":", color="#555", label="take-profit (probabilistic)")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(f"{vn} - BTCUSDT perp 15m, last 30 days {kst(t[a]):%m-%d} ~ {kst(t[z - 1]):%m-%d %H:%M} KST, market orders\n"
                    f"{sm['n']} trades ({sm['per_day']:.2f}/day), win {(sm['win'] or 0):.0%}, 5x {sm['x5']['ret']:+.1%} "
                    f"(max drawdown {sm['x5']['mdd']:.0%}). Exits: {sm['why']}", loc="left", fontsize=10)  # fmt: skip
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


def pt_chart(b, trs, vn, path):
    """정리 장치가 있는 거래 중 p(t) 점이 많은 것 하나(닿은 것·정리된 것 각 1개)."""
    t = b["t"]
    cand = [x for x in trs if len(x.get("pser", [])) >= 4]
    picks = []
    for why in ("bail", "tp", "struct", "time"):
        xs = [x for x in cand if x["why"] == why]
        if xs:
            picks.append(max(xs, key=lambda x: len(x["pser"])))
        if len(picks) == 2:
            break
    fig, axs = plt.subplots(1, max(len(picks), 1), figsize=(14, 4.5), squeeze=False)
    for ax, x in zip(axs[0], picks):
        hh = [p[0] for p in x["pser"]]
        pp = [p[1] for p in x["pser"]]
        ax.step(hh, pp, where="post", color="#2c7bb6", lw=1.8)
        ax.axhline(G2.BAIL, color=R, ls="--", lw=1, label=f"bail-out line {G2.BAIL:.2f}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("hours since the leg was confirmed")
        ax.set_ylabel("P(reach take-profit) from matched past legs")
        side = "long" if x["d"] == 1 else "short"
        ax.set_title(f"{side} {kst(t[x['e']]):%Y-%m-%d %H:%M} entry {x['ep']:,.0f}, TP {x['tp']:,.0f} -> "
                     f"{WHY.get(x['why'], x['why'])}, 5x {DW.LEV * x['r'] * 100:+.1f}%", fontsize=9, loc="left")  # fmt: skip
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{vn}: probability of reaching TP, re-estimated every hour (survival of matched past legs)", x=0.01, ha="left")
    plt.tight_layout()
    plt.savefig(path, dpi=95)
    plt.close(fig)


def equity_chart(b, runs, res, W, path):
    t = b["t"]
    fig, ax = plt.subplots(figsize=(15, 6))
    a, z = W["fresh"][0], len(t)
    for (vn, fn), col in zip(runs.items(), ("#2c7bb6", "#fdae61")):
        tr = fn(a, z)
        eq, xs, ys = 1.0, [kst(t[a])], [0.0]
        for x in tr:
            eq = max(eq * (1 + DW.LEV * x["r"]), 0)
            xs.append(kst(t[min(x["x"], z - 1)]))
            ys.append(np.log10(max(eq, 1e-4)))
        lab = ", ".join(f"{w} {res[vn][w]['x5']['ret']:+.0%}" for w in G2.MAIN)
        ax.step(xs, ys, where="post", color=col, lw=1.5, label=f"{vn}: {lab}")
    for w in ("design", "val", "year"):
        ax.axvline(kst(t[W[w][0]]), color="#999", lw=0.6, ls=":")
    ax.set_ylabel("5x equity, log10 (0 = start)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Equity 2020-03 ~ now, 5x, market orders, fees + funding (dotted = window starts)", loc="left", fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=95)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("names", nargs=3, help="매매·p(t) 그림 변형, 자산 곡선 변형 2개")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b, s = RL.load_all(DW.R5)
    F = DW.features(b)
    C = DW.all_cands(b, F)
    W = RL.windows(b["t"])
    res = json.loads((out / "result.json").read_text())["res"]
    runs = {vn: build(b, F, C, s, vn) for vn in a_.names}
    first = a_.names[0]
    eq = {vn: runs[vn] for vn in a_.names[1:]}
    a, z = W["d30"]
    trades_chart(b, F, runs[first](a, z), first, a, z, out / "chart_trades_d30.png")
    a, z = W["year"]
    pt_chart(b, runs[first](a, z), first, out / "chart_p_reach.png")
    equity_chart(b, eq, res, W, out / "equity.png")
    print("ok")


if __name__ == "__main__":
    main()
