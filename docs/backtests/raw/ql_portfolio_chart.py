"""QL 포트폴리오 차트: 최근 30일 매매(가격 + Stoch·CCI·RSI), 구간별 5배 자산 곡선(증거금 100%/40%), 우위별 횟수·수수료 부담.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익(증거금 전액 기준).
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 ql_portfolio_chart.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import ql_portfolio as Q
import regime_lines as RL
from daily_wave_chart import G, R, dt
from matplotlib.lines import Line2D

LAB = {"fresh": "2020-03~2021-10 (untouched)", "design": "2022-01~2024-06", "val": "2024-07~2025-10",
       "year": "2025-11~ (last year)"}  # fmt: skip


def trades_chart(b, F, tr, a, z, s, title, path):
    t = b["t"]
    X = mdates.date2num([dt(t, i) for i in range(a, z)])
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, b["c"][a:z], color="#555", lw=0.8)
    for x in tr:
        lg = x["d"] == 1
        xe, xx = mdates.date2num(dt(t, x["e"])), mdates.date2num(dt(t, x["x"]))
        ax[0].scatter(xe, x["ep"], marker="o" if lg else "v", s=70, color=G if lg else R, zorder=5, edgecolor="white")
        ax[0].scatter(xx, x["xp"], marker="o" if lg else "^", s=70, color=R if lg else G, zorder=5, edgecolor="white")
        ax[0].annotate(f"{Q.LEV * x['r'] * 100:+.1f}% {x.get('edge', '')}", (xx, x["xp"]), fontsize=7,
                       color=G if x["r"] > 0 else R, xytext=(3, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(
        f"{title} - BTCUSDT perp 15m {dt(t, a):%Y-%m-%d} ~ {dt(t, z - 1):%m-%d %H:%M} KST\n"
        f"{s['n']} trades ({s['per_day']:.2f}/day), win {(s['win'] or 0):.0%}, 5x full margin "
        f"{s['x5_f1.0']['ret']:+.1%} (mdd {s['x5_f1.0']['mdd']:.0%}), 5x on 40% margin {s['x5_f0.4']['ret']:+.1%}. "
        "Maker limit entry, fees + funding included", loc="left", fontsize=10)  # fmt: skip
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


def equity_chart(b, F, C, W, path):
    t = b["t"]
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    for k, w in enumerate(LAB):
        a, z = W[w]
        for fee, f, col, ls in (("FB", 0.4, "#2c7bb6", "-"), ("FB", 1.0, "#2c7bb6", ":"),
                                ("FA", 0.4, "#d73027", "-")):  # fmt: skip
            tr = Q.sim(b, F, C, fee, a, z)
            eq, xs, ys = 1.0, [dt(t, a)], [0.0]
            for x in tr:
                pos = -1.0 if Q.LEV * x["worst"] <= Q.LIQ else max(Q.LEV * x["r"], -1.0)
                eq = max(eq * (1 + f * pos), 0)
                xs.append(dt(t, x["x"]))
                ys.append((eq - 1) * 100)
            lab = f"{'maker entry' if fee == 'FB' else 'market'}, {int(f * 100)}% margin: {ys[-1]:+.0f}%"
            ax[k].step(xs, ys, where="post", color=col, ls=ls, lw=1.3, label=lab)
        ax[k].axhline(0, color="#999", lw=0.6)
        ax[k].set_title(LAB[w], fontsize=10)
        ax[k].legend(fontsize=7, loc="upper left")
        ax[k].tick_params(axis="x", labelsize=7, rotation=30)
        ax[k].grid(alpha=0.2)
    ax[0].set_ylabel("5x return %")
    fig.suptitle("QL portfolio (NY-open breakout, any admitted daily gate) - 5x equity by window", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def freq_chart(res, path):
    ed = res["edges"]
    names = list(ed)
    pdays = [ed[e]["FB"]["pool"]["per_day"] for e in names]
    d1 = [(ed[e]["FB"]["design"]["avg1"] or 0) * 100 for e in names]
    cols = ["#1a9850" if ed[e]["admit"] else "#bbbbbb" for e in names]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].barh(names, pdays, color=cols)
    ax[0].axvline(1.0, color="#d73027", ls="--", lw=1)
    ax[0].set_xlabel("trades per day (2020-03~2025-10, each edge alone, maker entry)")
    ax[0].set_title("Green = admitted (design window positive after costs)", fontsize=10)
    ax[1].barh(names, d1, color=cols)
    ax[1].axvline(0, color="#555", lw=0.8)
    ax[1].set_xlabel("avg net % per trade at 1x, design window 2022-01~2024-06")
    pd = res["portfolio"]["FB"]["pool"]["per_day"]
    fig.suptitle(f"Edges: frequency vs edge after fees. Portfolio = {pd:.2f} trades/day (one position at a time)",
                 fontsize=11)  # fmt: skip
    for q in ax:
        q.grid(alpha=0.2, axis="x")
        q.invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).expanduser()
    res = json.loads((out / "result.json").read_text())
    b, s = RL.load_all(DW.R5)
    F = DW.features(b)
    Q.add_gates(b, s, F)
    E = Q.edge_cands(b, F)
    W = RL.windows(b["t"])
    C = Q.merged(E, res["admitted"], res["prio"])
    for w, title in (("d30", "QL portfolio, last 30 days"), ("m6", "QL portfolio, last 6 months")):
        a, z = W[w]
        tr = Q.sim(b, F, C, "FB", a, z)
        trades_chart(b, F, tr, a, z, Q.summ(tr, a, z, b), title, out / f"chart_{w}.png")
    equity_chart(b, F, C, W, out / "equity.png")
    freq_chart(res, out / "edges.png")
    print("charts done")


if __name__ == "__main__":
    main()
