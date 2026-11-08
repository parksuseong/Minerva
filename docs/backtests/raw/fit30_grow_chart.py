"""fit30 키우기 차트: 걷기 시험 버전이 최근 30일에 낸 매매(chart_d30.png)와 자산 곡선(equity.png).

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익(100% 증거금).
점수 판은 주마다 θ가 달라 s/θ로 그린다(±1 = 진입선).
실행: ... python3 docs/backtests/raw/fit30_grow_chart.py --out $OUT [--key BASE-90]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import fit30_grow as G
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from fit30_chart import G as GREEN
from fit30_chart import RD
from matplotlib.lines import Line2D


def dt(ms_: int) -> datetime:
    return datetime.fromtimestamp(ms_ / 1000, G.KST).replace(tzinfo=None)


def score_series(C, folds, N, a, z, feats):
    """매매 주마다 그 주에 쓴 표준화·w·θ로 s/θ."""
    t = C.t
    out = np.full(len(t), np.nan)
    for f in folds:
        fa, fz = f["a"], f["z"]
        if fz <= a or fa >= z:
            continue
        ta = int(np.searchsorted(t, f["start"] - N * G.DAY))
        Z, _ = G.zmat(C, feats, ta, fa)
        out[fa:fz] = (np.array(f["w"]) @ Z[:, fa:fz]) / f["th"]
    return out


def trade_chart(C, tr, sn, a, z, title, path):
    t = C.t
    X = mdates.date2num([dt(int(t[i])) for i in range(a, z)])
    fig, ax = plt.subplots(5, 1, figsize=(16, 14), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1.2]})
    ax[0].plot(X, C.c[a:z], color="#555", lw=0.8)
    for x in tr:
        long = x["d"] == 1
        e, xi = x["e"], min(x["x"], z - 1)
        ax[0].scatter(mdates.date2num(dt(int(t[e]))), x["ep"], marker="o" if long else "v", s=60,
                      color=GREEN if long else RD, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].scatter(mdates.date2num(dt(int(t[xi]))), x["xp"], marker="o" if long else "^", s=60,
                      color=RD if long else GREEN, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].annotate(f"{5 * x['r'] * 100:+.1f}", (mdates.date2num(dt(int(t[xi]))), x["xp"]), fontsize=6,
                       color=GREEN if x["r"] > 0 else RD, xytext=(2, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=GREEN, label="long open"),
                          Line2D([], [], marker="o", ls="", color=RD, label="long close"),
                          Line2D([], [], marker="v", ls="", color=RD, label="short open"),
                          Line2D([], [], marker="^", ls="", color=GREEN, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(title, loc="left", fontsize=10)
    R = C.R
    for k, (y, nm, lo, hi) in enumerate(((R["k"], "Stoch 5", 20, 80), (R["cci"], "CCI 20", -100, 100),
                                          (R["rsi"], "RSI 14", 30, 70), (sn, "score / theta", -1, 1)), 1):  # fmt: skip
        ax[k].plot(X, y[a:z], lw=0.6, color="#2c7bb6" if k < 4 else "#7b3294")
        ax[k].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[k].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[k].set_ylabel(nm)
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for q in ax:
        q.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def curve(tr, frac):
    eq, xs, ys = 1.0, [], []
    for x in tr:
        if G.LEV * x["worst"] <= G.LIQ:
            eq *= 1 - frac
        else:
            eq *= 1 + frac * max(G.LEV * x["r"], -1.0)
        xs.append(dt(x["te"]))
        ys.append(max(eq, 1e-4))
    return xs, ys


def equity_chart(allwf, rep, best, end, path):
    W = G.wins(end)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    cols = {"30": "#1b9e77", "60": "#d95f02", "90": "#7570b3", "180": "#e7298a"}
    q = axs[0, 0]
    for N, folds in allwf["BASE"].items():
        tr = [x for f in folds for x in f["base"]]
        xs, ys = curve(tr, 0.4)
        q.plot(xs, ys, color=cols[N], lw=1, label=f"WF-{N}d")
    q.set_yscale("log")
    q.axhline(1, color="#999", lw=0.6)
    q.set_title("weekly refit, all weeks 2020-09 ~ now, 5x on 40% margin (equity, log)", fontsize=9, loc="left")
    q.legend(fontsize=8)
    for q, wn in ((axs[0, 1], "A"), (axs[1, 0], "Y1")):
        wa, wz = W[wn]
        for N, folds in allwf["BASE"].items():
            tr = [x for f in folds if wa <= f["start"] < wz for x in f["base"]]
            xs, ys = curve(tr, 0.4)
            r = rep[f"BASE-{N}"][wn]
            q.plot(xs, [(y - 1) * 100 for y in ys], color=cols[N], lw=1,
                   label=f"WF-{N}d: {r['x5_f0.4']['ret']:+.0%} (100% margin {r['x5_f1.0']['ret']:+.0%}), "
                         f"p={r['random']['p']:.2f}")  # fmt: skip
        q.axhline(0, color="#999", lw=0.6)
        q.legend(fontsize=7)
        lab = "2024-07 ~ now" if wn == "A" else "last 1 year"
        q.set_title(f"{lab}: 5x on 40% margin (%), per refit length", fontsize=9, loc="left")
    q = axs[1, 1]
    wa, wz = W["A"]
    N = best.split("-")[1]
    for addon, col in (("BASE", "#333"), ("GTGATE", "#1b9e77"), ("GTFEAT", "#d95f02"), ("DCFEAT", "#7570b3")):
        if addon not in allwf or N not in allwf[addon]:
            continue
        tr = [x for f in allwf[addon][N] if wa <= f["start"] < wz for x in f["base"]]
        xs, ys = curve(tr, 0.4)
        r = rep[f"{addon}-{N}"]["A"]
        q.plot(xs, [(y - 1) * 100 for y in ys], color=col, lw=1, label=f"{addon}: {r['x5_f0.4']['ret']:+.0%}")
    q.axhline(0, color="#999", lw=0.6)
    q.legend(fontsize=8)
    q.set_title(f"add-ons, refit {N}d, 2024-07 ~ now, 5x on 40% margin (%)", fontsize=9, loc="left")
    for q in axs.flat:
        q.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        q.grid(alpha=0.2)
    fig.suptitle("fit30 grown: weekly refit walk-forward - BTCUSDT perp 15m, fees + funding + chase limit",
                 fontsize=11)  # fmt: skip
    plt.tight_layout()
    plt.savefig(path, dpi=95)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    rep = json.loads((out / "report.json").read_text())
    allwf = {}
    for addon in G.ADDONS:
        fp = out / f"wf_{addon}.json"
        if fp.exists():
            allwf[addon] = json.loads(fp.read_text())
    best = a.key or max((k for k in rep if k.startswith("BASE-")), key=lambda k: rep[k]["A"]["x5_f0.4"]["ret"])
    C = G.Ctx()
    t = C.t
    end = int(t[-1]) + G.M15
    addon, N = best.split("-")
    if addon != "BASE":
        C.add_extra()
    folds = allwf[addon][N]
    za = int(np.searchsorted(t, end - 30 * G.DAY))
    z = len(t)
    tr = [x for f in folds for x in f["base"] if x["e"] >= za]
    sn = score_series(C, folds, int(N), za, z, G.feats_for(addon))
    r = rep[best]["D30"]
    trade_chart(C, tr, sn, za, z,
                f"WALK-FORWARD (refit weekly on prior {N}d, trade next 7d) last 30 days {G.kst(int(t[za]))} ~ "
                f"{G.kst(int(t[-1]))} KST - {r['n']} trades ({r['per_day']:.1f}/day, L {r['long']} / S {r['short']}), "
                f"win {(r['win'] or 0):.0%}, 5x {r['x5_f1.0']['ret']:+.0%} on 100% margin, "
                f"{r['x5_f0.4']['ret']:+.0%} on 40%\nentry next-bar open, exit opposite signal / 1.5 ATR stop / 32 bars; "
                f"labels = 5x net % per trade",
                out / "chart_d30.png")  # fmt: skip
    equity_chart(allwf, rep, best, end, out / "equity.png")
    print("charts ok", best)


if __name__ == "__main__":
    main()
