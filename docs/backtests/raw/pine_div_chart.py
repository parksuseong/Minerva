"""다이버전스 신호 차트: 최근 30일 15분봉 매매(chart_d30.png), 구간별 자산 곡선(equity.png).

현실 진입·청산은 진한 표시, 트레이딩뷰가 라벨을 그리는 피벗 봉은 흐린 표시로 둬서 사후 5봉 간격이 보이게 한다.
표시 규칙: 롱 = 원, 숏 = 삼각형. 롱 진입·숏 청산 = 초록, 롱 청산·숏 진입 = 빨강.
사용: python3 pine_div_chart.py OUT [quick|full] [안 이름]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pine_div as PD

GREEN, RED = "#1a9641", "#d7191c"


def dt(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc)


def bars15(m: dict, a: int, z: int) -> dict:
    k0, k1 = np.searchsorted(m["t"], a), np.searchsorted(m["t"], z)
    t = m["t"][k0:k1]
    g = t // (15 * PD.MIN)
    st = np.r_[0, np.nonzero(np.diff(g))[0] + 1]
    en = np.r_[st[1:] - 1, len(t) - 1]
    sl = lambda x: x[k0:k1]  # noqa: E731
    return {"t": g[st] * 15 * PD.MIN, "o": sl(m["o"])[st], "c": sl(m["c"])[en],
            "h": np.maximum.reduceat(sl(m["h"]), st), "l": np.minimum.reduceat(sl(m["l"]), st)}  # fmt: skip


def chart(out: Path, trades: list, sigs: list, title: str, fname: str, rib: bool = False) -> None:
    m = PD.load_1m(out)
    z = int(m["t"][-1]) + PD.MIN
    a = z - 30 * PD.DAY
    B = bars15(m, a - 3 * PD.DAY, z)
    rsi, cci = PD.rsi(B["c"]), PD.cci(B["h"], B["l"], B["c"])
    k, dline = PD.stoch(B["h"], B["l"], B["c"])
    show = B["t"] >= a
    x = mdates.date2num([dt(t + 7.5 * PD.MIN) for t in B["t"]])
    w = 15 / 1440 * 0.7
    np_ = 5 if rib else 4
    fig = plt.figure(figsize=(22, 14 if rib else 13))
    gs = fig.add_gridspec(np_, 1, height_ratios=[5, 1.1, 1.1, 1.1] + ([0.9] if rib else []), hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axs = [ax] + [fig.add_subplot(gs[i], sharex=ax) for i in range(1, np_)]
    up = B["c"] >= B["o"]
    for msk, col in ((show & up, "#9db8a5"), (show & ~up, "#d2a3a3")):
        ax.vlines(x[msk], B["l"][msk], B["h"][msk], color=col, lw=0.5)
        ax.bar(x[msk], np.abs(B["c"] - B["o"])[msk] + 1e-6, w, np.minimum(B["o"], B["c"])[msk], color=col, lw=0)
    # 받지 않은 신호(포지션 중·추격 제한): 피벗 봉 위치에 아주 흐린 점
    taken = {(t["pt"], t["d"]) for t in trades}
    for s in sigs:
        if (s["pt"], s["d"]) in taken:
            continue
        ax.scatter(mdates.date2num(dt(s["pt"] - 7.5 * PD.MIN)), s["ext"], s=14,
                   marker="o" if s["d"] == 1 else "^", facecolors="none",
                   edgecolors=GREEN if s["d"] == 1 else RED, alpha=0.25, lw=0.6)  # fmt: skip
    for t in trades:
        d, mk = t["d"], "o" if t["d"] == 1 else "^"
        c_in, c_out = (GREEN, RED) if d == 1 else (RED, GREEN)
        px_ = mdates.date2num(dt(t["pt"] - 7.5 * PD.MIN))
        ex = mdates.date2num(dt(t["T"]))
        # 트레이딩뷰 라벨 자리(피벗 봉, 사후 5봉): 흐린 표시 + 현실 진입까지 점선
        if not rib:
            ax.scatter(px_, t["pext"], s=70, marker=mk, color=c_in, alpha=0.25, lw=0)
            ax.plot([px_, ex], [t["pext"], t["e"]], color="#777", lw=0.7, ls=":")
        ax.scatter(ex, t["e"], s=70, marker=mk, color=c_in, edgecolors="black", lw=0.6, zorder=5)
        ax.hlines(t["stop"], ex, mdates.date2num(dt(int(m["t"][t["kx"]]))), color="#555", lw=0.6)
        for kx, px, wgt in t["exits"]:
            tx = mdates.date2num(dt(int(m["t"][kx])))
            ax.scatter(tx, px, s=55 if wgt > 0.5 else 30, marker=mk, color=c_out, edgecolors="black", lw=0.4, zorder=5)
        lab = f"{t['tier']} {t['r'] / t['Rp']:+.1f}R"
        ax.annotate(lab, (ex, t["e"]), xytext=(0, 12 if d == 1 else -16), textcoords="offset points",
                    fontsize=7, ha="center", color="#222")  # fmt: skip
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(alpha=0.15)
    ax.set_ylabel("BTCUSDT perp")
    from matplotlib.lines import Line2D

    hs = [Line2D([], [], marker="o", ls="", color=GREEN, mec="k", label="long open (real, after confirm)"),
          Line2D([], [], marker="o", ls="", color=RED, mec="k", label="long close"),
          Line2D([], [], marker="^", ls="", color=RED, mec="k", label="short open (real)"),
          Line2D([], [], marker="^", ls="", color=GREEN, mec="k", label="short close"),
          Line2D([], [], marker="o", ls="", color=GREEN, alpha=0.3, label="pivot bar = where TradingView draws the label (hindsight: known only R bars later)"),
          Line2D([], [], color="#777", ls=":", label="hindsight gap: pivot -> real entry"),
          Line2D([], [], color="#555", lw=0.6, label="stop (pivot extreme)")]  # fmt: skip
    if rib:
        hs = hs[:4] + [hs[6]]
    ax.legend(handles=hs, fontsize=8, loc="upper left", ncol=2)
    if rib:
        F = {"c": B["c"]}
        PD.ribbon(F)
        cols = {1: "#32cd32", 2: "#800000", -1: "#ff0000", -2: "#006400", 0: "#cccccc"}
        lead = F["lead"]
        sa = axs[4]
        for v, col in cols.items():
            mk_ = show & (lead == v)
            sa.bar(x[mk_], 1, 15 / 1440, 0, color=col, lw=0)
        mas = np.array([PD.ema(B["c"], n) for n in range(10, 95, 5)])
        full = (np.diff(mas, axis=1) > 0).mean(0)
        sa.plot(x[1:][show[1:]], full[show[1:]], color="black", lw=0.6)
        sa.set_ylim(0, 1)
        sa.set_ylabel("ribbon 15m\nlead / rising share", fontsize=7)
        from matplotlib.patches import Patch

        sa.legend(handles=[Patch(color=cols[1], label="LIME up"), Patch(color=cols[2], label="MAROON dip in up"),
                           Patch(color=cols[-1], label="RED down"), Patch(color=cols[-2], label="GREEN bounce in down")],
                  fontsize=7, ncol=4, loc="upper left")  # fmt: skip
    for axx, ys, lv, nm in ((axs[1], [(k, "#333"), (dline, "#e66101")], (20, 80), "Stoch 5,3"),
                            (axs[2], [(cci, "#5e3c99")], (-100, 0, 100), "CCI 20"),
                            (axs[3], [(rsi, "#0571b0")], (30, 50, 70), "RSI 14")):  # fmt: skip
        for y, col in ys:
            axx.plot(x[show], y[show], color=col, lw=0.7)
        for v in lv:
            axx.axhline(v, color="#999", lw=0.5, ls="--")
        axx.set_ylabel(nm, fontsize=8)
        axx.grid(alpha=0.15)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for axx in axs[:-1]:
        plt.setp(axx.get_xticklabels(), visible=False)
    ax.set_xlim(x[show][0], x[-1] + 0.1)
    plt.savefig(out / fname, dpi=90, bbox_inches="tight")
    plt.close()


def equity(out: Path, name: str) -> None:
    tj = json.loads((out / "trades.json").read_text())[name]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    for w, col in (("fresh", "#7b3294"), ("design", "#2c7bb6"), ("val", "#1a9641"), ("year", "#d7191c")):
        tr = tj[w]
        if not tr:
            continue
        xs = [dt(t["T"]) for t in tr]
        eq = np.cumprod([1 + PD.lev1(t) * t["r"] for t in tr])
        axs[0].plot(xs, eq, color=col, lw=1, label=w)
        for frac, ls in ((1.0, "-"), (0.4, "--")):
            e5 = np.cumprod([1 + frac * max(5 * t["r"], -1) for t in tr])
            axs[1].plot(xs, e5, color=col, ls=ls, lw=1, label=f"{w} {int(frac * 100)}%")
    axs[0].set_title(f"{name}: 1% risk sizing (5x cap)", loc="left", fontsize=10)
    axs[1].set_title(f"{name}: 5x, margin 100% (solid) / 40% (dashed)", loc="left", fontsize=10)
    for a_ in axs:
        a_.set_yscale("log")
        a_.axhline(1, color="#333", lw=0.6)
        a_.grid(alpha=0.2)
        a_.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close()


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    mode = sys.argv[2] if len(sys.argv) > 2 else "quick"
    q = json.loads((out / "quick_d30.json").read_text())
    r = q["real"]
    title = (f"15m Pine divergence (all tiers S/M/W) + Minerva exit (stop=pivot, BE at +1R, 1/3 at +2R, trail prior 15m bar) "
             f"- last 30 days, 1m fills, fees+funding: {r['n']} trades, win {100 * r['win']:.0f}%, "
             f"1% risk {100 * r['risk1']['ret']:+.1f}%, 5x/100% {100 * r['x5_100']['ret']:+.1f}%. "
             f"Same trades at the pivot bar (chart-visible, impossible): 1% risk {100 * q['hind']['risk1']['ret']:+.1f}%")  # fmt: skip
    chart(out, q["trades"], q["sigs"], title, "chart_d30.png")
    if mode == "full":
        allr = json.loads((out / "result.json").read_text())
        res, vd = allr["res"], allr["verdict"]
        tj = json.loads((out / "trades.json").read_text())
        rank = lambda n: (sum(vd[n][g] for g in ("g1", "g2", "g3", "g4", "g5")), res[n]["design"]["risk1"]["ret"])  # noqa: E731
        bf = max([n for n in res if n.startswith("F")], key=rank)
        br = max([n for n in res if n.startswith("R")], key=rank)
        for nm, fn, rb in ((bf, "chart_d30_fixed.png", False), (br, "chart_d30_ribbon.png", True)):
            r = res[nm]["d30"]
            if not r["n"]:
                print("no d30 trades", nm)
                continue
            ti = (f"{nm} (best by registered rule) - last 30 days, 15m view, 1m fills, fees+funding: {r['n']} trades, "
                  f"win {100 * r['win']:.0f}%, 1% risk {100 * r['risk1']['ret']:+.1f}%, 5x/100% {100 * r['x5_100']['ret']:+.1f}%, "
                  f"5x/40% {100 * r['x5_40']['ret']:+.1f}%. Last 30 days are in-sample for this fix")  # fmt: skip
            chart(out, tj[nm]["d30"], [] if rb else q["sigs"], ti, fn, rib=rb)
            print(fn, nm)
        equity(out, sys.argv[3] if len(sys.argv) > 3 else bf)
    print("ok")


if __name__ == "__main__":
    main()
