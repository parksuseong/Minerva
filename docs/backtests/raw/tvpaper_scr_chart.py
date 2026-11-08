"""SCR + TV/논문 조합 그림(통과·근접 규칙만): chart_d30.png, equity.png.

표시: 롱 = 원, 숏 = 삼각형. 롱 진입·숏 청산 = 초록, 롱 청산·숏 진입 = 빨강. Stoch·CCI·RSI 칸, 한글(NanumGothic).
사용: python3 tvpaper_scr_chart.py OUT 규칙이름
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
from matplotlib import font_manager
from matplotlib.lines import Line2D

GREEN, RED = "#1a9641", "#d7191c"
FONT = Path("~/.minerva/fonts/NanumGothic-Regular.ttf").expanduser()
font_manager.fontManager.addfont(str(FONT))
plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def dt(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc)


def bars(m: dict, a: int, z: int, tf: int) -> dict:
    k0, k1 = np.searchsorted(m["t"], a), np.searchsorted(m["t"], z)
    t = m["t"][k0:k1]
    g = t // (tf * PD.MIN)
    st = np.r_[0, np.nonzero(np.diff(g))[0] + 1]
    en = np.r_[st[1:] - 1, len(t) - 1]
    sl = lambda x: x[k0:k1]  # noqa: E731
    return {"t": g[st] * tf * PD.MIN, "o": sl(m["o"])[st], "c": sl(m["c"])[en],
            "h": np.maximum.reduceat(sl(m["h"]), st), "l": np.minimum.reduceat(sl(m["l"]), st)}  # fmt: skip


def chart(out: Path, m: dict, trades: list, tf: int, title: str) -> None:
    z = int(m["t"][-1]) + PD.MIN
    a = z - 30 * PD.DAY
    B = bars(m, a - 3 * PD.DAY, z, tf)
    rsi, cci = PD.rsi(B["c"]), PD.cci(B["h"], B["l"], B["c"])
    k, dl = PD.stoch(B["h"], B["l"], B["c"])
    show = B["t"] >= a
    x = mdates.date2num([dt(t + tf / 2 * PD.MIN) for t in B["t"]])
    w = tf / 1440 * 0.7
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[5, 1.1, 1.1, 1.1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axs = [ax] + [fig.add_subplot(gs[i], sharex=ax) for i in range(1, 4)]
    up = B["c"] >= B["o"]
    for msk, col in ((show & up, "#9db8a5"), (show & ~up, "#d2a3a3")):
        ax.vlines(x[msk], B["l"][msk], B["h"][msk], color=col, lw=0.5)
        ax.bar(x[msk], np.abs(B["c"] - B["o"])[msk] + 1e-6, w, np.minimum(B["o"], B["c"])[msk], color=col, lw=0)
    for t in trades:
        d, mk = t["d"], "o" if t["d"] == 1 else "^"
        c_in, c_out = (GREEN, RED) if d == 1 else (RED, GREEN)
        ex = mdates.date2num(dt(t["T"]))
        ax.scatter(ex, t["e"], s=80, marker=mk, color=c_in, edgecolors="black", lw=0.6, zorder=5)
        ax.hlines(t["stop"], ex, mdates.date2num(dt(int(m["t"][t["kx"]]))), color="#555", lw=0.6)
        for kx, px, wgt in t["exits"]:
            tx = mdates.date2num(dt(int(m["t"][kx])))
            ax.scatter(tx, px, s=60 if wgt > 0.5 else 32, marker=mk, color=c_out, edgecolors="black", lw=0.4, zorder=5)
        ax.annotate(f"{t['r'] / t['Rp']:+.1f}R", (ex, t["e"]), xytext=(0, 12 if d == 1 else -16),
                    textcoords="offset points", fontsize=8, ha="center")  # fmt: skip
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(alpha=0.15)
    hs = [Line2D([], [], marker="o", ls="", color=GREEN, mec="k", label="롱 진입"),
          Line2D([], [], marker="o", ls="", color=RED, mec="k", label="롱 청산"),
          Line2D([], [], marker="^", ls="", color=RED, mec="k", label="숏 진입"),
          Line2D([], [], marker="^", ls="", color=GREEN, mec="k", label="숏 청산"),
          Line2D([], [], color="#555", lw=0.6, label="구조 손절")]  # fmt: skip
    ax.legend(handles=hs, fontsize=9, loc="upper left", ncol=5)
    for axx, ys, lv, nm in ((axs[1], [(k, "#333"), (dl, "#e66101")], (20, 80), "Stoch 5"),
                            (axs[2], [(cci, "#5e3c99")], (-100, 0, 100), "CCI 20"),
                            (axs[3], [(rsi, "#0571b0")], (30, 50, 70), "RSI 14")):  # fmt: skip
        for y, col in ys:
            axx.plot(x[show], y[show], color=col, lw=0.7)
        for v in lv:
            axx.axhline(v, color="#999", lw=0.5, ls="--")
        axx.set_ylabel(nm, fontsize=9)
        axx.grid(alpha=0.15)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for axx in axs[:-1]:
        plt.setp(axx.get_xticklabels(), visible=False)
    ax.set_xlim(x[show][0], x[-1] + 0.1)
    plt.savefig(out / "chart_d30.png", dpi=90, bbox_inches="tight")
    plt.close()


def equity(out: Path, name: str, tj: dict) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    tr = tj["pool"] + tj["year"]
    xs = [dt(t["T"]) for t in tr]
    axs[0].plot(xs, np.cumprod([1 + PD.lev1(t) * t["r"] for t in tr]), color="#2c7bb6", lw=1)
    for frac, ls in ((1.0, "-"), (0.4, "--")):
        axs[1].plot(xs, np.cumprod([1 + frac * max(5 * t["r"], -1) for t in tr]), color="#d7191c", ls=ls, lw=1,
                    label=f"5배 증거금 {int(frac * 100)}%")  # fmt: skip
    axs[0].set_title(f"{name}: 1% 위험(5배 상한), 2020-03~", loc="left")
    axs[1].set_title(f"{name}: 5배 고정", loc="left")
    for a_ in axs:
        a_.set_yscale("log")
        a_.axhline(1, color="#333", lw=0.6)
        a_.grid(alpha=0.2)
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=100)
    plt.close()


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    name = sys.argv[2]
    res = json.loads((out / "result.json").read_text())
    tj = json.loads((out / "trades.json").read_text())[name]
    rule = next(r for r in res["rules"] if r["name"] == name)
    r = res["res"][name]["d30"]
    m = PD.load_1m(out)
    title = (f"{name} 최근 30일 {rule['tf']}분봉: {r['n']}회, 승률 {100 * r.get('win', 0):.0f}%, "
             f"1% 위험 {100 * r['risk1']['ret']:+.1f}%, 5배·100% {100 * r['x5_100']['ret']:+.1f}% (수수료·펀딩 포함)")  # fmt: skip
    chart(out, m, tj["d30"], rule["tf"], title)
    equity(out, name, tj)
    print("ok")


if __name__ == "__main__":
    main()
