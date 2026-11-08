"""QL 고도화 차트: 최근 30일 매매(가격 + Stoch·CCI·RSI)와 자산 곡선(지금 모형 vs 등록 최종 vs 사후 P1·P2, 5배 증거금 100%)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import daily_wave as DW
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import ql_upgrade as U
from btc_fractal import cci, rsi, stoch
from matplotlib.lines import Line2D
from r5_data import load

KST = timezone(timedelta(hours=9))
G, R = "#1a9850", "#d73027"


def kdt(ms):
    return datetime.fromtimestamp(ms / 1000, KST).replace(tzinfo=None)


def trades_chart(b, tr, title, path):
    t = b["t"]
    a = int(np.searchsorted(t, t[-1] - 30 * U.DAY))
    X = md.date2num([kdt(v) for v in t[a:]])
    k, cc, rr = stoch(b["h"], b["l"], b["c"], 5), cci(b["h"], b["l"], b["c"], 20), rsi(b["c"], 14)
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, b["c"][a:], color="#555", lw=0.8)
    for x in tr:
        long = x["d"] == 1
        ax[0].scatter(md.date2num(kdt(x["t_in"])), x["ep"], marker="o" if long else "v", s=80,
                      color=G if long else R, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].scatter(md.date2num(kdt(x["t_out"])), x["xp"], marker="o" if long else "^", s=80,
                      color=R if long else G, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].annotate(f"{5 * x['r'] * 100:+.1f}% ({x['why']})", (md.date2num(kdt(x["t_out"])), x["xp"]),
                       fontsize=8, color=G if x["r"] > 0 else R, xytext=(4, 6), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")], loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(title, loc="left", fontsize=10)
    for i_, (y, nm, lo, hi) in enumerate(((k, "Stoch 5", 20, 80), (cc, "CCI 20", -100, 100), (rr, "RSI 14", 30, 70)), 1):
        ax[i_].plot(X, y[a:], lw=0.6, color="#2c7bb6")
        ax[i_].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[i_].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[i_].set_ylabel(nm)
    ax[3].xaxis.set_major_formatter(md.DateFormatter("%m-%d"))
    for z in ax:
        z.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def curve(tr, f=None):
    eq, xs, ys = 1.0, [], []
    for i, x in enumerate(tr):
        w = 1.0 if f is None else f[i]
        eq *= 1 + w * max(U.LEV * x["r"], -1.0)
        xs.append(kdt(x["t_out"]))
        ys.append(eq)
    return xs, ys


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b = load(DW.R5)
    ph = json.loads((out / "posthoc_trades.json").read_text())
    tr2 = ph["P2"]["d30"]
    trades_chart(b, tr2, "P2 (post-hoc, not pre-registered): NY first-hour breakout, limit at 15m close, exit by NY 16:00, "
                 f"maker TP, vol sizing - last 30 days: {len(tr2)} trades, labels = 5x net per trade", out / "chart_d30.png")  # fmt: skip
    # 자산: 전체(fresh 시작~지금) — result/posthoc 의 구간별 거래를 이어 붙이지 않고 결과 수치로 막대
    res = json.loads((out / "result.json").read_text())
    pj = json.loads((out / "posthoc.json").read_text())
    rows = [("current model (1m re-sim)", res["old"]), ("pre-registered final (+A1)", res["final"]),
            ("P1 post-hoc: maker TP", pj["P1"]["res"]), ("P2 post-hoc: 16:00 exit + maker TP + vol size", pj["P2"]["res"])]  # fmt: skip
    wins = ("fresh", "design", "val", "year", "m6", "d30")
    lab = ["2020-03~21-10\n(untouched)", "2022-01~24-06", "2024-07~25-10", "last year", "6 months", "30 days"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    wdt = 0.2
    cols = ["#888", "#d73027", "#2c7bb6", "#1a9850"]
    for j, (nm, r) in enumerate(rows):
        v = [r[w]["x5_f1.0"]["ret"] * 100 for w in wins]
        dd = [r[w]["x5_f1.0"]["mdd"] * 100 for w in wins]
        xx = np.arange(len(wins)) + (j - 1.5) * wdt
        ax[0].bar(xx, np.sign(v) * np.log10(1 + np.abs(v)), wdt, color=cols[j], label=nm)
        ax[1].bar(xx, dd, wdt, color=cols[j], label=nm)
    ax[0].set_xticks(range(len(wins)), lab, fontsize=8)
    ax[1].set_xticks(range(len(wins)), lab, fontsize=8)
    ax[0].set_yticks([-2, -1, 0, 1, 2, 3, 4], ["-100%", "-10%", "0", "+10%", "+100%", "+1,000%", "+10,000%"])
    ax[0].axhline(0, color="#333", lw=0.6)
    ax[0].set_title("5x, 100% margin: return by window (log scale)", loc="left", fontsize=10)
    ax[1].set_title("max drawdown % (5x, 100% margin)", loc="left", fontsize=10)
    ax[0].legend(fontsize=8)
    for z in ax:
        z.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out / "windows.png", dpi=100)
    plt.close()
    print("ok")


if __name__ == "__main__":
    main()
