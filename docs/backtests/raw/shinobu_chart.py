"""Shinobu 이식 차트: 최근 30일 매매(가격 + Stoch5·CCI20·RSI5)와 6개월 5배 자산 곡선(오늘 SCR 3단 비교 포함).

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 shinobu_chart.py --out <폴더>
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
from matplotlib.lines import Line2D

KST = timezone(timedelta(hours=9))
G, R = "#1a9850", "#d73027"
LEV = 5


def kdt(ms_):
    return datetime.fromtimestamp(int(ms_) / 1000, KST).replace(tzinfo=None)


def trade_chart(out: Path, name: str, trades: list, s: dict) -> Path:
    tf = name.split("|")[2]
    z = np.load(out / f"bars_{tf}.npz")
    off = int(z["a"])
    t = z["t"]
    a = int(np.searchsorted(t, t[-1] - 30 * 86_400_000))
    X = mdates.date2num([kdt(v) for v in t[a:]])
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    ax[0].plot(X, z["c"][a:], color="#555", lw=0.7)
    for x in trades:
        long = x["d"] == 1
        e, xi = x["e"] - off, min(x["x"] - off, len(t) - 1)
        if e < a:
            continue
        ax[0].scatter(mdates.date2num(kdt(t[e])), x["ep"], marker="o" if long else "v", s=70,
                      color=G if long else R, zorder=5, edgecolor="white")  # fmt: skip
        if x["why"] != "end":
            ax[0].scatter(mdates.date2num(kdt(t[xi])), x["xp"], marker="o" if long else "^", s=70,
                          color=R if long else G, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].annotate(f"{LEV * x['r'] * 100:+.1f}%", (mdates.date2num(kdt(t[xi])), x["xp"]), fontsize=7,
                       color=G if x["r"] > 0 else R, xytext=(3, 6), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=R, label="long close"),
                          Line2D([], [], marker="v", ls="", color=R, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(
        f"Shinobu {name} on BTCUSDT perp, last 30 days ({kdt(t[a]):%m-%d} ~ {kdt(t[-1]):%m-%d %H:%M} KST)\n"
        f"{s['n']} trades (L {s['nl']} / S {s['ns']}), win {s['wins']}/{s['n']}, 5x {s['x5']['ret']:+.1%} "
        f"(max drawdown {s['x5']['mdd']:.0%}, liq {s['x5']['liq']:.0f}), 1x {s['x1']['ret']:+.1%}. "
        "Fill = open 2 bars after the signal bar; fees 0.05%/side + funding. Labels = 5x net per trade",
        fontsize=10, loc="left")  # fmt: skip
    for i_, (key, nm, lo, hi) in enumerate((("k", "Stoch 5", 20, 80), ("cci", "CCI 20", -100, 100),
                                            ("rsi", "RSI 5", 30, 70)), 1):  # fmt: skip
        ax[i_].plot(X, z[key][a:], lw=0.5, color="#2c7bb6")
        ax[i_].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[i_].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[i_].set_ylabel(nm)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for q in ax:
        q.grid(alpha=0.2)
    plt.tight_layout()
    p = out / f"chart_{name.replace('|', '_')}_d30.png"
    plt.savefig(p, dpi=100)
    plt.close(fig)
    return p


def curve(trades: list, t: np.ndarray, off: int, start: int):
    eq, xs, ys = 1.0, [kdt(start)], [0.0]
    for x in trades:
        eq *= 1 + LEV * x["r"]
        xs.append(kdt(t[min(x["x"] - off, len(t) - 1)]))
        ys.append((eq - 1) * 100)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).expanduser()
    res = json.loads((out / "result.json").read_text())
    trades = json.loads((out / "trades.json").read_text())
    names = sorted(res["res"], key=lambda n: (res["res"][n]["m6"]["win"] or 0), reverse=True)[:2]
    for n in res["pick"]:
        if n not in names:
            names.append(n)
    paths = [trade_chart(out, n, trades[n]["d30"], res["res"][n]["d30"]) for n in names]
    # 오늘 SCR 3단(15분봉) 비교: 같은 6개월
    import scr15_timing as X

    b = X.load()
    F = X.features(b)
    end = int(b["t"][-1]) + X.M15
    W = {"d30": end - 30 * X.DAY, "m6": end - 183 * X.DAY, "year": X.ms("2025-11-01"),
         "mid": (X.ms("2022-01-01"), X.ms("2025-11-01")), "fresh": (X.ms("2020-03-01"), X.ms("2021-11-01"))}  # fmt: skip
    base, base_tr = {}, {}
    for v in (("B0", "S1", None), ("A8", "S1", None)):
        nm = X.vname(v)
        base[nm] = {}
        for w, span in W.items():
            a_, z_ = (span, end) if isinstance(span, int) else span
            a, z = int(np.searchsorted(b["t"], a_)), int(np.searchsorted(b["t"], z_))
            tr, cnt = X.sim(b, F, v, a, z)
            base[nm][w] = X.summ(tr, cnt)
            if w == "m6":
                base_tr[nm] = (tr, a)
    (out / "baselines.json").write_text(json.dumps(base, indent=1, default=float))
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
    z15 = np.load(out / "bars_15m.npz")
    t15 = z15["t"]
    a15 = int(np.searchsorted(t15, t15[-1] + 900_000 - 183 * 86_400_000))
    ax[0].plot([kdt(v) for v in t15[a15::4]], z15["c"][a15::4], color="#555", lw=0.7)
    ax[0].set_ylabel("BTCUSDT perp")
    cols = ["#2c7bb6", "#fdae61", "#7b3294", "#1a9850"]
    for i, n in enumerate(names):
        tf = n.split("|")[2]
        z = np.load(out / f"bars_{tf}.npz")
        a = int(np.searchsorted(z["t"], z["t"][-1] - 183 * 86_400_000))
        xs, ys = curve(trades[n]["m6"], z["t"], int(z["a"]), z["t"][a])
        ax[1].step(xs, ys, where="post", color=cols[i % 4], lw=1.6, label=f"Shinobu {n}: {ys[-1]:+.0f}%")
    for nm, col in (("B0+S1", "#888"), ("A8+S1", "#555")):
        tr, a = base_tr[nm]
        eq, xs, ys = 1.0, [kdt(b["t"][a])], [0.0]
        for x in tr:
            eq *= 1 + LEV * x["r"]
            xs.append(kdt(b["t"][min(x["x"], len(b["t"]) - 1)]))
            ys.append((eq - 1) * 100)
        lab = {"B0+S1": "today's SCR 3-level baseline", "A8+S1": "today's SCR 3-level A8 confirm"}[nm]
        ax[1].step(xs, ys, where="post", color=col, lw=1.2, ls="--", label=f"{lab}: {ys[-1]:+.0f}%")
    ax[1].axhline(0, color="#999", lw=0.6)
    ax[1].set_ylabel("5x return %")
    ax[1].legend(fontsize=9, loc="upper left")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax[0].set_title("Shinobu SRC port vs today's SCR 3-level, BTCUSDT perp, last 6 months - 5x, fees 0.05%/side "
                    "+ funding, chase limit on", loc="left", fontsize=10)  # fmt: skip
    for q in ax:
        q.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "equity_m6.png", dpi=100)
    print("charts", [str(p) for p in paths], out / "equity_m6.png")


if __name__ == "__main__":
    main()
