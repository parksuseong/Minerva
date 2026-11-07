"""137번 차트: (a) 전진 최적화 표본 밖 곡선 vs BTC 보유·R1, (b) 칸 하나의 최근 N일 매매(이치모쿠 구름, SCR·MACD 패널).

실행: PYTHONPATH=<numpy·matplotlib>:src:docs/backtests/raw python <이 파일> <t137 폴더> [칸 키] [일수]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import btc_scr_macd_ichi as T
from btc_signal_scan import load

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"


def equity_chart(res: dict, out: Path) -> None:
    comp = res["compare"]
    ys = sorted(comp, key=int)
    lab = [str(y) for y in ys]
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, col, name in (("wf", "#d73027", "walk-forward pick (OOS)"), ("r1", "#4575b4", "R1"),
                         ("btc_hold", "#999999", "BTC hold")):
        v = [1.0]
        for y in ys:
            r = comp[y][k]
            v.append(v[-1] * (1 + (r or 0) / 100))
        ax.plot(range(len(v)), v, marker="o", color=col, label=name)
    ax.set_xticks(range(len(ys) + 1))
    ax.set_xticklabels(["start"] + lab)
    ax.set_yscale("log")
    ax.grid(color=GRID)
    ax.legend(frameon=False)
    ax.set_title("Test 137: SCR+MACD+Ichimoku walk-forward (pick by prior 2y, trade next year), 1x, "
                 "year-end compounding", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def parse(key: str) -> dict:
    fn, sc, mc, ic, zx = key.split("|")
    _, ls, w = sc[3:].split("-")
    k = int(sc[3])
    scr = (k, ls, int(w[1:]))
    macd = None if mc == "M-" else (tuple(int(x) for x in mc[1:].split("-")[0].split(".")), mc.split("-")[1])
    ichi = None if ic == "I-" else (tuple(int(x) for x in ic[1:].split("-")[0].split(".")), ic.split("-")[1])
    return {"frame": fn, "scr": scr, "macd": macd, "ichi": ichi, "exit": zx}


def trade_chart(key: str, arr: np.ndarray, days: int, out: Path) -> None:
    d5 = load(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    m = parse(key)
    F = T.Frame(d5, T.FRAMES[m["frame"]], d5["t"])
    t5, o5 = d5["t"], d5["o"]
    end = F.t[-1]
    lo = int(np.searchsorted(F.t, end - days * 86_400_000))
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in F.close_t])
    xs = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in t5])
    e5, side, g, x5 = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2], arr[:, 3].astype(int)
    sel = t5[e5] >= F.t[lo]
    fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1]})
    p = ax[0]
    p.plot(x[lo:], F.c[lo:], color=INK, lw=0.8)
    n1, n2, n3 = (m["ichi"][0] if m["ichi"] else (9, 26, 52))
    ten, kij = T.mid(F.h, F.l, n1), T.mid(F.h, F.l, n2)
    a_, b_ = T.shift((ten + kij) / 2, n2), T.shift(T.mid(F.h, F.l, n3), n2)
    p.fill_between(x[lo:], a_[lo:], b_[lo:], where=a_[lo:] >= b_[lo:], color=BUY, alpha=0.12, lw=0)
    p.fill_between(x[lo:], a_[lo:], b_[lo:], where=a_[lo:] < b_[lo:], color=SELL, alpha=0.12, lw=0)
    p.plot(x[lo:], kij[lo:], color="#4575b4", lw=0.7, label=f"kijun {n2}")
    wins = 0
    for ei, si, gi, xi in zip(e5[sel], side[sel], g[sel], x5[sel]):
        net = gi - 2 * T.COST
        wins += net > 0
        px = o5[ei]
        pxo = px * (1 + (gi) * si)
        mk = "o" if si > 0 else "^"
        p.scatter(xs[ei], px, marker=mk, s=100, zorder=5, edgecolor="white", color=BUY if si > 0 else SELL)
        p.scatter(xs[min(xi, len(xs) - 1)], pxo, marker=mk, s=100, zorder=5, edgecolor="white",
                  color=SELL if si > 0 else BUY)
        p.plot([xs[ei], xs[min(xi, len(xs) - 1)]], [px, pxo], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (xs[min(xi, len(xs) - 1)], pxo), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=9, fontweight="bold", color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures, last {days} days: {key}  |  {wins}/{int(sel.sum())} wins, net of fees+funding "
                f"(cloud shaded, KST)", loc="left", fontsize=10, color=INK)
    fa, sl_, sg = (m["macd"][0] if m["macd"] else (12, 26, 9))
    mm = T.ema(F.c, fa) - T.ema(F.c, sl_)
    hist = mm - T.ema(mm, sg)
    for a_x, (name, v, lines) in zip(ax[1:4], (("Stoch 5", F.sto, (20, 80)), ("CCI 20", F.cci, (-100, 100)),
                                                ("RSI 14", F.rsi, (30, 70)))):
        a_x.plot(x[lo:], v[lo:], color=INK, lw=0.6)
        for ln in lines:
            a_x.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_x.set_ylabel(name, color=INK)
    ax[4].bar(x[lo:], hist[lo:], width=(x[1] - x[0]) * 0.8, color=np.where(hist[lo:] >= 0, BUY, SELL))
    ax[4].plot(x[lo:], mm[lo:], color=INK, lw=0.6)
    ax[4].set_ylabel(f"MACD {fa}/{sl_}/{sg}", color=INK)
    for a_x in ax:
        a_x.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_x.spines[sd].set_visible(False)
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(key, int(sel.sum()), "trades", int(wins), "wins")


def main() -> None:
    d = Path(sys.argv[1])
    res = json.loads((d / "result.json").read_text())
    equity_chart(res, d / "t137_walkforward.png")
    picks = json.loads((d / "picks.json").read_text())
    z = np.load(d / "picks.npz")
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    key = sys.argv[2] if len(sys.argv) > 2 else picks["picks"]["2026"]
    yk = next(y for y, k in picks["picks"].items() if k == key) if key in picks["picks"].values() else None
    if yk is None:
        print("key not among picks; pass a pick key")
        return
    trade_chart(key, z[f"y{yk}"], days, d / f"t137_{yk}_pick.png")


if __name__ == "__main__":
    main()
