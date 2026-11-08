"""DMI 시험 차트: 가격 + Stoch·CCI·RSI + 신호 시간대의 ADX·PDI·MDI 패널, 자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 10배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 dmi15_chart.py --out <폴더> 이름:구간 ...
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import dmi15_eval as V
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nonprice_chart as NC
import numpy as np
import pine_chart as C
import pine_eval as E
import pine_ports as P

LAB = {"fresh": "untouched 2020-03~2021-10", "design": "design 2022-01~2024-06",
       "val": "validation 2024-07~2025-10", "seen": "seen 2025-11~", "last30": "last 30 days"}  # fmt: skip


def ffill15(n: int, last15: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.full(n, np.nan)
    out[last15] = v
    idx = np.where(~np.isnan(out), np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    return out[idx]


def chart(b, key, tr, s, a, z, ms, lines, fname, title) -> None:
    n = len(b["c"])
    wa = max(0, a - 200 * ms // E.M15)
    fr = NC.to_frame(b, wa, z, ms, b["c"])
    m = fr["t"] >= b["t"][a]
    fw = {k: v[m] for k, v in fr.items()}
    fig, ax = plt.subplots(5, 1, figsize=(18, 13), sharex=True,
                           gridspec_kw={"height_ratios": [4.5, 1.1, 1.1, 1.1, 1.4]})  # fmt: skip
    p = ax[0]
    NC.candles(p, fw, ms)
    for t in tr:
        d = t["d"]
        mk = "o" if d > 0 else "^"
        x0, x1 = C.kx(b["t"][t["e"]])[0], C.kx(b["t"][t["x"]])[0]
        p.scatter(x0, t["ep"], marker=mk, s=90, zorder=6, edgecolor="white", lw=1.0,
                  color=C.BUY if d > 0 else C.SELL)  # fmt: skip
        p.scatter(x1, t["xp"], marker=mk, s=90, zorder=6, edgecolor="white", lw=1.0,
                  color=C.SELL if d > 0 else C.BUY)  # fmt: skip
        good = t["r"] > 0
        p.plot([x0, x1], [t["ep"], t["xp"]], color="#555555", lw=0.9, ls="-" if good else ":")
        if len(tr) <= 120:
            p.annotate(f"{t['r'] * 1000:+.1f}%", (x1, t["xp"]),
                       xytext=(4, 8) if good else (4, -14), textcoords="offset points",
                       fontsize=7, fontweight="bold", color=C.BUY if good else C.SELL)  # fmt: skip
    win = "-" if s["win"] is None else f"{s['win'] * 100:.0f}%"
    p.set_title(
        f"{key}: {s['n']} trades (L {s['nl']} / S {s['ns']}), win {win}, "
        f"10x {s['x10']['ret'] * 100:+.1f}% (MDD {s['x10']['mdd'] * 100:.0f}%, liq {s['x10']['liq']}), "
        f"5x {s['x5']['ret'] * 100:+.1f}%, 3x {s['x3']['ret'] * 100:+.1f}%, "
        f"1x {s['x1']['ret'] * 100:+.1f}%. Labels = net per trade at 10x (fees + funding)",
        loc="left", color=C.INK, fontsize=10)  # fmt: skip
    p.set_ylabel("BTCUSDT perp (Binance)", color=C.INK)
    C.style(p)
    p.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", ms=9, color=C.BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", ms=9, color=C.SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", ms=9, color=C.SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", ms=9, color=C.BUY, label="short close"),
    ], loc="upper right", ncol=4, frameon=False, fontsize=9)  # fmt: skip
    hh, ll, cc = fr["h"], fr["l"], fr["c"]
    tfl = f"{ms // 3_600_000}h"
    ind = ((f"Stoch 5 ({tfl})", P.stoch(cc, hh, ll, 5), (20, 80)),
           (f"CCI 20 ({tfl})", P.cci((hh + ll + cc) / 3, 20), (-100, 100)),
           (f"RSI 14 ({tfl})", P.rsi(cc, 14), (30, 70)))  # fmt: skip
    xh = C.kx(fr["t"] + ms)
    for q, (nm, v, lns) in zip(ax[1:4], ind):
        q.plot(xh[m], v[m], color="#2a78d6", lw=1)
        for ln in lns:
            q.axhline(ln, color="#999999", lw=0.8, ls="--")
        q.set_ylabel(nm, color=C.INK, fontsize=9)
        C.style(q)
    tf = key.split("@")[1].split("|")[0]
    last15, (pdi, mdi, adx) = lines[tf]
    xf = C.kx(b["t"][a:z] + E.M15)
    q = ax[4]
    for v, col, nm in ((adx, "#222222", "ADX"), (pdi, "#d73027", "PDI"), (mdi, "#2a78d6", "MDI")):
        q.plot(xf, ffill15(n, last15, v)[a:z], color=col, lw=1.1, label=nm)
    q.axhline(20, color="#999999", lw=0.8, ls="--")
    q.set_ylabel(f"DMI 14 ({tf})", color=C.INK, fontsize=9)
    q.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    C.style(q)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    fig.suptitle(title, x=0.01, ha="left", color=C.INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=85)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("items", nargs="+", help="이름:구간 (예: EN-A@1d|X1:val)")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b, W, sets, _, lines, B = V.setup()
    rows = []
    for it in a_.items:
        key, _, w = it.rpartition(":")
        a, z = W[w]
        tr = V.run(B, sets, key, a, z)
        s = V.summ(tr)
        ms = 4 * 3_600_000 if w == "last30" else DAYMS
        when = (f"{datetime.fromtimestamp(b['t'][a] / 1000, C.KST):%Y-%m-%d} ~ "
                f"{datetime.fromtimestamp((b['t'][z - 1] + E.M15) / 1000, C.KST):%Y-%m-%d} KST")  # fmt: skip
        safe = key.replace("|", "_").replace("@", "_")
        f = out / f"chart_{safe}_{w}.png"
        cand = "daily" if ms == DAYMS else "4h"
        chart(b, key, tr, s, a, z, ms, lines, f,
              f"{key} - {LAB[w]} {when}, {cand} candles, fills on 15m")  # fmt: skip
        print(f)
        rows.append((f"{key} {LAB[w]}", tr, a, z))
    C.equity_chart(b, rows, out / "equity.png",
                   "Equity (mark-to-market on 15m closes), 10x / 3x / 1x vs buy & hold", log=True)
    print(out / "equity.png")


DAYMS = 24 * 3_600_000

if __name__ == "__main__":
    main()
