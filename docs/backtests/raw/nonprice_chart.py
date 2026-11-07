"""비가격 가설 차트: 검증·최근 30일 매매(가격 + Stoch·CCI·RSI + 비가격 동인 패널)와 자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 10배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 nonprice_chart.py --out <폴더> H5:T1/S1 ...
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nonprice_eval as N
import numpy as np
import pine_chart as C
import pine_eval as E
import pine_ports as P
import pine_tv_eval as T
from nonprice_data import load

DRIVER = {
    "H1": ("zOI4", "OI 4h change z (30d)"), "H10": ("zOI4", "OI 4h change z (30d)"),
    "H2": ("oi1", "OI 1h log change"), "H3": ("oi1", "OI 1h log change"),
    "H4": ("prem8", "premium index 8h mean"), "H5": ("zP", "premium 1h z (30d)"),
    "H6": ("dprem", "premium 1h change"), "H7": ("tb1", "taker buy share 1h"),
    "H8": ("zD", "top-trader vs crowd L/S divergence z"), "H9": ("zC", "crowd L/S z (30d)"),
    "H11": ("f_last", "last settled funding"),
}  # fmt: skip


def to_frame(b: dict, a: int, z: int, ms: int, feat: np.ndarray) -> dict:
    t = b["t"][a:z]
    k = t // ms
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    en = np.r_[st[1:], len(t)]
    return {
        "t": k[st] * ms,
        "o": b["o"][a:z][st],
        "h": np.maximum.reduceat(b["h"][a:z], st),
        "l": np.minimum.reduceat(b["l"][a:z], st),
        "c": b["c"][a:z][en - 1],
        "f": feat[a:z][en - 1],
    }


def candles(ax, fr, ms):
    x = C.kx(fr["t"])
    w = ms / 86_400_000 * 0.7
    up = fr["c"] >= fr["o"]
    for m, col in ((up, "#9a9a9a"), (~up, "#4a4a4a")):
        ax.vlines(x[m] + w / 2, fr["l"][m], fr["h"][m], color=col, lw=0.5)
        ax.bar(x[m], np.abs(fr["c"][m] - fr["o"][m]) + 1e-6, bottom=np.minimum(fr["o"], fr["c"])[m],
               width=w, align="edge", color=col, lw=0)  # fmt: skip


def trade_chart(b, h, var, tr, s, a, z, ms, fname, title, thr_lines):
    key, dlab = DRIVER[h]
    feat = b["f_" + key]
    wa = max(0, a - 200 * ms // E.M15)
    fr = to_frame(b, wa, z, ms, feat)
    m = fr["t"] >= b["t"][a]
    fw = {k: v[m] for k, v in fr.items()}
    fig, ax = plt.subplots(5, 1, figsize=(18, 13), sharex=True,
                           gridspec_kw={"height_ratios": [4.5, 1.1, 1.1, 1.1, 1.3]})  # fmt: skip
    p = ax[0]
    candles(p, fw, ms)
    lab_each = len(tr) <= 120
    for t in tr:
        d = t["d"]
        mk = "o" if d > 0 else "^"
        x0 = C.kx(b["t"][t["e"]])[0]
        x1 = C.kx(b["t"][t["x"]])[0]
        p.scatter(x0, t["ep"], marker=mk, s=90, zorder=6, edgecolor="white", lw=1.0,
                  color=C.BUY if d > 0 else C.SELL)  # fmt: skip
        p.scatter(x1, t["xp"], marker=mk, s=90, zorder=6, edgecolor="white", lw=1.0,
                  color=C.SELL if d > 0 else C.BUY)  # fmt: skip
        good = t["r"] > 0
        p.plot([x0, x1], [t["ep"], t["xp"]], color="#555555", lw=0.9, ls="-" if good else ":")
        if lab_each:
            liq = 10 * t["worst"] <= E.LIQ
            p.annotate("LIQ" if liq else f"{t['r'] * 1000:+.1f}%", (x1, t["xp"]),
                       xytext=(4, 8) if good else (4, -14), textcoords="offset points",
                       fontsize=7, fontweight="bold", color=C.BUY if good else C.SELL)  # fmt: skip
    win = "-" if s["win"] is None else f"{s['win'] * 100:.0f}%"
    p.set_title(
        f"{h} [{var}]: {s['n']} trades (L {s['nl']} / S {s['ns']}), win {win}, "
        f"10x {s['x10']['ret'] * 100:+.1f}% (MDD {s['x10']['mdd'] * 100:.0f}%, liquidations "
        f"{s['x10']['liq']}), 3x {s['x3']['ret'] * 100:+.1f}%, 1x {s['x1']['ret'] * 100:+.1f}%. "
        "Labels = net per trade at 10x (fees + funding)",
        loc="left",
        color=C.INK,
        fontsize=10,
    )
    p.set_ylabel("BTCUSDT perp (Binance)", color=C.INK)
    C.style(p)
    p.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", ms=9, color=C.BUY, label="long open"),
        plt.Line2D([], [], marker="o", ls="", ms=9, color=C.SELL, label="long close"),
        plt.Line2D([], [], marker="^", ls="", ms=9, color=C.SELL, label="short open"),
        plt.Line2D([], [], marker="^", ls="", ms=9, color=C.BUY, label="short close"),
    ], loc="upper right", ncol=4, frameon=False, fontsize=9)  # fmt: skip
    hh, ll, cc = fr["h"], fr["l"], fr["c"]
    tf = f"{ms // 3_600_000}h"
    ind = ((f"Stoch 5 ({tf})", P.stoch(cc, hh, ll, 5), (20, 80)),
           (f"CCI 20 ({tf})", P.cci((hh + ll + cc) / 3, 20), (-100, 100)),
           (f"RSI 14 ({tf})", P.rsi(cc, 14), (30, 70)))  # fmt: skip
    xh = C.kx(fr["t"] + ms)
    for q, (nm, v, lines) in zip(ax[1:4], ind):
        q.plot(xh[m], v[m], color="#2a78d6", lw=1)
        for ln in lines:
            q.axhline(ln, color="#999999", lw=0.8, ls="--")
        q.set_ylabel(nm, color=C.INK, fontsize=9)
        C.style(q)
    # 비가격 동인: 15분봉 값 그대로
    xf = C.kx(b["t"][a:z] + E.M15)
    ax[4].plot(xf, feat[a:z], color="#7b3294", lw=0.7)
    for ln in thr_lines:
        ax[4].axhline(ln, color="#d73027", lw=0.8, ls="--")
    ax[4].set_ylabel(dlab + " (15m)", color=C.INK, fontsize=9)
    C.style(ax[4])
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    fig.suptitle(title, x=0.01, ha="left", color=C.INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=85)
    plt.close(fig)


THR_KEYS = {"H1": ("zOI4", "95"), "H10": ("zOI4", "97"), "H2": ("oi1", "5"), "H3": ("oi1", "5"),
            "H4": ("prem8", "2", "98"), "H5": ("zP", "0.5", "99.5"), "H6": ("dprem", "0.5", "99.5"),
            "H7": ("tb1", "1", "99"), "H8": ("zD", "1", "99"), "H9": ("zC", "0.5", "99.5"),
            "H11": None}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("items", nargs="+", help="H:변형[:tag]")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b = load(out)
    W = N.windows(b)
    g = json.loads((out / "grid.json").read_text())
    Sx, thr = N.build_hyps(b, W, g["thresholds"])
    B = N.bars(b)
    rows = []
    for it in a_.items:
        parts = it.split(":")
        h, var = parts[0], parts[1]
        tag = parts[2] if len(parts) > 2 else ""
        v = T.parse_v(var)
        tk = THR_KEYS[h]
        lines = [thr[tk[0]][q] for q in tk[1:]] if tk else [0.0]
        for w, ms in (("val", 4 * 3_600_000), ("last30", 3_600_000)):
            a, z = W[w]
            tr = T.simulate(B, Sx[h], v, a, z)
            s = N.summ(tr)
            when = (f"{datetime.fromtimestamp(b['t'][a] / 1000, C.KST):%Y-%m-%d} ~ "
                    f"{datetime.fromtimestamp((b['t'][z - 1] + E.M15) / 1000, C.KST):%Y-%m-%d %H:%M} KST")  # fmt: skip
            lab = "validation" if w == "val" else "last 30 days (inside seen period)"
            f = out / f"chart_{h}_{var.replace('/', '-')}_{w}.png"
            trade_chart(b, h, var, tr, s, a, z, ms, f,
                        f"{h} {tag} - {lab} {when}, {ms // 3_600_000}h candles, fills on 15m",
                        lines)  # fmt: skip
            print(f)
        for w in ("val", "seen"):
            a, z = W[w]
            rows.append((f"{h} [{var}] {tag} {w}", T.simulate(B, Sx[h], v, a, z), a, z))
    C.equity_chart(b, rows, out / "equity.png",
                   "Equity (mark-to-market on 15m closes), 10x / 3x / 1x vs buy & hold; "
                   "validation 2025-01~10 and seen 2025-11~", log=True)  # fmt: skip
    print(out / "equity.png")


if __name__ == "__main__":
    main()
