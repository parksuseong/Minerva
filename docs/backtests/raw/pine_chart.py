"""파인 10개 검증 차트: 결선(설계로 고름) + 표본 밖 상위(표본 밖으로 고름, 낙관적) 매매·자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 10배 손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 pine_chart.py --out <폴더> 3D:R/S0 ...
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pine_eval as E
import pine_ports as P

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, MUTED, GRID = "#222222", "#888888", "#ebebeb"
SER = {"10x": "#d73027", "3x": "#eb6834", "1x": "#2a78d6", "B&H 1x": "#888888"}
H1 = 3_600_000


def kx(ms):
    return mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in np.atleast_1d(ms)])


def style(ax):
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)


def to_h1(b: dict, a: int, z: int) -> dict:
    t = b["t"][a:z]
    k = t // H1
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    en = np.r_[st[1:], len(t)]
    return {
        "t": k[st] * H1,
        "o": b["o"][a:z][st],
        "h": np.array([b["h"][a:z][s:e].max() for s, e in zip(st, en)]),
        "l": np.array([b["l"][a:z][s:e].min() for s, e in zip(st, en)]),
        "c": b["c"][a:z][en - 1],
    }


def candles(ax, h1):
    x = kx(h1["t"])
    w = (1 / 24) * 0.7
    up = h1["c"] >= h1["o"]
    for m, col in ((up, "#9a9a9a"), (~up, "#4a4a4a")):
        ax.vlines(x[m] + w / 2, h1["l"][m], h1["h"][m], color=col, lw=0.5)
        ax.bar(
            x[m],
            np.abs(h1["c"][m] - h1["o"][m]) + 1e-6,
            bottom=np.minimum(h1["o"], h1["c"])[m],
            width=w,
            align="edge",
            color=col,
            lw=0,
        )


def entry_ms(b, tr) -> float:
    return float(b["t"][tr["e"]])


def exit_ms(b, tr) -> float:
    return float(b["t"][tr["x"]])


def trade_chart(b, name, var, tr, s, a, z, fname, title, warm_from):
    # 지표는 예열을 위해 창 앞 200시간부터 계산
    wa = max(warm_from, a - 800)
    h1 = to_h1(b, wa, z)
    m = h1["t"] >= b["t"][a]
    hw = {k: v[m] for k, v in h1.items()}
    fig, ax = plt.subplots(
        4, 1, figsize=(18, 11), sharex=True, gridspec_kw={"height_ratios": [4.5, 1.1, 1.1, 1.1]}
    )
    p = ax[0]
    candles(p, hw)
    for t in tr:
        d = t["d"]
        mk = "o" if d > 0 else "^"
        x0 = kx(entry_ms(b, t))[0]
        x1 = kx(exit_ms(b, t))[0]
        p.scatter(
            x0,
            t["ep"],
            marker=mk,
            s=120,
            zorder=6,
            edgecolor="white",
            lw=1.2,
            color=BUY if d > 0 else SELL,
        )
        p.scatter(
            x1,
            t["xp"],
            marker=mk,
            s=120,
            zorder=6,
            edgecolor="white",
            lw=1.2,
            color=SELL if d > 0 else BUY,
        )
        good = t["r"] > 0
        p.plot([x0, x1], [t["ep"], t["xp"]], color="#555555", lw=1.0, ls="-" if good else ":")
        liq = 10 * t["worst"] <= E.LIQ
        lab = "LIQ" if liq else f"{t['r'] * 1000:+.1f}%"
        p.annotate(
            lab,
            (x1, t["xp"]),
            xytext=(4, 8) if good else (4, -14),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=BUY if good else SELL,
        )
    x10, x1_ = s["x10"], s["x1"]
    win = "-" if s["win"] is None else f"{s['win'] * 100:.0f}%"
    p.set_title(
        f"{name} [{var}]: {s['n']} trades (L {s['nl']} / S {s['ns']}), win {win}, "
        f"10x {x10['ret'] * 100:+.1f}% (MDD {x10['mdd'] * 100:.0f}%, liquidations {x10['liq']}), "
        f"3x {s['x3']['ret'] * 100:+.1f}%, 1x {x1_['ret'] * 100:+.1f}%. "
        "Labels = net per trade at 10x (fees + funding)",
        loc="left",
        color=INK,
        fontsize=10,
    )
    p.set_ylabel("BTCUSDT perp (Binance)", color=INK)
    style(p)
    p.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", ms=9, color=BUY, label="long open"),
            plt.Line2D([], [], marker="o", ls="", ms=9, color=SELL, label="long close"),
            plt.Line2D([], [], marker="^", ls="", ms=9, color=SELL, label="short open"),
            plt.Line2D([], [], marker="^", ls="", ms=9, color=BUY, label="short close"),
        ],
        loc="upper right",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    hh, ll, cc = h1["h"], h1["l"], h1["c"]
    st = P.stoch(cc, hh, ll, 5)
    ind = (
        ("Stoch 5 (1h)", st, (20, 80)),
        ("CCI 20 (1h)", P.cci((hh + ll + cc) / 3, 20), (-100, 100)),
        ("RSI 14 (1h)", P.rsi(cc, 14), (30, 70)),
    )
    xh = kx(h1["t"] + H1)
    for q, (nm, v, lines) in zip(ax[1:], ind):
        q.plot(xh[m], v[m], color="#2a78d6", lw=1)
        for ln in lines:
            q.axhline(ln, color="#999999", lw=0.8, ls="--")
        q.set_ylabel(nm, color=INK, fontsize=9)
        style(q)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=90)
    plt.close(fig)


def mtm_curve(b, tr, a, z, L) -> np.ndarray:
    eq = np.ones(z - a)
    cur = 1.0
    c = b["c"]
    last = a
    dead = False
    for t in tr:
        e, x = t["e"], t["x"]
        eq[last - a : e - a] = cur
        if dead:
            continue
        seg = c[e : x + 1]
        mt = t["d"] * (seg / t["ep"] - 1) - E.FEE
        eq[e - a : x + 1 - a] = cur * np.maximum(1 + L * mt, 0)
        if L * t["worst"] <= E.LIQ:
            cur = 0.0
            dead = True
        else:
            cur *= 1 + L * t["r"]
        eq[min(x + 1, z) - a - 1] = cur
        last = x + 1
    eq[last - a :] = cur
    return eq


def equity_chart(b, rows, fname, title, log=False):
    fig, ax = plt.subplots(len(rows), 1, figsize=(14, 3.6 * len(rows) + 0.6), squeeze=False)
    for p, (lab, tr, a, z) in zip(ax[:, 0], rows):
        x = kx(b["t"][a:z] + E.M15)
        for L, k in ((10.0, "10x"), (3.0, "3x"), (1.0, "1x")):
            y = mtm_curve(b, tr, a, z, L)
            p.plot(x, np.maximum(y, 1e-3) if log else y, color=SER[k], lw=1.3, label=k)
        p.plot(x, b["c"][a:z] / b["o"][a], color=SER["B&H 1x"], lw=1.1, ls="--", label="B&H 1x")
        p.axhline(1, color="#bbbbbb", lw=0.8)
        if log:
            p.set_yscale("log")
        p.set_title(lab, loc="left", color=INK, fontsize=10)
        p.set_ylabel("equity (start = 1)", color=INK, fontsize=9)
        p.legend(loc="upper left", frameon=False, fontsize=8, ncol=4)
        p.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d" if log else "%m-%d"))
        style(p)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("items", nargs="+", help="이름:변형[:tag]")
    ap.add_argument("--long", nargs="*", default=[], help="장기 자산 곡선을 그릴 이름")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b, S, B, _ = E.load_all(out)
    ha, hz = E.win_idx(b["t"], "holdout")
    la = int(np.searchsorted(b["t"], b["t"][-1] + E.M15 - 30 * E.DAY))
    rows = []
    for it in a_.items:
        parts = it.split(":")
        name, var = parts[0], parts[1]
        tag = parts[2] if len(parts) > 2 else ""
        v = E.parse_v(var)
        for w, (a, z) in (("holdout", (ha, hz)), ("last30", (la, hz))):
            tr = E.simulate(B, S[name], v, a, z)
            s = E.summarize(tr)
            f = out / f"chart_{name}_{w}.png"
            when = (
                f"{datetime.fromtimestamp(b['t'][a] / 1000, KST):%Y-%m-%d %H:%M} ~ "
                f"{datetime.fromtimestamp((b['t'][z - 1] + E.M15) / 1000, KST):%m-%d %H:%M} KST"
            )
            trade_chart(
                b,
                name,
                var,
                tr,
                s,
                a,
                z,
                f,
                f"{name} {tag} - {w} {when}, 1h candles, fills on 15m",
                ha - 4000,
            )
            if w == "holdout":
                rows.append((f"{name} [{var}] {tag} holdout", tr, a, z))
            print(f)
    equity_chart(
        b,
        rows,
        out / "equity_holdout.png",
        "Holdout equity (mark-to-market on 15m closes), 10x / 3x / 1x vs buy & hold",
    )
    if a_.long:
        la_, lz_ = E.win_idx(b["t"], "long")
        lrows = []
        for it in a_.items:
            name, var = it.split(":")[:2]
            if name in a_.long:
                tr = E.simulate(B, S[name], E.parse_v(var), la_, lz_)
                lrows.append(
                    (
                        f"{name} [{var}] 2020-08 ~ 2025-11 (log scale; 10x line hits 0 at the "
                        "first liquidation)",
                        tr,
                        la_,
                        lz_,
                    )
                )
        equity_chart(b, lrows, out / "equity_long.png", "Long-window robustness", log=True)
        print(out / "equity_long.png")


if __name__ == "__main__":
    main()
