"""140번 차트: L2(ETH/BTC 비율 문 + AS4) 최근 30일 매매. 가격 + Stoch/CCI/RSI + ETH/BTC 비율 패널.

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python <이 파일> <출력 png>
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from btc_eth_leadlag import align, ema
from btc_fractal import cci, load5, resample, rsi, stoch
from btc_ml_bracket_grid import RULES, run_ml
from btc_ml_futures_reprice import futures_on
from btc_ml_regime_gate import on_15m
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model
from btc_strategy_combo import RES, ts

KST = timezone(timedelta(hours=9))
BUY, SELL, INK, GRID = "#1a9850", "#d73027", "#333333", "#e6e6e6"
M5, BAR, DAYS = 300_000, 900_000, 30


def main() -> None:
    out = Path(sys.argv[1])
    fb = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    fe = load5(Path("~/.minerva/data/eth/futures/klines_5m.csv").expanduser())
    bd, ed = align(resample(fb, 288), resample(fe, 288))
    rt = ed["c"] / bd["c"]
    em = ema(rt, 20)
    sl5 = np.r_[np.full(5, np.nan), rt[5:] - rt[:-5]]
    with np.errstate(invalid="ignore"):
        up_d, dn_d = (rt > em) & (sl5 > 0), (rt < em) & (sl5 < 0)
    dclose = bd["t"] + 288 * M5
    z = np.load(RES / "ml15/pred.npz")
    tp_t = z["t"]
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    pm = np.mean(ps, 0)
    P, _ = futures_on(tp_t, {k: z[k] for k in "ohlc"})
    P["t"] = tp_t
    end = float(tp_t[-1])
    f5k = {k: v[fb["t"] <= end + BAR] for k, v in fb.items()}
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp_t)
    G = Grid(P, f5k)
    i0 = int(np.searchsorted(tp_t, ts(2023)))
    up = on_15m(dclose, up_d.astype(float), tp_t) > 0.5
    dn = on_15m(dclose, dn_d.astype(float), tp_t) > 0.5
    q, k, _, _ = RULES["AS4"]
    li, si = rule_masks(pm, tp_t, q, k)
    Lm, Sm = li & (tr4 > 0), si & (tr4 < 0)
    grp = run_ml(G, Lm & up, Sm & dn, i0, Model(True, 0.0005), 0.015, "1.5", None)
    lo = int(np.searchsorted(tp_t, end - DAYS * 86_400_000))
    view = max(0, lo - 16)
    c, h, low = P["c"], P["h"], P["l"]
    x = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in tp_t])
    s_, cc_, r_ = stoch(h, low, c, 5), cci(h, low, c, 20), rsi(c, 14)
    fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True,
                           gridspec_kw={"height_ratios": [4, 1, 1, 1, 1.2]})
    p = ax[0]
    p.plot(x[view:], c[view:], color=INK, lw=0.8)
    for i in range(view, len(tp_t) - 1):
        if up[i] or dn[i]:
            p.axvspan(x[i], x[i + 1], color=BUY if up[i] else SELL, alpha=0.05, lw=0)
    wins = n = 0
    for j, s, segs, b in grp:
        if j < lo:
            continue
        px, xp = segs[0][1], segs[0][4]
        net = (xp / px - 1) * s - 0.0007
        n += 1
        wins += net > 0
        mk = "o" if s > 0 else "^"
        p.scatter(x[j], px, marker=mk, s=110, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
        p.scatter(x[b], xp, marker=mk, s=110, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
        p.plot([x[j], x[b]], [px, xp], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        p.annotate(f"{net * 100:+.2f}%", (x[b], xp), xytext=(4, 6) if net > 0 else (-8, -16),
                   textcoords="offset points", fontsize=9, fontweight="bold",
                   color=BUY if net > 0 else SELL)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    p.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    p.set_title(f"BTCUSDT futures 15m, last {DAYS} days to predictions end: test 140 L2 "
                f"(AS4 + ETH/BTC ratio gate; shade green = longs allowed, red = shorts allowed)  |  "
                f"{wins}/{n} wins, approx. net (KST)", loc="left", color=INK, fontsize=10)
    for a_, (name, v, lines) in zip(ax[1:4], (("Stoch 5", s_, (20, 80)), ("CCI 20", cc_, (-100, 100)),
                                              ("RSI 14", r_, (30, 70)))):
        a_.plot(x[view:], v[view:], color=INK, lw=0.6)
        for ln in lines:
            a_.axhline(ln, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(name, color=INK)
    dx = mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in dclose])
    dm = dclose >= tp_t[view]
    ax[4].plot(dx[dm], rt[dm], color=INK, lw=1.0, marker=".", label="ETH/BTC (daily close)")
    ax[4].plot(dx[dm], em[dm], color="#4575b4", lw=1.0, label="EMA 20")
    ax[4].legend(loc="upper left", frameon=False, fontsize=8)
    ax[4].set_ylabel("ETH/BTC", color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(n, wins)


if __name__ == "__main__":
    main()
