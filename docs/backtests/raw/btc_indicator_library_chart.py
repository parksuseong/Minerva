"""139번 차트: L1 가중치 열지도, P1 최선 vs AS4 자산 곡선, 최선 규칙 최근 30일 매매.

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python <이 파일> <t139 폴더> <t128 series.npz> <P1 변형> <규칙 이름> <출력 폴더>
규칙 이름: P1 변형(예: XGB-q95) 또는 "P2:<번호>"(final3 순서).
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

import btc_indicator_library as L

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, GRID = "#333333", "#e6e6e6"


def daily_equity(tr, t15, t5, c5, d0, d1):
    days = np.arange(d0 // L.DAY, d1 // L.DAY + 1) * L.DAY
    eq = np.ones(len(days))
    cur = 1.0
    k = 0
    for i, s, net, xt in tr:
        e_t = t15[i] + L.B15 + L.M5
        while k < len(days) and days[k] + L.DAY <= e_t:
            eq[k] = cur
            k += 1
        e = int(np.searchsorted(t5, e_t))
        px = c5[e - 1]
        while k < len(days) and days[k] + L.DAY <= xt:
            j = int(np.searchsorted(t5, days[k] + L.DAY)) - 1
            eq[k] = cur * (1 + (c5[j] / px - 1) * s)
            k += 1
        cur *= 1 + net
    eq[k:] = cur
    return days, eq


def main() -> None:
    d = Path(sys.argv[1])
    as4f, var, rule, out = Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
    W = json.load(open(d / "p1_weights.json"))
    names = W["names"]
    yrs = sorted(W["weights"])
    M = np.array([W["weights"][y] for y in yrs])
    keep = np.flatnonzero(np.abs(M).max(0) > 0)
    order = keep[np.argsort(-np.abs(M[:, keep]).mean(0))][:40]
    fig, ax = plt.subplots(figsize=(10, 12))
    v = np.abs(M[:, order]).max()
    im = ax.imshow(M[:, order].T, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(yrs)), yrs)
    ax.set_yticks(range(len(order)), [names[i] for i in order], fontsize=8)
    ax.set_title("L1-logistic weights (standardised), top 40 features by mean |weight|; red = favours long", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.5)
    fig.tight_layout()
    fig.savefig(out / "t139_weights.png", dpi=110)
    plt.close(fig)

    base = Path("~/.minerva/data/btc").expanduser()
    F = np.load(d / "feat.npz", allow_pickle=True)
    t15 = F["t"]
    O = np.load(d / "outcomes.npz")  # noqa: E741
    ln, sn, lx, sx, t5 = O["ln"], O["sn"], O["lx"], O["sx"], O["t5"]
    f5 = L.load_fut5(base / "futures/klines_5m.csv")
    c5 = f5["c"]
    P = np.load(d / "p1_pred.npz")
    model, q = var.split("-q")
    Ls, Ss = L.signals_from(P[model], t15, int(q) / 100)
    i23 = int(np.searchsorted(t15, L.ts(2023)))
    iend = len(t15) - 2
    tr = L.walk(Ls, Ss, ln, sn, lx, sx, t5, t15, i23, iend)
    days, eq = daily_equity(tr, t15, t5, c5, int(L.ts(2023)), int(t15[iend]))
    A = np.load(as4f, allow_pickle=True)
    adt, adr = A["AS4|bc|mean__dt"], A["AS4|bc|mean__dr"]
    m = adt >= L.ts(2023)
    aeq = np.cumprod(1 + adr[m])
    x1 = mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in days])
    x2 = mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in adt[m]])
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for x, e, lab, col in ((x1, eq, f"P1 {var} (market, 5-min delay)", "#d73027"),
                           (x2, aeq, "AS4 (test 128, limit, realistic)", "#4575b4")):
        ax[0].plot(x, e, color=col, lw=1.2, label=lab)
        ax[1].plot(x, (e / np.maximum.accumulate(e) - 1) * 100, color=col, lw=0.9)
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False)
    ax[0].set_title("BTCUSDT futures 2023~: 1x daily mark-to-market equity", loc="left")
    ax[1].set_ylabel("drawdown %")
    for a in ax:
        a.grid(color=GRID)
    fig.tight_layout()
    fig.savefig(out / "t139_equity.png", dpi=110)
    plt.close(fig)

    # 최근 30일 매매 (최선 규칙)
    if rule.startswith("P2:"):
        R2 = json.load(open(d / "p2_result.json"))
        r = R2["final3"][int(rule[3:])]
        X = F["X"]
        i20 = int(np.searchsorted(t15, L.ts(2020) + 60 * L.DAY))
        Qv = np.nanquantile(X[i20:i23], (0.1, 0.25, 0.75, 0.9), axis=0)
        bn = [n.split("_", 1)[1] if n.split("_", 1)[0] in ("15m", "1h", "4h", "1d") else n for n in names]
        ml = np.ones(len(t15), bool)
        ms = np.ones(len(t15), bool)
        for f, k in r["key"]:
            x = X[:, f]
            with np.errstate(invalid="ignore"):
                ml &= x <= Qv[k, f] if k < 2 else x >= Qv[k, f]
            g, kk = f, k
            if bn[f] in L.DIRECTIONAL:
                kk = 3 - k
            elif bn[f] in ("pdi", "mdi"):
                g = names.index(names[f].replace("pdi", "mdi") if bn[f] == "pdi" else names[f].replace("mdi", "pdi"))
            x = X[:, g]
            with np.errstate(invalid="ignore"):
                ms &= x <= Qv[kk, g] if kk < 2 else x >= Qv[kk, g]
        el, es = ml & ~np.r_[False, ml[:-1]], ms & ~np.r_[False, ms[:-1]]
        title = "P2 rule: " + r["rule"].replace("분위", "pct")
    else:
        el, es = Ls, Ss
        title = f"P1 {var}"
    i0 = int(np.searchsorted(t15, t15[iend] - 30 * L.DAY))
    tr30 = L.walk(el, es, ln, sn, lx, sx, t5, t15, i0, iend)
    d15 = L.resample(f5, 3)
    from btc_fractal import cci, rsi, stoch
    h, lo, c = d15["h"], d15["l"], d15["c"]
    st, cc, rs = stoch(h, lo, c, 5), cci(h, lo, c, 20), rsi(c, 14)
    X15 = d15["t"]
    j = int(np.searchsorted(X15, t15[i0]))
    sl = slice(max(0, j - 8), len(c))
    xs = mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in X15[sl]])
    names_i = {n: i for i, n in enumerate(names)}
    Xf = F["X"]
    fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1]})
    ax[0].plot(xs, c[sl], color=INK, lw=0.8)
    wins = 0
    for i, s, net, xt in tr30:
        e_t = t15[i] + L.B15 + L.M5
        e = int(np.searchsorted(t5, e_t))
        pin = f5["o"][e]
        xe = mdates.date2num(datetime.fromtimestamp(e_t / 1000, KST))
        xx = mdates.date2num(datetime.fromtimestamp(xt / 1000, KST))
        pout = pin * (1 + (net + 2 * L.FEE) * s)
        mk = "o" if s > 0 else "^"
        ax[0].scatter(xe, pin, marker=mk, s=100, zorder=5, edgecolor="white", color=BUY if s > 0 else SELL)
        ax[0].scatter(xx, pout, marker=mk, s=100, zorder=5, edgecolor="white", color=SELL if s > 0 else BUY)
        ax[0].plot([xe, xx], [pin, pout], color="#777777", lw=1.1, ls="-" if net > 0 else ":")
        ax[0].annotate(f"{net * 100:+.2f}%", (xx, pout), xytext=(4, 6) if net > 0 else (-8, -16),
                       textcoords="offset points", fontsize=9, fontweight="bold", color=BUY if net > 0 else SELL)
        wins += net > 0
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUY, label="long open"),
               plt.Line2D([], [], marker="o", ls="", color=SELL, label="long close"),
               plt.Line2D([], [], marker="^", ls="", color=SELL, label="short open"),
               plt.Line2D([], [], marker="^", ls="", color=BUY, label="short close")]
    ax[0].legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    ax[0].set_title(f"BTCUSDT futures 15m, last 30 days: {title[:150]}  |  {wins}/{len(tr30)} wins, "
                    "net of fees + funding, ±1.5% (KST)", loc="left", fontsize=10)
    for a_, (nm, v_, lines) in zip(ax[1:4], (("Stoch 5", st, (20, 80)), ("CCI 20", cc, (-100, 100)),
                                             ("RSI 14", rs, (30, 70)))):
        a_.plot(xs, v_[sl], color=INK, lw=0.6)
        for ln_ in lines:
            a_.axhline(ln_, color="#999999", lw=0.8, ls="--")
        a_.set_ylabel(nm)
    xt15 = mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in t15[i0:]])
    for key, col in (("1h_adx", INK), ("1h_pdi", BUY), ("1h_mdi", SELL)):
        ax[4].plot(xt15, Xf[i0:, names_i[key]], color=col, lw=0.8, label=key)
    ax[4].legend(frameon=False, fontsize=8, ncol=3)
    ax[4].set_ylabel("ADX/DI (1h)")
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(out / "t139_trades.png", dpi=110)
    plt.close(fig)
    print("charts done", len(tr30), wins)


if __name__ == "__main__":
    main()
