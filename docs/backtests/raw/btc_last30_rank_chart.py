"""최근 30일 순위 차트: 순위 막대, 자산 곡선(10배·1배), 상위 기법 매매 차트(1시간 캔들 + Stoch·CCI·RSI).

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강.
실행: PYTHONPATH=<pyml>:src:docs/backtests/raw python3 <이 파일> <결과 폴더>
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import btc_fut_daily_regime as R
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from btc_fractal import cci, resample, rsi, stoch
from btc_last30_rank import BAR, ENS, SINGLES, TRACKER, equity, split_equity
from btc_scr_learned import bars

KST = timezone(timedelta(hours=9))
BUY, SELL = "#1a9850", "#d73027"
INK, MUTED, GRID = "#222222", "#888888", "#ebebeb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#e87ba4")
H1 = 3_600_000


def kx(ms):
    return mdates.date2num([datetime.fromtimestamp(v / 1000, KST) for v in np.atleast_1d(ms)])


def load(res: Path):
    rep = json.loads((res / "result.json").read_text())
    T = json.loads((res / "trades.json").read_text())
    rows = {}
    with (Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()).open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if x[-1] != "okx":
                rows[int(x[0])] = [float(v) for v in x[1:6]]
    for x in csv.reader((res / "fut5m_fapi.csv").open()):
        rows[int(x[0])] = [float(v) for v in x[1:6]]
    t = np.array(sorted(rows), float)
    v = np.array([rows[int(k)] for k in t])
    now = rep["info"]["now"]
    m = t < now
    d5 = {"t": t[m], "o": v[m, 0], "h": v[m, 1], "l": v[m, 2], "c": v[m, 3], "v": v[m, 4]}
    return rep, T, d5


def style(ax):
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)


def ranking(rep, res):
    names = sorted(SINGLES + ENS, key=lambda k: rep[k]["ret10"])
    v = [rep[k]["ret10"] for k in names]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names, v, color=[BUY if x > 0 else SELL for x in v], height=0.6)
    for i, (k, x) in enumerate(zip(names, v)):
        r = rep[k]
        ax.annotate(
            f"{x:+.1f}%  ({r['n']} trades, win {r['win'] if r['win'] is not None else '-'}%, "
            f"MDD {r['mdd10']:.0f}%)",
            (x, i),
            xytext=(5 if x >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.axvline(0, color="#999999", lw=0.8)
    bh = rep["info"]["buy_hold_pct"]
    ax.set_xlim(min(v) * 2.0, max(v) * 3.2)
    ax.set_title(
        f"BTCUSDT perp, last 30 days ({rep['info']['start_kst'][:16]} ~ {rep['info']['end_kst'][:16]} KST)"
        f"\n10x compounded, fees 0.05%/side + funding. Buy & hold 1x {bh:+.1f}% / 10x {bh * 10:+.0f}%",
        loc="left",
        color=INK,
        fontsize=10,
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(res / "last30_ranking.png", dpi=110)
    plt.close(fig)


def equity_chart(rep, T, d, a, res, keys):
    x = kx(d["t"][a:] + BAR)
    c = d["c"]
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, lev in zip(axs, (10.0, 1.0)):
        bh = (c[a:] / c[a - 1] - 1) * lev + 1
        ax.plot(
            x,
            np.maximum(bh, 0) * 100 - 100,
            color=MUTED,
            lw=1.2,
            ls="--",
            label=f"BTC buy & hold {lev:g}x {bh[-1] * 100 - 100:+.1f}%",
        )
        for col, k in zip(SERIES, keys):
            if k in ENS:
                mem = [T[m] for m in (TRACKER if k == "E4" else rep[k]["members"])]
                e, _ = split_equity(d, mem, a, lev)
            else:
                e, _ = equity(d, T[k], a, lev)
            lab = k + (f" ({'+'.join(rep[k]['members'])})" if k == "E5" else "")
            ax.plot(x, e * 100 - 100, color=col, lw=2, label=f"{lab} {e[-1] * 100 - 100:+.1f}%")
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_ylabel(f"{lev:g}x return %", color=INK)
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        style(ax)
    axs[0].set_title(
        "Equity, 15m mark-to-market, fees + funding (one position at a time, compounded)",
        loc="left",
        color=INK,
        fontsize=10,
    )
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(res / "last30_equity.png", dpi=110)
    plt.close(fig)


def candles(ax, h1):
    x = kx(h1["t"])
    w = (1 / 24) * 0.7
    up = h1["c"] >= h1["o"]
    for m, col in ((up, "#9a9a9a"), (~up, "#4a4a4a")):
        ax.vlines(x[m] + w / 2, h1["l"][m], h1["h"][m], color=col, lw=0.6)
        ax.bar(
            x[m],
            np.abs(h1["c"][m] - h1["o"][m]) + 1e-6,
            bottom=np.minimum(h1["o"], h1["c"])[m],
            width=w,
            align="edge",
            color=col,
            lw=0,
        )


def trades_chart(rep, T, d, d5, a, res, keys, fname, title, shade=()):
    t15 = d["t"]
    a_t = t15[a]
    h1 = resample(d5, 12)
    m = h1["t"] >= a_t - 4 * H1
    h1w = {k: v[m] for k, v in h1.items()}
    full = {k: v for k, v in h1.items()}
    n = len(keys)
    fig, ax = plt.subplots(
        n + 3,
        1,
        figsize=(16, 4.6 * n + 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [4.5] * n + [1.1, 1.1, 1.1]},
    )
    reg, _ = R.regime(d5, d, 0)
    for p, k in zip(ax[:n], keys):
        candles(p, h1w)
        if k in shade:
            xs = kx(t15 + BAR)
            for i in range(a, len(t15) - 1):
                if reg[i]:
                    p.axvspan(xs[i], xs[i + 1], color=BUY if reg[i] > 0 else SELL, alpha=0.05, lw=0)
        for z in T[k]:
            s = z["s"]
            mk = "o" if s > 0 else "^"
            x0 = kx(t15[z["i"]] + BAR)[0]
            p.scatter(
                x0,
                z["px"],
                marker=mk,
                s=110,
                zorder=6,
                edgecolor="white",
                lw=1.2,
                color=BUY if s > 0 else SELL,
            )
            if z["open"]:
                p.annotate(
                    f"open {z['net'] * 1000:+.1f}% @10x",
                    (x0, z["px"]),
                    xytext=(6, 10),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK,
                )
                continue
            x1 = kx(t15[z["x"]] + BAR)[0]
            p.scatter(
                x1,
                z["xp"],
                marker=mk,
                s=110,
                zorder=6,
                edgecolor="white",
                lw=1.2,
                color=SELL if s > 0 else BUY,
            )
            good = z["net"] > 0
            p.plot([x0, x1], [z["px"], z["xp"]], color="#555555", lw=1.0, ls="-" if good else ":")
            p.annotate(
                f"{z['net'] * 1000:+.1f}%",
                (x1, z["xp"]),
                xytext=(4, 8) if good else (4, -14),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=BUY if good else SELL,
            )
        r = rep[k]
        sd = r["side"]
        lt = ", ".join(
            f"{nm} {v['n']} ({v['win']}% win, 10x {v['x10']:+.1f}%)" for nm, v in sd.items()
        )
        p.set_title(
            f"{k}: {r['n']} trades (L {r['long']} / S {r['short']}, open {r['open']}), win {r['win']}%, "
            f"10x {r['ret10']:+.1f}% (MDD {r['mdd10']:.0f}%), 1x {r['ret1']:+.2f}% (MDD {r['mdd1']:.1f}%)"
            f"\n{lt}. Labels = net per trade at 10x",
            loc="left",
            color=INK,
            fontsize=10,
        )
        p.set_ylabel("BTCUSDT perp", color=INK)
        style(p)
    ax[0].legend(
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
    hh, ll, cc = full["h"], full["l"], full["c"]
    ind = (
        ("Stoch 5 (1h)", stoch(hh, ll, cc, 5), (20, 80)),
        ("CCI 20 (1h)", cci(hh, ll, cc, 20), (-100, 100)),
        ("RSI 14 (1h)", rsi(cc, 14), (30, 70)),
    )
    xh = kx(full["t"] + H1)
    for p, (nm, v, lines) in zip(ax[n:], ind):
        p.plot(xh[m], v[m], color="#2a78d6", lw=1)
        for ln in lines:
            p.axhline(ln, color="#999999", lw=0.8, ls="--")
        p.set_ylabel(nm, color=INK, fontsize=9)
        style(p)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(res / fname, dpi=95)
    plt.close(fig)


def main() -> None:
    res = Path(sys.argv[1])
    rep, T, d5 = load(res)
    d = bars(d5, 0)
    a = rep["info"]["a"]
    assert len(d["t"]) - a == rep["info"]["bars15"]
    ranking(rep, res)
    singles = sorted(SINGLES, key=lambda k: -rep[k]["ret10"])[:2]
    ens = max(ENS, key=lambda k: rep[k]["ret10"])
    equity_chart(rep, T, d, a, res, singles + [ens])
    for k in singles:
        trades_chart(
            rep,
            T,
            d,
            d5,
            a,
            res,
            [k],
            f"last30_trades_{k}.png",
            f"{k}: last 30 days, BTCUSDT perp, market fill at signal close",
            shade=("K2B", "K2", "R1"),
        )
    if ens not in singles:
        mem = rep[ens].get("members") or []
        rest = [k for k in mem if k not in singles]
        if rest:
            trades_chart(
                rep,
                T,
                d,
                d5,
                a,
                res,
                rest,
                f"last30_trades_{ens}_rest.png",
                f"{ens} = equal split of {' + '.join(mem)}; panel shows the member(s) "
                "not charted separately",
            )
    print("ok", singles, ens)


if __name__ == "__main__":
    main()
