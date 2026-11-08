"""30일 맞춤 점수 차트: 맞춤 구간 매매(가격 + Stoch·CCI·RSI + 점수)와 확장 구간 5배 자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 5배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 fit30_chart.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import fit30 as F
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

G, RD = "#1a9850", "#d73027"


def dt(t, i):
    return datetime.fromtimestamp(int(t[i]) / 1000, F.KST).replace(tzinfo=None)


def trade_chart(b, R, s, th, tr, a, z, title, path):
    t = b["t"]
    X = mdates.date2num([dt(t, i) for i in range(a, z)])
    fig, ax = plt.subplots(5, 1, figsize=(16, 14), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1.2]})
    ax[0].plot(X, b["c"][a:z], color="#555", lw=0.8)
    for x in tr:
        long = x["d"] == 1
        e, xi = x["e"], min(x["x"], z - 1)
        ax[0].scatter(mdates.date2num(dt(t, e)), x["ep"], marker="o" if long else "v", s=60,
                      color=G if long else RD, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].scatter(mdates.date2num(dt(t, xi)), x["xp"], marker="o" if long else "^", s=60,
                      color=RD if long else G, zorder=5, edgecolor="white")  # fmt: skip
        ax[0].annotate(f"{5 * x['r'] * 100:+.1f}", (mdates.date2num(dt(t, xi)), x["xp"]), fontsize=6,
                       color=G if x["r"] > 0 else RD, xytext=(2, 5), textcoords="offset points")  # fmt: skip
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=G, label="long open"),
                          Line2D([], [], marker="o", ls="", color=RD, label="long close"),
                          Line2D([], [], marker="v", ls="", color=RD, label="short open"),
                          Line2D([], [], marker="^", ls="", color=G, label="short close")],
                 loc="upper right", fontsize=8)  # fmt: skip
    ax[0].set_title(title, loc="left", fontsize=10)
    for k, (y, nm, lo, hi) in enumerate(((R["k"], "Stoch 5", 20, 80), (R["cci"], "CCI 20", -100, 100),
                                          (R["rsi"], "RSI 14", 30, 70), (s, "score", -th, th)), 1):  # fmt: skip
        ax[k].plot(X, y[a:z], lw=0.6, color="#2c7bb6" if k < 4 else "#7b3294")
        ax[k].axhline(lo, ls="--", lw=0.6, color="#888")
        ax[k].axhline(hi, ls="--", lw=0.6, color="#888")
        ax[k].set_ylabel(nm)
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for q in ax:
        q.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def equity_chart(b, R, s, th, P, W, ext, wf, path):
    t = b["t"]
    names = [("P30", "prev 30d"), ("M6", "6 months excl. fit"), ("Y1", "1 year excl. fit"),
             ("W22", "2022-01 ~ 2025-10"), ("W20", "2020-03 ~ 2021-10 (untouched)")]  # fmt: skip
    fig, axs = plt.subplots(3, 2, figsize=(15, 11))
    for q, (k, lab) in zip(axs.flat, names):
        a, z = W[k]
        tr = F.sim(b, R, s, th, P, a, z)
        eq, xs, ys = 1.0, [dt(t, a)], [0.0]
        for x in tr:
            eq = max(eq * (1 + 5 * x["r"]), 0.0)
            xs.append(dt(t, min(x["x"], z - 1)))
            ys.append((eq - 1) * 100)
        q.step(xs, ys, where="post", color="#2c7bb6", lw=1.2)
        q.axhline(0, color="#999", lw=0.6)
        r = ext[k]
        rp = r.get("random", {})
        q.set_title(f"{lab}: {r['n']} trades ({r['per_day']:.1f}/day), win {r['win'] or 0:.0%}, "
                    f"5x {r['x5']['ret']:+.0%}, MDD {r['x5']['mdd']:.0%}, random p {rp.get('p', float('nan')):.2f}",
                    fontsize=9, loc="left")  # fmt: skip
        q.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        q.grid(alpha=0.2)
    q = axs.flat[5]
    if wf:
        for k, col in (("WF1y", "#d95f02"), ("WF6m", "#1b9e77")):
            wk = np.array(wf[k]["weekly_x5"])
            q.plot(np.arange(len(wk)), (np.cumprod(1 + wk) - 1) * 100, color=col,
                   label=f"{k}: 5x {wf[k]['x5']['ret']:+.0%}, {wf[k]['per_day']:.1f}/day")  # fmt: skip
        q.axhline(0, color="#999", lw=0.6)
        q.legend(fontsize=8)
        q.set_title("walk-forward: refit every week on prior 30d, trade next 7d (x = week)", fontsize=9,
                    loc="left")  # fmt: skip
        q.grid(alpha=0.2)
    fig.suptitle("Fitted on last 30 days, frozen, tested on other periods - BTCUSDT perp 15m, 5x, "
                 "fees + funding + chase limit (y = 5x return %)", fontsize=11)  # fmt: skip
    plt.tight_layout()
    plt.savefig(path, dpi=95)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).expanduser()
    b = F.load()
    R = F.raw_features(b)
    W = F.windows(b["t"])
    rule = json.loads((out / "rule.json").read_text())
    ext = json.loads((out / "ext.json").read_text())
    wf = json.loads((out / "wf.json").read_text()) if (out / "wf.json").exists() else None
    Z = F.apply_std(R, rule["std"])
    w = np.array([rule["weights"][n] for n in F.FEATS])
    s, th, P = w @ Z, rule["theta"], rule["P"]
    a, z = W["F"]
    tr = F.sim(b, R, s, th, P, a, z)
    f = ext["F"]
    trade_chart(b, R, s, th, tr, a, z,
                f"FITTED (in-sample) last 30 days {rule['fit_window'][0]} ~ {rule['fit_window'][1]} KST - "
                f"{f['n']} trades ({f['per_day']:.1f}/day, L {f['long']} / S {f['short']}), win {f['win']:.0%}, "
                f"5x {f['x5']['ret']:+.0%} (MDD {f['x5']['mdd']:.0%})\nentry {P['entry']} exit {P['exit']} "
                f"T={P['T']} bars; labels = 5x net % per trade", out / "chart_fit30.png")  # fmt: skip
    a, z = W["P30"]
    tr = F.sim(b, R, s, th, P, a, z)
    p = ext["P30"]
    trade_chart(b, R, s, th, tr, a, z,
                f"OUT OF SAMPLE previous 30 days {p['span'][0]} ~ {p['span'][1]} KST (same frozen rule) - "
                f"{p['n']} trades, win {p['win'] or 0:.0%}, 5x {p['x5']['ret']:+.0%} (MDD {p['x5']['mdd']:.0%})",
                out / "chart_prev30.png")  # fmt: skip
    equity_chart(b, R, s, th, P, W, ext, wf, out / "equity_ext.png")
    print("charts ok")


if __name__ == "__main__":
    main()
