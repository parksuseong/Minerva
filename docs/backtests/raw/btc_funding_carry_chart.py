"""142번 펀딩 캐리 차트: (a) 2020~ 자산 곡선 + 펀딩 + 헤지 구간, (b) 최근 30일.

실행: PYTHONPATH=<numpy·matplotlib>:src:docs/backtests/raw python <이 파일> <t142 폴더>
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

from btc_fractal import load5

KST = timezone(timedelta(hours=9))
INK, GRID = "#333333", "#e6e6e6"
COL = {"C0": "#4575b4", "C1": "#d73027", "C2": "#1a9850"}


def dn(ms):
    return mdates.date2num([datetime.fromtimestamp(t / 1000, KST) for t in ms])


def spans(log, t_end):
    out, start = [], None
    for e in log:
        if e["ev"] == "in":
            start = e["t"]
        elif e["ev"] in ("out", "liq") and start is not None:
            out.append((start, e["t"]))
            start = None
    if start is not None:
        out.append((start, t_end))
    return out


def main() -> None:
    d = Path(sys.argv[1])
    s = np.load(d / "series.npz")
    logs = json.loads((d / "logs.json").read_text())
    ft, fr = s["ft"], s["fr"]
    t_end = float(s["C0_days"][-1])
    fig, ax = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for v in ("C0", "C1", "C2"):
        ax[0].plot(dn(s[f"{v}_days"]), (s[f"{v}_eq"] - 1) * 100, color=COL[v], lw=1.2,
                   label=f"{v} ({(s[f'{v}_eq'][-1] - 1) * 100:+.1f}%)")
    for a, b in spans(logs["C2"], t_end):
        ax[0].axvspan(*dn([a, b]), color=COL["C2"], alpha=0.08, lw=0)
    ax[0].set_ylabel("return on capital % (1x, spot fee 0.1%)", color=INK)
    ax[0].legend(loc="upper left", frameon=False)
    ax[0].set_title("BTC funding carry: long spot + short BTCUSDT perp, 2020~ (shade = C2 hedged)", loc="left",
                    color=INK)
    ax[1].bar(dn(ft), fr * 100, width=0.33, color=np.where(fr >= 0, "#1a9850", "#d73027"))
    ax[1].axhline(0.01, color="#999999", lw=0.8, ls="--")
    ax[1].axhline(0.02, color="#999999", lw=0.8, ls=":")
    ax[1].set_ylabel("funding % / 8h", color=INK)
    ax[1].set_ylim(-0.05, 0.12)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(d / "t142_equity.png", dpi=110)
    plt.close(fig)

    fut = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    lo = t_end - 30 * 86_400_000
    m = fut["t"] >= lo
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(dn(fut["t"][m]), fut["c"][m], color=INK, lw=0.7)
    for v, mk in (("C1", "o"), ("C2", "^")):
        for e in logs[v]:
            if e["t"] < lo:
                continue
            i = int(np.searchsorted(fut["t"], e["t"]))
            ax[0].scatter(dn([e["t"]]), [fut["c"][min(i, len(fut["c"]) - 1)]], marker=mk, s=90, zorder=5,
                          color="#1a9850" if e["ev"] == "in" else "#d73027", edgecolor="white",
                          label=f"{v} hedge {'on' if e['ev'] == 'in' else 'off'}")
    h, lab = ax[0].get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax[0].legend(uniq.values(), uniq.keys(), loc="upper right", frameon=False, fontsize=9)
    ax[0].set_title("Last 30 days: BTCUSDT perp price, hedge on (green) / off (red); C1 circle, C2 triangle (KST)",
                    loc="left", color=INK)
    mf = ft >= lo
    ax[1].bar(dn(ft[mf]), fr[mf] * 100, width=0.25, color=np.where(fr[mf] >= 0, "#1a9850", "#d73027"))
    ax[1].axhline(0.01, color="#999999", lw=0.8, ls="--")
    ax[1].set_ylabel("funding % / 8h", color=INK)
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
        for sd in ("top", "right"):
            a_.spines[sd].set_visible(False)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(d / "t142_recent.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
