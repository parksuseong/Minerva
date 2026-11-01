"""패턴 신호의 지정가 체결 5분봉 재현. 등록: docs/backtests/2026-11-01-btc-pattern-hypotheses.md 끝

실행: PYTHONPATH=src:<numpy> python docs/backtests/raw/btc_limit.py <패턴 pkl>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import csv
import pickle
import sys
from datetime import datetime, timezone

import numpy as np

from minerva.strategy.btc_data import OUT

TP, SL = 0.02, 0.01
MAKER, TAKER, SLIP = 0.0002, 0.0005, 0.001
MAX_HOURS = 500
YEAR_DAYS = 365.25


def load(frame: str):
    t, o, h, l, c = [], [], [], [], []  # noqa: E741
    with (OUT / f"{frame}.csv").open() as f:
        for r in csv.reader(f):
            if r[0] == "open_time":
                continue
            t.append(int(r[0]))
            o.append(float(r[1]))
            h.append(float(r[2]))
            l.append(float(r[3]))
            c.append(float(r[4]))
    return tuple(np.array(x) for x in (t, o, h, l, c))


def first(mask):
    return int(mask.argmax()) if mask.any() else -1


def sim(sig, t1, c1, m5, wait, pen):
    t5, o5, h5, l5 = m5
    n = len(t5)
    free = -1
    trades, missed = [], []
    for i, s, _ in sig:
        close_ms = t1[i] + 3_600_000
        a = int(np.searchsorted(t5, close_ms))
        if a <= free or a >= n - 1:
            continue
        px = c1[i]
        z = min(a + wait, n)
        seg = l5[a:z] <= px * (1 - pen) if s == 1 else h5[a:z] >= px * (1 + pen)
        j = first(seg)
        # 체결 여부와 상관없이 결과를 재서 놓친 신호의 승패를 본다
        e = a + j if j >= 0 else a
        end = min(e + MAX_HOURS * 12, n)
        goal = px * (1 + s * TP)
        stop = px * (1 - s * SL)
        hs = l5[e:end] <= stop if s == 1 else h5[e:end] >= stop
        hg = h5[e:end] >= goal * (1 + pen) if s == 1 else l5[e:end] <= goal * (1 - pen)
        fs, fg = first(hs), first(hg)
        fs = fs if fs >= 0 else 10**9
        fg = fg if fg >= 0 else 10**9
        if fs == fg == 10**9:
            continue
        win = fg < fs
        if j < 0:
            missed.append(win)
            continue
        if win:
            r = s * TP - MAKER - MAKER * (1 + s * TP)
            last = e + fg
        else:
            b = e + fs
            bad = min(o5[b], stop) if s == 1 else max(o5[b], stop)
            bad *= 1 - s * SLIP
            r = s * (bad / px - 1) - MAKER - TAKER * (bad / px)
            last = b
        trades.append((i, s, last, float(r), bool(win)))
        free = last
    return trades, missed


def show(tag, trades, missed, sig_n, years, days, s25):
    r = np.array([x[3] for x in trades])
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    mdd = float((eq / peak - 1).min())
    cagr = eq[-1] ** (YEAR_DAYS / days) - 1
    rest = float(np.prod(np.sort(1 + r)[:-5]))
    rec = float(np.prod([1 + x[3] for x in trades if x[0] >= s25]))
    by: dict = {}
    for x in trades:
        y = int(years[x[0]])
        by[y] = by.get(y, 1.0) * (1 + x[3])
    print(
        f"{tag}: 체결 {len(r)}/{len(r) + len(missed)} ({len(r) / (len(r) + len(missed)) * 100:.0f}%) "
        f"승률 {(r > 0).mean() * 100:.1f} 놓친 승률 {np.mean(missed) * 100 if missed else 0:.1f} "
        f"연 {cagr * 100:+.1f}% 낙폭 {mdd * 100:.1f}% 비 {cagr / abs(mdd):.2f} "
        f"평균 {r.mean() * 100:.3f}% 상위5제외 {rest:.2f} 2025~ {rec:.3f} "
        f"연도 { {y: round(v - 1, 3) for y, v in sorted(by.items())} }",
        flush=True,
    )


def main(path: str) -> None:
    d = pickle.load(open(path, "rb"))
    t1, _, _, _, c1 = load("1h")
    t5, o5, h5, l5, _ = load("5m")
    print("5m bars", len(t5), flush=True)
    years = np.array([datetime.fromtimestamp(t / 1000, timezone.utc).year for t in t1])
    s20 = int(np.argmax(years == 2020))
    s25 = int(np.argmax(years == 2025))
    days = (t1[-1] - t1[s20]) / 86_400_000
    for wait, label in ((3, "15분"), (12, "60분")):
        for pen in (0.0, 0.0002, 0.0005):
            tr, ms = sim(d["sig"], t1, c1, (t5, o5, h5, l5), wait, pen)
            show(f"대기 {label} 관통 {pen * 100:.2f}%", tr, ms, len(d["sig"]), years, days, s25)


if __name__ == "__main__":
    main(sys.argv[1])
