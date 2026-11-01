"""BTCUSDT 15분봉 문헌 가설 (사전 등록: docs/backtests/2026-11-01-btc-literature-hypotheses.md).

numpy가 필요하다. 실행: PYTHONPATH=src:<numpy 경로> python docs/backtests/raw/btc_lit.py
주문과 연결되지 않는다.
"""

from __future__ import annotations

import csv
import random
from collections import deque
from datetime import datetime, timezone

import numpy as np

from minerva.strategy.btc_data import OUT

STOP = 0.01
COST = 0.0005
YEAR_DAYS = 365.25
SHIFTS = 1000
LOOK = 1920


def load() -> dict:
    with (OUT / "15m.csv").open() as f:
        rows = list(csv.DictReader(f))
    get = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    return {
        "t": [datetime.fromtimestamp(int(r["open_time"]) / 1000, timezone.utc) for r in rows],
        "o": get("open"),
        "h": get("high"),
        "l": get("low"),
        "c": get("close"),
        "v": get("volume"),
        "b": get("taker_buy_base"),
    }


def run(d, entries, stop=STOP, cost=COST, entry_slip=0.0, stop_slip=0.0, lo=0, hi=None):
    """entries: (신호 봉, 쪽, 보유 봉) 시간순. (청산 봉, 수익) 목록을 낸다."""
    o, h, l = d["o"], d["h"], d["l"]  # noqa: E741
    n = len(o) if hi is None else hi
    out, free = [], -1
    for i, side, hold in entries:
        e = i + 1
        if e <= free or e < lo or e + hold >= n:
            continue
        x = e + hold
        px0 = o[e]
        fill = px0 * (1 + side * entry_slip)
        px, last = o[x], x - 1
        if stop is not None:
            line = px0 * (1 - stop) if side == 1 else px0 * (1 + stop)
            hit = l[e:x] <= line if side == 1 else h[e:x] >= line
            if hit.any():
                last = e + int(hit.argmax())
                bad = min(o[last], line) if side == 1 else max(o[last], line)
                px = bad * (1 - side * stop_slip)
        r = (side * (px / fill - 1) + 1) * (1 - cost) - (px / fill) * cost - 1
        out.append((last, float(r)))
        free = last
    return out


def stats(trades, years, days):
    if not trades:
        return {"n": 0, "cagr": 0.0, "mdd": 0.0, "final": 1.0}
    eq, peak, mdd, by = 1.0, 1.0, 0.0, {}
    for j, r in trades:
        eq *= 1 + r
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
        by[int(years[j])] = by.get(int(years[j]), 1.0) * (1 + r)
    rets = sorted((r for _, r in trades), reverse=True)
    cagr = eq ** (YEAR_DAYS / days) - 1
    return {
        "n": len(trades),
        "win": sum(r > 0 for r in rets) / len(rets),
        "avg": float(np.mean(rets)),
        "final": eq,
        "cagr": cagr,
        "mdd": mdd,
        "ratio": cagr / abs(mdd) if mdd else 0.0,
        "no_top5": float(np.prod([1 + r for r in rets[5:]])),
        "years": {y: round(v - 1, 3) for y, v in sorted(by.items())},
    }


def breakouts(c, look=LOOK):
    """직전 look봉 최고·최저 종가를 처음 넘은 봉."""
    hi, lo = deque(), deque()
    up, dn = np.zeros(len(c), bool), np.zeros(len(c), bool)
    for i, v in enumerate(c):
        if hi and hi[0] <= i - look - 1:
            hi.popleft()
        if lo and lo[0] <= i - look - 1:
            lo.popleft()
        if i >= look:
            up[i], dn[i] = v > c[hi[0]], v < c[lo[0]]
        while hi and c[hi[-1]] <= v:
            hi.pop()
        hi.append(i)
        while lo and c[lo[-1]] >= v:
            lo.pop()
        lo.append(i)
    edge = lambda x: x & ~np.r_[False, x[:-1]]  # noqa: E731
    return edge(up), edge(dn)


def signals(d) -> dict:
    t, c, o = d["t"], d["c"], d["o"]
    n = len(c)
    hour = np.array([x.hour for x in t])
    minute = np.array([x.minute for x in t])
    wday = np.array([x.weekday() for x in t])
    up, dn = breakouts(c)
    ups = [(int(i), 1, 96) for i in np.nonzero(up)[0]]
    downs = [(int(i), -1, 96) for i in np.nonzero(dn)[0]]
    sig = {"L1": ups, "L2": sorted(ups + downs)}
    sig["L3"] = [(int(i), 1, 8) for i in np.nonzero((hour == 20) & (minute == 45))[0]]
    sunday = np.nonzero((wday == 6) & (hour == 22) & (minute == 45))[0]
    sig["L4"] = [(int(i), int(np.sign(c[i] / c[i - 96] - 1)) or 1, 96) for i in sunday if i >= 96]
    last = []
    for i in np.nonzero((hour == 0) & (minute == 15))[0]:
        j = int(i) + 92  # 23:15 봉
        if j < n and t[j].hour == 23 and t[j].minute == 15:
            last.append((j, 1 if c[i] >= o[i - 1] else -1, 2))
    sig["L5"] = last
    share = np.where(d["v"] > 0, d["b"] / np.where(d["v"] > 0, d["v"], 1), 0.5) - 0.5
    cs, cq = np.r_[0, np.cumsum(share)], np.r_[0, np.cumsum(share**2)]
    win = 2880
    mean = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    mean[win:] = (cs[win:n] - cs[: n - win]) / win
    var = (cq[win:n] - cq[: n - win]) / win - mean[win:] ** 2
    sd[win:] = np.sqrt(np.maximum(var, 1e-12))
    z = (share - mean) / sd
    sig["L6"] = [
        (int(i), 1 if z[i] > 0 else -1, 32)
        for i in np.nonzero((minute == 0) & (np.abs(z) >= 2))[0]
    ]
    return sig


def shift(entries, lo, n, off):
    m = n - lo
    moved = [((i - lo + off) % m + lo, s, h) for i, s, h in entries if i >= lo]
    return sorted(moved)


def main() -> None:
    d = load()
    n = len(d["c"])
    years = np.array([x.year for x in d["t"]])
    s20 = int(np.argmax(years == 2020))
    s25 = int(np.argmax(years == 2025))
    days = (d["t"][-1] - d["t"][s20]).days
    days25 = (d["t"][-1] - d["t"][s25]).days
    rng = random.Random(7)
    offs = [rng.randrange(2000, n - s20 - 2000) for _ in range(SHIFTS)]
    for name, ent in signals(d).items():
        ok = [e for e in ent if e[0] >= s20]
        base = run(d, ok, lo=s20)
        a = stats(base, years, days)
        c10 = stats(run(d, ok, cost=0.001, lo=s20), years, days)
        slip = stats(run(d, ok, entry_slip=0.0002, stop_slip=0.001, lo=s20), years, days)
        rec = stats(run(d, ok, lo=s25), years, days25)
        free = stats(run(d, ok, stop=None, lo=s20), years, days)
        p = sum(
            stats(run(d, shift(ok, s20, n, f), lo=s20), years, days)["final"] >= a["final"]
            for f in offs
        ) / SHIFTS
        print(name, a, flush=True)
        print(
            f"  cost10 {c10['cagr']:.3f} slip {slip['cagr']:.3f} recent {rec.get('cagr'):.3f} "
            f"n25 {rec['n']} p {p:.4f} no_stop {free['cagr']:.3f}/{free['mdd']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
