"""패턴 신호 + 합의 투표. 등록: docs/backtests/2026-11-01-btc-pattern-hypotheses.md 끝

실행: PYTHONPATH=src:<numpy>:docs/backtests/raw python <이 파일> <패턴 pkl> <조합 pkl>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import pickle
import random
import sys

import btc_mit as mit
import btc_regress as br
import btc_swing_label as sl
import numpy as np

from minerva.strategy import btc_bt as bt
from minerva.strategy.indicators import tm_states

TP = 0.02
SHIFTS = 300


def main(pat: str, com: str) -> None:
    sig = pickle.load(open(pat, "rb"))["sig"]
    combo = pickle.load(open(com, "rb"))["sig"]
    bars = bt.load("1h")
    o, h, l, c = [np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low", "close")]  # noqa: E741
    n = len(o)
    years = np.array([b.day.year for b in bars])
    s20 = int(np.argmax(years == 2020))
    s25 = int(np.argmax(years == 2025))
    days = (bars[-1].day - bars[s20].day).days
    d25 = (bars[-1].day - bars[s25].day).days
    br.MAX_BARS, br.STOP = 500, 0.01
    keyed = {(TP, s): br.outcomes(o, h, l, TP, s) for s in (1, -1)}
    rec = {1: np.zeros(n, bool), -1: np.zeros(n, bool)}
    for i, s, _ in combo:
        rec[s][i : i + 3] = True
    lead_l, lead_s = mit.range_filter(c)
    up, down = mit.rqk(c)
    rf = {1: lead_l & up, -1: lead_s & down}
    st = tm_states(bars)
    scr = {
        1: np.array([x.long_level >= 1 for x in st]),
        -1: np.array([x.short_level >= 1 for x in st]),
    }
    hi, lo = np.zeros(n), np.zeros(n)
    k, a, b = 0, 0.0, 0.0
    ev = sl.confirmed(bars)
    for i in range(n):
        while k < len(ev) and ev[k][0] <= i:
            _, s, text, _ = ev[k]
            if s == 1:
                a = 1.0 if text == "HH" else -1.0
            else:
                b = 1.0 if text == "HL" else -1.0
            k += 1
        hi[i], lo[i] = a, b
    struct = {1: (hi > 0) & (lo > 0), -1: (hi < 0) & (lo < 0)}
    votes = [
        (i, s, t, int(rec[s][i]) + int(rf[s][i]) + int(struct[s][i]) + int(scr[s][i]))
        for i, s, t in sig
    ]
    print("표 분포", {m: sum(v[3] == m for v in votes) for m in range(5)}, flush=True)
    for m in (0, 1, 2, 3, 4):
        chosen = [(i, s, t) for i, s, t, v in votes if v >= m]
        tr = br.trade(chosen, keyed, br.COST, o)
        if not tr:
            print(f"m{m}: 거래 없음", flush=True)
            continue
        got = br.score(tr, years, days)
        c10 = br.score(br.trade(chosen, keyed, 0.001, o), years, days)
        slip = br.score(
            [(e, r - 0.0004 - (0.001 if r < 0 else 0), w) for e, r, w in tr], years, days
        )
        r25 = br.score(br.trade([x for x in chosen if x[0] >= s25], keyed, br.COST, o), years, d25)
        rng = random.Random(7)
        vals, span = [], n - s20
        for _ in range(SHIFTS):
            d = rng.randrange(1000, span - 1000)
            moved = sorted(((i - s20 + d) % span + s20, s, t) for i, s, t in chosen)
            vals.append(
                br.score(
                    br.trade([x for x in moved if x[0] < n - 2], keyed, br.COST, o), years, days
                )["final"]
            )
        p = (sum(v >= got["final"] for v in vals) + 1) / (SHIFTS + 1)
        print(
            f"m{m}: n {got['trades']} win {got['win'] * 100:.1f} cagr {got['cagr'] * 100:+.1f} "
            f"mdd {got['mdd'] * 100:.1f} ratio {got['ratio']:.2f} cost10 {c10['cagr'] * 100:+.1f} "
            f"slip {slip['cagr'] * 100:+.1f} y25 {r25.get('final')} n25 {r25.get('trades')} "
            f"win25 {r25.get('win')} top5 {got['no_top5']:.2f} p {p:.3f} years {got['years']}",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
