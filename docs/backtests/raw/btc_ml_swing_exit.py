"""120번: ML 앙상블 AS·BS 진입 + RSI 7 스윙 구조 청산. 등록: 2026-11-07-btc-ml-swing-exit-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_swing_exit.py \
    --out <폴더> [--research ~/.minerva/research/btc-2026-11-04]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
from btc_fractal import rsi
from btc_fut_rsi_swing import bd_signals
from btc_futures_compare import load_funding
from btc_ml_edge_cost import MAKER, OFF, REST, SLIP, TAKER, THRU, rest_sim, summ_net
from btc_ml_indicator_exit import RULES, load_probs
from btc_ml_seeds import rule_masks

STOP = 0.01
MAXH = 192
BUF = 0.001


def swing_state(n: int, sig: list) -> tuple:
    """봉 b 마감 때 확정돼 있는 마지막 스윙 저점·고점 가격, 그리고 봉 b에서 새로 확정된 스윙(+1 저점, −1 고점)."""
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    new = np.zeros(n, int)
    for i, side, px, _ in sig:
        new[i] = side  # bd_signals의 +1은 저점 확정, −1은 고점 확정
        (lo if side > 0 else hi)[i] = px
    # 앞으로 채우기
    for a in (lo, hi):
        idx = np.where(np.isfinite(a), np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        a[:] = a[idx]
    return lo, hi, new


def one_e0(P, t, s, j, px, mk, tk, sp):
    """rest_sim과 같은 ±TP/SL 청산 한 건. (1배 순손익, 청산 봉, 승)."""
    o, h, l = P["o"], P["h"], P["l"]  # noqa: E741
    up, dn = px * (1 + F.TP), px * (1 - F.SL)
    b = j
    while b < len(o):
        first = b == j
        if s > 0:
            if l[b] <= dn:
                return (min(o[b], dn) if not first else dn) * (1 - sp) / px - 1 - mk - tk, b, 0
            if not first and h[b] >= up * (1 + THRU):
                return F.TP - 2 * mk, b, 1
        else:
            if h[b] >= up:
                return -((max(o[b], up) if not first else up) * (1 + sp) / px - 1) - mk - tk, b, 0
            if not first and l[b] <= dn * (1 - THRU):
                return F.SL - 2 * mk, b, 1
        b += 1
    return None


def swing_sim(P, t, li, si, i0, i1, sw, mode, k=1.0, stats=None):
    """mode 'S1' 또는 'S2'. 거래: (진입시각, 1배 순손익, 방향, 청산시각, 승, 손절, 익절, 'S'/'E0'/'F1')."""
    o, h, l, c = P["o"], P["h"], P["l"], P["c"]  # noqa: E741
    lo, hi, new = sw
    mk, tk, sp = MAKER * k, TAKER * k, SLIP * k
    out, i = [], i0
    n = len(o)
    while i < i1 - 2:
        s = 1 if li[i] else (-1 if si[i] else 0)
        if not s:
            i += 1
            continue
        lim = c[i] * (1 - OFF * s)
        j = None
        for b in range(i + 1, min(i + 1 + REST, n)):
            if (l[b] <= lim * (1 - THRU)) if s > 0 else (h[b] >= lim * (1 + THRU)):
                j = b
                break
        if j is None:
            i += 1
            continue
        px = lim
        lv_stop = lo[i] if s > 0 else hi[i]
        lv_tgt = hi[i] if s > 0 else lo[i]
        stop = lv_stop * (1 - BUF * s) if np.isfinite(lv_stop) else np.nan
        bad_stop = not np.isfinite(stop) or (stop >= px if s > 0 else stop <= px)
        bad_tgt = not np.isfinite(lv_tgt) or (lv_tgt <= px if s > 0 else lv_tgt >= px)
        kind = "S"
        if mode == "S1" and (bad_stop or bad_tgt):
            res = one_e0(P, t, s, j, px, mk, tk, sp)
            if res is None:
                break
            g, xb, win = res
            out.append((float(t[j]), float(g), s, float(t[xb]), int(win), None, None, "E0"))
            if stats is not None:
                stats["E0"] = stats.get("E0", 0) + 1
            i = xb
            continue
        if bad_stop:
            stop, kind = px * (1 - STOP * s), "F1"
        elif abs(stop / px - 1) > STOP:
            stop = px * (1 - STOP * s)
        tgt = lv_tgt if mode == "S1" else None
        if stats is not None:
            stats[kind] = stats.get(kind, 0) + 1
        res, b = None, j
        while b < n:
            first = b == j
            if (l[b] <= stop) if s > 0 else (h[b] >= stop):
                fill = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
                g = (fill * (1 - sp * s) / px - 1) * s - mk - tk
                res = (g, b)
                break
            if tgt is not None and not first and ((h[b] >= tgt * (1 + THRU)) if s > 0 else (l[b] <= tgt * (1 - THRU))):
                res = ((tgt / px - 1) * s - 2 * mk, b)
                break
            if b + 1 < n and ((mode == "S2" and new[b] == -s) or b + 1 - j >= MAXH):
                g = (o[b + 1] * (1 - sp * s) / px - 1) * s - mk - tk
                res = (g, b + 1)
                break
            b += 1
        if res is None:
            break
        g, xb = res
        out.append((float(t[j]), float(g), s, float(t[xb]), int(g > 0), float(stop),
                    None if tgt is None else float(tgt), kind))
        i = xb
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--research", default=str(Path("~/.minerva/research/btc-2026-11-04").expanduser()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z, seeds = load_probs(Path(args.research))
    t = z["t"]
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    n = len(t)
    ft, fr = load_funding()
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    i1y = int(np.searchsorted(t, t[-1] - F.YEAR))
    sig = bd_signals(P["h"], P["l"], rsi(P["c"], 7))
    sw = swing_state(n, sig)
    pa = np.mean(seeds, 0)
    pers = {"2023": (i23, i24), "2024~": (i24, n), "1y": (i1y, n)}

    def run(probs, rule, ex, a0, a1, sides=(1, -1), k=1.0, stats=None):
        li, si = rule_masks(probs, t, *RULES[rule])
        li = li & (1 in sides)
        si = si & (-1 in sides)
        yrs = (t[a1 - 1] - t[a0]) / F.YEAR
        if ex == "E0":
            tr = rest_sim(P, t, li, si, a0, a1, sides, k)
        else:
            tr = swing_sim(P, t, li, si, a0, a1, sw, ex, k, stats)
        return summ_net(tr, ft, fr, yrs), tr

    rep: dict = {"swings": len(sig)}
    trades: dict = {}
    for rule in RULES:
        for ex in ("E0", "S1", "S2"):
            r = {}
            for per, (a0, a1) in pers.items():
                st: dict = {}
                r[per], tr = run(pa, rule, ex, a0, a1, stats=st)
                if per == "2024~":
                    g = np.array([x[1] for x in tr]) if tr else np.zeros(1)
                    r["x1_2024~"] = round(float((np.cumprod(1 + g)[-1] - 1) * 100), 1)
                    r["hold_bars"] = round(float(np.mean([(x[3] - x[0]) / 900_000 for x in tr])), 1) if tr else 0
                    r["kinds_2024~"] = st
                    trades[f"{rule}-{ex}"] = tr
            r["long_2024~"], _ = run(pa, rule, ex, i24, n, (1,))
            r["short_2024~"], _ = run(pa, rule, ex, i24, n, (-1,))
            r["cost1.5_2024~"], _ = run(pa, rule, ex, i24, n, k=1.5)
            r["seeds_2024~"] = [run(p, rule, ex, i24, n)[0]["x3"] for p in seeds]
            rep[f"{rule}-{ex}"] = r
            print(rule, ex, r["2023"]["x3"], r["2024~"]["x3"], r["2024~"]["n"], r.get("kinds_2024~"), flush=True)
    verdict = {}
    for rule in RULES:
        e0 = rep[f"{rule}-E0"]
        for ex in ("S1", "S2"):
            r = rep[f"{rule}-{ex}"]
            c1 = all(r[p]["x3"] > 0 and r[p]["x3"] > e0[p]["x3"] for p in ("2023", "2024~"))
            c2 = r["long_2024~"].get("ev3", -1) > 0 and r["short_2024~"].get("ev3", -1) > 0
            c3 = r["cost1.5_2024~"]["x3"] > 0
            beat = sum(a > b for a, b in zip(r["seeds_2024~"], e0["seeds_2024~"]))
            verdict[f"{rule}-{ex}"] = {"1": bool(c1), "2": bool(c2), "3": bool(c3), "4": f"{beat}/5",
                                       "pass": bool(c1 and c2 and c3 and beat >= 3)}
    rep["verdict"] = verdict
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (out / "trades.json").write_text(json.dumps(trades))
    print(json.dumps(verdict, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
