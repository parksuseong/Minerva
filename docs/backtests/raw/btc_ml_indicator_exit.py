"""119번: ML 앙상블 AS·BS 진입 + 지표 청산. 등록: 2026-11-07-btc-ml-indicator-exit-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_indicator_exit.py \
    --out <폴더> [--research ~/.minerva/research/btc-2026-11-04]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
from btc_fractal import cci, rsi, stoch
from btc_futures_compare import load_funding
from btc_ml_edge_cost import MAKER, OFF, REST, SLIP, TAKER, THRU, rest_sim, summ_net
from btc_ml_seeds import rule_masks
from btc_scr_learned import direction

STOP = 0.01
MAXH = 192
HOUR = 3_600_000
RULES = {"AS": (0.90, 4), "BS": (0.95, 5)}


def hour_dir(P: dict, t: np.ndarray) -> np.ndarray:
    """15분봉 b 마감 때 마감돼 있는 가장 최근 1시간봉 방향."""
    key = (t // HOUR).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    st, en = edges[:-1], edges[1:]
    full = (en - st) == 4
    st, en = st[full], en[full]
    u = {"h": np.maximum.reduceat(P["h"], edges[:-1])[full],
         "l": np.minimum.reduceat(P["l"], edges[:-1])[full], "c": P["c"][en - 1]}
    dr = direction(u)
    close = key[st].astype(float) * HOUR + HOUR
    j = np.searchsorted(close, t + 900_000, side="right") - 1
    out = np.zeros(len(t), int)
    out[j >= 0] = dr[j[j >= 0]]
    return out


def exit_masks(P: dict, t: np.ndarray) -> dict:
    h, l, c = P["h"], P["l"], P["c"]
    s, cc, r = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    hd = hour_dir(P, t)
    with np.errstate(invalid="ignore"):
        ob = (s > 80).astype(int) + (cc >= 100) + (r >= 70)
        os_ = (s < 20).astype(int) + (cc <= -100) + (r <= 30)
        return {"E1": (r >= 70, r <= 30), "E2": (ob >= 2, os_ >= 2), "E3": (hd == -1, hd == 1)}


def ind_sim(P, t, li, si, i0, i1, xl, xs, k=1.0):
    """rest_sim과 같은 진입. 청산은 지표(다음 시가), 1% 손절, 192봉. (진입시각, 1배 순손익, 방향, 청산시각, 승)."""
    o, h, l, c = P["o"], P["h"], P["l"], P["c"]  # noqa: E741
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
        stop = px * (1 - STOP * s)
        xm = xl if s > 0 else xs
        res, b = None, j
        while b < n:
            first = b == j
            hit = (l[b] <= stop) if s > 0 else (h[b] >= stop)
            if hit:
                fill = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
                res, xb = fill, b
                break
            if b + 1 < n and (xm[b] or b + 1 - j >= MAXH):
                res, xb = o[b + 1], b + 1
                break
            b += 1
        if res is None:
            break
        g = (res * (1 - sp * s) / px - 1) * s - mk - tk
        out.append((float(t[j]), float(g), s, float(t[xb]), int(g > 0)))
        i = xb
    return out


def load_probs(research: Path) -> tuple:
    z = np.load(research / "ml15/pred.npz")
    seeds = [z["probs"]]
    for k in range(1, 5):
        w = np.load(research / f"seeds/pred_s{k}.npz")
        assert np.array_equal(w["t"], z["t"])
        seeds.append(w["probs"])
    return z, seeds


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
    X = exit_masks(P, t)
    pa = np.mean(seeds, 0)
    pers = {"2023": (i23, i24), "2024~": (i24, n), "1y": (i1y, n)}

    def run(probs, rule, ex, a0, a1, sides=(1, -1), k=1.0):
        li, si = rule_masks(probs, t, *RULES[rule])
        li = li & (1 in sides)
        si = si & (-1 in sides)
        yrs = (t[a1 - 1] - t[a0]) / F.YEAR
        if ex == "E0":
            tr = rest_sim(P, t, li, si, a0, a1, sides, k)
        else:
            tr = ind_sim(P, t, li, si, a0, a1, *X[ex], k)
        return summ_net(tr, ft, fr, yrs), tr

    rep: dict = {}
    trades: dict = {}
    for rule in RULES:
        for ex in ("E0", "E1", "E2", "E3"):
            r = {}
            for per, (a0, a1) in pers.items():
                r[per], tr = run(pa, rule, ex, a0, a1)
                if per == "2024~":
                    g = np.array([x[1] for x in tr]) if tr else np.zeros(1)
                    r["x1_2024~"] = round(float((np.cumprod(1 + g)[-1] - 1) * 100), 1)
                    r["hold_bars"] = round(float(np.mean([(x[3] - x[0]) / 900_000 for x in tr])), 1) if tr else 0
                    r["stopped_pct"] = round(float(np.mean([x[1] < -0.009 for x in tr]) * 100), 1) if tr else 0
                    trades[f"{rule}-{ex}"] = tr
            r["long_2024~"], _ = run(pa, rule, ex, i24, n, (1,))
            r["short_2024~"], _ = run(pa, rule, ex, i24, n, (-1,))
            r["cost1.5_2024~"], _ = run(pa, rule, ex, i24, n, k=1.5)
            r["seeds_2024~"] = [run(p, rule, ex, i24, n)[0]["x3"] for p in seeds]
            rep[f"{rule}-{ex}"] = r
            print(rule, ex, r["2023"]["x3"], r["2024~"]["x3"], r["2024~"]["n"], flush=True)
    verdict = {}
    for rule in RULES:
        e0 = rep[f"{rule}-E0"]
        for ex in ("E1", "E2", "E3"):
            r = rep[f"{rule}-{ex}"]
            c1 = all(r[p]["x3"] > 0 and r[p]["x3"] > e0[p]["x3"] for p in ("2023", "2024~"))
            c2 = r["long_2024~"].get("ev3", -1) > 0 and r["short_2024~"].get("ev3", -1) > 0
            c3 = r["cost1.5_2024~"]["x3"] > 0
            beat = sum(a > b for a, b in zip(r["seeds_2024~"], e0["seeds_2024~"]))
            c4 = beat >= 3
            verdict[f"{rule}-{ex}"] = {"1": bool(c1), "2": bool(c2), "3": bool(c3), "4": f"{beat}/5",
                                       "pass": bool(c1 and c2 and c3 and c4)}
    rep["verdict"] = verdict
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (out / "trades.json").write_text(json.dumps(trades))
    print(json.dumps(verdict, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
