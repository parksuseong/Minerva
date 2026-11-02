"""SCR에 이평이 몇 개, 몇 봉이 필요한가. 등록: 2026-11-02-btc-scr-ma-count-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src python docs/backtests/raw/btc_scr_ma_count.py --out <폴더>
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from btc_scr_ma import COSTS, WIN, YEAR, fit_logit, load, raw_signals, run, sma, stats

GRID = (10, 20, 30, 50, 60, 80, 120, 160, 240, 320, 480, 720, 1440, 2160, 2880, 3120)
SLOPE = 10
KS = (1, 2, 3, 4)
SIGS = ("S3", "R")
EXITS = ("O", "M")
MIN_TRAIN = 20

Rule = Tuple  # ("P", n), ("D", n), ("O", n1, n2, ...)


def rules(k: int) -> List[Rule]:
    if k == 1:
        return [("P", n) for n in GRID] + [("D", n) for n in GRID]
    return [("O",) + c for c in itertools.combinations(GRID, k)]


def rule_mask(rule: Rule, c: np.ndarray, ma: Dict[int, np.ndarray]) -> tuple:
    """(롱 허용, 숏 허용, 롱 청산, 숏 청산). 청산 기준은 가장 긴 이평."""
    if rule[0] == "P":
        ref = ma[rule[1]]
        lo, sh = c > ref, c < ref
    elif rule[0] == "D":
        ref = ma[rule[1]]
        prev = np.r_[np.full(SLOPE, np.nan), ref[:-SLOPE]]
        lo, sh = ref > prev, ref < prev
    else:
        ns = rule[1:]
        ref = ma[ns[-1]]
        lo = np.ones(len(c), bool)
        sh = lo.copy()
        for f, s in zip(ns, ns[1:]):
            lo &= ma[f] > ma[s]
            sh &= ma[f] < ma[s]
    below, above = c < ref, c > ref
    return lo, sh, below & np.r_[False, above[:-1]], above & np.r_[False, below[:-1]]


def none_mask(n: int) -> tuple:
    ok = np.ones(n, bool)
    no = np.zeros(n, bool)
    return ok, ok, no, no


def walk(res: Dict[tuple, tuple], months: list) -> dict:
    """7번째 구간부터 그 전 구간으로 승률 최고를 골라 다음 구간에 쓴다."""
    start = months[0][0]
    picks: list = []
    pnl: Dict[float, list] = {c: [] for c in COSTS}
    for m in range(6, 12):
        lo, hi = months[m]
        best = None
        for key, (ei, xi, _, gr) in res.items():
            sel = (ei >= start) & (xi < lo)
            if sel.sum() < MIN_TRAIN:
                continue
            net = gr[sel].astype(float) - 2 * COSTS[0]
            total = float(np.prod(1 + net) - 1)
            if total <= 0:
                continue
            cand = (float((net > WIN).mean()), total, key)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None:
            picks.append(None)
            continue
        picks.append(best[2])
        ei, _, _, gr = res[best[2]]
        k = (ei >= lo) & (ei < hi)
        for c in COSTS:
            pnl[c] += (gr[k].astype(float) - 2 * c).tolist()
    return {"picks": picks, "pnl": pnl}


def pooled(v: list) -> dict:
    a = np.array(v, dtype=float)
    if not len(a):
        return {"n": 0, "win": 0.0, "ret": 0.0, "avg": 0.0, "med": 0.0, "worst": 0.0}
    return {
        "n": len(a),
        "win": round(float((a > WIN).mean() * 100), 1),
        "ret": round(float((np.prod(1 + a) - 1) * 100), 1),
        "avg": round(float(a.mean() * 100), 3),
        "med": round(float(np.median(a) * 100), 3),
        "worst": round(float(a.min() * 100), 2),
    }


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def predict(x: np.ndarray, fit: tuple) -> np.ndarray:
    w, mu, s = fit
    z = np.c_[np.ones(len(x)), (x - mu) / s]
    return 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))


def stepwise(c: np.ndarray, ma: dict, base: tuple, months: list) -> dict:
    """서명 거리 16개에서 앞으로 선택. 이평 k개의 표본 밖 로그손실을 잰다."""
    ei, xi, sd, gr = base
    dist = np.column_stack([np.log(c / ma[n]) for n in GRID])
    x = dist[ei - 1] * sd[:, None]
    y = ((gr.astype(float) - 2 * COSTS[0]) > WIN).astype(float)
    loss = {k: 0.0 for k in range(0, max(KS) + 1)}
    count = 0
    chosen: list = []
    for m in range(6, 12):
        lo, hi = months[m]
        tr = (xi < lo) & (ei >= months[0][0])
        te = (ei >= lo) & (ei < hi)
        if te.sum() == 0:
            continue
        count += int(te.sum())
        rate = float(np.clip(y[tr].mean(), 1e-3, 1 - 1e-3))
        loss[0] += logloss(y[te], np.full(int(te.sum()), rate))
        cols: List[int] = []
        for k in KS:
            best = None
            for j in range(len(GRID)):
                if j in cols:
                    continue
                f = fit_logit(x[tr][:, cols + [j]], y[tr])
                v = logloss(y[tr], predict(x[tr][:, cols + [j]], f))
                if best is None or v < best[0]:
                    best = (v, j, f)
            cols.append(best[1])
            loss[k] += logloss(y[te], predict(x[te][:, cols], best[2]))
        chosen.append([GRID[j] for j in cols])
    return {"n": count, "loss": {k: round(v / max(count, 1), 4) for k, v in loss.items()},
            "chosen": chosen}


def describe(c: np.ndarray, ma: dict, base: tuple) -> list:
    """서술용. 신호 시점 위치별 승률과 평균. 선택에 쓰지 않는다."""
    ei, _, sd, gr = base
    net = gr.astype(float) - 2 * COSTS[0]
    out = []
    for n in GRID:
        v = ma[n][ei - 1]
        with_ = (c[ei - 1] - v) * sd > 0
        prev = ma[n][ei - 1 - SLOPE]
        slope = (v - prev) * sd > 0
        row: dict = {"n": n}
        for name, k in (("above", with_), ("below", ~with_), ("up", slope), ("down", ~slope)):
            row[name] = [int(k.sum()),
                         round(float((net[k] > WIN).mean() * 100), 1) if k.any() else 0.0,
                         round(float(net[k].mean() * 100), 3) if k.any() else 0.0]
        out.append(row)
    return out


def family(d: dict, sig: tuple, table: dict, a: int, b: int) -> Dict[int, Dict[tuple, tuple]]:
    """이평 개수별 결과. 키는 (규칙, 청산)."""
    out: Dict[int, Dict[tuple, tuple]] = {}
    for k in KS:
        res = {}
        for rule in rules(k):
            perm = table[rule]
            for ex in EXITS:
                res[rule, ex] = run(d, sig, perm, ex, a, b)
        out[k] = res
    return out


def rolled(table: dict, sh: int, a: int, b: int) -> dict:
    out = {}
    for rule, q in table.items():
        r = []
        for arr in q:
            c = arr.copy()
            c[a:b] = np.roll(arr[a:b], sh)
            r.append(c)
        out[rule] = tuple(r)
    return out


def neighbours(rule: Rule) -> List[Rule]:
    out = []
    for i, n in enumerate(rule[1:], 1):
        j = GRID.index(n)
        for q in (j - 1, j + 1):
            if not 0 <= q < len(GRID):
                continue
            ns = list(rule[1:])
            ns[i - 1] = GRID[q]
            if rule[0] == "O" and (len(set(ns)) < len(ns) or ns != sorted(ns)):
                continue
            out.append((rule[0],) + tuple(ns))
    return out


def label(key: tuple) -> str:
    rule, ex = key
    return "_".join(map(str, rule)) + "_" + ex


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    p.add_argument("--rotations", type=int, default=100)
    p.add_argument("--timing", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)

    d = load(data / "30m.csv")
    c = d["c"]
    n = len(c)
    a, b = n - YEAR, n
    step = YEAR // 12
    months = [(a + i * step, a + (i + 1) * step if i < 11 else b) for i in range(12)]
    cut = months[9][0]
    sigs = raw_signals(d)
    ma = {g: sma(c, g) for g in GRID}
    table = {r: rule_mask(r, c, ma) for k in KS for r in rules(k)}
    print("rules", len(table), "bars", n, flush=True)

    if args.timing:
        t0 = time.time()
        res = family(d, sigs["S3"], table, a, b)
        t1 = time.time()
        for k in KS:
            walk(res[k], months)
        print("grid_s", round(t1 - t0, 1), "walk_s", round(time.time() - t1, 1), flush=True)
        return

    base = {s: run(d, sigs[s], none_mask(n), "O", a, b) for s in SIGS}
    rep: dict = {"bars": n, "rules": len(table), "hold_year": round((c[-1] / d["o"][a] - 1) * 100, 1)}
    fams = {}
    for s in SIGS:
        fams[s] = family(d, sigs[s], table, a, b)
        ei0, _, _, gr0 = base[s]
        tail = gr0[ei0 >= months[6][0]].astype(float)
        row: dict = {
            "k0": {
                "year": stats(base[s], a, b, COSTS[0]),
                "wf": {str(c_): pooled((tail - 2 * c_).tolist()) for c_ in COSTS},
            },
            "describe": describe(c, ma, base[s]),
            "stepwise": stepwise(c, ma, base[s], months),
        }
        for k in KS:
            res = fams[s][k]
            wf = walk(res, months)
            top = []
            for key, tr in res.items():
                st = stats(tr, a, b, COSTS[0])
                if st["n"] >= 30:
                    top.append((st["win"], st["ret"], key, st))
            top.sort(key=lambda r: (-r[0], -r[1]))
            cards = []
            for t in top[:2]:
                key = t[2]
                near = []
                for nb in neighbours(key[0]):
                    if (nb, key[1]) in res:
                        near.append(stats(res[nb, key[1]], a, b, COSTS[0])["ret"])
                cards.append({
                    "key": label(key), **t[3],
                    "year_c10": stats(res[key], a, b, COSTS[1])["ret"],
                    "last3": stats(res[key], cut, b, COSTS[0]),
                    "near_n": len(near),
                    "near_med": float(np.median(near)) if near else None,
                    "near_pos": int(sum(v > 0 for v in near)),
                })
            row[f"k{k}"] = {
                "picks": [None if q is None else label(q) for q in wf["picks"]],
                "wf": {str(c_): pooled(v) for c_, v in wf["pnl"].items()},
                "top2": cards,
                "n_ok": len(top),
            }
            print(s, k, row[f"k{k}"]["wf"], flush=True)
        rep[s] = row
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))

    d2 = load(data / "15m.csv", offset_min=15, merge=True)
    sig2 = raw_signals(d2)
    ma2 = {g: sma(d2["c"], g) for g in GRID}
    n2 = len(d2["c"])
    a2 = n2 - YEAR
    rep["offset"] = []
    for s in SIGS:
        for k in KS:
            res = fams[s][k]
            wf = walk(res, months)
            seen = {q for q in wf["picks"] if q is not None}
            for key in seen:
                t2 = run(d2, sig2[s], rule_mask(key[0], d2["c"], ma2), key[1], a2, n2)
                rep["offset"].append({
                    "sig": s, "k": k, "key": label(key),
                    "orig": stats(fams[s][k][key], a, b, COSTS[0])["ret"],
                    "shift": stats(t2, a2, n2, COSTS[0])["ret"],
                })
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))

    rng = random.Random(20261103)
    real = {s: {k: rep[s][f"k{k}"]["wf"][str(COSTS[0])]["ret"] for k in KS} for s in SIGS}
    beat = {s: {k: 0 for k in KS} for s in SIGS}
    for it in range(args.rotations):
        sh = rng.randrange(30 * 48, (b - a) - 30 * 48)
        t2 = rolled(table, sh, a, b)
        for s in SIGS:
            f2 = family(d, sigs[s], t2, a, b)
            for k in KS:
                v = pooled(walk(f2[k], months)["pnl"][COSTS[0]])["ret"]
                beat[s][k] += v >= real[s][k]
        if (it + 1) % 10 == 0:
            print("rot", it + 1, beat, flush=True)
    rep["rotation"] = {
        s: {str(k): round((beat[s][k] + 1) / (args.rotations + 1), 3) for k in KS} for s in SIGS
    }
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("done", rep["rotation"], flush=True)


if __name__ == "__main__":
    main()
