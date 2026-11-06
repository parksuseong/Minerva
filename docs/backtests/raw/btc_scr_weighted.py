"""15분봉 SCR 가중 점수 신호. 등록: 2026-11-06-btc-scr15-weighted-score-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_weighted.py \
    --out <폴더> [--offset 5] [--futures] [--perm <규칙 이름>]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from btc_fractal import cci, crossed, load5, recent, resample, rsi, stoch
from btc_scr_entry_stop import label, stop_price, summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_swing_patterns import SPLIT

W = (0, 0.5, 1, 1.5, 2)
T = tuple(x / 2 for x in range(1, 13))
CASES = list(itertools.product((0, 1), repeat=3))  # (S, C, R)


def rules() -> dict:
    """참·거짓표가 같은 (가중치, 기준)은 하나로 묶는다. 이름은 켜지는 경우를 적는다."""
    out: dict = {}
    for ws, wc, wr in itertools.product(W, repeat=3):
        for t in T:
            tab = tuple(ws * s + wc * c + wr * r >= t for s, c, r in CASES)
            if not any(tab) or all(tab):
                continue
            name = "|".join("".join(k for k, v in zip("SCR", cs) if v) for cs, on in zip(CASES, tab) if on)
            out.setdefault(tab, {"name": name, "w": (ws, wc, wr, t), "n_combos": 0})
            out[tab]["n_combos"] += 1
    return out


def comps(d: dict) -> dict:
    s, cc, r = stoch(d["h"], d["l"], d["c"]), cci(d["h"], d["l"], d["c"]), rsi(d["c"])
    return {1: (recent(crossed(s, 20, True)), recent(crossed(cc, -100, True)), recent(crossed(r, 30, True))),
            -1: (recent(crossed(s, 80, False)), recent(crossed(cc, 100, False)), recent(crossed(r, 70, False)))}


def signal(cp: tuple, tab: tuple) -> np.ndarray:
    s, c, r = (x.astype(int) for x in cp)
    lv = np.asarray(tab)[s * 4 + c * 2 + r]
    return lv & ~np.r_[False, lv[:-1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--futures", action="store_true")
    ap.add_argument("--perm", default="", help="검증 구간 무작위 대조를 돌릴 규칙 이름")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / ("futures/klines_5m.csv" if args.futures else "5m.csv"))
    fund = load_funding(base / "futures/funding.csv") if args.futures else None
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, 3)
    d["t"] = d["t"] + args.offset * 60_000
    n = len(d["c"])
    cp = comps(d)
    rep: dict = {"offset": args.offset, "futures": args.futures, "rules": {}}
    for tab, meta in rules().items():
        on = {1: signal(cp[1], tab), -1: signal(cp[-1], tab)}
        cands = []
        for side in (1, -1):
            opp = on[-side]
            for i in np.flatnonzero(on[side]):
                if i < 6000 or i >= n - 2:
                    continue
                res = label(d, int(i), side, opp, stop_price(d, int(i), side, "S1"))
                if res is None:
                    continue
                g, x, dist, stopped = res
                g = with_funding(g, d, int(i), x, side, fund)
                cands.append((int(i), side, float(g), int(x), float(d["t"][i]), stopped, dist))
        cands.sort()
        tr = walk(cands)
        te = [z for z in tr if z[4] >= SPLIT]
        r = {**meta, "train": summ([z for z in tr if z[4] < SPLIT]), "test": summ(te),
             "test_c10": summ(te, 0.001), "all": summ(tr)}
        if args.perm == meta["name"] and te:
            rng = np.random.default_rng(7)
            idx = np.flatnonzero((d["t"] >= SPLIT) & (np.arange(n) >= 6000) & (np.arange(n) < n - 2))
            nl = sum(1 for z in te if z[1] > 0)
            actual = float(np.mean([z[2] for z in te]))
            cache: dict = {}

            def outcome(i: int, side: int):
                if (i, side) not in cache:
                    res = label(d, i, side, on[-side], stop_price(d, i, side, "S1"))
                    cache[(i, side)] = None if res is None else (with_funding(res[0], d, i, res[1], side, fund), res[1])
                return cache[(i, side)]

            hits = 0
            for _ in range(2000):
                picks = sorted([(int(k), 1) for k in rng.choice(idx, nl, replace=False)]
                               + [(int(k), -1) for k in rng.choice(idx, len(te) - nl, replace=False)])
                busy, gs = -1, []
                for i, side in picks:
                    if i <= busy:
                        continue
                    oc = outcome(i, side)
                    if oc is None:
                        continue
                    gs.append(oc[0])
                    busy = oc[1]
                if gs and np.mean(gs) >= actual:
                    hits += 1
            r["perm_p"] = round(hits / 2000, 4)
        if meta["name"] in ("SCR", args.perm):
            r["trades"] = [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]
        rep["rules"][meta["name"]] = r
    elig = {k: v for k, v in rep["rules"].items() if v["train"].get("n", 0) >= 100}
    rep["picked"] = max(elig, key=lambda k: elig[k]["train"]["ev"])
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for k, v in sorted(rep["rules"].items(), key=lambda kv: -kv[1]["train"].get("ev", -9)):
        print(f"{k:16s} w={v['w']} train n={v['train'].get('n')} ev={v['train'].get('ev')} | "
              f"test n={v['test'].get('n')} ev={v['test'].get('ev')} c10={v['test_c10'].get('ev')} "
              f"drop5={v['test'].get('drop5')} p={v.get('perm_p', '')}", flush=True)
    print("picked", rep["picked"])


if __name__ == "__main__":
    main()
