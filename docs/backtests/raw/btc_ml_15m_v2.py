"""15분봉 ML + SCR v2: 진입 문턱을 넓혀 거래 수 늘리기. 등록: 2026-11-04-btc-ml-15m-v2-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_15m_v2.py --pred <ml15/pred.npz> --out <폴더>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import btc_ml_ensemble as E
from btc_fractal import recent
from btc_ml_15m import BE, simulate, summ

WS = (4, 8, 16)
MS = (0.0, 0.02, 0.04)


def rules(probs, on_l, on_s, w, m):
    known = np.isfinite(probs).all(0)
    with np.errstate(invalid="ignore"):
        pm = np.where(known, np.nanmean(np.where(known, probs, 0.5), 0), np.nan)
        votes = (probs > BE).sum(0)
        ent = known & recent(on_l, w) & (votes >= 3) & (pm >= BE + m)
        ex = known & recent(on_s, w) & (pm < 0.5)
    return ent, ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.pred)
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    probs, t, on_l, on_s = z["probs"], z["t"], z["on_l"], z["on_s"]
    n = len(t)
    i22, i24 = (int(np.searchsorted(t, E.ts(v))) for v in (2022, 2024))
    rows = {}
    for w, m in itertools.product(WS, MS):
        ent, ex = rules(probs, on_l, on_s, w, m)
        rows[(w, m)] = {"sel": summ(simulate(P, ent, ex, i22, i24)),
                        "test": summ(simulate(P, ent, ex, i24, n))}
        print(f"W{w} m{m}", rows[(w, m)], flush=True)
    best = max(rows, key=lambda k: rows[k]["sel"]["x1"] if rows[k]["sel"]["n"] else -1e9)
    w, m = best
    test = rows[best]["test"]
    scr_only = summ(simulate(P, recent(on_l, w), recent(on_s, w), i24, n))
    rng = np.random.default_rng(91)
    seg = probs[:, i24:].copy()
    L = seg.shape[1]
    rand = []
    for _ in range(200):
        k = int(rng.integers(96 * 30, L - 96 * 30))
        p2 = probs.copy()
        p2[:, i24:] = np.roll(seg, k, axis=1)
        rand.append(summ(simulate(P, *rules(p2, on_l, on_s, w, m), i24, n))["x1"])
    rand = np.array(rand)
    p = float((rand >= test["x1"]).mean())
    crit = [test["n"] >= 30 and test["x1"] > 0 and test.get("ev", -1) > 0,
            test["x1"] > scr_only["x1"], p < 0.05]
    rep = {"rows": {f"W{k[0]}|m{k[1]}": v for k, v in rows.items()}, "best": f"W{w}|m{m}",
           "test": test, "scr_only": scr_only, "rand_median": float(np.median(rand)), "p": p,
           "criteria": crit}
    print("PICK", rep["best"], "test", test, "scr_only", scr_only, "p", p, "crit", crit, flush=True)
    (out / "result.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
