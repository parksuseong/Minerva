"""A평균 신호를 4시간 추세 나이로 나누기. 등록: 2026-11-05-btc-ml-trend-age-hypotheses.md

실행: PYTHONPATH=<pyml>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ml_trend_age.py \
      --pred <ml15/pred.npz> --seeds <seeds 폴더> --out <결과.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
import numpy as np
from btc_futures_compare import load_funding
from btc_ml_edge_cost import rest_sim, summ_net
from btc_ml_mtf_trend import BAR15, load_frame
from btc_ml_seeds import RULES, SEEDS, rule_masks

YOUNG, OLD = 6, 30
YEAR_MS = 365 * 86_400_000


def trend_age(t15: np.ndarray):
    ct, tr = load_frame("4h")
    age = np.ones(len(tr), int)
    for i in range(1, len(tr)):
        age[i] = age[i - 1] + 1 if tr[i] == tr[i - 1] else 1
    j = np.searchsorted(ct, t15 + BAR15 - 1, side="right") - 1
    ok = j >= 0
    j = np.maximum(j, 0)
    return np.where(ok, tr[j], 0.0), np.where(ok, age[j], 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.pred)
    t = z["t"]
    n = len(t)
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    ft, fr = load_funding()
    sets = [z["probs"]] + [np.load(Path(a.seeds) / f"pred_s{k}.npz")["probs"] for k in SEEDS]
    avg = np.mean(np.stack(sets), 0)
    tr, age = trend_age(t)
    young, old = age <= YOUNG, age > OLD
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    iy = int(np.searchsorted(t, t[-1] - YEAR_MS))
    W = {"2023": (i23, i24), "2024~": (i24, n), "pre1y": (i24, iy), "1y": (iy, n)}

    def run(li, si, per, sides=(1, -1), k=1.0):
        a0, a1 = W[per]
        return summ_net(rest_sim(P, t, li, si, a0, a1, sides, k), ft, fr,
                        (t[a1 - 1] - t[a0]) / F.YEAR)

    def variant(li, si, g):
        with_l, with_s = li & (tr > 0), si & (tr < 0)
        if g == "G1":
            return with_l, with_s
        if g == "G2":
            return with_l & young, with_s & young
        return with_l | (li & (tr < 0) & old), with_s | (si & (tr > 0) & old)

    q, kk = RULES["A"]
    li, si = rule_masks(avg, t, q, kk)
    res = {"buckets": {}}
    agebins = {"young": young, "mid": ~young & ~old, "old": old}
    rels = {"with": (tr > 0, tr < 0), "against": (tr < 0, tr > 0)}
    buckets = {f"{r}/{ab}": (ml & am, ms & am)
               for r, (ml, ms) in rels.items() for ab, am in agebins.items()}
    buckets["neutral"] = (tr == 0, tr == 0)
    for key, (ml, ms) in buckets.items():
        bl, bs = li & ml, si & ms
        res["buckets"][key] = {per: run(bl, bs, per) for per in ("2023", "pre1y", "1y")}
        print("bucket", key, res["buckets"][key], flush=True)
    tab = {}
    for g in ("G1", "G2", "G3"):
        vl, vs = variant(li, si, g)
        tab[g] = {per: run(vl, vs, per) for per in W}
        print(g, tab[g], flush=True)
    pick = max(("G2", "G3"), key=lambda g: tab[g]["2023"]["x3"])
    vl, vs = variant(li, si, pick)
    lo_, sh_ = run(vl, vs, "2024~", (1,)), run(vl, vs, "2024~", (-1,))
    fee = run(vl, vs, "2024~", k=1.5)
    beat, seeds = 0, {}
    for k, pr in enumerate(sets):
        l1, s1 = rule_masks(pr, t, q, kk)
        x1 = run(*variant(l1, s1, "G1"), "2024~")["x3"]
        x2 = run(*variant(l1, s1, pick), "2024~")["x3"]
        seeds[f"s{k}"] = {"G1": x1, pick: x2}
        beat += x2 > x1
    c, b = tab[pick], tab["G1"]
    crit = [all(c[p]["x3"] > b[p]["x3"] for p in ("2023", "2024~", "1y")) and c["1y"]["x3"] > 0,
            lo_["ev3"] > 0 and sh_["ev3"] > 0, fee["x3"] > 0, beat >= 4,
            c["1y"]["per_year"] >= 50]
    res.update({"table": tab, "pick": pick, "long": lo_, "short": sh_, "fee1.5": fee,
                "seeds": seeds, "seed_beat": beat, "crit": [bool(x) for x in crit],
                "pass": bool(all(crit))})
    print("PICK", pick, {k: v for k, v in res.items() if k not in ("buckets", "table")},
          flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
