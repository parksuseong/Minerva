"""씨드 평균 A·B + 30분·1시간·4시간 추세 필터. 등록: 2026-11-05-btc-ml-mtf-trend-hypotheses.md

실행: PYTHONPATH=<pyml>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ml_mtf_trend.py \
      --pred <ml15/pred.npz> --seeds <seeds 폴더> --out <결과.json>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
import numpy as np
from btc_futures_compare import load_funding
from btc_ml_edge_cost import mean_score, rest_sim, summ_net
from btc_ml_seeds import RULES, SEEDS, rule_masks

FRAMES = ("30m", "1h", "4h")
VARIANTS = ("F0", "F30", "F1h", "F4h", "FALL")
EMA_N, SLOPE = 50, 5
BAR15 = 900_000
YEAR_MS = 365 * 86_400_000


def load_frame(name: str):
    with open(Path(f"~/.minerva/data/btc/{name}.csv").expanduser()) as f:
        rows = list(csv.DictReader(f))
    ct = np.array([float(r["close_time"]) for r in rows])
    c = np.array([float(r["close"]) for r in rows])
    a = 2 / (EMA_N + 1)
    ema = np.empty_like(c)
    ema[0] = c[0]
    for i in range(1, len(c)):
        ema[i] = a * c[i] + (1 - a) * ema[i - 1]
    tr = np.zeros(len(c))
    up = (c > ema) & (ema > np.r_[np.full(SLOPE, np.inf), ema[:-SLOPE]])
    dn = (c < ema) & (ema < np.r_[np.full(SLOPE, -np.inf), ema[:-SLOPE]])
    tr[up], tr[dn] = 1, -1
    tr[:EMA_N] = 0
    return ct, tr


def trend_at(t15: np.ndarray, frame) -> np.ndarray:
    """신호 봉 마감(t+15분) 시점에 이미 마감된 마지막 봉의 추세."""
    ct, tr = frame
    j = np.searchsorted(ct, t15 + BAR15 - 1, side="right") - 1
    return np.where(j >= 0, tr[np.maximum(j, 0)], 0.0)


def filt(li, si, v, T):
    if v == "F0":
        return li, si
    if v == "FALL":
        u = (T["30m"] > 0) & (T["1h"] > 0) & (T["4h"] > 0)
        d = (T["30m"] < 0) & (T["1h"] < 0) & (T["4h"] < 0)
    else:
        f = {"F30": "30m", "F1h": "1h", "F4h": "4h"}[v]
        u, d = T[f] > 0, T[f] < 0
    return li & u, si & d


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
    sets = {"s0": z["probs"]}
    for k in SEEDS:
        w = np.load(Path(a.seeds) / f"pred_s{k}.npz")
        assert np.array_equal(w["t"], t)
        sets[f"s{k}"] = w["probs"]
    avg = np.mean(np.stack(list(sets.values())), 0)
    T = {f: trend_at(t, load_frame(f)) for f in FRAMES}
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    iy = int(np.searchsorted(t, t[-1] - YEAR_MS))
    W = {"2023": (i23, i24), "2024~": (i24, n), "1y": (iy, n)}

    def run(li, si, per, sides=(1, -1), k=1.0):
        a0, a1 = W[per]
        return summ_net(rest_sim(P, t, li, si, a0, a1, sides, k), ft, fr,
                        (t[a1 - 1] - t[a0]) / F.YEAR)

    def winrate(li, si, per):
        a0, a1 = W[per]
        tr = rest_sim(P, t, li, si, a0, a1, (1, -1))
        return round(100 * float(np.mean([x[1] > 0 for x in tr])), 1) if tr else None

    res = {"trend_share": {f: {"up": float((T[f][i23:] > 0).mean()),
                               "down": float((T[f][i23:] < 0).mean())} for f in FRAMES}}
    rng = np.random.default_rng(1105)
    for r, (q, kk) in RULES.items():
        rule = r + "S"
        li0, si0 = rule_masks(avg, t, q, kk)
        tab = {}
        for v in VARIANTS:
            li, si = filt(li0, si0, v, T)
            tab[v] = {per: run(li, si, per) for per in W}
            tab[v]["win_2024"] = winrate(li, si, "2024~")
            print(rule, v, tab[v], flush=True)
        best = max(VARIANTS, key=lambda v: tab[v]["2023"]["x3"])
        out = {"table": tab, "pick": best}
        if best != "F0":
            li, si = filt(li0, si0, best, T)
            base = tab[best]["2024~"]
            lo_, sh_ = run(li, si, "2024~", (1,)), run(li, si, "2024~", (-1,))
            fee = run(li, si, "2024~", k=1.5)
            # 무작위 제거: 2024~ 원래 신호 중 필터가 남긴 비율만큼 무작위로 남긴다
            keep_l = li[i24:].sum() / max(li0[i24:].sum(), 1)
            keep_s = si[i24:].sum() / max(si0[i24:].sum(), 1)
            rand = []
            for _ in range(200):
                rl, rs = li0.copy(), si0.copy()
                rl[i24:] &= rng.random(n - i24) < keep_l
                rs[i24:] &= rng.random(n - i24) < keep_s
                rand.append(run(rl, rs, "2024~")["x3"])
            p = float((np.array(rand) >= base["x3"]).mean())
            seed_rows, beat, both = {}, 0, 0
            for s, pr in sets.items():
                l1, s1 = rule_masks(pr, t, q, kk)
                f0 = {per: run(l1, s1, per)["x3"] for per in ("2023", "2024~")}
                l2, s2 = filt(l1, s1, best, T)
                fv = {per: run(l2, s2, per)["x3"] for per in ("2023", "2024~")}
                seed_rows[s] = {"F0": f0, best: fv}
                beat += fv["2024~"] > f0["2024~"]
                both += fv["2023"] > 0 and fv["2024~"] > 0
            y1 = tab[best]["1y"]
            crit = [base["x3"] > 0 and base["ev3"] > 0 and base["x3"] > tab["F0"]["2024~"]["x3"],
                    lo_["ev3"] > 0 and sh_["ev3"] > 0, fee["x3"] > 0, p < 0.05,
                    beat >= 4 and both >= 3, y1["x3"] > 0 and y1["per_year"] >= 50]
            out.update({"long": lo_, "short": sh_, "fee1.5": fee, "p": p,
                        "rand_median": float(np.median(rand)), "seeds": seed_rows,
                        "seed_beat": beat, "seed_both": both, "crit": crit,
                        "pass": bool(all(crit))})
        print(rule, "PICK", best, {k: v for k, v in out.items() if k != "table"}, flush=True)
        res[rule] = out
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
