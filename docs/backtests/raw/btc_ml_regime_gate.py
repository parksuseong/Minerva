"""135번: ML 규칙(AS4, BS, B)에 추세·횡보 진입 문. 등록: 2026-11-07-btc-ml-regime-gate-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_regime_gate.py --out <폴더> \
    [--null 200]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample
from btc_futures_compare import load_funding
from btc_ml_bracket_grid import RULES, run_ml
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, evaluate
from btc_strategy_combo import RES, ts

DAY = 86_400_000
M5 = 300_000
BAR = 900_000
GATE_RULES = ("AS4", "BS", "B")
GATES = ("G1", "G2", "G3", "G4")


def on_15m(close_t: np.ndarray, val: np.ndarray, t15: np.ndarray) -> np.ndarray:
    """15분봉 마감 시각에 이미 마감된 상위 봉의 값. 없으면 nan."""
    j = np.searchsorted(close_t, t15 + BAR, side="right") - 1
    out = np.full(len(t15), np.nan)
    ok = j >= 0
    out[ok] = val[j[ok]]
    return out


def gate_g1(f5: dict, t15: np.ndarray) -> np.ndarray:
    u = resample(f5, 288)
    c = u["c"]
    er = np.full(len(c), np.nan)
    ad = np.abs(np.diff(c))
    for i in range(10, len(c)):
        den = ad[i - 10:i].sum()
        er[i] = abs(c[i] - c[i - 10]) / den if den > 0 else np.nan
    v = on_15m(u["t"] + 288 * M5, er, t15)
    with np.errstate(invalid="ignore"):
        return v >= 0.30


def gate_g2(f5: dict, t15: np.ndarray) -> np.ndarray:
    u = resample(f5, 48)
    h, lo, c = u["h"], u["l"], u["c"]
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - lo, np.abs(h - pc), np.abs(lo - pc)], 1)
    ci = np.full(len(c), np.nan)
    for i in range(14, len(c)):
        rng = h[i - 13:i + 1].max() - lo[i - 13:i + 1].min()
        s = tr[i - 13:i + 1].sum()
        ci[i] = 100 * np.log10(s / rng) / np.log10(14) if rng > 0 else np.nan
    v = on_15m(u["t"] + 48 * M5, ci, t15)
    with np.errstate(invalid="ignore"):
        return v < 50


def gate_g3(f5: dict, t15: np.ndarray) -> np.ndarray:
    u = resample(f5, 48)
    lr = np.r_[np.nan, np.diff(np.log(u["c"]))]
    rv = np.full(len(lr), np.nan)
    for i in range(6, len(lr)):
        rv[i] = np.std(lr[i - 5:i + 1])
    pct = np.full(len(rv), np.nan)
    for i in range(1080 + 6, len(rv)):
        past = rv[i - 1080:i]
        past = past[np.isfinite(past)]
        if len(past) > 500 and np.isfinite(rv[i]):
            pct[i] = (past < rv[i]).mean()
    v = on_15m(u["t"] + 48 * M5, pct, t15)
    with np.errstate(invalid="ignore"):
        return v >= 0.50


def ratio(row, p):
    x = row.get(f"{p} 1x", {}).get("ratio")
    return x if x is not None else -9.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--null", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    ft, fr = load_funding()
    z = np.load(RES / "ml15/pred.npz")
    tp_t = z["t"]
    spot = {k: z[k] for k in "ohlc"}
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    pm = np.mean(ps, 0)
    P, _ = futures_on(tp_t, spot)
    P["t"] = tp_t
    end = float(tp_t[-1])
    f5k = {k: v[f5["t"] <= end + BAR] for k, v in f5.items()}
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp_t)
    G = Grid(P, f5k)
    i0 = int(np.searchsorted(tp_t, ts(2023)))
    lm0 = datetime(2026, 10, 6, 15, tzinfo=timezone.utc).timestamp() * 1000
    periods = {"2023": (ts(2023), ts(2024)), "2024~": (ts(2024), end + 1), "lastmonth": (lm0, end + 1)}
    g1, g2, g3 = gate_g1(f5k, tp_t), gate_g2(f5k, tp_t), gate_g3(f5k, tp_t)
    gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g1 & g2}
    w0 = datetime(2026, 10, 8, 15, tzinfo=timezone.utc).timestamp() * 1000   # 10/09 00:00 KST
    w1 = datetime(2026, 10, 28, 15, tzinfo=timezone.utc).timestamp() * 1000  # 10/29 00:00 KST
    win = (tp_t >= w0) & (tp_t < w1)
    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(),
                 "gate_on_2023~": {k: round(float(v[i0:].mean()) * 100, 1) for k, v in gates.items()},
                 "gate_off_1009_1028": {k: round(float((~v[win]).mean()) * 100, 1) for k, v in gates.items()}}
    print(rep, flush=True)

    def masks(rule, probs):
        q, k, _, gate = RULES[rule]
        li, si = rule_masks(probs, tp_t, q, k)
        if gate:
            li, si = li & (tr4 > 0), si & (tr4 < 0)
        return li, si

    model, model15 = Model(True, 0.0005), Model(True, 0.0005, 1.5)
    keep, series = {}, {}
    t0 = time.time()
    for r in GATE_RULES:
        main_probs = pm if RULES[r][2] else ps[0]
        Lm, Sm = masks(r, main_probs)
        seed_masks = [masks(r, ps[k]) for k in range(5)] if RULES[r][2] else []
        for gname in ("none",) + GATES:
            g = np.ones(len(tp_t), bool) if gname == "none" else gates[gname]
            key = f"{r}|{gname}"
            grp = run_ml(G, Lm & g, Sm & g, i0, model, 0.015, "1.5", None)
            row = evaluate(grp, G, ft, fr, periods, (1.0, 3.0))
            series[key] = row.pop("_series")
            g15 = run_ml(G, Lm & g, Sm & g, i0, model15, 0.015, "1.5", None)
            r15 = evaluate(g15, G, ft, fr, {p: periods[p] for p in ("2023", "2024~")}, (1.0,))
            r15.pop("_series")
            row["c15"] = {p: r15[f"{p} 1x"] for p in ("2023", "2024~")}
            if seed_masks:
                row["seed_ratio_2024"] = []
                for Ls, Ss in seed_masks:
                    gs = run_ml(G, Ls & g, Ss & g, i0, model, 0.015, "1.5", None)
                    rs = evaluate(gs, G, ft, fr, {"2024~": periods["2024~"]}, (1.0,))
                    row["seed_ratio_2024"].append(ratio(rs, "2024~"))
            rep[key] = row
            keep[key] = grp
            print(key, row["groups"], {p: (row[f"{p} 1x"].get("cagr"), row[f"{p} 1x"].get("mdd"),
                                           row[f"{p} 1x"].get("ratio")) for p in ("2023", "2024~")},
                  round(time.time() - t0), flush=True)
    # 판정
    span = G.n - i0
    rng = np.random.default_rng(7)
    shifts = rng.integers(96 * 7, span - 96 * 7, size=a.null)
    passed = []
    for r in GATE_RULES:
        base = rep[f"{r}|none"]
        Lm, Sm = masks(r, pm if RULES[r][2] else ps[0])
        for gname in GATES:
            key = f"{r}|{gname}"
            row = rep[key]
            c = {}
            c["1"] = all(ratio(row, p) > ratio(base, p) and
                         ((row.get(f"{p} trades") or {}).get("avg_pct") or -9) >
                         ((base.get(f"{p} trades") or {}).get("avg_pct") or -9) for p in ("2023", "2024~"))
            tr = row.get("2024~ trades") or {}
            c["2"] = (tr.get("long_pct") if tr.get("long_pct") is not None else 0) >= 0 and \
                     (tr.get("short_pct") if tr.get("short_pct") is not None else 0) >= 0
            if RULES[r][2]:
                c["3"] = sum(x > y for x, y in zip(row["seed_ratio_2024"], base["seed_ratio_2024"])) >= 3
            else:
                c["3"] = None
            c["4"] = all((row["c15"][p].get("total") or -1) > 0 for p in ("2023", "2024~"))
            # 문 원형 이동 대조
            g = gates[gname]
            actual = ratio(row, "2024~")
            null = []
            for k in shifts:
                gr = g.copy()
                gr[i0:] = np.roll(g[i0:], int(k))
                grp = run_ml(G, Lm & gr, Sm & gr, i0, model, 0.015, "1.5", None)
                rr = evaluate(grp, G, ft, fr, {"2024~": periods["2024~"]}, (1.0,))
                null.append(ratio(rr, "2024~"))
            null = np.array(null)
            c["null_mean"] = round(float(null.mean()), 3)
            c["p"] = round(float((null >= actual).mean()), 4)
            c["5"] = c["p"] < 0.0042
            c["pass"] = bool(c["1"] and c["2"] and (c["3"] is not False) and c["4"] and c["5"])
            rep[f"crit|{key}"] = c
            if c["pass"]:
                passed.append(key)
            print("crit", key, c, round(time.time() - t0), flush=True)
    rep["passed"] = passed
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (out / "groups.json").write_text(json.dumps(
        {k: [[int(e), int(s), float(segs[0][1]), int(segs[0][3]), float(segs[0][4])] for e, s, segs, _ in g]
         for k, g in keep.items()}))
    np.savez(out / "series.npz", **{k.replace("|", "__"): np.c_[v[0], v[1]] for k, v in series.items()},
             t15=tp_t, **{g: v for g, v in gates.items()}, c=P["c"])


if __name__ == "__main__":
    main()
