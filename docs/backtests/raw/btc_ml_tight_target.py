"""149번: ML 진입(B, BS, AS4) + 짧은 익절 T75(+0.75/−1.5), T50(+0.5/−1.0). 승률 70% 목표.
등록: 2026-11-07-btc-ml-tight-target-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_tight_target.py --out <폴더> \
    [--null 300]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_bracket_grid import run_ml
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, evaluate, path_returns
from btc_strategy_combo import RES, ts

CELLS = {"T75": (0.0075, "1.5"), "T50": (0.005, "1.0")}
RULES = {"B": (0.95, 5, False, False), "BS": (0.95, 5, True, False), "AS4": (0.90, 4, True, True)}


def trade_stats(g, G, ft, fr, p0, p1) -> dict:
    _, pg, _ = path_returns(g, G.A, 1.0, ft, fr)
    p = np.array([q for e, _, q in pg if p0 <= G.A["t"][e] < p1])
    if not len(p):
        return {"n": 0}
    w = p[p > 0]
    return {"n": len(p), "win": round(float((p > 0).mean() * 100), 1), "avg_pct": round(float(p.mean() * 100), 3),
            "avg_win": round(float(w.mean() * 100), 3) if len(w) else None,
            "avg_loss": round(float(p[p <= 0].mean() * 100), 3) if (p <= 0).any() else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--null", type=int, default=300)
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
    f5k = {k: v[f5["t"] <= end + 900_000] for k, v in f5.items()}
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp_t)
    G = Grid(P, f5k)
    i0 = int(np.searchsorted(tp_t, ts(2023)))
    periods = {"2023": (ts(2023), ts(2024)), "2024~": (ts(2024), end + 1)}

    def masks(rule, probs):
        q, k, _, gate = RULES[rule]
        li, si = rule_masks(probs, tp_t, q, k)
        if gate:
            li, si = li & (tr4 > 0), si & (tr4 < 0)
        return li, si

    model, model15 = Model(True, 0.0005), Model(True, 0.0005, 1.5)
    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat()}
    t0 = time.time()
    rng = np.random.default_rng(7)
    span = G.n - i0
    shifts = rng.integers(96 * 7, span - 96 * 7, size=a.null)
    for r in RULES:
        Lm, Sm = masks(r, pm if RULES[r][2] else ps[0])
        for cname, (tp, sl) in CELLS.items():
            key = f"{r}|{cname}"
            g = run_ml(G, Lm, Sm, i0, model, tp, sl, None)
            row = evaluate(g, G, ft, fr, periods, (1.0, 3.0))
            row.pop("_series")
            for pn, (p0, p1) in periods.items():
                row[f"{pn} stats"] = trade_stats(g, G, ft, fr, p0, p1)
            g15 = run_ml(G, Lm, Sm, i0, model15, tp, sl, None)
            r15 = evaluate(g15, G, ft, fr, {"2024~": periods["2024~"]}, (1.0,))
            row["c15_2024~"] = r15["2024~ trades"]
            if RULES[r][2]:
                pos = []
                for k in range(5):
                    gs = run_ml(G, *masks(r, ps[k]), i0, model, tp, sl, None)
                    rs = evaluate(gs, G, ft, fr, {"2024~": periods["2024~"]}, (1.0,))
                    pos.append(rs["2024~ trades"].get("avg_pct"))
                row["seeds_avg_2024~"] = pos
            act = row["2024~ trades"]
            hits, nulls = 0, []
            for s in shifts:
                Lr, Sr = Lm.copy(), Sm.copy()
                Lr[i0:], Sr[i0:] = np.roll(Lm[i0:], int(s)), np.roll(Sm[i0:], int(s))
                gn = run_ml(G, Lr, Sr, i0, model, tp, sl, None)
                rn = evaluate(gn, G, ft, fr, {"2024~": periods["2024~"]}, (1.0,))["2024~ trades"]
                ev, wn = rn.get("avg_pct"), rn.get("win")
                nulls.append((ev, wn))
                if ev is not None and ev >= act.get("avg_pct", 9) and wn >= act.get("win", 999):
                    hits += 1
            row["null_p"] = round(hits / len(shifts), 4)
            ok = [x for x in nulls if x[0] is not None]
            row["null_mean"] = [round(float(np.mean([x[0] for x in ok])), 3), round(float(np.mean([x[1] for x in ok])), 1)]
            rep[key] = row
            print(key, row["2023 stats"], row["2024~ stats"], act, row.get("seeds_avg_2024~"), row["c15_2024~"].get("avg_pct"),
                  row["null_p"], row["null_mean"], round(time.time() - t0), flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
