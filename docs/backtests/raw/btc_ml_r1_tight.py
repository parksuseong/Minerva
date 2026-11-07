"""150번: AS4 신호를 R1 일봉 방향과 같은 쪽만 + 짧은 익절 T75, T100.
등록: 2026-11-07-btc-ml-r1-tight-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_r1_tight.py --out <폴더> \
    [--null 200]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import btc_fut_daily_regime as R
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_bracket_grid import run_ml
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tight_target import trade_stats
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, evaluate
from btc_strategy_combo import RES, ts

CELLS = {"T75": (0.0075, "1.5"), "T100": (0.010, "1.5")}


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
    f5k = {k: v[f5["t"] <= end + 900_000] for k, v in f5.items()}
    reg, _ = R.regime(f5k, P, 0)
    reg = np.asarray(reg)
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp_t)
    G = Grid(P, f5k)
    i0 = int(np.searchsorted(tp_t, ts(2023)))
    periods = {"2023": (ts(2023), ts(2024)), "2024~": (ts(2024), end + 1)}

    def as4(probs):
        li, si = rule_masks(probs, tp_t, 0.90, 4)
        return li & (tr4 > 0), si & (tr4 < 0)

    def gated(L, S, rg):
        return L & (rg > 0), S & (rg < 0)

    model, model15 = Model(True, 0.0005), Model(True, 0.0005, 1.5)
    base = as4(pm)
    rng = np.random.default_rng(7)
    span = G.n - i0
    shifts = rng.integers(96 * 7, span - 96 * 7, size=a.null)

    def roll(x, k):
        y = x.copy()
        y[i0:] = np.roll(x[i0:], int(k))
        return y

    rep: dict = {}
    t0 = time.time()
    for cname, (tp, sl) in CELLS.items():
        Lm, Sm = gated(*base, reg)
        g = run_ml(G, Lm, Sm, i0, model, tp, sl, None)
        row = evaluate(g, G, ft, fr, periods, (1.0, 3.0))
        row.pop("_series")
        for pn, (p0, p1) in periods.items():
            row[f"{pn} stats"] = trade_stats(g, G, ft, fr, p0, p1)
        row["c15_2024~"] = evaluate(run_ml(G, Lm, Sm, i0, model15, tp, sl, None), G, ft, fr,
                                    {"2024~": periods["2024~"]}, (1.0,))["2024~ trades"]
        row["seeds_avg_2024~"] = [evaluate(run_ml(G, *gated(*as4(ps[k]), reg), i0, model, tp, sl, None), G, ft, fr,
                                           {"2024~": periods["2024~"]}, (1.0,))["2024~ trades"].get("avg_pct")
                                  for k in range(5)]
        act = row["2024~ trades"]
        for nname in ("ml_shift", "dir_shift"):
            hits, nulls = 0, []
            for s in shifts:
                if nname == "ml_shift":
                    Ln, Sn = gated(roll(base[0], s), roll(base[1], s), reg)
                else:
                    Ln, Sn = gated(base[0], base[1], roll(reg, s))
                rn = evaluate(run_ml(G, Ln, Sn, i0, model, tp, sl, None), G, ft, fr,
                              {"2024~": periods["2024~"]}, (1.0,))["2024~ trades"]
                ev, wn = rn.get("avg_pct"), rn.get("win")
                if ev is None:
                    continue
                nulls.append((ev, wn))
                hits += ev >= act["avg_pct"] and wn >= act["win"]
            row[f"{nname}_p"] = round(hits / len(shifts), 4)
            row[f"{nname}_mean"] = [round(float(np.mean([x[0] for x in nulls])), 3),
                                    round(float(np.mean([x[1] for x in nulls])), 1)]
        rep[cname] = row
        print(cname, row["2023 stats"], row["2024~ stats"], act, row["c15_2024~"].get("avg_pct"), row["seeds_avg_2024~"],
              row["ml_shift_p"], row["ml_shift_mean"], row["dir_shift_p"], row["dir_shift_mean"],
              {p: (row[p + " 3x"].get("cagr"), row[p + " 3x"].get("mdd")) for p in periods}, round(time.time() - t0),
              flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
