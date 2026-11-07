"""130번: ML 규칙 익절·손절 거리 격자. 등록: 2026-11-07-btc-ml-bracket-grid-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_bracket_grid.py --out <폴더> \
    [--null 20]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import btc_ml_edge_cost as C
import numpy as np
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, evaluate, lim_fill
from btc_r1_k2_stop_fix import daily_atr_adx
from btc_strategy_combo import RES, ts

DAY = 86_400_000
TPS = (0.010, 0.0125, 0.015, 0.020)
SLS = ("1.5", "1.0", "atr0.5")
RULES = {"A": (0.90, 4, False, False), "B": (0.95, 5, False, False), "AS": (0.90, 4, True, False),
         "BS": (0.95, 5, True, False), "AS4": (0.90, 4, True, True)}


def sl_dist(sl: str, atr_p, i: int):
    if sl == "atr0.5":
        a = atr_p[i]
        return None if (np.isnan(a) or a <= 0) else 0.5 * float(a)
    return float(sl) / 100


def bracket(G: Grid, e: int, s: int, px: float, tp: float, sd: float, model: Model, delayed_first: bool):
    """같은 봉이면 손절 먼저, 진입 봉에서는 익절 안 봄. 128번 bracket과 같고 거리만 인자."""
    A = G.A
    o, h, l = A["o"], A["h"], A["l"]  # noqa: E741
    stop, tgt = px * (1 - s * sd), px * (1 + s * tp)
    for b in range(e, G.n - 1):
        first = b == e
        hi, lo = G.first_hl(b, model) if (first and delayed_first) else (h[b], l[b])
        if (lo <= stop) if s > 0 else (hi >= stop):
            f = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
            return b, f * (1 - s * model.slip), model.taker, True
        if not first and ((hi >= tgt * (1 + model.thru)) if s > 0 else (lo <= tgt * (1 - model.thru))):
            return b, tgt, model.maker, False
    return None


def run_ml(G, Lm, Sm, i0, model, tp, sl, atr_p):
    out, i = [], i0
    while i < G.n - 2:
        s = 1 if Lm[i] else (-1 if Sm[i] else 0)
        if not s:
            i += 1
            continue
        sd = sl_dist(sl, atr_p, i)
        if sd is None:
            i += 1
            continue
        f = lim_fill(G, i, s, model)
        if f is None:
            i += 1
            continue
        j, px = f
        x = bracket(G, j, s, px, tp, sd, model, model.delay and j == i + 1)
        if x is None:
            break
        b, xp, xf, _ = x
        out.append((j, s, [(j, px, 1.0, b, float(xp), model.maker, xf)], b))
        i = b
    return out


def cell(tp, sl) -> str:
    return f"TP{tp * 100:g}/SL{sl}"


def crit_basic(row, base) -> dict:
    """기준 1·2·3·5 중 1·2·3 (5는 따로)."""
    def g(r, p, k):
        return r.get(f"{p} 1x", {}).get(k)
    c1 = all(((row.get(f"{p} trades") or {}).get("avg_pct") or -1) > 0 and
             (row.get(f"{p} 3x", {}).get("total") or -1) > 0 for p in ("2023", "2024~"))
    c2 = all((g(row, p, "ratio") if g(row, p, "ratio") is not None else -9) >
             (g(base, p, "ratio") if g(base, p, "ratio") is not None else -9) for p in ("2023", "2024~"))
    tr = row.get("2024~ trades") or {}
    c3 = (tr.get("long_pct") if tr.get("long_pct") is not None else 0) >= 0 and \
         (tr.get("short_pct") if tr.get("short_pct") is not None else 0) >= 0
    return {"1": bool(c1), "2": bool(c2), "3": bool(c3)}


def c5_of(row15) -> bool:
    return all((row15.get(f"{p} 1x", {}).get("total") or -1) > 0 for p in ("2023", "2024~"))


def neighbors(tp, sl):
    ti, si = TPS.index(tp), SLS.index(sl)
    out = []
    for dt in (-1, 1):
        if 0 <= ti + dt < len(TPS):
            out.append((TPS[ti + dt], sl))
    for ds in (-1, 1):
        if 0 <= si + ds < len(SLS):
            out.append((tp, SLS[si + ds]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--null", type=int, default=20)
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
    P, ok = futures_on(tp_t, spot)
    P["t"] = tp_t
    end = float(tp_t[-1])
    f5k = {k: v[f5["t"] <= end + 900_000] for k, v in f5.items()}
    atr_p, _ = daily_atr_adx(f5k, P, 0)
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp_t)
    G = Grid(P, f5k)
    i0 = int(np.searchsorted(tp_t, ts(2023)))
    lm0 = datetime(2026, 10, 6, 15, tzinfo=timezone.utc).timestamp() * 1000  # 10/07 00:00 KST
    periods = {"2023": (ts(2023), ts(2024)), "2024~": (ts(2024), end + 1), "1y": (end - 365 * DAY, end + 1),
               "lastmonth": (lm0, end + 1)}

    def masks(rule, probs):
        q, k, _, gate = RULES[rule]
        li, si = rule_masks(probs, tp_t, q, k)
        if gate:
            li, si = li & (tr4 > 0), si & (tr4 < 0)
        return li, si

    mains = {r: masks(r, pm if RULES[r][2] else ps[0]) for r in RULES}
    model, model15 = Model(True, 0.0005), Model(True, 0.0005, 1.5)
    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(),
                 "atr_nan_2023~": int(np.isnan(atr_p[i0:]).sum()),
                 "note": "ML 라벨은 ±1.5% 먼저 닿기로 학습됨. 다른 거리는 라벨 밖"}
    keep = {}
    t0 = time.time()
    for r in RULES:
        Lm, Sm = mains[r]
        for tp in TPS:
            for sl in SLS:
                key = f"{r}|{cell(tp, sl)}"
                g = run_ml(G, Lm, Sm, i0, model, tp, sl, atr_p)
                row = evaluate(g, G, ft, fr, periods, (1.0, 3.0))
                row.pop("_series")
                g15 = run_ml(G, Lm, Sm, i0, model15, tp, sl, atr_p)
                r15 = evaluate(g15, G, ft, fr, periods, (1.0,))
                r15.pop("_series")
                row["c15"] = {p: r15[f"{p} 1x"] for p in ("2023", "2024~")}
                if RULES[r][2]:
                    pos = 0
                    for k in range(5):
                        gs = run_ml(G, *masks(r, ps[k]), i0, model, tp, sl, atr_p)
                        rs = evaluate(gs, G, ft, fr, {"2024~": periods["2024~"]}, (1.0,))
                        pos += (rs["2024~ 1x"].get("total") or -1) > 0
                    row["seeds_pos"] = pos
                rep[key] = row
                keep[key] = g
                print(key, row["groups"], {p: (row[f"{p} 1x"].get("cagr"), row[f"{p} 1x"].get("mdd"))
                                           for p in ("2023", "2024~")}, round(time.time() - t0), flush=True)
    # 판정
    passed = []
    for r in RULES:
        base = rep[f"{r}|{cell(0.015, '1.5')}"]
        for tp in TPS:
            for sl in SLS:
                if (tp, sl) == (0.015, "1.5"):
                    continue
                key = f"{r}|{cell(tp, sl)}"
                row = rep[key]
                c = crit_basic(row, base)
                c["5"] = c5_of({f"{p} 1x": row["c15"][p] for p in ("2023", "2024~")})
                c["4"] = (row["seeds_pos"] >= 3) if RULES[r][2] else None
                c["pass"] = bool(c["1"] and c["2"] and c["3"] and c["5"] and (c["4"] is not False))
                c["basic"] = bool(c["1"] and c["2"] and c["3"] and c["5"])
                rep[f"crit|{key}"] = c
                if c["pass"]:
                    passed.append((r, tp, sl))
    stable = [(r, tp, sl) for r, tp, sl in passed if any((r, a, b) in passed for a, b in neighbors(tp, sl))]
    rep["passed"] = [f"{r}|{cell(tp, sl)}" for r, tp, sl in passed]
    rep["stable"] = [f"{r}|{cell(tp, sl)}" for r, tp, sl in stable]
    actual_basic = sum(1 for k, v in rep.items() if k.startswith("crit|") and v["basic"])
    rep["actual_basic_count"] = actual_basic
    print("passed", rep["passed"], "stable", rep["stable"], "basic", actual_basic, flush=True)
    # 우연 대조: 신호 시각 원형 이동(2023~ 안), 기준 1·2·3·5 통과 칸 수
    rng = np.random.default_rng(7)
    span = G.n - i0
    shifts = rng.integers(96 * 7, span - 96 * 7, size=a.null)
    counts = []
    for k in shifts:
        cnt = 0
        for r in RULES:
            Lm, Sm = mains[r]
            Lr, Sr = Lm.copy(), Sm.copy()
            Lr[i0:], Sr[i0:] = np.roll(Lm[i0:], int(k)), np.roll(Sm[i0:], int(k))
            rows = {}
            for tp in TPS:
                for sl in SLS:
                    g = run_ml(G, Lr, Sr, i0, model, tp, sl, atr_p)
                    rows[(tp, sl)] = evaluate(g, G, ft, fr, {p: periods[p] for p in ("2023", "2024~")}, (1.0, 3.0))
            base = rows[(0.015, "1.5")]
            for (tp, sl), row in rows.items():
                if (tp, sl) == (0.015, "1.5"):
                    continue
                c = crit_basic(row, base)
                if c["1"] and c["2"] and c["3"]:
                    g15 = run_ml(G, Lr, Sr, i0, model15, tp, sl, atr_p)
                    r15 = evaluate(g15, G, ft, fr, {p: periods[p] for p in ("2023", "2024~")}, (1.0,))
                    cnt += c5_of(r15)
        counts.append(int(cnt))
        print("null", int(k), cnt, round(time.time() - t0), flush=True)
    rep["null_basic_counts"] = counts
    rep["null_p"] = round(float(np.mean([x >= actual_basic for x in counts])), 3) if counts else None
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (out / "groups.json").write_text(json.dumps(
        {k: [[int(e), int(s), float(segs[0][1]), int(segs[0][3]), float(segs[0][4])] for e, s, segs, _ in g]
         for k, g in keep.items()}))


if __name__ == "__main__":
    main()
