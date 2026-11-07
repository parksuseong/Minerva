"""134번: K2 + BS 진입, 다수결 청산(손절 완화). 등록: 2026-11-07-btc-k2bs-vote-exit-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_k2bs_vote.py --out <폴더> \
    [--null 200]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import btc_fut_daily_regime as R
import btc_ml_edge_cost as C
from btc_best_entry_vote import k2_first, positions
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, evaluate, lim_fill, r1_manage, run_k, run_r1
from btc_scr_week_opt import gate_dir
from btc_strategy_combo import RES, ts

BAR = 900_000
R15 = 0.015
MAXH = 672
# 변형: (모드, 검문 간격, 들고 갈 때 검문가에서 물러나는 폭, K2를 R1식으로)
VARIANTS = {
    "X0": ("base", 0.015, None, False),
    "X1": ("vote", 0.015, 0.005, False),
    "X2": ("vote", 0.020, 0.015, False),
    "X3": ("vote", 0.015, 0.005, True),
}


def run(G, sigK, sigB, i0, model, V, var, reg, log=None):
    mode, step, back, k2r1 = VARIANTS[var]
    A = G.A
    o, h, l = A["o"], A["h"], A["l"]  # noqa: E741
    n = G.n
    out, i = [], i0
    while i < n - 3:
        if sigK[i]:
            s, src = int(sigK[i]), "K2"
        elif sigB[i]:
            s, src = int(sigB[i]), "BS"
        else:
            i += 1
            continue
        f = lim_fill(G, i, s, model)
        if f is None:
            i += 1
            continue
        e, px = f
        ef = model.maker
        dfirst = model.delay and e == i + 1
        cps = []
        if k2r1 and src == "K2":
            r = r1_manage(G, e, s, px, reg, model, dfirst, model.taker, model.slip)
            if r is None:
                break
            xb, xp, xf, stopped = r
            x = (xb, xp, xf, "r1stop" if stopped else "r1flip")
        else:
            stop = px * (1 - s * R15)
            k, pend, holding, x = 1, None, False, None
            for b in range(e, n - 1):
                first = b == e
                hi, lo = G.first_hl(b, model) if (first and dfirst) else (h[b], l[b])
                if (lo <= stop) if s > 0 else (hi >= stop):
                    fp = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
                    x = (b, fp * (1 - s * model.slip), model.taker, "stop")
                    break
                if not first:
                    lv = px * (1 + s * step * k)
                    if (hi >= lv * (1 + model.thru)) if s > 0 else (lo <= lv * (1 - model.thru)):
                        if mode == "vote":
                            vv = V[:, b - 1]
                            nz = int((vv != 0).sum())
                            ours = int((vv == s).sum())
                            hold = nz > 0 and ours * 2 > nz
                        else:
                            nz = ours = 0
                            hold = False
                        cps.append((b, float(lv), ours, nz, bool(hold)))
                        if not hold:
                            x = (b, lv, model.maker, "target")
                            break
                        pend = px * (1 + s * (step * k - back))
                        k += 1
                        holding = True
                if mode == "vote" and holding:
                    vv = V[:, b]
                    nz = int((vv != 0).sum())
                    if nz > 0 and int((vv == -s).sum()) * 2 > nz:
                        x = (b + 1, o[b + 1] * (1 - s * model.slip), model.taker, "flip")
                        break
                    if b - e >= MAXH:
                        x = (b + 1, o[b + 1] * (1 - s * model.slip), model.taker, "maxhold")
                        break
                if pend is not None:
                    stop = pend  # 옮긴 손절은 다음 봉부터
                    pend = None
            if x is None:
                break
        xb, xp, xf, why = x
        out.append((e, s, [(e, float(px), 1.0, xb, float(xp), ef, xf)], xb))
        if log is not None:
            log.append({"e": int(e), "s": s, "src": src, "px": float(px), "xb": int(xb), "xp": float(xp),
                        "why": why, "cps": [[int(a), b_, c_, d_, bool(f_)] for a, b_, c_, d_, f_ in cps]})
        i = xb
    return out


def per_trade(groups, t, p0, p1):
    v = [s * (g[0][4] / g[0][1] - 1) - g[0][5] - g[0][6] for e, s, g, _ in groups if p0 <= t[e] < p1]
    return float(np.mean(v)) if v else float("nan")


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
    tp = z["t"]
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    pm = np.mean(ps, 0)
    P, ok = futures_on(tp, {k: z[k] for k in "ohlc"})
    P["t"] = tp
    end = float(tp[-1])
    f5k = {k: v[f5["t"] <= end + BAR] for k, v in f5.items()}
    reg, newday = R.regime(f5k, P, 0)
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp)
    G = Grid(P, f5k)
    n = G.n
    i0 = int(np.searchsorted(tp, ts(2023)))
    model = Model(True, 0.0005)
    m15 = Model(True, 0.0005, 1.5)
    periods = {"2023": (ts(2023), ts(2024)), "2024~": (ts(2024), end + 1)}

    def sigs_of(probs):
        AS_ = rule_masks(probs, tp, 0.90, 4)
        BS_ = rule_masks(probs, tp, 0.95, 5)
        L4, S4 = AS_[0] & (tr4 > 0), AS_[1] & (tr4 < 0)
        sk = k2_first(reg, L4, S4, i0)
        sb = np.where(BS_[0], 1, np.where(BS_[1], -1, 0)).astype(int)
        return sk, sb, (L4, S4)

    sigK, sigB, as4 = sigs_of(pm)
    zero = np.zeros(n, int)
    # 투표자: 133번 V-ex와 같다(R1 방향, 4시간, 1시간, ML 방향, K2·R1 포지션)
    vot = {"R1dir": reg.astype(int), "4h": gate_dir(f5k, P, 3, 48), "1h": gate_dir(f5k, P, 3, 12)}
    sc = C.mean_score(pm)
    with np.errstate(invalid="ignore"):
        vot["MLside"] = np.where(sc > 0.5, 1, np.where(sc < 0.5, -1, 0)).astype(int)
    vot["pos_K2"] = positions(run_k(G, reg, as4[0], as4[1], i0, model, False), n)
    vot["pos_R1"] = positions(run_r1(G, reg, newday, i0, model), n)
    V = np.vstack([vot[k] for k in vot])
    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(), "voters": list(vot),
                 "n_sigK_2023~": int((sigK[i0:] != 0).sum()), "n_sigB_2023~": int((sigB[i0:] != 0).sum())}
    res, keep, logs = {}, {}, {}
    rows = [(f"E-{v}", sigK, sigB, v) for v in VARIANTS] + \
           [("K2-X0", sigK, zero, "X0"), ("K2-X1", sigK, zero, "X1"), ("BS-X0", zero, sigB, "X0"),
            ("BS-X1", zero, sigB, "X1")]
    for name, sk_, sb_, var in rows:
        for mk, mdl in (("bc", model), ("c15", m15)):
            lg = [] if mk == "bc" else None
            g = run(G, sk_, sb_, i0, mdl, V, var, reg, lg)
            row = evaluate(g, G, ft, fr, periods, (1.0, 3.0) if mk == "bc" else (1.0,))
            s_ = row.pop("_series")
            res[f"{name}|{mk}"] = row
            if mk == "bc":
                keep[name] = (g, s_)
                logs[name] = lg
                why, src = {}, {}
                for x in lg:
                    why[x["why"]] = why.get(x["why"], 0) + 1
                    src[x["src"]] = src.get(x["src"], 0) + 1
                row["exit_reasons"] = why
                row["sources"] = src
                row["holds"] = sum(1 for x in lg for c_ in x["cps"] if c_[4])
            print(name, mk, {p: row[f"{p} 1x"] for p in periods}, {p: row[f"{p} trades"] for p in periods},
                  flush=True)
    rep["res"] = res
    seeds = {}
    for sk in range(5):
        k_, b_, _ = sigs_of(ps[sk])
        seeds[f"s{sk}"] = {v: per_trade(run(G, k_, b_, i0, model, V, v, reg), tp, ts(2024), end + 1)
                           for v in VARIANTS}
        print("seed", sk, seeds[f"s{sk}"], flush=True)
    rep["seeds"] = seeds
    rng = np.random.default_rng(7)
    shifts = rng.integers(96 * 7, n - i0 - 96 * 7, size=a.null)
    null = {}
    for var in ("X1", "X2", "X3"):
        act = per_trade(keep[f"E-{var}"][0], tp, ts(2024), end + 1)
        vals = []
        for k_ in shifts:
            Vr = V.copy()
            Vr[:, i0:] = np.roll(V[:, i0:], int(k_), axis=1)
            vals.append(per_trade(run(G, sigK, sigB, i0, model, Vr, var, reg), tp, ts(2024), end + 1))
        vals = np.array(vals)
        null[var] = {"actual_pct": round(act * 100, 4), "null_mean_pct": round(float(np.nanmean(vals)) * 100, 4),
                     "p": round(float((vals >= act).mean()), 4)}
        print("null", var, null[var], flush=True)
    rep["null"] = null
    crit = {}
    b = res["E-X0|bc"]
    for var in ("X1", "X2", "X3"):
        v = res[f"E-{var}|bc"]
        c1 = all((v[f"{p} trades"].get("avg_pct") or -9) > (b[f"{p} trades"].get("avg_pct") or -9) and
                 (v[f"{p} 1x"].get("ratio") or -9) > (b[f"{p} 1x"].get("ratio") or -9) for p in periods)
        tr = v["2024~ trades"]
        c2 = (tr.get("long_pct") or 0) >= 0 and (tr.get("short_pct") or 0) >= 0
        cc = res[f"E-{var}|c15"]
        c3 = all((cc[f"{p} 1x"].get("cagr") or -1) > 0 for p in periods)
        pos = sum(1 for r_ in seeds.values() if r_[var] > r_["X0"])
        c4 = pos >= 3
        c5 = null[var]["p"] < 0.0167
        crit[var] = {"1": bool(c1), "2": bool(c2), "3": bool(c3), "4": bool(c4), "seeds_better": pos,
                     "5": bool(c5), "pass": bool(c1 and c2 and c3 and c4 and c5)}
        print("crit", var, crit[var], flush=True)
    rep["crit"] = crit
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    np.savez(out / "series.npz", **{f"{k}__dt": v[1][0] for k, v in keep.items()},
             **{f"{k}__dr": v[1][1] for k, v in keep.items()})
    (out / "logs.json").write_text(json.dumps(logs))


if __name__ == "__main__":
    main()
