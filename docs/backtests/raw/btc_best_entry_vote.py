"""133번: 가장 좋은 진입 찾기 + 다수결 청산. 등록: 2026-11-07-btc-best-entry-vote-exit-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_best_entry_vote.py --out <폴더> \
    [--null 200]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import btc_ml_edge_cost as C
import numpy as np
from btc_fractal import load5, resample, scr3
from btc_futures_compare import load_funding
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, lim_fill, run_k, run_ml, run_r1, evaluate
from btc_scr_week_opt import gate_dir, signals
from btc_strategy_combo import RES, ts

DAY = 86_400_000
M5 = 300_000
BAR = 900_000
R15 = 0.015
HOR = 192  # 2일
MAXH = 672  # 7일
LIMIT_FAMILY = ("A", "B", "AS", "BS", "AS4", "K2")
ML_FAMILY = LIMIT_FAMILY


def measure(A, e, s, px):
    """체결 봉 다음 봉부터 2일 안에 +1.5%가 −1.5%보다 먼저면 1. 같은 봉이면 손절."""
    h, l = A["h"][e + 1:e + 1 + HOR], A["l"][e + 1:e + 1 + HOR]  # noqa: E741
    if s > 0:
        win, loss = h >= px * (1 + R15), l <= px * (1 - R15)
    else:
        win, loss = l <= px * (1 - R15), h >= px * (1 + R15)
    iw = int(np.argmax(win)) if win.any() else 10**9
    il = int(np.argmax(loss)) if loss.any() else 10**9
    return 1.0 if iw < il else 0.0


def fwd(A, e, s, px):
    c, h, l = A["c"], A["h"], A["l"]  # noqa: E741
    n = len(c)
    r4 = (c[e + 16] / px - 1) * s if e + 16 < n else np.nan
    r24 = (c[e + 96] / px - 1) * s if e + 96 < n else np.nan
    hh, ll = h[e + 1:e + 1 + HOR], l[e + 1:e + 1 + HOR]
    mfe = ((hh.max() / px - 1) if s > 0 else (1 - ll.min() / px)) if len(hh) else np.nan
    mae = ((1 - ll.min() / px) if s > 0 else (hh.max() / px - 1)) if len(hh) else np.nan
    return r4, r24, mfe, mae


def entries(G, sig, kind, i0, model):
    """sig: +1/−1/0 배열. kind 'lim' 또는 'mkt'. 진입 하나 뒤 2일 측정 동안 다음 신호는 보지 않는다."""
    out, i = [], i0
    n = G.n
    while i < n - HOR - 2:
        s = int(sig[i])
        if not s:
            i += 1
            continue
        if kind == "lim":
            f = lim_fill(G, i, s, model)
            if f is None:
                i += 1
                continue
            e, px = f
        else:
            e = i + 1
            px = G.market_px(e, model) * (1 + s * model.slip)
        out.append((e, s, float(px)))
        i = e + HOR
    return out


def k2_first(reg, L4, S4, i0):
    """R1 방향 구간마다 첫 같은 방향 AS4 신호."""
    sig = np.zeros(len(reg), int)
    used, prev = False, 0
    for i in range(i0, len(reg)):
        d = int(reg[i])
        if d != prev:
            used, prev = False, d
        if d and not used and (L4[i] if d > 0 else S4[i]):
            sig[i] = d
            used = True
    return sig


def on_grid(tp, close_t, L, S):
    """상위 봉 마감 시각을 15분봉(그 봉이 마감되는 15분봉)에 맞춘다."""
    sig = np.zeros(len(tp), int)
    j = np.searchsorted(tp, close_t - BAR)
    ok = (j < len(tp)) & (tp[np.minimum(j, len(tp) - 1)] == close_t - BAR)
    sig[j[ok & L]] = 1
    sig[j[ok & S & ~L]] = -1
    return sig


def positions(groups, n):
    p = np.zeros(n, int)
    for e, s, segs, end in groups:
        p[e:end] = s
    return p


def run_vote(G, sig, kind, i0, model, V, mode, log=None):
    """mode: 'base'(±1.5%), 'vote'. V: 투표자 × 봉 (+1/−1/0). 반환은 evaluate용 묶음."""
    A = G.A
    o, h, l = A["o"], A["h"], A["l"]  # noqa: E741
    n = G.n
    out, i = [], i0
    while i < n - 3:
        s = int(sig[i])
        if not s:
            i += 1
            continue
        if kind == "lim":
            f = lim_fill(G, i, s, model)
            if f is None:
                i += 1
                continue
            e, px = f
            ef = model.maker
            dfirst = model.delay and e == i + 1
        else:
            e = i + 1
            px = G.market_px(e, model) * (1 + s * model.slip)
            ef = model.taker
            dfirst = model.delay
        stop = px * (1 - s * R15)
        k = 1
        pend = None
        holding = False
        x = None
        cps = []
        for b in range(e, n - 1):
            first = b == e
            hi, lo = G.first_hl(b, model) if (first and dfirst) else (h[b], l[b])
            if (lo <= stop) if s > 0 else (hi >= stop):
                fp = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
                x = (b, fp * (1 - s * model.slip), model.taker, "stop")
                break
            if not first:
                lv = px * (1 + s * R15 * k)
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
                    pend = px * (1 + s * (R15 * k - 0.010))
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
                stop = pend
                pend = None
        if x is None:
            break
        xb, xp, xf, why = x
        out.append((e, s, [(e, float(px), 1.0, xb, float(xp), ef, xf)], xb))
        if log is not None:
            log.append({"e": int(e), "s": s, "px": float(px), "xb": int(xb), "xp": float(xp), "why": why,
                        "cps": [[int(a), b_, c_, d_, bool(f_)] for a, b_, c_, d_, f_ in cps]})
        i = xb
    return out


def per_trade(groups, t, p0, p1):
    v = [s * (segs[0][4] / segs[0][1] - 1) - segs[0][5] - segs[0][6] for e, s, segs, _ in groups if p0 <= t[e] < p1]
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
    spot = {k: z[k] for k in "ohlc"}
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    pm = np.mean(ps, 0)
    P, ok = futures_on(tp, spot)
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

    def ml_masks(probs_a, probs_mean):
        A_ = rule_masks(probs_a, tp, 0.90, 4)
        B_ = rule_masks(probs_a, tp, 0.95, 5)
        AS_ = rule_masks(probs_mean, tp, 0.90, 4)
        BS_ = rule_masks(probs_mean, tp, 0.95, 5)
        AS4_ = (AS_[0] & (tr4 > 0), AS_[1] & (tr4 < 0))
        return {"A": A_, "B": B_, "AS": AS_, "BS": BS_, "AS4": AS4_}

    def to_sig(m):
        return np.where(m[0], 1, np.where(m[1], -1, 0))

    masks = ml_masks(ps[0], pm)
    sigs = {k: to_sig(v) for k, v in masks.items()}
    sigs["K2"] = k2_first(reg, masks["AS4"][0], masks["AS4"][1], i0)
    sigs["R1"] = np.where(newday & (reg != 0), reg, 0).astype(int)
    on_l, on_s, *_ = scr3(P)
    sigs["SCR3_15m"] = np.where(on_l, 1, np.where(on_s, -1, 0))
    u30 = resample(f5k, 6)
    c30 = u30["t"] + 6 * M5
    l30, s30, *_ = scr3(u30)
    sigs["SCR3_30m"] = on_grid(tp, c30, l30, s30)
    lw, sw, *_ = signals(u30, "REL")
    sigs["S1W"] = on_grid(tp, c30, lw, sw)
    kinds = {k: ("lim" if k in LIMIT_FAMILY else "mkt") for k in sigs}

    # 1단계
    rng = np.random.default_rng(7)
    month = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).strftime("%Y%m") for x in tp])
    valid = np.arange(i0, n - HOR - 2)
    by_month = {mm: valid[month[valid] == mm] for mm in dict.fromkeys(month[valid])}
    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(),
                 "futures_missing_2023~": int((~ok[tp >= ts(2023)]).sum())}
    stage1 = {}
    for name, sg in sigs.items():
        ents = entries(G, sg, kinds[name], i0, model)
        row = {}
        for pn, (p0, p1) in periods.items():
            es = [x for x in ents if p0 <= tp[x[0]] < p1]
            if not es:
                row[pn] = {"n": 0}
                continue
            hit = np.array([measure(P, e, s, px) for e, s, px in es])
            rnd = []
            for e, s, px in es:
                pool = by_month.get(month[e])
                if pool is None or not len(pool):
                    continue
                for r_ in rng.choice(pool, 20):
                    rnd.append(measure(P, int(r_), s, float(P["o"][int(r_)])))
            fw = np.array([fwd(P, e, s, px) for e, s, px in es])
            sd = np.array([s for _, s, _ in es])
            row[pn] = {"n": len(es), "n_long": int((sd > 0).sum()), "P": round(float(hit.mean()) * 100, 1),
                       "P_rand": round(float(np.mean(rnd)) * 100, 1),
                       "edge": round(float(hit.mean() - np.mean(rnd)) * 100, 1),
                       "r4h": round(float(np.nanmean(fw[:, 0])) * 100, 3),
                       "r24h": round(float(np.nanmean(fw[:, 1])) * 100, 3),
                       "mfe_med": round(float(np.nanmedian(fw[:, 2])) * 100, 2),
                       "mae_med": round(float(np.nanmedian(fw[:, 3])) * 100, 2)}
        stage1[name] = row
        print("stage1", name, row, flush=True)
    rep["stage1"] = stage1
    elig = [k for k, v in stage1.items() if v["2023"].get("edge", -99) > 0 and v["2024~"].get("n", 0) >= 100]
    ranking = sorted(stage1, key=lambda k: (stage1[k]["2024~"].get("edge", -99), stage1[k]["2023"].get("edge", -99)),
                     reverse=True)
    rep["ranking"] = ranking
    rep["eligible"] = elig
    if not elig:
        rep["winner"] = None
        (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
        print("no eligible winner", flush=True)
        return
    win = max(elig, key=lambda k: (stage1[k]["2024~"]["edge"], stage1[k]["2023"]["edge"]))
    rep["winner"] = win
    print("winner", win, flush=True)

    # 2단계: 투표자
    vot = {"R1dir": reg.astype(int), "4h": gate_dir(f5k, P, 3, 48), "1h": gate_dir(f5k, P, 3, 12)}
    sc = C.mean_score(pm)
    with np.errstate(invalid="ignore"):
        vot["MLside"] = np.where(sc > 0.5, 1, np.where(sc < 0.5, -1, 0)).astype(int)
    for k in ("A", "B", "AS", "BS", "AS4"):
        vot[f"pos_{k}"] = positions(run_ml(G, masks[k][0], masks[k][1], i0, model), n)
    vot["pos_K2"] = positions(run_k(G, reg, masks["AS4"][0], masks["AS4"][1], i0, model, False), n)
    vot["pos_R1"] = positions(run_r1(G, reg, newday, i0, model), n)
    own = {"R1": ["pos_R1"], "K2": ["pos_K2"], "A": ["pos_A"], "B": ["pos_B"], "AS": ["pos_AS"],
           "BS": ["pos_BS"], "AS4": ["pos_AS4"]}.get(win, [])
    v_all = [k for k in vot if k not in own]
    v_ex = [k for k in v_all if k not in ("pos_A", "pos_B", "pos_AS", "pos_BS", "pos_AS4")]
    rep["voters"] = {"V": v_all, "V-ex": v_ex}
    Vall = np.vstack([vot[k] for k in v_all])
    Vex = np.vstack([vot[k] for k in v_ex])
    sg, kd = sigs[win], kinds[win]
    res2, keep, logs = {}, {}, {}
    for var, mode, VV in (("base", "base", Vall), ("V", "vote", Vall), ("V-ex", "vote", Vex)):
        for mk, mdl in (("bc", model), ("c15", m15)):
            lg = [] if mk == "bc" else None
            g = run_vote(G, sg, kd, i0, mdl, VV, mode, lg)
            row = evaluate(g, G, ft, fr, periods, (1.0, 3.0) if mk == "bc" else (1.0,))
            s_ = row.pop("_series")
            res2[f"{var}|{mk}"] = row
            if mk == "bc":
                keep[var] = (g, s_)
                logs[var] = lg
                why = {}
                for x in lg:
                    why[x["why"]] = why.get(x["why"], 0) + 1
                row["exit_reasons"] = why
                row["holds"] = sum(1 for x in lg for c_ in x["cps"] if c_[4])
            print("stage2", var, mk, {p: row[f"{p} 1x"] for p in periods},
                  {p: row[f"{p} trades"] for p in periods}, flush=True)
    rep["stage2"] = res2
    # 씨드
    seeds = {}
    if win in ML_FAMILY:
        for sk in range(5):
            mm = ml_masks(ps[sk], ps[sk])
            if win == "K2":
                ss = k2_first(reg, mm["AS4"][0], mm["AS4"][1], i0)
            else:
                ss = to_sig(mm[win])
            r_ = {}
            for var, mode, VV in (("base", "base", Vall), ("V", "vote", Vall), ("V-ex", "vote", Vex)):
                g = run_vote(G, ss, kd, i0, model, VV, mode)
                r_[var] = per_trade(g, tp, ts(2024), end + 1)
            seeds[f"s{sk}"] = r_
            print("seed", sk, r_, flush=True)
    rep["seeds"] = seeds
    # 무작위 대조: 투표자 시계열을 2023~ 안에서 원형 이동
    span = n - i0
    shifts = rng.integers(96 * 7, span - 96 * 7, size=a.null)
    null = {}
    for var, VV in (("V", Vall), ("V-ex", Vex)):
        act = per_trade(keep[var][0], tp, ts(2024), end + 1)
        vals = []
        for k_ in shifts:
            Vr = VV.copy()
            Vr[:, i0:] = np.roll(VV[:, i0:], int(k_), axis=1)
            g = run_vote(G, sg, kd, i0, model, Vr, "vote")
            vals.append(per_trade(g, tp, ts(2024), end + 1))
        vals = np.array(vals)
        null[var] = {"actual_pct": round(act * 100, 4), "null_mean_pct": round(float(np.nanmean(vals)) * 100, 4),
                     "p": round(float((vals >= act).mean()), 4)}
        print("null", var, null[var], flush=True)
    rep["null"] = null
    # 판정
    crit = {}
    b = res2["base|bc"]
    for var in ("V", "V-ex"):
        v = res2[f"{var}|bc"]
        c1 = all((v[f"{p} trades"].get("avg_pct") or -9) > (b[f"{p} trades"].get("avg_pct") or -9) and
                 (v[f"{p} 1x"].get("ratio") or -9) > (b[f"{p} 1x"].get("ratio") or -9) for p in periods)
        tr = v["2024~ trades"]
        c2 = (tr.get("long_pct") if tr.get("long_pct") is not None else 0) >= 0 and \
             (tr.get("short_pct") if tr.get("short_pct") is not None else 0) >= 0
        cc = res2[f"{var}|c15"]
        c3 = all((cc[f"{p} 1x"].get("cagr") or -1) > 0 for p in periods)
        if seeds:
            pos = sum(1 for r_ in seeds.values() if r_[var] > r_["base"])
            c4 = pos >= 3
        else:
            pos, c4 = None, True
        c5 = null[var]["p"] < 0.025
        crit[var] = {"1": bool(c1), "2": bool(c2), "3": bool(c3), "4": bool(c4), "seeds_better": pos, "5": bool(c5),
                     "pass": bool(c1 and c2 and c3 and c4 and c5)}
        print("crit", var, crit[var], flush=True)
    rep["crit"] = crit
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    np.savez(out / "series.npz", **{f"{k}__dt": v[1][0] for k, v in keep.items()},
             **{f"{k}__dr": v[1][1] for k, v in keep.items()})
    (out / "logs.json").write_text(json.dumps(logs))


if __name__ == "__main__":
    main()
