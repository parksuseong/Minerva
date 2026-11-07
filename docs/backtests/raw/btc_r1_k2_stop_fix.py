"""129번: R1·K2 손절·청산 변형(ATR 손절, 본전·추적, ADX 문). 등록: 2026-11-07-btc-r1-k2-stop-fix-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_r1_k2_stop_fix.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import btc_ml_edge_cost as C
import numpy as np
from btc_fractal import load5, resample
from btc_futures_compare import load_funding
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, evaluate, lim_fill
from btc_strategy_combo import RES, ts

DAY = 86_400_000
M5 = 300_000
H = 3_600_000
BASE_R = 0.01
CAP = 3.0
VARS = {
    "base": dict(atr=0.0, trail=False, adx=False),
    "V1": dict(atr=1.0, trail=False, adx=False),
    "V2": dict(atr=0.5, trail=False, adx=False),
    "V3": dict(atr=0.0, trail=True, adx=False),
    "V4": dict(atr=0.0, trail=False, adx=True),
    "V5": dict(atr=1.0, trail=True, adx=True),
}


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    for k in range(n, len(x)):
        out[k] = out[k - 1] + (x[k] - out[k - 1]) / n
    return out


def daily_atr_adx(f5: dict, A: dict, shift_h: int) -> tuple:
    """15분봉 마감 때 이미 마감된 일봉의 ATR14(가격 대비 비율)와 ADX14."""
    sh = dict(f5)
    sh["t"] = f5["t"] - shift_h * H
    u = resample(sh, 288)
    h, l, c = u["h"], u["l"], u["c"]  # noqa: E741
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    atr = wilder(tr, 14)
    up, dn = h - np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]] - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = mdm[0] = 0.0
    tr14, p14, m14 = wilder(tr, 14), wilder(pdm, 14), wilder(mdm, 14)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi, mdi = 100 * p14 / tr14, 100 * m14 / tr14
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = np.full(len(dx), np.nan)
    ok = np.flatnonzero(~np.isnan(dx))
    if len(ok) >= 14:
        adx[ok[0]:] = wilder(dx[ok[0]:], 14)
    close = u["t"] + shift_h * H + 288 * M5
    j = np.searchsorted(close, A["t"] + 900_000, side="right") - 1
    atr_p, adx_p = np.full(len(A["t"]), np.nan), np.full(len(A["t"]), np.nan)
    m = j >= 0
    atr_p[m] = (atr / c)[j[m]]
    adx_p[m] = adx[j[m]]
    return atr_p, adx_p


def manage(G: Grid, e: int, s: int, px: float, reg, model: Model, delayed_first: bool, fee_exit: float,
           slip: float, sd: float, trail: bool):
    """손절 폭 sd(비율). trail이면 +1R 본전, +2R부터 최대 유리 폭 − 1R. 옮긴 손절은 다음 봉부터."""
    A = G.A
    o, h, l = A["o"], A["h"], A["l"]  # noqa: E741
    stop = px * (1 - s * sd)
    r = px * sd
    best = px
    for b in range(e, G.n - 1):
        first = b == e
        hi, lo = G.first_hl(b, model) if (first and delayed_first) else (h[b], l[b])
        if (lo <= stop) if s > 0 else (hi >= stop):
            f = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
            return b, f * (1 - s * slip), fee_exit, True
        if reg[b] != s:
            return b, o[b + 1] * (1 - s * slip), fee_exit, False
        if trail:
            best = max(best, hi) if s > 0 else min(best, lo)
            mfe = (best - px) * s
            if mfe >= r:
                stop = max(stop, px) if s > 0 else min(stop, px)
            if mfe >= 2 * r:
                ns = px + s * (mfe - r)
                stop = max(stop, ns) if s > 0 else min(stop, ns)
    return None


def stop_dist(cfg: dict, atr_p, i: int) -> float | None:
    if cfg["atr"] <= 0:
        return BASE_R
    a = atr_p[i]
    if np.isnan(a) or a <= 0:
        return None
    return float(cfg["atr"] * a)


def weight(sd: float, sizing: str) -> float:
    return 1.0 if sizing == "1x" else min(CAP, BASE_R / sd)


def seg(e, s, px, w, x, ef):
    b, xp, xf, stopped = x
    return (e, s, [(e, px, w, b, float(xp), ef, xf)], b), stopped


def run_r1(G, reg, newday, atr_p, adx_p, i0, model, cfg, sizing):
    out, i, can = [], i0, True
    while i < G.n - 2:
        if newday[i]:
            can = True
        s = int(reg[i])
        if s and can:
            sd = stop_dist(cfg, atr_p, i)
            blocked = cfg["adx"] and not (adx_p[i] >= 20)
            if sd is None or blocked:
                can = False  # 이 일봉에서는 들어가지 않는다. 다음 일봉 마감에 다시 본다
                i += 1
                continue
            e = i + 1
            px = G.market_px(e, model)
            x = manage(G, e, s, px, reg, model, model.delay, model.r1_fee, 0.0, sd, cfg["trail"])
            if x is None:
                break
            g, stopped = seg(e, s, px, weight(sd, sizing), x, model.r1_fee)
            out.append(g)
            if stopped:
                can = False
            i = x[0]
            continue
        i += 1
    return out


def run_k2(G, reg, Lm, Sm, atr_p, adx_p, i0, model, cfg, sizing):
    out, i = [], i0
    while i < G.n - 2:
        d = int(reg[i])
        if d == 0 or not (Lm[i] if d > 0 else Sm[i]):
            i += 1
            continue
        sd = stop_dist(cfg, atr_p, i)
        if sd is None or (cfg["adx"] and not (adx_p[i] >= 20)):
            i += 1
            continue
        f = lim_fill(G, i, d, model)
        if f is None:
            i += 1
            continue
        e, px = f
        x = manage(G, e, d, px, reg, model, model.delay and e == i + 1, model.taker, model.slip, sd, cfg["trail"])
        if x is None:
            break
        g, _ = seg(e, d, px, weight(sd, sizing), x, model.maker)
        out.append(g)
        i = x[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    ft, fr = load_funding()
    z = np.load(RES / "ml15/pred.npz")
    tp = z["t"]
    spot = {k: z[k] for k in "ohlc"}
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    P, ok = futures_on(tp, spot)
    P["t"] = tp
    end = float(tp[-1])
    f5k = {k: v[f5["t"] <= end + 900_000] for k, v in f5.items()}
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp)
    G = Grid(P, f5k)
    model = Model(True, 0.0005)
    i_r1 = int(np.searchsorted(tp, ts(2020, 3, 1)))
    i_k2 = int(np.searchsorted(tp, ts(2023)))
    lm = ts(2026, 10, 7) - 9 * H
    per_r1 = {"2020-23": (ts(2020, 3, 1), ts(2024)), "2024~": (ts(2024), end + 1), "lastmonth": (lm, end + 1)}
    per_k2 = {"2023": (ts(2023), ts(2024)), "2024~": (ts(2024), end + 1), "lastmonth": (lm, end + 1)}

    def as4(probs):
        li, si = rule_masks(probs, tp, 0.90, 4)
        return li & (tr4 > 0), si & (tr4 < 0)

    masks = {"mean": as4(np.mean(ps, 0)), **{f"s{k}": as4(ps[k]) for k in range(5)}}
    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(),
                 "fut_missing_2020-03~": int((~ok[tp >= ts(2020, 3, 1)]).sum())}
    keep, series = {}, {}
    for shift in (0, 8, 16):
        reg, newday = R.regime(f5k, P, shift)
        atr_p, adx_p = daily_atr_adx(f5k, P, shift)
        for vn, cfg in VARS.items():
            for sizing in (("eq", "1x") if shift == 0 else ("eq",)):
                g = run_r1(G, reg, newday, atr_p, adx_p, i_r1, model, cfg, sizing)
                row = evaluate(g, G, ft, fr, per_r1, (1.0,))
                s_ = row.pop("_series")
                key = f"R1|{vn}|{sizing}|sh{shift}"
                rep[key] = row
                if shift == 0 and sizing == "eq":
                    keep[key], series[key] = g, s_
                print(key, row["groups"], {p: (row[f"{p} 1x"].get("cagr"), row[f"{p} 1x"].get("mdd"))
                                           for p in per_r1}, flush=True)
                seeds = ["mean"] + ([f"s{k}" for k in range(5)] if (shift == 0 and sizing == "eq") else [])
                for sk in seeds:
                    Lm, Sm = masks[sk]
                    g = run_k2(G, reg, Lm, Sm, atr_p, adx_p, i_k2, model, cfg, sizing)
                    row = evaluate(g, G, ft, fr, per_k2, (1.0,))
                    s_ = row.pop("_series")
                    key = f"K2|{vn}|{sizing}|sh{shift}|{sk}"
                    rep[key] = row
                    if shift == 0 and sizing == "eq" and sk == "mean":
                        keep[key], series[key] = g, s_
                    if sk == "mean":
                        print(key, row["groups"], {p: (row[f"{p} 1x"].get("cagr"), row[f"{p} 1x"].get("mdd"))
                                                   for p in per_k2}, flush=True)

    def ratio(k, p):
        v = rep[k][f"{p} 1x"].get("ratio")
        return -99.0 if v is None else v

    def cagr(k, p):
        v = rep[k][f"{p} 1x"].get("cagr")
        return -99.0 if v is None else v

    for rule, pers in (("R1", ("2020-23", "2024~")), ("K2", ("2023", "2024~"))):
        sfx = "|mean" if rule == "K2" else ""
        for vn in VARS:
            if vn == "base":
                continue
            k0, kb = f"{rule}|{vn}|eq|sh0{sfx}", f"{rule}|base|eq|sh0{sfx}"
            c1 = all(cagr(k0, p) > 0 and ratio(k0, p) > ratio(kb, p) for p in pers)
            tr = rep[k0]["2024~ trades"]
            c2 = (tr.get("long_pct") is None or tr["long_pct"] >= 0) and \
                 (tr.get("short_pct") is None or tr["short_pct"] >= 0)
            c3 = all(ratio(f"{rule}|{vn}|eq|sh{s}{sfx}", p) > ratio(f"{rule}|base|eq|sh{s}{sfx}", p)
                     for s in (8, 16) for p in pers)
            crit = {"1": bool(c1), "2": bool(c2), "3": bool(c3)}
            ok_all = c1 and c2 and c3
            if rule == "K2":
                pos = sum(1 for k in range(5)
                          if cagr(f"K2|{vn}|eq|sh0|s{k}", "2024~") > 0 and
                          ratio(f"K2|{vn}|eq|sh0|s{k}", "2024~") > ratio(f"K2|base|eq|sh0|s{k}", "2024~"))
                crit["4"] = pos >= 3
                crit["seeds"] = pos
                ok_all = ok_all and pos >= 3
            crit["pass"] = bool(ok_all)
            rep[f"crit|{rule}|{vn}"] = crit
            print("crit", rule, vn, crit, flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    np.savez(out / "series.npz", **{f"{k}__dt": v[0] for k, v in series.items()},
             **{f"{k}__dr": v[1] for k, v in series.items()})
    (out / "groups.json").write_text(json.dumps(
        {k: [[int(e), int(s), float(segs[0][1]), int(segs[0][3]), float(segs[0][4]), float(segs[0][2])]
             for e, s, segs, _ in g] for k, g in keep.items()}))


if __name__ == "__main__":
    main()
