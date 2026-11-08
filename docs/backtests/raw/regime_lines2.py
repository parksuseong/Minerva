"""큰 구간 선 2: 확률 목표가 + 정리 장치. 등록: 2026-11-08-btc-regime-lines-2-hypotheses.md

사건(꺾임점 확정) 때 진행 중 다리 방향으로 들어가고, 닮은 과거 다리들이 "확정 시점부터 더 간 거리" 분포로 목표가를 정한다.
1시간마다 아직 안 닿고 살아 있는 이웃 중 나중에 닿은 비율 p(t)를 다시 재서 0.30 밑이면 정리한다. 판정은 실제 15분 체결 손익.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/regime-lines2-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/regime_lines2.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import daily_wave as DW
import numpy as np
import regime_lines as RL
from pine_eval import lev_path

M15, H1, H4, DAY = 900_000, 3_600_000, 14_400_000, 86_400_000
SEGS = ("Z5", "Z8", "L1.5", "L3")
TPS = ("Q30", "Q50", "Q70", "EV")
QGRID = np.round(np.arange(0.1, 0.91, 0.1), 2)
K_NB, MIN_LIB, KLEG = 20, 10, 3
BAIL = 0.30
DISASTER = 0.06
FEE_RT = 0.001
N_VAR = 32
N_TONIGHT_BEFORE = 5899
MAIN = ("fresh", "design", "val", "year")


# ---------------- 사건 통계 (확정 시점부터) ----------------
def ev_stats(s: dict, name: str) -> dict:
    E = RL.events(s, name)
    x, lh, ll = s["x"], np.log(s["h"]), np.log(s["l"])
    P, C = E["P"], E["C"]
    n = len(P)
    d = np.sign(x[C] - x[P]).astype(int)
    mh = np.abs(E["m"])
    end = np.full(n, -1)  # 방향이 바뀌는 다음 사건(구조 정리 시점)
    for e in range(n):
        m_ = e + 1
        while m_ < n and d[m_] == d[e]:
            m_ += 1
        end[e] = m_ if m_ < n else -1
    fav, ext, v, dur = [None] * n, np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for e in range(n):
        if end[e] < 0 or d[e] == 0:
            continue
        a, z = C[e] + 1, C[end[e]]
        if z < a:
            continue
        f = (lh[a : z + 1] - x[C[e]]) if d[e] == 1 else (x[C[e]] - ll[a : z + 1])
        fav[e] = np.maximum.accumulate(f)
        ext[e] = max(fav[e][-1], 0.0)
        v[e] = d[e] * (x[z] - x[C[e]])
        dur[e] = z - C[e]  # 4시간봉 수
    return {"E": E, "d": d, "mh": mh, "end": end, "fav": fav, "ext": ext, "v": v, "dur": dur}


def reach_time(fav, level) -> float:
    if fav is None:
        return np.inf
    hit = np.nonzero(fav >= level)[0]
    return float(hit[0] + 1) if len(hit) else np.inf


def decisions(s: dict, S: dict, tp: str, cond: str) -> dict:
    """사건마다 진입 여부·목표 비율·이웃 정보. cond: E(닮은 이웃) / U(같은 방향 과거 전체)."""
    E = S["E"]
    n = len(E["P"])
    F_, prog, key = RL.feats(E, s["x"], KLEG)
    ok = np.isfinite(S["ext"]) & (S["mh"] > 0)
    endC = np.where(S["end"] >= 0, E["C"][np.maximum(S["end"], 0)], 10**9)
    out = {}
    for e in range(KLEG + 1, n):
        if S["d"][e] == 0 or not np.isfinite(F_[e, 0, 0]) or S["mh"][e] <= 0:
            continue
        lib = np.nonzero(ok & (endC < E["C"][e]))[0]
        if cond == "E":
            lib = lib[np.isfinite(F_[lib, 0, 0])]
            if len(lib) < MIN_LIB:
                continue
            dd = RL.dist_E(F_, prog, key, e, lib)
            o = np.argsort(dd)[:K_NB]
            nb = lib[o[np.isfinite(dd[o])]]
        else:
            nb = lib[S["d"][lib] == S["d"][e]]
        if len(nb) < MIN_LIB:
            continue
        rj = S["ext"][nb] / S["mh"][nb]
        vj = S["v"][nb] / S["mh"][nb]
        m_now = S["mh"][e]

        def ev(r):
            return float(np.mean(np.where(rj >= r, r, vj)) * m_now - FEE_RT)

        if tp == "EV":
            cands = np.quantile(rj, QGRID)
            evs = [ev(r) for r in cands]
            kbest = int(np.argmax(evs))
            r, val = float(cands[kbest]), evs[kbest]
            take = val > 0.002
        else:
            q = {"Q30": 0.3, "Q50": 0.5, "Q70": 0.7}[tp]
            r = float(np.quantile(rj, q))
            val = ev(r)
            take = val > 0
        tpd = float(np.clip(r * m_now, 0.005, 0.20))
        r_eff = tpd / m_now
        tre = np.array([reach_time(S["fav"][j], r_eff * S["mh"][j]) for j in nb])
        durj = S["dur"][nb]
        out[e] = {"take": bool(take), "ev": val, "tp": tpd, "r": r_eff, "tre": tre, "dur": durj,
                  "tmax_h": float(np.quantile(durj, 0.9) * 4), "p0": float(np.mean(np.isfinite(tre)))}  # fmt: skip
    return out


def p_at(dec: dict, hours: float, prev: float) -> float:
    t4 = hours / 4.0
    alive = (dec["dur"] > t4) & (dec["tre"] > t4)
    if alive.sum() < 5:
        return prev
    return float(np.mean(np.isfinite(dec["tre"][alive])))


# ---------------- 모의 매매 ----------------
def simulate(b, F, s, S, D, a, z, bail=True, fee="FA", delay=0, fee_mult=1.0) -> list:
    o, h, lo, c, t = b["o"], b["h"], b["l"], b["c"], b["t"]
    ft, fr = b["ft"], b["fr"]
    E = S["E"]
    tk, mk = DW.TAKER * fee_mult, DW.MAKER * fee_mult
    trades, busy = [], -1
    for e in sorted(D):
        dec = D[e]
        if not dec["take"]:
            continue
        i = int(s["i15"][E["C"][e]])
        if i < a or i >= z - 2 or i <= busy:
            continue
        d = int(S["d"][e])
        if fee == "FA":
            en = i + 1 + delay
            ep = o[en]
            lim = 0.03 if F["agree"][i] else 0.01
            if abs(ep / c[i] - 1) > lim:
                continue
            fin = tk
        else:
            lp, en = c[i], None
            for j in range(i + 1 + delay, min(i + 5 + delay, z - 1)):
                if (d == 1 and lo[j] < lp - DW.TICK) or (d == -1 and h[j] > lp + DW.TICK):
                    en = j
                    break
            if en is None:
                continue
            ep, fin = lp, mk
        tp = ep * np.exp(d * dec["tp"])
        stop = ep * (1 - d * DISASTER)
        t0 = int(s["t"][E["C"][e]]) + H4  # 사건 확정 시각
        endi = int(s["i15"][E["C"][S["end"][e]]]) if S["end"][e] >= 0 else 10**12
        p, pser, worst = dec["p0"], [(0.0, dec["p0"])], 0.0
        xp = why = xi = None
        for j in range(en, z):
            adv = (lo[j] / ep - 1) if d == 1 else (1 - h[j] / ep)
            worst = min(worst, adv)
            if (lo[j] <= stop) if d == 1 else (h[j] >= stop):
                gap = (o[j] < stop) if d == 1 else (o[j] > stop)
                xp, why, xi = (o[j] if gap and j > en else stop), "stop", j
                break
            if (h[j] >= tp) if d == 1 else (lo[j] <= tp):
                xp, why, xi = tp, "tp", j
                break
            if j + 1 >= z:
                break
            if j >= endi:
                xp, why, xi = o[j + 1], "struct", j + 1
                break
            tc = int(t[j]) + M15
            if tc % H1 == 0:
                hrs = (tc - t0) / H1
                p = p_at(dec, hrs, p)
                pser.append((hrs, p))
                if hrs > dec["tmax_h"]:
                    xp, why, xi = o[j + 1], "time", j + 1
                    break
                if bail and p < BAIL:
                    xp, why, xi = o[j + 1], "bail", j + 1
                    break
        if xp is None:
            xi = z - 1
            xp, why = c[xi], "end"
        t_in, t_out = int(t[en]), int(t[xi]) + M15
        sel = (ft > t_in) & (ft <= t_out)
        fund = -d * float(fr[sel].sum())
        r = d * (xp / ep - 1) - fin - tk + fund
        trades.append({"i": i, "e": int(en), "x": int(xi), "d": d, "ep": float(ep), "xp": float(xp), "tp": float(tp),
                       "r": float(r), "worst": float(min(worst, r)), "why": why, "ev_id": int(e),
                       "pser": pser[:: max(1, len(pser) // 60)]})  # fmt: skip
        busy = xi
    return trades


def gate_array(b, s, S, D, n15) -> np.ndarray:
    """R-B·EV가 진입할 만하다고 본 방향을 다리 끝(구조)·p(t)<0.30·시간 정리까지 문으로 쓴다."""
    t = b["t"]
    g = np.zeros(n15)
    E = S["E"]
    for e, dec in D.items():
        if not dec["take"]:
            continue
        i = int(s["i15"][E["C"][e]])
        if i < 0:
            continue
        t0 = int(s["t"][E["C"][e]]) + H4
        endi = int(s["i15"][E["C"][S["end"][e]]]) if S["end"][e] >= 0 else n15 - 1
        stop_i, p = min(endi, n15 - 1), dec["p0"]
        hrs = 1.0
        while True:
            j = int(np.searchsorted(t, t0 + int(hrs * H1))) - 1
            if j >= stop_i:
                break
            p = p_at(dec, hrs, p)
            if p < BAIL or hrs > dec["tmax_h"]:
                stop_i = j
                break
            hrs += 1
        g[i + 1 : stop_i + 1] = S["d"][e]
    return g


def equiv_cands(tr: list) -> list:
    """시각 맞춘 무작위용: 같은 방향·익절 거리·비상 손절·보유 시간."""
    out = []
    for x in tr:
        c0 = x["ep"]
        out.append(DW.cand(x["i"], x["d"], c0 * (1 - x["d"] * DISASTER), x["tp"], x["x"]))
    return out


def summ(tr, a, z, b) -> dict:
    sm = DW.summ(tr, a, z, b)
    sm["why"] = {k: sum(1 for x in tr if x["why"] == k) for k in ("tp", "bail", "time", "struct", "stop", "end")}
    return sm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=100)
    ap.add_argument("--nsim-big", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b, s = RL.load_all(DW.R5)
    F = DW.features(b)
    C = DW.all_cands(b, F)
    W = RL.windows(b["t"])
    n15 = len(b["t"])
    print("data", round(time.time() - t0, 1), flush=True)
    res, keep, variants = {}, {}, {}
    for name in SEGS:
        S = ev_stats(s, name)
        decs = {tp: decisions(s, S, tp, "E") for tp in TPS}
        decU = decisions(s, S, "EV", "U")
        for tp in TPS:
            variants[f"RB|{name}|{tp}"] = ("R", S, decs[tp], True)
        variants[f"RN|{name}|EV"] = ("R", S, decs["EV"], False)
        variants[f"RU|{name}|EV"] = ("R", S, decU, True)
        F["gate"][f"G2_{name}"] = gate_array(b, s, S, decs["EV"], n15)
        for en in ("S-NY", "P8"):
            variants[f"G|{name}|{en}"] = ("G", en, f"G2_{name}", None)
        print(name, "events", len(S["E"]["P"]), "decisions", {k: sum(v["take"] for v in d_.values()) for k, d_ in decs.items()},
              round(time.time() - t0), flush=True)  # fmt: skip

    def run(vn, a, z, **kw):
        kind, x1, x2, x3 = variants[vn]
        if kind == "R":
            fee = kw.pop("fee", "FA")
            return simulate(b, F, s, x1, x2, a, z, bail=x3, fee=fee, **kw)
        return DW.simulate(b, F, C[x1], x2, kw.pop("fee", "FA"), a, z, **kw)[0]

    for vn in variants:
        r = {}
        for w, (a, z) in W.items():
            tr = run(vn, a, z)
            r[w] = summ(tr, a, z, b)
            if w in ("pool", "d30", "year", "fresh"):
                keep.setdefault(vn, {})[w] = tr
        a, z = W["pool"]
        r["pool_maker"] = summ(run(vn, a, z, fee="FB"), a, z, b)
        r["redteam"] = {w: {"delay1": summ(run(vn, *W[w], delay=1), *W[w], b)["x5"],
                            "fee15": summ(run(vn, *W[w], fee_mult=1.5), *W[w], b)["x5"]} for w in MAIN}  # fmt: skip
        r["pos_all"] = all(r[w]["x5"]["ret"] > 0 for w in MAIN)
        r["liq0"] = all(r[w]["x5"]["liq"] == 0 for w in MAIN)
        r["mdd_ok"] = all(r[w]["x5"]["mdd"] <= 0.40 for w in MAIN)
        res[vn] = r
        print(vn, {w: (r[w]["n"], round(r[w]["win"] or 0, 2), round(r[w]["x5"]["ret"], 3)) for w in MAIN + ("d30",)},
              "pd", round(r["year"]["per_day"], 2), r["year"]["why"], round(time.time() - t0), flush=True)  # fmt: skip
    a, z = W["pool"]
    for vn in variants:
        tr = keep[vn]["pool"]
        if not tr:
            res[vn]["random"], res[vn]["pass"] = None, False
            continue
        ns = args.nsim_big if res[vn]["pos_all"] else args.nsim
        kind, x1, x2, _ = variants[vn]
        if kind == "R":
            rr = DW.random_p(b, F, equiv_cands(tr), "G0", "FA", a, z, tr, ns, 41)
        else:
            rr = DW.random_p(b, F, C[x1], x2, "FA", a, z, tr, ns, 41)
        res[vn]["random"] = rr
        res[vn]["bonf"] = min(1.0, rr["p"] * N_VAR)
        res[vn]["pass"] = bool(res[vn]["pos_all"] and res[vn]["liq0"] and res[vn]["mdd_ok"] and res[vn]["bonf"] < 0.05)
        print("random", vn, rr, flush=True)
    by_year = {}
    for vn in variants:
        tr = keep[vn]["pool"] + keep[vn]["year"]
        yrs: dict = {}
        for x in tr:
            y = int(np.datetime64(int(b["t"][x["e"]]), "ms").astype("datetime64[Y]").astype(int) + 1970)
            yrs.setdefault(y, []).append(x)
        by_year[vn] = {y: {"n": len(v), "x5": float(lev_path(v, 5)["ret"])} for y, v in sorted(yrs.items())}
    design_best = {name: max(("Q30", "Q50", "Q70"), key=lambda q: res[f"RB|{name}|{q}"]["design"]["x5"]["ret"]) for name in SEGS}
    save = {"n_variants": N_VAR, "n_tonight": N_TONIGHT_BEFORE + N_VAR, "passed": [v for v in res if res[v].get("pass")],
            "design_best_q": design_best, "res": res, "by_year": by_year, "sec": time.time() - t0}  # fmt: skip
    (out / "result.json").write_text(json.dumps(save, indent=1, default=float))
    (out / "trades.json").write_text(json.dumps({vn: {w: d_[w] for w in ("d30", "year")} for vn, d_ in keep.items()}, default=float))
    print("passed", save["passed"], "design_best", design_best, "sec", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
