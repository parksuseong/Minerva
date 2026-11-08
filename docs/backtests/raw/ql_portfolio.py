"""QL 포트폴리오: 따로 문을 가진 우위 16개 → 설계 구간에서 채택 → 한 포지션으로 합친다.

등록: 2026-11-08-btc-layered-model-hypotheses.md (개정 04:10). 진입·모의 매매는 daily_wave, 레짐 문은 regime_lines2.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/ql-portfolio-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/ql_portfolio.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import daily_wave as DW
import numpy as np
import pine_ports as P
import regime_lines as RL
import regime_lines2 as R2
import tvc_components as K
from dmi15_eval import frame
from htf_live import to15

M15, DAY = DW.M15, DW.DAY
LEV, LIQ = 5, -0.95
K2_LO, K2_HI = -2.2090162406812377, 2.8074433692152576
EDGES = [(s, g) for s in ("S-NY", "S-LN", "S-AS") for g in ("GT", "V2", "L3", "Z5")]
EDGES += [("OV", "GT"), ("OV", "V2"), ("PR", "G0"), ("PR", "GT")]
N_TESTS = len(EDGES) + 2  # 18
N_BEFORE = 5931
MAIN = DW.MAIN
SHOW = ("fresh", "design", "val", "year", "m6", "prev30", "d30", "pool")
FRACS = (1.0, 0.4)


# ---------------- 방향 문 ----------------
def add_gates(b: dict, s: dict, F: dict) -> None:
    n = len(b["c"])
    D = frame(b, 1440)
    c = D["c"]
    prev = np.r_[np.full(30, np.nan), c[:-30]]
    with np.errstate(invalid="ignore"):
        ts = np.where(c > prev, 1.0, np.where(c < prev, -1.0, 0.0))
    tsm = np.nan_to_num(to15(D, ts, n), nan=0.0)
    sm = F["gate"]["GT"] + F["gate"]["GD"] + tsm
    F["gate"]["V2"] = np.where(sm >= 2, 1.0, np.where(sm <= -2, -1.0, 0.0))
    for name in ("L3", "Z5"):
        S = R2.ev_stats(s, name)
        dec = R2.decisions(s, S, "EV", "E")
        F["gate"][name] = R2.gate_array(b, s, S, dec, n)


# ---------------- 진입 후보 ----------------
def ov_cands(b: dict) -> list:
    h, lo, c = b["h"], b["l"], b["c"]
    n = len(c)
    r = np.r_[0.0, c[1:] / c[:-1] - 1]
    cs, cs2 = np.cumsum(np.r_[0.0, r]), np.cumsum(np.r_[0.0, r * r])
    out = []
    for i in range(98, n - 2):
        m = (cs[i] - cs[i - 96]) / 96
        v = (cs2[i] - cs2[i - 96]) / 96 - m * m
        if v <= 0 or abs(r[i]) < 3 * np.sqrt(v):
            continue
        d = 1 if r[i] > 0 else -1
        stop = lo[i] * (1 - 0.0005) if d == 1 else h[i] * (1 + 0.0005)
        risk = abs(c[i] - stop)
        if risk > 0:
            out.append(DW.cand(i, d, stop, c[i] + d * 1.5 * risk, min(i + 32, n - 1)))
    return out


def pr_cands(b: dict) -> list:
    h, lo, c, v = b["h"], b["l"], b["c"], b["v"]
    n = len(c)
    zP = np.nan_to_num(b["f_zP"])
    z15 = np.nan_to_num(K.vwz(h, lo, c, v))
    atr = P.atr(h, lo, c, 14)
    on_l = (zP <= K2_LO) & (z15 <= -1.5)
    on_s = (zP >= K2_HI) & (z15 >= 1.5)
    out = []
    for on, d in ((on_l, 1), (on_s, -1)):
        first = on & ~np.r_[False, on[:-1]]
        for i in np.nonzero(first)[0]:
            if i < 20 or i >= n - 2 or not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            out.append(DW.cand(i, d, c[i] * (1 - d * 0.07), c[i] + d * 0.5 * atr[i], min(i + 96, n - 1)))
    return out


def edge_cands(b: dict, F: dict) -> dict:
    base = {k: DW.session_cands(b, F, k) for k in ("S-NY", "S-LN", "S-AS")}
    base["OV"] = ov_cands(b)
    base["PR"] = pr_cands(b)
    out = {}
    for en, g in EDGES:
        gg = F["gate"].get(g)
        keep = [x for x in base[en] if g == "G0" or gg[x["i"]] == x["d"]]
        out[f"{en}|{g}"] = sorted(keep, key=lambda x: x["i"])
    return out


# ---------------- 자산 곡선 ----------------
def path(tr: list, f: float) -> dict:
    eq, peak, mdd, liq = 1.0, 1.0, 0.0, 0
    for x in tr:
        if LEV * x["worst"] <= LIQ:
            liq += 1
            pos = -1.0
        else:
            pos = max(LEV * x["r"], -1.0)
            mdd = max(mdd, 1 - eq * (1 + f * LEV * x["worst"]) / peak)
        eq *= 1 + f * pos
        if eq <= 0:
            return {"ret": -1.0, "mdd": 1.0, "liq": liq}
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {"ret": eq - 1, "mdd": mdd, "liq": liq}


def summ(tr: list, a: int, z: int, b: dict) -> dict:
    days = max((int(b["t"][z - 1]) - int(b["t"][a])) / DAY, 1)
    r = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    out = {"n": len(tr), "per_day": len(tr) / days, "win": float((r > 0).mean()) if len(r) else None,
           "avg1": float(r.mean()) if len(r) else None, "x1": float(np.prod(1 + r) - 1) if len(r) else 0.0,
           "nl": int(sum(x["d"] == 1 for x in tr)), "ns": int(sum(x["d"] == -1 for x in tr))}  # fmt: skip
    for f in FRACS:
        out[f"x5_f{f}"] = {k: float(v) for k, v in path(tr, f).items()}
    return out


def merged(E: dict, edges: list, prio: dict) -> list:
    lst = []
    for e in edges:
        for x in E[e]:
            lst.append({**x, "edge": e})
    lst.sort(key=lambda x: (x["i"], -prio.get(x["edge"], 0)))
    return lst


def sim(b, F, cands, fee, a, z, **kw) -> list:
    tr, _ = DW.simulate(b, F, cands, "G0", fee, a, z, **kw)
    src = {}
    for x in cands:
        src.setdefault((x["i"], x["d"]), x)
    for x in tr:
        x["edge"] = src[(x["i"], x["d"])].get("edge")
    return tr


def random_p(b, F, cands, fee, a, z, real_tr, nsim, seed, f=0.4) -> dict:
    rng = np.random.default_rng(seed)
    t, c = b["t"], b["c"]
    n = len(t)
    by_slot: dict = {}
    for i in range(a, z - 2):
        by_slot.setdefault(int(t[i] % DAY), []).append(i)
    by_slot = {k: np.array(v) for k, v in by_slot.items()}
    src = {}
    for x in cands:
        src.setdefault((x["i"], x["d"]), x)
    vals = []
    for _ in range(nsim):
        rc = []
        for tr_ in real_tr:
            x = src[(tr_["i"], tr_["d"])]
            j = int(rng.choice(by_slot[int(t[x["i"]] % DAY)]))
            sd, td = x["stop"] / c[x["i"]] - 1, x["tp"] / c[x["i"]] - 1
            rc.append(DW.cand(j, x["d"], c[j] * (1 + sd), c[j] * (1 + td), min(j + x["tmax"] - x["i"], n - 1)))
        rc.sort(key=lambda y: y["i"])
        tr, _ = DW.simulate(b, F, rc, "G0", fee, a, z)
        vals.append(path(tr, f)["ret"])
    vals = np.array(vals)
    real = path(real_tr, f)["ret"]
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "real": float(real), "nsim": nsim}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b, s = RL.load_all(DW.R5)
    F = DW.features(b)
    add_gates(b, s, F)
    E = edge_cands(b, F)
    W = RL.windows(b["t"])
    print("built", round(time.time() - t0, 1), {k: len(v) for k, v in E.items()}, flush=True)

    # 1) 우위별 (FB 기본, FA 비교)
    edges = {}
    for e in E:
        edges[e] = {}
        for fee in ("FB", "FA"):
            edges[e][fee] = {w: summ(sim(b, F, E[e], fee, *W[w]), *W[w], b) for w in SHOW}
        d = edges[e]["FB"]["design"]
        edges[e]["admit"] = bool(d["n"] >= 30 and (d["avg1"] or 0) > 0 and d["x1"] > 0)
        print(e, "admit", edges[e]["admit"], {w: (edges[e]["FB"][w]["n"], round(edges[e]["FB"][w]["x5_f1.0"]["ret"], 2))
                                               for w in MAIN}, flush=True)  # fmt: skip
    admitted = [e for e in E if edges[e]["admit"]]
    prio = {e: edges[e]["FB"]["design"]["avg1"] or 0 for e in admitted}
    C = merged(E, admitted, prio)
    print("admitted", admitted, flush=True)

    # 2) 포트폴리오
    port, keep = {}, {}
    for fee in ("FB", "FA"):
        port[fee] = {}
        for w in SHOW:
            tr = sim(b, F, C, fee, *W[w])
            port[fee][w] = summ(tr, *W[w], b)
            port[fee][w]["by_edge"] = {e: sum(x["edge"] == e for x in tr) for e in admitted}
            if fee == "FB":
                keep[w] = tr
        print("portfolio", fee, {w: (port[fee][w]["n"], round(port[fee][w]["per_day"], 2),
                                     round(port[fee][w]["x5_f0.4"]["ret"], 2), round(port[fee][w]["x5_f0.4"]["mdd"], 2))
                                 for w in SHOW}, flush=True)  # fmt: skip
    # 수수료 부담: 같은 FB 거래를 수수료 0으로
    drag = {}
    for w in MAIN + ("pool",):
        g0 = summ(sim(b, F, C, "FB", *W[w], fee_mult=0.0), *W[w], b)
        drag[w] = {"gross_avg1": g0["avg1"], "fb_avg1": port["FB"][w]["avg1"], "fa_avg1": port["FA"][w]["avg1"],
                   "per_day": port["FB"][w]["per_day"], "gross_x5_f0.4": g0["x5_f0.4"]["ret"],
                   "fb_x5_f0.4": port["FB"][w]["x5_f0.4"]["ret"], "fa_x5_f0.4": port["FA"][w]["x5_f0.4"]["ret"]}  # fmt: skip
    # 3) 무작위·레드팀·걷기
    rnd = {}
    for fee in ("FB", "FA"):
        rnd[fee] = {}
        for w in ("pool",) + MAIN:
            a, z = W[w]
            tr = sim(b, F, C, fee, a, z)
            rnd[fee][w] = random_p(b, F, C, fee, a, z, tr, args.nsim if w == "pool" else 300, 41)
            print("random", fee, w, rnd[fee][w], flush=True)
    red = {}
    for fee in ("FB", "FA"):
        red[fee] = {w: {"delay1": path(sim(b, F, C, fee, *W[w], delay=1), 0.4),
                        "fee15": path(sim(b, F, C, fee, *W[w], fee_mult=1.5), 0.4)} for w in MAIN}  # fmt: skip
        tr = sim(b, F, C, fee, W["fresh"][0], len(b["t"]))
        yrs: dict = {}
        for x in tr:
            yrs.setdefault(datetime.fromtimestamp(int(b["t"][x["e"]]) / 1000, timezone.utc).year, []).append(x)
        red[fee]["by_year"] = {y: {"n": len(v), "x5_f0.4": path(v, 0.4)["ret"], "x5_f1": path(v, 1.0)["ret"]}
                               for y, v in sorted(yrs.items())}  # fmt: skip
    t = b["t"]
    rolls, s0, end = [], DW.ms("2020-03-01"), int(t[-1]) + M15
    while s0 + 90 * DAY <= end:
        a, z = int(np.searchsorted(t, s0)), int(np.searchsorted(t, s0 + 90 * DAY))
        rolls.append(path(sim(b, F, C, "FB", a, z), 0.4)["ret"])
        s0 += 30 * DAY
    rolls = np.array(rolls)
    roll = {"n": len(rolls), "pos_share": float((rolls > 0).mean()), "median": float(np.median(rolls)),
            "worst": float(rolls.min()), "best": float(rolls.max())}  # fmt: skip
    print("red", red, roll, flush=True)

    fb = port["FB"]
    gate = {"pos_all": all(fb[w]["x5_f0.4"]["ret"] > 0 for w in MAIN),
            "liq0": all(fb[w]["x5_f0.4"]["liq"] == 0 for w in MAIN),
            "mdd40": all(fb[w]["x5_f0.4"]["mdd"] <= 0.40 for w in MAIN),
            "random_bonf": rnd["FB"]["pool"]["p"] * N_TESTS < 0.05,
            "freq": fb["pool"]["per_day"] >= 0.7}  # fmt: skip
    (out / "result.json").write_text(json.dumps(
        {"n_tests": N_TESTS, "n_before": N_BEFORE, "admitted": admitted, "prio": prio, "edges": edges,
         "portfolio": port, "fee_drag": drag, "random": rnd, "redteam": red, "rolling": roll, "gate": gate,
         "pass": all(gate.values()), "windows": {k: [int(t[a]), int(t[z - 1])] for k, (a, z) in W.items()},
         "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    keep_out = {w: [{k: (v if not isinstance(v, (np.integer, np.floating)) else v.item()) for k, v in x.items()}
                    for x in tr] for w, tr in keep.items() if w in ("d30", "fresh", "design", "val", "year", "m6")}  # fmt: skip
    (out / "trades.json").write_text(json.dumps(keep_out, default=float))
    print("gate", gate, "pass", all(gate.values()), round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
