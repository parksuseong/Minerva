"""5차 탐색: 체결 쏠림(O)·K2 시장 상태 문(G)·거래소 간(V)·나눠 들어가기(S). 판정·무작위 기준.

문서: docs/backtests/2026-11-08-btc-round5-hypotheses.md (사전 등록 그대로)
구간: 새 구간 2020-03~2021-10(프리미엄 신호로는 처음), 설계 2022-01~2024-06, 검증 2024-07~2025-10,
이미 본 구간 2025-11~. 문턱은 설계 구간에서 정하고 고정한다. K2 문턱은 앞 시험 값 그대로.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/round5-2026-11-08
  P="systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
     PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3"
  $P docs/backtests/raw/r5_data.py --out $OUT
  $P docs/backtests/raw/r5_eval.py --out $OUT grid
  $P docs/backtests/raw/r5_eval.py --out $OUT random --nsim 2000
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from pathlib import Path

import numpy as np
import pine_eval as E
import pine_ports as P
import pine_tv_eval as T
import tvc_components as K
from numpy.lib.stride_tricks import sliding_window_view
from r5_data import load
from tvc_exits import make_native

M15 = E.M15
DAY = E.DAY
FEE = E.FEE
WIN = {
    "fresh": (E.ms("2020-03-01"), E.ms("2021-11-01")),
    "design": (E.ms("2022-01-01"), E.ms("2024-07-01")),
    "val": (E.ms("2024-07-01"), E.ms("2025-11-01")),
    "seen": (E.ms("2025-11-01"), None),
}
K2_LO, K2_HI = -2.2090162406812377, 2.8074433692152576  # 앞 시험(tvc) 설계 2%·98% 분위, 고정
ZP_DEEP = (-2.9094584772122682, 4.438499847131847)  # 같은 시험 0.5%·99.5% 분위
EXITS = ("X1", "X2", "X3", "X9", "X10")
SIGS = ("K2", "G1", "G2", "G3", "G4", "G5", "G6", "O1", "O2", "O3", "O4", "O5",
        "V1", "V2", "V3", "V4")  # fmt: skip
SCALE = ("S1", "S2", "S3")
N_TOTAL = len(SIGS) * len(EXITS) + len(SCALE)  # 83
N_TONIGHT = 5171 + N_TOTAL
K2_FAMILY = {"K2", "G1", "G2", "G3", "G4", "G5", "G6", "O4", "O5", "V2"}
NO_FRESH = {"G6"}
PART_FRESH = {"G1", "V4"}


def windows(t: np.ndarray) -> dict:
    w = {}
    for k, (a, z) in WIN.items():
        w[k] = (int(np.searchsorted(t, a)), len(t) if z is None else int(np.searchsorted(t, z)))
    w["last30"] = (int(np.searchsorted(t, t[-1] + M15 - 30 * DAY)), len(t))
    return w


def bars(b: dict) -> E.Bars:
    return E.Bars({k: b[k] for k in ("t", "o", "h", "l", "c", "ft", "fr")})


def summ(tr: list) -> dict:
    s = T.summ(tr)
    s.pop("why", None)
    return s


def q_design(x: np.ndarray, W: dict, qs: tuple) -> list:
    a, z = W["design"]
    v = x[a:z]
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, q)) for q in qs]


def atr_rank(atrp: np.ndarray, step: int = 16, k: int = 180) -> np.ndarray:
    """ATR%의 직전 30일 순위(16봉 간격 표본 180개, 현재 제외)."""
    n = len(atrp)
    out = np.full(n, np.nan)
    for r in range(step):
        s = atrp[r::step]
        if len(s) <= k:
            continue
        win = sliding_window_view(s[:-1], k)  # win[m] = s[m .. m+k-1], 비교 대상은 s[m+k]
        cur = s[k:]
        with np.errstate(invalid="ignore"):
            rk = (win < cur[:, None]).sum(1) / k
        rk[np.isnan(cur) | np.isnan(win).any(1)] = np.nan
        out[r + step * k :: step][: len(rk)] = rk
    return out


def arm_confirm(sig: np.ndarray, okl: np.ndarray, oks: np.ndarray, k: int = 8) -> np.ndarray:
    out = np.zeros(len(sig), np.int8)
    d, left = 0, 0
    for i in range(len(sig)):
        if sig[i] != 0:
            d, left = int(sig[i]), k
            continue
        if left > 0:
            left -= 1
            if d == 1 and okl[i]:
                out[i], left = 1, 0
            elif d == -1 and oks[i]:
                out[i], left = -1, 0
    return out


def build(b: dict, W: dict) -> tuple:
    t, o, h, lo, c, v, tbv = (b[k] for k in ("t", "o", "h", "l", "c", "v", "tbv"))
    n = len(c)
    nz = K.nz
    atr = P.atr(h, lo, c, 14)
    st_l = P.lowest(lo, 8) - 0.25 * atr
    st_s = P.highest(h, 8) + 0.25 * atr
    zP = b["f_zP"]
    z15 = K.vwz(h, lo, c, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        tbb = tbv / v
        k2l = (nz(zP) <= K2_LO) & (nz(z15) <= -1.5)
        k2s = (nz(zP) >= K2_HI) & (nz(z15) >= 1.5)
    thr: dict = {}
    dvol = b["x_dvol"]
    thr["dvol_med"] = q_design(dvol, W, (0.5,))[0]
    rk = atr_rank(atr / c)
    D = P.daily(b)
    sma50 = P.daily_state_15(D, P.sma(D["c"], 50), n)
    dclose = P.daily_state_15(D, D["c"], n)
    fund = b["f_f_last"]
    thr["fund_med"] = q_design(fund, W, (0.5,))[0]
    zoi = b["f_zOI4"]
    tb4, tb1 = b["f_tb4"], b["f_tb1"]
    thr["tb4_q2"], thr["tb4_q98"] = q_design(tb4, W, (0.02, 0.98))
    thr["tb1_q10"], thr["tb1_q90"] = q_design(tb1, W, (0.10, 0.90))
    G: dict = {}
    with np.errstate(invalid="ignore"):
        calm = ~np.isnan(dvol) & (dvol <= thr["dvol_med"])
        G["G1"] = (calm, calm)
        g2 = ~np.isnan(rk) & (rk <= 0.7)
        G["G2"] = (g2, g2)
        up = ~np.isnan(sma50) & (dclose > sma50)
        dn = ~np.isnan(sma50) & (dclose < sma50)
        G["G3"] = (up, dn)
        G["G4"] = (dn, up)
        G["G5"] = (~np.isnan(fund) & (fund <= thr["fund_med"]),
                   ~np.isnan(fund) & (fund >= thr["fund_med"]))  # fmt: skip
        G["G6"] = (~np.isnan(zoi) & (zoi <= 0), ~np.isnan(zoi) & (zoi >= 0))
        cond: dict = {"K2": (k2l, k2s)}
        for g, (gl, gs) in G.items():
            cond[g] = (k2l & gl, k2s & gs)
        # 체결 쏠림
        cond["O1"] = (~np.isnan(tb4) & (tb4 <= thr["tb4_q2"]) & (c > o),
                      ~np.isnan(tb4) & (tb4 >= thr["tb4_q98"]) & (c < o))  # fmt: skip
        from nonprice_data import roll_z

        vz = roll_z(v, 96, 80)
        body = np.abs(c - o) <= 0.3 * atr
        rng = h - lo
        upper = (c - lo) >= 0.5 * rng
        lower = (h - c) >= 0.5 * rng
        cond["O2"] = (
            (nz(vz) >= 2) & (nz(tbb, 0.5) <= 0.45) & body & upper,
            (nz(vz) >= 2) & (nz(tbb, 0.5) >= 0.55) & body & lower,
        )
        delta = 2 * tbv - v
        cs = np.r_[0.0, np.cumsum(delta)]
        i = np.arange(n)
        d96 = cs[i + 1] - cs[np.maximum(i - 95, 0)]
        prev_lo = P.shift(P.lowest(lo, 96), 1)
        prev_hi = P.shift(P.highest(h, 96), 1)
        cond["O3"] = ((lo < nz(prev_lo, -np.inf)) & (d96 > 0) & (i >= 96),
                      (h > nz(prev_hi, np.inf)) & (d96 < 0) & (i >= 96))  # fmt: skip
        sell = K.recent(~np.isnan(tb1) & (tb1 <= thr["tb1_q10"]), 4)
        buy = K.recent(~np.isnan(tb1) & (tb1 >= thr["tb1_q90"]), 4)
        cond["O4"] = (k2l & sell, k2s & buy)
        zc, ze, zd = b["x_zcbp"], b["x_zeth"], b["x_zdvc"]
        cond["V1"] = (nz(zc) >= 2.5, nz(zc) <= -2.5)
        cond["V2"] = (k2l & ~np.isnan(zc) & (zc >= 0), k2s & ~np.isnan(zc) & (zc <= 0))
        cond["V3"] = (nz(ze) >= 2.5, nz(ze) <= -2.5)
        cond["V4"] = ((nz(zd) >= 2.5) & (nz(z15) <= -1.5), (nz(zd) >= 2.5) & (nz(z15) >= 1.5))
    sig = {k: K.sig_from(*cond[k]) for k in cond}
    with np.errstate(invalid="ignore"):
        sig["O5"] = arm_confirm(sig["K2"], nz(tbb, 0.5) > 0.55, nz(tbb, 0.5) < 0.45)
    own_k2 = (nz(zP) >= 0, nz(zP) <= 0)
    own_z = (nz(z15) >= 0, nz(z15) <= 0)
    sets = {}
    for s in SIGS:
        own = own_k2 if s in K2_FAMILY else own_z
        for x in EXITS:
            sets[f"{s}|{x}"] = {"sig": sig[s],
                                "native": make_native(x, atr, st_l, st_s, own[0], own[1])}  # fmt: skip
    comp = {"sig": sig, "atr": atr, "zP": zP, "z15": z15, "thr": thr}
    return sets, comp


# ---------------- 나눠 들어가기 ----------------
def sim_scale(B, b: dict, sig: np.ndarray, atr: np.ndarray, zP: np.ndarray, mode: str,
              a: int, z: int) -> list:  # fmt: skip
    """반 크기 진입 → 조건이면 반 더(한 번). 손익·최악 평가손익은 전체 계획 크기(1.0) 기준."""
    o, h, lo, c = B.o, B.h, B.l, B.c
    idx = (np.flatnonzero(np.asarray(sig[a:z]) != 0) + a).tolist()
    sg = sig.tolist()
    at = atr.tolist()
    zp = np.nan_to_num(zP).tolist()
    trades = []
    i = a
    while True:
        k = bisect.bisect_left(idx, i)
        if k >= len(idx):
            break
        j = idx[k]
        if j + 1 >= z:
            break
        dr = sg[j]
        e = j + 1
        A = at[j]
        fills = [(0.5, o[e], int(B.t[e]))]
        fees = FEE * 0.5
        added = False
        stop = o[e] * (1 - dr * 0.07)
        trig = o[e] - dr * (1.0 if mode == "S1" else 2.0) * A
        pend = False
        worst = -fees
        xi, xp, why = None, None, None
        ii = e
        while ii < z:
            add_now = False
            if not added:
                px = None
                if pend:
                    px = o[ii]
                elif mode in ("S1", "S2"):
                    if (o[ii] <= trig) if dr == 1 else (o[ii] >= trig):
                        px = o[ii]
                    elif (lo[ii] <= trig) if dr == 1 else (h[ii] >= trig):
                        px = trig
                if px is not None:
                    fills.append((0.5, px, int(B.t[ii]) + 1))
                    fees += FEE * 0.5
                    added, add_now, pend = True, True, False
            size = sum(f for f, _, _ in fills)
            avg = sum(f * p for f, p, _ in fills) / size
            adverse = lo[ii] if dr == 1 else h[ii]
            hit_o = (o[ii] <= stop) if dr == 1 else (o[ii] >= stop)
            hit = (lo[ii] <= stop) if dr == 1 else (h[ii] >= stop)
            if hit_o or hit:
                xi, xp, why = ii, (o[ii] if hit_o else stop), "stop"
                break
            mtm = sum(f * dr * (adverse / p - 1) for f, p, _ in fills) - fees
            worst = min(worst, mtm)
            tp = avg + dr * 0.5 * A
            if not add_now:
                if (o[ii] >= tp) if dr == 1 else (o[ii] <= tp):
                    xi, xp, why = ii, o[ii], "tp"
                    break
                if (h[ii] >= tp) if dr == 1 else (lo[ii] <= tp):
                    xi, xp, why = ii, tp, "tp"
                    break
            if mode == "S3" and not added and not pend:
                if (zp[ii] <= ZP_DEEP[0]) if dr == 1 else (zp[ii] >= ZP_DEEP[1]):
                    pend = True
            if ii - e >= 96:
                if ii + 1 < z:
                    xi, xp, why = ii + 1, o[ii + 1], "time"
                else:
                    xi, xp, why = ii, c[ii], "end"
                break
            ii += 1
        if xi is None:
            xi, xp, why = z - 1, c[z - 1], "end"
        size = sum(f for f, _, _ in fills)
        fees += FEE * size * 1.0
        gross = sum(f * dr * (xp / p - 1) for f, p, _ in fills)
        t_out = int(B.t[xi]) + (1 if why in ("stop", "tp") else 0)
        fund = sum(f * E.funding(B, dr, p, ti, t_out, None) for f, p, ti in fills)
        r = gross - fees - fund
        mtm_x = gross - fees
        worst = min(worst, mtm_x, r)
        trades.append({"d": dr, "j": j, "e": e, "x": xi, "ep": fills[0][1], "xp": xp, "r": r,
                       "worst": worst, "fund": fund, "why": why, "added": added})  # fmt: skip
        i = xi + 1
    return trades


# ---------------- 판정 ----------------
def passes(r: dict, name: str) -> dict:
    d, v, f = r["design"], r["val"], r["fresh"]
    out = {
        "win70": (v["win"] or 0) >= 0.70, "x10_pos": v["x10"]["ret"] > 0,
        "liq0_val": v["x10"]["liq"] == 0, "liq0_design": d["x10"]["liq"] == 0,
        "n30": v["n"] >= 30, "design1_pos": d["x1"]["ret"] > 0,
    }  # fmt: skip
    if name.split("|")[0] not in NO_FRESH:
        out["fresh_ok"] = ((f["win"] or 0) >= 0.70 and f["x10"]["ret"] > 0
                           and f["x10"]["liq"] == 0 and f["n"] >= 20)  # fmt: skip
    return out


def run_one(B, b, comp, sets, key, a, z, sig=None) -> list:
    if key in SCALE:
        s = comp["sig"]["K2"] if sig is None else sig
        return sim_scale(B, b, s, comp["atr"], comp["zP"], key, a, z)
    if sig is None:
        return T.simulate(B, sets[key], ("N",), a, z)
    return T.simulate(B, sets[key], ("N",), a, z, sig=sig, epx=None)


def setup(out: Path):
    b = load(out)
    W = windows(b["t"])
    sets, comp = build(b, W)
    return b, W, sets, comp, bars(b)


def cmd_grid(out: Path) -> None:
    t0 = time.time()
    b, W, sets, comp, B = setup(out)
    keys = list(sets) + list(SCALE)
    assert len(keys) == N_TOTAL, len(keys)
    print("built", round(time.time() - t0, 1), flush=True)
    res = {}
    for key in keys:
        r = {w: summ(run_one(B, b, comp, sets, key, *W[w]))
             for w in ("fresh", "design", "val", "seen", "last30")}  # fmt: skip
        r["pass"] = {k: bool(x) for k, x in passes(r, key).items()}
        r["pass_all"] = bool(all(r["pass"].values()))
        res[key] = r
    sigc = {s: {w: {"long": int((comp["sig"][s][a:z] == 1).sum()),
                    "short": int((comp["sig"][s][a:z] == -1).sum())}
                for w, (a, z) in W.items()} for s in SIGS}  # fmt: skip
    pool = [(k, r) for k, r in res.items() if r["val"]["n"] >= 30
            and r["val"]["x10"]["liq"] == 0 and r["design"]["x10"]["liq"] == 0]  # fmt: skip
    pool.sort(key=lambda x: ((x[1]["val"]["win"] or 0), x[1]["val"]["x10"]["ret"]), reverse=True)
    top_win = [k for k, _ in pool[:2]]
    pool.sort(key=lambda x: (x[1]["val"]["x10"]["ret"], x[1]["val"]["x1"]["ret"]), reverse=True)
    top_ret = [k for k, _ in pool[:2]]
    cand = [k for k, r in res.items() if r["pass_all"]]
    bh = {w: float(b["c"][z - 1] / b["o"][a] - 1) for w, (a, z) in W.items()}
    (out / "grid.json").write_text(json.dumps(
        {"variants_evaluated": len(res), "thresholds": comp["thr"], "signals": sigc, "bh": bh,
         "candidates": cand, "top2_val_win": top_win, "top2_val_x10": top_ret,
         "sec": time.time() - t0, "res": res}, indent=1))  # fmt: skip
    print("variants", len(res), "candidates", cand, "top", top_win, top_ret, flush=True)


def cmd_random(out: Path, nsim: int, items: list | None) -> None:
    b, W, sets, comp, B = setup(out)
    g = json.loads((out / "grid.json").read_text())
    if not items:
        items = list(dict.fromkeys(g["candidates"] + g["top2_val_win"] + g["top2_val_x10"]))
    rng = np.random.default_rng(20261108)
    res = {}
    for it in items:
        ow = {}
        for w in ("fresh", "design", "val"):
            a, z = W[w]
            tr = run_one(B, b, comp, sets, it, a, z)
            real = summ(tr)
            if real["n"] == 0:
                continue
            r1, wn, r10 = [], [], []
            for _ in range(nsim):
                sg = T.random_sigs(rng, tr, a, z, b["t"], False)
                rs = summ(run_one(B, b, comp, sets, it, a, z, sig=sg))
                r1.append(rs["x1"]["ret"])
                r10.append(rs["x10"]["ret"])
                wn.append(rs["win"] or 0.0)
            r1a, wna = np.array(r1), np.array(wn)
            z1 = (real["x1"]["ret"] - r1a.mean()) / (r1a.std() + 1e-12)
            pn1 = 0.5 * math.erfc(z1 / math.sqrt(2))
            ow[w] = {"n": real["n"], "win": real["win"], "real1": real["x1"]["ret"],
                     "real10": real["x10"]["ret"], "rand1_mean": float(r1a.mean()),
                     "rand10_median": float(np.median(r10)), "rand_win_mean": float(wna.mean()),
                     "p1": float((r1a >= real["x1"]["ret"]).mean()),
                     "pwin": float((wna >= (real["win"] or 0)).mean()),
                     "p1_normal": pn1, "p1_bonf": min(1.0, pn1 * N_TOTAL)}  # fmt: skip
        res[it] = ow
        print(it, {w: (v["n"], round(v["win"], 3), round(v["p1"], 4), round(v["p1_bonf"], 3))
                   for w, v in ow.items()}, flush=True)  # fmt: skip
    (out / "random.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("grid")
    r = sp.add_parser("random")
    r.add_argument("--nsim", type=int, default=2000)
    r.add_argument("--items", nargs="*")
    a = ap.parse_args()
    p = Path(a.out).expanduser()
    if a.cmd == "grid":
        cmd_grid(p)
    else:
        cmd_random(p, a.nsim, a.items)
