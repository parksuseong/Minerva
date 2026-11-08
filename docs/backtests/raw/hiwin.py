"""고승률(2~3일에 한 번): 신호 7 × 청산 4 + 합집합 2 = 30개, 1분봉 체결, 같은 기하 무작위 기준.

등록: docs/backtests/2026-11-08-btc-hiwin.md. 연구 전용(주문 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/hiwin-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/hiwin.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import daily_wave as DW
import numpy as np
import pine_ports as P
import ql_upgrade as U
import regime_lines as RL
import rr_asym as RA
from btc_fractal import scr3
from dmi15_eval import frame

MIN, M15, DAY = U.MIN, U.M15, U.DAY
TAKER, MAKER, TICK, SLIP = U.TAKER, U.MAKER, U.TICK, 0.0001
HOLD = 72 * 60
N_BONF, N_BEFORE = 30, 6450
N_CUM = N_BEFORE + N_BONF
MAINW = ("fresh", "design", "val", "year")
WINS = MAINW + ("m6", "d30", "pool")
SIGS = ("R4a", "R4b", "R1", "S4", "DC", "NRg", "NR0")
EXITS = ("E1", "E2", "E3", "E4")


def norm_p(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---------------- 신호 ----------------
def first_in(x, lo_thr, hi_thr):
    with np.errstate(invalid="ignore"):
        lo, hi = x <= lo_thr, x >= hi_thr
    return lo & ~np.r_[False, lo[:-1]], hi & ~np.r_[False, hi[:-1]]


def on15(Fr, mask, n):
    out = np.zeros(n, bool)
    out[Fr["last15"][mask]] = True
    return out


def build_signals(b, F) -> dict:
    """반환 {이름: [(i15, d, 신호가격, 진입 1분 k 또는 None), ...]} — k None이면 i15 마감 직후 1분 시가."""
    n = len(b["c"])
    gt = F["gate"]["GT"]
    H4, H1 = frame(b, 240), frame(b, 60)
    out = {}

    def gated(L, S, name):
        i = np.r_[np.nonzero(L)[0], np.nonzero(S)[0]]
        d = np.r_[np.ones(L.sum(), int), -np.ones(S.sum(), int)]
        o = np.argsort(i, kind="stable")
        i, d = i[o], d[o]
        keep = gt[i] == d
        out[name] = [(int(a), int(z), float(b["c"][a]), None) for a, z in zip(i[keep], d[keep])]

    r4 = P.rsi(H4["c"], 2)
    for name, (lo, hi) in (("R4a", (10, 90)), ("R4b", (5, 95))):
        L, S = first_in(r4, lo, hi)
        # RSI2 낮음 = 과매도 → 롱
        gated(on15(H4, L, n), on15(H4, S, n), name)
    r1 = P.rsi(H1["c"], 2)
    L, S = first_in(r1, 5, 95)
    gated(on15(H1, L, n), on15(H1, S, n), "R1")
    l4, s4, *_ = scr3({"h": H4["h"], "l": H4["l"], "c": H4["c"]})
    gated(on15(H4, l4, n), on15(H4, s4, n), "S4")
    L, S = RA.donchian_sig(b, 96)
    gated(L, S, "DC")
    return out


def nr4_signals(b, F, m) -> tuple[list, list]:
    D = frame(b, 1440)
    rng = D["h"] - D["l"]
    gt = F["gate"]["GT"]
    t1 = m["t"]
    sg, s0 = [], []
    for j in range(3, len(rng) - 1):
        if rng[j] > rng[j - 3 : j + 1].min():
            continue
        i_end = int(D["last15"][j])
        day_a = int(b["t"][i_end]) + M15
        ka, kz = np.searchsorted(t1, day_a), np.searchsorted(t1, day_a + DAY)
        if kz - ka < 600:
            continue
        hi, lo = D["h"][j], D["l"][j]
        up, dn = m["h"][ka:kz] >= hi, m["l"][ka:kz] <= lo
        ku = int(np.argmax(up)) if up.any() else None
        kd = int(np.argmax(dn)) if dn.any() else None
        if ku is None and kd is None:
            continue
        if ku is not None and kd is not None and ku == kd:
            continue
        if kd is None or (ku is not None and ku < kd):
            k, d, lvl = ka + ku, 1, hi
        else:
            k, d, lvl = ka + kd, -1, lo
        i15 = int(np.searchsorted(b["t"], t1[k], "right")) - 2  # 진입 분 직전에 마감된 15분봉
        rec = (i15, d, float(lvl), int(k))
        s0.append(rec)
        if gt[i_end] == d:
            sg.append(rec)
    return sg, s0


# ---------------- 체결 ----------------
class Sim:
    def __init__(self, m, b, F, fund_fn):
        self.m, self.b, self.F, self.fund_fn = m, b, F, fund_fn
        self.n1 = len(m["t"])
        self.atrp = F["atr_d"] / b["c"]
        g = F["gate"]["GT"]
        n = len(g)
        # 다음 반대 GT 위치(i 이후)
        self.nxt = {}
        for dv in (1, -1):
            pos = np.where(g == dv, np.arange(n), n)
            self.nxt[dv] = np.minimum.accumulate(pos[::-1])[::-1]

    def kmin(self, i15, lat=0):
        ts = int(self.b["t"][i15]) + M15 + lat * MIN
        k = int(np.searchsorted(self.m["t"], ts))
        if k >= self.n1 - 2 or int(self.m["t"][k]) - ts > 5 * MIN:
            return -1
        return k

    def geom(self, ex, i15, kq):
        a = float(self.atrp[i15]) if np.isfinite(self.atrp[i15]) else 0.03
        if ex == "E1":
            return 0.03, 0.01
        if ex == "E2":
            return 0.03, 0.015
        S = min(max(a, 0.01), 0.035)
        if ex == "E3":
            return S, max(0.35 * a, 0.004)
        return S, max(kq * a, 0.004)

    def bail_k(self, i15, d):
        j = int(self.nxt[-d][min(i15 + 1, len(self.b["t"]) - 1)])
        if j >= len(self.b["t"]):
            return None
        return self.kmin(j)

    def trade(self, k, d, S, T, bail=False, i15=None, ep=None, k0=None, fee_mult=1.0, slip=SLIP, hold=HOLD):
        m = self.m
        ep = float(m["o"][k]) if ep is None else ep
        stop, tp = ep * (1 - d * S), ep * (1 + d * T)
        tend = min(k + hold, self.n1 - 1)
        if bail:
            kb = self.bail_k(i15, d)
            if kb is not None and kb > k and kb - 1 < tend:
                tend = kb - 1
        k0 = k if k0 is None else k0
        hit = U.first_hit(m, d, k0, tend, stop, tp, k0) if tend >= k0 else None
        if hit is None:
            j, why, px = tend, "time", float(m["c"][tend])
        else:
            j, why = hit
            if why == "stop":
                o = float(m["o"][j])
                px = (min(stop, o) if d == 1 else max(stop, o)) - d * TICK
            else:
                px = tp
        fin = TAKER * fee_mult + slip
        fout = MAKER * fee_mult if why == "tp" else TAKER * fee_mult + slip
        h, lo = m["h"][k : j + 1], m["l"][k : j + 1]
        worst = (lo.min() / ep - 1) if d == 1 else (1 - h.max() / ep)
        t_in, t_out = int(m["t"][k]), int(m["t"][j]) + MIN
        r = d * (px / ep - 1) - fin - fout + self.fund_fn(t_in, t_out, d)
        return {"k": int(k), "x": int(j), "d": int(d), "ep": ep, "xp": float(px), "r": float(r), "S": S, "T": T,
                "worst": float(min(worst, r)), "why": why, "size": 1.0, "t_in": t_in, "t_out": t_out}  # fmt: skip

    def entry(self, rec, lat=0, slip=SLIP):
        """반환 (k, ep, k0) 또는 None(빈 구간·추격 제한)."""
        i15, d, sp, kk = rec
        if kk is None:
            k = self.kmin(i15, lat)
            if k < 0:
                return None
            ep, k0 = float(self.m["o"][k]), k
        else:
            k = kk + lat
            if k >= self.n1 - 2:
                return None
            o = float(self.m["o"][k])
            ep = max(sp, o) if d == 1 else min(sp, o)
            if lat:
                ep = o
            k0 = k + 1
        lim = 0.03 if self.F["agree"][i15] else 0.01  # 15분·4시간·일봉 방향 일치
        if abs(ep / sp - 1) >= lim:
            return None
        return k, ep, k0

    def run(self, recs, a_ms, z_ms, ex, kq=0.0, lat=0, **kw):
        out, busy = [], -1
        bail = ex == "E4"
        t = self.b["t"]
        for rec in recs:
            i15, d = rec[0], rec[1]
            if not (a_ms <= t[i15] and t[i15] + M15 < z_ms):
                continue
            e = self.entry(rec, lat, kw.get("slip", SLIP))
            if e is None or e[0] <= busy:
                continue
            k, ep, k0 = e
            S, T = self.geom(ex, i15, kq)
            x = self.trade(k, d, S, T, bail, i15, ep, k0, **kw)
            if x["t_out"] > z_ms:
                continue
            x["i15"] = int(i15)
            x["ex"] = ex
            out.append(x)
            busy = x["x"]
        return out


def mfe_quantile(sim, recs, a_ms, z_ms, q=0.22) -> float:
    """설계 신호 전부: 손절 1×ATR 전·72시간 안 최대 유리 이동 / ATR%의 q 백분위."""
    m, t = sim.m, sim.b["t"]
    vals = []
    for rec in recs:
        i15, d = rec[0], rec[1]
        if not (a_ms <= t[i15] and t[i15] + M15 + HOLD * MIN < z_ms):
            continue
        e = sim.entry(rec)
        if e is None:
            continue
        k, ep, k0 = e
        a = float(sim.atrp[i15])
        if not np.isfinite(a):
            continue
        S = min(max(a, 0.01), 0.035)
        h, lo = m["h"][k0 : k0 + HOLD], m["l"][k0 : k0 + HOLD]
        adv = (1 - lo / ep) if d == 1 else (h / ep - 1)
        fav = (h / ep - 1) if d == 1 else (1 - lo / ep)
        js = np.nonzero(adv >= S)[0]
        end = js[0] if len(js) else len(fav)
        mf = fav[:end].max() if end > 0 else 0.0
        vals.append(max(mf, 0.0) / a)
    return float(np.quantile(vals, q)) if vals else 0.35


def summ(tr, a_ms, z_ms) -> dict:
    s = U.summ(tr, a_ms, z_ms)
    for dv, nm in ((1, "L"), (-1, "S")):
        r = [x["r"] for x in tr if x["d"] == dv]
        s[f"win{nm}"] = float(np.mean(np.array(r) > 0)) if r else None
    s["why"] = {w: int(sum(x["why"] == w for x in tr)) for w in ("stop", "tp", "time")}
    s["avgS"] = float(np.mean([x["S"] for x in tr])) if tr else None
    s["avgT"] = float(np.mean([x["T"] for x in tr])) if tr else None
    return s


# ---------------- 무작위(같은 기하) ----------------
def random_geom(sim, tr, nsim=200, seed=7) -> np.ndarray:
    """거래마다 ±15일 안 아무 15분봉·같은 방향·같은 S·T·보유(E4면 같은 무효화). 반환 r 배열 [거래, nsim]."""
    rng = np.random.default_rng(seed)
    t = sim.b["t"]
    out = np.full((len(tr), nsim), np.nan)
    last_ok = int(np.searchsorted(t, int(t[-1]) - HOLD * MIN - DAY))
    for a, x in enumerate(tr):
        c = x["i15"]
        lo, hi = max(c - 15 * 96, 200), min(c + 15 * 96, last_ok)
        bail = x["ex"] == "E4"
        for s in range(nsim):
            for _ in range(5):
                i = int(rng.integers(lo, hi))
                k = sim.kmin(i)
                if k >= 0:
                    break
            else:
                continue
            out[a, s] = sim.trade(k, x["d"], x["S"], x["T"], bail, i)["r"]
    return out


def rnd_stats(tr, R) -> dict:
    if len(tr) < 5:
        return {}
    r = np.array([x["r"] for x in tr])
    wr = (r > 0).mean()
    rw = np.array([np.mean(R[~np.isnan(R[:, s]), s] > 0) for s in range(R.shape[1])])
    rm = np.nanmean(R, 0)
    zw = (wr - rw.mean()) / (rw.std() + 1e-12)
    zm = (r.mean() - rm.mean()) / (rm.std() + 1e-12)
    return {"n": len(tr), "win": float(wr), "rnd_win": float(rw.mean()), "excess": float(wr - rw.mean()),
            "p_win": norm_p(zw), "p_win_emp": float((np.sum(rw >= wr) + 1) / (len(rw) + 1)),
            "mean": float(r.mean()), "rnd_mean": float(rm.mean()), "p_mean": norm_p(zm),
            "p_mean_emp": float((np.sum(rm >= r.mean()) + 1) / (len(rm) + 1))}  # fmt: skip


# ---------------- 실행 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b, _ = RL.load_all(DW.R5)
    F = DW.features(b)
    t_end = int(b["t"][-1]) + M15
    m = U.load_1m(t_end)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    sim = Sim(m, b, F, fund_fn)
    W = RA.windows(t_end)
    sigs = build_signals(b, F)
    sigs["NRg"], sigs["NR0"] = nr4_signals(b, F, m)
    print("loaded", {k: len(v) for k, v in sigs.items()}, round(time.time() - t0), flush=True)

    kq = {s: mfe_quantile(sim, sigs[s], *W["design"]) for s in SIGS}
    print("kq", {s: round(v, 3) for s, v in kq.items()}, flush=True)

    spec = {f"{s}|{e}": (s, e) for s in SIGS for e in EXITS}
    res, trades = {}, {}

    def recs_of(s):
        if "+" in s:
            allr = [r for p in s.split("+") for r in sigs[p]]
            return sorted(allr, key=lambda r: (r[0], r[3] or 0))
        return sigs[s]

    def kq_of(s):
        return kq.get(s, float(np.median([kq[p] for p in s.split("+")])))

    def run_one(s, e, a_ms, z_ms, **kw):
        if "+" in s and e == "E4":  # 합집합 E4: 신호마다 자기 k
            outs, busy = [], -1
            allr = sorted([(r, p) for p in s.split("+") for r in sigs[p]], key=lambda z: (z[0][0], z[0][3] or 0))
            for rec, p in allr:
                tr = sim.run([rec], a_ms, z_ms, e, kq[p], **kw)
                if tr and tr[0]["k"] > busy:
                    outs.append(tr[0])
                    busy = tr[0]["x"]
            return outs
        return sim.run(recs_of(s), a_ms, z_ms, e, kq_of(s), **kw)

    def evaluate(name):
        s, e = spec[name]
        r, k = {}, {}
        for w in WINS:
            tr = run_one(s, e, *W[w])
            r[w] = summ(tr, *W[w])
            k[w] = tr
        return r, k

    def show(name, r):
        print(name, {w: (r[w]["n"], round(r[w]["per_day"], 2), r[w]["win"] and round(r[w]["win"], 3),
                         round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f0.4"]["ret"], 2))
                     for w in MAINW + ("d30",)}, round(time.time() - t0), flush=True)  # fmt: skip

    for name in list(spec):
        res[name], trades[name] = evaluate(name)
        show(name, res[name])

    def pick(names, freq=True):
        ok = [n for n in names if res[n]["design"]["n"] >= 60 and (res[n]["design"]["win"] or 0) >= 0.70
              and (not freq or 0.25 <= res[n]["design"]["per_day"] <= 0.6)]  # fmt: skip
        if not ok:
            return None
        top = max(res[n]["design"]["x5_f1.0"]["ret"] for n in ok)
        near = [n for n in ok if res[n]["design"]["x5_f1.0"]["ret"] >= top - 0.10 * abs(top)]
        return min(near, key=lambda n: res[n]["design"]["x5_f1.0"]["mdd"])

    base = list(spec)
    p0 = pick(base)
    relaxed = p0 is None
    if relaxed:
        p0 = pick(base, False)
    if p0 is None:
        p0 = max(base, key=lambda n: res[n]["design"]["x5_f1.0"]["ret"])
    e0 = spec[p0][1]
    top_sig = sorted(SIGS, key=lambda s: -res[f"{s}|{e0}"]["design"]["x5_f1.0"]["ret"])[:2]
    c1 = "+".join(top_sig)
    spec[f"C1:{c1}|{e0}"] = (c1, e0)
    spec["C2|E4"] = ("R4a+R4b+R1+S4+DC+NRg", "E4")
    for name in (f"C1:{c1}|{e0}", "C2|E4"):
        res[name], trades[name] = evaluate(name)
        show(name, res[name])
    fin = pick(list(spec)) or (pick(list(spec), False) if relaxed else None) or p0
    print("PICK", fin, "relaxed" if relaxed else "", flush=True)

    # 승률 ≥70% 네 구간 모두 + 고른 안 → 무작위·판정
    hi = [n for n in spec if all((res[n][w]["win"] or 0) >= 0.70 for w in MAINW)]
    cand = list(dict.fromkeys(hi + [fin]))
    full = {}
    rnd, verdict, red, years = {}, {}, {}, {}
    for n in cand:
        s, e = spec[n]
        tr = run_one(s, e, W["fresh"][0], t_end)
        R = random_geom(sim, tr, 200, 7)
        full[n] = tr
        tin = np.array([x["t_in"] for x in tr])
        rnd[n] = {}
        for w in MAINW + ("pool",):
            sel = (tin >= W[w][0]) & (tin < W[w][1])
            rnd[n][w] = rnd_stats([x for x, z in zip(tr, sel) if z], R[sel])
        print("rnd", n, {w: (round(v.get("excess", 0), 3), round(v.get("p_win", 1), 4), round(v.get("p_mean", 1), 4))
                         for w, v in rnd[n].items()}, round(time.time() - t0), flush=True)  # fmt: skip
        red[n] = {}
        for nm, kw in (("lat1m", {"lat": 1}), ("lat15m", {"lat": 15}), ("slip+0.02", {"slip": SLIP + 0.0002}),
                       ("fee1.5", {"fee_mult": 1.5})):  # fmt: skip
            red[n][nm] = {w: summ(run_one(s, e, *W[w], **kw), *W[w]) for w in ("val", "year")}
        years[n] = {}
        for y in range(2020, 2027):
            a_, z_ = RA.ms(f"{y}-01-01"), min(RA.ms(f"{y + 1}-01-01"), t_end)
            years[n][y] = summ(run_one(s, e, a_, z_), a_, z_)
        fx, rp = res[n], rnd[n]["pool"]
        g1 = all((fx[w]["win"] or 0) >= 0.70 for w in MAINW)
        g2 = all(fx[w][f"x5_f{f}"]["ret"] > 0 for w in MAINW for f in (1.0, 0.4)) and all(
            fx[w]["x5_f1.0"]["liq"] == 0 for w in MAINW)
        g3a = bool(rp) and min(rp["p_mean"], rp["p_win"]) * N_CUM < 0.05
        g3b = bool(rp) and rp["p_win"] < 0.01 and all(rnd[n][w].get("excess", -1) > 0 for w in MAINW)
        g4 = all(0.25 <= fx[w]["per_day"] <= 0.6 for w in ("design", "year"))
        g5 = fx["pool"]["n"] > 0 and min(fx["pool"]["nl"], fx["pool"]["ns"]) >= 0.15 * fx["pool"]["n"]
        g6 = all(red[n][k][w]["x5_f1.0"]["ret"] > 0 for k in red[n] for w in ("val", "year"))
        verdict[n] = {"g1_win70": g1, "g2_ret_liq": g2, "g3a_bonf": g3a, "g3b_consistent": g3b, "g4_freq": g4,
                      "g5_sides": g5, "g6_red": g6, "pass": bool(g1 and g2 and (g3a or g3b) and g4 and g5 and g6)}  # fmt: skip
        print("VERDICT", n, verdict[n], {k: {w: round(v["x5_f1.0"]["ret"], 2) for w, v in red[n][k].items()}
                                         for k in red[n]},
              {y: round(v["x5_f1.0"]["ret"], 2) for y, v in years[n].items()}, flush=True)  # fmt: skip

    n_win = {w: sum((res[n][w]["win"] or 0) > 0.70 for n in spec) for w in MAINW}
    json.dump({"windows": W, "spec": spec, "kq": kq, "res": res, "final": fin, "relaxed": relaxed, "hi": hi,
               "rnd": rnd, "red": red, "years": years, "verdict": verdict, "n_win70": n_win,
               "n_bonf": N_BONF, "n_cum": N_CUM}, (out / "result.json").open("w"), default=float)  # fmt: skip
    keep = list(dict.fromkeys([fin] + hi))[:6]
    json.dump({n: {w: trades[n][w] for w in WINS} for n in keep}, (out / "trades.json").open("w"))
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
