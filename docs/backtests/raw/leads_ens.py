"""단서 모음(앙상블): NR4 일·12h·8h, NR7 일, DC24h 다리 × 문 GT/ZP/GZ, 공통 청산 틀, 지정가 진입 안. 20개.

등록: docs/backtests/2026-11-08-btc-leads-ensemble.md. 연구 전용(주문 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/leads-ens-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/leads_ens.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import daily_wave as DW
import numpy as np
import papers_swing as PS
import pine_ports as P
import ql_upgrade as U
import regime_lines as RL
import rr_asym as RA
from dmi15_eval import frame
from hiwin import norm_p, summ
from htf_live import to15

MIN, M15, DAY = U.MIN, U.M15, U.DAY
TAKER, MAKER, TICK, SLIP = U.TAKER, U.MAKER, U.TICK, 0.0001
N_BONF, N_BEFORE = 20, 6480
N_CUM = N_BEFORE + N_BONF
MAINW = ("fresh", "design", "val", "year")
WINS = MAINW + ("m6", "d30", "pool")
NR = ("N4D", "N4H12", "N4H8")


def V(legs, gate="GT", mode=None, q=0.22, H=72, sz=False):
    return {"legs": legs, "gate": gate, "mode": mode, "q": q, "H": H, "sz": sz}


SPEC = {
    "N4D·GT": V(["N4D"]), "N7D·GT": V(["N7D"]), "N4H12·GT": V(["N4H12"]), "N4H8·GT": V(["N4H8"]),
    "N4D·ZP": V(["N4D"], "ZP"), "N4D·GZ": V(["N4D"], "GZ"), "DC·GT": V(["DC"]),
    "EN": V(list(NR)), "END": V([*NR, "DC"]), "EN·GZ": V(list(NR), "GZ"), "END·GZ": V([*NR, "DC"], "GZ"),
    "EN·ZP": V(list(NR), "ZP"), "END·q15": V([*NR, "DC"], q=0.15), "END·q30": V([*NR, "DC"], q=0.30),
    "END·H24": V([*NR, "DC"], H=24), "END·SZ": V([*NR, "DC"], sz=True),
    "EN·M3": V(list(NR), mode=(0.003, 4)), "END·M3": V([*NR, "DC"], mode=(0.003, 4)),
    "END·M5": V([*NR, "DC"], mode=(0.005, 8)), "END·GZ·M3": V([*NR, "DC"], "GZ", (0.003, 4)),
}  # fmt: skip
assert len(SPEC) == N_BONF


# ---------------- 신호 ----------------
def nr_signals(b, m, minutes, N) -> list:
    """NR-N 봉 → 다음 같은 길이 봉 동안 고·저 첫 돌파. 반환 [(i15, d, 수준, 돌파 분 k, 문 볼 i_end)]."""
    Fr = frame(b, minutes)
    rng = Fr["h"] - Fr["l"]
    t1 = m["t"]
    step = minutes * MIN
    out = []
    for j in range(N - 1, len(rng)):
        if rng[j] > rng[j - N + 1 : j + 1].min():
            continue
        i_end = int(Fr["last15"][j])
        a = int(b["t"][i_end]) + M15
        ka, kz = np.searchsorted(t1, a), np.searchsorted(t1, a + step)
        if kz - ka < 0.4 * minutes:
            continue
        hi, lo = Fr["h"][j], Fr["l"][j]
        up, dn = m["h"][ka:kz] >= hi, m["l"][ka:kz] <= lo
        ku = int(np.argmax(up)) if up.any() else None
        kd = int(np.argmax(dn)) if dn.any() else None
        if (ku is None and kd is None) or (ku is not None and ku == kd):
            continue
        if kd is None or (ku is not None and ku < kd):
            k, d, lvl = ka + ku, 1, hi
        else:
            k, d, lvl = ka + kd, -1, lo
        i15 = int(np.searchsorted(b["t"], t1[k], "right")) - 2
        out.append((i15, d, float(lvl), int(k), i_end))
    return out


def dc_signals(b) -> list:
    L, S = RA.donchian_sig(b, 96)
    i = np.r_[np.nonzero(L)[0], np.nonzero(S)[0]]
    d = np.r_[np.ones(L.sum(), int), -np.ones(S.sum(), int)]
    o = np.argsort(i, kind="stable")
    return [(int(a), int(z), float(b["c"][a]), None, int(a)) for a, z in zip(i[o], d[o])]


def zpb_gate(b, m) -> np.ndarray:
    D = PS.build(b, m)["D"]
    s = PS.zpb(D, False, False)  # 날 j 목표(날 j−1 마감까지)
    day = (b["t"] // DAY) * DAY
    j = np.searchsorted(D["T"], day)
    j = np.clip(j, 0, len(s) - 1)
    ok = D["T"][j] == day
    return np.where(ok, np.sign(s[j]), 0.0)


def all_down(b) -> np.ndarray:
    n = len(b["c"])

    def dn(Fr):
        hlc3 = (Fr["h"] + Fr["l"] + Fr["c"]) / 3
        with np.errstate(invalid="ignore"):
            return ((P.cci(hlc3, 20) < 0) & (P.rsi(Fr["c"], 14) < 50)).astype(float)

    D, F4 = frame(b, 1440), frame(b, 240)
    x15 = dn({"h": b["h"], "l": b["l"], "c": b["c"]})
    xd = np.nan_to_num(to15(D, dn(D), n), nan=0.0)
    x4 = np.nan_to_num(to15(F4, dn(F4), n), nan=0.0)
    return (x15 > 0) & (xd > 0) & (x4 > 0)


# ---------------- 체결 ----------------
class Sim:
    def __init__(self, m, b, F, gates, fund_fn):
        self.m, self.b, self.F, self.gates, self.fund_fn = m, b, F, gates, fund_fn
        self.n1 = len(m["t"])
        self.atrp = F["atr_d"] / b["c"]
        n = len(b["t"])
        self.nxt = {}
        for g, arr in gates.items():
            for dv in (1, -1):
                pos = np.where(arr != dv, np.arange(n), n)
                self.nxt[(g, dv)] = np.minimum.accumulate(pos[::-1])[::-1]

    def kmin(self, i15, lat=0):
        ts = int(self.b["t"][i15]) + M15 + lat * MIN
        k = int(np.searchsorted(self.m["t"], ts))
        if k >= self.n1 - 2 or int(self.m["t"][k]) - ts > 5 * MIN:
            return -1
        return k

    def geom(self, i15, kq, mult=1.0):
        a = float(self.atrp[i15]) if np.isfinite(self.atrp[i15]) else 0.03
        return min(max(a, 0.01), 0.035), max(kq * a * mult, 0.004)

    def limit_fill(self, k_s, d, L, hours):
        if k_s < 0 or k_s >= self.n1 - 2:
            return None
        kz = min(k_s + hours * 60, self.n1 - 2)
        hit = (self.m["l"][k_s:kz] < L) if d == 1 else (self.m["h"][k_s:kz] > L)
        if not hit.any():
            return None
        return k_s + int(np.argmax(hit))

    def entry(self, rec, mode, lat=0):
        """반환 (k, ep, k0 = 익절 보기 시작 분, maker) 또는 None."""
        i15, d, sp, kk = rec[0], rec[1], rec[2], rec[3]
        if mode is not None:
            x, hours = mode
            k_s = self.kmin(i15, lat) if kk is None else kk + 1 + lat
            L = sp * (1 - d * x)
            k = self.limit_fill(k_s, d, L, hours)
            if k is None:
                return None
            return k, L, k + 1, True
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
            ep = o if lat else (max(sp, o) if d == 1 else min(sp, o))
            k0 = k + 1
        lim = 0.03 if self.F["agree"][i15] else 0.01
        if abs(ep / sp - 1) >= lim:
            return None
        return k, ep, k0, False

    def trade(self, k, d, ep, k0, S, T, gate, i15, H, maker=False, fee_mult=1.0, slip=SLIP):
        m = self.m
        stop, tp = ep * (1 - d * S), ep * (1 + d * T)
        tend = min(k + H * 60, self.n1 - 1)
        j = int(self.nxt[(gate, d)][min(i15 + 1, len(self.b["t"]) - 1)])
        if j < len(self.b["t"]):
            kb = self.kmin(j)
            if kb > k and kb - 1 < tend:
                tend = kb - 1
        hit = U.first_hit(m, d, k, tend, stop, tp, k0) if tend >= k else None
        if hit is None:
            j, why, px = tend, "time", float(m["c"][tend])
        else:
            j, why = hit
            if why == "stop":
                o = float(m["o"][j])
                px = (min(stop, o) if d == 1 else max(stop, o)) - d * TICK
            else:
                px = tp
        fin = MAKER * fee_mult if maker else TAKER * fee_mult + slip
        fout = MAKER * fee_mult if why == "tp" else TAKER * fee_mult + slip
        h, lo = m["h"][k : j + 1], m["l"][k : j + 1]
        worst = (lo.min() / ep - 1) if d == 1 else (1 - h.max() / ep)
        t_in, t_out = int(m["t"][k]), int(m["t"][j]) + MIN
        r = d * (px / ep - 1) - fin - fout + self.fund_fn(t_in, t_out, d)
        return {"k": int(k), "x": int(j), "d": int(d), "ep": float(ep), "xp": float(px), "r": float(r), "S": S,
                "T": T, "worst": float(min(worst, r)), "why": why, "size": 1.0, "t_in": t_in, "t_out": t_out}  # fmt: skip


# ---------------- 안 실행 ----------------
class Runner:
    def __init__(self, sim, sigs, sz):
        self.sim, self.sigs, self.sz = sim, sigs, sz
        self.kq = {}

    def recs(self, v):
        g = self.sim.gates[v["gate"]]
        out = []
        for leg in v["legs"]:
            for r in self.sigs[leg]:
                if g[r[4]] == r[1]:
                    out.append((*r, leg))
        return out

    def mfe_k(self, v, leg, a_ms, z_ms):
        key = (leg, v["gate"], v["mode"], v["q"], v["H"])
        if key in self.kq:
            return self.kq[key]
        sim, m, t = self.sim, self.sim.m, self.sim.b["t"]
        H = v["H"] * 60
        vals = []
        for rec in self.recs(v | {"legs": [leg]}):
            i15, d = rec[0], rec[1]
            if not (a_ms <= t[i15] and t[i15] + M15 + H * MIN < z_ms):
                continue
            e = sim.entry(rec, v["mode"])
            if e is None:
                continue
            k, ep, k0, _ = e
            a = float(sim.atrp[i15])
            if not np.isfinite(a):
                continue
            S = min(max(a, 0.01), 0.035)
            h, lo = m["h"][k : k + H], m["l"][k : k + H]
            adv = (1 - lo / ep) if d == 1 else (h / ep - 1)
            fav = (h / ep - 1) if d == 1 else (1 - lo / ep)
            js = np.nonzero(adv >= S)[0]
            end = js[0] if len(js) else len(fav)
            mf = fav[:end].max() if end > 0 else 0.0
            vals.append(max(mf, 0.0) / a)
        self.kq[key] = float(np.quantile(vals, v["q"])) if len(vals) >= 10 else 0.25
        return self.kq[key]

    def run(self, v, a_ms, z_ms, lat=0, fee_mult=1.0, slip=SLIP):
        sim, t = self.sim, self.sim.b["t"]
        cands = []
        for rec in self.recs(v):
            i15 = rec[0]
            if not (a_ms <= t[i15] and t[i15] + M15 < z_ms):
                continue
            e = sim.entry(rec, v["mode"], lat)
            if e is not None:
                cands.append((e, rec))
        cands.sort(key=lambda z: z[0][0])
        out, busy = [], -1
        for (k, ep, k0, maker), rec in cands:
            if k <= busy:
                continue
            i15, d, leg = rec[0], rec[1], rec[5]
            mult = 1.5 if v["sz"] and self.sz[i15] else 1.0
            S, T = sim.geom(i15, self.kq_leg[leg], mult)
            x = sim.trade(k, d, ep, k0, S, T, v["gate"], i15, v["H"], maker, fee_mult, slip)
            if x["t_out"] > z_ms:
                continue
            x.update(i15=int(i15), leg=leg, gate=v["gate"], H=v["H"], mode=v["mode"])
            out.append(x)
            busy = x["x"]
        return out


def random_gate(sim, tr, nsim=200, seed=7) -> np.ndarray:
    """문 맞춘 같은 기하 무작위: ±15일 안 그 거래 문이 같은 방향인 15분봉, 같은 S·T·H·무효화·진입 방식."""
    rng = np.random.default_rng(seed)
    t, c = sim.b["t"], sim.b["c"]
    out = np.full((len(tr), nsim), np.nan)
    last_ok = int(np.searchsorted(t, int(t[-1]) - 96 * 60 * MIN - DAY))
    for a, x in enumerate(tr):
        g = sim.gates[x["gate"]]
        lo, hi = max(x["i15"] - 15 * 96, 200), min(x["i15"] + 15 * 96, last_ok)
        pool = np.arange(lo, hi)
        pool = pool[g[pool] == x["d"]]
        if not len(pool):
            continue
        for s in range(nsim):
            for _ in range(20):
                i = int(rng.choice(pool))
                k0 = sim.kmin(i)
                if k0 < 0:
                    continue
                if x["mode"] is None:
                    e = (k0, float(sim.m["o"][k0]), k0, False)
                else:
                    xx, hours = x["mode"]
                    L = float(c[i]) * (1 - x["d"] * xx)
                    kf = sim.limit_fill(k0, x["d"], L, hours)
                    if kf is None:
                        continue
                    e = (kf, L, kf + 1, True)
                break
            else:
                continue
            k, ep, kk0, mk = e
            out[a, s] = sim.trade(k, x["d"], ep, kk0, x["S"], x["T"], x["gate"], i, x["H"], mk)["r"]
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
            "mean": float(r.mean()), "rnd_mean": float(rm.mean()), "ex_mean": float(r.mean() - rm.mean()),
            "z_mean": float(zm), "p_mean": norm_p(zm),
            "p_mean_emp": float((np.sum(rm >= r.mean()) + 1) / (len(rm) + 1))}  # fmt: skip


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda z: z[1])
    mtot, run, out = len(items), 0.0, {}
    for i, (n, p) in enumerate(items):
        run = max(run, min(1.0, (mtot - i) * p))
        out[n] = run
    return out


# ---------------- 실행 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=200)
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

    gt = F["gate"]["GT"]
    zp = zpb_gate(b, m)
    gates = {"GT": gt, "ZP": zp, "GZ": np.where(gt == zp, gt, 0.0)}
    sim = Sim(m, b, F, gates, fund_fn)
    W = RA.windows(t_end)
    sigs = {"N4D": nr_signals(b, m, 1440, 4), "N7D": nr_signals(b, m, 1440, 7), "N4H12": nr_signals(b, m, 720, 4),
            "N4H8": nr_signals(b, m, 480, 4), "DC": dc_signals(b)}  # fmt: skip
    sz = all_down(b)
    print("loaded", {k: len(v) for k, v in sigs.items()}, "zp+", float((zp > 0).mean()), "zp-",
          float((zp < 0).mean()), "gz0", float((gates["GZ"] == 0).mean()), "alldown", float(sz.mean()),
          round(time.time() - t0), flush=True)  # fmt: skip
    R = Runner(sim, sigs, sz)

    def prep(v):
        R.kq_leg = {leg: R.mfe_k(v, leg, *W["design"]) for leg in v["legs"]}
        return dict(R.kq_leg)

    res, trades, kqs = {}, {}, {}
    for name, v in SPEC.items():
        kqs[name] = prep(v)
        res[name], trades[name] = {}, {}
        for w in WINS:
            tr = R.run(v, *W[w])
            res[name][w] = summ(tr, *W[w])
            trades[name][w] = tr
        r = res[name]
        print(name, {k: round(x, 3) for k, x in kqs[name].items()},
              {w: (r[w]["n"], round(r[w]["per_day"], 2), r[w]["win"] and round(r[w]["win"], 3),
                   round(r[w]["x5_f0.4"]["ret"], 2), round(r[w]["x5_f1.0"]["ret"], 2)) for w in MAINW + ("d30",)},
              round(time.time() - t0), flush=True)  # fmt: skip

    def pick(freq=True):
        ok = [n for n in SPEC if res[n]["design"]["n"] >= 60 and (res[n]["design"]["win"] or 0) >= 0.70
              and (not freq or 0.25 <= res[n]["design"]["per_day"] <= 0.6)]  # fmt: skip
        if not ok:
            return None
        top = max(res[n]["design"]["x5_f1.0"]["ret"] for n in ok)
        near = [n for n in ok if res[n]["design"]["x5_f1.0"]["ret"] >= top - 0.10 * abs(top)]
        return min(near, key=lambda n: res[n]["design"]["x5_f1.0"]["mdd"])

    fin = pick()
    relaxed = fin is None
    if relaxed:
        fin = pick(False) or max(SPEC, key=lambda n: res[n]["design"]["x5_f1.0"]["ret"])
    print("PICK", fin, "relaxed" if relaxed else "", flush=True)

    # 무작위: 20개 모두(가족 보정에 필요)
    rnd, full = {}, {}
    for name, v in SPEC.items():
        prep(v)
        tr = R.run(v, W["fresh"][0], t_end)
        full[name] = tr
        Rm = random_gate(sim, tr, args.nsim, 7)
        tin = np.array([x["t_in"] for x in tr])
        rnd[name] = {}
        for w in MAINW + ("pool",):
            sel = (tin >= W[w][0]) & (tin < W[w][1])
            rnd[name][w] = rnd_stats([x for x, z in zip(tr, sel) if z], Rm[sel])
        print("rnd", name, {w: (round(q.get("excess", 0), 3), round(q.get("ex_mean", 0) * 100, 3),
                                round(q.get("p_win", 1), 4), round(q.get("p_mean", 1), 4))
                            for w, q in rnd[name].items()}, round(time.time() - t0), flush=True)  # fmt: skip
    pm = {n: rnd[n]["pool"].get("p_mean", 1.0) for n in SPEC}
    ph = holm(pm)

    hi = [n for n in SPEC if all((res[n][w]["win"] or 0) >= 0.70 for w in MAINW)]
    cand = list(dict.fromkeys(hi + [fin]))
    red, years, verdict = {}, {}, {}
    for n in cand:
        v = SPEC[n]
        prep(v)
        red[n] = {}
        for nm, kw in (("lat1m", {"lat": 1}), ("lat15m", {"lat": 15}), ("slip+0.02", {"slip": SLIP + 0.0002}),
                       ("fee1.5", {"fee_mult": 1.5})):  # fmt: skip
            red[n][nm] = {w: summ(R.run(v, *W[w], **kw), *W[w]) for w in ("val", "year")}
        years[n] = {}
        for y in range(2020, 2027):
            a_, z_ = max(RA.ms(f"{y}-01-01"), W["fresh"][0]), min(RA.ms(f"{y + 1}-01-01"), t_end)
            years[n][y] = summ(R.run(v, a_, z_), a_, z_)
        fx, rp = res[n], rnd[n]["pool"]
        g1 = all((fx[w]["win"] or 0) >= 0.70 for w in MAINW)
        g2 = all(fx[w][f"x5_f{f}"]["ret"] > 0 for w in MAINW for f in (1.0, 0.4)) and all(
            fx[w]["x5_f1.0"]["liq"] == 0 for w in MAINW)
        g3b = ph[n] < 0.05 and all(rnd[n][w].get("excess", -1) > 0 and rnd[n][w].get("ex_mean", -1) > 0
                                   for w in MAINW)  # fmt: skip
        g3a = bool(rp) and rp["p_mean"] * N_CUM < 0.05
        g4 = all(0.25 <= fx[w]["per_day"] <= 0.6 for w in ("design", "year"))
        g5 = fx["pool"]["n"] > 0 and min(fx["pool"]["nl"], fx["pool"]["ns"]) >= 0.15 * fx["pool"]["n"]
        g6 = all(red[n][k][w]["x5_f1.0"]["ret"] > 0 for k in red[n] for w in ("val", "year"))
        verdict[n] = {"g1_win70": g1, "g2_ret_liq": g2, "g3b_family": bool(g3b), "g3a_cum": bool(g3a),
                      "g4_freq": g4, "g5_sides": g5, "g6_red": g6, "holm_p": ph[n],
                      "pass": bool(g1 and g2 and g3b and g4 and g5 and g6)}  # fmt: skip
        print("VERDICT", n, verdict[n], {k: {w: round(q["x5_f1.0"]["ret"], 2) for w, q in red[n][k].items()}
                                         for k in red[n]},
              {y: (q["n"], round(q["win"] or 0, 2), round(q["x5_f0.4"]["ret"], 2), round(q["x5_f1.0"]["ret"], 2))
               for y, q in years[n].items()}, flush=True)  # fmt: skip

    n_win = {w: sum((res[n][w]["win"] or 0) > 0.70 for n in SPEC) for w in MAINW}
    json.dump({"windows": W, "spec": SPEC, "kq": kqs, "res": res, "final": fin, "relaxed": relaxed, "hi": hi,
               "rnd": rnd, "p_mean_pool": pm, "holm": ph, "red": red, "years": years, "verdict": verdict,
               "n_win70": n_win, "n_bonf": N_BONF, "n_cum": N_CUM}, (out / "result.json").open("w"), default=float)  # fmt: skip
    keep = list(dict.fromkeys([fin] + hi))[:8]
    json.dump({n: {w: trades[n][w] for w in WINS} for n in keep}, (out / "trades.json").open("w"), default=float)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
