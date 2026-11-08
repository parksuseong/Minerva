"""BTC 더 파기: A 변동성 확장 양방향 돌파(OCO, 1분봉 체결) 96개 + B metrics·테이커 상태 신호 12개.

등록: docs/backtests/2026-11-08-btc-deep.md. 연구 전용(주문 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/deep-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/deep.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import daily_wave as DW
import numpy as np
import ql_upgrade as U
import rr_asym as A
from dmi15_eval import frame
from htf_live import direction, to15
from numpy.lib.stride_tricks import sliding_window_view
from r5_data import load

MIN, M15, DAY = U.MIN, U.M15, U.DAY
TAKER, MAKER, TICK, SLIP = U.TAKER, U.MAKER, U.TICK, 0.0001
N_BEFORE, N_THIS = 6324, 108
N_CUM = N_BEFORE + N_THIS
PEND = 240  # 브래킷 대기 4시간(분)
MAXH = 48 * 60
MAINW = ("fresh", "design", "val", "year")
SHOWW = MAINW + ("m6", "d30", "dv")
VS = ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7")
RS = {"R2h": 8, "R8h": 32}
SS = ("SM", "SO")
XS = ("X2R", "XTR", "X24")
BS = ("B1", "B2", "B3", "B4", "B5", "B6")
BH = {"H4": 240, "H24": 1440}
METRIC_B = ("B1", "B2", "B3", "B6")


# ---------------- 공통 ----------------
def trailing_q(x: np.ndarray, win: int, q: float, chunk: int = 4000) -> np.ndarray:
    """x[j]에 대해 x[j-win:j](현재 제외)의 q 분위. NaN 무시."""
    n = len(x)
    out = np.full(n, np.nan)
    if n <= win:
        return out
    V = sliding_window_view(x[:-1], win)  # V[s] = x[s:s+win] → 대상 j = s+win
    for s0 in range(0, len(V), chunk):
        blk = V[s0 : s0 + chunk]
        with np.errstate(all="ignore"):
            out[s0 + win : s0 + win + len(blk)] = np.nanquantile(blk, q, axis=1)
    return out


def rmean(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def rstd(x: np.ndarray, n: int) -> np.ndarray:
    m = rmean(x, n)
    m2 = rmean(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0))


class Sim:
    def __init__(self, m, b, fund_fn):
        self.m, self.b, self.fund_fn = m, b, fund_fn
        self.n1 = len(m["t"])

    def kmin(self, i15, lat=0):
        ts = int(self.b["t"][i15]) + M15 + lat * MIN
        k = int(np.searchsorted(self.m["t"], ts))
        if k >= self.n1 - 2 or int(self.m["t"][k]) - ts > 5 * MIN:
            return -1
        return k

    def exit_(self, k, d, ep, stop, mode, hold, fee_in, fee_mult, slip):
        """체결 분 k부터. mode: X2R(2R 목표)·XTR(+1R 뒤 1R 추적)·T(시간, 손절만). 체결 분의 손절은 같은 분에 친다."""
        m = self.m
        R = abs(ep - stop)
        tend = min(k + hold, self.n1 - 1)
        h, lo = m["h"][k : tend + 1], m["l"][k : tend + 1]
        if mode == "XTR":
            fav = np.maximum.accumulate(h) if d == 1 else np.minimum.accumulate(lo)
            prev = np.r_[ep, fav[:-1]]
            prev = np.maximum(prev, ep) if d == 1 else np.minimum(prev, ep)
            armed = (prev >= ep + R) if d == 1 else (prev <= ep - R)
            tr_ = prev - d * R
            lvl = np.where(armed, np.maximum(tr_, stop) if d == 1 else np.minimum(tr_, stop), stop)
        else:
            lvl = np.full(len(h), stop)
        sh = (lo <= lvl) if d == 1 else (h >= lvl)
        tp = ep + d * 2 * R if mode == "X2R" else None
        if tp is not None:
            th = (h >= tp) if d == 1 else (lo <= tp)
            th[0] = False
        else:
            th = np.zeros(len(h), bool)
        ks = int(np.argmax(sh)) if sh.any() else None
        kt = int(np.argmax(th)) if th.any() else None
        if ks is None and kt is None:
            jj, why, px = len(h) - 1, "time", float(m["c"][tend])
        elif kt is None or (ks is not None and ks <= kt):
            jj, why = ks, "stop"
            o = float(m["o"][k + jj]) if jj > 0 else ep
            lv = float(lvl[jj])
            px = (min(lv, o) if d == 1 else max(lv, o)) - d * TICK
        else:
            jj, why, px = kt, "tp", tp
        fout = MAKER * fee_mult if why == "tp" else TAKER * fee_mult + slip
        worst = (lo[: jj + 1].min() / ep - 1) if d == 1 else (1 - h[: jj + 1].max() / ep)
        t_in, t_out = int(m["t"][k]), int(m["t"][k + jj]) + MIN
        r = d * (px / ep - 1) - fee_in - fout + self.fund_fn(t_in, t_out, d)
        return {"k": int(k), "x": int(k + jj), "d": int(d), "ep": float(ep), "xp": float(px), "r": float(r),
                "R": float(r / (R / ep)) if R > 0 else 0.0, "worst": float(min(worst, r)), "why": why,
                "size": 1.0, "t_in": t_in, "t_out": t_out}  # fmt: skip

    # ---------- A: 브래킷 ----------
    def bracket(self, i, N, S, X, lat=0, fee_mult=1.0, slip=SLIP):
        """반환 (거래 또는 None, 바쁜 끝 분)."""
        b, m = self.b, self.m
        if i - N + 1 < 0:
            return None, -1
        hi, lo = float(b["h"][i - N + 1 : i + 1].max()), float(b["l"][i - N + 1 : i + 1].min())
        mid = (hi + lo) / 2
        half = (hi - lo) / 2 / mid
        if half > 0.03:
            return None, -1
        if half < 0.003:
            hi, lo = mid * 1.003, mid * 0.997
        k0 = self.kmin(i, lat)
        if k0 < 0:
            return None, -1
        k1 = min(k0 + PEND, self.n1 - 1)
        bt, st = hi + TICK, lo - TICK
        uh = m["h"][k0:k1] >= bt
        dl = m["l"][k0:k1] <= st
        ku = int(np.argmax(uh)) if uh.any() else None
        kd = int(np.argmax(dl)) if dl.any() else None
        if ku is None and kd is None:
            return None, k1
        fee_in = TAKER * fee_mult + slip
        stops = {1: mid if S == "SM" else st, -1: mid if S == "SM" else bt}
        if ku is not None and kd is not None and ku == kd:  # 같은 분 양쪽: 손실 큰 쪽 체결 후 그 분 손절
            k = k0 + ku
            o = float(m["o"][k])
            best = None
            for d, trg in ((1, bt), (-1, st)):
                ep = (max(trg, o) if d == 1 else min(trg, o)) + d * TICK
                px = stops[d] - d * TICK
                r = d * (px / ep - 1) - fee_in - (TAKER * fee_mult + slip)
                if best is None or r < best[0]:
                    best = (r, d, ep, px)
            r, d, ep, px = best
            t_in = int(m["t"][k])
            x = {"k": k, "x": k, "d": d, "ep": ep, "xp": px, "r": float(r), "R": -1.0, "worst": float(r),
                 "why": "both", "size": 1.0, "t_in": t_in, "t_out": t_in + MIN}  # fmt: skip
            return x, k
        if kd is None or (ku is not None and ku < kd):
            d, k, trg = 1, k0 + ku, bt
        else:
            d, k, trg = -1, k0 + kd, st
        o = float(m["o"][k])
        ep = (max(trg, o) if d == 1 else min(trg, o)) + d * TICK
        if abs(ep / trg - 1) >= 0.01:  # 추격 한도(스톱 지정가)
            return None, k
        stop = stops[d]
        if (stop - ep) * d >= 0:
            return None, k
        mode = "T" if X == "X24" else X
        hold = 1440 if X == "X24" else MAXH
        x = self.exit_(k, d, ep, stop, mode, hold, fee_in, fee_mult, slip)
        return x, x["x"]

    def run_A(self, fires, N, S, X, a_ms, z_ms, **kw):
        t = self.b["t"]
        out, busy = [], -1
        for i in fires:
            if t[i] < a_ms or t[i] + M15 >= z_ms:
                continue
            k0 = self.kmin(int(i), kw.get("lat", 0))
            if k0 < 0 or k0 <= busy:
                continue
            x, end = self.bracket(int(i), N, S, X, **kw)
            if end >= 0:
                busy = end
            if x is None or x["t_out"] > z_ms:
                continue
            x["i15"] = int(i)
            out.append(x)
        return out

    # ---------- B: 시장가 ----------
    def market(self, i, d, H, chase_lim, lat=0, fee_mult=1.0, slip=SLIP):
        k = self.kmin(i, lat)
        if k < 0:
            return None, -1
        ep = float(self.m["o"][k]) + d * TICK
        if abs(ep / float(self.b["c"][i]) - 1) >= chase_lim:
            return None, -1
        stop = ep * (1 - d * 0.02)
        x = self.exit_(k, d, ep, stop, "T", H, TAKER * fee_mult + slip, fee_mult, slip)
        return x, x["x"]

    def run_B(self, sig, H, a_ms, z_ms, chase, **kw):
        i_, d_ = sig
        t = self.b["t"]
        out, busy = [], -1
        for i, d in zip(i_, d_):
            if t[i] < a_ms or t[i] + M15 >= z_ms:
                continue
            k = self.kmin(int(i), kw.get("lat", 0))
            if k < 0 or k <= busy:
                continue
            x, end = self.market(int(i), int(d), H, chase[i] if d == 1 else chase[i + len(t)], **kw)
            if x is None:
                continue
            busy = end
            if x["t_out"] > z_ms:
                continue
            x["i15"] = int(i)
            out.append(x)
        return out


# ---------------- 특징·신호 ----------------
def build(b) -> dict:
    n = len(b["c"])
    t, h, lo, c = b["t"], b["h"], b["l"], b["c"]
    F1, F4, D = frame(b, 60), frame(b, 240), frame(b, 1440)
    d15 = direction(h, lo, c).astype(float)
    d4 = np.nan_to_num(to15(F4, direction(F4["h"], F4["l"], F4["c"]).astype(float), n), nan=0.0)
    d1 = np.nan_to_num(to15(D, direction(D["h"], D["l"], D["c"]).astype(float), n), nan=0.0)
    fires = {}
    fires["V0"] = F4["last15"].copy()
    r4 = F4["h"] - F4["l"]
    nr7 = np.zeros(len(r4), bool)
    nr7[6:] = r4[6:] <= sliding_window_view(r4, 7).min(1)
    fires["V1"] = F4["last15"][nr7]
    r1 = F1["h"] - F1["l"]
    nr4 = np.zeros(len(r1), bool)
    nr4[3:] = r1[3:] <= sliding_window_view(r1, 4).min(1)
    fires["V2"] = F1["last15"][nr4]
    c1 = F1["c"]
    bw = 4 * rstd(c1, 20) / rmean(c1, 20)
    with np.errstate(invalid="ignore"):
        fires["V3"] = F1["last15"][bw <= trailing_q(bw, 720, 0.10)]
    pc = np.r_[c1[0], c1[:-1]]
    trr = np.maximum(F1["h"] - F1["l"], np.maximum(abs(F1["h"] - pc), abs(F1["l"] - pc)))
    ar = rmean(trr, 6) / rmean(trr, 48)
    with np.errstate(invalid="ignore"):
        fires["V4"] = F1["last15"][ar <= trailing_q(ar, 720, 0.10)]
    dn3 = (d15 == -1) & (d4 == -1) & (d1 == -1)
    fires["V5"] = np.nonzero(dn3 & ~np.r_[False, dn3[:-1]])[0]
    ib = np.zeros(len(r4), bool)
    ib[1:] = (F4["h"][1:] <= F4["h"][:-1]) & (F4["l"][1:] >= F4["l"][:-1])
    fires["V6"] = F4["last15"][ib]
    fires["V7"] = np.unique(np.concatenate([fires[v] for v in ("V1", "V2", "V3", "V4", "V5", "V6")]))
    fires = {k: np.sort(v.astype(np.int64)) for k, v in fires.items()}

    # B 특징
    W = A.windows(int(t[-1]) + M15)
    dz = (t >= W["design"][0]) & (t < W["design"][1])
    r1b = np.r_[np.full(4, np.nan), c[4:] / c[:-4] - 1]
    r4b = np.r_[np.full(16, np.nan), c[16:] / c[:-16] - 1]
    v, tb = b["v"].astype(float), b["tbv"].astype(float)
    with np.errstate(all="ignore"):
        tbS1 = rmean(tb, 4) / rmean(v, 4)
        tbS4 = rmean(tb, 16) / rmean(v, 16)
    oi1, oi4, zC = b["f_oi1"], b["f_oi4"], b["f_zC"]

    def q(x, p):
        return float(np.nanquantile(x[dz], p))

    th = {"r4q90": q(r4b, 0.9), "r4q10": q(r4b, 0.1), "oi4q70": q(oi4, 0.7), "oi4q30": q(oi4, 0.3),
          "tb1q90": q(tbS1, 0.9), "tb1q10": q(tbS1, 0.1), "tb4q95": q(tbS4, 0.95), "tb4q05": q(tbS4, 0.05),
          "r1q95": q(r1b, 0.95), "r1q05": q(r1b, 0.05), "oi1q03": q(oi1, 0.03)}  # fmt: skip
    with np.errstate(invalid="ignore"):
        cond = {
            "B1": (r4b >= th["r4q90"]) & (oi4 >= th["oi4q70"]), "B1s": (r4b <= th["r4q10"]) & (oi4 >= th["oi4q70"]),
            "B2": (r4b <= th["r4q10"]) & (oi4 <= th["oi4q30"]), "B2s": (r4b >= th["r4q90"]) & (oi4 <= th["oi4q30"]),
            "B3": zC <= -1.5, "B3s": zC >= 1.5,
            "B4": tbS1 >= th["tb1q90"], "B4s": tbS1 <= th["tb1q10"],
            "B5": (tbS4 <= th["tb4q05"]) & (r1b >= 0), "B5s": (tbS4 >= th["tb4q95"]) & (r1b <= 0),
            "B6": (r1b <= th["r1q05"]) & (oi1 <= th["oi1q03"]), "B6s": (r1b >= th["r1q95"]) & (oi1 <= th["oi1q03"]),
        }  # fmt: skip
    sigs = {}
    for k in BS:
        ii, dd = [], []
        for key, d in ((k, 1), (k + "s", -1)):
            x = np.nan_to_num(cond[key].astype(float), nan=0.0).astype(bool)
            on = np.nonzero(x & ~np.r_[False, x[:-1]])[0]
            ii.append(on)
            dd.append(np.full(len(on), d))
        ii, dd = np.concatenate(ii), np.concatenate(dd)
        o = np.argsort(ii, kind="stable")
        sigs[k] = (ii[o], dd[o])
    agree_l = (d15 == 1) & (d4 == 1) & (d1 == 1)
    agree_s = (d15 == -1) & (d4 == -1) & (d1 == -1)
    chase = np.r_[np.where(agree_l, 0.03, 0.01), np.where(agree_s, 0.03, 0.01)]
    return {"fires": fires, "sigs": sigs, "th": th, "chase": chase}


# ---------------- 요약·무작위 ----------------
def slice_w(tr, a, z):
    return [x for x in tr if a <= x["t_in"] and x["t_out"] <= z]


def summ(tr, a, z) -> dict:
    s = U.summ(tr, a, z)
    for d, nm in ((1, "L"), (-1, "S")):
        r = np.array([x["r"] for x in tr if x["d"] == d])
        s[f"win{nm}"] = float((r > 0).mean()) if len(r) else None
        s[f"x1{nm}"] = float(np.prod(1 + r) - 1) if len(r) else 0.0
    s["why"] = {w: int(sum(x["why"] == w for x in tr)) for w in ("stop", "tp", "time", "both")}
    return s


def slot_pool(b, a, z, pad):
    t = b["t"]
    ia, iz = int(np.searchsorted(t, a)), int(np.searchsorted(t, z - pad))
    idx = np.arange(max(ia, 40), iz)
    sl = (t[idx] % DAY).astype(np.int64)
    return {int(s): idx[sl == s] for s in np.unique(sl)}


def zp(real, vals):
    sd = float(np.std(vals))
    z = (real - float(np.mean(vals))) / sd if sd > 0 else 0.0
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def random_A(sim, tr, N, S, X, a, z, nsim, seed):
    rng = np.random.default_rng(seed)
    pool = slot_pool(sim.b, a, z, (MAXH + PEND) * MIN + M15)
    t = sim.b["t"]
    slots = [int(t[x["i15"]] % DAY) for x in tr]
    vals = []
    for _ in range(nsim):
        rs = []
        for s in slots:
            p = pool.get(s)
            for _try in range(30):
                x, _ = sim.bracket(int(rng.choice(p)), N, S, X)
                if x is not None:
                    rs.append(x["r"])
                    break
        vals.append(np.mean(rs))
    real = float(np.mean([x["r"] for x in tr]))
    zz, pz = zp(real, vals)
    return {"real": real, "rnd_mean": float(np.mean(vals)), "rnd_sd": float(np.std(vals)), "z": zz, "p_norm": pz,
            "p_emp": float((np.sum(np.array(vals) >= real) + 1) / (nsim + 1)), "n": len(tr)}  # fmt: skip


def random_B(sim, tr, H, a, z, nsim, seed):
    rng = np.random.default_rng(seed)
    pool = slot_pool(sim.b, a, z, (H + 60) * MIN + M15)
    t = sim.b["t"]
    spec = [(int(t[x["i15"]] % DAY), x["d"]) for x in tr]
    vals = []
    for _ in range(nsim):
        rs = []
        for s, d in spec:
            for _try in range(10):
                x, _ = sim.market(int(rng.choice(pool[s])), d, H, 1.0)
                if x is not None:
                    rs.append(x["r"])
                    break
        vals.append(np.mean(rs))
    real = float(np.mean([x["r"] for x in tr]))
    zz, pz = zp(real, vals)
    return {"real": real, "rnd_mean": float(np.mean(vals)), "rnd_sd": float(np.std(vals)), "z": zz, "p_norm": pz,
            "p_emp": float((np.sum(np.array(vals) >= real) + 1) / (nsim + 1)), "n": len(tr)}  # fmt: skip


# ---------------- 진단 ----------------
def diag_A(sim, fires, W, rng, cap=4000) -> dict:
    b = sim.b
    t, h, lo, c = b["t"], b["h"], b["l"], b["c"]
    n = len(c)
    Hh = np.r_[sliding_window_view(h[1:], 96).max(1), np.full(97, np.nan)][:n]
    Ll = np.r_[sliding_window_view(lo[1:], 96).min(1), np.full(97, np.nan)][:n]
    rng24 = (Hh - Ll) / c  # i 뒤 24시간 범위
    out = {}
    for w in ("fresh", "design", "val", "year"):
        a, z = W[w]
        pool = slot_pool(b, a, z, DAY + PEND * MIN + M15)
        slm = {s: float(np.nanmean(rng24[p])) for s, p in pool.items()}
        for v in VS:
            f = fires[v]
            f = f[(t[f] >= a) & (t[f] + DAY + M15 < z)]
            if not len(f):
                continue
            sl = (t[f] % DAY).astype(np.int64)
            real = float(np.nanmean(rng24[f]))
            base = float(np.mean([slm.get(int(s), np.nan) for s in sl]))
            rec = {"n_fire": int(len(f)), "per_day": len(f) / ((z - a) / DAY), "rng24": real, "rng24_rnd": base,
                   "ratio": real / base}  # fmt: skip
            sub = f if len(f) <= cap else np.sort(rng.choice(f, cap, replace=False))
            for rn, N in RS.items():
                for S in SS:
                    rr_, rp_ = [], []
                    for i in sub:
                        x, _ = sim.bracket(int(i), N, S, "X2R")
                        if x is not None:
                            rr_.append(x)
                        p = pool.get(int(t[i] % DAY))
                        for _try in range(10):
                            y, _ = sim.bracket(int(rng.choice(p)), N, S, "X2R")
                            if y is not None:
                                rp_.append(y)
                                break
                    rec[f"{rn}|{S}"] = {
                        "fills": len(rr_), "tp": float(np.mean([x["why"] == "tp" for x in rr_])) if rr_ else None,
                        "tp_rnd": float(np.mean([x["why"] == "tp" for x in rp_])) if rp_ else None,
                        "r": float(np.mean([x["r"] for x in rr_])) if rr_ else None,
                        "r_rnd": float(np.mean([x["r"] for x in rp_])) if rp_ else None,
                        "both": float(np.mean([x["why"] == "both" for x in rr_])) if rr_ else None}  # fmt: skip
            out[f"{v}|{w}"] = rec
            print("diagA", v, w, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()
                                  if not isinstance(x, dict)},
                  {k: (x["fills"], x["tp"] and round(x["tp"], 3), x["tp_rnd"] and round(x["tp_rnd"], 3),
                       x["r"] and round(x["r"] * 100, 3), x["r_rnd"] and round(x["r_rnd"] * 100, 3))
                   for k, x in rec.items() if isinstance(x, dict)}, flush=True)  # fmt: skip
    return out


def diag_B(sim, sigs, W, rng) -> dict:
    m, b = sim.m, sim.b
    t = b["t"]
    out = {}
    hs = (60, 240, 1440)

    def fwd(i, d):
        k = sim.kmin(int(i))
        if k < 0 or k + 1441 >= sim.n1:
            return None
        ep = m["o"][k]
        rets = [d * (m["c"][k + hh - 1] / ep - 1) for hh in hs]
        hh_, ll_ = m["h"][k : k + 1440], m["l"][k : k + 1440]
        up = (hh_ >= ep * 1.01) if d == 1 else (ll_ <= ep * 0.99)
        dn = (ll_ <= ep * 0.99) if d == 1 else (hh_ >= ep * 1.01)
        ku = int(np.argmax(up)) if up.any() else 10**9
        kd = int(np.argmax(dn)) if dn.any() else 10**9
        return rets + [float(ku < kd and ku < 10**9)]

    for w in ("fresh", "design", "val", "year"):
        a, z = W[w]
        pool = slot_pool(b, a, z, DAY + 2 * M15)
        for k, (ii, dd) in sigs.items():
            sel = (t[ii] >= a) & (t[ii] + DAY + M15 < z)
            ii, dd = ii[sel], dd[sel]
            if len(ii) < 10:
                continue
            R, Q = [], []
            for i, d in zip(ii, dd):
                x = fwd(i, d)
                if x is None:
                    continue
                R.append(x)
                y = fwd(rng.choice(pool[int(t[i] % DAY)]), d)
                if y is not None:
                    Q.append(y)
            R, Q = np.array(R), np.array(Q)
            rec = {"n": int(len(R)), "per_day": len(ii) / ((z - a) / DAY),
                   "nl": int((dd == 1).sum()), "ns": int((dd == -1).sum())}  # fmt: skip
            for j, nm in enumerate(("f1h", "f4h", "f24h", "p1before1")):
                rec[nm] = float(R[:, j].mean())
                rec[nm + "_rnd"] = float(Q[:, j].mean())
            se = R[:, 1].std() / math.sqrt(len(R))
            rec["t4h"] = float((rec["f4h"] - rec["f4h_rnd"]) / se) if se > 0 else 0.0
            out[f"{k}|{w}"] = rec
            print("diagB", k, w, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in rec.items()}, flush=True)
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
    b = load(DW.R5)
    t_end = int(b["t"][-1]) + M15
    m = U.load_1m(t_end)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    sim = Sim(m, b, fund_fn)
    W = A.windows(t_end)
    W["dv"] = (W["design"][0], W["val"][1])
    S_ = build(b)
    fires, sigs = S_["fires"], S_["sigs"]
    print("loaded", len(m["t"]), {k: len(v) for k, v in fires.items()}, {k: len(v[0]) for k, v in sigs.items()},
          S_["th"], round(time.time() - t0), flush=True)  # fmt: skip
    rng = np.random.default_rng(5)

    # 1. 진단
    dA = diag_A(sim, fires, W, rng)
    dB = diag_B(sim, sigs, W, rng)
    json.dump({"A": dA, "B": dB, "th": S_["th"]}, (out / "diag.json").open("w"), default=float)
    print("diag done", round(time.time() - t0), flush=True)

    # 2. 변형 전부 (2020-03 ~ 끝 한 번 돌리고 구간으로 자른다)
    full = (A.ms("2020-03-01"), t_end)
    res, trades, spec = {}, {}, {}
    for v in VS:
        for rn, N in RS.items():
            for S in SS:
                for X in XS:
                    spec[f"{v}|{rn}|{S}|{X}"] = ("A", v, N, S, X)
    for k in BS:
        for hn in BH:
            spec[f"{k}|{hn}"] = ("B", k, hn)

    def run(name, a=None, z=None, **kw):
        sp = spec[name]
        a = full[0] if a is None else a
        z = full[1] if z is None else z
        if sp[0] == "A":
            return sim.run_A(fires[sp[1]], sp[2], sp[3], sp[4], a, z, **kw)
        return sim.run_B(sigs[sp[1]], BH[sp[2]], a, z, S_["chase"], **kw)

    def evaluate(tr):
        return {w: summ(slice_w(tr, *W[w]), *W[w]) for w in SHOWW}

    for name in spec:
        tr = run(name)
        trades[name] = tr
        res[name] = evaluate(tr)
        r = res[name]
        print(name, {w: (r[w]["n"], round(r[w]["per_day"], 2), r[w]["win"] and round(r[w]["win"], 2),
                         round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f0.4"]["ret"], 2))
                     for w in MAINW + ("d30",)}, round(time.time() - t0), flush=True)  # fmt: skip

    def pick(fam):
        names = [n for n in spec if spec[n][0] == fam]
        ok = [n for n in names if res[n]["design"]["per_day"] >= 0.8] or names
        return sorted(ok, key=lambda n: (-res[n]["design"]["x5_f1.0"]["ret"], res[n]["design"]["x5_f1.0"]["mdd"]))[:2]

    finals = pick("A") + pick("B")
    print("FINALS", finals, flush=True)
    detail = {}
    for fn in finals:
        sp = spec[fn]
        dv = slice_w(trades[fn], *W["dv"])
        if sp[0] == "A":
            rnd = random_A(sim, dv, sp[2], sp[3], sp[4], *W["dv"], args.nsim, 7) if len(dv) >= 5 else None
        else:
            rnd = random_B(sim, dv, BH[sp[2]], *W["dv"], args.nsim, 7) if len(dv) >= 5 else None
        print("rnd", fn, rnd, round(time.time() - t0), flush=True)
        red = {}
        for nm, kw in (("lat1", {"lat": 1}), ("lat5", {"lat": 5}), ("slip+0.02", {"slip": SLIP + 0.0002}),
                       ("fee1.5", {"fee_mult": 1.5})):  # fmt: skip
            tr = run(fn, **kw)
            red[nm] = {w: summ(slice_w(tr, *W[w]), *W[w]) for w in MAINW + ("m6", "d30")}
            print("red", fn, nm, {w: round(red[nm][w]["x5_f1.0"]["ret"], 2) for w in red[nm]}, flush=True)
        years = {}
        for y in range(2020, 2027):
            a_, z_ = A.ms(f"{y}-01-01"), min(A.ms(f"{y + 1}-01-01"), t_end)
            years[y] = summ(slice_w(trades[fn], a_, z_), a_, z_)
        print("years", fn, {y: (v["n"], round(v["x5_f1.0"]["ret"], 2)) for y, v in years.items()}, flush=True)
        fx = res[fn]
        mw = [w for w in MAINW if not (sp[0] == "B" and sp[1] in METRIC_B and w == "fresh")]
        g1 = all(fx[w]["per_day"] >= 0.8 for w in ("design", "val", "year"))
        g2 = all(fx[w][f"x5_f{f}"]["ret"] > 0 for w in mw for f in (1.0, 0.4))
        g3 = all(fx[w]["x5_f1.0"]["liq"] == 0 for w in mw)
        g4 = rnd is not None and rnd["real"] > rnd["rnd_mean"] and rnd["p_norm"] * N_CUM < 0.05
        g5 = all(red[k][w]["x5_f1.0"]["ret"] > 0 for k in red for w in ("val", "year"))
        verdict = {"g1_freq": g1, "g2_all_windows": g2, "g3_no_liq": g3, "g4_random_bonf": g4, "g5_red": g5,
                   "pass": bool(g1 and g2 and g3 and g4 and g5)}  # fmt: skip
        print("VERDICT", fn, verdict, flush=True)
        detail[fn] = {"rnd": rnd, "red": red, "years": years, "verdict": verdict}

    n_pos = {w: sum(res[n][w]["x5_f1.0"]["ret"] > 0 for n in spec) for w in MAINW}
    json.dump({"windows": W, "spec": spec, "res": res, "finals": finals, "detail": detail, "n_pos": n_pos,
               "th": S_["th"], "n_this": N_THIS, "n_cum": N_CUM}, (out / "result.json").open("w"), default=float)  # fmt: skip
    json.dump({fn: {w: slice_w(trades[fn], *W[w]) for w in SHOWW} for fn in finals},
              (out / "trades.json").open("w"))  # fmt: skip
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
