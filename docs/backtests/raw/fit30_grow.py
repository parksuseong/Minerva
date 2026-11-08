"""fit30 키우기: 같은 가중 점수 모형을 긴 구간(60·90·180·365일)에 맞추고, 매주 다시 맞추는 걷기 시험.

등록: docs/backtests/2026-11-08-btc-fit30-grow.md. 연구 전용(주문 없음, 바이낸스 POST 없음).
fit30.py의 특징·표준화·진입 M·청산 R(+1.5·ATR 손절, 32봉)·목표 함수를 그대로 쓰고, 체결 루프만 numba로 옮겼다
(같은 난수 순서로 fit30 30일 맞춤을 재현해 일치 확인).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/fit30-grow-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env NUMBA_CACHE_DIR=$OUT/nbcache \
    PYTHONPATH=$OUT/pynb:$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/fit30_grow.py --out $OUT fixed
  ... fit30_grow.py --out $OUT wf --addon BASE      # 걷기 시험 (N = 30·60·90·180)
  ... fit30_grow.py --out $OUT wf --addon GTGATE|GTFEAT|DCFEAT
  ... fit30_grow.py --out $OUT report
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fit30 as F
import numba as nb
import numpy as np

M15, DAY = F.M15, F.DAY
TAKER, MAKER = F.TAKER, F.MAKER
LIQ, LEV = -0.95, 5
KST = timezone(timedelta(hours=9))
T_BARS, SL_ATR = 32, 1.5
NS = (30, 60, 90, 180)
FIXED = (30, 60, 90, 180, 365)
ADDONS = ("BASE", "GTGATE", "GTFEAT", "DCFEAT")
SLIP = 0.0002  # 시장가 체결 1회당 추가 미끄러짐(레드팀)
N_TESTS, N_BEFORE = 52, 6256


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def kst(x: int) -> str:
    return datetime.fromtimestamp(x / 1000, KST).strftime("%Y-%m-%d %H:%M")


# ---------------- 체결 (fit30.sim 의 entry M · exit R 경로와 같은 순서) ----------------
@nb.njit(cache=True)
def sim_core(o, h, l, c, t, atr, big, sig, ft, fcum, a, z, delay, feem, slip, T, slm):
    cap = (z - a) // 2 + 4
    D = np.zeros(cap, np.int64)
    E = np.zeros(cap, np.int64)
    Xi = np.zeros(cap, np.int64)
    EP = np.zeros(cap)
    XP = np.zeros(cap)
    RR = np.zeros(cap)
    WO = np.zeros(cap)
    WHY = np.zeros(cap, np.int64)  # 0 stop 1 time 2 rev 3 end
    SI = np.zeros(cap, np.int64)
    n = 0
    has = False
    d = 0
    e = 0
    ep = 0.0
    stop = 0.0
    worst = 0.0
    si = 0
    t0 = 0
    for i in range(a, z - 2):
        if has:
            adv = (l[i] / ep - 1.0) if d == 1 else (1.0 - h[i] / ep)
            if adv < worst:
                worst = adv
            hit = (d == 1 and l[i] <= stop) or (d == -1 and h[i] >= stop)
            px = 0.0
            why = -1
            if hit:
                gap = (d == 1 and o[i] < stop) or (d == -1 and o[i] > stop)
                px = o[i] if gap else stop
                why = 0
            elif i - e + 1 >= T:
                px = c[i]
                why = 1
            if why >= 0:
                fee = (TAKER + TAKER) * feem + 2 * slip
                i0 = np.searchsorted(ft, t0, side="right")
                i1 = np.searchsorted(ft, t[i] + M15, side="right")
                fu = -d * (fcum[i1] - fcum[i0])
                D[n], E[n], Xi[n], EP[n], XP[n] = d, e, i, ep, px
                RR[n] = d * (px / ep - 1.0) - fee + fu
                WO[n] = worst - TAKER * feem - slip
                WHY[n], SI[n] = why, si
                n += 1
                has = False
        q = 0
        if i - delay >= a:
            q = sig[i - delay]
        if q != 0:
            if has and d == -q:
                px = c[i]
                fee = (TAKER + TAKER) * feem + 2 * slip
                i0 = np.searchsorted(ft, t0, side="right")
                i1 = np.searchsorted(ft, t[i] + M15, side="right")
                fu = -d * (fcum[i1] - fcum[i0])
                D[n], E[n], Xi[n], EP[n], XP[n] = d, e, i, ep, px
                RR[n] = d * (px / ep - 1.0) - fee + fu
                WO[n] = worst - TAKER * feem - slip
                WHY[n], SI[n] = 2, si
                n += 1
                has = False
            if not has:
                lim = 0.03 if big[i] else 0.01
                if abs(o[i + 1] / c[i] - 1.0) <= lim:
                    has = True
                    d = q
                    e = i + 1
                    ep = o[i + 1]
                    stop = ep - d * slm * atr[i]
                    worst = 0.0
                    si = i
                    t0 = t[i + 1]
    if has:
        i = z - 1
        px = c[i]
        fee = (TAKER + TAKER) * feem + 2 * slip
        i0 = np.searchsorted(ft, t0, side="right")
        i1 = np.searchsorted(ft, t[i] + M15, side="right")
        fu = -d * (fcum[i1] - fcum[i0])
        D[n], E[n], Xi[n], EP[n], XP[n] = d, e, i, ep, px
        RR[n] = d * (px / ep - 1.0) - fee + fu
        WO[n] = worst - TAKER * feem - slip
        WHY[n], SI[n] = 3, si
        n += 1
    return D[:n], E[:n], Xi[:n], EP[:n], XP[:n], RR[:n], WO[:n], WHY[:n], SI[:n]


@nb.njit(cache=True)
def lev_stats(D, RR, WO, L, frac):
    """(5배 수익, 낙폭, 청산 수). frac=1.0이면 pine_eval.lev_path, 0.4면 ql_upgrade.path 와 같은 계산."""
    eq, peak, mdd, liq = 1.0, 1.0, 0.0, 0
    for k in range(len(RR)):
        if L * WO[k] <= LIQ:
            liq += 1
            if frac >= 1.0:
                return eq * 0.0 - 1.0, 1.0, liq
            pos = -1.0
        else:
            pos = max(L * RR[k], -1.0)
            tr = eq * (1 + frac * L * WO[k])
            mdd = max(mdd, 1 - tr / peak)
        eq *= 1 + frac * pos
        if eq <= 0:
            return -1.0, 1.0, liq
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return eq - 1.0, mdd, liq


@nb.njit(cache=True)
def objective_nb(D, RR, WO, days):
    n = len(D)
    if n < days or n == 0:
        return -10.0 + n / days
    nl = 0
    for k in range(n):
        if D[k] == 1:
            nl += 1
    fl, fs = nl / n, (n - nl) / n
    ret, mdd, liq = lev_stats(D, RR, WO, 5.0, 1.0)
    pen = 0.0
    mn = min(fl, fs)
    if mn < 0.3:
        pen += 2 + (0.3 - mn) * 10
    if mdd > 0.25:
        pen += 2 + (mdd - 0.25) * 10
    if liq > 0:
        pen += 5
    return ret - pen


@nb.njit(cache=True)
def score6(s, qs, gate, use_g, full, o, h, l, c, t, atr, big, ft, fcum, a, z, days):
    """θ 6개(|s| 분위수)마다 신호를 만들어 체결하고 목표값을 낸다. full은 재사용 버퍼(끝나면 0)."""
    ab = np.abs(s)
    obs = np.zeros(len(qs))
    ths = np.zeros(len(qs))
    for k in range(len(qs)):
        th = np.quantile(ab, qs[k])
        for j in range(len(s)):
            v = 1 if s[j] >= th else (-1 if s[j] <= -th else 0)
            if use_g and gate[a + j] != v:
                v = 0
            full[a + j] = v
        D, E, Xi, EP, XP, RR, WO, WHY, SI = sim_core(o, h, l, c, t, atr, big, full, ft, fcum, a, z, 0, 1.0,
                                                     0.0, T_BARS, SL_ATR)  # fmt: skip
        obs[k] = objective_nb(D, RR, WO, days)
        ths[k] = th
    for j in range(len(s)):
        full[a + j] = 0
    return obs, ths


# ---------------- 자료 ----------------
class Ctx:
    def __init__(self):
        b = F.load()
        R = F.raw_features(b)
        self.b, self.R = b, R
        self.o, self.h, self.l, self.c = b["o"], b["h"], b["l"], b["c"]
        self.t = b["t"]
        self.ft = b["ft"]
        self.fcum = np.r_[0.0, np.cumsum(b["fr"])]
        self.atr = np.nan_to_num(R["atr"])
        self.big = R["big"].astype(np.bool_)
        self.extra = None

    def add_extra(self):
        import daily_wave as DW
        from rr_asym import donchian_sig

        g = DW.features(self.b)["gate"]["GT"]
        up, dn = donchian_sig(self.b, 96)
        raw = up.astype(float) - dn.astype(float)
        dc = np.zeros(len(raw))  # 최근 4봉 안 마지막 24시간 돈치안 첫 이탈 방향
        last, lasti = 0.0, -999
        for i in range(len(raw)):
            if raw[i] != 0:
                last, lasti = raw[i], i
            dc[i] = last if i - lasti < 4 else 0.0
        self.extra = {"GT": np.asarray(g, float), "DC": dc}

    def run(self, sig, a, z, delay=0, feem=1.0, slip=0.0):
        return sim_core(self.o, self.h, self.l, self.c, self.t, self.atr, self.big, sig.astype(np.int64),
                        self.ft, self.fcum, int(a), int(z), int(delay), float(feem), float(slip),
                        T_BARS, SL_ATR)  # fmt: skip


def feats_for(addon: str) -> tuple:
    if addon == "GTFEAT":
        return F.FEATS + ("gt",)
    if addon == "DCFEAT":
        return F.FEATS + ("dc",)
    return F.FEATS


def zmat(C: Ctx, feats, a, z) -> tuple[np.ndarray, dict]:
    st, cols = {}, []
    for name in feats:
        x = C.R["f"][name] if name in C.R["f"] else C.extra["GT" if name == "gt" else "DC"]
        mu, sd = float(x[a:z].mean()), float(x[a:z].std()) or 1.0
        st[name] = (mu, sd)
        cols.append((x - mu) / sd)
    return np.clip(np.vstack(cols), -4, 4), st


def make_sig(s, th, gate=None):
    sig = np.where(s >= th, 1, np.where(s <= -th, -1, 0)).astype(np.int64)
    if gate is not None:
        sig = np.where(gate == sig, sig, 0).astype(np.int64)
    return sig


# ---------------- 탐색 (fit30.search 와 같은 난수 순서) ----------------
def search(C: Ctx, Z, a, z, n_rand, rng, n_refine, passes, gate=None):
    """반환: [(목표, w, θ)] 정렬. 신호는 맞춤 구간 [a, z)에서만 계산(앞뒤 0)."""
    nF = Z.shape[0]
    days = (z - a) / 96
    Zs = Z[:, a:z]
    full = np.zeros(len(C.t), np.int64)

    qs = np.array(F.QS)
    gfull = np.zeros(len(C.t), np.int64) if gate is None else gate.astype(np.int64)
    use_g = gate is not None

    def score(w):
        s = w @ Zs
        obs, ths = score6(s, qs, gfull, use_g, full, C.o, C.h, C.l, C.c, C.t, C.atr, C.big, C.ft, C.fcum,
                          a, z, days)  # fmt: skip
        return zip(obs.tolist(), ths.tolist())

    res = []
    for _ in range(n_rand):
        w = np.zeros(nF)
        nz = rng.choice(nF, rng.integers(3, 13), replace=False)
        w[nz] = rng.uniform(-1, 1, len(nz))
        for ob, th in score(w):
            res.append((ob, w.copy(), th))
    res.sort(key=lambda x: -x[0])
    top, seen = [], set()
    for sc, w, th in res:
        key = tuple(np.round(w, 2))
        if key in seen:
            continue
        seen.add(key)
        top.append([sc, w, th])
        if len(top) >= max(n_refine, 20):
            break
    for item in top[:n_refine]:
        for _ in range(passes):
            for j in range(nF):
                for dlt in (-0.25, 0.25):
                    w2 = item[1].copy()
                    w2[j] = np.clip(w2[j] + dlt, -1, 1)
                    if np.count_nonzero(w2) > 12:
                        continue
                    for ob, th in score(w2):
                        if ob > item[0]:
                            item[0], item[1], item[2] = ob, w2, th
    top.sort(key=lambda x: -x[0])
    return top


def trades_list(C: Ctx, out_tuple) -> list:
    D, E, Xi, EP, XP, RR, WO, WHY, SI = out_tuple
    return [{"d": int(D[k]), "e": int(E[k]), "x": int(Xi[k]), "ep": float(EP[k]), "xp": float(XP[k]),
             "r": float(RR[k]), "worst": float(WO[k]), "why": ("stop", "time", "rev", "end")[int(WHY[k])],
             "sig": int(SI[k]), "te": int(C.t[E[k]])} for k in range(len(D))]  # fmt: skip


def lev(tr, frac=1.0, L=5.0):
    if not tr:
        return {"ret": 0.0, "mdd": 0.0, "liq": 0}
    D = np.array([x["d"] for x in tr], np.int64)
    RR = np.array([x["r"] for x in tr])
    WO = np.array([x["worst"] for x in tr])
    r, m, q = lev_stats(D, RR, WO, float(L), float(frac))
    return {"ret": float(r), "mdd": float(m), "liq": int(q)}


def summ(tr, days) -> dict:
    out = {"n": len(tr), "per_day": len(tr) / days if days else 0.0,
           "long": sum(x["d"] == 1 for x in tr), "short": sum(x["d"] == -1 for x in tr)}  # fmt: skip
    if tr:
        r = np.array([x["r"] for x in tr])
        out.update(win=float((r > 0).mean()), sum1=float(r.sum()), avg1=float(r.mean()),
                   stops=sum(x["why"] == "stop" for x in tr))  # fmt: skip
    else:
        out.update(win=None, sum1=0.0, avg1=None, stops=0)
    out["x1"] = lev(tr, 1.0, 1.0)
    out["x5_f1.0"] = lev(tr, 1.0)
    out["x5_f0.4"] = lev(tr, 0.4)
    return out


# ---------------- 안전장치 ----------------
def guard(tr, frac, kill="pause", wk_starts=None, dd=0.20, nstop=5):
    """사후 거르기. kill='perm': 고점 대비 −20%면 끝까지 멈춤. 'pause': 다음 주 다시 맞춤까지 멈추고 고점 재설정.
    둘 다: 같은 UTC 날 손절 5연속이면 그날 남은 진입 건너뜀."""
    eq, peak, keep, halted_until, dead = 1.0, 1.0, [], -1, False
    day_, streak = None, 0
    halts = []
    ws = np.array(wk_starts or [], np.int64)
    for x in tr:
        te = x["te"]
        dday = te // DAY
        if dday != day_:
            day_, streak = dday, 0
        if dead or te < halted_until or streak >= nstop:
            continue
        keep.append(x)
        w = x["r"]
        if LEV * x["worst"] <= LIQ:
            eq *= 1 - frac
        else:
            eq *= 1 + frac * max(LEV * w, -1.0)
        streak = streak + 1 if x["why"] == "stop" else 0
        peak = max(peak, eq)
        if eq / peak - 1 <= -dd:
            halts.append(kst(te))
            if kill == "perm":
                dead = True
            else:
                k = int(np.searchsorted(ws, te, "right"))
                halted_until = int(ws[k]) if k < len(ws) else 2**62
                peak = eq
    return keep, halts


# ---------------- 1) 끝을 고정한 긴 구간 맞춤 ----------------
def cmd_fixed(out: Path) -> None:
    C = Ctx()
    t = C.t
    end = int(t[-1]) + M15
    res = {}
    for N in FIXED:
        t0 = time.time()
        a, z = int(np.searchsorted(t, end - N * DAY)), len(t)
        Z, st = zmat(C, F.FEATS, a, z)
        rng = np.random.default_rng(20261108)
        top = search(C, Z, a, z, 6000, rng, 10, 2)
        sc, w, th = top[0]
        s = w @ Z
        sig = make_sig(s, th)
        sig[:a] = 0
        tr_in = trades_list(C, C.run(sig, a, z))
        d30a = int(np.searchsorted(t, end - 30 * DAY))
        tr30 = [x for x in tr_in if x["e"] >= d30a]
        # 맞춤 구간 앞 전체(표본 밖): 같은 w·θ·표준화 고정
        oa = int(np.searchsorted(t, ms("2020-03-01")))
        sig2 = make_sig(s, th)
        tr_oos = trades_list(C, C.run(sig2, oa, a))
        by_year = {}
        for x in tr_oos:
            by_year.setdefault(datetime.fromtimestamp(x["te"] / 1000, KST).year, []).append(x)
        top_w = np.array([x[1] for x in top[:10]])
        res[N] = {
            "objective": float(sc), "theta": float(th), "q": None,
            "weights": {n: round(float(v), 3) for n, v in zip(F.FEATS, w)},
            "top10_mean": {n: round(float(v), 3) for n, v in zip(F.FEATS, top_w.mean(0))},
            "top10_pos_share": {n: round(float((top_w[:, j] > 0).mean()), 2) for j, n in enumerate(F.FEATS)},
            "std": st, "span": [kst(int(t[a])), kst(int(t[z - 1]))],
            "fit": summ(tr_in, (z - a) / 96), "last30": summ(tr30, 30),
            "oos_before": summ(tr_oos, (a - oa) / 96),
            "oos_by_year": {y: summ(v, 365) for y, v in sorted(by_year.items())},
            "sec": round(time.time() - t0, 1),
        }
        print(N, round(sc, 3), res[N]["fit"]["n"], round(res[N]["fit"]["x5_f1.0"]["ret"], 3),
              "last30", res[N]["last30"]["n"], round(res[N]["last30"]["x5_f1.0"]["ret"], 3),
              "oos", round(res[N]["oos_before"]["x5_f0.4"]["ret"], 3), res[N]["sec"], flush=True)  # fmt: skip
    keys = list(res)
    cos = {}
    for i, p in enumerate(keys):
        for q in keys[i + 1:]:
            u = np.array(list(res[p]["weights"].values()))
            v = np.array(list(res[q]["weights"].values()))
            cos[f"{p}-{q}"] = round(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)), 3)
    res["cos"] = cos
    (out / "fixed.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(cos))


# ---------------- 2) 매주 다시 맞추는 걷기 시험 ----------------
WF_START = ms("2020-09-06")


def cmd_wf(out: Path, addon: str, n_rand: int, n_refine: int) -> None:
    C = Ctx()
    if addon != "BASE":
        C.add_extra()
    t = C.t
    end = int(t[-1]) + M15
    feats = feats_for(addon)
    gate = C.extra["GT"] if addon == "GTGATE" else None
    start = WF_START if addon == "BASE" else ms("2024-01-07")
    nweeks = int((end - start) // (7 * DAY))
    res = {}
    for N in NS:
        t0 = time.time()
        rng = np.random.default_rng(1000 + N)
        folds = []
        for wk in range(nweeks, 0, -1):
            f1 = end - wk * 7 * DAY
            a, z = int(np.searchsorted(t, f1 - N * DAY)), int(np.searchsorted(t, f1))
            z2 = int(np.searchsorted(t, f1 + 7 * DAY))
            Z, _ = zmat(C, feats, a, z)
            top = search(C, Z, a, z, n_rand, rng, n_refine, 1, gate)
            sc, w, th = top[0]
            sig = np.zeros(len(t), np.int64)
            sig[z:z2] = make_sig(w @ Z[:, z:z2], th, None if gate is None else gate[z:z2])
            fold = {"start": int(t[z]) if z < len(t) else f1, "a": z, "z": z2, "obj": float(sc),
                    "w": [float(v) for v in w], "th": float(th)}  # fmt: skip
            for tag, kw in (("base", {}), ("delay1", {"delay": 1}), ("fee15", {"feem": 1.5}), ("slip", {"slip": SLIP})):
                fold[tag] = trades_list(C, C.run(sig, z, z2, **kw))
            # 무작위: 같은 주·같은 15분 칸·같은 방향, R 청산은 무작위 신호끼리 반전
            folds.append(fold)
            if wk % 26 == 0:
                print(addon, N, kst(f1), len(fold["base"]), round(time.time() - t0), flush=True)
        res[N] = folds
        print(addon, N, "done", round(time.time() - t0), flush=True)
    with open(out / f"wf_{addon}.json", "w") as fh:
        json.dump({str(k): v for k, v in res.items()}, fh)


@nb.njit(cache=True)
def rand_fold(o, h, l, c, t, atr, big, ft, fcum, a, z, sidx, dirs, nsim, seed):
    """같은 주 안, 같은 UTC 15분 칸(=하루 단위로만 섞음), 같은 방향 무작위 신호 → 1배 순손익 합 nsim개."""
    np.random.seed(seed)
    out = np.zeros(nsim)
    sig = np.zeros(len(t), np.int64)
    for k in range(nsim):
        for j in range(len(sidx)):
            hod = sidx[j] % 96
            base = a + ((hod - a % 96) % 96)
            m = (z - 3 - base + 95) // 96
            if m <= 0:
                continue
            sig[base + 96 * np.random.randint(m)] = dirs[j]
        D, E, Xi, EP, XP, RR, WO, WHY, SI = sim_core(o, h, l, c, t, atr, big, sig, ft, fcum, a, z, 0, 1.0, 0.0,
                                                     T_BARS, SL_ATR)  # fmt: skip
        out[k] = RR.sum()
        for j in range(a, z):
            sig[j] = 0
    return out


def wins(end: int) -> dict:
    return {"A": (ms("2024-07-01"), end), "Y1": (end - 365 * DAY, end), "M6": (end - 183 * DAY, end),
            "D30": (end - 30 * DAY, end), "B": (ms("2022-01-01"), ms("2024-07-01")),
            "C": (ms("2020-09-01"), ms("2022-01-01"))}  # fmt: skip


def cmd_report(out: Path, nsim: int) -> None:
    C = Ctx()
    t = C.t
    end = int(t[-1]) + M15
    W = wins(end)
    rep = {}
    for addon in ADDONS:
        fp = out / f"wf_{addon}.json"
        if not fp.exists():
            continue
        allf = json.loads(fp.read_text())
        for N, folds in allf.items():
            key = f"{addon}-{N}"
            t0 = time.time()
            # 무작위: 주마다 nsim개 1배 합
            RM = np.zeros((len(folds), nsim))
            for fi, f in enumerate(folds):
                if f["base"] and f["z"] - f["a"] >= 8:
                    sidx = np.array([x["sig"] for x in f["base"]], np.int64)
                    dirs = np.array([x["d"] for x in f["base"]], np.int64)
                    RM[fi] = rand_fold(C.o, C.h, C.l, C.c, t, C.atr, C.big, C.ft, C.fcum, f["a"], f["z"], sidx,
                                       dirs, nsim, 7 + fi)  # fmt: skip
            starts = np.array([f["start"] for f in folds], np.int64)
            r = {}
            for wn, (wa, wz) in W.items():
                sel = [i for i, f in enumerate(folds) if wa <= f["start"] < wz]
                if not sel:
                    continue
                if wn == "D30":  # 최근 30일은 주 경계와 안 맞아 매매 진입 시각으로 자른다
                    pick = lambda tag: [x for f in folds for x in f[tag] if wa <= x["te"] < wz]  # noqa: E731
                    days = 30.0
                else:
                    pick = lambda tag: [x for i in sel for x in folds[i][tag]]  # noqa: E731
                    days = len(sel) * 7.0
                tr = pick("base")
                sm = summ(tr, days)
                for tag in ("delay1", "fee15", "slip"):
                    q = summ(pick(tag), days)
                    sm[tag] = {"x5_f0.4": q["x5_f0.4"], "x5_f1.0": q["x5_f1.0"], "sum1": q["sum1"]}
                if wn != "D30":
                    rs = RM[sel].sum(0)
                    sm["random"] = {"p": float((rs >= sm["sum1"]).mean()), "med_sum1": float(np.median(rs)),
                                    "z": float((sm["sum1"] - rs.mean()) / (rs.std() + 1e-12))}  # fmt: skip
                ws = [int(starts[i]) for i in sel]
                for frac in (0.4, 1.0):
                    for kill in ("perm", "pause"):
                        kept, halts = guard(tr, frac, kill, ws)
                        g = summ(kept, days)
                        sm[f"guard_{kill}_f{frac}"] = {"n": g["n"], "x5": g[f"x5_f{frac}"],
                                                        "halts": halts[:5], "n_halts": len(halts)}  # fmt: skip
                r[wn] = sm
            years = {}
            for f in folds:
                for x in f["base"]:
                    years.setdefault(datetime.fromtimestamp(x["te"] / 1000, timezone.utc).year, []).append(x)
            r["by_year"] = {y: summ(v, 365) for y, v in sorted(years.items())}
            Wm = np.array([f["w"] for f in folds])
            nzm = Wm != 0
            r["w_pos_share"] = (np.where(nzm, Wm > 0, False).sum(0) / np.maximum(nzm.sum(0), 1)).round(2).tolist()
            r["w_mean"] = Wm.mean(0).round(3).tolist()
            cs = [float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)) for u, v in zip(Wm[:-1], Wm[1:])]
            r["w_cos_next_week"] = round(float(np.mean(cs)), 3)
            A = r.get("A")
            if A:
                ok = {
                    "1_A_pos_mdd_noliq": A["x5_f0.4"]["ret"] > 0 and A["x5_f0.4"]["mdd"] <= 0.4
                    and A["x5_f1.0"]["liq"] == 0,
                    "2_Y1_M6_B_pos": all(r[k]["x5_f0.4"]["ret"] > 0 for k in ("Y1", "M6", "B") if k in r)
                    and "B" in r,
                    "3_freq_sides": A["per_day"] >= 1 and min(A["long"], A["short"]) >= 0.3 * max(A["n"], 1),
                    "4_random_bonf": A["random"]["p"] < 0.05 / N_TESTS,
                    "5_delay_fee": A["delay1"]["x5_f0.4"]["ret"] > 0 and A["fee15"]["x5_f0.4"]["ret"] > 0,
                }
                r["gates"] = ok
                r["pass"] = all(ok.values())
            rep[key] = r
            print(key, round(time.time() - t0), {k: (r[k]["n"], round(r[k]["per_day"], 2), r[k]["win"] and round(r[k]["win"], 2),
                  round(r[k]["x5_f0.4"]["ret"], 3), round(r[k]["x5_f1.0"]["ret"], 3), r[k].get("random", {}).get("p"))
                  for k in W if k in r}, r.get("pass"), flush=True)  # fmt: skip
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", choices=("bench", "fixed", "wf", "report"))
    ap.add_argument("--addon", default="BASE", choices=ADDONS)
    ap.add_argument("--nrand", type=int, default=1500)
    ap.add_argument("--nrefine", type=int, default=3)
    ap.add_argument("--nsim", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "bench":
        C = Ctx()
        t = C.t
        end = int(t[-1]) + M15
        aa, z = int(np.searchsorted(t, end - 30 * DAY)), len(t)
        Z, _ = zmat(C, F.FEATS, aa, z)
        rule = json.loads(Path("~/.minerva/research/fit30-2026-11-08/rule.json").expanduser().read_text())
        w = np.array([rule["weights"][n] for n in F.FEATS])
        s = w @ Z
        sig = make_sig(s, rule["theta"])
        sig[:aa] = 0
        tr = trades_list(C, C.run(sig, aa, z))
        sm = summ(tr, 30)
        ref = F.summ(F.sim(C.b, C.R, s, rule["theta"], rule["P"], aa, z), 30)
        print("numba", sm["n"], sm["win"], sm["x5_f1.0"], "ref", ref["n"], ref["win"], ref["x5"])
        for N in (30, 180, 365):
            aa = int(np.searchsorted(t, end - N * DAY))
            Z, _ = zmat(C, F.FEATS, aa, z)
            rng = np.random.default_rng(1)
            t0 = time.time()
            search(C, Z, aa, z, 100, rng, 0, 0)
            print(N, "days: 100 weights x 6 theta", round(time.time() - t0, 2), "s")
    elif a.cmd == "fixed":
        cmd_fixed(out)
    elif a.cmd == "wf":
        cmd_wf(out, a.addon, a.nrand, a.nrefine)
    else:
        cmd_report(out, a.nsim)


if __name__ == "__main__":
    main()
