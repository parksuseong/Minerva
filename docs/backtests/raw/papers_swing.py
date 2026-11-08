"""스윙 논문 규칙 16개(14 + 조합 2): 일봉 목표 포지션 → 선물 1분봉 체결, 펀딩, 5배 청산 모형.

등록: docs/backtests/2026-11-08-btc-papers-swing.md. 연구 전용(주문 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/papers-swing-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/papers_swing.py --out $OUT
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
import ql_upgrade as U
import regime_lines as RL
from r5_data import load
from rr_asym import donchian_sig

MIN, M15, DAY, H1 = 60_000, 900_000, 86_400_000, 3_600_000
TAKER, MAKER, SLIP, TICK = 0.0005, 0.0002, 0.0001, 0.1
LEV = 5
N_BONF, N_BEFORE = 16, 6308
MAINW = ("fresh", "design", "val", "year")
WINS = MAINW + ("m6", "d30", "pool")
ZPB_N = (5, 10, 20, 30, 60, 90, 150, 250, 360)


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


# ---------------- 자료 ----------------
def build(b: dict, m: dict) -> dict:
    """가격 열(15분 현물 → 15분 선물 → 1분 선물)로 일봉·1시간 종가를 만든다. 시각 = 봉 마감 시각."""
    raw = np.genfromtxt(RL.SPOT_CSV, delimiter=",", skip_header=1, usecols=(0, 2, 3, 4))
    raw = raw[raw[:, 0] < b["t"][0]]
    m0 = int(m["t"][0])
    fb = b["t"] < m0
    t = np.r_[raw[:, 0].astype(np.int64) + M15, b["t"][fb].astype(np.int64) + M15, m["t"].astype(np.int64) + MIN]
    h = np.r_[raw[:, 1], b["h"][fb], m["h"]]
    lo = np.r_[raw[:, 2], b["l"][fb], m["l"]]
    c = np.r_[raw[:, 3], b["c"][fb], m["c"]]
    kd = (t - 1) // DAY
    s = np.r_[0, np.nonzero(np.diff(kd))[0] + 1]
    e = np.r_[s[1:] - 1, len(t) - 1]
    D = {"T": kd[s] * DAY, "h": np.maximum.reduceat(h, s), "l": np.minimum.reduceat(lo, s), "c": c[e]}
    kh = (t - 1) // H1
    sh = np.r_[0, np.nonzero(np.diff(kh))[0] + 1]
    eh = np.r_[sh[1:] - 1, len(t) - 1]
    Hh = {"T": kh[sh] * H1, "c": c[eh]}
    return {"D": D, "H": Hh}


# ---------------- 신호: tgt[j] = 날 j 동안의 목표(날 j−1 마감까지 자료) ----------------
def shift1(x):
    return np.r_[0.0, x[:-1]]


def zpb(D: dict, long_only: bool, vt: bool) -> np.ndarray:
    c = D["c"]
    n = len(c)
    acc = np.zeros(n)
    for N in ZPB_N:
        pos, stop = 0, np.nan
        st = np.zeros(n)
        for k in range(N, n):
            hh, ll = c[k - N : k].max(), c[k - N : k].min()
            mid = (c[k - N + 1 : k + 1].max() + c[k - N + 1 : k + 1].min()) / 2
            if pos == 1 and c[k] < stop:
                pos = 0
            elif pos == -1 and c[k] > stop:
                pos = 0
            elif pos == 1:
                stop = max(stop, mid)
            elif pos == -1:
                stop = min(stop, mid)
            if pos != 1 and c[k] > hh:
                pos, stop = 1, mid
            elif pos != -1 and c[k] < ll and not long_only:
                pos, stop = -1, mid
            st[k] = pos
        acc += st
    s = acc / len(ZPB_N)
    if vt:
        r = np.r_[np.nan, np.diff(np.log(c))]
        sig = np.full(n, np.nan)
        for k in range(91, n):
            sig[k] = r[k - 89 : k + 1].std(ddof=1) * np.sqrt(365)
        v = np.where(np.isnan(sig), 0.0, np.minimum(2.0, 0.25 / np.where(sig > 0, sig, np.inf)) / 2)
        s = s * v
    return shift1(s)


def dz(D: dict, N: int) -> np.ndarray:
    c = D["c"]
    ma = np.convolve(c, np.ones(N) / N, "full")[: len(c)]
    ma[: N - 1] = np.nan
    with np.errstate(invalid="ignore"):
        s = np.where(np.isnan(ma), 0.0, np.sign(c - ma))
    return shift1(s)


def dow(T):
    return ((T // DAY) + 3) % 7  # 0 = 월요일


def lt(D: dict) -> np.ndarray:
    c, T = D["c"], D["T"]
    n = len(c)
    tgt = np.zeros(n)
    mons = np.nonzero(dow(T) == 0)[0]
    wr = []
    for j in mons:
        if j < 8:
            continue
        x = np.log(c[j - 1] / c[j - 8])
        if len(wr) >= 52:
            hist = np.array(wr[-52:])
            s = 1.0 if x > np.quantile(hist, 0.8) else (-1.0 if x < np.quantile(hist, 0.2) else 0.0)
            tgt[j : j + 7] = s
        wr.append(x)
    return tgt


def skew(D: dict, Hh: dict) -> np.ndarray:
    T = D["T"]
    n = len(T)
    tgt = np.zeros(n)
    hr = np.r_[np.nan, np.diff(np.log(Hh["c"]))]
    ht = Hh["T"]
    hist = []
    for j in np.nonzero(dow(T) == 0)[0]:
        a, z = np.searchsorted(ht, T[j] - 7 * DAY), np.searchsorted(ht, T[j])
        r = hr[a:z]
        r = r[~np.isnan(r)]
        if len(r) < 150:
            continue
        x = np.sqrt(len(r)) * np.sum(r**3) / np.sum(r**2) ** 1.5
        if len(hist) >= 52:
            hh = np.array(hist[-52:])
            s = 1.0 if x < np.quantile(hh, 0.2) else (-1.0 if x > np.quantile(hh, 0.8) else 0.0)
            tgt[j : j + 7] = s
        hist.append(x)
    return tgt


def mon(D: dict) -> np.ndarray:
    return (dow(D["T"]) == 0).astype(float)


def ovr(D: dict) -> np.ndarray:
    c = D["c"]
    n = len(c)
    r = np.r_[np.nan, c[1:] / c[:-1] - 1]
    s = np.zeros(n)
    for k in range(32, n):
        w = r[k - 30 : k]
        mu, sd = w.mean(), w.std(ddof=1)
        s[k] = 1.0 if r[k] > mu + 2 * sd else (-1.0 if r[k] < mu - 2 * sd else 0.0)
    return shift1(s)


def fund(D: dict, ft, fr) -> np.ndarray:
    T = D["T"]
    n = len(T)
    f7 = np.full(n, np.nan)
    for j in range(n):
        z = np.searchsorted(ft, T[j])  # T[j] 전에 정산된 것
        if z >= 21:
            f7[j] = fr[z - 21 : z].mean()
    tgt = np.zeros(n)
    for j in range(n):
        if np.isnan(f7[j]):
            continue
        h = f7[max(0, j - 365) : j]
        h = h[~np.isnan(h)]
        if len(h) < 90:
            continue
        p = (h < f7[j]).mean()
        tgt[j] = -1.0 if p > 0.9 else (1.0 if p < 0.1 else 0.0)
    return tgt


def t6d(D: dict) -> np.ndarray:
    ev = P.s6_trend_dev(D["h"], D["l"], D["c"]).astype(float)
    st = np.where(ev != 0, ev, np.nan)
    idx = np.where(~np.isnan(st), np.arange(len(st)), 0)
    np.maximum.accumulate(idx, out=idx)
    return shift1(np.nan_to_num(st[idx], nan=0.0))


# ---------------- 일봉 엔진 ----------------
class Eng:
    def __init__(self, D, m, ft, fr, t_end):
        self.D, self.m, self.ft, self.fr = D, m, ft, fr
        self.t_end = t_end
        self.cache = {}

    def px(self, lat):
        if lat not in self.cache:
            T = self.D["T"]
            k = np.searchsorted(self.m["t"], T + lat * MIN)
            ok = (k < len(self.m["t"])) & (T >= self.m["t"][0])
            kk = np.clip(k, 0, len(self.m["t"]) - 1)
            p = np.where(ok, self.m["o"][kk], np.nan)
            # 체결 사이 1분봉 고가·저가(청산·낙폭용)
            hi, lo = np.full(len(T), np.nan), np.full(len(T), np.nan)
            for j in range(len(T) - 1):
                if ok[j] and ok[j + 1] and kk[j + 1] > kk[j]:
                    hi[j] = self.m["h"][kk[j] : kk[j + 1]].max()
                    lo[j] = self.m["l"][kk[j] : kk[j + 1]].min()
            fs = np.zeros(len(T))
            for j in range(len(T) - 1):
                a_, z_ = np.searchsorted(self.ft, T[j] + lat * MIN, "right"), np.searchsorted(
                    self.ft, T[j + 1] + lat * MIN, "right")  # fmt: skip
                fs[j] = self.fr[a_:z_].sum()
            self.cache[lat] = (p, hi, lo, fs)
        return self.cache[lat]

    def idx(self, a_ms, z_ms):
        T = self.D["T"]
        return np.nonzero((T >= a_ms) & (T + DAY <= z_ms))[0]

    def run(self, tgt, a_ms, z_ms, f=1.0, lev=LEV, lat=0, slip=SLIP, fee_mult=1.0, curve=None):
        p, hi, lo, fs = self.px(lat)
        js = self.idx(a_ms, z_ms)
        js = js[~np.isnan(p[js]) & ~np.isnan(p[js + 1])]
        cost_u = TAKER * fee_mult + slip
        eq, peak, mdd, liq, E = 1.0, 1.0, 0.0, 0, 0.0
        for j in js:
            Et = lev * f * tgt[j]
            eq -= abs(Et - E) * cost_u * eq
            E = Et
            if E == 0:
                continue
            r = p[j + 1] / p[j] - 1
            adv = (lo[j] / p[j] - 1) if E > 0 else (1 - hi[j] / p[j])
            if adv <= -0.95 / lev:  # 5배면 19%, 손실 = 그 포지션 증거금 |E|/lev
                liq += 1
                eq *= 1 - abs(E) / lev
                E = 0.0
            else:
                mdd = max(mdd, 1 - eq * (1 + abs(E) * adv) / peak)
                pnl = E * r - E * fs[j]
                n_ = E * (1 + r)
                eq *= 1 + pnl
                E = n_ / (1 + pnl) if 1 + pnl > 0 else 0.0
            if eq <= 0:
                return {"ret": -1.0, "mdd": 1.0, "liq": liq}
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak)
            if curve is not None:
                curve.append((int(self.D["T"][j + 1]), float(eq)))
        return {"ret": eq - 1, "mdd": mdd, "liq": liq}

    def trades(self, tgt, a_ms, z_ms, lat=0, slip=SLIP, fee_mult=1.0):
        """부호 구간 = 한 매매. 1배 수익(진입·청산 테이커, 펀딩 포함, 분수 크기는 무시)."""
        p, hi, lo, fs = self.px(lat)
        js = self.idx(a_ms, z_ms)
        js = js[~np.isnan(p[js]) & ~np.isnan(p[js + 1])]
        T = self.D["T"]
        out, cur = [], None
        cu = TAKER * fee_mult + slip
        for j in js:
            d = int(np.sign(tgt[j]))
            if cur is not None and d != cur["d"]:
                out.append(cur)
                cur = None
            if d != 0 and cur is None:
                cur = {"d": d, "t_in": int(T[j] + lat * MIN), "ep": float(p[j]), "fund": 0.0, "worst": 0.0}
            if cur is not None:
                cur["fund"] += -d * fs[j]
                w = (lo[j] / cur["ep"] - 1) if d == 1 else (1 - hi[j] / cur["ep"])
                cur["worst"] = min(cur["worst"], w)
                cur["t_out"], cur["xp"] = int(T[j + 1] + lat * MIN), float(p[j + 1])
        if cur is not None:
            out.append(cur)
        for x in out:
            x["r"] = x["d"] * (x["xp"] / x["ep"] - 1) - 2 * cu + x["fund"]
            x["why"] = "sig"
        return out

    def daily_pnl(self, tgt, a_ms, z_ms):
        p, _, _, fs = self.px(0)
        js = self.idx(a_ms, z_ms)
        js = js[~np.isnan(p[js]) & ~np.isnan(p[js + 1])]
        r = p[js + 1] / p[js] - 1
        return js, r, fs[js]


def summ(eng, tgt, a_ms, z_ms, **kw) -> dict:
    tr = eng.trades(tgt, a_ms, z_ms, **{k: v for k, v in kw.items() if k != "f"})
    days = max((z_ms - a_ms) / DAY, 1)
    rr = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    js = eng.idx(a_ms, z_ms)
    out = {"n": len(tr), "per_day": len(tr) / days, "nl": sum(x["d"] == 1 for x in tr),
           "ns": sum(x["d"] == -1 for x in tr), "win": float((rr > 0).mean()) if len(rr) else None,
           "avg1": float(rr.mean()) if len(rr) else None,
           "expo": float(np.mean(np.abs(tgt[js]))) if len(js) else 0.0}  # fmt: skip
    for f in (1.0, 0.4):
        out[f"x5_f{f}"] = eng.run(tgt, a_ms, z_ms, f=f, **kw)
    return out, tr


def random_p(eng, tgt, a_ms, z_ms, nsim=1000, seed=7) -> dict:
    js, r, fs = eng.daily_pnl(tgt, a_ms, z_ms)
    s = tgt[js]
    cu = TAKER + SLIP

    def pnl(x):
        return x * r - x * fs - np.abs(np.diff(np.r_[0.0, x])) * cu

    real = pnl(s).mean()
    rng = np.random.default_rng(seed)
    n = len(s)
    vals = np.array([pnl(np.roll(s, int(rng.integers(1, n)))).mean() for _ in range(nsim)])
    return {"real": float(real), "rnd_mean": float(vals.mean()), "rnd_sd": float(vals.std()),
            "p": float((np.sum(vals >= real) + 1) / (nsim + 1))}  # fmt: skip


# ---------------- C2: DC24h 첫 이탈 + BEST 방향 ----------------
def c2_trades(b, m, D, best, ft, fr, a_ms, z_ms, lat=0, slip=SLIP, fee_mult=1.0):
    up, dn = donchian_sig(b, 96)
    ev = sorted([(int(i), 1) for i in np.nonzero(up)[0]] + [(int(i), -1) for i in np.nonzero(dn)[0]])
    T = D["T"]
    mt = m["t"]
    cu = TAKER * fee_mult + slip

    def kmin(ts):
        k = int(np.searchsorted(mt, ts))
        return k if k < len(mt) - 1 and int(mt[k]) - ts <= 5 * MIN else -1

    def gate(ts):
        j = int(np.searchsorted(T, ts, "right")) - 1
        return int(np.sign(best[j])) if j >= 0 else 0

    out, pos = [], None
    for i, d in ev:
        ts = int(b["t"][i]) + M15 + lat * MIN
        if ts < a_ms or ts >= z_ms:
            continue
        if pos is not None:
            # 그 사이 일봉 시가에 방향이 바뀌었으면 그날 시가에 청산
            j0 = int(np.searchsorted(T, pos["t_in"], "right"))
            j1 = int(np.searchsorted(T, ts, "right"))
            xt = None
            for j in range(j0, j1):
                if int(np.sign(best[j])) != pos["d"]:
                    xt = int(T[j]) + lat * MIN
                    break
            if xt is None and d == -pos["d"]:
                xt = ts
            if xt is not None:
                k = kmin(xt)
                if k < 0:
                    continue
                pos["x"] = k
                out.append(pos)
                pos = None
        if pos is None and gate(ts) == d:
            k = kmin(ts)
            if k < 0:
                continue
            pos = {"k": k, "d": d, "t_in": int(mt[k])}
    if pos is not None:  # 구간 끝까지 방향 유지 시 방향 바뀌는 날 또는 구간 끝
        j0 = int(np.searchsorted(T, pos["t_in"], "right"))
        xt = z_ms - MIN
        for j in range(j0, len(T)):
            if T[j] >= z_ms:
                break
            if int(np.sign(best[j])) != pos["d"]:
                xt = int(T[j]) + lat * MIN
                break
        k = kmin(xt)
        if k > pos["k"]:
            pos["x"] = k
            out.append(pos)
    res = []
    for x in out:
        k, j, d = x["k"], x["x"], x["d"]
        ep, xp = float(m["o"][k]), float(m["o"][j])
        worst = (m["l"][k:j].min() / ep - 1) if d == 1 else (1 - m["h"][k:j].max() / ep)
        t_in, t_out = int(mt[k]), int(mt[j])
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        r = d * (xp / ep - 1) - 2 * cu - d * float(fr[a_:z_].sum())
        if t_out > z_ms:
            continue
        res.append({"k": k, "x": j, "d": d, "ep": ep, "xp": xp, "r": float(r), "worst": float(min(worst, r)),
                    "t_in": t_in, "t_out": t_out, "why": "sig", "size": 1.0})  # fmt: skip
    return res


def c2_random(m, tr, a_ms, z_ms, nsim=200, seed=9) -> dict:
    rng = np.random.default_rng(seed)
    mt = m["t"]
    real = float(np.mean([x["r"] for x in tr]))
    a, z = np.searchsorted(mt, a_ms), np.searchsorted(mt, z_ms)
    vals = []
    for _ in range(nsim):
        rs = []
        for x in tr:
            L = x["x"] - x["k"]
            k = int(rng.integers(a, max(a + 1, z - L - 1)))
            ep, xp = m["o"][k], m["o"][k + L]
            rs.append(x["d"] * (xp / ep - 1) - 2 * (TAKER + SLIP))
        vals.append(np.mean(rs))
    vals = np.array(vals)
    return {"real": real, "rnd_mean": float(vals.mean()), "rnd_sd": float(vals.std()),
            "p": float((np.sum(vals >= real) + 1) / (nsim + 1)), "note": "funding not in random"}  # fmt: skip


# ---------------- 실행 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = load(DW.R5)
    m = U.load_1m(10**15)
    t_end = int((int(m["t"][-1]) // DAY) * DAY)  # 마지막 완성 일봉 끝
    keep = m["t"] < t_end + DAY
    m = {k: v[keep] for k, v in m.items()}
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)
    X = build(b, m)
    D, Hh = X["D"], X["H"]
    print("loaded", len(m["t"]), len(D["T"]), datetime.fromtimestamp(t_end / 1000, timezone.utc),
          round(time.time() - t0), flush=True)  # fmt: skip
    eng = Eng(D, m, ft, fr, t_end)
    W = {k: (ms(a), ms(z)) for k, (a, z) in DW.WIN.items() if k in ("fresh", "design", "val")}
    W["year"] = (t_end - 365 * DAY, t_end)
    W["m6"] = (t_end - 183 * DAY, t_end)
    W["d30"] = (t_end - 30 * DAY, t_end)
    W["pool"] = (ms("2020-03-01"), ms("2025-11-01"))

    T6 = t6d(D)
    rules = {"ZPB-LS-VT": zpb(D, False, True), "ZPB-LO-VT": zpb(D, True, True), "ZPB-LS-EW": zpb(D, False, False),
             "LT": lt(D)}  # fmt: skip
    for N in (5, 10, 20, 50, 100):
        rules[f"DZ{N}"] = dz(D, N)
    rules["DZE"] = np.mean([rules[f"DZ{N}"] for N in (5, 10, 20, 50, 100)], axis=0)
    rules["MON"] = mon(D)
    rules["OVR"] = ovr(D)
    rules["FUND"] = fund(D, ft, fr)
    rules["SKEW"] = skew(D, Hh)
    print("signals", round(time.time() - t0), flush=True)

    res, trs, rnd = {}, {}, {}
    for nm, tg in rules.items():
        res[nm], trs[nm] = {}, {}
        for w in WINS:
            res[nm][w], trs[nm][w] = summ(eng, tg, *W[w])
        rnd[nm] = {w: random_p(eng, tg, *W[w]) for w in ("pool",) + MAINW}
        r = res[nm]
        print(nm, {w: (r[w]["n"], round(r[w]["per_day"], 2), r[w]["win"] and round(r[w]["win"], 2),
                       round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f0.4"]["ret"], 2), r[w]["x5_f1.0"]["liq"])
                   for w in MAINW + ("m6", "d30")}, "p", round(rnd[nm]["pool"]["p"], 4), round(time.time() - t0),
              flush=True)  # fmt: skip
    # 논문 원형 참고(5배 아님): 노출 = s_LO × min(2, 25%/σ)
    ref = {w: eng.run(2 * rules["ZPB-LO-VT"], *W[w], f=1.0, lev=1) for w in WINS}
    print("REF ZPB-LO-VT as published (1x notional, cap 2)", {w: round(ref[w]["ret"], 2) for w in WINS}, flush=True)

    best = max(rules, key=lambda n: res[n]["design"]["x5_f1.0"]["ret"])
    print("BEST", best, flush=True)
    rules["C1"] = np.where(np.sign(rules[best]) == T6, rules[best], 0.0)
    nm = "C1"
    res[nm], trs[nm] = {}, {}
    for w in WINS:
        res[nm][w], trs[nm][w] = summ(eng, rules[nm], *W[w])
    rnd[nm] = {w: random_p(eng, rules[nm], *W[w]) for w in ("pool",) + MAINW}
    print("C1", {w: (res[nm][w]["n"], round(res[nm][w]["x5_f1.0"]["ret"], 2)) for w in MAINW}, rnd[nm]["pool"],
          flush=True)  # fmt: skip
    # C2
    nm = "C2"
    res[nm], trs[nm] = {}, {}
    for w in WINS:
        tr = c2_trades(b, m, D, rules[best], ft, fr, *W[w])
        s = U.summ(tr, *W[w])
        res[nm][w], trs[nm][w] = s, tr
    rnd[nm] = {w: c2_random(m, trs[nm][w], *W[w]) for w in ("pool",) + MAINW if len(trs[nm][w]) >= 5}
    print("C2", {w: (res[nm][w]["n"], round(res[nm][w]["per_day"], 2), round(res[nm][w]["x5_f1.0"]["ret"], 2))
                 for w in MAINW + ("m6", "d30")}, rnd[nm].get("pool"), flush=True)  # fmt: skip

    # 판정 + 레드팀 + 연도별 (모든 규칙)
    allnames = list(rules) + ["C2"]
    red, years, verdict = {}, {}, {}
    for nm in allnames:
        r = res[nm]
        g1 = all(r[w][f"x5_f{f}"]["ret"] > 0 for w in MAINW for f in (1.0, 0.4))
        g2 = all(r[w]["x5_f1.0"]["liq"] == 0 and r[w]["x5_f0.4"]["liq"] == 0 for w in MAINW)
        g3 = "pool" in rnd[nm] and rnd[nm]["pool"]["p"] * N_BONF < 0.05
        red[nm] = {}
        for rn, kw in (("lat15m", {"lat": 15}), ("lat60m", {"lat": 60}), ("slip+0.02", {"slip": SLIP + 0.0002}),
                       ("fee1.5", {"fee_mult": 1.5})):  # fmt: skip
            red[nm][rn] = {}
            for w in ("val", "year"):
                if nm == "C2":
                    red[nm][rn][w] = U.summ(c2_trades(b, m, D, rules[best], ft, fr, *W[w], **kw), *W[w])["x5_f1.0"]
                else:
                    red[nm][rn][w] = eng.run(rules[nm], *W[w], f=1.0, **kw)
        g4 = all(red[nm][k][w]["ret"] > 0 for k in red[nm] for w in ("val", "year"))
        verdict[nm] = {"g1": g1, "g2": g2, "g3": g3, "g4_red": g4, "pass": bool(g1 and g2 and g3 and g4)}
        years[nm] = {}
        for y in range(2020, 2027):
            a_, z_ = max(ms(f"{y}-01-01"), ms("2020-03-01")), min(ms(f"{y + 1}-01-01"), t_end)
            if nm == "C2":
                years[nm][y] = U.summ(c2_trades(b, m, D, rules[best], ft, fr, a_, z_), a_, z_)["x5_f1.0"]["ret"]
            else:
                years[nm][y] = eng.run(rules[nm], a_, z_, f=1.0)["ret"]
        print("V", nm, verdict[nm], {k: {w: round(v["ret"], 2) for w, v in red[nm][k].items()} for k in red[nm]},
              {y: round(v, 2) for y, v in years[nm].items()}, flush=True)  # fmt: skip

    json.dump({"windows": W, "t_end": t_end, "res": res, "rnd": rnd, "ref_published": ref, "best": best,
               "red": red, "years": years, "verdict": verdict, "n_bonf": N_BONF,
               "n_cum": N_BEFORE + N_BONF}, (out / "result.json").open("w"), default=float)  # fmt: skip
    json.dump({nm: {w: trs[nm][w] for w in ("d30", "fresh", "design", "val", "year", "m6")} for nm in allnames},
              (out / "trades.json").open("w"), default=float)  # fmt: skip
    curves = {}
    for nm in rules:
        curves[nm] = {}
        for w in MAINW + ("d30",):
            for f in (1.0, 0.4):
                cv = []
                eng.run(rules[nm], *W[w], f=f, curve=cv)
                curves[nm][f"{w}|{f}"] = cv
    json.dump(curves, (out / "curves.json").open("w"))
    np.savez(out / "targets.npz", T=D["T"], **{k: v for k, v in rules.items()})
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
