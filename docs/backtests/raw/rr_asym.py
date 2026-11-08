"""손익비 비대칭(짧은 손절·긴 익절): 시계 없는 신호 4종 × 일봉 방향 문 2 × 손절/익절 6 (1분봉 체결).

등록: docs/backtests/2026-11-08-btc-rr-asym.md (52개). 연구 전용(주문 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/rr-asym-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/rr_asym.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import daily_wave as DW
import numpy as np
import ql_upgrade as U
import regime_lines as RL
from btc_fractal import scr3
from dmi15_eval import frame
from numpy.lib.stride_tricks import sliding_window_view

MIN, M15, DAY = U.MIN, U.M15, U.DAY
TAKER, MAKER, TICK, SLIP = U.TAKER, U.MAKER, U.TICK, 0.0001
HOLD = 48 * 60
N_BONF, N_BEFORE = 52, 6151
STOPS, TPS = (0.003, 0.005), (0.01, 0.02, 0.03)
DX, DY = (0.005, 0.01, 0.02, 0.03), (0.003, 0.005, 0.01)
WINS = ("fresh", "design", "val", "year", "m6", "d30", "pool")
MAINW = ("fresh", "design", "val", "year")


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def windows(t_end: int) -> dict:
    w = {k: (ms(a), t_end if z is None else ms(z)) for k, (a, z) in DW.WIN.items()}
    w["m6"] = (t_end - 183 * DAY, t_end)
    w["d30"] = (t_end - 30 * DAY, t_end)
    w["pool"] = (ms("2020-03-01"), ms("2025-11-01"))
    return w


# ---------------- 신호 ----------------
def first_cross(c, lvl_hi, lvl_lo):
    with np.errstate(invalid="ignore"):
        up = c > lvl_hi
        dn = c < lvl_lo
    up_f = up & ~np.r_[False, up[:-1]]
    dn_f = dn & ~np.r_[False, dn[:-1]]
    return up_f, dn_f


def donchian_sig(b, N):
    h, lo, c = b["h"], b["l"], b["c"]
    n = len(c)
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    hh[N:] = sliding_window_view(h, N).max(1)[:-1]  # 직전 N봉(현재 봉 제외)
    ll[N:] = sliding_window_view(lo, N).min(1)[:-1]
    return first_cross(c, hh, ll)


def signals(b, F) -> dict:
    n = len(b["c"])
    out = {"SCR15": (F["scr_l"], F["scr_s"])}
    H1 = frame(b, 60)
    l1, s1, *_ = scr3(H1)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    L[H1["last15"][l1]] = True
    S[H1["last15"][s1]] = True
    out["SCR1h"] = (L, S)
    out["DC4h"] = donchian_sig(b, 16)
    out["DC24h"] = donchian_sig(b, 96)
    sets = {}
    gt = F["gate"]["GT"]
    for k, (L, S) in out.items():
        i = np.r_[np.nonzero(L)[0], np.nonzero(S)[0]]
        d = np.r_[np.ones(L.sum(), int), -np.ones(S.sum(), int)]
        o = np.argsort(i, kind="stable")
        i, d = i[o], d[o]
        sets[f"{k}|G0"] = (i, d)
        keep = gt[i] == d
        sets[f"{k}|GT"] = (i[keep], d[keep])
    return sets


# ---------------- 체결 ----------------
class Sim:
    def __init__(self, m, b, fund_fn):
        self.m, self.b, self.fund_fn = m, b, fund_fn
        self.n1 = len(m["t"])

    def kmin(self, i15, lat=0):
        ts = int(self.b["t"][i15]) + M15 + lat * MIN
        k = int(np.searchsorted(self.m["t"], ts))
        if k >= self.n1 - 2 or int(self.m["t"][k]) - ts > 5 * MIN:  # 빈 구간이면 버린다
            return -1
        return k

    def trade(self, k, d, S, T, mode="fix", fee_mult=1.0, slip=SLIP):
        m = self.m
        ep = float(m["o"][k])
        stop0 = ep * (1 - d * S)
        tp = ep * (1 + d * T) if mode != "TR" else None
        tend = min(k + HOLD, self.n1 - 1)
        h, lo = m["h"][k : tend + 1], m["l"][k : tend + 1]
        if mode == "fix":
            hit = U.first_hit(m, d, k, tend, stop0, tp, k)
            j, why = (None, None) if hit is None else hit
        else:
            fav = np.maximum.accumulate(h) if d == 1 else np.minimum.accumulate(lo)
            prev = np.r_[ep, fav[:-1]]  # 앞 분봉까지의 최고 유리가(보수적)
            prev = np.maximum(prev, ep) if d == 1 else np.minimum(prev, ep)
            armed = (prev >= ep * 1.01) if d == 1 else (prev <= ep * 0.99)
            if mode == "BE":
                lvl = np.where(armed, ep * (1 + d * 0.001), stop0)
            else:
                trail = prev * (1 - d * 2 * S)
                lvl = np.where(armed, np.maximum(trail, stop0) if d == 1 else np.minimum(trail, stop0), stop0)
            sh = (lo <= lvl) if d == 1 else (h >= lvl)
            th = np.zeros(len(h), bool) if tp is None else ((h >= tp) if d == 1 else (lo <= tp))
            ks = int(np.argmax(sh)) if sh.any() else None
            kt = int(np.argmax(th)) if th.any() else None
            if ks is None and kt is None:
                j, why = None, None
            elif kt is None or (ks is not None and ks <= kt):
                j, why = k + ks, "stop"
                stop0 = float(lvl[ks])
            else:
                j, why = k + kt, "tp"
        if j is None:
            j, why, px = tend, "time", float(m["c"][tend])
        elif why == "stop":
            o = float(m["o"][j])
            px = (min(stop0, o) if d == 1 else max(stop0, o)) - d * TICK
        else:
            px = tp
        fin = (TAKER * fee_mult) + slip
        fout = MAKER * fee_mult if why == "tp" else TAKER * fee_mult + slip
        jj = j - k
        worst = (lo[: jj + 1].min() / ep - 1) if d == 1 else (1 - h[: jj + 1].max() / ep)
        t_in, t_out = int(m["t"][k]), int(m["t"][j]) + MIN
        fund = self.fund_fn(t_in, t_out, d)
        r = d * (px / ep - 1) - fin - fout + fund
        return {"k": int(k), "x": int(j), "d": int(d), "ep": ep, "xp": float(px), "r": float(r),
                "R": float(r / S), "worst": float(min(worst, r)), "why": why, "size": 1.0,
                "t_in": t_in, "t_out": t_out}  # fmt: skip

    def run(self, sig, a_ms, z_ms, S, T, mode="fix", lat=0, **kw):
        i, d = sig
        t = self.b["t"]
        sel = (t[i] >= a_ms) & (t[i] + M15 < z_ms)
        out, busy = [], -1
        for ii, dd in zip(i[sel], d[sel]):
            k = self.kmin(ii, lat)
            if k < 0 or k <= busy:
                continue
            x = self.trade(k, int(dd), S, T, mode, **kw)
            if x["t_out"] > z_ms:
                continue
            x["i15"] = int(ii)
            out.append(x)
            busy = x["x"]
        return out


def summ(tr, a_ms, z_ms) -> dict:
    s = U.summ(tr, a_ms, z_ms)
    s["avgR"] = float(np.mean([x["R"] for x in tr])) if tr else None
    s["why"] = {w: int(sum(x["why"] == w for x in tr)) for w in ("stop", "tp", "time")}
    return s


# ---------------- 무작위 ----------------
def by_slot_index(b, a_ms, z_ms):
    t = b["t"]
    a, z = int(np.searchsorted(t, a_ms)), int(np.searchsorted(t, z_ms - HOLD * MIN - M15))
    idx = np.arange(a, z)
    slot = (t[idx] % DAY).astype(np.int64)
    return {int(s): idx[slot == s] for s in np.unique(slot)}


def random_p(sim, tr, S, T, mode, a_ms, z_ms, nsim, seed) -> dict:
    rng = np.random.default_rng(seed)
    bs = by_slot_index(sim.b, a_ms, z_ms)
    t = sim.b["t"]
    real = float(np.mean([x["r"] for x in tr]))
    slots = [int(t[x["i15"]] % DAY) for x in tr]
    ds = [x["d"] for x in tr]
    vals, comp = [], []
    for _ in range(nsim):
        rs = []
        for s, d in zip(slots, ds):
            pool = bs.get(s)
            if pool is None or not len(pool):
                continue
            k = sim.kmin(int(rng.choice(pool)))
            if k < 0:
                continue
            rs.append(sim.trade(k, d, S, T, mode)["r"])
        vals.append(np.mean(rs))
        comp.append(rs)
    vals = np.array(vals)
    return {"real": real, "rnd_mean": float(vals.mean()), "rnd_sd": float(vals.std()),
            "p": float((np.sum(vals >= real) + 1) / (nsim + 1))}  # fmt: skip


# ---------------- 진단: +X가 −Y보다 먼저 ----------------
def first_passage(m, ks, ds, chunk=800) -> dict:
    """반환 {(X,Y): 성공 비율 배열 요소}. 같은 분이면 −Y 먼저."""
    h, lo, o = m["h"], m["l"], m["o"]
    n1 = len(h)
    ks = np.asarray(ks)
    ds = np.asarray(ds)
    ok = ks + HOLD < n1
    ks, ds = ks[ok], ds[ok]
    res = {(x, y): np.zeros(len(ks), bool) for x in DX for y in DY}
    Hv, Lv = sliding_window_view(h, HOLD), sliding_window_view(lo, HOLD)
    for s0 in range(0, len(ks), chunk):
        kk, dd = ks[s0 : s0 + chunk], ds[s0 : s0 + chunk]
        ep = o[kk][:, None]
        Hc, Lc = Hv[kk] / ep - 1, Lv[kk] / ep - 1
        up = np.where(dd[:, None] == 1, Hc, -Lc)  # 유리한 쪽 최대 이동
        dn = np.where(dd[:, None] == 1, -Lc, Hc)  # 불리한 쪽 최대 이동
        upm = {x: np.where((up >= x).any(1), np.argmax(up >= x, 1), HOLD + 1) for x in DX}
        dnm = {y: np.where((dn >= y).any(1), np.argmax(dn >= y, 1), HOLD + 1) for y in DY}
        for x in DX:
            for y in DY:
                res[(x, y)][s0 : s0 + chunk] = (upm[x] < dnm[y]) & (upm[x] <= HOLD)
        del Hc, Lc, up, dn
    return {f"{x}|{y}": float(v.mean()) for (x, y), v in res.items()} | {"n": int(len(ks))}


def diagnostics(sim, sets, a_ms, z_ms, seed=11, cap=12000) -> dict:
    rng = np.random.default_rng(seed)
    t = sim.b["t"]
    bs = by_slot_index(sim.b, a_ms, z_ms)
    out = {}
    for name, (i, d) in sets.items():
        sel = (t[i] >= a_ms) & (t[i] + HOLD * MIN < z_ms)
        i, d = i[sel], d[sel]
        if len(i) > cap:
            p = np.sort(rng.choice(len(i), cap, replace=False))
            i, d = i[p], d[p]
        ks = np.array([sim.kmin(int(x)) for x in i])
        good = ks >= 0
        real = first_passage(sim.m, ks[good], d[good])
        ri = np.array([rng.choice(bs[int(t[x] % DAY)]) for x in i[good]])
        rk = np.array([sim.kmin(int(x)) for x in ri])
        g2 = rk >= 0
        rnd = first_passage(sim.m, rk[g2], d[good][g2])
        out[name] = {"sig": real, "rnd": rnd}
        print("diag", name, real["n"], {k: (round(real[k], 3), round(rnd[k], 3))
                                        for k in ("0.01|0.003", "0.02|0.003", "0.02|0.005", "0.03|0.005")},
              flush=True)  # fmt: skip
    return out


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

    sim = Sim(m, b, fund_fn)
    W = windows(t_end)
    sets = signals(b, F)
    print("loaded", len(m["t"]), {k: len(v[0]) for k, v in sets.items()}, round(time.time() - t0), flush=True)

    res, trades = {}, {}
    spec = {}
    for sn in sets:
        for S in STOPS:
            for T in TPS:
                spec[f"{sn}|S{S * 100:g}|T{T * 100:g}|fix"] = (sn, S, T, "fix")

    def evaluate(name, wins=WINS, **kw):
        sn, S, T, mode = spec[name]
        r, k = {}, {}
        for w in wins:
            tr = sim.run(sets[sn], *W[w], S, T, mode, **kw)
            r[w] = summ(tr, *W[w])
            k[w] = tr
        return r, k

    def show(name, r):
        print(name, {w: (r[w]["n"], round(r[w]["per_day"], 2), r[w]["win"] and round(r[w]["win"], 2),
                         round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f0.4"]["ret"], 2))
                     for w in MAINW + ("d30",)}, round(time.time() - t0), flush=True)  # fmt: skip

    for name in list(spec):
        res[name], trades[name] = evaluate(name)
        show(name, res[name])

    def pick(names):
        ok = [n for n in names if res[n]["design"]["per_day"] >= 0.8]
        if not ok:
            return None
        top = max(res[n]["design"]["x5_f1.0"]["ret"] for n in ok)
        near = [n for n in ok if res[n]["design"]["x5_f1.0"]["ret"] >= top - 0.10 * abs(top)]
        return min(near, key=lambda n: res[n]["design"]["x5_f1.0"]["mdd"])

    base = list(spec)
    ok = sorted([n for n in base if res[n]["design"]["per_day"] >= 0.8],
                key=lambda n: -res[n]["design"]["x5_f1.0"]["ret"])  # fmt: skip
    top2 = ok[:2]
    for n in top2:
        sn, S, T, _ = spec[n]
        for mode in ("BE", "TR"):
            nm = f"{sn}|S{S * 100:g}|T{T * 100:g}|{mode}" if mode == "BE" else f"{sn}|S{S * 100:g}|{mode}"
            spec[nm] = (sn, S, T, mode)
            res[nm], trades[nm] = evaluate(nm)
            show(nm, res[nm])
    fin = pick(list(spec)) or max(spec, key=lambda n: res[n]["design"]["x5_f1.0"]["ret"])
    print("PICK", fin, flush=True)
    sn, S, T, mode = spec[fin]

    # 무작위
    rnd = {}
    for w in ("pool",) + MAINW:
        tr = trades[fin][w]
        if len(tr) < 5:
            continue
        rnd[w] = random_p(sim, tr, S, T, mode, *W[w], 200 if w == "pool" else 60, 7)
        print("rnd", w, rnd[w], round(time.time() - t0), flush=True)
    # 레드팀
    red = {}
    for nm, kw in (("lat1m", {"lat": 1}), ("lat15m", {"lat": 15}), ("slip+0.02", {"slip": SLIP + 0.0002}),
                   ("fee1.5", {"fee_mult": 1.5})):  # fmt: skip
        red[nm] = {}
        for w in MAINW + ("m6", "d30"):
            tr = sim.run(sets[sn], *W[w], S, T, mode, **kw)
            red[nm][w] = summ(tr, *W[w])
        print("red", nm, {w: round(red[nm][w]["x5_f1.0"]["ret"], 2) for w in red[nm]}, flush=True)
    # 연도별
    years = {}
    for y in range(2020, 2027):
        a_, z_ = ms(f"{y}-01-01"), min(ms(f"{y + 1}-01-01"), t_end)
        tr = sim.run(sets[sn], a_, z_, S, T, mode)
        years[y] = summ(tr, a_, z_)
    print("years", {y: round(v["x5_f1.0"]["ret"], 2) for y, v in years.items()}, flush=True)
    # 진단
    diag = diagnostics(sim, sets, *W["pool"])

    # 판정
    fx = res[fin]
    g1 = all(fx[w][f"x5_f{f}"]["ret"] > 0 for w in MAINW for f in (1.0, 0.4))
    g2 = all(fx[w]["x5_f1.0"]["liq"] == 0 for w in MAINW)
    g3 = "pool" in rnd and rnd["pool"]["p"] * N_BONF < 0.05
    g4 = fx["design"]["per_day"] >= 0.8 and fx["year"]["per_day"] >= 0.8
    g5 = all(red[k][w]["x5_f1.0"]["ret"] > 0 for k in red for w in ("val", "year"))
    verdict = {"g1_all_windows": g1, "g2_no_liq": g2, "g3_random": g3, "g4_freq": g4, "g5_red": g5,
               "pass": bool(g1 and g2 and g3 and g4 and g5)}  # fmt: skip
    print("VERDICT", verdict, flush=True)
    n_pos = {w: sum(res[n][w]["x5_f1.0"]["ret"] > 0 for n in spec) for w in MAINW}

    json.dump({"windows": W, "spec": spec, "res": res, "final": fin, "top2": top2, "rnd": rnd, "red": red,
               "years": years, "diag": diag, "verdict": verdict, "n_pos": n_pos,
               "n_bonf": N_BONF, "n_cum": N_BEFORE + N_BONF}, (out / "result.json").open("w"), default=float)  # fmt: skip
    json.dump({"final": {w: trades[fin][w] for w in ("d30", "pool", "year", "fresh", "design", "val", "m6")}},
              (out / "trades.json").open("w"))  # fmt: skip
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
