"""손익비 비대칭 2차: 넓은 손절(1~2%, ATR) × 익절/추적/BE × 보유 48·96시간 (1분봉 체결).

등록: docs/backtests/2026-11-08-btc-rr-asym-2.md (53개). 연구 전용(주문 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/rr-asym2-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/rr_asym2.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import daily_wave as DW
import numpy as np
import ql_upgrade as U
import regime_lines as RL
import rr_asym as A
from numpy.lib.stride_tricks import sliding_window_view

MIN, M15, DAY = U.MIN, U.M15, U.DAY
TAKER, MAKER, TICK, SLIP = U.TAKER, U.MAKER, U.TICK, A.SLIP
N_BONF, N_BEFORE = 53, 6203
ATR_CAP = 0.035
STOPS = ("S1", "S1.5", "S2", "A0.75", "A1")
DX, DY = (0.02, 0.03, 0.05), (0.01, 0.015, 0.02)
WINS, MAINW = A.WINS, A.MAINW


def tp_of(stop: str, tier: int, S: float) -> float:
    """tier 1~3. 고정 % 손절은 2/3/5%, ATR 손절은 1.5/2/3R."""
    if stop.startswith("S"):
        return (0.02, 0.03, 0.05)[tier - 1]
    return (1.5, 2.0, 3.0)[tier - 1] * S


class Sim(A.Sim):
    def __init__(self, m, b, fund_fn, atrp):
        super().__init__(m, b, fund_fn)
        self.atrp = atrp

    def stop_of(self, stop: str, i15: int) -> float:
        if stop.startswith("S"):
            return float(stop[1:]) / 100
        a = self.atrp[i15]
        if not np.isfinite(a) or a <= 0:
            return float("nan")
        return min(float(stop[1:]) * a, ATR_CAP)

    def trade2(self, k, d, S, T, mode, hold, fee_mult=1.0, slip=SLIP):
        m = self.m
        ep = float(m["o"][k])
        stop0 = ep * (1 - d * S)
        tp = None if mode == "TR" else ep * (1 + d * T)
        tend = min(k + hold, self.n1 - 1)
        h, lo = m["h"][k : tend + 1], m["l"][k : tend + 1]
        if mode == "fix":
            hit = U.first_hit(m, d, k, tend, stop0, tp, k)
            j, why = (None, None) if hit is None else hit
            lvl_hit = stop0
        else:
            fav = np.maximum.accumulate(h) if d == 1 else np.minimum.accumulate(lo)
            prev = np.r_[ep, fav[:-1]]
            prev = np.maximum(prev, ep) if d == 1 else np.minimum(prev, ep)
            armed = (prev >= ep * (1 + S)) if d == 1 else (prev <= ep * (1 - S))
            if mode == "BE":
                lvl = np.where(armed, ep * (1 + d * 0.001), stop0)
            else:
                trail = prev * (1 - d * S)
                lvl = np.where(armed, np.maximum(trail, stop0) if d == 1 else np.minimum(trail, stop0), stop0)
            sh = (lo <= lvl) if d == 1 else (h >= lvl)
            th = np.zeros(len(h), bool) if tp is None else ((h >= tp) if d == 1 else (lo <= tp))
            ks = int(np.argmax(sh)) if sh.any() else None
            kt = int(np.argmax(th)) if th.any() else None
            lvl_hit = stop0
            if ks is None and kt is None:
                j, why = None, None
            elif kt is None or (ks is not None and ks <= kt):
                j, why, lvl_hit = k + ks, "stop", float(lvl[ks])
            else:
                j, why = k + kt, "tp"
        if j is None:
            j, why, px = tend, "time", float(m["c"][tend])
        elif why == "stop":
            o = float(m["o"][j])
            px = (min(lvl_hit, o) if d == 1 else max(lvl_hit, o)) - d * TICK
        else:
            px = tp
        fin = TAKER * fee_mult + slip
        fout = MAKER * fee_mult if why == "tp" else TAKER * fee_mult + slip
        jj = j - k
        worst = (lo[: jj + 1].min() / ep - 1) if d == 1 else (1 - h[: jj + 1].max() / ep)
        t_in, t_out = int(m["t"][k]), int(m["t"][j]) + MIN
        fund = self.fund_fn(t_in, t_out, d)
        r = d * (px / ep - 1) - fin - fout + fund
        return {"k": int(k), "x": int(j), "d": int(d), "ep": ep, "xp": float(px), "r": float(r),
                "R": float(r / S), "S": float(S), "T": None if T is None else float(T),
                "worst": float(min(worst, r)), "why": why, "size": 1.0, "t_in": t_in, "t_out": t_out}  # fmt: skip

    def run2(self, sig, a_ms, z_ms, stop, tier, mode, hold, lat=0, **kw):
        i, d = sig
        t = self.b["t"]
        sel = (t[i] >= a_ms) & (t[i] + M15 < z_ms)
        out, busy = [], -1
        for ii, dd in zip(i[sel], d[sel]):
            k = self.kmin(ii, lat)
            if k < 0 or k <= busy:
                continue
            S = self.stop_of(stop, int(ii))
            if not np.isfinite(S):
                continue
            T = None if mode == "TR" else tp_of(stop, tier, S)
            x = self.trade2(k, int(dd), S, T, mode, hold, **kw)
            if x["t_out"] > z_ms:
                continue
            x["i15"] = int(ii)
            out.append(x)
            busy = x["x"]
        return out


def slot_pool(b, a_ms, z_ms, hold):
    t = b["t"]
    a, z = int(np.searchsorted(t, a_ms)), int(np.searchsorted(t, z_ms - hold * MIN - M15))
    idx = np.arange(a, z)
    slot = (t[idx] % DAY).astype(np.int64)
    return {int(s): idx[slot == s] for s in np.unique(slot)}


def random_p(sim, tr, mode, hold, a_ms, z_ms, nsim, seed) -> dict:
    rng = np.random.default_rng(seed)
    bs = slot_pool(sim.b, a_ms, z_ms, hold)
    t = sim.b["t"]
    real = float(np.mean([x["r"] for x in tr]))
    vals = []
    for _ in range(nsim):
        rs = []
        for x in tr:
            pool = bs.get(int(t[x["i15"]] % DAY))
            if pool is None or not len(pool):
                continue
            k = sim.kmin(int(rng.choice(pool)))
            if k < 0:
                continue
            rs.append(sim.trade2(k, x["d"], x["S"], x["T"], mode, hold)["r"])
        vals.append(np.mean(rs))
    vals = np.array(vals)
    return {"real": real, "rnd_mean": float(vals.mean()), "rnd_sd": float(vals.std()),
            "p": float((np.sum(vals >= real) + 1) / (nsim + 1)), "n": len(tr)}  # fmt: skip


def first_passage(m, ks, ds, hold, chunk=400) -> dict:
    h, lo, o = m["h"], m["l"], m["o"]
    n1 = len(h)
    ks, ds = np.asarray(ks), np.asarray(ds)
    ok = ks + hold < n1
    ks, ds = ks[ok], ds[ok]
    res = {(x, y): np.zeros(len(ks), bool) for x in DX for y in DY}
    Hv, Lv = sliding_window_view(h, hold), sliding_window_view(lo, hold)
    for s0 in range(0, len(ks), chunk):
        kk, dd = ks[s0 : s0 + chunk], ds[s0 : s0 + chunk]
        ep = o[kk][:, None]
        Hc, Lc = Hv[kk] / ep - 1, Lv[kk] / ep - 1
        up = np.where(dd[:, None] == 1, Hc, -Lc)
        dn = np.where(dd[:, None] == 1, -Lc, Hc)
        del Hc, Lc
        upm = {x: np.where((up >= x).any(1), np.argmax(up >= x, 1), hold + 1) for x in DX}
        dnm = {y: np.where((dn >= y).any(1), np.argmax(dn >= y, 1), hold + 1) for y in DY}
        for x in DX:
            for y in DY:
                res[(x, y)][s0 : s0 + chunk] = (upm[x] < dnm[y]) & (upm[x] <= hold)
        del up, dn
    return {f"{x}|{y}": float(v.mean()) for (x, y), v in res.items()} | {"n": int(len(ks))}


def diagnostics(sim, sets, names, a_ms, z_ms, seed=11, cap=8000) -> dict:
    rng = np.random.default_rng(seed)
    t = sim.b["t"]
    out = {}
    for hold in (2880, 5760):
        bs = slot_pool(sim.b, a_ms, z_ms, hold)
        for name in names:
            i, d = sets[name]
            sel = (t[i] >= a_ms) & (t[i] + hold * MIN < z_ms)
            i, d = i[sel], d[sel]
            if len(i) > cap:
                p = np.sort(rng.choice(len(i), cap, replace=False))
                i, d = i[p], d[p]
            ks = np.array([sim.kmin(int(x)) for x in i])
            g = ks >= 0
            real = first_passage(sim.m, ks[g], d[g], hold)
            ri = np.array([rng.choice(bs[int(t[x] % DAY)]) for x in i[g]])
            rk = np.array([sim.kmin(int(x)) for x in ri])
            g2 = rk >= 0
            rnd = first_passage(sim.m, rk[g2], d[g][g2], hold)
            out[f"{name}|H{hold // 60}"] = {"sig": real, "rnd": rnd}
            print("diag", name, hold // 60, real["n"],
                  {k: (round(real[k], 3), round(rnd[k], 3)) for k in ("0.03|0.01", "0.05|0.01", "0.05|0.02")},
                  flush=True)  # fmt: skip
    return out


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

    atrp = F["atr_d"] / b["c"]
    sim = Sim(m, b, fund_fn, atrp)
    W = A.windows(t_end)
    sets_all = A.signals(b, F)
    sets = {k: sets_all[k] for k in ("DC24h|GT", "DC24h|G0", "DC4h|GT")}
    print("loaded", {k: len(v[0]) for k, v in sets.items()}, round(time.time() - t0), flush=True)

    # 1) 진단 먼저
    diag = diagnostics(sim, sets, list(sets), *W["pool"])
    json.dump({"diag": diag}, (out / "diag.json").open("w"))

    # 2) 격자
    spec = {}
    for st in STOPS:
        for ex in ("T1", "T2", "T3", "TR"):
            spec[f"DC24h|GT|{st}|{ex}|H48"] = ("DC24h|GT", st, ex, 2880)
        for ex in ("T3", "TR"):
            spec[f"DC24h|GT|{st}|{ex}|H96"] = ("DC24h|GT", st, ex, 5760)
        for sn in ("DC24h|G0", "DC4h|GT"):
            for ex in ("T2", "TR"):
                spec[f"{sn}|{st}|{ex}|H48"] = (sn, st, ex, 2880)

    def mode_tier(ex):
        if ex == "TR":
            return "TR", 0
        if ex.endswith("BE"):
            return "BE", int(ex[1])
        return "fix", int(ex[1])

    res, trades = {}, {}

    def evaluate(name, wins=WINS, **kw):
        sn, st, ex, hold = spec[name]
        mode, tier = mode_tier(ex)
        r, k = {}, {}
        for w in wins:
            tr = sim.run2(sets[sn], *W[w], st, tier, mode, hold, **kw)
            r[w] = A.summ(tr, *W[w])
            if tr:
                Ss = np.array([x["S"] for x in tr])
                r[w]["S_q"] = [float(v) for v in np.quantile(Ss, (0.1, 0.5, 0.9))]
            k[w] = tr
        return r, k

    def show(name, r):
        print(name, {w: (r[w]["n"], round(r[w]["per_day"], 2), r[w]["win"] and round(r[w]["win"], 2),
                         round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f0.4"]["ret"], 2), r[w]["x5_f1.0"]["liq"])
                     for w in MAINW + ("d30",)}, round(time.time() - t0), flush=True)  # fmt: skip

    for name in list(spec):
        res[name], trades[name] = evaluate(name)
        show(name, res[name])

    # E: BE — DC24h·GT의 T ≥ 2R 고정 익절 중 설계 상위 3개
    def r_mult(st, ex):
        tier = int(ex[1])
        if st.startswith("A"):
            return (1.5, 2.0, 3.0)[tier - 1]
        return (0.02, 0.03, 0.05)[tier - 1] / (float(st[1:]) / 100)

    elig = [n for n, (sn, st, ex, h) in spec.items()
            if sn == "DC24h|GT" and ex != "TR" and r_mult(st, ex) >= 2.0 and res[n]["design"]["n"] >= 100]  # fmt: skip
    elig.sort(key=lambda n: -res[n]["design"]["x5_f1.0"]["ret"])
    for n in elig[:3]:
        sn, st, ex, h = spec[n]
        nm = f"{sn}|{st}|{ex}BE|H{h // 60}"
        spec[nm] = (sn, st, ex + "BE", h)
        res[nm], trades[nm] = evaluate(nm)
        show(nm, res[nm])

    ok = [n for n in spec if res[n]["design"]["n"] >= 100]
    top = max(res[n]["design"]["x5_f1.0"]["ret"] for n in ok)
    near = [n for n in ok if res[n]["design"]["x5_f1.0"]["ret"] >= top - 0.10 * abs(top)]
    fin = min(near, key=lambda n: res[n]["design"]["x5_f1.0"]["mdd"])
    print("PICK", fin, flush=True)
    sn, st, ex, hold = spec[fin]
    mode, tier = mode_tier(ex)

    rnd = {}
    for w in ("pool",) + MAINW:
        tr = trades[fin][w]
        if len(tr) < 5:
            continue
        rnd[w] = random_p(sim, tr, mode, hold, *W[w], 200 if w == "pool" else 60, 7)
        print("rnd", w, rnd[w], round(time.time() - t0), flush=True)
    red = {}
    for nm, kw in (("lat1m", {"lat": 1}), ("lat15m", {"lat": 15}), ("slip+0.02", {"slip": SLIP + 0.0002}),
                   ("fee1.5", {"fee_mult": 1.5})):  # fmt: skip
        red[nm] = {}
        for w in MAINW + ("m6", "d30"):
            tr = sim.run2(sets[sn], *W[w], st, tier, mode, hold, **kw)
            red[nm][w] = A.summ(tr, *W[w])
        print("red", nm, {w: round(red[nm][w]["x5_f1.0"]["ret"], 2) for w in red[nm]}, flush=True)
    years = {}
    for y in range(2020, 2027):
        a_, z_ = A.ms(f"{y}-01-01"), min(A.ms(f"{y + 1}-01-01"), t_end)
        years[y] = A.summ(sim.run2(sets[sn], a_, z_, st, tier, mode, hold), a_, z_)
    print("years", {y: round(v["x5_f1.0"]["ret"], 2) for y, v in years.items()}, flush=True)

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
    n_all4 = [n for n in spec if all(res[n][w]["x5_f1.0"]["ret"] > 0 for w in MAINW)]
    print("n_pos", n_pos, "all4", n_all4, flush=True)
    json.dump({"windows": W, "spec": spec, "res": res, "final": fin, "rnd": rnd, "red": red, "years": years,
               "diag": diag, "verdict": verdict, "n_pos": n_pos, "all4": n_all4,
               "n_bonf": N_BONF, "n_cum": N_BEFORE + N_BONF}, (out / "result.json").open("w"), default=float)  # fmt: skip
    json.dump({"final": {w: trades[fin][w] for w in ("d30", "pool", "year", "fresh", "design", "val", "m6")}},
              (out / "trades.json").open("w"))  # fmt: skip
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
