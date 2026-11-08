"""SCR 다섯 프레임 일치(5분·15분·30분·1시간·4시간) 진입 + 상위 프레임 청산. 1분봉 체결.

등록: docs/backtests/2026-11-08-btc-scr-mtf5-hypotheses.md. 연구 전용(주문 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/scr-mtf5-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/scr_mtf5.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import daily_wave as DW
import numpy as np
import r5_data
from btc_fractal import cci, crossed, recent, rsi, stoch

MIN, DAY = 60_000, 86_400_000
TAKER, MAKER, TICK = 0.0005, 0.0002, 0.1
LEV, LIQ, DSTOP = 5, -0.95, 0.02
DIR1M = Path("~/.minerva/data/btc/futures/1m").expanduser()
FRAMES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
FIVE = ("5m", "15m", "30m", "1h", "4h")
ENTRIES = ("D5", "D4", "S5k3", "S5k6", "S15k3", "M5", "M15", "M5NH", "M15NH")
EXITS = ("XH1", "XH4", "XT", "XP")
N_BONF, N_BEFORE = 40, 6111


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


# ---------------- 자료 ----------------
def load_1m() -> dict:
    parts = [np.load(p) for p in sorted(DIR1M.glob("*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("t", "o", "h", "l", "c")}


def frame(m: dict, mins: int) -> dict:
    """UTC 경계 봉. te = 마감 시각(ms). 지표·방향·3단 신호와 '최근 신호 위치'."""
    P = mins * MIN
    k = m["t"] // P
    st = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    en = np.r_[st[1:] - 1, len(k) - 1]
    f = {"t": k[st] * P, "o": m["o"][st], "c": m["c"][en],
         "h": np.maximum.reduceat(m["h"], st), "l": np.minimum.reduceat(m["l"], st)}  # fmt: skip
    f["te"] = f["t"] + P
    h, lo, c = f["h"], f["l"], f["c"]
    f["k"], f["cci"], f["rsi"] = stoch(h, lo, c, 5), cci(h, lo, c, 20), rsi(c, 14)
    with np.errstate(invalid="ignore"):
        up = (f["cci"] > 0) & (f["rsi"] > 50)
        dn = (f["cci"] < 0) & (f["rsi"] < 50)
    f["dir"] = np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)
    Lx = recent(crossed(f["k"], 20, True)) & recent(crossed(f["cci"], -100, True)) & recent(crossed(f["rsi"], 30, True))
    Sx = recent(crossed(f["k"], 80, False)) & recent(crossed(f["cci"], 100, False)) & recent(crossed(f["rsi"], 70, False))
    f["L"] = Lx & ~np.r_[False, Lx[:-1]]
    f["S"] = Sx & ~np.r_[False, Sx[:-1]]
    idx = np.arange(len(c))
    f["lastL"] = np.maximum.accumulate(np.where(f["L"], idx, -10**9))
    f["lastS"] = np.maximum.accumulate(np.where(f["S"], idx, -10**9))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    atr = np.empty_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / 14
    f["atr"] = atr
    return f


def at(f: dict, tc: np.ndarray) -> np.ndarray:
    """시각 tc에 이미 마감된 마지막 봉 위치(없으면 -1)."""
    return np.searchsorted(f["te"], tc, "right") - 1


# ---------------- 진입 신호 ----------------
def events(F: dict, E: str) -> list:
    """(결정 시각 ms, 방향, 신호 종가, 트리거 프레임) 목록."""
    trig = "15m" if E.startswith(("S15", "M15")) else "5m"
    T = F[trig]
    tc = T["te"]
    J = {n: at(F[n], tc) for n in FRAMES}
    ok = np.ones(len(tc), bool)
    for n in FRAMES:
        ok &= J[n] >= 30  # 지표가 자리 잡은 뒤

    def g(n, key):
        return F[n][key][np.maximum(J[n], 0)]

    out_d = np.zeros(len(tc), np.int8)
    if E in ("D5", "D4"):
        D = np.stack([g(n, "dir") for n in FIVE])
        nu, nd = (D == 1).sum(0), (D == -1).sum(0)
        need = 5 if E == "D5" else 4
        lo = (nu >= need) & (nd == 0)
        sh = (nd >= need) & (nu == 0)
        lo &= ~np.r_[False, lo[:-1]]
        sh &= ~np.r_[False, sh[:-1]]
        out_d[lo & ok], out_d[sh & ok] = 1, -1
    else:
        Lt, St = T["L"], T["S"]
        others = [n for n in FIVE if n != trig]
        if E.startswith("S"):
            kk = int(E[-1])
            cl, cs = Lt.copy(), St.copy()
            for n in others:
                j = np.maximum(J[n], 0)
                cl &= (j - F[n]["lastL"][j]) < kk
                cs &= (j - F[n]["lastS"][j]) < kk
        else:
            cl, cs = Lt.copy(), St.copy()
            for n in others:
                cl &= g(n, "dir") == 1
                cs &= g(n, "dir") == -1
            if E.endswith("NH"):
                for n in ("1h", "4h"):
                    with np.errstate(invalid="ignore"):
                        cl &= (g(n, "k") < 80) & (g(n, "rsi") < 70)
                        cs &= (g(n, "k") > 20) & (g(n, "rsi") > 30)
        out_d[cl & ok], out_d[cs & ok] = 1, -1
    ii = np.nonzero(out_d)[0]
    return [(int(tc[i]), int(out_d[i]), float(T["c"][i]), trig) for i in ii]


# ---------------- 청산 신호 시각 ----------------
def exit_times(F: dict, X: str, trig: str) -> dict:
    """방향별로 '반대 조건이 선 봉 마감 시각' 정렬 배열."""
    if X == "XP":
        return {1: np.zeros(0, np.int64), -1: np.zeros(0, np.int64)}
    n = {"XH1": "1h", "XH4": "4h", "XT": trig}[X]
    f = F[n]
    if X == "XH1":
        ag_l, ag_s = (f["dir"] == -1) | f["S"], (f["dir"] == 1) | f["L"]
    elif X == "XH4":
        ag_l, ag_s = f["dir"] == -1, f["dir"] == 1
    else:
        ag_l, ag_s = f["S"], f["L"]
    return {1: f["te"][ag_l], -1: f["te"][ag_s]}


# ---------------- 한 거래 ----------------
class Sim:
    def __init__(self, m, F, ft, fr):
        self.m, self.F, self.ft, self.fr = m, F, ft, fr
        self.t = m["t"]

    def fund(self, t_in, t_out, d):
        a_, z_ = np.searchsorted(self.ft, t_in, "right"), np.searchsorted(self.ft, t_out, "right")
        return -d * float(self.fr[a_:z_].sum())

    def trade(self, ev, cfg, xt, bail=None, lat=0, slip=0.0, fee_mult=1.0):
        m, t = self.m, self.t
        tc, d, sc, trig = ev
        k = int(np.searchsorted(t, tc)) + lat
        if k >= len(t) - 2:
            return None
        F = self.F
        if cfg.get("limit"):
            win = FRAMES[trig] * 3
            k1 = min(k + win, len(t) - 1)
            hit = (m["l"][k:k1] <= sc - TICK) if d == 1 else (m["h"][k:k1] >= sc + TICK)
            if not hit.any():
                return None
            k = k + int(np.argmax(hit))
            ep, fee_in = sc, MAKER
        else:
            ep = m["o"][k] + d * TICK + d * m["o"][k] * slip
            fee_in = TAKER
            agree = all(F[n]["dir"][max(at(F[n], np.array([tc]))[0], 0)] == d for n in ("15m", "4h", "1d"))
            if abs(ep / sc - 1) > (0.03 if agree else 0.01):
                return None
        t_in = int(t[k])
        stop = ep * (1 - d * DSTOP)
        tp = None
        k_end = len(t) - 1
        if cfg.get("tp") or bail is not None:
            j1 = at(F["1h"], np.array([tc]))[0]
            a1 = float(F["1h"]["atr"][max(j1, 0)])
            tp = ep + d * a1
            if cfg["X"] == "XP":
                stop = ep - d * min(a1, ep * DSTOP)
        if cfg["X"] == "XP":
            k_end = min(k_end, k + 8 * 60)
        else:
            arr = xt[d]
            q = int(np.searchsorted(arr, t_in, "right"))
            if q < len(arr):
                k_end = min(k_end, int(np.searchsorted(t, arr[q])))
        if bail is not None:
            # 1시간마다 목표 도달 확률 재추정, 0.30 미만이면 다음 분 시가 청산
            for hh in range(1, 9):
                kb = k + hh * 60
                if kb >= k_end:
                    break
                seg = self._hit(d, k, kb - 1, stop, tp)
                if seg is not None:
                    break
                prog = d * (m["c"][kb - 1] - ep) / abs(tp - ep)
                p = bail(hh, prog)
                if p is not None and p < 0.30:
                    k_end = kb
                    break
        hit = self._hit(d, k, k_end - 1 if k_end > k else k, stop, tp)
        if hit is not None:
            kx, why = hit
            if why == "stop":
                o = m["o"][kx]
                xp = (min(stop, o) if d == 1 else max(stop, o)) - d * TICK - d * stop * slip
                fee_out = TAKER
            else:
                xp, fee_out = tp, MAKER
            t_out = int(t[kx]) + MIN
        else:
            kx = k_end
            xp, fee_out, why = m["o"][kx], TAKER, ("time" if cfg["X"] == "XP" else "signal")
            t_out = int(t[kx])
        seg_l = m["l"][k : kx + 1] if d == 1 else m["h"][k : kx + 1]
        worst = float(d * (seg_l.min() if d == 1 else seg_l.max()) / ep - d)
        worst = min(worst, d * (xp / ep - 1))
        r = d * (xp / ep - 1) - (fee_in + fee_out) * fee_mult + self.fund(t_in, t_out, d)
        return {"t_in": t_in, "t_out": t_out, "d": d, "ep": float(ep), "xp": float(xp), "r": float(r),
                "worst": worst, "why": why, "tp": None if tp is None else float(tp), "trig": trig}  # fmt: skip

    def _hit(self, d, k0, k1, stop, tp):
        m = self.m
        if k1 < k0:
            return None
        h, lo = m["h"][k0 : k1 + 1], m["l"][k0 : k1 + 1]
        sh = (lo <= stop) if d == 1 else (h >= stop)
        ks = int(np.argmax(sh)) if sh.any() else None
        kt = None
        if tp is not None:
            th = (h >= tp) if d == 1 else (lo <= tp)
            th[0] = False  # 진입한 분에는 익절 안 셈(보수적)
            kt = int(np.argmax(th)) if th.any() else None
        if ks is None and kt is None:
            return None
        if kt is None or (ks is not None and ks <= kt):
            return k0 + ks, "stop"
        return k0 + kt, "tp"

    def run(self, evs, cfg, xt, a_ms, z_ms, bail=None, **kw):
        out, busy = [], -1
        for ev in evs:
            if ev[0] < a_ms or ev[0] >= z_ms or ev[0] < busy:
                continue
            x = self.trade(ev, cfg, xt, bail=bail, **kw)
            if x is None:
                continue
            out.append(x)
            busy = x["t_out"]
        return out


# ---------------- 성과 ----------------
def path(tr: list, frac: float) -> dict:
    eq, peak, mdd, liq = 1.0, 1.0, 0.0, 0
    for x in tr:
        if LEV * x["worst"] <= LIQ:
            liq += 1
            pos = -1.0
        else:
            pos = max(LEV * x["r"], -1.0)
            mdd = max(mdd, 1 - eq * (1 + frac * LEV * x["worst"]) / peak)
        eq *= 1 + frac * pos
        if eq <= 0:
            return {"ret": -1.0, "mdd": 1.0, "liq": liq}
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {"ret": eq - 1, "mdd": mdd, "liq": liq}


def summ(tr: list, a_ms: int, z_ms: int) -> dict:
    days = max((z_ms - a_ms) / DAY, 1)
    r = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    out = {"n": len(tr), "per_day": len(tr) / days, "win": float((r > 0).mean()) if len(r) else None,
           "avg1": float(r.mean()) if len(r) else None, "nl": int(sum(x["d"] == 1 for x in tr)),
           "ns": int(sum(x["d"] == -1 for x in tr)),
           "hold_h": float(np.median([(x["t_out"] - x["t_in"]) / 3.6e6 for x in tr])) if tr else None}  # fmt: skip
    for fr in (1.0, 0.4):
        out[f"x5_f{fr}"] = path(tr, fr)
    return out


def bail_table(tr: list, m: dict):
    """설계 구간 거래로 (경과 시간, 진행 칸) → 목표 도달 비율. 20건 미만 칸은 None."""
    bins = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    cnt, hit = {}, {}
    t = m["t"]
    for x in tr:
        if x["tp"] is None:
            continue
        k = int(np.searchsorted(t, x["t_in"]))
        reached = x["why"] == "tp"
        span = abs(x["tp"] - x["ep"])
        for hh in range(1, 9):
            kb = k + hh * 60
            if kb >= len(t) or t[kb] >= x["t_out"]:
                break
            prog = x["d"] * (m["c"][kb - 1] - x["ep"]) / span
            key = (hh, int(np.searchsorted(bins, prog)))
            cnt[key] = cnt.get(key, 0) + 1
            hit[key] = hit.get(key, 0) + int(reached)

    def fn(hh, prog):
        key = (hh, int(np.searchsorted(bins, prog)))
        n = cnt.get(key, 0)
        return hit[key] / n if n >= 20 else None

    fn.table = {f"{k[0]}h|{k[1]}": [hit[k], cnt[k]] for k in sorted(cnt)}
    return fn


def random_p(S, F, real, cfg, xt, bail, a_ms, z_ms, nsim, seed) -> dict:
    """변동성 맞춘 무작위: 진짜 거래마다 같은 방향, 진입 시점 1시간 ATR% 5분위 같은 칸의 5분 마감 시각."""
    rng = np.random.default_rng(seed)
    f1 = F["1h"]
    atrp = f1["atr"] / f1["c"]
    T5 = F["5m"]["te"]
    T5 = T5[(T5 >= a_ms) & (T5 < z_ms)]
    j5 = np.maximum(at(f1, T5), 0)
    qs = np.quantile(atrp[j5], [0.2, 0.4, 0.6, 0.8])
    b5 = np.searchsorted(qs, atrp[j5])
    pools = [T5[b5 == q] for q in range(5)]
    c5 = F["5m"]["c"]
    i5 = {int(v): i for i, v in enumerate(F["5m"]["te"])}

    def stat(tr):
        return float(sum(np.log(max(1 + 0.4 * max(LEV * x["r"], -1.0), 1e-9)) for x in tr))

    real_s = stat(real)
    sims = []
    for _ in range(nsim):
        tr = []
        for x in real:
            jb = max(at(f1, np.array([x["t_in"]]))[0], 0)
            q = int(np.searchsorted(qs, atrp[jb]))
            tcr = int(pools[q][rng.integers(len(pools[q]))])
            ev = (tcr, x["d"], float(c5[i5[tcr]]), "5m")
            y = S.trade(ev, {**cfg, "limit": False}, xt, bail=bail)
            if y is not None:
                tr.append(y)
        sims.append(stat(tr))
    sims = np.array(sims)
    return {"p": float((sims >= real_s).mean()), "real": real_s, "rand_med": float(np.median(sims)), "nsim": nsim}


def by_year(tr: list) -> dict:
    out: dict = {}
    for x in tr:
        y = datetime.fromtimestamp(x["t_in"] / 1000, timezone.utc).year
        out.setdefault(y, []).append(x)
    return {y: {"n": len(v), "x5_f0.4": path(v, 0.4)["ret"], "x5_f1.0": path(v, 1.0)["ret"]} for y, v in sorted(out.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = load_1m()
    b = r5_data.load(DW.R5)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)
    del b
    F = {n: frame(m, mins) for n, mins in FRAMES.items()}
    S = Sim(m, F, ft, fr)
    end = int(m["t"][-1]) + MIN
    W = {"fresh": (ms("2020-03-01"), ms("2021-11-01")), "design": (ms("2022-01-01"), ms("2024-07-01")),
         "val": (ms("2024-07-01"), ms("2025-11-01")), "year": (ms("2025-11-01"), end),
         "m6": (end - 183 * DAY, end), "d30": (end - 30 * DAY, end), "pool": (ms("2020-03-01"), ms("2025-11-01"))}  # fmt: skip
    print("loaded", len(m["t"]), round(time.time() - t0, 1), flush=True)

    # 일치 빈도(5분 마감 기준)
    T5 = F["5m"]["te"]
    sel = (T5 >= W["pool"][0]) & (T5 < W["pool"][1])
    D = np.stack([F[n]["dir"][np.maximum(at(F[n], T5[sel]), 0)] for n in FIVE])
    align = {"all5_up": float((D == 1).all(0).mean()), "all5_dn": float((D == -1).all(0).mean()),
             "4of5_up": float((((D == 1).sum(0) >= 4) & ((D == -1).sum(0) == 0)).mean()),
             "4of5_dn": float((((D == -1).sum(0) >= 4) & ((D == 1).sum(0) == 0)).mean())}  # fmt: skip
    sigs = {n: {"L": int(F[n]["L"].sum()), "S": int(F[n]["S"].sum())} for n in FIVE}
    print("align", align, flush=True)

    EV = {E: events(F, E) for E in ENTRIES}
    print("events", {E: len(v) for E, v in EV.items()}, round(time.time() - t0, 1), flush=True)
    XT = {(E, X): exit_times(F, X, EV[E][0][3] if EV[E] else "5m") for E in ENTRIES for X in EXITS}

    def evaluate(E, X, extra=None, bail=None, wins=("fresh", "design", "val", "year", "m6", "d30", "pool"), **kw):
        cfg = {"X": X, **(extra or {})}
        res, keep = {}, {}
        for w in wins:
            a_, z_ = W[w]
            tr = S.run(EV[E], cfg, XT[(E, X)], a_, z_, bail=bail, **kw)
            res[w] = summ(tr, a_, z_)
            keep[w] = tr
        return res, keep

    grid, trades = {}, {}
    for E in ENTRIES:
        for X in EXITS:
            r, kp = evaluate(E, X)
            grid[f"{E}|{X}"] = r
            trades[f"{E}|{X}"] = kp
            print(E, X, {w: (r[w]["n"], round(r[w]["x5_f0.4"]["ret"], 3), r[w]["x5_f0.4"]["liq"])
                         for w in ("fresh", "design", "val", "year")}, round(time.time() - t0, 1), flush=True)  # fmt: skip

    elig = [k for k, r in grid.items() if r["design"]["n"] >= 30 and r["design"]["x5_f0.4"]["liq"] == 0]
    elig.sort(key=lambda k: (grid[k]["design"]["x5_f0.4"]["ret"], -grid[k]["design"]["x5_f0.4"]["mdd"]), reverse=True)
    top2 = elig[:2]
    print("top2", top2, flush=True)

    extras = {}
    for key in top2:
        E, X = key.split("|")
        r, kp = evaluate(E, X, {"limit": True})
        extras[f"{key}|LIM"] = r
        trades[f"{key}|LIM"] = kp
        # 탈출 장치: XP가 아니면 같은 변형에 1×ATR 익절을 더한 뒤 표를 설계 구간에서 만든다
        base_extra = {"tp": True}
        _, kd = evaluate(E, X, base_extra, wins=("design",))
        bl = bail_table(kd["design"], m)
        r, kp = evaluate(E, X, base_extra, bail=bl)
        extras[f"{key}|BAIL"] = r
        extras[f"{key}|BAIL"]["table"] = bl.table
        trades[f"{key}|BAIL"] = kp
    allres = {**grid, **{k: {w: v for w, v in r.items() if w != "table"} for k, r in extras.items()}}
    final = top2[0] if top2 else None

    gate, rnd, red = {}, {}, {}
    if final:
        rf = allres[final]
        E, X = final.split("|")
        cfg = {"X": X}
        gate = {
            "pos_all4": all(rf[w]["x5_f0.4"]["ret"] > 0 for w in ("fresh", "design", "val", "year")),
            "liq0": all(rf[w]["x5_f0.4"]["liq"] == 0 for w in ("fresh", "design", "val", "year")),
            "mdd40": all(rf[w]["x5_f0.4"]["mdd"] <= 0.40 for w in ("fresh", "design", "val", "year")),
        }
        real = trades[final]["pool"]
        rnd["pool"] = random_p(S, F, real, cfg, XT[(E, X)], None, *W["pool"], args.nsim, 7)
        for w in ("fresh", "val", "year"):
            rnd[w] = random_p(S, F, trades[final][w], cfg, XT[(E, X)], None, *W[w], max(args.nsim // 2, 50), 11)
        print("random", rnd, flush=True)
        gate["random_bonf"] = rnd["pool"]["p"] * N_BONF < 0.05
        for w in ("fresh", "design", "val", "year"):
            a_, z_ = W[w]
            red[w] = {
                "lat1": summ(S.run(EV[E], cfg, XT[(E, X)], a_, z_, lat=1), a_, z_),
                "lat5": summ(S.run(EV[E], cfg, XT[(E, X)], a_, z_, lat=5), a_, z_),
                "slip2": summ(S.run(EV[E], cfg, XT[(E, X)], a_, z_, slip=0.0002), a_, z_),
                "fee15": summ(S.run(EV[E], cfg, XT[(E, X)], a_, z_, fee_mult=1.5), a_, z_),
            }
        red["by_year"] = by_year(S.run(EV[E], cfg, XT[(E, X)], W["fresh"][0], end))
    p2 = None
    pp = Path("~/.minerva/research/ql-upgrade-2026-11-08/posthoc.json").expanduser()
    if pp.exists():
        try:
            p2 = json.loads(pp.read_text())["P2"]["res"]
        except (KeyError, ValueError):
            p2 = None
    if final and p2:
        gate["beats_p2_ref"] = any(allres[final][w]["x5_f1.0"]["ret"] > p2[w]["x5_f1.0"]["ret"] for w in ("val", "year") if w in p2)
    passed = bool(final) and all(v for k, v in gate.items() if k != "beats_p2_ref")
    res = {"n_tests": N_BONF, "n_tonight": N_BEFORE + N_BONF, "align": align, "signals_total": sigs,
           "events": {E: len(v) for E, v in EV.items()}, "grid": grid, "extras": extras, "top2": top2,
           "final": final, "gate": gate, "pass": passed, "random": rnd, "redteam": red,
           "p2": p2, "windows": W, "sec": time.time() - t0}  # fmt: skip
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    keep = {k: {w: v for w, v in d.items() if w in ("d30", "m6", "year")} for k, d in trades.items() if k in ([final] if final else [])}
    keep["_pool_final"] = trades[final]["pool"] if final else []
    (out / "trades.json").write_text(json.dumps(keep, default=float))
    print("final", final, "gate", gate, "pass", passed, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
