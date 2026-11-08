"""QL 포트폴리오 고도화: 1분봉 체결로 진입(E)·문(G)·청산(X)·비중(S)·추가 우위(A)를 단계별로 고른다.

등록: 2026-11-08-btc-ql-upgrade-hypotheses.md. 연구 전용(주문 없음).
자료: fut1m_data.py가 받은 1분봉 + ql_portfolio 방향 문(15분봉).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/ql-upgrade-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/ql_upgrade.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import daily_wave as DW
import numpy as np
import ql_portfolio as QP
import regime_lines as RL

MIN, M15, DAY = 60_000, 900_000, 86_400_000
TAKER, MAKER, TICK = 0.0005, 0.0002, 0.1
LEV, LIQ = 5, -0.95
NY, LDN = ZoneInfo("America/New_York"), ZoneInfo("Europe/London")
DIR1M = Path("~/.minerva/data/btc/futures/1m").expanduser()
GATES4 = ("GT", "V2", "L3", "Z5")
MAIN = ("fresh", "design", "val", "year")
SHOW = ("fresh", "design", "val", "year", "m6", "d30", "pool")
N_TESTS, N_BONF, N_BEFORE = 30, 40, 5949


# ---------------- 자료 ----------------
def load_1m(t_end: int) -> dict:
    parts = [np.load(p) for p in sorted(DIR1M.glob("*.npz"))]
    m = {k: np.concatenate([p[k] for p in parts]) for k in ("t", "o", "h", "l", "c")}
    keep = m["t"] < t_end
    return {k: v[keep] for k, v in m.items()}


def local_ms(day_utc: datetime, tz, hh: int, mm: int) -> int:
    return int(datetime(day_utc.year, day_utc.month, day_utc.day, hh, mm, tzinfo=tz).timestamp() * 1000)


# ---------------- 세션(하루 한 번 셋업) ----------------
def sessions(m: dict, b: dict, gates: dict, kind: str) -> list:
    """kind: NY(9:30 60분, 창 4시간), NY2(10:30 60분), LDN(8:00 런던 60분), US830(8:30 15분, 창 45분)."""
    spec = {"NY": (NY, 9, 30, 60, 240), "NY2": (NY, 10, 30, 60, 240),
            "LDN": (LDN, 8, 0, 60, 240), "US830": (NY, 8, 30, 15, 45)}[kind]  # fmt: skip
    tz, hh, mm, rng, win = spec
    t = m["t"]
    out = []
    d0 = datetime.fromtimestamp(t[0] / 1000, timezone.utc).date()
    d1 = datetime.fromtimestamp(t[-1] / 1000, timezone.utc).date()
    d = d0
    while d <= d1:
        dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        d += timedelta(days=1)
        if dt.weekday() >= 5:
            continue
        st = local_ms(dt, tz, hh, mm)
        a = int(np.searchsorted(t, st))
        z = a + rng
        if z + win >= len(t) or t[a] != st or t[z] != st + rng * MIN or t[z + win] != st + (rng + win) * MIN:
            continue
        H, L = float(m["h"][a:z].max()), float(m["l"][a:z].min())
        if H <= L:
            continue
        i15 = int(np.searchsorted(b["t"], st + rng * MIN)) - 1  # 범위 끝에 마감된 15분봉
        if i15 < 0 or i15 >= len(b["t"]):
            continue
        g = {k: int(gates[k][i15]) for k in GATES4}
        out.append({"kind": kind, "st": st, "a": a, "z": z, "win": win, "H": H, "L": L, "i15": i15, "g": g,
                    "ny16": local_ms(dt, NY, 16, 0)})  # fmt: skip
    return out


def gate_ok(g: dict, d: int, mode: str) -> tuple[bool, float]:
    n = sum(1 for k in GATES4 if g[k] == d)
    if mode == "Gany":
        return n >= 1, 1.0
    if mode == "G2":
        return n >= 2, 1.0
    if mode == "GT":  # 런던 추가 우위용
        return g["GT"] == d, 1.0
    return n >= 1, n / 4.0  # Gsz


# ---------------- 진입 ----------------
def entry(m: dict, b: dict, gates: dict, s: dict, E: str, G: str, lat: int = 0, slip: float = 0.0):
    """반환: dict(k=체결 분 위치, d, ep, stop, tp, fin, size) 또는 None."""
    o, h, lo, c, t = m["o"], m["h"], m["l"], m["c"], m["t"]
    H, L, z, win = s["H"], s["L"], s["z"], s["win"]
    rngh = H - L
    end = z + win  # 진입 창 끝(이 분 전까지)
    if E in ("E1", "E4"):
        sides = [d for d in (1, -1) if gate_ok(s["g"], d, G)[0]]
        if not sides:
            return None
        p0 = z + lat
        upv = (h[p0:end] >= H + TICK) if 1 in sides else np.zeros(max(end - p0, 0), bool)
        dnv = (lo[p0:end] <= L - TICK) if -1 in sides else np.zeros(max(end - p0, 0), bool)
        any_ = upv | dnv
        if not any_.any():
            return None
        q = int(np.argmax(any_))
        k = p0 + q
        if upv[q] and dnv[q]:  # 같은 분에 둘 다: 시가에 가까운 쪽이 먼저
            d = 1 if abs(o[k] - (H + TICK)) <= abs(o[k] - (L - TICK)) else -1
        else:
            d = 1 if upv[q] else -1
        trig = H + TICK if d == 1 else L - TICK
        if E == "E1":
            ep = (max(trig, o[k]) if d == 1 else min(trig, o[k])) + d * TICK
        else:  # E4 스톱 지정가: 지정가 = 트리거 ± 0.05%
            lim = trig * (1 + d * 0.0005)
            px = (max(trig, o[k]) if d == 1 else min(trig, o[k])) + d * TICK
            if (px - lim) * d <= 0:
                ep = px
            else:  # 지정가 밖으로 튐 → 60분 안 되돌아오면 체결
                k2 = None
                for j in range(k + 1, min(k + 61, len(t))):
                    if (lo[j] <= lim) if d == 1 else (h[j] >= lim):
                        k2 = j
                        break
                if k2 is None:
                    return None
                k, ep = k2, lim
        ep *= 1 + d * slip
        _, size = gate_ok(s["g"], d, G)
        return {"k": k, "d": d, "ep": ep, "stop": L if d == 1 else H, "tp": ep + d * rngh,
                "fin": TAKER, "size": size, "fill_same_bar": True}  # fmt: skip
    # 15분/1분 종가 신호
    if E in ("E0m", "E3"):
        j = None
        for q in range(z + 14, end, 15):  # 15분봉 마감 분
            if c[q] > H or c[q] < L:
                j = q
                break
        if j is None:
            return None
        d = 1 if c[j] > H else -1
        i15 = int(np.searchsorted(b["t"], t[j] - 14 * MIN))  # 그 15분봉
        g = {kk: int(gates[kk][i15]) for kk in GATES4}
        ok, size = gate_ok(g, d, G)
        if not ok:
            return None
        lp = c[j] if E == "E0m" else (H if d == 1 else L)
        tp = (c[j] if E == "E0m" else lp) + d * rngh
    elif E == "E2":
        brk = (c[z:end] > H) | (c[z:end] < L)
        if not brk.any():
            return None
        j = z + int(np.argmax(brk))
        d = 1 if c[j] > H else -1
        ok, size = gate_ok(s["g"], d, G)
        if not ok:
            return None
        lp = H if d == 1 else L
        tp = lp + d * rngh
    else:
        raise ValueError(E)
    k = None
    for q in range(j + 1 + lat, min(j + 61 + lat, len(t))):
        if E != "E0m" and ((h[q] >= tp) if d == 1 else (lo[q] <= tp)):
            return None  # 체결 전에 익절 가격 도달 → 취소
        if (lo[q] < lp - TICK) if d == 1 else (h[q] > lp + TICK):
            k = q
            break
    if k is None:
        return None
    ep = lp * (1 + d * slip)
    return {"k": k, "d": d, "ep": ep, "stop": L if d == 1 else H, "tp": tp, "fin": MAKER, "size": size,
            "fill_same_bar": False}  # fmt: skip


# ---------------- 청산 ----------------
def first_hit(m, d, k0, k1, stop, tp, check_tp_from):
    """[k0, k1] 안에서 손절/익절 첫 분. 같은 분이면 손절. 반환 (분, 'stop'|'tp') 또는 None."""
    h, lo = m["h"][k0 : k1 + 1], m["l"][k0 : k1 + 1]
    sh = (lo <= stop) if d == 1 else (h >= stop)
    th = (h >= tp) if d == 1 else (lo <= tp)
    if check_tp_from > k0:
        th[: check_tp_from - k0] = False
    ks = int(np.argmax(sh)) if sh.any() else None
    kt = int(np.argmax(th)) if th.any() else None
    if ks is None and kt is None:
        return None
    if kt is None or (ks is not None and ks <= kt):
        return k0 + ks, "stop"
    return k0 + kt, "tp"


def stop_px(m, j, d, stop, slip):
    o = m["o"][j]
    px = min(stop, o) if d == 1 else max(stop, o)  # 시가가 이미 넘었으면 시가
    return px - d * TICK - d * px * slip


def exit_trade(m, e: dict, X: str, s: dict, fund_fn, fee_mult=1.0, slip=0.0, bail=None) -> dict:
    t, c = m["t"], m["c"]
    k, d, ep, stop, tp = e["k"], e["d"], e["ep"], e["stop"], e["tp"]
    R = abs(ep - stop)
    tend = k + 480
    if X in ("X3", "X6"):  # X6 = X3 + 익절 지정가(메이커), 사후 점검용
        tend = int(np.searchsorted(t, s["ny16"])) - 1
        if tend <= k:
            tend = k + 1
    tend = min(tend, len(t) - 1)
    ktp = k + 1  # 익절은 체결 다음 분부터(체결 분 안 순서를 모른다)
    tk, mk = TAKER * fee_mult, MAKER * fee_mult
    fin = e["fin"] * fee_mult
    legs = []  # (비중, 청산가, 이유, 분)

    if X in ("X0", "X3", "X5", "X6"):
        hit = first_hit(m, d, k, tend, stop, tp, ktp)
        if hit is None:
            legs.append((1.0, c[tend], "time", tend))
        elif hit[1] == "stop":
            legs.append((1.0, stop_px(m, hit[0], d, stop, slip), "stop", hit[0]))
        else:
            legs.append((1.0, tp, "tp", hit[0]))
    elif X == "X1":
        trig = ep + d * 0.5 * R
        hh, ll = m["h"][k + 1 : tend + 1], m["l"][k + 1 : tend + 1]
        reach = (hh >= trig) if d == 1 else (ll <= trig)
        q = k + 1 + int(np.argmax(reach)) if reach.any() else None
        hit = first_hit(m, d, k, q if q is not None else tend, stop, tp, ktp)
        if hit is not None:
            px = stop_px(m, hit[0], d, stop, slip) if hit[1] == "stop" else tp
            legs.append((1.0, px, hit[1], hit[0]))
        elif q is None:
            legs.append((1.0, c[tend], "time", tend))
        else:
            hit2 = first_hit(m, d, q + 1, tend, ep, tp, q + 1) if q + 1 <= tend else None
            if hit2 is None:
                legs.append((1.0, c[tend], "time", tend))
            elif hit2[1] == "stop":
                legs.append((1.0, stop_px(m, hit2[0], d, ep, slip), "be", hit2[0]))
            else:
                legs.append((1.0, tp, "tp", hit2[0]))
    elif X == "X2":
        hit = first_hit(m, d, k, tend, stop, tp, ktp)
        if hit is None:
            legs.append((1.0, c[tend], "time", tend))
        elif hit[1] == "stop":
            legs.append((1.0, stop_px(m, hit[0], d, stop, slip), "stop", hit[0]))
        else:
            q = hit[0]
            legs.append((0.5, tp, "tp1", q))
            tp2 = ep + d * 2 * abs(tp - ep)
            hit2 = first_hit(m, d, q + 1, tend, ep, tp2, q + 1) if q + 1 <= tend else None
            if hit2 is None:
                legs.append((0.5, c[tend], "time", tend))
            elif hit2[1] == "stop":
                legs.append((0.5, stop_px(m, hit2[0], d, ep, slip), "be", hit2[0]))
            else:
                legs.append((0.5, tp2, "tp2", hit2[0]))
    elif X == "X4":
        hit = first_hit(m, d, k, tend, stop, tp, ktp)
        done = False
        for hr in range(1, 8):
            j = k + 60 * hr
            if j >= tend or (hit is not None and hit[0] <= j):
                break
            u = d * (c[j] / ep - 1) * ep / R if R > 0 else 0.0
            bk = int(np.digitize(u, [-0.5, 0.0, 0.5]))
            p = bail.get((hr, bk)) if bail else None
            if p is not None and p < 0.30:
                legs.append((1.0, c[j], "bail", j))
                done = True
                break
        if not done:
            if hit is None:
                legs.append((1.0, c[tend], "time", tend))
            elif hit[1] == "stop":
                legs.append((1.0, stop_px(m, hit[0], d, stop, slip), "stop", hit[0]))
            else:
                legs.append((1.0, tp, "tp", hit[0]))
    else:
        raise ValueError(X)
    jx = max(x[3] for x in legs)
    lo_, hi_ = m["l"][k : jx + 1], m["h"][k : jx + 1]
    worst = (lo_.min() / ep - 1) if d == 1 else (1 - hi_.max() / ep)
    r = 0.0
    for w, px, why, j in legs:
        fout = mk if (X in ("X5", "X6") and why == "tp") else tk
        r += w * (d * (px / ep - 1) - fin - fout)
    t_in, t_out = int(t[k]), int(t[jx]) + MIN
    fund = fund_fn(t_in, t_out, d)
    r += fund
    return {"k": int(k), "x": int(jx), "d": int(d), "ep": float(ep), "xp": float(legs[-1][1]), "r": float(r),
            "worst": float(min(worst, r)), "why": legs[-1][2], "size": float(e["size"]), "fund": float(fund),
            "t_in": t_in, "t_out": t_out, "kind": s["kind"], "st": int(s["st"])}  # fmt: skip


# ---------------- 묶기 ----------------
def run(m, b, gates, sess_lists, cfg, fund_fn, a_ms, z_ms, **kw) -> list:
    """sess_lists: [(우선순위, 세션 목록, E, G)]. 한 번에 1포지션, 시각 순."""
    ev = []
    for pri, lst, E, G in sess_lists:
        for s in lst:
            if a_ms <= s["st"] < z_ms:
                ev.append((s["st"], pri, s, E, G))
    ev.sort(key=lambda x: (x[0], x[1]))
    out, busy = [], -1
    for _, _, s, E, G in ev:
        e = entry(m, b, gates, s, E, G, lat=kw.get("lat", 0), slip=kw.get("slip", 0.0))
        if e is None or e["k"] <= busy or m["t"][e["k"]] >= z_ms:
            continue
        tr = exit_trade(m, e, cfg["X"], s, fund_fn, kw.get("fee_mult", 1.0), kw.get("slip", 0.0), cfg.get("bail"))
        out.append(tr)
        busy = tr["x"]
    return out


def sizes(tr: list, S: str, vol_fn, vtarget) -> np.ndarray:
    n = len(tr)
    f = np.array([x["size"] for x in tr], float)
    if S == "S1":
        rv = np.array([vol_fn(x["t_in"]) for x in tr])
        f = f * np.minimum(1.0, vtarget / np.where(rv > 0, rv, vtarget))
    elif S == "S2":
        streak = 0
        for i in range(n):
            if streak >= 3:
                f[i] *= 0.5
            streak = streak + 1 if tr[i]["r"] <= 0 else 0
    elif S == "S3":
        eq, hist = 1.0, []
        for i in range(n):
            if len(hist) >= 20 and eq < np.mean(hist[-20:]):
                f[i] *= 0.5
            eq *= 1 + LEV * tr[i]["r"]
            hist.append(eq)
    elif S == "S4":
        eq, wk, wk_start = 1.0, None, 1.0
        for i in range(n):
            w = (tr[i]["t_in"] // DAY + 3) // 7  # 월요일 시작 주
            if w != wk:
                wk, wk_start = w, eq
            if eq / wk_start - 1 <= -0.15:
                f[i] = 0.0
            eq *= 1 + f[i] * LEV * tr[i]["r"]
    return f


def path(tr: list, frac: float, f: np.ndarray | None = None) -> dict:
    eq, peak, mdd, liq = 1.0, 1.0, 0.0, 0
    for i, x in enumerate(tr):
        w = frac * (1.0 if f is None else f[i])
        if LEV * x["worst"] <= LIQ:
            liq += 1
            pos = -1.0
        else:
            pos = max(LEV * x["r"], -1.0)
            mdd = max(mdd, 1 - eq * (1 + w * LEV * x["worst"]) / peak)
        eq *= 1 + w * pos
        if eq <= 0:
            return {"ret": -1.0, "mdd": 1.0, "liq": liq}
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {"ret": eq - 1, "mdd": mdd, "liq": liq}


def summ(tr, a_ms, z_ms, f=None) -> dict:
    days = max((z_ms - a_ms) / DAY, 1)
    r = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    out = {"n": len(tr), "per_day": len(tr) / days, "win": float((r > 0).mean()) if len(r) else None,
           "avg1": float(r.mean()) if len(r) else None, "x1": float(np.prod(1 + r) - 1) if len(r) else 0.0,
           "nl": int(sum(x["d"] == 1 for x in tr)), "ns": int(sum(x["d"] == -1 for x in tr))}  # fmt: skip
    for fr in (1.0, 0.4):
        out[f"x5_f{fr}"] = path(tr, fr, f)
    return out


# ---------------- 무작위 ----------------
def random_p(m, b, gates, tr_real, cfg, fund_fn, sess_all, a_ms, z_ms, nsim, seed, f=None) -> dict:
    rng = np.random.default_rng(seed)
    pool = [s for s in sess_all if a_ms <= s["st"] < z_ms]
    real = path(tr_real, 1.0, f)["ret"]
    vals = []
    for _ in range(nsim):
        rt = []
        for x in tr_real:
            s = pool[int(rng.integers(len(pool)))]
            k = int(s["a"] + (x["k"] - x["_sa"]))
            if k >= len(m["t"]) - 2:
                continue
            ep = m["o"][k]
            stop = ep * (1 - x["d"] * x["_sd"])
            tp = ep * (1 + x["d"] * x["_td"])
            e = {"k": k, "d": x["d"], "ep": ep, "stop": stop, "tp": tp, "fin": x["_fin"], "size": x["size"],
                 "fill_same_bar": True}  # fmt: skip
            rt.append(exit_trade(m, e, cfg["X"], s, fund_fn, bail=cfg.get("bail")))
        rt.sort(key=lambda y: y["k"])
        vals.append(path(rt, 1.0, f)["ret"] if f is None else path(rt, 1.0, np.array([y["size"] for y in rt]))["ret"])
    vals = np.array(vals)
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "real": float(real), "nsim": nsim}


def annotate(tr, m, sess_by_st):
    for x in tr:
        s = sess_by_st[(x["kind"], x["st"])]
        x["_sa"] = s["a"]
        x["_sd"] = abs(x["ep"] - (s["L"] if x["d"] == 1 else s["H"])) / x["ep"]
        x["_td"] = abs(s["H"] - s["L"]) / x["ep"]
        x["_fin"] = TAKER if x.get("_e") in ("E1", "E4") else MAKER


# ---------------- 실행 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b, s4 = RL.load_all(DW.R5)
    F = DW.features(b)
    QP.add_gates(b, s4, F)
    gates = {k: F["gate"][k] for k in GATES4}
    t_end = int(b["t"][-1]) + M15
    m = load_1m(t_end)
    print("loaded", len(m["t"]), round(time.time() - t0, 1), flush=True)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    # 일봉 실현 변동성(전날까지 20일)
    dc = b["c"][np.r_[np.nonzero(np.diff(b["t"] // DAY))[0], len(b["t"]) - 1]]
    dday = (b["t"] // DAY)[np.r_[np.nonzero(np.diff(b["t"] // DAY))[0], len(b["t"]) - 1]]
    dr = np.r_[np.nan, dc[1:] / dc[:-1] - 1]
    rv = np.array([np.nanstd(dr[max(0, i - 20) : i]) if i >= 5 else np.nan for i in range(len(dr))])

    def vol_fn(t_in):
        i = int(np.searchsorted(dday, t_in // DAY)) - 0  # 그날 이전까지(rv[i]는 i 전날까지)
        i = min(max(i, 0), len(rv) - 1)
        return float(rv[i]) if np.isfinite(rv[i]) else 0.0

    Wb = RL.windows(b["t"])
    W = {k: (int(b["t"][a]), int(b["t"][min(z, len(b["t"]) - 1)]) + (M15 if z >= len(b["t"]) else 0))
         for k, (a, z) in Wb.items() if k in SHOW}  # fmt: skip
    sess = {k: sessions(m, b, gates, k) for k in ("NY", "NY2", "LDN", "US830")}
    sess_by = {(s["kind"], s["st"]): s for v in sess.values() for s in v}
    print("sessions", {k: len(v) for k, v in sess.items()}, round(time.time() - t0, 1), flush=True)

    def evaluate(lists, cfg, S="S0", vt=None, windows=SHOW, **kw):
        res, keep = {}, {}
        for w in windows:
            a_ms, z_ms = W[w]
            tr = run(m, b, gates, lists, cfg, fund_fn, a_ms, z_ms, **kw)
            f = sizes(tr, S, vol_fn, vt) if tr else None
            res[w] = summ(tr, a_ms, z_ms, f)
            keep[w] = (tr, f)
        return res, keep

    # 0) 지금 모형 15분 모의 기록(ql-portfolio 결과)과 1분 재현
    # 1단계
    st1 = {}
    for E in ("E0m", "E1", "E2", "E3", "E4"):
        for G in ("Gany", "G2", "Gsz"):
            r, _ = evaluate([(0, sess["NY"], E, G)], {"X": "X0"})
            st1[f"{E}|{G}"] = r
            print("S1", E, G, {w: (r[w]["n"], round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f1.0"]["mdd"], 2))
                                for w in MAIN}, round(time.time() - t0), flush=True)  # fmt: skip

    def key_ret(name, res):
        d = res[name]["design"]
        return (d["n"] >= 30, d["x5_f1.0"]["ret"])

    best1 = max(st1, key=lambda n: key_ret(n, st1))
    top = st1[best1]["design"]["x5_f1.0"]["ret"]
    near = [n for n in st1 if st1[n]["design"]["n"] >= 30 and st1[n]["design"]["x5_f1.0"]["ret"] >= top - 0.10 * abs(top)]
    best1 = min(near, key=lambda n: st1[n]["design"]["x5_f1.0"]["mdd"])
    E1c, G1c = best1.split("|")
    print("pick1", best1, flush=True)

    # 2단계: 청산 (X4 표는 design 구간 X0 거래로)
    _, kp = evaluate([(0, sess["NY"], E1c, G1c)], {"X": "X0"}, windows=("design",))
    dtr = kp["design"][0]
    cnt: dict = {}
    for x in dtr:
        R = abs(x["ep"] - (sess_by[(x["kind"], x["st"])]["L"] if x["d"] == 1 else sess_by[(x["kind"], x["st"])]["H"]))
        for hr in range(1, 8):
            j = x["k"] + 60 * hr
            if j >= x["x"]:
                break
            u = x["d"] * (m["c"][j] - x["ep"]) / R if R > 0 else 0.0
            bk = int(np.digitize(u, [-0.5, 0.0, 0.5]))
            c_ = cnt.setdefault((hr, bk), [0, 0])
            c_[0] += 1
            c_[1] += x["why"] == "tp"
    bail = {k: v[1] / v[0] for k, v in cnt.items() if v[0] >= 20}
    st2 = {}
    for X in ("X0", "X1", "X2", "X3", "X4", "X5"):
        r, _ = evaluate([(0, sess["NY"], E1c, G1c)], {"X": X, "bail": bail})
        st2[X] = r
        print("S2", X, {w: (r[w]["n"], round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f1.0"]["mdd"], 2))
                        for w in MAIN}, flush=True)  # fmt: skip
    top = max(st2[x]["design"]["x5_f1.0"]["ret"] for x in st2)
    near = [x for x in st2 if st2[x]["design"]["x5_f1.0"]["ret"] >= top - 0.10 * abs(top)]
    X2c = min(near, key=lambda x: st2[x]["design"]["x5_f1.0"]["mdd"])
    cfg = {"X": X2c, "bail": bail}
    print("pick2", X2c, flush=True)

    # 3단계: 비중
    dsg = [vol_fn(x["t_in"]) for x in dtr]
    vt = float(np.median([v for v in dsg if v > 0]))
    st3 = {}
    for S in ("S0", "S1", "S2", "S3", "S4"):
        r, _ = evaluate([(0, sess["NY"], E1c, G1c)], cfg, S=S, vt=vt)
        st3[S] = r
        print("S3", S, {w: (round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f1.0"]["mdd"], 2)) for w in MAIN},
              flush=True)  # fmt: skip
    S3c = max(st3, key=lambda S: (1 + st3[S]["design"]["x5_f1.0"]["ret"]) / max(st3[S]["design"]["x5_f1.0"]["mdd"], 1e-6))
    print("pick3", S3c, flush=True)

    # 4단계: 추가 우위 (각자 단독 심사)
    st4, admit = {}, []
    for A, kind, G in (("A1", "NY2", G1c), ("A2", "LDN", "GT"), ("A3", "US830", G1c)):
        r, _ = evaluate([(0, sess[kind], E1c, G)], cfg)
        d = r["design"]
        st4[A] = {**r, "admit": bool(d["n"] >= 30 and (d["avg1"] or 0) > 0 and d["x1"] > 0)}
        if st4[A]["admit"]:
            admit.append((A, kind, G))
        print("S4", A, st4[A]["admit"], {w: (r[w]["n"], round(r[w]["x5_f1.0"]["ret"], 2)) for w in MAIN}, flush=True)

    lists_final = [(0, sess["NY"], E1c, G1c)] + [(i + 1, sess[k], E1c, G) for i, (_, k, G) in enumerate(admit)]
    lists_old = [(0, sess["NY"], "E0m", "Gany")]
    fin, kfin = evaluate(lists_final, cfg, S=S3c, vt=vt)
    old, kold = evaluate(lists_old, {"X": "X0"})
    print("final", {w: (fin[w]["n"], round(fin[w]["per_day"], 2), round(fin[w]["x5_f1.0"]["ret"], 2),
                        round(fin[w]["x5_f1.0"]["mdd"], 2)) for w in SHOW}, flush=True)  # fmt: skip

    # 무작위 (pool, 각 주구간)
    sess_all = [s for L_ in lists_final for s in L_[1]]
    rnd = {}
    for name, keep, cf, E in (("final", kfin, cfg, E1c), ("old", kold, {"X": "X0"}, "E0m")):
        rnd[name] = {}
        for w in ("pool",) + MAIN:
            tr, f = keep[w]
            for x in tr:
                x["_e"] = E
            annotate(tr, m, sess_by)
            pool_s = sess_all if name == "final" else sess["NY"]
            rnd[name][w] = random_p(m, b, gates, tr, cf, fund_fn, pool_s, *W[w],
                                    args.nsim if w == "pool" else 200, 7, f)  # fmt: skip
            print("random", name, w, rnd[name][w], flush=True)

    # 레드팀
    red = {}
    for name, lists, cf, S in (("final", lists_final, cfg, S3c), ("old", lists_old, {"X": "X0"}, "S0")):
        red[name] = {}
        for tag, kw in (("slip2bp", {"slip": 0.0002}), ("fee15", {"fee_mult": 1.5}), ("lat1", {"lat": 1}),
                        ("lat5", {"lat": 5})):  # fmt: skip
            r, _ = evaluate(lists, cf, S=S, vt=vt, windows=MAIN, **kw)
            red[name][tag] = {w: r[w]["x5_f1.0"] for w in MAIN}
        a_ms, z_ms = W["fresh"][0], W["year"][1]
        tr = run(m, b, gates, lists, cf, fund_fn, a_ms, z_ms)
        f = sizes(tr, S, vol_fn, vt)
        yrs: dict = {}
        for i, x in enumerate(tr):
            yrs.setdefault(datetime.fromtimestamp(x["t_in"] / 1000, timezone.utc).year, []).append(i)
        red[name]["by_year"] = {y: {"n": len(ix), "x5_f1": path([tr[i] for i in ix], 1.0, f[ix])["ret"],
                                    "mdd_f1": path([tr[i] for i in ix], 1.0, f[ix])["mdd"]}
                                for y, ix in sorted(yrs.items())}  # fmt: skip
        rolls, s0 = [], a_ms
        while s0 + 90 * DAY <= z_ms:
            ix = [i for i, x in enumerate(tr) if s0 <= x["t_in"] < s0 + 90 * DAY]
            rolls.append(path([tr[i] for i in ix], 1.0, f[ix])["ret"] if ix else 0.0)
            s0 += 30 * DAY
        rolls = np.array(rolls)
        red[name]["rolling90"] = {"n": len(rolls), "pos_share": float((rolls > 0).mean()),
                                  "median": float(np.median(rolls)), "worst": float(rolls.min())}  # fmt: skip
        print("red", name, red[name], flush=True)

    beat = sum(fin[w]["x5_f1.0"]["ret"] > old[w]["x5_f1.0"]["ret"] and fin[w]["x5_f1.0"]["mdd"] <= old[w]["x5_f1.0"]["mdd"]
               for w in ("design", "val", "year"))  # fmt: skip
    gate = {"pos_all_f1": all(fin[w]["x5_f1.0"]["ret"] > 0 for w in MAIN),
            "liq0": all(fin[w]["x5_f1.0"]["liq"] == 0 for w in MAIN),
            "random_bonf": rnd["final"]["pool"]["p"] * N_BONF < 0.05,
            "beats_old_2of3": beat >= 2}  # fmt: skip
    pick = {"E": E1c, "G": G1c, "X": X2c, "S": S3c, "addons": [a for a, _, _ in admit], "vtarget": vt}

    def slim(keep):
        return {w: [{k: v for k, v in x.items() if not k.startswith("_")} for x in keep[w][0]]
                for w in ("d30", "m6", "year", "fresh", "design", "val")}  # fmt: skip

    (out / "trades.json").write_text(json.dumps({"final": slim(kfin), "old": slim(kold)}, default=float))
    (out / "result.json").write_text(json.dumps(
        {"n_tests": N_TESTS, "n_bonf": N_BONF, "n_before": N_BEFORE, "pick": pick, "stage1": st1, "stage2": st2,
         "stage3": st3, "stage4": st4, "final": fin, "old": old, "random": rnd, "redteam": red, "gate": gate,
         "pass": all(gate.values()), "bail_table": {f"{k[0]}h_b{k[1]}": v for k, v in bail.items()},
         "windows": W, "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    print("gate", gate, "pick", pick, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
