"""뉴욕장 돌파 모형(P2) 진입 위치(A)·횟수(B) 고도화. 등록: 2026-11-08-btc-ql-entry-freq-hypotheses.md

연구 전용(주문 없음). 1분봉 체결, 청산 X6(뉴욕 16시·익절 지정가), 비중 S1(변동성 맞춤), 문 Gany.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/ql-entry-freq-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/ql_entry_freq.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import daily_wave as DW
import numpy as np
import ql_portfolio as QP
import ql_upgrade as U
import regime_lines as RL

MIN, M15, H1, DAY = U.MIN, U.M15, 3_600_000, U.DAY
TAKER, MAKER, TICK = U.TAKER, U.MAKER, U.TICK
NY = U.NY
MAIN, SHOW = U.MAIN, U.SHOW
N_TESTS, N_BEFORE = 21, 5981
A_NAMES = ("A0", "A1", "A1b", "A2", "A3a", "A3b", "A3c", "A4", "A5")  # A0 = 기준 E0m(P2)
B_NAMES = ("B1a", "B1b", "B1c", "B1u", "B2", "B3a", "B3b", "B4a", "B4b", "B5")


# ---------------- 세션 ----------------
def make_sessions(m, b, gates, kind, tz, hh, mm, rng, win, days="wk", exit_at="ny16") -> list:
    t = m["t"]
    out = []
    d = datetime.fromtimestamp(t[0] / 1000, timezone.utc).date()
    d1 = datetime.fromtimestamp(t[-1] / 1000, timezone.utc).date()
    while d <= d1:
        dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        d += timedelta(days=1)
        wk = dt.weekday() >= 5
        if (days == "wk" and wk) or (days == "we" and not wk):
            continue
        st = U.local_ms(dt, tz, hh, mm)
        a = int(np.searchsorted(t, st))
        z = a + rng
        if (
            z + win >= len(t)
            or t[a] != st
            or t[z] != st + rng * MIN
            or t[z + win] != st + (rng + win) * MIN
        ):
            continue
        H, L = float(m["h"][a:z].max()), float(m["l"][a:z].min())
        if H <= L:
            continue
        i15 = int(np.searchsorted(b["t"], st + rng * MIN)) - 1
        if i15 < 0 or i15 >= len(b["t"]):
            continue
        g = {k: int(gates[k][i15]) for k in U.GATES4}
        end = U.local_ms(dt, NY, 16, 0) if exit_at == "ny16" else st + (rng + win) * MIN
        out.append({"kind": kind, "st": st, "a": a, "z": z, "win": win, "H": H, "L": L, "i15": i15, "g": g,
                    "ny16": end, "ny930": U.local_ms(dt, NY, 9, 30)})  # fmt: skip
    return out


def gate_at(gates, b, tms) -> dict:
    i15 = int(np.searchsorted(b["t"], tms - 14 * MIN))
    i15 = min(max(i15, 0), len(b["t"]) - 1)
    return {k: int(gates[k][i15]) for k in U.GATES4}


# ---------------- 진입 ----------------
def breakout(m, b, gates, s, z0=None):
    """돌파 15분 종가 분 j, 방향 d (문 Gany 통과). 없으면 None."""
    c, t = m["c"], m["t"]
    H, L = s["H"], s["L"]
    z = s["z"] if z0 is None else z0
    end = s["z"] + s["win"]
    for q in range(z + 14, end, 15):
        if c[q] > H or c[q] < L:
            d = 1 if c[q] > H else -1
            ok, size = U.gate_ok(gate_at(gates, b, t[q]), d, "Gany")
            return (q, d, size) if ok else None
    return None


def limit_fill(m, j, d, lp, tp, wait, lat, cancel_tp=True):
    h, lo, t = m["h"], m["l"], m["t"]
    for q in range(j + 1 + lat, min(j + 1 + wait + lat, len(t))):
        if cancel_tp and ((h[q] >= tp) if d == 1 else (lo[q] <= tp)):
            return None
        if (lo[q] < lp - TICK) if d == 1 else (h[q] > lp + TICK):
            return q
    return None


def vwap15(b, a_ms, j_ms):
    """세션 VWAP: a_ms(9:30)부터 j_ms에 마감된 15분봉까지, 15분봉 대표가·거래량."""
    i0 = int(np.searchsorted(b["t"], a_ms))
    i1 = int(np.searchsorted(b["t"], j_ms - 14 * MIN)) + 1
    if i1 <= i0:
        return None
    tp_ = (b["h"][i0:i1] + b["l"][i0:i1] + b["c"][i0:i1]) / 3
    v = b["v"][i0:i1]
    return float((tp_ * v).sum() / v.sum()) if v.sum() > 0 else None


def entries(m, b, gates, s, A, lat=0, slip=0.0) -> list:
    """진입 다리 목록 [dict(k, d, ep, stop, tp, fin, size, j)]. 신호 없거나 안 걸리면 []."""
    o, h, lo, c, t = m["o"], m["h"], m["l"], m["c"], m["t"]
    H, L = s["H"], s["L"]
    r = H - L
    if A == "A4":
        ds = {s["g"][k] for k in U.GATES4} - {0}
        if len(ds) != 1:
            return []
        d = ds.pop()
        lp = L + 0.25 * r if d == 1 else H - 0.25 * r
        stop = L - 0.25 * r if d == 1 else H + 0.25 * r
        tp = H + r if d == 1 else L - r
        k = limit_fill(m, s["z"] - 1, d, lp, tp, 60, lat)
        if k is None:
            return []
        return [{"k": k, "d": d, "ep": lp * (1 + d * slip), "stop": stop, "tp": tp, "fin": MAKER, "size": 1.0,
                 "j": s["z"], "ref": c[s["z"] - 1]}]  # fmt: skip
    br = breakout(m, b, gates, s)
    if br is None:
        return []
    j, d, size = br
    stop = L if d == 1 else H
    edge = H if d == 1 else L
    legs = []

    def lim(lp, wait, w=1.0, cancel=True):
        tp = lp + d * r
        k = limit_fill(m, j, d, lp, tp, wait, lat, cancel)
        if k is not None:
            legs.append({"k": k, "d": d, "ep": lp * (1 + d * slip), "stop": stop, "tp": tp, "fin": MAKER,
                         "size": size * w, "j": j, "ref": c[j]})  # fmt: skip

    if A == "A0":
        lim(c[j], 60, cancel=False)
    elif A == "A1":
        lim(edge, 60)
    elif A == "A1b":
        lim(edge, 120)
    elif A == "A2":
        lim(c[j], 60, 0.5, cancel=False)
        lim(edge, 60, 0.5)
    elif A in ("A3a", "A3b"):
        f = 0.382 if A == "A3a" else 0.5
        lim(c[j] - d * f * abs(c[j] - edge), 60)
    elif A == "A3c":
        vw = vwap15(b, s["ny930"], int(t[j]) + MIN)
        lp = (
            edge
            if vw is None or (vw - edge) * d < 0
            else min(vw, c[j])
            if d == 1
            else max(vw, c[j])
        )
        lim(lp, 60)
    elif A == "A5":
        red = False
        for q in range(j + 1 + lat, min(j + 61 + lat, len(t) - 1)):
            if (c[q] < o[q]) if d == 1 else (c[q] > o[q]):
                red = True
                continue
            if red and ((c[q] > h[q - 1]) if d == 1 else (c[q] < lo[q - 1])):
                ep = o[q + 1] + d * TICK
                legs.append({"k": q + 1, "d": d, "ep": ep * (1 + d * slip), "stop": stop, "tp": ep + d * r,
                             "fin": TAKER, "size": size, "j": j, "ref": c[j]})  # fmt: skip
                break
    else:
        raise ValueError(A)
    return legs


def intraday_entry(m, b, gates, s, kind, hour_ema, lat=0, slip=0.0) -> list:
    """B3a VWAP 되찾기 / B3b 1시간 EMA20 눌림. 손절·익절 = 진입가 ∓/± 60분 범위 높이."""
    t, c = m["t"], m["c"]
    r = s["H"] - s["L"]
    day0 = s["ny930"]
    lo_ms, hi_ms = (
        (day0 + 60 * MIN, day0 + 330 * MIN)
        if kind == "B3a"
        else (day0 + 30 * MIN, day0 + 330 * MIN)
    )
    step = 15 if kind == "B3a" else 60
    first = lo_ms + (step - 1) * MIN if kind == "B3a" else None
    if kind == "B3b":  # 1시간봉 마감(정시) 10:00~15:00
        first = day0 + 30 * MIN + 59 * MIN
    q0 = int(np.searchsorted(t, first))
    prev_side = None
    while q0 < len(t) and t[q0] < hi_ms:
        g = gate_at(gates, b, t[q0])
        if kind == "B3a":
            vw = vwap15(b, day0, int(t[q0]) + MIN)
            if vw is None:
                q0 += step
                continue
            side = 1 if c[q0] > vw else -1
            if prev_side is not None and side != prev_side and U.gate_ok(g, side, "Gany")[0]:
                d, lp = side, vw
            else:
                prev_side = side
                q0 += step
                continue
        else:
            hi = int(np.searchsorted(hour_ema[0], int(t[q0]) + MIN - H1))
            if hi >= len(hour_ema[0]) or hour_ema[0][hi] != int(t[q0]) + MIN - H1:
                q0 += step
                continue
            e = hour_ema[1][hi]
            d = 1 if c[q0] > e else -1
            if not U.gate_ok(g, d, "Gany")[0]:
                q0 += step
                continue
            lp = e
        tp = lp + d * r
        k = limit_fill(m, q0, d, lp, tp, 60, lat)
        if k is None:
            return []
        return [{"k": k, "d": d, "ep": lp * (1 + d * slip), "stop": lp - d * r, "tp": tp, "fin": MAKER,
                 "size": 1.0, "j": q0, "ref": c[q0]}]  # fmt: skip
    return []


# ---------------- 묶기 ----------------
def run(m, b, gates, plan, fund_fn, a_ms, z_ms, hour_ema, lat=0, slip=0.0, fee_mult=1.0) -> list:
    """plan: [(우선순위, 세션들, 방식, 재진입여부)]. 한 번에 1포지션(A2 두 다리는 같은 포지션)."""
    ev = []
    for pri, lst, how, reent in plan:
        for s in lst:
            if a_ms <= s["st"] < z_ms:
                ev.append((s["z"], pri, s, how, reent))
    ev.sort(key=lambda x: (x[0], x[1]))
    out, busy = [], -1
    for _, _, s, how, reent in ev:
        z0 = None
        for _rep in range(3 if reent else 1):
            if how in ("B3a", "B3b"):
                legs = intraday_entry(m, b, gates, s, how, hour_ema, lat, slip)
            else:
                ss = s if z0 is None else {**s, "z": z0, "win": s["z"] + s["win"] - z0}
                legs = entries(m, b, gates, ss, how, lat, slip)
            legs = [e for e in legs if e["k"] > busy and m["t"][e["k"]] < z_ms]
            if not legs:
                break
            trs = []
            for e in legs:
                tr = U.exit_trade(m, e, "X6", s, fund_fn, fee_mult, slip)
                tr.update(
                    stop=float(e["stop"]),
                    tp=float(e["tp"]),
                    how=how,
                    j=int(e["j"]),
                    ref=float(e["ref"]),
                )
                trs.append(tr)
            out.extend(trs)
            busy = max(x["x"] for x in trs)
            last = max(trs, key=lambda x: x["x"])
            if not reent or last["why"] != "stop":
                break
            nxt = s["a"] + 15 * int(np.ceil((last["x"] + 1 - s["a"]) / 15))
            if nxt + 15 >= s["z"] + s["win"]:
                break
            z0 = nxt
    out.sort(key=lambda x: x["k"])
    return out


def random_p(m, tr_real, fund_fn, pools, f, nsim, seed) -> dict:
    """시각 맞춘 무작위: 같은 종류 세션에서 같은 범위 끝 기준 오프셋에 시가 진입, 같은 손절·익절 거리."""
    rng = np.random.default_rng(seed)
    real = U.path(tr_real, 1.0, f)["ret"]
    vals = []
    for _ in range(nsim):
        rt = []
        for x in tr_real:
            pool = pools[x["kind"]]
            s = pool[int(rng.integers(len(pool)))]
            k = int(s["z"] + (x["k"] - x["_z"]))
            if k >= len(m["t"]) - 2 or k <= 0:
                continue
            ep = m["o"][k]
            e = {"k": k, "d": x["d"], "ep": ep, "stop": ep * (1 - x["d"] * x["_sd"]), "tp": ep * (1 + x["d"] * x["_td"]),
                 "fin": x["_fin"], "size": x["size"]}  # fmt: skip
            rt.append(U.exit_trade(m, e, "X6", s, fund_fn))
        rt.sort(key=lambda y: y["k"])
        fr = np.array([y["size"] for y in rt]) if f is not None else None
        vals.append(U.path(rt, 1.0, fr)["ret"])
    vals = np.array(vals)
    return {
        "p": float((vals >= real).mean()),
        "rand_med": float(np.median(vals)),
        "real": float(real),
        "nsim": nsim,
    }


def setup() -> dict:
    """자료·문·세션·함수를 한 번에 만든다(본 실행과 사후 점검이 같이 쓴다)."""
    b, s4 = RL.load_all(DW.R5)
    F = DW.features(b)
    QP.add_gates(b, s4, F)
    gates = {k: F["gate"][k] for k in U.GATES4}
    t_end = int(b["t"][-1]) + M15
    m = U.load_1m(t_end)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    prev = json.loads(
        Path("~/.minerva/research/ql-upgrade-2026-11-08/result.json").expanduser().read_text()
    )
    W = {k: tuple(v) for k, v in prev["windows"].items()}
    vt = prev["pick"]["vtarget"]
    dc_i = np.r_[np.nonzero(np.diff(b["t"] // DAY))[0], len(b["t"]) - 1]
    dc, dday = b["c"][dc_i], (b["t"] // DAY)[dc_i]
    dr = np.r_[np.nan, dc[1:] / dc[:-1] - 1]
    rv = np.array([np.nanstd(dr[max(0, i - 20) : i]) if i >= 5 else np.nan for i in range(len(dr))])

    def vol_fn(t_in):
        i = min(max(int(np.searchsorted(dday, t_in // DAY)), 0), len(rv) - 1)
        return float(rv[i]) if np.isfinite(rv[i]) else 0.0

    # 1시간봉 EMA20 (15분봉에서, 정시 마감)
    hk = b["t"] // H1
    hs = np.r_[0, np.nonzero(np.diff(hk))[0] + 1]
    he = np.r_[hs[1:] - 1, len(hk) - 1]
    full = (he - hs) == 3
    ht, hc = (hk[hs] * H1)[full], b["c"][he][full]
    ema = np.empty_like(hc)
    ema[0] = hc[0]
    for i in range(1, len(hc)):
        ema[i] = ema[i - 1] + (hc[i] - ema[i - 1]) * 2 / 21
    hour_ema = (ht.astype(np.int64), ema)

    S = {
        "NY60": make_sessions(m, b, gates, "NY60", NY, 9, 30, 60, 240),
        "NY15": make_sessions(m, b, gates, "NY15", NY, 9, 30, 15, 240),
        "NY30": make_sessions(m, b, gates, "NY30", NY, 9, 30, 30, 240),
        "NY90": make_sessions(m, b, gates, "NY90", NY, 9, 30, 90, 240),
        "AE": make_sessions(m, b, gates, "AE", timezone.utc, 6, 0, 60, 240, exit_at="win"),
        "WE": make_sessions(m, b, gates, "WE", NY, 9, 30, 60, 240, days="we"),
    }

    return {"b": b, "gates": gates, "m": m, "fund_fn": fund_fn, "W": W, "vt": vt, "vol_fn": vol_fn,
            "hour_ema": hour_ema, "S": S}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    C = setup()
    b, gates, m, fund_fn, W, vt, vol_fn, hour_ema, S = (
        C[k] for k in ("b", "gates", "m", "fund_fn", "W", "vt", "vol_fn", "hour_ema", "S")
    )
    print("sessions", {k: len(v) for k, v in S.items()}, round(time.time() - t0, 1), flush=True)

    def ev(plan, windows=SHOW, sz=True, **kw):
        res, keep = {}, {}
        for w in windows:
            tr = run(m, b, gates, plan, fund_fn, *W[w], hour_ema, **kw)
            f = U.sizes(tr, "S1", vol_fn, vt) if (tr and sz) else None
            res[w] = U.summ(tr, *W[w], f)
            keep[w] = (tr, f)
        return res, keep

    def brief(r):
        return {w: (r[w]["n"], round(r[w]["per_day"], 2), round(r[w]["x5_f1.0"]["ret"], 2),
                    round(r[w]["x5_f1.0"]["mdd"], 2)) for w in MAIN}  # fmt: skip

    # ---- A: 진입 위치 ----
    resA, fillA = {}, {}
    nsig = {}
    for w in MAIN:
        nsig[w] = sum(
            1
            for s in S["NY60"]
            if W[w][0] <= s["st"] < W[w][1] and breakout(m, b, gates, s) is not None
        )
    for A in A_NAMES:
        r, kp = ev([(0, S["NY60"], A, False)])
        resA[A] = r
        tr = kp["design"][0]
        imp = [x["d"] * (x["ref"] - x["ep"]) / x["ref"] * 1e4 for x in tr]
        fillA[A] = {"fill_rate_design": len({x["j"] for x in tr}) / max(nsig["design"], 1),
                    "improve_bp_design": float(np.mean(imp)) if imp else None}  # fmt: skip
        print("A", A, brief(r), fillA[A], round(time.time() - t0), flush=True)
    base = resA["A0"]["design"]["x5_f1.0"]
    candA = [A for A in A_NAMES if A != "A0" and resA[A]["design"]["n"] >= 30
             and resA[A]["design"]["x5_f1.0"]["ret"] > base["ret"]
             and resA[A]["design"]["x5_f1.0"]["mdd"] <= base["mdd"] + 0.05]  # fmt: skip
    pickA = max(candA, key=lambda A: resA[A]["design"]["x5_f1.0"]["ret"]) if candA else "A0"
    print("pickA", pickA, flush=True)

    # ---- B: 횟수 ----
    planB = {
        "B1a": [(0, S["NY15"], "A0", False)],
        "B1b": [(0, S["NY30"], "A0", False)],
        "B1c": [(0, S["NY90"], "A0", False)],
        "B1u": [
            (1, S["NY15"], "A0", False),
            (2, S["NY30"], "A0", False),
            (3, S["NY60"], "A0", False),
            (4, S["NY90"], "A0", False),
        ],  # fmt: skip
        "B2": [(0, S["NY60"], "A0", True)],
        "B3a": [(0, S["NY60"], "B3a", False)],
        "B3b": [(0, S["NY60"], "B3b", False)],
        "B4a": [(0, S["AE"], "A0", False)],
        "B4b": [(0, S["AE"], "A0g", False)],
        "B5": [(0, S["WE"], "A0", False)],
    }
    resB, admit = {}, []
    for B in B_NAMES:
        plan = planB[B]
        if B == "B4b":  # 문 = T6D만: 세션 g를 T6D만 남긴 복사본
            se = [
                {**s, "g": {k: (s["g"]["GT"] if k == "GT" else 0) for k in U.GATES4}}
                for s in S["AE"]
            ]
            gT = {k: (gates["GT"] if k == "GT" else np.zeros_like(gates["GT"])) for k in U.GATES4}
            res, kp = {}, {}
            for w in SHOW:
                tr = run(m, b, gT, [(0, se, "A0", False)], fund_fn, *W[w], hour_ema)
                f = U.sizes(tr, "S1", vol_fn, vt) if tr else None
                res[w] = U.summ(tr, *W[w], f)
            r = res
        else:
            r, _ = ev(plan)
        d = r["design"]
        ok = bool(d["n"] >= 30 and (d["avg1"] or 0) > 0 and d["x1"] > 0)
        resB[B] = {**r, "admit": ok}
        if ok:
            admit.append(B)
        print("B", B, ok, brief(r), round(time.time() - t0), flush=True)

    # ---- 최종안 ----
    plan_f = [(0, S["NY60"], pickA, "B2" in admit)]
    pri = 1
    if "B1u" in admit:
        plan_f += [
            (pri, S["NY15"], pickA, False),
            (pri + 1, S["NY30"], pickA, False),
            (pri + 2, S["NY90"], pickA, False),
        ]
        pri += 3
    else:
        for B, kind in (("B1a", "NY15"), ("B1b", "NY30"), ("B1c", "NY90")):
            if B in admit:
                plan_f.append((pri, S[kind], pickA, False))
                pri += 1
    rest = [
        B for B in ("B3a", "B3b", "B4a", "B5") if B in admit
    ]  # B4b는 T6D 전용 문이라 묶음 밖(같은 세션 B4a와 겹침)
    rest.sort(key=lambda B: -(resB[B]["design"]["avg1"] or 0))
    for B in rest:
        sess = {"B3a": S["NY60"], "B3b": S["NY60"], "B4a": S["AE"], "B5": S["WE"]}[B]
        how = B if B in ("B3a", "B3b") else pickA
        plan_f.append((pri, sess, how, False))
        pri += 1
    fin, kfin = ev(plan_f)
    p2, kp2 = ev([(0, S["NY60"], "A0", False)])
    print("final", plan_f and [(p, how, rt) for p, _, how, rt in plan_f], brief(fin), flush=True)

    # ---- 무작위 ----
    pools = {k: v for k, v in S.items()}
    rnd = {}
    for name, keep in (("final", kfin), ("p2", kp2)):
        rnd[name] = {}
        for w in ("pool",) + MAIN:
            tr, f = keep[w]
            if not tr:
                continue
            for x in tr:
                x["_z"] = next(s["z"] for s in pools[x["kind"]] if s["st"] == x["st"])
                x["_sd"] = abs(x["ep"] - x["stop"]) / x["ep"]
                x["_td"] = abs(x["tp"] - x["ep"]) / x["ep"]
                x["_fin"] = TAKER if x["how"] == "A5" else MAKER
            rnd[name][w] = random_p(m, tr, fund_fn, pools, f, args.nsim if w == "pool" else 200, 7)
            print("random", name, w, rnd[name][w], flush=True)

    # ---- 레드팀 ----
    red = {}
    for name, plan in (("final", plan_f), ("p2", [(0, S["NY60"], "A0", False)])):
        red[name] = {}
        for tag, kw in (("lat1", {"lat": 1}), ("lat5", {"lat": 5}), ("slip2bp", {"slip": 0.0002}),
                        ("fee15", {"fee_mult": 1.5})):  # fmt: skip
            r, _ = ev(plan, windows=MAIN, **kw)
            red[name][tag] = {w: r[w]["x5_f1.0"] for w in MAIN}
        tr = run(m, b, gates, plan, fund_fn, W["fresh"][0], W["year"][1], hour_ema)
        f = U.sizes(tr, "S1", vol_fn, vt)
        yrs: dict = {}
        for i, x in enumerate(tr):
            yrs.setdefault(datetime.fromtimestamp(x["t_in"] / 1000, timezone.utc).year, []).append(
                i
            )
        red[name]["by_year"] = {
            y: {"n": len(ix), **U.path([tr[i] for i in ix], 1.0, f[ix])}
            for y, ix in sorted(yrs.items())
        }
        rolls, s0 = [], W["fresh"][0]
        while s0 + 90 * DAY <= W["year"][1]:
            ix = [i for i, x in enumerate(tr) if s0 <= x["t_in"] < s0 + 90 * DAY]
            rolls.append(U.path([tr[i] for i in ix], 1.0, f[ix])["ret"] if ix else 0.0)
            s0 += 30 * DAY
        rolls = np.array(rolls)
        red[name]["rolling90"] = {"n": len(rolls), "pos_share": float((rolls > 0).mean()),
                                  "median": float(np.median(rolls)), "worst": float(rolls.min())}  # fmt: skip
        print("red", name, {k: v for k, v in red[name].items() if k != "by_year"}, flush=True)
        print(
            "by_year",
            name,
            {y: (v["n"], round(v["ret"], 2)) for y, v in red[name]["by_year"].items()},
            flush=True,
        )

    beat = sum(fin[w]["x5_f1.0"]["ret"] > p2[w]["x5_f1.0"]["ret"] and fin[w]["x5_f1.0"]["mdd"] <= p2[w]["x5_f1.0"]["mdd"] + 0.05
               for w in ("design", "val", "year"))  # fmt: skip
    gate = {"pos_all_f1": all(fin[w]["x5_f1.0"]["ret"] > 0 for w in MAIN),
            "liq0": all(fin[w]["x5_f1.0"]["liq"] == 0 for w in MAIN),
            "random_bonf": rnd["final"].get("pool", {"p": 1})["p"] * N_TESTS < 0.05,
            "beats_p2_2of3": beat >= 2}  # fmt: skip

    def slim(keep):
        return {
            w: [{k: v for k, v in x.items() if not k.startswith("_")} for x in keep[w][0]]
            for w in ("d30", "m6", "year")
        }

    (out / "trades.json").write_text(
        json.dumps({"final": slim(kfin), "p2": slim(kp2)}, default=float)
    )
    (out / "result.json").write_text(json.dumps(
        {"n_tests": N_TESTS, "n_before": N_BEFORE, "pickA": pickA, "admitB": admit,
         "plan": [(p, how, rt) for p, _, how, rt in plan_f], "A": resA, "A_fill": fillA, "nsig": nsig, "B": resB,
         "final": fin, "p2": p2, "random": rnd, "redteam": red, "gate": gate, "pass": all(gate.values()),
         "windows": W, "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    print("gate", gate, "pickA", pickA, "admit", admit, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
