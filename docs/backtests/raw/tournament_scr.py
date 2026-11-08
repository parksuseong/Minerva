"""토너먼트 계열 S: 돌파 + SCR 과열 거르기(A1~A4), 일봉 방향 + SCR 3단 눌림(B 8개). 1분봉 체결.

등록: 2026-11-08-btc-technique-tournament-hypotheses.md (계열 S, 12개). 조사 전용(주문 없음).
진입·청산·비중·문은 시간대 없는 돌파 시험(sf_eval)과 같은 코드 경로.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/tournament-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/tournament_scr.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import daily_wave as DW
import numpy as np
import ql_portfolio as QP
import ql_upgrade as U
import regime_lines as RL
import sf_eval as SF
from btc_fractal import cci, crossed, recent, rsi, stoch

MIN, M15, DAY, HOUR = U.MIN, U.M15, U.DAY, 3_600_000
TAKER, MAKER, TICK = U.TAKER, U.MAKER, U.TICK
MAIN, SHOW = U.MAIN, U.SHOW
SF_RES = Path("~/.minerva/research/session-free-2026-11-08/result.json").expanduser()


def frame_scr(b: dict, period: int) -> dict:
    """큰 봉 SCR 값과 3단 신호. 값은 그 큰 봉이 끝난 15분봉 위치(e)에 둔다(그 봉 마감 뒤에 앎)."""
    g = b["t"] // period
    s = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    e = np.r_[s[1:], len(g)] - 1
    if int(b["t"][e[-1]]) + M15 < (int(g[e[-1]]) + 1) * period:
        s, e = s[:-1], e[:-1]
    H = np.maximum.reduceat(b["h"], s)[: len(s)]
    L = np.minimum.reduceat(b["l"], s)[: len(s)]
    C = b["c"][e]
    k, c_, r = stoch(H, L, C, 5), cci(H, L, C, 20), rsi(C, 14)
    lx = recent(crossed(k, 20, True)) & recent(crossed(c_, -100, True)) & recent(crossed(r, 30, True))
    sx = recent(crossed(k, 80, False)) & recent(crossed(c_, 100, False)) & recent(crossed(r, 70, False))
    L3 = lx & ~np.r_[False, lx[:-1]]
    S3 = sx & ~np.r_[False, sx[:-1]]
    return {"e": e, "s": s, "H": H, "L": L, "C": C, "k": k, "c": c_, "r": r, "L3": L3, "S3": S3}


def extreme_count(fs: dict, j: int, d: int) -> int:
    k, c_, r = fs["k"][j], fs["c"][j], fs["r"][j]
    if d == 1:
        return int(k >= 80) + int(c_ >= 100) + int(r >= 70)
    return int(k <= 20) + int(c_ <= -100) + int(r <= 30)


def last_closed(fs: dict, i15: int) -> int:
    """15분봉 i15 마감 때 이미 끝나 있던 큰 봉 번호."""
    return int(np.searchsorted(fs["e"], i15, side="right")) - 1


def breakout_signal(m, b, s):
    """sf/ql E0m과 같은 돌파 15분봉 찾기: (j 분, d, i15) 또는 None."""
    c, t = m["c"], m["t"]
    H, L, z, win = s["H"], s["L"], s["z"], s["win"]
    for q in range(z + 14, z + win, 15):
        if c[q] > H or c[q] < L:
            return q, (1 if c[q] > H else -1), int(np.searchsorted(b["t"], t[q] - 14 * MIN))
    return None


def limit_fill(m, j, d, lp, lat=0, tp=None):
    h, lo, t = m["h"], m["l"], m["t"]
    for q in range(j + 1 + lat, min(j + 61 + lat, len(t))):
        if (lo[q] < lp - TICK) if d == 1 else (h[q] > lp + TICK):
            return q
    return None


def entry_a(m, b, gates, s, G, mode, f15, f1h, lat=0, slip=0.0):
    """A1~A4: 돌파 + SCR 과열 거르기/미룸. 반환 (entry dict, scr 값) 또는 None."""
    sig = breakout_signal(m, b, s)
    if sig is None:
        return None
    j, d, i15 = sig
    g = {kk: int(gates[kk][i15]) for kk in U.GATES4}
    ok, size = U.gate_ok(g, d, G)
    if not ok:
        return None
    j15 = last_closed(f15, i15)
    ext15 = extreme_count(f15, j15, d)
    vals = (float(f15["k"][j15]), float(f15["c"][j15]), float(f15["r"][j15]))
    if mode == "A1" and ext15 >= 2:
        return None
    if mode == "A2" and ext15 >= 3:
        return None
    if mode == "A3":
        jh = last_closed(f1h, i15)
        if jh >= 0 and extreme_count(f1h, jh, d) >= 2:
            return None
    rngh = s["H"] - s["L"]
    lp = m["c"][j]
    if mode == "A4" and ext15 >= 2:
        went = False
        found = None
        for jj in range(j15 + 1, min(j15 + 17, len(f15["k"]))):
            kv = f15["k"][jj]
            if not went:
                went = kv < 50 if d == 1 else kv > 50
            elif (kv >= 50) if d == 1 else (kv <= 50):
                found = jj
                break
        if found is None:
            return None
        i2 = int(f15["e"][found])
        g2 = {kk: int(gates[kk][i2]) for kk in U.GATES4}
        ok, size = U.gate_ok(g2, d, G)
        if not ok:
            return None
        j = int(np.searchsorted(m["t"], b["t"][i2] + 14 * MIN))
        if j >= len(m["t"]):
            return None
        lp = m["c"][j]
        stop = s["L"] if d == 1 else s["H"]
        if (lp - stop) * d <= 0:
            return None
        vals = (float(f15["k"][found]), float(f15["c"][found]), float(f15["r"][found]))
    k = limit_fill(m, j, d, lp, lat)
    if k is None:
        return None
    ep = lp * (1 + d * slip)
    e = {"k": k, "d": d, "ep": ep, "stop": s["L"] if d == 1 else s["H"], "tp": lp + d * rngh, "fin": MAKER,
         "size": size, "fill_same_bar": False}  # fmt: skip
    return e, vals


def scr_setups(b, fs, nb=8):
    """B: SCR 3단 신호 셋업. st = 신호 큰 봉 마감 시각."""
    out = []
    for j in range(len(fs["e"])):
        for d, arr in ((1, fs["L3"]), (-1, fs["S3"])):
            if not arr[j] or j < nb:
                continue
            i15 = int(fs["e"][j])
            lo_ = float(fs["L"][j - nb + 1 : j + 1].min())
            hi_ = float(fs["H"][j - nb + 1 : j + 1].max())
            out.append({"kind": "SCR", "st": int(b["t"][i15]) + M15, "i15": i15, "d": d, "j": j,
                        "stop": lo_ - TICK if d == 1 else hi_ + TICK,
                        "vals": (float(fs["k"][j]), float(fs["c"][j]), float(fs["r"][j]))})  # fmt: skip
    out.sort(key=lambda x: x["st"])
    return out


def entry_b(m, b, gates, s, G, lat=0, slip=0.0):
    d, i15 = s["d"], s["i15"]
    g = {kk: int(gates[kk][i15]) for kk in U.GATES4}
    ok, size = U.gate_ok(g, d, G)
    if not ok:
        return None
    j = int(np.searchsorted(m["t"], b["t"][i15] + 14 * MIN))
    if j >= len(m["t"]) or m["t"][j] != b["t"][i15] + 14 * MIN:
        return None
    lp = m["c"][j]
    R = (lp - s["stop"]) * d
    if R <= 0:
        return None
    k = limit_fill(m, j, d, lp, lat)
    if k is None:
        return None
    ep = lp * (1 + d * slip)
    return {"k": k, "d": d, "ep": ep, "stop": s["stop"], "tp": lp + d * R, "fin": MAKER, "size": size,
            "fill_same_bar": False}  # fmt: skip


def run_family(m, b, gates, kind, lst, G, cfg, fund_fn, a_ms, z_ms, f15=None, f1h=None, **kw):
    out, busy = [], -1
    lat, slip = kw.get("lat", 0), kw.get("slip", 0.0)
    for s in lst:
        if s["st"] < a_ms:
            continue
        if s["st"] >= z_ms:
            break
        if kind == "B":
            e = entry_b(m, b, gates, s, G, lat, slip)
            vals = s["vals"]
        else:
            if s["z"] <= busy:
                continue
            r = entry_a(m, b, gates, s, G, kind, f15, f1h, lat, slip)
            e, vals = (r if r else (None, None))
        if e is None or e["k"] <= busy or m["t"][e["k"]] >= z_ms:
            continue
        tr = SF.sf_exit(m, e, s, cfg["X"], cfg.get("hold", 480), fund_fn, kw.get("fee_mult", 1.0), slip,
                        cfg.get("bail"))  # fmt: skip
        tr["scr"] = vals
        out.append(tr)
        busy = tr["x"]
    return out


def bail_table(m, trs):
    cnt: dict = {}
    for x in trs:
        R = abs(x["ep"] - x["_stop"])
        for hr in range(1, 8):
            j = x["k"] + 60 * hr
            if j >= x["x"]:
                break
            u = x["d"] * (m["c"][j] - x["ep"]) / R if R > 0 else 0.0
            bk = int(np.digitize(u, [-0.5, 0.0, 0.5]))
            c_ = cnt.setdefault((hr, bk), [0, 0])
            c_[0] += 1
            c_[1] += x["why"] == "tp"
    return {k: v[1] / v[0] for k, v in cnt.items() if v[0] >= 20}


def scr_stats(tr):
    out = {}
    for d, nm in ((1, "long"), (-1, "short")):
        v = np.array([x["scr"] for x in tr if x["d"] == d and x.get("scr")])
        if len(v):
            out[nm] = {k: {"med": float(np.median(v[:, i])), "lo": float(v[:, i].min()), "hi": float(v[:, i].max())}
                       for i, k in enumerate(("stoch", "cci", "rsi"))}  # fmt: skip
    return out


def random_b(m, tr_real, rvq_min, cfg, fund_fn, a_ms, z_ms, nsim, seed, f, t_m):
    """같은 방향·같은 손절/익절 거리, 같은 30일 변동성 5분위 분에서 무작위 진입."""
    rng = np.random.default_rng(seed)
    real = U.path(tr_real, 1.0, f)["ret"]
    a, z = int(np.searchsorted(t_m, a_ms)), int(np.searchsorted(t_m, z_ms))
    pools = {q: np.flatnonzero(rvq_min[a:z] == q) + a for q in range(5)}
    vals = []
    for _ in range(nsim):
        rt = []
        for x in tr_real:
            pool = pools[int(rvq_min[x["k"]])]
            k = int(pool[int(rng.integers(len(pool)))]) if len(pool) else x["k"]
            if k >= len(t_m) - 600:
                continue
            ep = m["o"][k]
            sd, td = abs(x["ep"] - x["_stop"]) / x["ep"], abs(x["_tp"] - x["ep"]) / x["ep"]
            e = {"k": k, "d": x["d"], "ep": ep, "stop": ep * (1 - x["d"] * sd), "tp": ep * (1 + x["d"] * td),
                 "fin": MAKER, "size": x["size"]}  # fmt: skip
            rt.append(SF.sf_exit(m, e, {"kind": "R", "st": 0}, cfg["X"], cfg.get("hold", 480), fund_fn,
                                 bail=cfg.get("bail")))  # fmt: skip
        rt.sort(key=lambda y: y["k"])
        vals.append(U.path(rt, 1.0, np.array([y["size"] for y in rt]) * (f.mean() if len(f) else 1))["ret"])
    vals = np.array(vals)
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "real": float(real), "nsim": nsim}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    t0 = time.time()
    b, s4 = RL.load_all(DW.R5)
    F = DW.features(b)
    QP.add_gates(b, s4, F)
    gates = {k: F["gate"][k] for k in U.GATES4}
    m = U.load_1m(int(b["t"][-1]) + M15)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    sfr = json.loads(SF_RES.read_text())
    W = {k: tuple(v) for k, v in sfr["windows"].items()}
    up = json.loads(SF.QL_UP.read_text())
    vt = up["pick"]["vtarget"]
    dc_i = np.r_[np.nonzero(np.diff(b["t"] // DAY))[0], len(b["t"]) - 1]
    dc, dday = b["c"][dc_i], (b["t"] // DAY)[dc_i]
    dr = np.r_[np.nan, dc[1:] / dc[:-1] - 1]
    rv = np.array([np.nanstd(dr[max(0, i - 20) : i]) if i >= 5 else np.nan for i in range(len(dr))])

    def vol_fn(t_in):
        i = min(max(int(np.searchsorted(dday, t_in // DAY)), 0), len(rv) - 1)
        return float(rv[i]) if np.isfinite(rv[i]) else 0.0

    # 무작위용: 분마다 30일 변동성 5분위(설계 구간 기준 경계)
    rv30 = np.array([np.nanstd(dr[max(0, i - 30) : i]) if i >= 10 else np.nan for i in range(len(dr))])
    dsel = (dday * DAY >= W["design"][0]) & (dday * DAY < W["design"][1])
    qs = np.nanquantile(rv30[dsel], [0.2, 0.4, 0.6, 0.8])
    dq = np.searchsorted(qs, np.nan_to_num(rv30, nan=float(np.nanmedian(rv30[dsel]))))
    mi = np.clip(np.searchsorted(dday, m["t"] // DAY), 0, len(dq) - 1)
    rvq_min = dq[mi]

    f15, f1h = frame_scr(b, M15), frame_scr(b, HOUR)
    boxes = SF.donchian(m, b, 96, "D24h")
    setups_b = {"15m": scr_setups(b, f15), "1h": scr_setups(b, f1h)}
    print("loaded", len(m["t"]), {k: len(v) for k, v in setups_b.items()}, len(boxes), round(time.time() - t0), flush=True)

    def evaluate(kind, lst, G, cfg, windows=SHOW, **kw):
        res, keep = {}, {}
        for w in windows:
            tr = run_family(m, b, gates, kind, lst, G, cfg, fund_fn, *W[w], f15=f15, f1h=f1h, **kw)
            for x in tr:
                x["_stop"], x["_tp"] = x.get("_stop", 0), x.get("_tp", 0)
            f = U.sizes(tr, "S1", vol_fn, vt) if tr else np.zeros(0)
            res[w] = U.summ(tr, *W[w], f if len(f) else None)
            res[w]["scr"] = scr_stats(tr)
            keep[w] = (tr, f)
        return res, keep

    res, keeps, specs = {}, {}, {}
    for mode in ("A1", "A2", "A3", "A4"):
        nm = f"{mode}|D24h|Gsz|X5"
        specs[nm] = (mode, boxes, "Gsz", {"X": "X5", "hold": 480})
        res[nm], keeps[nm] = evaluate(*specs[nm])
        print(nm, {w: (res[nm][w]["n"], round(res[nm][w]["per_day"], 2), round(res[nm][w]["x5_f1.0"]["ret"], 2),
                       round(res[nm][w]["x5_f1.0"]["mdd"], 2)) for w in MAIN + ("d30",)}, round(time.time() - t0),
              flush=True)  # fmt: skip
    for fr_ in ("15m", "1h"):
        for G in ("Gsz", "Gany"):
            nm = f"B|SCR{fr_}|{G}|X5"
            specs[nm] = ("B", setups_b[fr_], G, {"X": "X5", "hold": 480})
            res[nm], keeps[nm] = evaluate(*specs[nm])
            # X4: 설계 구간 X5 거래로 이탈 확률 표 (손절가를 다시 붙여 R 계산)
            dtr = keeps[nm]["design"][0]
            st_by = {s["st"]: s for s in setups_b[fr_]}
            for x in dtr:
                x["_stop"] = st_by[x["st"]]["stop"]
            bail = bail_table(m, dtr)
            nm4 = f"B|SCR{fr_}|{G}|X4"
            specs[nm4] = ("B", setups_b[fr_], G, {"X": "X4", "bail": bail})
            res[nm4], keeps[nm4] = evaluate(*specs[nm4])
            for n_ in (nm, nm4):
                print(n_, {w: (res[n_][w]["n"], round(res[n_][w]["per_day"], 2),
                               round(res[n_][w]["x5_f1.0"]["ret"], 2), round(res[n_][w]["x5_f1.0"]["mdd"], 2))
                           for w in MAIN + ("d30",)}, round(time.time() - t0), flush=True)  # fmt: skip
    assert len(res) == 12, len(res)

    # 고르기: 설계 구간 5배·40% Calmar (본 등록과 같은 기준)
    def cal(n):
        x = res[n]["design"]["x5_f0.4"]
        return (x["ret"] / max(x["mdd"], 0.05)) if (x["liq"] == 0 and res[n]["design"]["n"] >= 20) else -9

    best = max(res, key=cal)
    print("best S", best, flush=True)
    kind, lst, G, cfg = specs[best]
    st_all = {}
    for fr_ in ("15m", "1h"):
        for s in setups_b[fr_]:
            st_all[(fr_, s["st"])] = s
    rnd = {}
    for w in ("pool",) + MAIN:
        tr, f = keeps[best][w]
        for x in tr:
            if kind == "B":
                x["_stop"] = next(s["stop"] for s in lst if s["st"] == x["st"])
            else:
                bx = next(s for s in lst if s["st"] == x["st"])
                x["_stop"] = bx["L"] if x["d"] == 1 else bx["H"]
            x["_tp"] = x["ep"] + x["d"] * (abs(x["ep"] - x["_stop"]) if kind == "B" else 0)
            if kind != "B":
                bx = next(s for s in lst if s["st"] == x["st"])
                x["_tp"] = x["ep"] + x["d"] * (bx["H"] - bx["L"])
        rnd[w] = random_b(m, tr, rvq_min, cfg, fund_fn, *W[w], 200 if w == "pool" else 100, 7, f, m["t"])
        print("random", w, rnd[w], round(time.time() - t0), flush=True)
    red = {}
    for tag, kw in (("lat1", {"lat": 1}), ("lat5", {"lat": 5}), ("slip2bp", {"slip": 0.0002}),
                    ("fee15", {"fee_mult": 1.5})):  # fmt: skip
        r, _ = evaluate(kind, lst, G, cfg, windows=MAIN, **kw)
        red[tag] = {w: r[w]["x5_f1.0"] for w in MAIN}
    a_ms, z_ms = W["fresh"][0], W["year"][1]
    tr = run_family(m, b, gates, kind, lst, G, cfg, fund_fn, a_ms, z_ms, f15=f15, f1h=f1h)
    f = U.sizes(tr, "S1", vol_fn, vt)
    yrs: dict = {}
    for i, x in enumerate(tr):
        yrs.setdefault(datetime.fromtimestamp(x["t_in"] / 1000, timezone.utc).year, []).append(i)
    red["by_year"] = {y: {"n": len(ix), **U.path([tr[i] for i in ix], 1.0, f[ix])} for y, ix in sorted(yrs.items())}
    print("red", red, flush=True)

    def slim(keep):
        return {w: [{k: v for k, v in x.items() if not k.startswith("_")} for x in keep[w][0]]
                for w in ("d30", "m6", "year")}  # fmt: skip

    (out / "scr_trades.json").write_text(json.dumps({"best": slim(keeps[best])}, default=float))
    (out / "scr_result.json").write_text(json.dumps(
        {"n_tests": 12, "best": best, "res": res, "random": rnd, "redteam": red,
         "sf_final": sfr["final"], "sf_res": sfr["res"][sfr["final"]], "windows": W,
         "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
