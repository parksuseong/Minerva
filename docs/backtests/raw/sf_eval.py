"""시간대 없는 돌파 모형: 압축 상자·돈치안·레짐 횡보 다리를 일봉 방향으로 돌파 (1분봉 체결).

등록: docs/backtests/2026-11-08-btc-session-free-hypotheses.md (27개). 연구 전용(주문 없음).
진입·손절·익절·비중은 P2(ql_upgrade_posthoc)와 같고, 범위만 시계 대신 변동성 상태로 잡는다.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/session-free-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/sf_eval.py --out $OUT
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
from numpy.lib.stride_tricks import sliding_window_view

MIN, M15, DAY, H4 = U.MIN, U.M15, U.DAY, RL.H4
TAKER, MAKER, TICK = U.TAKER, U.MAKER, U.TICK
MIN_H = 0.0025  # 상자 높이 하한(가격 대비)
N_BONF, N_BEFORE = 27, 6003
QL_UP = Path("~/.minerva/research/ql-upgrade-2026-11-08/result.json").expanduser()
MAIN, SHOW = U.MAIN, U.SHOW


# ---------------- 상자 ----------------
def minute_index(m: dict, ts: np.ndarray) -> np.ndarray:
    k = np.searchsorted(m["t"], ts)
    ok = (k < len(m["t"])) & (m["t"][np.minimum(k, len(m["t"]) - 1)] == ts)
    return np.where(ok, k, -1)


def mk_setups(m, b, kind, ib, H, L, win_min, start_ms) -> list:
    """ib: 상자가 끝난 15분봉 위치(그 봉 마감 뒤에 앎). start_ms: 상자 시작 시각.
    메모리 때문에 진입 창 안 15분봉 종가가 한 번도 상자 밖으로 안 나가는 셋업은 미리 뺀다(결과는 같다)."""
    nb = win_min // 15
    c = b["c"]
    ext = np.r_[c, np.full(nb, np.nan)]
    nxt = sliding_window_view(ext[1:], nb)[: len(c)]
    with np.errstate(invalid="ignore"):
        brk = (np.nanmax(nxt[ib], 1) > H) | (np.nanmin(nxt[ib], 1) < L)
    ib, H, L, start_ms = ib[brk], H[brk], L[brk], start_ms[brk]
    zt = b["t"][ib].astype(np.int64) + M15
    z = minute_index(m, zt)
    a = np.searchsorted(m["t"], start_ms)
    out = []
    n1 = len(m["t"])
    for q in range(len(ib)):
        if z[q] < 0 or z[q] + win_min + 1 >= n1 or H[q] <= L[q]:
            continue
        if (H[q] - L[q]) / b["c"][ib[q]] < MIN_H:
            continue
        out.append({"kind": kind, "st": int(zt[q]), "a": int(a[q]), "z": int(z[q]), "win": int(win_min),
                    "H": float(H[q]), "L": float(L[q])})  # fmt: skip
    return out


def compression(m, b, N: int, q: int) -> list:
    h, lo, c, t = b["h"], b["l"], b["c"], b["t"]
    n = len(c)
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    hh[N - 1 :] = sliding_window_view(h, N).max(1)
    ll[N - 1 :] = sliding_window_view(lo, N).min(1)
    w = (hh - ll) / c
    # 문턱: 그날 0시(UTC) 전 30일(2,880봉) 값의 q 분위 — 하루 한 번 갱신(구현 메모)
    day = t // DAY
    starts = np.r_[0, np.nonzero(np.diff(day))[0] + 1]
    thr = np.full(n, np.nan)
    for k, s0 in enumerate(starts):
        e0 = starts[k + 1] if k + 1 < len(starts) else n
        hist = w[max(0, s0 - 2880) : s0]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= 2000:
            thr[s0:e0] = np.quantile(hist, q / 100)
    with np.errstate(invalid="ignore"):
        ib = np.nonzero(w <= thr)[0]
    return mk_setups(m, b, f"C{N}q{q}", ib, hh[ib], ll[ib], 16 * 15, t[np.maximum(ib - N + 1, 0)].astype(np.int64))


def donchian(m, b, N: int, name: str) -> list:
    h, lo, t = b["h"], b["l"], b["t"]
    n = len(h)
    ib = np.arange(N - 1, n)
    H = sliding_window_view(h, N).max(1)
    L = sliding_window_view(lo, N).min(1)
    return mk_setups(m, b, name, ib, H, L, 15, t[ib - N + 1].astype(np.int64))


def sideways(m, b, s4, seg: str) -> list:
    E = RL.events(s4, seg)
    P, C, lab = E["P"], E["C"], E["lab"]
    ib, H, L, st = [], [], [], []
    for e in range(1, len(P)):
        if lab[e] != 0:
            continue
        i15 = int(s4["i15"][C[e]])
        if i15 < 0:
            continue
        p0, p1 = int(P[e - 1]), int(P[e])
        ib.append(i15)
        H.append(float(s4["h"][p0 : p1 + 1].max()))
        L.append(float(s4["l"][p0 : p1 + 1].min()))
        st.append(int(s4["t"][p0]))
    return mk_setups(m, b, f"R{seg}", np.array(ib, int), np.array(H), np.array(L), 1440, np.array(st, np.int64))


# ---------------- 청산 (X5: 익절 지정가, 시간 청산은 분 단위) ----------------
def sf_exit(m, e, s, X, hold, fund_fn, fee_mult=1.0, slip=0.0, bail=None):
    if X == "X4":
        return U.exit_trade(m, e, "X4", s, fund_fn, fee_mult, slip, bail)
    t, c = m["t"], m["c"]
    k, d, ep, stop, tp = e["k"], e["d"], e["ep"], e["stop"], e["tp"]
    tend = min(k + hold, len(t) - 1)
    hit = U.first_hit(m, d, k, tend, stop, tp, k + 1)
    if hit is None:
        px, why, jx = c[tend], "time", tend
    elif hit[1] == "stop":
        px, why, jx = U.stop_px(m, hit[0], d, stop, slip), "stop", hit[0]
    else:
        px, why, jx = tp, "tp", hit[0]
    fin = e["fin"] * fee_mult
    fout = (MAKER if why == "tp" else TAKER) * fee_mult
    lo_, hi_ = m["l"][k : jx + 1], m["h"][k : jx + 1]
    worst = (lo_.min() / ep - 1) if d == 1 else (1 - hi_.max() / ep)
    t_in, t_out = int(t[k]), int(t[jx]) + MIN
    fund = fund_fn(t_in, t_out, d)
    r = d * (px / ep - 1) - fin - fout + fund
    return {"k": int(k), "x": int(jx), "d": int(d), "ep": float(ep), "xp": float(px), "r": float(r),
            "worst": float(min(worst, r)), "why": why, "size": float(e["size"]), "fund": float(fund),
            "t_in": t_in, "t_out": t_out, "kind": s["kind"], "st": int(s["st"])}  # fmt: skip


_ST: dict = {}


def sf_run(m, b, gates, lists, cfg, fund_fn, a_ms, z_ms, **kw) -> list:
    ev = []
    for pri, lst, G in lists:
        key = id(lst)
        if key not in _ST:
            _ST[key] = np.array([s["st"] for s in lst], np.int64)
        lo_i = int(np.searchsorted(_ST[key], a_ms))
        for s in lst[lo_i:]:
            if s["st"] >= z_ms:
                break
            ev.append((s["st"], pri, s, G))
    ev.sort(key=lambda x: (x[0], x[1]))
    out, busy = [], -1
    lat, slip = kw.get("lat", 0), kw.get("slip", 0.0)
    for st, _, s, G in ev:
        if s["z"] <= busy:
            continue
        e = U.entry(m, b, gates, s, "E0m", G, lat=lat, slip=slip)
        if e is None or e["k"] <= busy or m["t"][e["k"]] >= z_ms:
            continue
        tr = sf_exit(m, e, s, cfg["X"], cfg.get("hold", 480), fund_fn, kw.get("fee_mult", 1.0), slip, cfg.get("bail"))
        out.append(tr)
        busy = tr["x"]
    return out


def sf_random(m, tr_real, cfg, fund_fn, pool, by_key, nsim, seed, f) -> dict:
    """변동성 상태 맞춤: 같은 상자 종류의 셋업에서 무작위로 고르고, 상자 시작 기준 진입 위치·방향·손절·익절 거리 유지."""
    rng = np.random.default_rng(seed)
    real = U.path(tr_real, 1.0, f)["ret"]
    pools = {}
    for s in pool:
        pools.setdefault(s["kind"], []).append(s)
    vals = []
    for _ in range(nsim):
        rt = []
        for x in tr_real:
            src = by_key[(x["kind"], x["st"])]
            cand = pools[x["kind"]]
            s = cand[int(rng.integers(len(cand)))]
            k = int(s["z"] + (x["k"] - src["z"]))
            if k >= len(m["t"]) - 2:
                continue
            ep = m["o"][k]
            sd = abs(x["ep"] - (src["L"] if x["d"] == 1 else src["H"])) / x["ep"]
            td = abs(src["H"] - src["L"]) / x["ep"]
            e = {"k": k, "d": x["d"], "ep": ep, "stop": ep * (1 - x["d"] * sd), "tp": ep * (1 + x["d"] * td),
                 "fin": MAKER, "size": x["size"]}  # fmt: skip
            rt.append(sf_exit(m, e, s, cfg["X"], cfg.get("hold", 480), fund_fn, bail=cfg.get("bail")))
        rt.sort(key=lambda y: y["k"])
        vals.append(U.path(rt, 1.0, np.array([y["size"] for y in rt]))["ret"])
    vals = np.array(vals)
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "real": float(real), "nsim": nsim}


def us_share(tr: list) -> dict:
    """뉴욕 정규장(9:30~16:00, 평일)에 들어간 거래 비율과 손익 비율 — 맥락 확인용."""
    if not tr:
        return {"trades": None, "pnl": None}
    inside, pin, ptot = 0, 0.0, 0.0
    for x in tr:
        lt = datetime.fromtimestamp(x["t_in"] / 1000, U.NY)
        mins = lt.hour * 60 + lt.minute
        ins = lt.weekday() < 5 and 570 <= mins < 960
        inside += ins
        pin += x["r"] if ins else 0.0
        ptot += x["r"]
    return {"trades": inside / len(tr), "pnl_in": pin, "pnl_all": ptot}


# ---------------- 실행 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
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

    up = json.loads(QL_UP.read_text())
    W = {k: tuple(v) for k, v in up["windows"].items()}
    vt = up["pick"]["vtarget"]
    dc_i = np.r_[np.nonzero(np.diff(b["t"] // DAY))[0], len(b["t"]) - 1]
    dc, dday = b["c"][dc_i], (b["t"] // DAY)[dc_i]
    dr = np.r_[np.nan, dc[1:] / dc[:-1] - 1]
    rv = np.array([np.nanstd(dr[max(0, i - 20) : i]) if i >= 5 else np.nan for i in range(len(dr))])

    def vol_fn(t_in):
        i = min(max(int(np.searchsorted(dday, t_in // DAY)), 0), len(rv) - 1)
        return float(rv[i]) if np.isfinite(rv[i]) else 0.0

    print("loaded", len(m["t"]), round(time.time() - t0, 1), flush=True)
    boxes = {}
    for N in (4, 8, 16):
        for q in (20, 30):
            boxes[f"C{N}q{q}"] = compression(m, b, N, q)
    for N, nm in ((16, "D4h"), (32, "D8h"), (96, "D24h")):
        boxes[nm] = donchian(m, b, N, nm)
    for seg in ("L3", "Z5"):
        boxes[f"R{seg}"] = sideways(m, b, s4, seg)
    by_key = {(s["kind"], s["st"]): s for v in boxes.values() for s in v}
    print("setups", {k: len(v) for k, v in boxes.items()}, round(time.time() - t0, 1), flush=True)

    def evaluate(lists, cfg, windows=SHOW, **kw):
        res, keep = {}, {}
        for w in windows:
            tr = sf_run(m, b, gates, lists, cfg, fund_fn, *W[w], **kw)
            f = U.sizes(tr, "S1", vol_fn, vt) if tr else None
            res[w] = U.summ(tr, *W[w], f)
            res[w]["us"] = us_share(tr)
            keep[w] = (tr, f)
        return res, keep

    def show(name, r):
        print(name, {w: (r[w]["n"], round(r[w]["per_day"], 2), round(r[w]["x5_f1.0"]["ret"], 2),
                         round(r[w]["x5_f1.0"]["mdd"], 2)) for w in MAIN + ("d30",)},
              round(time.time() - t0), flush=True)  # fmt: skip

    # 기준: P2 (뉴욕 60분, X6, S1) 같은 코드 경로로 다시
    sess = U.sessions(m, b, gates, "NY")
    p2 = {}
    p2_keep = {}
    for w in SHOW:
        tr = U.run(m, b, gates, [(0, sess, "E0m", "Gany")], {"X": "X6"}, fund_fn, *W[w])
        f = U.sizes(tr, "S1", vol_fn, vt) if tr else None
        p2[w] = U.summ(tr, *W[w], f)
        p2[w]["us"] = us_share(tr)
        p2_keep[w] = (tr, f)
    show("P2", p2)

    res, keeps, spec = {}, {}, {}
    for bx, lst in boxes.items():
        for G in ("Gany", "Gsz"):
            name = f"{bx}|{G}|X5"
            spec[name] = ([(0, lst, G)], {"X": "X5", "hold": 480})
            res[name], keeps[name] = evaluate(*spec[name])
            show(name, res[name])

    def pick(names):
        ok = [n for n in names if res[n]["design"]["n"] >= 30]
        if not ok:
            return None
        top = max(res[n]["design"]["x5_f1.0"]["ret"] for n in ok)
        near = [n for n in ok if res[n]["design"]["x5_f1.0"]["ret"] >= top - 0.10 * abs(top)]
        return min(near, key=lambda n: res[n]["design"]["x5_f1.0"]["mdd"])

    base = list(res)
    ranked = sorted([n for n in base if res[n]["design"]["n"] >= 30],
                    key=lambda n: res[n]["design"]["x5_f1.0"]["ret"], reverse=True)  # fmt: skip
    top2 = ranked[:2]
    print("top2", top2, flush=True)
    for n in top2:
        bx, G, _ = n.split("|")
        lst = boxes[bx]
        nm = f"{bx}|{G}|X5h4"
        spec[nm] = ([(0, lst, G)], {"X": "X5", "hold": 240})
        res[nm], keeps[nm] = evaluate(*spec[nm])
        show(nm, res[nm])
        # 이탈 확률 표: 설계 구간 X5 거래로
        dtr = keeps[n]["design"][0]
        cnt: dict = {}
        for x in dtr:
            s = by_key[(x["kind"], x["st"])]
            R = abs(x["ep"] - (s["L"] if x["d"] == 1 else s["H"]))
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
        nm = f"{bx}|{G}|X4"
        spec[nm] = ([(0, lst, G)], {"X": "X4", "bail": bail})
        res[nm], keeps[nm] = evaluate(*spec[nm])
        show(nm, res[nm])
    kinds, seen = [], set()
    for n in ranked:
        bx = n.split("|")[0]
        if bx not in seen:
            seen.add(bx)
            kinds.append(bx)
        if len(kinds) == 3:
            break
    G0 = ranked[0].split("|")[1]
    nm = "U3|" + G0 + "|X5"
    spec[nm] = ([(i, boxes[k], G0) for i, k in enumerate(kinds)], {"X": "X5", "hold": 480})
    res[nm], keeps[nm] = evaluate(*spec[nm])
    show(nm + str(kinds), res[nm])
    assert len(res) == N_BONF, len(res)

    final = pick(list(res))
    print("final", final, flush=True)
    lists_f, cfg_f = spec[final]
    fin, kfin = res[final], keeps[final]

    # 무작위 (변동성 상태 맞춤)
    pool_all = [s for _, lst, _ in lists_f for s in lst]
    rnd = {}
    for w in ("pool",) + MAIN:
        tr, f = kfin[w]
        pw = [s for s in pool_all if W[w][0] <= s["st"] < W[w][1]]
        rnd[w] = sf_random(m, tr, cfg_f, fund_fn, pw, by_key, 200 if w == "pool" else 100, 7, f)
        print("random", w, rnd[w], round(time.time() - t0), flush=True)

    # 레드팀
    red = {}
    for tag, kw in (("lat1", {"lat": 1}), ("lat5", {"lat": 5}), ("slip2bp", {"slip": 0.0002}),
                    ("fee15", {"fee_mult": 1.5})):  # fmt: skip
        r, _ = evaluate(lists_f, cfg_f, windows=MAIN, **kw)
        red[tag] = {w: r[w]["x5_f1.0"] for w in MAIN}
    a_ms, z_ms = W["fresh"][0], W["year"][1]
    tr = sf_run(m, b, gates, lists_f, cfg_f, fund_fn, a_ms, z_ms)
    f = U.sizes(tr, "S1", vol_fn, vt)
    yrs: dict = {}
    for i, x in enumerate(tr):
        yrs.setdefault(datetime.fromtimestamp(x["t_in"] / 1000, timezone.utc).year, []).append(i)
    red["by_year"] = {y: {"n": len(ix), **U.path([tr[i] for i in ix], 1.0, f[ix])} for y, ix in sorted(yrs.items())}
    rolls, s0 = [], a_ms
    while s0 + 90 * DAY <= z_ms:
        ix = [i for i, x in enumerate(tr) if s0 <= x["t_in"] < s0 + 90 * DAY]
        rolls.append(U.path([tr[i] for i in ix], 1.0, f[ix])["ret"] if ix else 0.0)
        s0 += 30 * DAY
    rolls = np.array(rolls)
    red["rolling90"] = {"n": len(rolls), "pos_share": float((rolls > 0).mean()), "median": float(np.median(rolls)),
                        "worst": float(rolls.min())}  # fmt: skip
    print("red", red, flush=True)

    beat = sum(fin[w]["x5_f1.0"]["ret"] > p2[w]["x5_f1.0"]["ret"] and fin[w]["x5_f1.0"]["mdd"] <= p2[w]["x5_f1.0"]["mdd"]
               for w in ("design", "val", "year"))  # fmt: skip
    gate = {"pos_all_f1": all(fin[w]["x5_f1.0"]["ret"] > 0 for w in MAIN),
            "liq0": all(fin[w]["x5_f1.0"]["liq"] == 0 for w in MAIN),
            "random_bonf": rnd["pool"]["p"] * N_BONF < 0.05,
            "beats_p2_2of3": beat >= 2}  # fmt: skip

    def slim(keep):
        return {w: [{k: v for k, v in x.items() if not k.startswith("_")} for x in keep[w][0]]
                for w in ("d30", "m6", "year", "fresh", "design", "val")}  # fmt: skip

    (out / "trades.json").write_text(json.dumps({"final": slim(kfin), "p2": slim(p2_keep)}, default=float))
    boxes_d30 = [{k: s[k] for k in ("st", "a", "z", "H", "L", "kind")} for _, lst, _ in lists_f for s in lst
                 if W["d30"][0] - 2 * DAY <= s["st"]]  # fmt: skip
    (out / "boxes_d30.json").write_text(json.dumps(boxes_d30, default=float))
    (out / "result.json").write_text(json.dumps(
        {"n_tests": N_BONF, "n_before": N_BEFORE, "final": final, "top2": top2, "union_kinds": kinds,
         "setups": {k: len(v) for k, v in boxes.items()}, "res": res, "p2": p2, "random": rnd, "redteam": red,
         "gate": gate, "pass": all(gate.values()), "windows": W, "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    print("gate", gate, "final", final, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
