"""125번: 패턴·거래량·지표 신호 전수 탐색, 발견(2020~23)에서 BH-FDR로 고르고 확인(2024~) 한 번.

등록: 2026-11-07-btc-signal-scan-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_signal_scan.py --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import btc_book2 as K
import btc_fut_daily_regime as R
import btc_swing_patterns as S
from btc_fractal import cci, rsi, scr3, stoch
from btc_scr_gate15 import load_funding

M5 = 300_000
SPLIT = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
START = datetime(2020, 3, 1, tzinfo=timezone.utc).timestamp() * 1000
FRAMES = {"15m": 3, "1h": 12, "4h": 48}
SIZES = (0.01, 0.02, 0.04)
GATES = ("G0", "GR", "GV")
EXITS = {"X1": (288, 0.02), "X2": (16, None), "X3": (96, None)}
STOP = 0.01
COST = 0.0005
ZIG = ("DB", "DBP", "TB", "IHS", "HARM", "ABCD", "3DRV", "GP", "ATRI", "SFP", "CHOCH", "FLAG", "QM",
       "VBRK", "LVRT", "OBVD", "RSID", "CCID", "STOD", "WSMA")
FREE = ("VPA", "CLMX", "TKC", "TKM", "SCR3")


def load(path: Path) -> dict:
    cols = {k: [] for k in ("t", "o", "h", "l", "c", "v", "tb")}
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            cols["t"].append(int(row[0]))
            for k, j in (("o", 1), ("h", 2), ("l", 3), ("c", 4), ("v", 5)):
                cols[k].append(float(row[j]))
            tb = row[9]
            cols["tb"].append(float(tb) if tb not in ("nan", "") else np.nan)
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def agg(d5: dict, factor: int, off_min: int) -> dict:
    span = factor * M5
    key = ((d5["t"] - off_min * 60_000) // span).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    st, en = edges[:-1], edges[1:]
    full = (en - st) == factor
    hh = np.maximum.reduceat(d5["h"], st)
    ll = np.minimum.reduceat(d5["l"], st)
    vv = np.add.reduceat(d5["v"], st)
    tb = np.add.reduceat(np.nan_to_num(d5["tb"], nan=np.nan), st)
    st2, en2 = st[full], en[full]
    return {"t": d5["t"][st2], "o": d5["o"][st2], "h": hh[full], "l": ll[full], "c": d5["c"][en2 - 1],
            "v": vv[full], "tb": tb[full], "span": span}


def zscore_prev(x: np.ndarray, n: int) -> np.ndarray:
    """직전 n봉(현재 제외) 평균·표준편차로 본 z."""
    cs, cs2 = np.cumsum(np.r_[0.0, x]), np.cumsum(np.r_[0.0, x * x])
    out = np.full(len(x), np.nan)
    i = np.arange(n, len(x))
    m = (cs[i] - cs[i - n]) / n
    var = (cs2[i] - cs2[i - n]) / n - m * m
    sd = np.sqrt(np.maximum(var, 1e-18))
    out[i] = (x[i] - m) / sd
    return out


def last_piv(piv: list, n: int, kind: int) -> tuple:
    """봉마다 확정된 마지막·그 앞 kind 피벗(가격, 극점 봉, 피벗 번호). 없으면 nan/−1."""
    p1, p2 = np.full(n, np.nan), np.full(n, np.nan)
    e1, k1 = np.full(n, -1), np.full(n, -1)
    sel = [(cf, pr, ex, k) for k, (ex, pr, kd, cf) in enumerate(piv) if kd == kind]
    prev = None
    for j, (cf, pr, ex, k) in enumerate(sel):
        end = sel[j + 1][0] if j + 1 < len(sel) else n
        p1[cf:end], e1[cf:end], k1[cf:end] = pr, ex, k
        if prev is not None:
            p2[cf:end] = prev
        prev = pr
    return p1, p2, e1, k1


def first_per(cond: np.ndarray, key: np.ndarray) -> np.ndarray:
    """같은 key(피벗 번호)마다 조건이 처음 성립한 봉만."""
    idx = np.flatnonzero(cond & (key >= 0))
    if not len(idx):
        return idx
    keep = np.r_[True, key[idx][1:] != key[idx][:-1]]
    return idx[keep]


def zig_signals(f: dict, th: float, side: int) -> dict:
    """롱 기준 신호(봉 번호 배열). 숏은 가격을 뒤집은 자료로 부른다."""
    o, h, l, c, v = f["o"], f["h"], f["l"], f["c"], f["v"]
    n = len(c)
    base = S.zigzag(f["h_raw"], f["l_raw"], th)
    piv = base if side > 0 else [(i, -p, -k, cf) for i, p, k, cf in base]
    out = {}
    for kind in ("DB", "DBP", "TB", "IHS"):
        out[kind] = np.array(sorted({s[0] for s in S.find(piv, c, kind)}), int)
    dd = {"c_side": c}
    harm, other = set(), {"ABCD": set(), "3DRV": set(), "GP": set(), "ATRI": set()}
    for name, start, *_ in K.scan(dd, piv, 1):
        if name in other:
            other[name].add(start)
        else:
            harm.add(start)
    out["HARM"] = np.array(sorted(harm), int)
    for k2, s2 in other.items():
        out[k2] = np.array(sorted(s2), int)
    lo1, lo2, loe, lok = last_piv(piv, n, -1)
    hi1, hi2, hie, hik = last_piv(piv, n, 1)
    with np.errstate(invalid="ignore"):
        out["SFP"] = first_per((l < lo1) & (c > lo1), lok)
        choch = (hi1 < hi2) & (lo1 < lo2) & (c > hi1)
        out["CHOCH"] = first_per(choch, hik)
        vz = f["vz"]
        out["VBRK"] = first_per((c > hi1) & (vz >= 2), hik)
        rng = h - l
        wick = np.minimum(o, c) - l
        out["LVRT"] = first_per((l <= lo1 * 1.002) & (c > lo1) & (wick >= 0.5 * rng) & (vz <= -0.5), lok)
    # 깃발: 마지막 확정 피벗이 고점 H0, 그 앞 저점 L0, 다리 ≥ 2·th, 이후 저가가 다리의 50% 위, 종가 > H0
    flag = np.zeros(n, bool)
    qm = np.zeros(n, bool)
    qkey = np.full(n, -1)
    for k in range(1, len(piv)):
        if piv[k][2] == 1 and piv[k - 1][2] == -1:
            L0, H0 = piv[k - 1], piv[k]
            leg = H0[1] - L0[1]
            if leg <= 0 or leg / abs(L0[1]) < 2 * th:
                continue
            end = piv[k + 1][3] if k + 1 < len(piv) else n
            floor = H0[1] - 0.5 * leg
            for i in range(H0[3] + 1, min(end + 1, n)):
                if l[i] < floor:
                    break
                if c[i] > H0[1]:
                    flag[i] = True
                    break
        # 콰지모도: L1 → H1 → LL → HH 확정, 이후 저가가 L1에 닿고 종가 > LL
        if k >= 3 and piv[k][2] == 1 and piv[k - 3][2] == -1:
            L1, H1, LL, HH = piv[k - 3:k + 1]
            if LL[1] < L1[1] and HH[1] > H1[1]:
                end = piv[k + 1][3] if k + 1 < len(piv) else n
                for i in range(HH[3] + 1, min(end + 1, n)):
                    if c[i] < LL[1]:
                        break
                    if l[i] <= L1[1]:
                        qm[i] = True
                        qkey[i] = k
                        break
    out["FLAG"] = np.flatnonzero(flag)
    out["QM"] = np.flatnonzero(qm)
    # 다이버전스: 새로 확정된 저점이 직전 저점보다 낮고 지표는 높음, 저점 봉 지표가 깊은 과매도
    inds = {"OBVD": (f["obv"], None), "RSID": (f["rsi"], 30), "CCID": (f["cci"], -100), "STOD": (f["sto"], 20)}
    lows = [p for p in piv if p[2] == -1]
    for name, (ind, deep) in inds.items():
        ev = []
        for a, b in zip(lows[:-1], lows[1:]):
            if b[1] < a[1] and ind[b[0]] > ind[a[0]] and (deep is None or ind[b[0]] < deep):
                ev.append(b[3])
        out[name] = np.array(ev, int)
    # 파동 길이 SMA: 마지막으로 끝난 다리의 봉 수
    legn = np.full(n, 0)
    for k in range(1, len(piv)):
        end = piv[k + 1][3] if k + 1 < len(piv) else n
        legn[piv[k][3]:end] = max(piv[k][0] - piv[k - 1][0], 0)
    cs = np.cumsum(np.r_[0.0, c])
    i = np.arange(n)
    ok = (legn >= 5) & (i - legn >= 1)
    sma = np.full(n, np.nan)
    smp = np.full(n, np.nan)
    sma[ok] = (cs[i[ok] + 1] - cs[i[ok] + 1 - legn[ok]]) / legn[ok]
    smp[ok] = (cs[i[ok]] - cs[i[ok] - legn[ok]]) / legn[ok]
    with np.errstate(invalid="ignore"):
        touch = ok & (sma > smp) & (l <= sma) & (c > sma)
    out["WSMA"] = np.flatnonzero(touch & ~np.r_[False, touch[:-1]])
    return out


def free_signals(f: dict) -> dict:
    o, h, l, c, vz = f["o"], f["h"], f["l"], f["c"], f["vz"]
    n = len(c)
    rng = h - l
    body = np.abs(c - o)
    lo20 = np.full(n, np.inf)
    for k in range(20):
        lo20 = np.minimum(lo20, np.r_[np.full(k, np.inf), l[:n - k]])
    with np.errstate(invalid="ignore"):
        vpa = (body <= 0.3 * rng) & (vz >= 2) & (l <= lo20)
        rq = np.full(n, np.nan)
        # 범위가 직전 100봉 상위 5%: 직전 100봉 z ≥ 1.645로 근사
        rq = zscore_prev(rng, 100)
        clm_prev = (c < o) & (vz >= 3) & (rq >= 1.645)
        mid = (h + l) / 2
        clmx = np.r_[False, clm_prev[:-1]] & (c > np.r_[np.nan, mid[:-1]])
        tz = f["tkz"]
        tkc, tkm = tz <= -1.645, tz >= 1.645
    on = lambda m: np.flatnonzero(m & ~np.r_[False, m[:-1]])  # noqa: E731
    return {"VPA": on(vpa), "CLMX": on(clmx), "TKC": on(tkc), "TKM": on(tkm)}


def prep(f: dict, side: int) -> dict:
    """롱 기준 자료. 숏은 가격을 뒤집고 시장가 매수 비율을 1−비율로."""
    g = {"o": f["o"] * side, "c": f["c"] * side, "v": f["v"]}
    g["h"], g["l"] = (f["h"], f["l"]) if side > 0 else (-f["l"], -f["h"])
    g["h_raw"], g["l_raw"] = f["h"], f["l"]
    g["vz"] = f["vz"]
    sgn = np.sign(np.r_[0.0, np.diff(f["c"])])
    obv = np.cumsum(sgn * f["v"])
    g["obv"] = obv * side
    if side > 0:
        g["rsi"], g["cci"], g["sto"] = f["rsi"], f["cci"], f["sto"]
    else:
        g["rsi"], g["cci"], g["sto"] = 100 - f["rsi"], -f["cci"], 100 - f["sto"]
    g["tkz"] = f["tkz"] * side
    return g


def frame_data(d5: dict, factor: int, off: int) -> dict:
    f = agg(d5, factor, off)
    f["vz"] = zscore_prev(f["v"], 20)
    with np.errstate(invalid="ignore", divide="ignore"):
        tr20 = np.convolve(np.nan_to_num(f["tb"]), np.ones(20), "full")[:len(f["v"])]
        v20 = np.convolve(f["v"], np.ones(20), "full")[:len(f["v"])]
        ratio = tr20 / v20
    ratio[:20] = np.nan
    ratio[np.isnan(f["tb"])] = np.nan
    f["tkz"] = zscore_prev(np.nan_to_num(ratio, nan=0.5), 2000)
    f["tkz"][np.isnan(ratio)] = np.nan
    f["rsi"], f["cci"], f["sto"] = rsi(f["c"]), cci(f["h"], f["l"], f["c"]), stoch(f["h"], f["l"], f["c"])
    return f


def exits(b: dict, ent: np.ndarray, side: int, fund) -> dict:
    """15분봉 진입 시가 배열에 대한 청산 (총손익, 청산 봉). 같은 봉에서 손절 먼저."""
    o, h, l, t = b["o"], b["h"], b["l"], b["t"]
    n = len(o)
    ft, fr = fund
    fcs = np.r_[0.0, np.cumsum(fr)]
    res = {}
    for xn, (w, tgt) in EXITS.items():
        g = np.full(len(ent), np.nan)
        xi = np.full(len(ent), -1)
        for a in range(0, len(ent), 4000):
            e = ent[a:a + 4000]
            ok = e + w < n
            idx = np.minimum(e[:, None] + np.arange(w)[None, :], n - 1)
            px = o[e]
            sp = px * (1 - STOP * side)
            if side > 0:
                hit_s = l[idx] <= sp[:, None]
                hit_t = h[idx] >= (px * (1 + tgt))[:, None] if tgt else np.zeros_like(hit_s)
            else:
                hit_s = h[idx] >= sp[:, None]
                hit_t = l[idx] <= (px * (1 - tgt))[:, None] if tgt else np.zeros_like(hit_s)
            fs = np.where(hit_s.any(1), hit_s.argmax(1), w)
            ftg = np.where(hit_t.any(1), hit_t.argmax(1), w)
            gg = np.empty(len(e))
            xx = np.empty(len(e), int)
            st = fs <= ftg
            st &= fs < w
            tg = (~st) & (ftg < w)
            tm = ~(st | tg)
            j = e + fs
            oj = o[np.minimum(j, n - 1)]
            fill = np.where(side > 0, np.minimum(oj, sp), np.maximum(oj, sp))
            gg[st] = ((fill / px - 1) * side)[st]
            xx[st] = j[st]
            if tgt:
                jt = e + ftg
                ot = o[np.minimum(jt, n - 1)]
                tp = px * (1 + tgt * side)
                tfill = np.where(side > 0, np.maximum(ot, tp), np.minimum(ot, tp))
                gg[tg] = ((tfill / px - 1) * side)[tg]
                xx[tg] = jt[tg]
            je = np.minimum(e + w, n - 1)
            gg[tm] = ((o[je] / px - 1) * side)[tm]
            xx[tm] = je[tm]
            a0 = np.searchsorted(ft, t[e], side="right")
            a1 = np.searchsorted(ft, t[np.minimum(xx, n - 1)], side="right")
            gg = gg - side * (fcs[a1] - fcs[a0])
            gg[~ok] = np.nan
            g[a:a + 4000] = gg
            xi[a:a + 4000] = xx
        res[xn] = (g, xi)
    return res


def walk(ev: list) -> list:
    """(진입, 방향, 손익, 청산) 시간 순, 보유 중 신호는 버린다."""
    out, busy = [], -1
    for e, s, g, x in ev:
        if e <= busy or not np.isfinite(g):
            continue
        out.append((e, s, g, x))
        busy = x
    return out


def stats(tr: list, cost: float = COST) -> dict:
    if not tr:
        return {"n": 0}
    net = np.array([z[2] for z in tr]) - 2 * cost
    sd = net.std(ddof=1) if len(net) > 1 else np.nan
    tv = net.mean() / (sd / math.sqrt(len(net))) if sd and sd > 0 else 0.0
    lg = [z for z in tr if z[1] > 0]
    sh = [z for z in tr if z[1] < 0]
    return {"n": len(net), "ev": round(float(net.mean() * 100), 4), "win": round(float((net > 0).mean() * 100), 1),
            "t": round(float(tv), 2),
            "long": round(float((np.array([z[2] for z in lg]) - 2 * cost).mean() * 100), 4) if lg else None,
            "short": round(float((np.array([z[2] for z in sh]) - 2 * cost).mean() * 100), 4) if sh else None,
            "nl": len(lg), "ns": len(sh)}


def build(d5: dict, fund, off: int, only: set | None = None) -> tuple:
    """칸 → (진입 15분봉 번호, 방향) 사건 목록과 15분봉 자료."""
    b = agg(d5, 3, off)
    t15 = b["t"]
    b5 = {"t": d5["t"], "o": d5["o"], "h": d5["h"], "l": d5["l"], "c": d5["c"], "v": d5["v"]}
    reg, _ = R.regime(b5, {"t": t15, "o": b["o"]}, 0)
    cells = {}
    for fn, fac in FRAMES.items():
        f = frame_data(d5, fac, off)
        close = f["t"] + f["span"]
        ent_of = np.searchsorted(t15, close)  # 마감 시각에 시작하는 15분봉
        valid_e = (ent_of < len(t15)) & (t15[np.minimum(ent_of, len(t15) - 1)] == close)
        for side in (1, -1):
            g = prep(f, side)
            sig_sets = {}
            for th in SIZES:
                key_pref = [(nm, fn, f"{th:g}") for nm in ZIG]
                if only is not None and not any((nm, fn, sz) in only for nm, _, sz in key_pref):
                    continue
                for nm, arr in zig_signals(g, th, side).items():
                    sig_sets[(nm, fn, f"{th:g}")] = arr
            if only is None or any(k[1] == fn and k[2] == "-" for k in only):
                fs = free_signals(g)
                fs["SCR3"] = np.flatnonzero(scr3(f)[0 if side > 0 else 1])
                for nm, arr in fs.items():
                    sig_sets[(nm, fn, "-")] = arr
            for key, arr in sig_sets.items():
                if only is not None and key not in only:
                    continue
                arr = arr[(arr >= 0) & (arr < len(close))]
                arr = arr[valid_e[arr]]
                e = ent_of[arr]
                vz = f["vz"][arr]
                for gate in GATES:
                    if gate == "G0":
                        m = np.ones(len(e), bool)
                    elif gate == "GR":
                        m = reg[np.maximum(e - 1, 0)] == side
                    else:
                        m = vz >= 1.5
                    cells.setdefault(key + (gate,), []).append((e[m], side))
    return b, cells


def run_cells(b, cells, fund, cost=COST, split=True):
    sides = {1: set(), -1: set()}
    for lst in cells.values():
        for e, s in lst:
            sides[s].update(e.tolist())
    table = {}
    for s in (1, -1):
        ent = np.array(sorted(sides[s]), int)
        ent = ent[ent < len(b["o"]) - 1]
        ex = exits(b, ent, s, fund)
        table[s] = (ent, ex)
    out = {}
    t15 = b["t"]
    for key, lst in cells.items():
        for xn in EXITS:
            ev = []
            for e, s in lst:
                ent, ex = table[s]
                pos = np.searchsorted(ent, e)
                ok = (pos < len(ent)) & (ent[np.minimum(pos, len(ent) - 1)] == e)
                g, xi = ex[xn]
                for ee, pp in zip(e[ok], pos[ok]):
                    ev.append((int(ee), s, float(g[pp]), int(xi[pp])))
            ev.sort()
            ev = [z for z in ev if t15[z[0]] >= START]
            disc = walk([z for z in ev if t15[z[0]] < SPLIT])
            hold = walk([z for z in ev if t15[z[0]] >= SPLIT])
            out[key + (xn,)] = (disc, hold)
    return out


def circ_p(b, cells_key_events, fund, xn, actual, reps=200, seed=7):
    """발견 구간 진입 시각 원형 이동."""
    rng = np.random.default_rng(seed)
    t15 = b["t"]
    lo, hi = np.searchsorted(t15, START), np.searchsorted(t15, SPLIT)
    span = hi - lo
    null = []
    for _ in range(reps):
        k = int(rng.integers(96 * 30, span - 96 * 30))
        sh = []
        for e, s in cells_key_events:
            e = e[(e >= lo) & (e < hi)]
            sh.append((lo + (e - lo + k) % span, s))
        r = run_cells(b, {"x": sh}, fund)
        null.append(stats(r[("x", xn)][0]).get("ev", 0.0))
    null = np.array(null)
    return round(float((null >= actual).mean()), 4), round(float(null.mean()), 4)


def norm_sf(t: float) -> float:
    return 0.5 * math.erfc(t / math.sqrt(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = Path(args.data)
    d5 = load(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    b, cells = build(d5, fund, 0)
    res = run_cells(b, cells, fund)
    print("cells", len(res), round(time.time() - t0), "s", flush=True)
    rows = []
    for key, (disc, hold) in res.items():
        sd = stats(disc)
        rows.append({"key": list(key), "disc": sd, "p": norm_sf(sd["t"]) if sd["n"] >= 60 else None})
    tested = [r for r in rows if r["p"] is not None]
    m = len(tested)
    tested.sort(key=lambda r: r["p"])
    surv = []
    for k, r in enumerate(tested, 1):
        if r["p"] <= 0.10 * k / m:
            surv = tested[:k]
    surv_keys = {tuple(r["key"]) for r in surv}
    top10 = sorted(tested, key=lambda r: -r["disc"]["t"])[:10]
    look = {tuple(r["key"]) for r in surv} | {tuple(r["key"]) for r in top10}
    # 확인 구간: 살아남은 칸과 참고 상위 10칸만 본다
    only = {k[:3] for k in look}
    b5, cells5 = build(d5, fund, 5, only)
    res5 = run_cells(b5, {k: v for k, v in cells5.items()}, fund)
    rep = []
    for key in sorted(look, key=lambda k: tuple(map(str, k))):
        disc, hold = res[key]
        cellkey = key[:4]
        h0, h10 = stats(hold), stats(hold, 0.001)
        h5 = stats(res5[key][1]) if key in res5 else {"n": 0}
        sd = stats(disc)
        row = {"key": list(key), "fdr": key in surv_keys, "disc": sd, "hold": h0, "hold_c10": h10, "hold_off5": h5}
        if key in surv_keys:
            row["circ_p"], row["circ_null"] = circ_p(b, cells[cellkey], fund, key[4], sd["ev"])
        ok = (h0["n"] >= 30 and h0["ev"] > 0 and (h0["long"] or 0) >= 0 and (h0["short"] or 0) >= 0
              and h10.get("ev", -1) > 0 and h5.get("ev", -1) > 0)
        row["pass"] = bool(ok and key in surv_keys)
        row["trades_hold"] = [[z[0], z[1], round(z[2], 5), z[3]] for z in hold]
        rep.append(row)
    summary = {"cells_total": len(res), "cells_tested": m, "fdr_survivors": len(surv),
               "runtime_s": round(time.time() - t0), "rows": rep,
               "disc_t_hist": np.histogram([r["disc"]["t"] for r in tested], bins=[-99, -3, -2, -1, 0, 1, 2, 3, 99])[0].tolist()}
    (out / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("done", m, len(surv), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
