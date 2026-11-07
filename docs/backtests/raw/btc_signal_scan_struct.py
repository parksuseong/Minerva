"""126번: 125번 신호 전수 탐색을 구조 손절·책 목표·R 평가로 다시. 하모닉 등 4종은 D 지정가.

등록: 2026-11-07-btc-signal-scan-structure-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_signal_scan_struct.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import btc_book2 as K
import btc_fut_daily_regime as R
import btc_signal_scan as SS
import btc_swing_patterns as S
import numpy as np
from btc_fractal import scr3
from btc_scr_gate15 import load_funding

COST = 0.0005
BUF = 0.001
SD_MIN, SD_MAX = 0.003, 0.08
DAY15 = 96
MAXH = 30 * DAY15
YS = ("Y1", "Y2", "Y3")
LIMIT = ("HARM", "ABCD", "3DRV", "GP")


def piv_tables(piv: list, n: int) -> dict:
    """롱 기준 피벗에서: 확정 봉, 극점 봉, 가격, 종류."""
    cf = np.array([p[3] for p in piv], int)
    ex = np.array([p[0] for p in piv], int)
    pr = np.array([p[1] for p in piv], float)
    kd = np.array([p[2] for p in piv], int)
    lo1, _, _, lok = SS.last_piv(piv, n, -1)
    return {"cf": cf, "ex": ex, "pr": pr, "kd": kd, "lo1": lo1, "lok": lok}


def formation(pt: dict, sig: int) -> int:
    cnt = int(np.searchsorted(pt["cf"], sig, side="right"))
    k = cnt - 3
    if k < 0:
        return 4
    return max(sig - int(pt["ex"][k]), 4)


def struct_events(g: dict, th: float, side: int, piv: list, pt: dict) -> list:
    """(이름, 신호 봉, 방식, 손절 수준, 목표, 대기 시작, 대기 길이, D, 취소 수준) 롱 기준 가격.

    방식 0 = 다음 시가 시장가, 1 = D 지정가.
    """
    l, c = g["l"], g["c"]  # noqa: E741
    n = len(c)
    nan = float("nan")
    ev = []
    for kind in ("DB", "DBP", "TB", "IHS"):
        for sig, stop, tgt, nm in S.find(piv, c, kind):
            ev.append((nm, int(sig), 0, float(stop), float(tgt), 0, 0, nan, nan))
    for name, start, lim, dp, stop, tgt, extra, kind in K.scan({"c_side": c}, piv, 1):
        if kind == "limit":
            nm = name if name in LIMIT else "HARM"
            ev.append((nm, int(start) - 1, 1, float(stop), float(tgt), int(start), int(lim), float(dp),
                       float(extra)))
        else:
            ev.append((name, int(start), 0, float(stop), float(tgt), 0, 0, nan, nan))
    idx = SS.zig_signals(g, th, side)
    lo1 = pt["lo1"]
    for nm in ("SFP", "CHOCH", "VBRK", "LVRT", "WSMA"):
        for i in idx[nm]:
            if nm == "SFP":
                lv = l[i]
            elif nm == "LVRT":
                lv = min(l[i], lo1[i])
            else:
                lv = lo1[i]
            if np.isfinite(lv):
                ev.append((nm, int(i), 0, float(lv), nan, 0, 0, nan, nan))
    # 깃발·콰지모도: 125번과 같은 탐지, 손절·목표를 함께 기록
    for k in range(1, len(piv)):
        if piv[k][2] == 1 and piv[k - 1][2] == -1:
            L0, H0 = piv[k - 1], piv[k]
            leg = H0[1] - L0[1]
            if leg > 0 and leg / abs(L0[1]) >= 2 * th:
                end = piv[k + 1][3] if k + 1 < len(piv) else n
                floor = H0[1] - 0.5 * leg
                lowest = np.inf
                for i in range(H0[3] + 1, min(end + 1, n)):
                    if l[i] < floor:
                        break
                    lowest = min(lowest, l[i])
                    if c[i] > H0[1]:
                        ev.append(("FLAG", i, 0, float(lowest), float(H0[1] + leg), 0, 0, nan, nan))
                        break
        if k >= 3 and piv[k][2] == 1 and piv[k - 3][2] == -1:
            L1, H1, LL, HH = piv[k - 3:k + 1]
            if LL[1] < L1[1] and HH[1] > H1[1]:
                end = piv[k + 1][3] if k + 1 < len(piv) else n
                for i in range(HH[3] + 1, min(end + 1, n)):
                    if c[i] < LL[1]:
                        break
                    if l[i] <= L1[1]:
                        ev.append(("QM", i, 0, float(LL[1]), float(HH[1]), 0, 0, nan, nan))
                        break
    inds = {"OBVD": (g["obv"], None), "RSID": (g["rsi"], 30), "CCID": (g["cci"], -100), "STOD": (g["sto"], 20)}
    lows = [p for p in piv if p[2] == -1]
    for name, (ind, deep) in inds.items():
        for a, b in zip(lows[:-1], lows[1:]):
            if b[1] < a[1] and ind[b[0]] > ind[a[0]] and (deep is None or ind[b[0]] < deep):
                ev.append((name, int(b[3]), 0, float(b[1]), nan, 0, 0, nan, nan))
    return ev


def free_events(g: dict, f: dict, side: int, pt: dict) -> list:
    nan = float("nan")
    fs = SS.free_signals(g)
    fs["SCR3"] = np.flatnonzero(scr3(f)[0 if side > 0 else 1])
    lo1 = pt["lo1"]
    out = []
    for nm, arr in fs.items():
        for i in arr:
            if np.isfinite(lo1[i]):
                out.append((nm, int(i), 0, float(lo1[i]), nan, 0, 0, nan, nan))
    return out


class Sim:
    """롱 기준 가격(숏은 부호를 뒤집은 15분봉)으로 진입·청산."""

    def __init__(self, b: dict, fund):
        self.t = b["t"]
        self.n = len(b["t"])
        self.L = {1: (b["o"], b["h"], b["l"]), -1: (-b["o"], -b["l"], -b["h"])}
        ft, fr = fund
        self.ft, self.fcs = ft, np.r_[0.0, np.cumsum(fr)]

    def funding(self, side: int, e: int, x: int) -> float:
        a0 = np.searchsorted(self.ft, self.t[e], side="right")
        a1 = np.searchsorted(self.ft, self.t[min(x, self.n - 1)], side="right")
        return side * (self.fcs[a1] - self.fcs[a0])

    def fill_limit(self, side, ws, we, dp, inval):
        o, h, l = self.L[side]  # noqa: E741
        we = min(we, self.n - 1)
        if we <= ws:
            return None
        hh, ll = h[ws:we], l[ws:we]
        fill = ll <= dp
        canc = (hh > inval) & ~fill
        jf = int(fill.argmax()) if fill.any() else None
        jc = int(canc.argmax()) if canc.any() else None
        if jf is None or (jc is not None and jc < jf):
            return None
        i = ws + jf
        return i, min(o[i], dp)

    def first(self, mask: np.ndarray) -> int:
        return int(mask.argmax()) if mask.any() else -1

    def run(self, side, e, px, stop, tgt1, hold, trail, limit):
        """Y1·Y2·Y3의 (총손익−펀딩, 청산 봉). 거래 안 하면 None."""
        o, h, l = self.L[side]  # noqa: E741
        n = self.n
        ap = abs(px)
        sd = (px - stop) / ap
        end = min(e + hold, n - 1)
        if end <= e + 1:
            return None
        hs = l[e:end] <= stop
        js = self.first(hs)
        res = {}
        risk = px - stop

        def stop_fill(j):
            return stop if j == 0 else min(o[e + j], stop)

        def bracket(T):
            ht = h[e:end] >= T
            if limit:
                ht[0] = False
            jt = self.first(ht)
            if js >= 0 and (jt < 0 or js <= jt):
                x = e + js
                xp = stop_fill(js)
            elif jt >= 0:
                x = e + jt
                xp = T if jt == 0 else max(o[x], T)
            else:
                x, xp = end, o[end]
            return (xp - px) / ap, x

        if np.isfinite(tgt1) and tgt1 > px:
            res["Y1"] = bracket(tgt1)
        elif not np.isfinite(tgt1):
            res["Y1"] = bracket(px + 2 * risk)
        res["Y2"] = bracket(px + risk)
        # Y3: +2R 전까지 처음 손절, 닿으면 진입 + 1R, 이후 새 스윙마다 끌어올림
        h2 = h[e:end] >= px + 2 * risk
        if limit:
            h2[0] = False
        j2 = self.first(h2)
        if js >= 0 and (j2 < 0 or js <= j2):
            res["Y3"] = ((stop_fill(js) - px) / ap, e + js)
        elif j2 < 0:
            res["Y3"] = ((o[end] - px) / ap, end)
        else:
            cs = px + risk
            a = e + j2 + 1
            cf15, lv = trail
            k0 = int(np.searchsorted(cf15, a, side="left"))
            done = None
            pts = [(int(cf15[k]), float(lv[k])) for k in range(k0, len(cf15)) if cf15[k] < end]
            pts.append((end, cs))
            for nb, nl in pts:
                if nb > a:
                    seg = l[a:nb] <= cs
                    j = self.first(seg)
                    if j >= 0:
                        x = a + j
                        done = ((min(o[x], cs) - px) / ap, x)
                        break
                    a = nb
                cs = max(cs, nl)
            res["Y3"] = done if done else ((o[end] - px) / ap, end)
        out = {}
        for y, (g, x) in res.items():
            out[y] = (g - self.funding(side, e, x), x)
        return sd, out


def build(d5: dict, fund, off: int, only: set | None = None) -> tuple:
    b = SS.agg(d5, 3, off)
    t15 = b["t"]
    b5 = {"t": d5["t"], "o": d5["o"], "h": d5["h"], "l": d5["l"], "c": d5["c"], "v": d5["v"]}
    reg, _ = R.regime(b5, {"t": t15, "o": b["o"]}, 0)
    sim = Sim(b, fund)
    cells: dict = {}
    skips = {"sd_small": 0, "sd_large": 0, "beyond_stop": 0, "no_fill": 0, "y1_target_passed": 0, "events": 0}
    for fn, fac in SS.FRAMES.items():
        f = SS.frame_data(d5, fac, off)
        nf = len(f["c"])
        close = f["t"] + f["span"]
        ent_of = np.searchsorted(t15, close)
        valid_e = (ent_of < len(t15)) & (t15[np.minimum(ent_of, len(t15) - 1)] == close)
        for side in (1, -1):
            g = SS.prep(f, side)
            zz = {}
            for th in SS.SIZES + (0.02,):
                if th in zz:
                    continue
                base = S.zigzag(f["h"], f["l"], th)
                piv = base if side > 0 else [(i, -p, -k, cf) for i, p, k, cf in base]
                pt = piv_tables(piv, nf)
                lowk = pt["kd"] == -1
                cfl = pt["cf"][lowk]
                okc = cfl < nf
                cf15 = np.where(valid_e[np.minimum(cfl, nf - 1)], ent_of[np.minimum(cfl, nf - 1)], 10**9)
                lv = pt["pr"][lowk] - np.abs(pt["pr"][lowk]) * BUF  # 롱 기준 저점 − 0.1%
                zz[th] = (piv, pt, (cf15[okc], lv[okc]))
            groups = []
            for th in SS.SIZES:
                if only is not None and not any(k[0] in SS.ZIG and k[1] == fn and k[2] == f"{th:g}" for k in only):
                    continue
                piv, pt, trail = zz[th]
                groups.append((f"{th:g}", struct_events(g, th, side, piv, pt), pt, trail))
            if only is None or any(k[1] == fn and k[2] == "-" for k in only):
                piv, pt, trail = zz[0.02]
                groups.append(("-", free_events(g, f, side, pt), pt, trail))
            for sz, evs, pt, trail in groups:
                for nm, sig, mode, stop_l, tgt_l, ws_f, lim, dp, inval in evs:
                    key = (nm, fn, sz)
                    if only is not None and key not in only:
                        continue
                    if sig < 0 or sig >= nf or not valid_e[sig]:
                        continue
                    skips["events"] += 1
                    stop = stop_l - abs(stop_l) * BUF  # 롱 기준 가격에서 0.1% 더 아래
                    dec = int(ent_of[sig])
                    if mode == 1:
                        if ws_f >= nf:
                            continue
                        ws = int(np.searchsorted(t15, f["t"][ws_f]))
                        we_f = ws_f + lim
                        we = int(np.searchsorted(t15, f["t"][we_f])) if we_f < nf else len(t15) - 1
                        fl = sim.fill_limit(side, ws, we, dp, inval)
                        if fl is None:
                            skips["no_fill"] += 1
                            continue
                        e, px = fl
                        dec = ws
                    else:
                        e = dec
                        if e >= len(t15) - 2:
                            continue
                        px = sim.L[side][0][e]
                    if px <= stop:
                        skips["beyond_stop"] += 1
                        continue
                    sd = (px - stop) / abs(px)
                    if sd < SD_MIN:
                        skips["sd_small"] += 1
                        continue
                    if sd > SD_MAX:
                        skips["sd_large"] += 1
                        continue
                    hold = min(10 * formation(pt, sig) * fac // 3, MAXH)
                    r = sim.run(side, e, px, stop, tgt_l, hold, trail, mode == 1)
                    if r is None:
                        continue
                    sd, res = r
                    if "Y1" not in res:
                        skips["y1_target_passed"] += 1
                    vz = f["vz"][sig]
                    gr = reg[max(dec - 1, 0)] == side
                    for y, (gg, x) in res.items():
                        rec = (e, side, gg, x, sd, float(side * px), float(side * stop),
                               float(side * tgt_l) if np.isfinite(tgt_l) else None)
                        for gate in SS.GATES:
                            if gate == "GR" and not gr:
                                continue
                            if gate == "GV" and not (vz >= 1.5):
                                continue
                            cells.setdefault(key + (gate, y), []).append(rec)
    return b, cells, skips


def walk(ev: list) -> list:
    out, busy = [], -1
    for z in sorted(ev, key=lambda z: (z[0], z[1])):
        if z[0] <= busy:
            continue
        out.append(z)
        busy = z[3]
    return out


def stats(tr: list, cost: float = COST) -> dict:
    if not tr:
        return {"n": 0}
    g = np.array([z[2] for z in tr])
    sd = np.array([z[4] for z in tr])
    rr = (g - 2 * cost) / sd
    r0 = g / sd
    s = rr.std(ddof=1) if len(rr) > 1 else np.nan
    tv = rr.mean() / (s / math.sqrt(len(rr))) if s and s > 0 else 0.0
    lg = np.array([z[1] > 0 for z in tr])
    return {"n": len(rr), "R": round(float(rr.mean()), 4), "R_pre": round(float(r0.mean()), 4),
            "t": round(float(tv), 2), "win": round(float((rr > 0).mean() * 100), 1),
            "pct": round(float((g - 2 * cost).mean() * 100), 4), "sd_pct": round(float(sd.mean() * 100), 3),
            "R_sd": round(float(s), 3) if np.isfinite(s) else None,
            "long": round(float(rr[lg].mean()), 4) if lg.any() else None,
            "short": round(float(rr[~lg].mean()), 4) if (~lg).any() else None,
            "nl": int(lg.sum()), "ns": int((~lg).sum())}


def equity(tr: list, cost: float = COST) -> dict:
    eq, peak, mdd, big = 1.0, 1.0, 0.0, 0
    for z in tr:
        lev = min(0.01 / z[4], 10.0)
        big += lev > 5
        eq *= 1 + lev * (z[2] - 2 * cost)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    return {"ret_pct": round((eq - 1) * 100, 1), "mdd_pct": round(mdd * 100, 1),
            "lev_gt5_pct": round(100 * big / max(len(tr), 1), 1)}


def split(b: dict, cells: dict) -> dict:
    t15 = b["t"]
    out = {}
    for key, ev in cells.items():
        ev = [z for z in ev if t15[z[0]] >= SS.START]
        out[key] = (walk([z for z in ev if t15[z[0]] < SS.SPLIT]), walk([z for z in ev if t15[z[0]] >= SS.SPLIT]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = Path(args.data)
    d5 = SS.load(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    b, cells, skips = build(d5, fund, 0)
    res = split(b, cells)
    print("cells", len(res), round(time.time() - t0), "s", skips, flush=True)
    rows = []
    for key, (disc, _hold) in res.items():
        sd = stats(disc)
        rows.append({"key": list(key), "disc": sd, "p": SS.norm_sf(sd["t"]) if sd["n"] >= 60 else None})
    tested = [r for r in rows if r["p"] is not None]
    m = len(tested)
    tested.sort(key=lambda r: r["p"])
    surv = []
    for k, r in enumerate(tested, 1):
        if r["p"] <= 0.10 * k / m:
            surv = tested[:k]
    surv_keys = {tuple(r["key"]) for r in surv}
    top10 = sorted(tested, key=lambda r: -r["disc"]["t"])[:10]
    look = surv_keys | {tuple(r["key"]) for r in top10}
    only = {k[:3] for k in look}
    b5, cells5, _ = build(d5, fund, 5, only)
    res5 = split(b5, cells5)
    rep = []
    for key in sorted(look, key=lambda k: tuple(map(str, k))):
        disc, hold = res[key]
        h0, h10 = stats(hold), stats(hold, 0.001)
        h5 = stats(res5[key][1]) if key in res5 else {"n": 0}
        ok = (h0["n"] >= 30 and h0["R"] > 0 and (h0["long"] or 0) >= 0 and (h0["short"] or 0) >= 0
              and h10.get("R", -1) > 0 and h5.get("R", -1) > 0)
        rep.append({"key": list(key), "fdr": key in surv_keys, "disc": stats(disc), "hold": h0, "hold_c10": h10,
                    "hold_off5": h5, "equity_hold": equity(hold), "pass": bool(ok and key in surv_keys),
                    "trades_hold": [[int(z[0]), int(z[1]), round(z[2], 5), int(z[3]), round(z[4], 5), z[5], z[6],
                                     z[7]] for z in hold]})
    # 검정력: 칸마다 걸리려면 필요한 평균 R ≈ 문턱 t × R 표준편차 / √n (중앙값 칸)
    sds = [r["disc"]["R_sd"] for r in tested if r["disc"].get("R_sd")]
    ns = [r["disc"]["n"] for r in tested]
    tcrit = None
    if m:
        from statistics import NormalDist
        tcrit = NormalDist().inv_cdf(1 - 0.10 / m)
    power = {"t_needed_single": round(tcrit, 2) if tcrit else None,
             "median_R_sd": float(np.median(sds)) if sds else None, "median_n": float(np.median(ns)) if ns else None}
    if tcrit and sds:
        power["R_needed_median_cell"] = round(tcrit * float(np.median(sds)) / math.sqrt(float(np.median(ns))), 3)
    hist_t = np.histogram([r["disc"]["t"] for r in tested], bins=[-99, -3, -2, -1, 0, 1, 2, 3, 99])[0].tolist()
    pre = [r["disc"]["R_pre"] for r in tested]
    summary = {"cells_total": len(res), "cells_tested": m, "fdr_survivors": len(surv), "skips": skips,
               "runtime_s": round(time.time() - t0), "power": power, "disc_t_hist": hist_t,
               "disc_pre_fee_R_mean_of_cells": round(float(np.mean(pre)), 4) if pre else None,
               "disc_pre_fee_t_max": max((r["disc"]["R_pre"] / (r["disc"]["R_sd"] / math.sqrt(r["disc"]["n"]))
                                          for r in tested if r["disc"].get("R_sd")), default=None),
               "rows": rep}
    (out / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print("done", m, len(surv), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
