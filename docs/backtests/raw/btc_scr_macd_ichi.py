"""137번: SCR(방아쇠) + MACD(상태) + 이치모쿠(상태) 조합 최적화, 롱·숏 거울.

등록: 2026-11-07-btc-scr-macd-ichimoku-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_scr_macd_ichi.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

import btc_fut_daily_regime as R
from btc_fractal import cci, crossed, load5, recent, rsi, stoch
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars
from btc_signal_scan import agg, load

M5 = 300_000
COST = 0.0005
FRAMES = {"15m": 3, "30m": 6, "1h": 12, "4h": 48}
SCRS = [(k, ls, w) for k in (2, 3) for ls in ("std", "rlx") for w in (3, 5)]
LEVELS = {"std": (20.0, 100.0, 30.0), "rlx": (25.0, 80.0, 35.0)}
MACDS = [None] + [(p, v) for p in ((8, 21, 5), (12, 26, 9), (19, 39, 9)) for v in ("hist", "line")]
ICHIS = [None] + [(p, v) for p in ((9, 26, 52), (10, 30, 60), (20, 60, 120)) for v in ("cloud", "tk")]
EXITS = ("Z1", "Z2", "Z3", "Z4")
W30, W7 = 8640, 2016
SPLIT = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
START = datetime(2020, 2, 1, tzinfo=timezone.utc).timestamp() * 1000


def ystart(y: int) -> float:
    return datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2 / (n + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def mid(h: np.ndarray, l: np.ndarray, n: int) -> np.ndarray:  # noqa: E741
    out = np.full(len(h), np.nan)
    out[n - 1:] = (swv(h, n).max(1) + swv(l, n).min(1)) / 2
    return out


def shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


class Frame:
    def __init__(self, d5: dict, factor: int, t5: np.ndarray):
        f = agg(d5, factor, 0)
        self.t, self.o, self.h, self.l, self.c = f["t"], f["o"], f["h"], f["l"], f["c"]
        self.close_t = self.t + factor * M5
        self.e5 = np.searchsorted(t5, self.close_t + M5)  # 봉 마감 5분 뒤 5분봉
        h, l, c = self.h, self.l, self.c  # noqa: E741
        self.sto, self.cci, self.rsi = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
        self.scr = {}
        for k, ls, w in SCRS:
            sl, cl, rl = LEVELS[ls]
            for side in (1, -1):
                up = side > 0
                a = recent(crossed(self.sto, sl if up else 100 - sl, up), w)
                a &= recent(crossed(self.cci, -cl if up else cl, up), w)
                if k == 3:
                    a &= recent(crossed(self.rsi, rl if up else 100 - rl, up), w)
                self.scr[(k, ls, w, side)] = a & ~np.r_[False, a[:-1]]
        self.macd = {None: (np.ones(len(c), bool), np.ones(len(c), bool))}
        for (fa, sl_, sg), v in [m for m in MACDS if m]:
            m = ema(c, fa) - ema(c, sl_)
            x = m - ema(m, sg) if v == "hist" else m
            self.macd[((fa, sl_, sg), v)] = (x > 0, x < 0)
        self.ichi = {None: (np.ones(len(c), bool), np.ones(len(c), bool))}
        self.kij = {}
        for (n1, n2, n3), v in [i for i in ICHIS if i]:
            ten, kij = mid(h, l, n1), mid(h, l, n2)
            self.kij[n2] = kij
            if v == "cloud":
                a_, b_ = shift((ten + kij) / 2, n2), shift(mid(h, l, n3), n2)
                with np.errstate(invalid="ignore"):
                    self.ichi[((n1, n2, n3), v)] = (c > np.fmax(a_, b_), c < np.fmin(a_, b_))
            else:
                with np.errstate(invalid="ignore"):
                    self.ichi[((n1, n2, n3), v)] = (ten > kij, ten < kij)
        # 기준선 반대 교차(새로 켜질 때만): 롱 청산은 종가가 기준선 아래로, 숏은 위로
        self.kx = {}
        for n2, kij in self.kij.items():
            with np.errstate(invalid="ignore"):
                below, above = c < kij, c > kij
            self.kx[(n2, 1)] = below & ~np.r_[False, below[:-1]]
            self.kx[(n2, -1)] = above & ~np.r_[False, above[:-1]]


def first_hit(arr: np.ndarray, e: np.ndarray, lvl: np.ndarray, w: int, below: bool) -> np.ndarray:
    n = len(arr)
    out = np.full(len(e), w)
    for a in range(0, len(e), 256):
        ee = e[a:a + 256]
        idx = np.minimum(ee[:, None] + np.arange(w)[None, :], n - 1)
        v = arr[idx]
        hit = v <= lvl[a:a + 256, None] if below else v >= lvl[a:a + 256, None]
        hit &= (ee[:, None] + np.arange(w)[None, :]) < n
        out[a:a + 256] = np.where(hit.any(1), hit.argmax(1), w)
    return out


class Exec:
    def __init__(self, d5: dict, fund):
        self.t, self.o, self.h, self.l, self.c = d5["t"], d5["o"], d5["h"], d5["l"], d5["c"]
        self.n = len(self.o)
        ft, fr = fund
        self.ft, self.fcs = ft, np.r_[0.0, np.cumsum(fr)]

    def funding(self, e: np.ndarray, x: np.ndarray, side) -> np.ndarray:
        a0 = np.searchsorted(self.ft, self.t[e], side="right")
        a1 = np.searchsorted(self.ft, self.t[np.minimum(x, self.n - 1)], side="right")
        return side * (self.fcs[a1] - self.fcs[a0])

    def bracket(self, e: np.ndarray, side: int, tp: float, sl: float, w: int):
        o, n = self.o, self.n
        px = o[e]
        sp, tpx = px * (1 - sl * side), px * (1 + tp * side)
        ks = first_hit(self.l if side > 0 else self.h, e, sp, w, side > 0)
        kt = first_hit(self.h if side > 0 else self.l, e, tpx, w, side < 0)
        g = np.empty(len(e))
        x = np.empty(len(e), int)
        st = (ks <= kt) & (ks < w)
        tg = (~st) & (kt < w)
        tm = ~(st | tg)
        js = np.minimum(e + ks, n - 1)
        fill = np.minimum(o[js], sp) if side > 0 else np.maximum(o[js], sp)
        g[st], x[st] = ((fill / px - 1) * side)[st], js[st]
        jt = np.minimum(e + kt, n - 1)
        tf = np.maximum(o[jt], tpx) if side > 0 else np.minimum(o[jt], tpx)
        g[tg], x[tg] = ((tf / px - 1) * side)[tg], jt[tg]
        je = np.minimum(e + w, n - 1)
        g[tm], x[tm] = ((o[je] / px - 1) * side)[tm], je[tm]
        return g - self.funding(e, x, side), x

    def stop_k(self, e: np.ndarray, side: int, sl: float, w: int = W30) -> np.ndarray:
        sp = self.o[e] * (1 - sl * side)
        return first_hit(self.l if side > 0 else self.h, e, sp, w, side > 0)

    def stop_exit(self, e: np.ndarray, side: int, sl: float, other: np.ndarray, w: int = W30, ks=None):
        """손절 sl 또는 other(5분봉 번호, 그 시가에 청산) 중 먼저, 최대 w. ks: 미리 계산한 손절 도달 봉."""
        o, n = self.o, self.n
        px = o[e]
        sp = px * (1 - sl * side)
        if ks is None:
            ks = self.stop_k(e, side, sl, w)
        js = e + ks
        cap = np.minimum(np.minimum(other, e + w), n - 1)
        st = (ks < w) & (js < cap)
        g = np.empty(len(e))
        x = np.empty(len(e), int)
        jss = np.minimum(js, n - 1)
        fill = np.minimum(o[jss], sp) if side > 0 else np.maximum(o[jss], sp)
        g[st], x[st] = ((fill / px - 1) * side)[st], jss[st]
        g[~st], x[~st] = ((o[cap] / px - 1) * side)[~st], cap[~st]
        return g - self.funding(e, x, side), x


def walk(e5: np.ndarray, side: np.ndarray, g: np.ndarray, x5: np.ndarray):
    order = np.argsort(e5, kind="stable")
    keep, busy = [], -1
    for j in order:
        if e5[j] >= busy and np.isfinite(g[j]):
            keep.append(j)
            busy = x5[j]
    return np.array(keep, int)


def summ(net: np.ndarray, side: np.ndarray) -> dict:
    if len(net) == 0:
        return {"n": 0}
    sd = net.std(ddof=1) if len(net) > 1 else 0.0
    t = float(net.mean() / (sd / math.sqrt(len(net)))) if sd > 0 else 0.0
    eq = np.cumprod(1 + net)
    pk = np.maximum.accumulate(np.r_[1.0, eq])
    eq3 = np.cumprod(1 + np.maximum(3 * net, -1))
    pk3 = np.maximum.accumulate(np.r_[1.0, eq3])
    lg, sh = net[side > 0], net[side < 0]
    return {"n": int(len(net)), "ev": round(float(net.mean() * 100), 4), "t": round(t, 2),
            "win": round(float((net > 0).mean() * 100), 1), "ret": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / pk[1:] - 1).min() * 100), 1), "ret3": round(float((eq3[-1] - 1) * 100), 1),
            "mdd3": round(float((eq3 / pk3[1:] - 1).min() * 100), 1),
            "long": round(float(lg.mean() * 100), 4) if len(lg) else None,
            "short": round(float(sh.mean() * 100), 4) if len(sh) else None, "nl": int(len(lg)), "ns": int(len(sh))}


def bh(p: np.ndarray, q: float) -> np.ndarray:
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * (np.arange(1, m + 1) / m)
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
    out = np.zeros(m, bool)
    out[o[:k]] = True
    return out


def norm_sf(t: float) -> float:
    return 0.5 * math.erfc(t / math.sqrt(2))


def ratio(net: np.ndarray, years: float) -> float:
    if len(net) == 0:
        return -9.0
    eq = np.cumprod(1 + net)
    pk = np.maximum.accumulate(np.r_[1.0, eq])
    mdd = (eq / pk[1:] - 1).min()
    cagr = max(eq[-1], 1e-9) ** (1 / years) - 1
    return float(cagr / abs(mdd)) if mdd < 0 else float(cagr * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--null", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.data)
    t0 = time.time()
    d5 = load(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    X = Exec(d5, fund)
    t5 = d5["t"]
    cells = {}  # key → (e5, side, gnet_before_cost)
    meta = {}
    for fn, fac in FRAMES.items():
        F = Frame(d5, fac, t5)
        valid = (F.t >= START) & (F.e5 < X.n - 2)
        print(fn, "frame bars", len(F.t), round(time.time() - t0), flush=True)
        # 합집합 진입의 청산 미리 계산
        cache = {}
        for side in (1, -1):
            uni = np.zeros(len(F.t), bool)
            for k, ls, w in SCRS:
                uni |= F.scr[(k, ls, w, side)]
            uni &= valid
            fi = np.flatnonzero(uni)
            e = F.e5[fi]
            pos = np.full(len(F.t), -1)
            pos[fi] = np.arange(len(fi))
            z2 = X.bracket(e, side, 0.015, 0.015, W7)
            z3 = X.bracket(e, side, 0.03, 0.015, W7)
            z4 = {}
            for n2 in (26, 30, 60):
                kx = F.kx[(n2, side)]
                nxt = np.full(len(F.t) + 1, len(F.t))
                for j in range(len(F.t) - 1, -1, -1):
                    nxt[j] = j if kx[j] else nxt[j + 1]
                jx = nxt[np.minimum(fi + 1, len(F.t))]
                other = np.where(jx < len(F.t), F.e5[np.minimum(jx, len(F.t) - 1)], X.n - 1)
                z4[n2] = X.stop_exit(e, side, 0.03, other)
            k1 = X.stop_k(e, side, 0.01)
            cache[side] = (pos, e, z2, z3, z4, k1)
        print(fn, "exits cached", round(time.time() - t0), flush=True)
        for (k, ls, w), mc, ic in product(SCRS, MACDS, ICHIS):
            ev = {}
            for side in (1, -1):
                mstate = F.macd[mc][0 if side > 0 else 1]
                istate = F.ichi[ic][0 if side > 0 else 1]
                ev[side] = np.flatnonzero(F.scr[(k, ls, w, side)] & mstate & istate & valid)
            n2 = 26 if ic is None else ic[0][1]
            for zx in EXITS:
                E, S, G, XX = [], [], [], []
                for side in (1, -1):
                    pos, e_all, z2, z3, z4, k1 = cache[side]
                    fi = ev[side]
                    if len(fi) == 0:
                        continue
                    p = pos[fi]
                    e = e_all[p]
                    if zx == "Z2":
                        g, x = z2[0][p], z2[1][p]
                    elif zx == "Z3":
                        g, x = z3[0][p], z3[1][p]
                    elif zx == "Z4":
                        g, x = z4[n2][0][p], z4[n2][1][p]
                    else:
                        opp = F.e5[ev[-side]]
                        j = np.searchsorted(opp, e, side="right")
                        other = np.where(j < len(opp), opp[np.minimum(j, len(opp) - 1)], X.n - 1)
                        g, x = X.stop_exit(e, side, 0.01, other, ks=k1[p])
                    E.append(e), S.append(np.full(len(e), side)), G.append(g), XX.append(x)
                if not E:
                    continue
                e, s, g, x = map(np.concatenate, (E, S, G, XX))
                keep = walk(e, s, g, x)
                key = f"{fn}|SCR{k}-{ls}-w{w}|M{'-' if mc is None else f'{mc[0][0]}.{mc[0][1]}.{mc[0][2]}-{mc[1]}'}" \
                      f"|I{'-' if ic is None else f'{ic[0][0]}.{ic[0][1]}.{ic[0][2]}-{ic[1]}'}|{zx}"
                cells[key] = (e[keep].astype(np.int32), s[keep].astype(np.int8), g[keep].astype(np.float32),
                              x[keep].astype(np.int32))
                meta[key] = {"frame": fn, "scr": (k, ls, w), "macd": mc, "ichi": ic, "exit": zx}
        print(fn, "cells", len(cells), round(time.time() - t0), flush=True)
    keys = list(cells)
    rep: dict = {"n_cells": len(keys)}
    # 평가 A
    rows = {}
    for kk in keys:
        e, s, g, x = cells[kk]
        tt = t5[e]
        net = g.astype(float) - 2 * COST
        d, h = tt < SPLIT, tt >= SPLIT
        rows[kk] = {"disc": summ(net[d], s[d]), "hold": summ(net[h], s[h]),
                    "hold_c10": summ(net[h] - 2 * COST, s[h])}
    elig = [kk for kk in keys if rows[kk]["disc"]["n"] >= 60]
    pv = np.array([norm_sf(rows[kk]["disc"]["t"]) for kk in elig])
    surv = [kk for kk, ok in zip(elig, bh(pv, 0.10)) if ok] if len(elig) else []
    rep["eligible"] = len(elig)
    rep["min_p"] = float(pv.min()) if len(pv) else None
    rep["survivors"] = surv
    top = sorted(elig, key=lambda kk: -rows[kk]["disc"]["t"])[:10]
    rep["top10"] = [{"cell": kk, **rows[kk]} for kk in top]
    rep["survivor_rows"] = [{"cell": kk, **rows[kk]} for kk in surv]
    tdist = np.array([rows[kk]["disc"]["t"] for kk in elig])
    rep["disc_t_hist"] = {str(b): int(((tdist >= b) & (tdist < b + 1)).sum()) for b in range(-6, 6)}
    # 살아남은 칸: 원형 이동 대조 (확인 구간 안)
    if surv and a.null:
        rng = np.random.default_rng(7)
        frames = {fn: Frame(d5, fac, t5) for fn, fac in FRAMES.items() if any(meta[kk]["frame"] == fn for kk in surv)}
        for kk in surv:
            m = meta[kk]
            F = frames[m["frame"]]
            valid = (F.t >= START) & (F.e5 < X.n - 2)
            i0 = int(np.searchsorted(F.t, SPLIT))
            span = len(F.t) - i0
            k, ls, w = m["scr"]
            evb = {}
            for side in (1, -1):
                evb[side] = (F.scr[(k, ls, w, side)] & F.macd[m["macd"]][0 if side > 0 else 1]
                             & F.ichi[m["ichi"]][0 if side > 0 else 1] & valid)
            act = rows[kk]["hold"]["ev"]
            null = []
            for _ in range(a.null):
                sh = int(rng.integers(96, span - 96))
                ev = {}
                for side in (1, -1):
                    b_ = evb[side].copy()
                    b_[i0:] = np.roll(evb[side][i0:], sh)
                    ev[side] = np.flatnonzero(b_ & valid)
                E, S, G, XX = [], [], [], []
                for side in (1, -1):
                    fi = ev[side][ev[side] >= i0]
                    if len(fi) == 0:
                        continue
                    e = F.e5[fi]
                    zx = m["exit"]
                    if zx == "Z2":
                        g, x = X.bracket(e, side, 0.015, 0.015, W7)
                    elif zx == "Z3":
                        g, x = X.bracket(e, side, 0.03, 0.015, W7)
                    elif zx == "Z4":
                        n2 = 26 if m["ichi"] is None else m["ichi"][0][1]
                        kx = F.kx[(n2, side)]
                        jx = np.array([i + 1 + int(np.argmax(kx[i + 1:])) if kx[i + 1:].any() else len(F.t)
                                       for i in fi])
                        other = np.where(jx < len(F.t), F.e5[np.minimum(jx, len(F.t) - 1)], X.n - 1)
                        g, x = X.stop_exit(e, side, 0.03, other)
                    else:
                        opp = F.e5[ev[-side]]
                        j = np.searchsorted(opp, e, side="right")
                        other = np.where(j < len(opp), opp[np.minimum(j, len(opp) - 1)], X.n - 1)
                        g, x = X.stop_exit(e, side, 0.01, other)
                    E.append(e), S.append(np.full(len(e), side)), G.append(g), XX.append(x)
                if not E:
                    null.append(-9)
                    continue
                e, s, g, x = map(np.concatenate, (E, S, G, XX))
                keep = walk(e, s, g, x)
                null.append(float((g[keep] - 2 * COST).mean() * 100) if len(keep) else -9)
            null = np.array(null)
            rows[kk]["null_mean"] = round(float(null.mean()), 4)
            rows[kk]["null_p"] = round(float((null >= act).mean()), 4)
            hh = rows[kk]["hold"]
            rows[kk]["pass"] = bool(hh["n"] >= 30 and hh["ev"] > 0 and (hh["long"] or 0) >= 0
                                    and (hh["short"] or 0) >= 0 and rows[kk]["hold_c10"]["ev"] > 0
                                    and rows[kk]["null_p"] < 0.05)
        rep["survivor_rows"] = [{"cell": kk, **rows[kk]} for kk in surv]
    # 평가 B: 전진 최적화
    wf = []
    oos = []
    for y in range(2022, 2027):
        lo, hi = ystart(y - 2), ystart(y)
        best, bv = None, -1e9
        for kk in keys:
            e, s, g, x = cells[kk]
            tt = t5[e]
            msk = (tt >= lo) & (tt < hi)
            if msk.sum() < 40:
                continue
            v = ratio(g[msk].astype(float) - 2 * COST, 2.0)
            if v > bv:
                bv, best = v, kk
        e, s, g, x = cells[best]
        tt = t5[e]
        msk = (tt >= ystart(y)) & (tt < ystart(y + 1))
        net = g[msk].astype(float) - 2 * COST
        wf.append({"year": y, "pick": best, "train_ratio": round(bv, 3), "oos": summ(net, s[msk])})
        oos.append((e[msk], s[msk], net, x[msk]))
    rep["walk_forward"] = wf
    e = np.concatenate([z[0] for z in oos])
    s = np.concatenate([z[1] for z in oos])
    net = np.concatenate([z[2] for z in oos])
    xx = np.concatenate([z[3] for z in oos])
    rep["wf_total"] = summ(net, s)
    rep["wf_years_pos"] = int(sum(1 for z in wf if z["oos"].get("ret", -1) > 0))
    # 일별 평가 낙폭 (표본 밖 이어 붙임)
    day = (t5 // 86_400_000).astype(np.int64)
    last_of_day = np.flatnonzero(np.r_[day[1:] != day[:-1], True])
    eq, peak, mdd = 1.0, 1.0, 0.0
    for ei, si, ni, xi in zip(e, s, net, xx):
        px = X.o[ei]
        for j in last_of_day[(last_of_day > ei) & (last_of_day < xi)]:
            v = eq * (1 + si * (X.c[j] / px - 1) - COST)
            peak = max(peak, v)
            mdd = min(mdd, v / peak - 1)
        eq *= 1 + ni
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    rep["wf_daily_mdd"] = round(mdd * 100, 1)
    rep["wf_pass"] = bool(rep["wf_total"].get("ret", -1) > 0 and rep["wf_years_pos"] >= 4
                          and (rep["wf_total"].get("long") or 0) >= 0 and (rep["wf_total"].get("short") or 0) >= 0)
    # 비교: BTC 보유, R1 (해별)
    d5b = load5(base / "futures/klines_5m.csv")
    db = bars(d5b, 0)
    reg, newday = R.regime(d5b, db, 0)
    st = int(np.searchsorted(db["t"], db["t"][0] + 60 * 86_400_000))
    r1 = R.run(db, reg, newday, st, True, load_funding(base / "futures/funding.csv"))
    comp = {}
    for y in range(2022, 2027):
        i0, i1 = np.searchsorted(t5, ystart(y)), min(np.searchsorted(t5, ystart(y + 1)), len(t5) - 1)
        hold = (X.c[i1] / X.o[i0] - 1) * 100
        rr = [z[2] - 2 * COST for z in r1 if ystart(y) <= z[4] < ystart(y + 1)]
        comp[y] = {"btc_hold": round(float(hold), 1), "r1": round(float((np.prod([1 + v for v in rr]) - 1) * 100), 1),
                   "wf": next(z["oos"].get("ret") for z in wf if z["year"] == y)}
    rep["compare"] = comp
    # 기여 분석
    att = {"macd_only": [], "ichi_only": [], "both": []}
    for kk in keys:
        m = meta[kk]
        if m["macd"] is None and m["ichi"] is None:
            continue
        bk = kk.split("|")
        bk[2], bk[3] = "M-", "I-"
        bkey = "|".join(bk)
        if bkey not in rows:
            continue
        r0, r1_ = rows[bkey], rows[kk]
        if r0["disc"]["n"] < 30 or r1_["disc"]["n"] < 30 or r0["hold"]["n"] < 30 or r1_["hold"]["n"] < 30:
            continue
        grp = "both" if m["macd"] and m["ichi"] else ("macd_only" if m["macd"] else "ichi_only")
        att[grp].append((r1_["disc"]["ev"] - r0["disc"]["ev"], r1_["hold"]["ev"] - r0["hold"]["ev"]))
    rep["attribution"] = {g: {"n": len(v), "disc_median": round(float(np.median([a_ for a_, _ in v])), 4) if v else None,
                              "disc_pos": round(float(np.mean([a_ > 0 for a_, _ in v])), 3) if v else None,
                              "hold_median": round(float(np.median([b_ for _, b_ in v])), 4) if v else None,
                              "hold_pos": round(float(np.mean([b_ > 0 for _, b_ in v])), 3) if v else None,
                              "both_pos": round(float(np.mean([a_ > 0 and b_ > 0 for a_, b_ in v])), 3) if v else None}
                          for g, v in att.items()}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    np.savez_compressed(out / "picks.npz", **{f"y{z['year']}": np.c_[cells[z["pick"]][0], cells[z["pick"]][1],
                                                                          cells[z["pick"]][2], cells[z["pick"]][3]]
                                              for z in wf})
    json.dump({"picks": {z["year"]: z["pick"] for z in wf}, "survivors": surv},
              (out / "picks.json").open("w"), ensure_ascii=False)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
