"""138번: 이치모쿠 구름 가장자리 진입(I1 구름 안 되돌림, I2 구름 밖 눌림), 롱·숏 거울.

등록: 2026-11-07-btc-ichimoku-cloud-edge-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ichi_cloud_edge.py --out <폴더>
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

from btc_fractal import cci, crossed, recent, rsi, stoch
from btc_scr_gate15 import load_funding
from btc_scr_macd_ichi import bh, mid, norm_sf, shift
from btc_signal_scan import agg, load

M5 = 300_000
COST = 0.0005
FRAMES = {"30m": 6, "1h": 12, "4h": 48, "1d": 288}
ICHIS = ((9, 26, 52), (20, 60, 120))
INTERP = ("I1", "I2")
CONFS = ("none", "scr2", "scr3")
TARGETS = ("T1", "T2", "T3")
HOLD = 260  # 26 × 10 봉
START = datetime(2020, 3, 1, tzinfo=timezone.utc).timestamp() * 1000
SPLIT = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
MIN_D, MAX_D = 0.003, 0.03


def first_hit(arr: np.ndarray, e: np.ndarray, lvl: np.ndarray, w: int, below: bool) -> np.ndarray:
    """e부터 w개 5분봉 안에서 arr가 lvl에 처음 닿는 상대 위치(없으면 w)."""
    n = len(arr)
    out = np.full(len(e), w)
    step = max(1, int(1.5e7 // max(w, 1)))
    ar = np.arange(w)
    for a in range(0, len(e), step):
        ee = e[a:a + step]
        pos = ee[:, None] + ar[None, :]
        v = arr[np.minimum(pos, n - 1)]
        hit = (v <= lvl[a:a + step, None]) if below else (v >= lvl[a:a + step, None])
        hit &= pos < n
        out[a:a + step] = np.where(hit.any(1), hit.argmax(1), w)
    return out


class Exec:
    def __init__(self, d5: dict, fund):
        self.t, self.o, self.h, self.l = d5["t"], d5["o"], d5["h"], d5["l"]
        self.n = len(self.o)
        ft, fr = fund
        self.ft, self.fcs = ft, np.r_[0.0, np.cumsum(fr)]

    def run(self, e: np.ndarray, side: int, sp: np.ndarray, tp: np.ndarray, w: int):
        """손절가 sp, 목표가 tp 배열. 같은 봉에서 둘 다 닿으면 손절. (총손익−펀딩, 청산 5분봉)."""
        o, n = self.o, self.n
        px = o[e]
        ks = first_hit(self.l if side > 0 else self.h, e, sp, w, side > 0)
        kt = first_hit(self.h if side > 0 else self.l, e, tp, w, side < 0)
        st = (ks <= kt) & (ks < w)
        tg = (~st) & (kt < w)
        js = np.minimum(e + ks, n - 1)
        jt = np.minimum(e + kt, n - 1)
        je = np.minimum(e + w, n - 1)
        sfill = np.minimum(o[js], sp) if side > 0 else np.maximum(o[js], sp)
        tfill = np.maximum(o[jt], tp) if side > 0 else np.minimum(o[jt], tp)
        xp = np.where(st, sfill, np.where(tg, tfill, o[je]))
        x = np.where(st, js, np.where(tg, jt, je))
        g = (xp / px - 1) * side
        a0 = np.searchsorted(self.ft, self.t[e], side="right")
        a1 = np.searchsorted(self.ft, self.t[x], side="right")
        return g - side * (self.fcs[a1] - self.fcs[a0]), x, np.where(st, 1, np.where(tg, 2, 0))


class Frame:
    def __init__(self, d5: dict, fac: int, t5: np.ndarray):
        f = agg(d5, fac, 0)
        self.fac = fac
        self.t, self.o, self.h, self.l, self.c = f["t"], f["o"], f["h"], f["l"], f["c"]
        self.e5 = np.searchsorted(t5, self.t + fac * M5 + M5)  # 봉 마감 5분 뒤 5분봉
        h, l, c = self.h, self.l, self.c  # noqa: E741
        s, cc, r = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
        self.conf = {("none", 1): np.ones(len(c), bool), ("none", -1): np.ones(len(c), bool)}
        for k in (2, 3):
            for side in (1, -1):
                up = side > 0
                a = recent(crossed(s, 20 if up else 80, up), 3) & recent(crossed(cc, -100 if up else 100, up), 3)
                if k == 3:
                    a &= recent(crossed(r, 30 if up else 70, up), 3)
                self.conf[(f"scr{k}", side)] = recent(a, 3)
        self.cloud = {}
        for n1, n2, n3 in ICHIS:
            ten, kij = mid(h, l, n1), mid(h, l, n2)
            a_, b_ = shift((ten + kij) / 2, n2), shift(mid(h, l, n3), n2)
            self.cloud[(n1, n2, n3)] = (np.fmin(a_, b_), np.fmax(a_, b_))

    def events(self, ich, interp: str, side: int):
        lo, up = self.cloud[ich]
        h, l, c = self.h, self.l, self.c  # noqa: E741
        pc = np.r_[np.nan, c[:-1]]
        plo, pup = np.r_[np.nan, lo[:-1]], np.r_[np.nan, up[:-1]]
        with np.errstate(invalid="ignore"):
            if interp == "I1":
                inside = (pc >= plo) & (pc <= pup)
                if side > 0:
                    ev = inside & (l <= lo) & (c > lo)
                    stop = lo * (1 - 0.001)
                else:
                    ev = inside & (h >= up) & (c < up)
                    stop = up * (1 + 0.001)
            else:
                cs = [np.r_[np.full(k, np.nan), c[:-k]] for k in (1, 2, 3)]
                us = [np.r_[np.full(k, np.nan), up[:-k]] for k in (1, 2, 3)]
                ls = [np.r_[np.full(k, np.nan), lo[:-k]] for k in (1, 2, 3)]
                if side > 0:
                    prior = (cs[0] > us[0]) & (cs[1] > us[1]) & (cs[2] > us[2])
                    ev = prior & (l <= up) & (c > up)
                    stop = lo * (1 - 0.001)
                else:
                    prior = (cs[0] < ls[0]) & (cs[1] < ls[1]) & (cs[2] < ls[2])
                    ev = prior & (h >= lo) & (c < lo)
                    stop = up * (1 + 0.001)
        ev = ev & np.isfinite(lo) & np.isfinite(up)
        return ev, stop, up - lo


def trades_for(F: Frame, X: Exec, ich, interp, conf, tgt, mask_shift: int = 0, lo_t: float = START):
    """칸 하나의 거래(한 포지션). mask_shift는 무작위 대조용 원형 이동(봉 수)."""
    w = HOLD * F.fac
    rows = []
    skipped = 0
    for side in (1, -1):
        ev, stop, thick = F.events(ich, interp, side)
        ev = ev & F.conf[(conf, side)]
        if mask_shift:
            i0 = int(np.searchsorted(F.t, lo_t))
            seg = ev[i0:].copy()
            ev = ev.copy()
            ev[i0:] = np.roll(seg, mask_shift)
            # 밀린 시각의 구름으로 손절·목표를 다시 잡는다(같은 방식)
            ev &= np.isfinite(stop)
        fi = np.flatnonzero(ev & (F.t >= lo_t) & (F.e5 < X.n - 2))
        if len(fi) == 0:
            continue
        e = F.e5[fi]
        px = X.o[e]
        sp = stop[fi]
        d = side * (px - sp) / px
        ok = (d >= MIN_D) & (d <= MAX_D)
        skipped += int((~ok).sum())
        fi, e, px, sp, d = fi[ok], e[ok], px[ok], sp[ok], d[ok]
        if len(fi) == 0:
            continue
        if tgt == "T1":
            tp = px + side * thick[fi]
        elif tgt == "T2":
            tp = px + side * d * px
        else:
            tp = px + side * 2 * d * px
        g, x, why = X.run(e, side, sp, tp, w)
        for j in range(len(fi)):
            rows.append((int(e[j]), side, float(g[j]), int(x[j]), float(d[j]), float(X.t[e[j]]), int(why[j])))
    rows.sort()
    keep, busy = [], -1
    for r in rows:
        if r[0] >= busy:
            keep.append(r)
            busy = r[3]
    return keep, skipped


def stats(tr: list, cost: float = COST) -> dict:
    if not tr:
        return {"n": 0}
    g = np.array([r[2] for r in tr])
    d = np.array([r[4] for r in tr])
    sd = np.array([r[1] for r in tr])
    net = g - 2 * cost
    R = net / d
    s = R.std(ddof=1) if len(R) > 1 else 0.0
    t = float(R.mean() / (s / math.sqrt(len(R)))) if s > 0 else 0.0
    eq = np.cumprod(1 + net)
    pk = np.maximum.accumulate(np.r_[1.0, eq])
    out = {"n": len(tr), "R": round(float(R.mean()), 4), "t": round(t, 2), "ev": round(float(net.mean() * 100), 4),
           "win": round(float((net > 0).mean() * 100), 1), "ret": round(float((eq[-1] - 1) * 100), 1),
           "mdd": round(float((eq / pk[1:] - 1).min() * 100), 1), "d_med": round(float(np.median(d) * 100), 2),
           "nl": int((sd > 0).sum()), "ns": int((sd < 0).sum())}
    out["R_long"] = round(float(R[sd > 0].mean()), 4) if (sd > 0).any() else None
    out["R_short"] = round(float(R[sd < 0].mean()), 4) if (sd < 0).any() else None
    return out


def describe(key) -> str:
    fr, ich, interp, conf, tgt = key
    a = "구름 안에서 하단(롱)·상단(숏)에 닿고 되돌아 마감" if interp == "I1" else \
        "구름 위(롱)·아래(숏)에서 상단(롱)·하단(숏)까지 눌렸다 지지·저항 마감"
    c = {"none": "확인 없음", "scr2": "SCR 2단 같은 방향 3봉 안", "scr3": "SCR 3단 같은 방향 3봉 안"}[conf]
    tg = {"T1": "목표 = 구름 두께", "T2": "목표 1R", "T3": "목표 2R"}[tgt]
    return f"{fr}, 이치모쿠 {ich}, {a}, {c}, 손절 구름 반대편 −0.1%, {tg}"


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
    X = Exec(d5, load_funding(base / "futures/funding.csv"))
    frames = {fn: Frame(d5, fac, d5["t"]) for fn, fac in FRAMES.items()}
    print("frames", round(time.time() - t0), flush=True)
    cells, trades = {}, {}
    for fn, ich, interp, conf, tgt in product(FRAMES, ICHIS, INTERP, CONFS, TARGETS):
        key = (fn, ich, interp, conf, tgt)
        tr, sk = trades_for(frames[fn], X, ich, interp, conf, tgt)
        disc = [r for r in tr if r[5] < SPLIT]
        hold = [r for r in tr if r[5] >= SPLIT]
        cells[key] = {"disc": stats(disc), "hold": stats(hold), "hold_c10": stats(hold, 0.001), "skipped": sk}
        trades[key] = tr
    print("cells", len(cells), round(time.time() - t0), flush=True)
    keys = list(cells)
    elig = [k for k in keys if cells[k]["disc"].get("n", 0) >= 40]
    p = np.array([norm_sf(cells[k]["disc"]["t"]) for k in elig])
    surv = bh(p, 0.10) if len(p) else np.zeros(0, bool)
    survivors = [k for k, s in zip(elig, surv) if s]
    rep = {"n_cells": len(keys), "n_eligible": len(elig), "min_p": round(float(p.min()), 5) if len(p) else None,
           "survivors": [], "top10": []}
    for k in survivors:
        c = cells[k]
        h = c["hold"]
        ok = (h.get("n", 0) >= 25 and h.get("R", -1) > 0 and (h.get("R_long") or 0) >= 0
              and (h.get("R_short") or 0) >= 0 and c["hold_c10"].get("R", -1) > 0)
        nullp = None
        if ok and a.null:
            F = frames[k[0]]
            i0 = int(np.searchsorted(F.t, SPLIT))
            span = len(F.t) - i0
            rng = np.random.default_rng(7)
            null = []
            for _ in range(a.null):
                sh = int(rng.integers(26, span - 26))
                tr, _ = trades_for(F, X, *k[1:], mask_shift=sh, lo_t=SPLIT)
                null.append(stats(tr).get("R", 0.0))
            nullp = round(float((np.array(null) >= h["R"]).mean()), 4)
            ok = ok and nullp < 0.05
        rep["survivors"].append({"key": str(k), "rule": describe(k), "disc": c["disc"], "hold": h,
                                 "hold_c10": c["hold_c10"], "null_p": nullp, "pass": bool(ok)})
    order = sorted(elig, key=lambda k: -cells[k]["disc"]["t"])
    for k in order[:10]:
        c = cells[k]
        rep["top10"].append({"key": str(k), "rule": describe(k), "disc": c["disc"], "hold": c["hold"],
                             "hold_c10": c["hold_c10"], "skipped": c["skipped"]})
    rep["all"] = {str(k): v for k, v in cells.items()}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "trades.json").write_text(json.dumps({str(k): v for k, v in trades.items()}))
    print("done", round(time.time() - t0), "survivors", len(survivors), flush=True)


if __name__ == "__main__":
    main()
