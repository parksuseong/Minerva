"""136번: 코인베이스 프리미엄, 김치 프리미엄, DVOL(내재 변동성). 등록: 2026-11-07-btc-premium-dvol-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_premium_dvol.py --out <폴더> [--null 200]
자료: ~/.minerva/data/btc/{coinbase,upbit,fx,deribit}/ (btc_premium_fetch.py), futures/klines_5m.csv, 5m.csv(현물), futures/funding.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from btc_fractal import cci, load5, resample, rsi
from btc_scr_gate15 import load_funding

H = 3_600_000
M5 = 300_000
DAY = 86_400_000
COST = 0.0005
BASE = Path("~/.minerva/data/btc").expanduser()
DISC = (datetime(2021, 4, 1, tzinfo=timezone.utc).timestamp() * 1000,
        datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def load_csv(path: Path, tcol: int = 0, vcol: int = 4) -> tuple[np.ndarray, np.ndarray]:
    t, v = [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(float(row[tcol]))
            v.append(float(row[vcol]))
    t, v = np.array(t), np.array(v)
    o = np.argsort(t, kind="stable")
    return t[o], v[o]


def on_grid(grid: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """같은 시작 시각의 값. 없으면 NaN."""
    j = np.searchsorted(t, grid)
    ok = (j < len(t)) & (t[np.minimum(j, len(t) - 1)] == grid)
    out = np.full(len(grid), np.nan)
    out[ok] = v[j[ok]]
    return out


def roll_z(x: np.ndarray, w: int) -> np.ndarray:
    """끝이 현재 봉인 w봉 창의 z. NaN은 건너뛴다(창 안 유효값 ≥ w/2)."""
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    xs = np.where(ok, x, 0.0)
    c1 = np.r_[0.0, np.cumsum(xs)]
    c2 = np.r_[0.0, np.cumsum(xs * xs)]
    cn = np.r_[0, np.cumsum(ok)]
    for_i = np.arange(w - 1, len(x))
    n = cn[for_i + 1] - cn[for_i + 1 - w]
    s1 = c1[for_i + 1] - c1[for_i + 1 - w]
    s2 = c2[for_i + 1] - c2[for_i + 1 - w]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - m * m, 0))
        z = (x[for_i] - m) / sd
    z[(n < w // 2) | ~ok[for_i]] = np.nan
    out[for_i] = z
    return out


def month_thresholds(t: np.ndarray, x: np.ndarray, qs: tuple, look_days: int = 180) -> dict:
    """달마다, 그 달 시작 전 look_days일 값의 분위수(과거만). 반환: q → 봉별 문턱."""
    out = {q: np.full(len(t), np.nan) for q in qs}
    months = (t.astype("datetime64[ms]").astype("datetime64[M]"))
    for m in np.unique(months):
        ms = m.astype("datetime64[ms]").astype(np.int64)
        idx = np.flatnonzero(months == m)
        past = (t < ms) & (t >= ms - look_days * DAY) & np.isfinite(x)
        if past.sum() < 24 * 60:
            continue
        for q in qs:
            out[q][idx] = np.quantile(x[past], q)
    return out


def daily_dir(d5: dict, t_close: np.ndarray) -> np.ndarray:
    """마감된 일봉(UTC 0시) CCI20>0·RSI14>50 = +1, 둘 다 아래 = −1."""
    u = resample(d5, 288)
    c, r = cci(u["h"], u["l"], u["c"]), rsi(u["c"])
    with np.errstate(invalid="ignore"):
        dr = np.where((c > 0) & (r > 50), 1, np.where((c < 0) & (r < 50), -1, 0))
    close = u["t"] + DAY
    j = np.searchsorted(close, t_close, side="right") - 1
    out = np.zeros(len(t_close), int)
    out[j >= 0] = dr[j[j >= 0]]
    return out


def fresh(x: np.ndarray) -> np.ndarray:
    return x & ~np.r_[False, x[:-1]]


class Exec:
    """5분봉 선물 체결. 진입: 1시간봉 마감 +5분의 5분봉 시가."""

    def __init__(self, f5: dict, fund) -> None:
        self.t, self.o, self.h, self.l = f5["t"], f5["o"], f5["h"], f5["l"]
        self.ft, self.fr = fund

    def trade(self, t_close: float, side: int, y: str) -> tuple | None:
        e = int(np.searchsorted(self.t, t_close + M5))
        if e >= len(self.t) or self.t[e] != t_close + M5:
            return None
        px = self.o[e]
        if y == "Y1":
            stop, tgt, horizon = px * (1 - 0.015 * side), px * (1 + 0.03 * side), DAY
        else:
            stop, tgt, horizon = px * (1 - 0.03 * side), None, DAY
        end_t = self.t[e] + horizon
        j = e
        while j < len(self.t) and self.t[j] < end_t:
            hi, lo = self.h[j], self.l[j]
            if (side > 0 and lo <= stop) or (side < 0 and hi >= stop):
                fill = (min(self.o[j], stop) if side > 0 else max(self.o[j], stop)) if j > e else stop
                return self._done(px, fill, side, e, j, "stop")
            if tgt is not None and j > e and ((side > 0 and hi >= tgt) or (side < 0 and lo <= tgt)):
                fill = max(self.o[j], tgt) if side > 0 else min(self.o[j], tgt)
                return self._done(px, fill, side, e, j, "target")
            j += 1
        if j >= len(self.t):
            return None
        return self._done(px, self.o[j], side, e, j, "time")

    def _done(self, px, fill, side, e, x, why) -> tuple:
        g = (fill / px - 1) * side
        m = (self.ft > self.t[e]) & (self.ft <= self.t[x])
        g -= side * float(self.fr[m].sum())
        return (float(self.t[e]), side, float(g), float(self.t[x]), why)


def run_cell(ex: Exec, tc: np.ndarray, sig: np.ndarray, side: np.ndarray, y: str, t0: float, t1: float) -> list:
    out, busy_until, last_sig = [], -1.0, -1e18
    for i in np.flatnonzero(sig):
        if not (t0 <= tc[i] < t1):
            continue
        if tc[i] < busy_until or tc[i] - last_sig < DAY:
            continue
        r = ex.trade(tc[i], int(side[i]), y)
        if r is None:
            continue
        out.append(r)
        busy_until = r[3]
        last_sig = tc[i]
    return out


def summ(tr: list, cost: float = COST) -> dict:
    if not tr:
        return {"n": 0}
    g = np.array([z[2] for z in tr]) - 2 * cost
    sd = g.std(ddof=1) if len(g) > 1 else np.nan
    t = g.mean() / sd * math.sqrt(len(g)) if sd and sd > 0 else 0.0
    eq = np.cumprod(1 + g)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    lo = [z for z in tr if z[1] > 0]
    sh = [z for z in tr if z[1] < 0]
    return {"n": len(g), "win": round(float((g > 0).mean() * 100), 1), "ev": round(float(g.mean() * 100), 3),
            "t": round(float(t), 2), "ret": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
            "long_n": len(lo), "short_n": len(sh),
            "long_ev": round(float((np.array([z[2] for z in lo]) - 2 * cost).mean() * 100), 3) if lo else None,
            "short_ev": round(float((np.array([z[2] for z in sh]) - 2 * cost).mean() * 100), 3) if sh else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--null", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f5 = load5(BASE / "futures/klines_5m.csv")
    s5 = load5(BASE / "5m.csv")
    fh = resample(f5, 12)
    sh_ = resample(s5, 12)
    grid = fh["t"]
    tc = grid + H
    spot_c = on_grid(grid, sh_["t"], sh_["c"])
    cbt, cbc = load_csv(BASE / "coinbase/btcusd_1h.csv")
    cb_c = on_grid(grid, cbt, cbc)
    upt, upc = load_csv(BASE / "upbit/krwbtc_1h.csv")
    up_c = on_grid(grid, upt, upc)
    # 환율: 봉 날짜(UTC) 이전 날짜의 ECB 값만 쓴다
    fxd, fxv = [], []
    with (BASE / "fx/usdkrw_daily.csv").open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            fxd.append(datetime.strptime(row[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            fxv.append(float(row[1]))
    fxd, fxv = np.array(fxd), np.array(fxv)
    day0 = (grid // DAY) * DAY
    j = np.searchsorted(fxd, day0, side="left") - 1
    fx = np.where(j >= 0, fxv[np.maximum(j, 0)], np.nan)
    dvt, dvc = load_csv(BASE / "deribit/btc_dvol_1h.csv")
    dv = on_grid(grid, dvt, dvc)

    with np.errstate(invalid="ignore", divide="ignore"):
        cp = cb_c / spot_c - 1
        kp = up_c / (fx * spot_c) - 1
        lr = np.r_[np.nan, np.diff(np.log(fh["c"]))]
    cp_z = roll_z(cp, 720)
    kp_z = roll_z(kp, 720)
    cp4 = cp - np.r_[np.full(4, np.nan), cp[:-4]]
    cp4_z = roll_z(cp4, 720)
    rv = np.full(len(lr), np.nan)
    w = 720
    lr0 = np.where(np.isfinite(lr), lr, 0.0)
    c1 = np.r_[0.0, np.cumsum(lr0)]
    c2 = np.r_[0.0, np.cumsum(lr0 * lr0)]
    idx = np.arange(w, len(lr))
    m = (c1[idx + 1] - c1[idx + 1 - w]) / w
    rv[idx] = np.sqrt(np.maximum((c2[idx + 1] - c2[idx + 1 - w]) / w - m * m, 0)) * math.sqrt(8760) * 100
    spread = dv - rv
    dv24 = dv - np.r_[np.full(24, np.nan), dv[:-24]]
    th_cp4 = month_thresholds(tc, cp4_z, (0.1, 0.9))
    th_sp = month_thresholds(tc, spread, (0.9,))
    th_d24 = month_thresholds(tc, dv24, (0.9,))
    ddir = daily_dir(f5, tc)
    open24 = np.r_[np.full(23, np.nan), fh["o"][:-23]]
    close = fh["c"]

    with np.errstate(invalid="ignore"):
        sigs = {}
        L, S = cp_z >= 2, cp_z <= -2
        sigs["H1"] = (fresh(L) | fresh(S), np.where(L, 1, -1))
        L, S = cp4_z >= th_cp4[0.9], cp4_z <= th_cp4[0.1]
        sigs["H2"] = (fresh(L) | fresh(S), np.where(L, 1, -1))
        L, S = kp_z <= -2, kp_z >= 2  # 역발상: 김프 과열이면 숏
        sigs["H3"] = (fresh(L) | fresh(S), np.where(L, 1, -1))
        on = (spread >= th_sp[0.9]) & (ddir != 0)
        sigs["H4"] = (fresh(on), np.where(ddir > 0, 1, -1))
        hot = dv24 >= th_d24[0.9]
        S, L = hot & (close < open24), hot & (close > open24)
        sigs["H5"] = (fresh(S) | fresh(L), np.where(L, 1, -1))

    ex = Exec(f5, load_funding(BASE / "futures/funding.csv"))
    end = float(min(tc[np.isfinite(spot_c)].max(), tc[np.isfinite(dv)].max(), f5["t"][-1] - DAY))
    hold = (DISC[1], end)
    cov = {k: {"first": datetime.fromtimestamp(float(tc[np.isfinite(v)].min()) / 1000, timezone.utc).isoformat(),
               "last": datetime.fromtimestamp(float(tc[np.isfinite(v)].max()) / 1000, timezone.utc).isoformat(),
               "nan_share_2021_04on": round(float(np.isnan(v[tc >= DISC[0]]).mean()), 4)}
           for k, v in (("cp", cp), ("kp", kp), ("dvol", dv), ("spot", spot_c))}
    rep: dict = {"coverage": cov, "end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(), "cells": {}}
    pvals = {}
    for hk, (sig, side) in sigs.items():
        for y in ("Y1", "Y2"):
            k = f"{hk}-{y}"
            dtr = run_cell(ex, tc, sig, side, y, DISC[0], DISC[1])
            htr = run_cell(ex, tc, sig, side, y, *hold)
            ds = summ(dtr)
            p = 0.5 * math.erfc(ds["t"] / math.sqrt(2)) if ds.get("n", 0) >= 40 else 1.0
            pvals[k] = p
            rep["cells"][k] = {"disc": ds, "disc_p": round(p, 4), "hold": summ(htr), "hold_c10": summ(htr, 0.001),
                               "signals_disc": int(sig[(tc >= DISC[0]) & (tc < DISC[1])].sum()),
                               "signals_hold": int(sig[(tc >= hold[0]) & (tc < hold[1])].sum()),
                               "hold_trades": [[z[0], z[1], round(z[2], 5), z[3], z[4]] for z in htr]}
    # BH
    ks = sorted(pvals, key=pvals.get)
    m_ = len(ks)
    surv = []
    for r_, k in enumerate(ks, 1):
        if pvals[k] <= 0.10 * r_ / m_:
            surv = ks[:r_]
    rep["bh_survivors"] = surv
    rng = np.random.default_rng(7)
    hmask = (tc >= hold[0]) & (tc < hold[1])
    hi = np.flatnonzero(hmask)
    for k in surv:
        hk, y = k.split("-")
        sig, side = sigs[hk]
        cell = rep["cells"][k]
        act = cell["hold"].get("ev", -9)
        null = []
        for _ in range(a.null):
            sh = int(rng.integers(24 * 30, len(hi) - 24 * 30))
            s2, d2 = sig.copy(), side.copy()
            s2[hi], d2[hi] = np.roll(sig[hi], sh), np.roll(side[hi], sh)
            null.append(summ(run_cell(ex, tc, s2, d2, y, *hold)).get("ev", 0.0))
        null = np.array(null)
        hs = cell["hold"]
        cell["null_ev_mean"] = round(float(null.mean()), 3)
        cell["null_p"] = round(float((null >= act).mean()), 4)
        cell["pass"] = bool(hs.get("n", 0) >= 25 and hs["ev"] > 0 and (hs.get("long_ev") or 0) >= 0
                            and (hs.get("short_ev") or 0) >= 0 and cell["hold_c10"]["ev"] > 0
                            and cell["null_p"] < 0.05)
    np.savez_compressed(out / "series.npz", tc=tc, cp=cp, kp=kp, dv=dv, spread=spread, close=close)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({"coverage": cov, "bh": surv,
                      "cells": {k: {"disc": v["disc"], "p": v["disc_p"], "hold": v["hold"]}
                                for k, v in rep["cells"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
