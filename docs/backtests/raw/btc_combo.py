"""BTCUSDT 전체 조합 회귀. 사전 등록: docs/backtests/2026-11-01-btc-combo-hypotheses.md

numpy가 필요하다. 실행: PYTHONPATH=src:<numpy 경로>:docs/backtests/raw python docs/backtests/raw/btc_combo.py <결과.json>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import csv
import json
import pickle
import random
import sys
from datetime import datetime, timezone

import btc_mit as mit
import btc_regress as br
import btc_swing_label as sl
import numpy as np

from minerva.strategy import btc_bt as bt
from minerva.strategy import btc_features as bf
from minerva.strategy.btc_data import OUT
from minerva.strategy.indicators import cci, dmi, rsi, stoch_fast_k, tm_states

TARGETS = (0.01, 0.02, 0.03)
TOP = 0.98
MARGIN = 0.001
SHIFTS = 300
YEARS = range(2020, 2027)
PER_DAY = {"1h": 24, "15m": 96}


def nanarr(x) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in x], float)


def swing_features(bars, atr):
    ev = sl.confirmed(bars)
    n = len(bars)
    t_hi, t_lo = np.zeros(n), np.zeros(n)
    d_hi, d_lo = np.full(n, np.nan), np.full(n, np.nan)
    k, hi, lo, a, b = 0, None, None, 0.0, 0.0
    for i in range(n):
        while k < len(ev) and ev[k][0] <= i:
            _, s, text, price = ev[k]
            if s == 1:
                hi, a = price, 1.0 if text == "HH" else -1.0
            else:
                lo, b = price, 1.0 if text == "HL" else -1.0
            k += 1
        t_hi[i], t_lo[i] = a, b
        c = bars[i].close
        if hi is not None and atr[i] == atr[i] and atr[i] > 0:
            d_hi[i] = (hi - c) / atr[i]
        if lo is not None and atr[i] == atr[i] and atr[i] > 0:
            d_lo[i] = (c - lo) / atr[i]
    return t_hi, t_lo, d_hi, d_lo


def far_ends(c: np.ndarray, look: int):
    w = np.lib.stride_tricks.sliding_window_view(c, look)
    hi, lo = np.full(len(c), np.nan), np.full(len(c), np.nan)
    hi[look:] = w.max(1)[:-1]
    lo[look:] = w.min(1)[:-1]
    return hi / c - 1, c / lo - 1


def build(frame: str):
    bars = bt.load(frame)
    with (OUT / f"{frame}.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(bars)
    taker = [float(r["taker_buy_base"]) for r in rows]
    ts = [int(r["open_time"]) for r in rows]
    closes = [b.close for b in bars]
    c = np.array(closes)
    vols = [b.volume for b in bars]
    feat: dict[str, np.ndarray] = {}
    feat["Stoch5"] = nanarr(stoch_fast_k(bars, 5))
    for n in (20, 30):
        feat[f"CCI{n}"] = nanarr(cci(bars, n))
    for n in (7, 14):
        feat[f"RSI{n}"] = nanarr(rsi(closes, n))
    st = tm_states(bars)
    feat["SCR롱"] = np.array([s.long_level for s in st], float)
    feat["SCR숏"] = np.array([s.short_level for s in st], float)
    feat["ADX"] = nanarr(dmi(bars, 14)[2])
    tr = nanarr(bf.rolling_mean(bf.true_range(bars), 14))
    feat["ATR비율"] = tr / c
    vol20 = nanarr(bf.rolling_mean(vols, 20))
    feat["거래량비율"] = np.array(vols) / np.where(vol20 > 0, vol20, np.nan)
    tk, vs = np.array(bf.rolling_sum(taker)), np.array(bf.rolling_sum(vols))
    feat["체결방향"] = tk / np.where(vs > 0, vs, np.nan) - 0.5
    feat["%B"] = nanarr(bf.percent_b(closes))
    feat["EMA200거리"] = c / np.array(bf.ema(closes, 200)) - 1
    four = bt.load("4h", OUT)
    e4 = bf.ema([b.close for b in four], 50)
    dist4 = [b.close / e - 1 for b, e in zip(four, e4)]
    feat["4hEMA50거리"] = nanarr(
        bt.align(bt.close_times(frame, OUT), bt.close_times("4h", OUT), dist4)
    )
    for n in (4, 16, 96):
        r = np.full(len(c), np.nan)
        r[n:] = c[n:] / c[:-n] - 1
        feat[f"수익률{n}"] = r
    hour = np.array([datetime.fromtimestamp(t / 1000, timezone.utc).hour for t in ts], float)
    feat["sin"], feat["cos"] = np.sin(hour / 24 * 2 * np.pi), np.cos(hour / 24 * 2 * np.pi)
    a, b, d1, d2 = swing_features(bars, tr)
    feat["고점구조"], feat["저점구조"], feat["고점거리"], feat["저점거리"] = a, b, d1, d2
    lead_l, lead_s = mit.range_filter(c)
    rf = lead_l.astype(float) - lead_s.astype(float)
    feat["RF방향"] = rf
    feat["RF연속"] = np.minimum(mit.runs(lead_l), 10.0) - np.minimum(mit.runs(lead_s), 10.0)
    up, down = mit.rqk(c)
    feat["RQK"] = up.astype(float) - down.astype(float)
    hi, lo = far_ends(c, 20 * PER_DAY[frame])
    feat["20일고거리"], feat["20일저거리"] = hi, lo
    names = list(feat)
    X = np.c_[[feat[k] for k in names]].T
    return bars, X, names, ts


def run_frame(frame: str, report: dict) -> None:
    bars, X, names, ts = build(frame)
    br.MAX_BARS = 500 if frame == "1h" else 2000
    ok = np.isfinite(X).all(axis=1)
    days_of = [b.day for b in bars]
    years_of = np.array([d.year for d in days_of])
    bad = np.array([br.BAD[0] <= d <= br.BAD[1] for d in days_of])
    o = np.array([b.open for b in bars])
    h = np.array([b.high for b in bars])
    l = np.array([b.low for b in bars])  # noqa: E741
    print(frame, X.shape, "ok", int(ok.sum()), flush=True)
    s20 = int(np.argmax(years_of == 2020))
    s25 = int(np.argmax(years_of == 2025))
    days = (days_of[-1] - days_of[s20]).days
    days25 = (days_of[-1] - days_of[s25]).days
    n = len(o)
    br.STOP = 0.01
    for tp in TARGETS:
        res = {s: br.outcomes(o, h, l, tp, s) for s in (1, -1)}
        keyed = {(tp, s): res[s] for s in (1, -1)}
        probs = {s: np.zeros(n) for s in (1, -1)}
        cuts = {}
        coef = {}
        for side in (1, -1):
            y = res[side][0].astype(float)
            for yr in YEARS:
                t0 = int(np.argmax(years_of == yr))
                train = np.zeros(n, bool)
                train[: t0 - br.MAX_BARS] = True
                train &= ok & ~bad & (y >= 0)
                mu, sd = X[train].mean(0), X[train].std(0) + 1e-12

                def design(m, mu=mu, sd=sd):
                    return np.c_[np.ones(int(m.sum())), np.clip((X[m] - mu) / sd, -5, 5)]

                beta = br.fit(design(train), y[train])
                inner = br.predict(design(train), beta)
                cuts[(side, yr)] = float(np.quantile(inner, TOP))
                mask = (years_of == yr) & ok
                probs[side][mask] = br.predict(design(mask), beta)
                coef[(side, yr)] = beta[1:]
        for rule in ("T", "E"):
            sig = []
            for i in np.nonzero(ok & (years_of >= YEARS[0]))[0][:-1]:
                yr = int(years_of[i])
                pl, ps = probs[1][i], probs[-1][i]
                if rule == "T":
                    tl, tsh = cuts[(1, yr)], cuts[(-1, yr)]
                else:
                    tl = tsh = (br.STOP + 2 * br.COST + MARGIN) / (br.STOP + tp)
                hit_l, hit_s = pl >= tl, ps >= tsh
                if hit_l and (not hit_s or pl >= ps):
                    sig.append((int(i), 1, tp))
                elif hit_s:
                    sig.append((int(i), -1, tp))
            trades = br.trade(sig, keyed, br.COST, o)
            got = br.score(trades, years_of, days)
            key = f"{frame}/{tp}/{rule}"
            if not trades:
                print(key, "no trades", flush=True)
                report[key] = got
                continue
            t10 = br.trade(sig, keyed, 0.001, o)
            slip = [(e, r - 0.0004 - (0.001 if r < 0 else 0), w) for e, r, w in trades]
            recent = [x for x in sig if x[0] >= s25]
            rng = random.Random(7)
            m, vals = n - s20, []
            for _ in range(SHIFTS):
                d = rng.randrange(1000, m - 1000)
                moved = sorted(((i - s20 + d) % m + s20, s, t) for i, s, t in sig)
                moved = [x for x in moved if x[0] < n - 2]
                vals.append(br.score(br.trade(moved, keyed, br.COST, o), years_of, days)["final"])
            p = (sum(v >= got["final"] for v in vals) + 1) / (SHIFTS + 1)
            got.update(
                signals=len(sig),
                cost10=br.score(t10, years_of, days)["cagr"],
                slip=br.score(slip, years_of, days)["cagr"],
                recent=br.score(br.trade(recent, keyed, br.COST, o), years_of, days25).get("final"),
                p=p,
                longs=sum(s == 1 for _, s, _ in sig),
            )
            report[key] = got
            print(
                key,
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in got.items()},
                flush=True,
            )
            pickle.dump(
                {"sig": sig, "tp": tp, "trades": trades, "ts": ts},
                open(sys.argv[1] + f".{frame}.{tp}.{rule}.pkl", "wb"),
            )
        avg = {(s): np.mean([coef[(s, y)] for y in YEARS], axis=0) for s in (1, -1)}
        for s, v in avg.items():
            top = sorted(zip(names, v), key=lambda q: -abs(q[1]))[:8]
            print(" coef", frame, tp, s, [(a, round(float(b), 3)) for a, b in top], flush=True)


def main() -> None:
    report: dict = {}
    for frame in ("1h", "15m"):
        run_frame(frame, report)
    with open(sys.argv[1], "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
