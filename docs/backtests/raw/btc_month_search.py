"""BTCUSDT 15분봉 한 달 탐색 (사전 등록: docs/backtests/2026-11-01-btc-monthly-search-hypotheses.md).

numpy가 필요하다. 실행: PYTHONPATH=src:<numpy 경로> python docs/backtests/raw/btc_month_search.py
주문과 연결되지 않는다.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import urllib.request
from datetime import datetime, timezone

import numpy as np

from minerva.strategy.btc_data import OUT
from minerva.strategy.indicators import Bar, cci, rsi, stoch_fast_k

STOP = 0.01
COST = 0.0005
DAILY = 5
FIRST = 400
ROUNDS = 6
KLINES = (
    "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT"
    "&interval=15m&limit=1000&startTime={}"
)
GRID = {
    "kind": ["os", "bb", "brk"],
    "conf": ["touch", "back"],
    "stoch_n": [5, 9, 14],
    "rsi_n": [7, 14],
    "cci_n": [14, 20, 30],
    "s_lv": [10, 15, 20, 30],
    "r_lv": [20, 25, 30, 35, 40, 50],
    "c_lv": [50, 100, 150, 200],
    "bb_n": [20, 40],
    "bb_k": [1.5, 2.0, 2.5, 3.0],
    "brk_n": [8, 16, 32, 48, 96, 192],
    "trend": ["none", "with", "against"],
    "atr": ["any", "low", "high"],
    "hours": ["all", "asia", "eu", "us"],
    "tp": [0.003, 0.004, 0.005, 0.006, 0.0075, 0.01, 0.015],
    "hold": [4, 8, 16, 32, 96],
}
HOURS = {"asia": range(0, 8), "eu": range(8, 14), "us": range(14, 22)}


def load() -> dict:
    since = int(datetime(2025, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows: dict[int, list[float]] = {}
    with (OUT / "15m.csv").open() as f:
        for r in csv.reader(f):
            if r[0] != "open_time" and int(r[0]) >= since:
                rows[int(r[0])] = [float(x) for x in r[1:6]]
    cur = max(rows) + 1
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    while cur < now:
        with urllib.request.urlopen(KLINES.format(cur), timeout=20) as res:
            page = json.load(res)
        if not page:
            break
        for p in page:
            rows[int(p[0])] = [float(x) for x in p[1:6]]
        cur = page[-1][0] + 1
        if len(page) < 1000:
            break
    ts = [t for t in sorted(rows) if t + 900000 <= now]
    t = [datetime.fromtimestamp(x / 1000, timezone.utc) for x in ts]
    bars = [Bar(x.date(), *rows[k]) for x, k in zip(t, ts)]
    arr = lambda k: np.array([getattr(b, k) for b in bars])  # noqa: E731
    d = {k: arr(k) for k in ("open", "high", "low", "close")}
    d.update(t=t, bars=bars)
    closes = list(d["close"])
    nan = lambda s: np.array([np.nan if v is None else v for v in s], float)  # noqa: E731
    d["stoch"] = {n: nan(stoch_fast_k(bars, n)) for n in GRID["stoch_n"]}
    d["rsi"] = {n: nan(rsi(closes, n)) for n in GRID["rsi_n"]}
    d["cci"] = {n: nan(cci(bars, n)) for n in GRID["cci_n"]}
    c = d["close"]
    d["bb"] = {}
    for n in GRID["bb_n"]:
        w = np.lib.stride_tricks.sliding_window_view(c, n)
        z = np.full(len(c), np.nan)
        z[n - 1 :] = (c[n - 1 :] - w.mean(1)) / (w.std(1) + 1e-9)
        d["bb"][n] = z
    d["hi"], d["lo"] = {}, {}
    for n in GRID["brk_n"]:
        hi, lo = np.full(len(c), np.nan), np.full(len(c), np.nan)
        w = np.lib.stride_tricks.sliding_window_view(c, n)
        hi[n:] = w.max(1)[:-1]
        lo[n:] = w.min(1)[:-1]
        d["hi"][n], d["lo"][n] = hi, lo
    ema, k = [], 2 / 201
    v = c[0]
    for x in c:
        v = x * k + v * (1 - k)
        ema.append(v)
    d["ema"] = np.array(ema)
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([d["high"] - d["low"], abs(d["high"] - prev), abs(d["low"] - prev)])
    atr = np.convolve(tr, np.ones(14) / 14, "full")[: len(c)] / c
    # 롤링 30일 중앙값 대비로 잘라 미래 값을 쓰지 않는다
    med = np.full(len(c), np.nan)
    for i in range(2880, len(c), 96):
        med[i : i + 96] = np.median(atr[i - 2880 : i])
    d["atr_high"] = atr > med
    d["hour"] = np.array([x.hour for x in t])
    d["day"] = np.array([x.toordinal() for x in t])
    d["ym"] = np.array([x.year * 100 + x.month for x in t])
    return d


def flags(d, g):
    """(롱 신호, 숏 신호) 봉 마감 기준."""
    c = d["close"]
    if g["kind"] == "os":
        s, r, q = d["stoch"][g["stoch_n"]], d["rsi"][g["rsi_n"]], d["cci"][g["cci_n"]]
        lo = (s < g["s_lv"]) & (r < g["r_lv"]) & (q < -g["c_lv"])
        up = (s > 100 - g["s_lv"]) & (r > 100 - g["r_lv"]) & (q > g["c_lv"])
    elif g["kind"] == "bb":
        z = d["bb"][g["bb_n"]]
        lo, up = z < -g["bb_k"], z > g["bb_k"]
    else:
        up, lo = c > d["hi"][g["brk_n"]], c < d["lo"][g["brk_n"]]
    if g["kind"] == "brk":
        lng, sht = up, lo
    else:
        lng, sht = lo, up
    pv = lambda x: np.r_[False, x[:-1]]  # noqa: E731
    if g["conf"] == "touch":
        lng, sht = lng & ~pv(lng), sht & ~pv(sht)
    else:
        lng, sht = pv(lng) & ~lng, pv(sht) & ~sht
    above = c > d["ema"]
    if g["trend"] == "with":
        lng, sht = lng & above, sht & ~above
    elif g["trend"] == "against":
        lng, sht = lng & ~above, sht & above
    if g["atr"] != "any":
        ok = d["atr_high"] if g["atr"] == "high" else ~d["atr_high"]
        lng, sht = lng & ok, sht & ok
    if g["hours"] != "all":
        ok = np.isin(d["hour"], HOURS[g["hours"]])
        lng, sht = lng & ok, sht & ok
    return lng, sht


def sim(d, g, a, z, entry_slip=0.0, stop_slip=0.0):
    """[a, z) 구간 수익률 목록. 포지션은 그 안에서 끝낸다."""
    o, h, l = d["open"], d["high"], d["low"]  # noqa: E741
    lng, sht = flags(d, g)
    idx = np.nonzero((lng | sht)[a : z - 1])[0] + a
    tp, hold = g["tp"], g["hold"]
    out = []
    free, today, count = -1, -1, 0
    for i in idx.tolist():
        e = i + 1
        if e <= free or lng[i] == sht[i]:
            continue
        if d["day"][e] != today:
            today, count = d["day"][e], 0
        if count >= DAILY:
            continue
        side = 1 if lng[i] else -1
        px0 = o[e]
        x = min(e + hold, z - 1)
        if x <= e:
            continue
        up, dn = (px0 * (1 + tp), px0 * (1 - STOP)) if side == 1 else (
            px0 * (1 + STOP),
            px0 * (1 - tp),
        )
        hu, hd = h[e:x] >= up, l[e:x] <= dn
        hit = hu | hd
        px, last = o[x], x - 1
        if hit.any():
            j = int(hit.argmax())
            last = e + j
            win = (hu[j] and not hd[j]) if side == 1 else (hd[j] and not hu[j])
            if win:
                px = up if side == 1 else dn
            elif side == 1:
                px = min(o[last], dn) * (1 - stop_slip)
            else:
                px = max(o[last], up) * (1 + stop_slip)
        fill = px0 * (1 + side * entry_slip)
        out.append(side * (px / fill - 1) - COST - (px / fill) * COST)
        free, count = last, count + 1
    return out


def summary(rets):
    if not rets:
        return {"n": 0, "cum": 0.0, "win": 0.0}
    a = np.array(rets)
    return {"n": len(a), "cum": float(np.prod(1 + a) - 1), "win": float((a > 0).mean())}


SPAN = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def legal(g, m):
    return m["n"] >= 20 * SPAN and m["win"] >= 0.60


def key(g):
    return tuple(sorted(g.items()))


def search(d, a, z, rng):
    seen, base = {}, []
    for _ in range(FIRST * 15):
        g = {k: rng.choice(v) for k, v in GRID.items()}
        m = summary(sim(d, g, a, z))
        seen[key(g)] = (g, m)
        base.append(g)
    best, tries = -9.0, 0
    for tries in range(1, ROUNDS + 1):
        pool = [x for x in seen.values() if legal(*x)]
        pool.sort(key=lambda x: -x[1]["cum"])
        top = pool[:30]
        if not top or top[0][1]["cum"] <= best + 0.002:
            break
        best = top[0][1]["cum"]
        for g, _ in top:
            for k, vals in GRID.items():
                for v in vals:
                    h = {**g, k: v}
                    if key(h) not in seen:
                        seen[key(h)] = (h, summary(sim(d, h, a, z)))
    pool = [x for x in seen.values() if legal(*x)]
    pool.sort(key=lambda x: -x[1]["cum"])
    return pool[:20], base[:6000], tries, len(seen)


def month_span(d, ym):
    m = np.nonzero(d["ym"] == ym)[0]
    return int(m[0]), int(m[-1]) + 1


def evaluate(d, tr, te, rng):
    a = month_span(d, tr[0])[0]
    z = month_span(d, tr[-1])[1]
    top, base, rounds, tried = search(d, a, z, rng)
    a2, z2 = month_span(d, te)
    tests = [summary(sim(d, g, a2, z2)) for g, _ in top]
    rand = [summary(sim(d, g, a2, z2))["cum"] for g in base]
    res = {
        "train": tr,
        "test": te,
        "rounds": rounds,
        "tried": tried,
        "train_best": top[0][1] if top else None,
        "best": top[0][0] if top else None,
        "test_best": tests[0] if tests else None,
        "test_best_slip": summary(sim(d, top[0][0], a2, z2, 0.0002, 0.001)) if top else None,
        "top20_median": statistics.median(t["cum"] for t in tests) if tests else None,
        "top20_positive": sum(t["cum"] > 0 for t in tests),
        "random_median": statistics.median(rand),
        "hold": float(d["close"][z2 - 1] / d["open"][a2] - 1),
    }
    print(json.dumps(res, ensure_ascii=False, default=str), flush=True)
    return res


def main() -> None:
    d = load()
    print("bars", len(d["close"]), d["t"][-1], flush=True)
    rng = random.Random(11)
    months = [202512 + 0] + [202600 + m for m in range(1, 11)]
    out = []
    for i in range(len(months) - 1, SPAN - 1, -1):  # 최근 달부터
        out.append(evaluate(d, months[i - SPAN : i], months[i], rng))
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
