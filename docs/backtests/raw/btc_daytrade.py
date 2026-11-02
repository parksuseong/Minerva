"""데이트레이딩 지표 조합 7개, 손절 없음. 등록: 2026-11-02-btc-daytrade-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_daytrade.py --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import btc_scr_ma as B

B.STOP = 10.0  # 손절 없음: 닿을 수 없는 가격
DAY = 48
WINDOWS = {"30d": 30 * DAY, "6m": 183 * DAY}
STRATS = ("A", "B", "C", "D", "A4", "B4", "D4")


def load(path: Path, offset_min: int = 0, merge: bool = False) -> dict:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append((int(r["open_time"]), float(r["open"]), float(r["high"]),
                         float(r["low"]), float(r["close"]), float(r["volume"])))
    rows = rows[-40_000 * (2 if merge else 1):]
    if merge:
        g: dict = {}
        for t, o, h, l, c, v in rows:
            k = (t - offset_min * 60_000) // 1_800_000
            x = g.get(k)
            if x is None:
                g[k] = [t, o, h, l, c, v, 1]
            else:
                x[2], x[3], x[4], x[5], x[6] = max(x[2], h), min(x[3], l), c, x[5] + v, x[6] + 1
        rows = [tuple(v[:6]) for _, v in sorted(g.items()) if v[6] == 2]
    a = np.array(rows, dtype=float)
    return {"t": a[:, 0], "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": a[:, 5]}


def prev(x: np.ndarray, k: int = 1) -> np.ndarray:
    return np.r_[np.full(k, np.nan), x[:-k]]


def roll(x: np.ndarray, n: int, fn) -> np.ndarray:
    out = np.full(len(x), np.nan)
    w = np.lib.stride_tricks.sliding_window_view(x, n)
    out[n - 1:] = fn(w, axis=1)
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return B.ema(x, n)


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in B.rsi(c.tolist(), n)], dtype=float)


def any_last(x: np.ndarray, k: int) -> np.ndarray:
    out = x.copy()
    for j in range(1, k):
        out |= np.r_[np.zeros(j, bool), x[:-j]]
    return out


def h4_trend(d: dict, path: Path) -> tuple:
    """4시간봉 EMA20 대 EMA50, 30분봉 마감 때 이미 마감된 4시간봉."""
    with path.open() as f:
        rows = [(int(r["open_time"]), float(r["close"])) for r in csv.DictReader(f)]
    t4 = np.array([r[0] for r in rows], dtype=float) + 4 * 3_600_000
    c4 = np.array([r[1] for r in rows])
    up4, dn4 = ema(c4, 20) > ema(c4, 50), ema(c4, 20) < ema(c4, 50)
    j = np.searchsorted(t4, d["t"] + 1_800_000, side="right") - 1
    ok = j >= 0
    up = np.zeros(len(d["t"]), bool)
    dn = up.copy()
    up[ok], dn[ok] = up4[j[ok]], dn4[j[ok]]
    return up, dn


def signals(d: dict, h4: tuple) -> dict:
    """전략별 ((롱 신호, 숏 신호), (롱 허용, 숏 허용, 롱 청산, 숏 청산), 청산 종류)."""
    o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
    n = len(c)
    mid = roll(c, 20, np.mean)
    sd = roll(c, 20, np.std)
    up, lo = mid + 2 * sd, mid - 2 * sd
    hh, ll = roll(h, 5, np.max), roll(l, 5, np.min)
    rng = hh - ll
    k = np.where(rng > 0, (c - ll) / np.where(rng > 0, rng, 1) * 100, 50.0)
    dd = roll(k, 3, np.mean)
    kp, dp = prev(k), prev(dd)
    gold = (k > dd) & (kp <= dp) & (np.fmin(k, kp) < 20)
    dead = (k < dd) & (kp >= dp) & (np.fmax(k, kp) > 80)
    a_l = any_last(l <= lo, 3) & gold
    a_s = any_last(h >= up, 3) & dead

    e12, e26 = ema(c, 12), ema(c, 26)
    r = rsi(c)
    rp = prev(r)
    t_up, t_dn = e12 > e26, e12 < e26
    b_l = t_up & (rp <= 40) & (r > 40)
    b_s = t_dn & (rp >= 60) & (r < 60)
    flip_dn = t_dn & prev(e12 > e26).astype(bool)
    flip_up = t_up & prev(e12 < e26).astype(bool)

    width = (up - lo) / mid
    q20 = np.full(n, np.nan)
    w = np.lib.stride_tricks.sliding_window_view(width, 120)
    q20[120:] = np.nanpercentile(w[:-1], 20, axis=1)
    sq = width <= q20
    sq5 = np.zeros(n, bool)
    for j in range(1, 6):
        sq5 |= np.r_[np.zeros(j, bool), sq[:-j]]
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(np.nan_to_num(macd), 9)
    hp = prev(hist)
    c_l = sq5 & (c > up) & (prev(c) <= prev(up)) & (hist > 0) & (hist > hp)
    c_s = sq5 & (c < lo) & (prev(c) >= prev(lo)) & (hist < 0) & (hist < hp)
    below_mid = (c < mid) & (prev(c) >= prev(mid))
    above_mid = (c > mid) & (prev(c) <= prev(mid))

    sup = prev(roll(l, DAY, np.min))
    res = prev(roll(h, DAY, np.max))
    vavg = prev(roll(v, DAY, np.mean))
    spike = v >= 2 * vavg
    d_l = (l <= sup * 1.001) & (c > o) & (c > sup) & spike
    d_s = (h >= res * 0.999) & (c < o) & (c < res) & spike

    ok = np.ones(n, bool)
    no = np.zeros(n, bool)
    u4, d4 = h4
    clean = lambda x: np.nan_to_num(x.astype(float)).astype(bool)  # noqa: E731
    base = {
        "A": ((a_l, a_s), (ok, ok, no, no), "O"),
        "B": ((b_l, b_s), (ok, ok, flip_dn, flip_up), "M"),
        "C": ((c_l, c_s), (ok, ok, below_mid, above_mid), "M"),
        "D": ((d_l, d_s), (ok, ok, no, no), "O"),
    }
    out = {}
    for name, ((sl, ss), (pl, ps, xl, xs), ex) in base.items():
        out[name] = ((clean(sl), clean(ss)), (pl, ps, clean(xl), clean(xs)), ex)
    for name in ("A", "B", "D"):
        sig, (pl, ps, xl, xs), ex = out[name]
        out[name + "4"] = (sig, (u4, d4, xl, xs), ex)
    return out


def summary(tr: tuple, a: int, b: int, cost: float, d: dict) -> dict:
    ei, xi, sd, gr = tr
    k = (ei >= a) & (ei < b)
    s = B.stats(tr, a, b, cost)
    if not k.any():
        return s
    ei, xi, sd = ei[k], xi[k], sd[k]
    mae = []
    for e, x, side in zip(ei, xi, sd):
        seg_l, seg_h = d["l"][e:x + 1], d["h"][e:x + 1]
        worst = seg_l.min() / d["o"][e] - 1 if side > 0 else 1 - seg_h.max() / d["o"][e]
        mae.append(worst)
    s["mae_worst"] = round(float(min(mae)) * 100, 2)
    s["hold_med_h"] = round(float(np.median(xi - ei)) / 2, 1)
    s["hold_max_h"] = round(float((xi - ei).max()) / 2, 1)
    s["open_at_end"] = bool(xi[-1] >= b - 1)
    return s


def evaluate(d: dict, sigs: dict, win: int) -> dict:
    n = len(d["c"])
    a = n - win
    out = {}
    for name in STRATS:
        sig, perm, ex = sigs[name]
        tr = B.run(d, sig, perm, ex, a, n)
        out[name] = {"c05": summary(tr, a, n, 0.0005, d), "c10": B.stats(tr, a, n, 0.001)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    d = load(data / "30m.csv")
    sigs = signals(d, h4_trend(d, data / "4h.csv"))
    rep = {"last_bar": int(d["t"][-1]), "30d": evaluate(d, sigs, WINDOWS["30d"])}
    passed = [s for s in STRATS
              if rep["30d"][s]["c05"]["n"] >= 10 and rep["30d"][s]["c05"]["win"] > 70
              and rep["30d"][s]["c05"]["ret"] > 0]
    rep["pass30"] = passed
    if passed:
        six = evaluate(d, sigs, WINDOWS["6m"])
        rep["6m"] = {s: six[s] for s in passed}
        n = len(d["c"])
        rep["6m_before30"] = {}
        for s in passed:
            sig, perm, ex = sigs[s]
            a = n - WINDOWS["6m"]
            tr = B.run(d, sig, perm, ex, a, n - WINDOWS["30d"])
            rep["6m_before30"][s] = summary(tr, a, n - WINDOWS["30d"], 0.0005, d)
        d2 = load(data / "15m.csv", offset_min=15, merge=True)
        s2 = signals(d2, h4_trend(d2, data / "4h.csv"))
        sh = evaluate(d2, s2, WINDOWS["6m"])
        rep["6m_shift"] = {s: sh[s]["c05"] for s in passed}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("pass30", passed, flush=True)


if __name__ == "__main__":
    main()
