"""외부 글의 XGBoost 일봉 추세 분류를 그대로 옮겨 매매로 평가. 등록: 2026-11-03-btc-xgb-trend-hypotheses.md

실행: PYTHONPATH=<numpy·xgboost 경로>:docs/backtests/raw python docs/backtests/raw/btc_xgb_trend.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

L = 60
COST = 0.0005
LIQ = 0.32
CUT_A = datetime(2022, 9, 27, tzinfo=timezone.utc).timestamp() * 1000


def load_daily(path: Path) -> dict:
    rows = {}
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            rows[int(x[0])] = [float(v) for v in x[1:6]]
    t = np.array(sorted(rows), dtype=float)
    a = np.array([rows[k] for k in sorted(rows)])
    return {"t": t, "o": a[:, 0], "h": a[:, 1], "l": a[:, 2], "c": a[:, 3], "v": a[:, 4]}


# ---- 글의 라벨링 (DataLaeling) ----
def find_max(c, idx):
    return not any(c[i] > c[idx] for i in range(idx + 1, idx + L + 1))


def find_min(c, idx):
    return not any(c[i] < c[idx] for i in range(idx + 1, idx + L + 1))


def labeling(c: np.ndarray) -> tuple:
    n = len(c)
    first = list(c[:L])
    max_num, min_num = first.index(max(first)), first.index(min(first))
    if max_num > min_num:
        ck, start = 1, max_num
        if not find_max(c, max_num):
            ck, start = 0, min_num
    else:
        ck, start = 0, min_num
        if not find_min(c, min_num):
            ck, start = 1, max_num
    highs, lows = [], []
    prev_max = prev_min = None
    if ck == 0:
        lows.append(start - 1)
        mx = c[start - 1]
        prev_min = c[start - 1]
        mn = 999999999.0
    else:
        highs.append(start - 1)
        prev_max = c[start - 1]
        mn = c[start - 1]
        mx = 0.0
    temp_index = 0
    for i in range(start, n - L):
        if ck == 0:
            if c[i] > mx and c[i] >= prev_min:
                if find_max(c, i):
                    highs.append(i)
                    prev_max, mx, ck, temp_index = c[i], 0.0, 1, i
                else:
                    mx = c[i]
        else:
            if c[i] < mn and c[i] <= prev_max:
                if find_min(c, i):
                    lows.append(i)
                    prev_min, mn, ck, temp_index = c[i], 999999999.0, 0, i
                else:
                    mn = c[i]
    last = -1
    for i in range(temp_index, n):
        if c[i] == mx or c[i] == mn:
            last = i
        elif c[i] > mx and ck == 0:
            last, mx = i, c[i]
        elif c[i] < mn and ck == 1:
            last, mn = i, c[i]
    (highs if ck == 0 else lows).append(last % n)
    mark = np.zeros(n, dtype=int)
    mark[lows] = -1
    mark[highs] = 1  # transpose는 고점을 먼저 본다
    pts = list(np.flatnonzero(mark != 0))
    ang = []
    for a, b in zip(pts[:-1], pts[1:]):
        hgt, wid = abs(c[a] - c[b]), abs(a - b)
        ang.append(math.asin(wid / math.sqrt(hgt * hgt + wid * wid)))
    M = int(len(sorted(ang)) * 7 / 10)  # 글 그대로: 각도를 순위 정수와 비교
    lab = np.zeros(n, dtype=int)
    for (a, b), test in zip(zip(pts[:-1], pts[1:]), ang):
        if c[a] < c[b] and test < M:
            lab[a:b] = 0
        elif c[a] > c[b] and test < M:
            lab[a:b] = 1
        else:
            lab[a:b] = 2
    return lab, pts[0], pts[-1]


# ---- 글의 지표 (pandas_ta 기본값을 numpy로) ----
def ewm(x, alpha, adjust=True, min_periods=0):
    out = np.full(len(x), np.nan)
    num = den = 0.0
    cnt = 0
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        cnt += 1
        if adjust:
            num = v + (1 - alpha) * num
            den = 1 + (1 - alpha) * den
            cur = num / den
        else:
            cur = v if np.isnan(prev) else alpha * v + (1 - alpha) * prev
            prev = cur
        if cnt >= max(min_periods, 1):
            out[i] = cur
    return out


def sma(x, n):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.r_[0.0, x])
    out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def std(x, n):
    out = np.full(len(x), np.nan)
    w = np.lib.stride_tricks.sliding_window_view(x, n)
    out[n - 1 :] = w.std(1, ddof=1)
    return out


def rsi(c, n=14):
    d = np.r_[np.nan, np.diff(c)]
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    up[0] = dn[0] = np.nan
    pu, pd_ = ewm(up, 1 / n, True, n), ewm(dn, 1 / n, True, n)
    return 100 * pu / (pu + pd_)


def ema_ta(c, n):
    x = c.copy()
    x[: n - 1] = np.nan
    x[n - 1] = c[:n].mean()
    out = ewm(x, 2 / (n + 1), adjust=False)
    out[: n - 1] = np.nan
    return out


def roll_min1(x, n, fn):
    return np.array([fn(x[max(0, i - n + 1) : i + 1]) for i in range(len(x))])


def features(d: dict) -> np.ndarray:
    o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]  # noqa: E741
    hh, ll = roll_min1(h, 14, np.max), roll_min1(l, 14, np.min)
    fk = (c - ll) / (hh - ll) * 100
    sk = ewm(fk, 2 / (5 + 1))
    sd = ewm(sk, 2 / (5 + 1))
    bbl = sma(c, 60) - 2 * std(c, 60)
    cols = [o, h, l, c, v, rsi(c), sma(c, 20), ema_ta(c, 30), sma(c, 60), ema_ta(c, 60),
            fk, sk, sk, sd, bbl, bbl, bbl, bbl, 100 * o / sma(c, 60)]
    return np.stack(cols, 1)


# ---- 매매로 옮기기 ----
def trades(full: dict, test_idx: np.ndarray, pos: np.ndarray) -> list:
    """test_idx는 full 기준 위치. pos[k]는 test_idx[k] 종가 뒤 다음 날 시가부터의 방향."""
    o, h, l = full["o"], full["h"], full["l"]  # noqa: E741
    n = len(o)
    out, k = [], 0
    while k < len(pos):
        if pos[k] == 0:
            k += 1
            continue
        k2 = k
        while k2 + 1 < len(pos) and pos[k2 + 1] == pos[k]:
            k2 += 1
        a, b = test_idx[k] + 1, min(test_idx[k2] + 2, n - 1)
        if a >= n:
            break
        sd, px = pos[k], o[a]
        g = (o[b] / px - 1) * sd
        adv = ((px - l[a:b].min()) / px if sd > 0 else (h[a:b].max() - px) / px) if b > a else 0.0
        out.append((sd, g, adv, int(full["t"][a]), int(full["t"][b])))
        k = k2 + 1
    return out


def summ(tr: list) -> dict:
    if not tr:
        return {"n": 0}
    g = np.array([x[1] for x in tr])
    adv = np.array([x[2] for x in tr])
    n1 = g - 2 * COST
    n3 = np.where(adv >= LIQ, -1.0, 3 * g - 6 * COST)
    eq = np.cumprod(1 + n3)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {"n": len(tr), "win3": round(float((n3 > 0).mean() * 100), 1),
            "ev3": round(float(n3.mean() * 100), 2), "x3": round(float((eq[-1] - 1) * 100), 1),
            "mdd3": round(float((eq / peak[1:] - 1).min() * 100), 1), "liq": int((adv >= LIQ).sum()),
            "x1": round(float((np.prod(1 + n1) - 1) * 100), 1),
            "long": int(sum(1 for x in tr if x[0] > 0))}


def run(full: dict, upto: float) -> dict:
    m = full["t"] < upto
    d = {k: v[m] for k, v in full.items()}
    lab, p0, p1 = labeling(d["c"])
    sl = slice(p0, p1)
    dd = {k: v[sl] for k, v in d.items()}
    X = features(dd)
    y = lab[sl]
    ok = ~np.isnan(X).any(1)
    X, y, idx = X[ok], y[ok], np.arange(p0, p1)[ok]
    xtr, ytr, xte, yte = X[:-500], y[:-500], X[-500:], y[-500:]
    classes = np.unique(ytr)
    enc = {c: i for i, c in enumerate(classes)}
    # XGBClassifier() 기본값과 같은 설정을 기본 API로 (sklearn 없이): 나무 100개, 나머지 기본
    ye = np.array([enc[c] for c in ytr])
    if len(classes) > 2:
        prm = {"objective": "multi:softprob", "num_class": len(classes)}
    else:
        prm = {"objective": "binary:logistic"}
    bst = xgb.train(prm, xgb.DMatrix(xtr, label=ye), num_boost_round=100)
    pr = bst.predict(xgb.DMatrix(xte))
    pred = classes[pr.argmax(1) if pr.ndim == 2 else (pr > 0.5).astype(int)]
    acc = float((pred == yte).mean())
    te_idx = idx[-500:]
    pos = np.where(pred == 0, 1, np.where(pred == 1, -1, 0))
    ma60 = sma(full["c"], 60)[te_idx]
    pos_ma = np.where(full["c"][te_idx] > ma60, 1, -1)
    hold = np.ones(len(te_idx), dtype=int)
    day = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    res = {"data": [day(d["t"][0]), day(d["t"][-1])], "labeled": [day(d["t"][p0]), day(d["t"][p1 - 1])],
           "test": [day(full["t"][te_idx[0]]), day(full["t"][te_idx[-1]])],
           "label_mix_test": {int(k): int((yte == k).sum()) for k in np.unique(yte)},
           "pred_mix": {int(k): int((pred == k).sum()) for k in np.unique(pred)},
           "accuracy": round(acc * 100, 1),
           "xgb": summ(trades(full, te_idx, pos)), "ma60": summ(trades(full, te_idx, pos_ma)),
           "hold": summ(trades(full, te_idx, hold))}
    return res


def main() -> None:
    full = load_daily(Path("~/.minerva/data/btc/1d.csv").expanduser())
    rep = {"A_2022-09": run(full, CUT_A), "B_now": run(full, full["t"][-1] + 1)}
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    for k, v in rep.items():
        print(k, json.dumps(v, ensure_ascii=False))
    if out:
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
