"""학습 가중치 모델에 붙이는 체결 흐름·선물 포지션 특징. 등록: 2026-11-06-btc-scr15-learned-flow-hypotheses.md"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BAR = 900_000
M5 = 300_000


def taker15(path: Path, d: dict, offset: int) -> tuple:
    """15분봉 d와 같은 묶음의 (거래량, 시장가 매수 체결량)."""
    t, v, tk = [], [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(int(row[0]))
            v.append(float(row[5]))
            tk.append(float(row[9]))
    t, v, tk = np.array(t, float), np.array(v), np.array(tk)
    key = ((t - offset * 60_000) // BAR).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    st, en = edges[:-1], edges[1:]
    full = (en - st) == 3
    st = st[full]
    tt = key[st].astype(float) * BAR + offset * 60_000
    vs, ks = np.add.reduceat(v, st), np.add.reduceat(tk, st)
    # reduceat은 다음 시작까지 더하므로 꽉 찬 3개만 다시 더한다
    vs = v[st] + v[st + 1] + v[st + 2]
    ks = tk[st] + tk[st + 1] + tk[st + 2]
    pos = {x: k for k, x in enumerate(tt)}
    idx = np.array([pos.get(x, -1) for x in d["t"]])
    V = np.where(idx >= 0, vs[np.maximum(idx, 0)], np.nan)
    K = np.where(idx >= 0, ks[np.maximum(idx, 0)], np.nan)
    return V, K


def roll(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.r_[0.0, np.nan_to_num(x)])
    out = np.full(len(x), np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def flow(V: np.ndarray, K: np.ndarray) -> np.ndarray:
    cols = []
    for w in (1, 4, 16):
        sv, sk = roll(V, w), roll(K, w)
        r = np.where(sv > 0, sk / np.where(sv > 0, sv, 1) - 0.5, 0.0)
        r[(sk == 0) & (sv > 0)] = np.nan  # 2017-11 전 체결 방향 자료 없음
        cols.append(r)
    for w, base in ((4, 96), (16, 384)):
        sv = roll(V, w)
        avg = roll(sv, base) / base
        cols.append(np.log(np.where((sv > 0) & (avg > 0), sv / np.where(avg > 0, avg, 1), np.nan)))
    return np.c_[tuple(cols)]


def oi_feats(path: Path, d: dict) -> np.ndarray:
    """선물 지표: 15분봉 마감 5분 전까지 기록된 마지막 값. 빈 값은 앞 값으로 채운다(과거만)."""
    ts, rows = [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ts.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
            rows.append([float(x) if x else np.nan for x in (row[2], row[5], row[6], row[7])])
    ts, a = np.array(ts), np.array(rows)
    o = np.argsort(ts)
    ts, a = ts[o], a[o]
    a[a <= 0] = np.nan
    for k in range(1, len(a)):
        bad = np.isnan(a[k])
        a[k, bad] = a[k - 1, bad]
    at = lambda T: np.searchsorted(ts, T, side="right") - 1
    T = d["t"] + BAR - M5
    j = at(T)
    ok = j >= 0
    def val(jj, c):
        out = np.full(len(T), np.nan)
        g = jj >= 0
        out[g] = a[jj[g], c]
        return out
    oi = val(j, 0)
    cols = [np.log(oi / val(at(T - h * 3_600_000), 0)) for h in (1, 4, 24)]
    cols += [np.log(val(j, c)) for c in (1, 2, 3)]
    out = np.c_[tuple(cols)]
    out[~ok] = np.nan
    return out
