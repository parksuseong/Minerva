"""선물 전용 특징: 프리미엄 지수, 펀딩. 등록: 2026-11-06-btc-fut15-native-learned-hypotheses.md

모두 15분봉 마감 시각 T에 이미 알 수 있는 값만 쓴다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

M5 = 300_000
BAR = 900_000


def _rmean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def premium(path: Path, d: dict) -> np.ndarray:
    """프리미엄 지수 6개. 5분봉 개수로 평균을 낸다. 마지막 값이 1시간보다 오래되면 비운다."""
    t, c = [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(int(row[0]))
            c.append(float(row[4]))
    t, c = np.array(t, float), np.array(c)
    m1, m4, m24 = _rmean(c, 12), _rmean(c, 48), _rmean(c, 288)
    ch4 = m4 - np.r_[np.full(48, np.nan), m4[:-48]]
    T = d["t"] + BAR
    j = np.searchsorted(t + M5, T, side="right") - 1
    ok = (j >= 0) & (T - (t[np.maximum(j, 0)] + M5) <= 3_600_000)
    out = np.full((len(T), 6), np.nan)
    jj = j[ok]
    out[ok] = np.c_[c[jj], m1[jj], m4[jj], m24[jj], m4[jj] - m24[jj], ch4[jj]]
    return out * 1e4  # bp 단위


def funding(path: Path, d: dict) -> np.ndarray:
    """정산된 마지막 펀딩률과 최근 3회 평균(bp)."""
    t, f = [], []
    with path.open() as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            t.append(int(row[0]))
            f.append(float(row[2]))
    o = np.argsort(t)
    t, f = np.array(t, float)[o], np.array(f)[o]
    m3 = _rmean(f, 3)
    T = d["t"] + BAR
    j = np.searchsorted(t, T, side="right") - 1
    out = np.full((len(T), 2), np.nan)
    ok = j >= 2
    out[ok] = np.c_[f[j[ok]], m3[j[ok]]]
    return out * 1e4
