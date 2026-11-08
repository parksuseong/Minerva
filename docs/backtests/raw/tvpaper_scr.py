"""SCR 3단 바탕 + 트레이딩뷰 구성요소 10개 / 논문 거르개 4개: 진단 30칸 → 규칙 최대 30개.

등록: docs/backtests/2026-11-08-btc-tvpaper-scr.md. 연구 전용(주문 없음, 바이낸스 POST 없음).
트레이딩뷰 원문 코드는 받지 않았다. 널리 공개된 표준 공식만 파이썬으로 옮겼다.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/tvpaper-scr-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/tvpaper_scr.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pine_div as P
from btc_fractal import crossed, recent
from btc_fractal import stoch as stoch_raw

DAY, HOUR, MIN = P.DAY, 3_600_000, P.MIN
MAINW = P.MAINW
N_DIAG, N_RULE_MAX = 30, 30
N_CUM = 6685 + N_DIAG + N_RULE_MAX
NR = 20  # 진단 무작위
NSIM = 100  # 규칙 무작위
TRAIL = {15: 60, 30: 120}  # 위 봉(분)
TV = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10")
PAP = ("P1", "P2", "P3", "P4")
COMPS = TV + PAP
NAMES = {"T1": "WaveTrend 교차(과매도)", "T2": "QQE 추세(1h)", "T3": "UT Bot(1h)", "T4": "SSL 채널(1h)",
         "T5": "Hull 55(1h)", "T6": "Range Filter(1h)", "T7": "Heikin-Ashi(4h)", "T8": "FVG 재진입(1h)",
         "T9": "POC/VAL(7일)", "T10": "Cipher B 자금흐름(1h)", "P1": "14일 모멘텀", "P2": "종가/SMA20 일봉",
         "P3": "유동성 충격 뒤 되돌림", "P4": "테이커 흐름 4봉"}  # fmt: skip


# ---------------- 자료 ----------------
def fetch_tail7(out: Path, last: int) -> np.ndarray:
    """5분봉 꼬리(공개 GET klines, 키 없음): t,o,h,l,c,v,테이커 매수."""
    tail = out / "tail_5m7.json"
    rows, now, s, step = [], int(time.time() * 1000), last + 300_000, 300_000
    try:
        while s < now - step:
            u = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&startTime={s}&limit=1500"
            got = [g for g in json.loads(urllib.request.urlopen(u, timeout=20).read()) if g[6] < now]
            if not got:
                break
            rows += [[g[0], *map(float, (g[1], g[2], g[3], g[4], g[5], g[9]))] for g in got]
            s = got[-1][0] + step
        tail.write_text(json.dumps(rows))
    except Exception as e:  # noqa: BLE001
        print("tail fetch failed", e)
        rows = json.loads(tail.read_text()) if tail.exists() else []
    return np.array(rows, dtype=float).reshape(-1, 7)


def load_5m(out: Path) -> dict:
    a = np.loadtxt(P.FUT / "klines_5m.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 9))
    a = np.vstack([a, fetch_tail7(out, int(a[-1, 0]))])
    return {"t": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": a[:, 5],
            "tb": a[:, 6]}  # fmt: skip


def agg(b: dict, minutes: int, t_end: int) -> dict:
    k = b["t"] // (minutes * MIN)
    st = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    en = np.r_[st[1:] - 1, len(k) - 1]
    F = {"t": k[st] * minutes * MIN, "o": b["o"][st], "c": b["c"][en], "h": np.maximum.reduceat(b["h"], st),
         "l": np.minimum.reduceat(b["l"], st), "v": np.add.reduceat(b["v"], st), "tb": np.add.reduceat(b["tb"], st)}  # fmt: skip
    full = (en - st + 1) == minutes // 5
    full &= F["t"] + minutes * MIN <= t_end
    F = {x: F[x][full] for x in F}
    F["ct"] = F["t"] + minutes * MIN
    F["len"] = minutes * MIN
    F["rsi"], F["cci"] = P.rsi(F["c"]), P.cci(F["h"], F["l"], F["c"])
    F["k"] = stoch_raw(F["h"], F["l"], F["c"], 5)
    F["dir"] = np.where((F["cci"] > 0) & (F["rsi"] > 50), 1, np.where((F["cci"] < 0) & (F["rsi"] < 50), -1, 0))
    F["atr"] = atr(F, 14)
    return F


# ---------------- 지표 ----------------
def ema(x, n):
    return P.ema(np.nan_to_num(np.asarray(x, float)), n)


def atr(F, n):
    pc = np.r_[F["c"][0], F["c"][:-1]]
    tr = np.maximum(F["h"] - F["l"], np.maximum(np.abs(F["h"] - pc), np.abs(F["l"] - pc)))
    out = np.empty_like(tr)
    out[0] = tr[0]
    for i in range(1, len(tr)):
        out[i] = out[i - 1] + (tr[i] - out[i - 1]) / n
    return out


def wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    out = np.full(len(x), np.nan)
    out[n - 1 :] = np.lib.stride_tricks.sliding_window_view(x, n) @ w / w.sum()
    return out


def scr_events(F) -> np.ndarray:
    """SCR 3단 사건(첫 봉): +1 롱, −1 숏, 0 없음."""
    k, cc, r = F["k"], F["cci"], F["rsi"]
    lv = recent(crossed(k, 20, True)) & recent(crossed(cc, -100, True)) & recent(crossed(r, 30, True))
    sv = recent(crossed(k, 80, False)) & recent(crossed(cc, 100, False)) & recent(crossed(r, 70, False))
    lv0 = lv & ~np.r_[False, lv[:-1]]
    sv0 = sv & ~np.r_[False, sv[:-1]]
    return np.where(lv0 & ~sv0, 1, np.where(sv0 & ~lv0, -1, 0))


def wavetrend(F):
    ap = (F["h"] + F["l"] + F["c"]) / 3
    esa = ema(ap, 10)
    d = ema(np.abs(ap - esa), 10)
    ci = (ap - esa) / (0.015 * np.where(d == 0, 1e-9, d))
    wt1 = ema(ci, 21)
    wt2 = P.sma(wt1, 4)
    up = crossed(wt1 - wt2, 0, True) & (wt2 <= -53)
    dn = crossed(wt1 - wt2, 0, False) & (wt2 >= 53)
    return recent(up), recent(dn)


def qqe(F, rp=14, sf=5, q=4.238):
    rs = ema(np.nan_to_num(F["rsi"], nan=50.0), sf)
    wp = rp * 2 - 1
    dar = ema(ema(np.abs(np.r_[0, np.diff(rs)]), wp), wp) * q
    n = len(rs)
    lb, sb, tr = np.zeros(n), np.zeros(n), np.ones(n)
    lb[0], sb[0] = rs[0] - dar[0], rs[0] + dar[0]
    for i in range(1, n):
        nl, ns = rs[i] - dar[i], rs[i] + dar[i]
        lb[i] = max(lb[i - 1], nl) if (rs[i - 1] > lb[i - 1] and rs[i] > lb[i - 1]) else nl
        sb[i] = min(sb[i - 1], ns) if (rs[i - 1] < sb[i - 1] and rs[i] < sb[i - 1]) else ns
        cu = cd = False
        if i > 1:
            cu = np.sign(rs[i] - sb[i - 1]) != np.sign(rs[i - 1] - sb[i - 2])
            cd = np.sign(lb[i - 1] - rs[i]) != np.sign(lb[i - 2] - rs[i - 1])
        tr[i] = 1 if cu else (-1 if cd else tr[i - 1])
    return tr


def ut_bot(F, key=1.0, n=10):
    c, a = F["c"], atr(F, n) * key
    tr = np.empty_like(c)
    tr[0] = c[0] - a[0]
    for i in range(1, len(c)):
        p = tr[i - 1]
        if c[i] > p and c[i - 1] > p:
            tr[i] = max(p, c[i] - a[i])
        elif c[i] < p and c[i - 1] < p:
            tr[i] = min(p, c[i] + a[i])
        else:
            tr[i] = c[i] - a[i] if c[i] > p else c[i] + a[i]
    return np.where(c > tr, 1, -1)


def ssl(F, n=10):
    sh, sl = P.sma(F["h"], n), P.sma(F["l"], n)
    out = np.zeros(len(F["c"]))
    for i in range(1, len(out)):
        c = F["c"][i]
        out[i] = 1 if c > sh[i] else (-1 if c < sl[i] else out[i - 1])
    return out


def hull(F, n=55):
    c = F["c"]
    h = wma(2 * wma(c, n // 2) - wma(c, n), int(round(math.sqrt(n))))
    h2 = np.r_[np.nan, np.nan, h[:-2]]
    return np.where(h > h2, 1, np.where(h < h2, -1, 0))


def range_filter(F, per=100, mult=3.0):
    x = F["c"]
    r = ema(ema(np.abs(np.r_[0, np.diff(x)]), per), per * 2 - 1) * mult
    f = np.empty_like(x)
    f[0] = x[0]
    up = np.zeros(len(x))
    dn = np.zeros(len(x))
    for i in range(1, len(x)):
        p = f[i - 1]
        if x[i] > p:
            f[i] = p if x[i] - r[i] < p else x[i] - r[i]
        else:
            f[i] = p if x[i] + r[i] > p else x[i] + r[i]
        up[i] = up[i - 1] + 1 if f[i] > p else (0 if f[i] < p else up[i - 1])
        dn[i] = dn[i - 1] + 1 if f[i] < p else (0 if f[i] > p else dn[i - 1])
    return np.where(up > 0, 1, np.where(dn > 0, -1, 0))


def heikin(F):
    hc = (F["o"] + F["h"] + F["l"] + F["c"]) / 4
    ho = np.empty_like(hc)
    ho[0] = (F["o"][0] + F["c"][0]) / 2
    for i in range(1, len(hc)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return np.where(hc > ho, 1, -1)


def money_flow(F):
    rng = np.where(F["h"] - F["l"] == 0, np.nan, F["h"] - F["l"])
    mf = P.sma(np.nan_to_num((F["c"] - F["o"]) / rng * 150), 60) - 2.5
    return np.where(mf > 0, 1, np.where(mf < 0, -1, 0))


def fvg_state(H, F):
    """매매 봉마다 (롱 켜짐, 숏 켜짐): 살아 있는 1h FVG 구역에 신호 봉이 들어옴."""
    j = P.last_closed(H, F["ct"])
    n1 = len(H["c"])
    hi, lo, c = H["h"], H["l"], H["c"]
    bull = np.zeros(n1, bool)
    bear = np.zeros(n1, bool)
    bull[2:] = lo[2:] > hi[:-2]
    bear[2:] = hi[2:] < lo[:-2]
    btop, bbot = np.r_[np.nan, np.nan, lo[2:]], np.r_[np.nan, np.nan, hi[:-2]]  # 상승: [고가(i−2), 저가(i)]
    stop_, sbot = np.r_[np.nan, np.nan, lo[:-2]], np.r_[np.nan, np.nan, hi[2:]]  # 하락: [고가(i), 저가(i−2)]
    onL = np.zeros(len(F["c"]), bool)
    onS = np.zeros(len(F["c"]), bool)
    ok = j >= 0
    jj = np.clip(j, 0, None)
    rmin, rmax = _runs(c, 48), _runs(c, 48, True)
    for k in range(48):
        z = jj - k
        v = ok & (z >= 2)
        zc = np.clip(z, 0, None)
        # 소멸: 생성 봉 뒤 마감 종가(k개) 중 하나라도 구역 밖
        bl = v & bull[zc] & (rmin[k][jj] >= bbot[zc]) & (F["l"] <= btop[zc]) & (F["c"] >= bbot[zc])
        br = v & bear[zc] & (rmax[k][jj] <= stop_[zc]) & (F["h"] >= sbot[zc]) & (F["c"] <= stop_[zc])
        onL |= bl
        onS |= br
    return onL, onS


_RUNS = {}


def _runs(c, K, mx=False):
    """k마다 x에서 끝나는 길이 k 창의 최소(최대) 종가, k=0은 ±inf."""
    key = (id(c), mx)
    if key not in _RUNS:
        out = [np.full(len(c), -np.inf if mx else np.inf)]
        cur = out[0].copy()
        for k in range(1, K):
            sh = np.r_[np.full(k - 1, np.nan), c[: len(c) - k + 1]]
            sh = np.nan_to_num(sh, nan=(-np.inf if mx else np.inf))
            cur = np.maximum(cur, sh) if mx else np.minimum(cur, sh)
            out.append(cur.copy())
        _RUNS[key] = out
    return _RUNS[key]


def volume_profile(F15, H, step=0.0025, days=7):
    """1시간 마감마다 직전 7일 15분봉 거래량(hlc3 칸)으로 POC·VAL·VAH."""
    tp = (F15["h"] + F15["l"] + F15["c"]) / 3
    b = np.floor(np.log(tp) / math.log(1 + step)).astype(np.int64)
    ct15 = F15["ct"]
    n = len(H["ct"])
    poc, val, vah = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        z = np.searchsorted(ct15, H["ct"][i], "right")
        a = np.searchsorted(ct15, H["ct"][i] - days * DAY, "right")
        if z - a < 600:
            continue
        bb = b[a:z]
        m0 = bb.min()
        hist = np.bincount(bb - m0, weights=F15["v"][a:z])
        p = int(np.argmax(hist))
        tot, acc, lo, hi = hist.sum(), hist[p], p, p
        while acc < 0.7 * tot:
            up = hist[hi + 1] if hi + 1 < len(hist) else -1
            dn = hist[lo - 1] if lo > 0 else -1
            if up >= dn:
                hi += 1
                acc += up
            else:
                lo -= 1
                acc += dn
        poc[i], val[i], vah[i] = (1 + step) ** (p + m0 + 0.5), (1 + step) ** (lo + m0), (1 + step) ** (hi + m0 + 1)
    return poc, val, vah


def htf(Fs, key, F, arr):
    j = P.last_closed(Fs[key], F["ct"])
    return np.where(j >= 0, arr[np.clip(j, 0, None)], 0)


def components(F, Fs, VP) -> dict:
    """구성요소별 (롱 켜짐, 숏 켜짐) 불 배열, 매매 봉 F 위치."""
    C = {}
    C["T1"] = wavetrend(F)
    H, H4, D = Fs["1h"], Fs["4h"], Fs["1d"]
    for k, f in (("T2", qqe), ("T3", ut_bot), ("T4", ssl), ("T5", hull), ("T6", range_filter), ("T10", money_flow)):
        s = htf(Fs, "1h", F, f(H))
        C[k] = (s == 1, s == -1)
    s = htf(Fs, "4h", F, heikin(H4))
    C["T7"] = (s == 1, s == -1)
    C["T8"] = fvg_state(H, F)
    poc, val, vah = (htf(Fs, "1h", F, x) for x in VP)
    with np.errstate(invalid="ignore"):
        C["T9"] = ((F["c"] < poc) & (F["l"] <= val * 1.0025), (F["c"] > poc) & (F["h"] >= vah * 0.9975))
    dc = D["c"]
    r14 = np.r_[np.full(14, np.nan), dc[14:] / dc[:-14] - 1]
    s = htf(Fs, "1d", F, np.nan_to_num(np.sign(r14)))
    C["P1"] = (s == 1, s == -1)
    s = htf(Fs, "1d", F, np.nan_to_num(np.sign(dc / P.sma(dc, 20) - 1)))
    C["P2"] = (s == 1, s == -1)
    vs = P.sma(F["v"], 20)
    big = (F["v"] >= 2 * vs) & (F["h"] - F["l"] >= 1.5 * np.r_[np.nan, F["atr"][:-1]])
    C["P3"] = (recent(big & (F["c"] < F["o"]), 4), recent(big & (F["c"] > F["o"]), 4))
    dl = 2 * F["tb"] - F["v"]
    s4 = np.convolve(dl, np.ones(4))[: len(dl)]
    C["P4"] = (s4 > 0, s4 < 0)
    return C


# ---------------- 진단 ----------------
class Path1m:
    def __init__(self, m):
        self.t, self.o, self.h, self.l = m["t"], m["o"], m["h"], m["l"]
        self.cache = {}

    def fav(self, k0, d, sd, cap=14 * 1440):
        key = (k0, d, round(sd, 6))
        if key in self.cache:
            return self.cache[key]
        e = self.o[k0]
        stop = e * (1 - d * sd)
        z = min(len(self.t), k0 + cap)
        s, w, ks = k0, 720, -1
        while s < z:
            q = min(z, s + w)
            hit = (self.l[s:q] <= stop) if d == 1 else (self.h[s:q] >= stop)
            if hit.any():
                ks = s + int(np.argmax(hit))
                break
            s, w = q, w * 4
        end = ks if ks >= 0 else z
        if end <= k0:
            v = 0.0
        else:
            v = float(self.h[k0:end].max() / e - 1) if d == 1 else float(1 - self.l[k0:end].min() / e)
        self.cache[key] = v
        return v


def struct_stop(F, i, d, e):
    lo = max(0, i - 7)
    ext = F["l"][lo : i + 1].min() * (1 - 0.0005) if d == 1 else F["h"][lo : i + 1].max() * (1 + 0.0005)
    sd = d * (e - ext) / e
    if sd > 0.03:
        return None
    if sd < 0.0015:
        sd = 0.0015
    return e * (1 - d * sd)


def events(F, ctx, m) -> list:
    """SCR 사건 → 체결 가능한 사건(추격 제한·구조 손절)."""
    ev = scr_events(F)
    t1, o1 = m["t"], m["o"]
    out = []
    for i in np.nonzero(ev)[0]:
        if i < 200:
            continue
        d = int(ev[i])
        T = int(F["ct"][i])
        k0 = int(np.searchsorted(t1, T))
        if k0 >= len(t1) - 2 or t1[k0] - T > 30 * MIN:
            continue
        e = o1[k0]
        lim = 0.03 if ctx["agree"][i] == d else 0.01
        if d * (e - F["c"][i]) / F["c"][i] >= lim:
            continue
        stop = struct_stop(F, i, d, e)
        if stop is None:
            continue
        out.append({"i": int(i), "d": d, "T": T, "k0": k0, "stop": float(stop), "sd": float(d * (e - stop) / e)})
    return out


def two_prop_p(h1, n1, h2, n2):
    if n1 == 0 or n2 == 0:
        return None
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0
    return P.norm_p((h1 / n1 - h2 / n2) / se) if se else None


def diag_cell(EV, F, on, PM, W, rng, comp, t_end):
    """on: [n_bars, 2] (롱, 숏) 켜짐. comp=None이면 바탕(무작위 = 아무 시각)."""
    ct = F["ct"]
    res = {}
    rows = []
    for x in EV:
        if x["T"] > t_end - 15 * DAY:
            continue
        c_on = True if comp is None else bool(on[x["d"]][x["i"]])
        r2 = PM.fav(x["k0"], x["d"], x["sd"]) >= 2 * x["sd"]
        r1 = PM.fav(x["k0"], x["d"], x["sd"]) >= x["sd"]
        rr = []
        if c_on:
            lo, hi = np.searchsorted(ct, x["T"] - 15 * DAY), np.searchsorted(ct, x["T"] + 15 * DAY)
            cand = np.arange(lo, hi) if comp is None else lo + np.nonzero(on[x["d"]][lo:hi])[0]
            if len(cand):
                for j in rng.choice(cand, NR):
                    k = int(np.searchsorted(PM.t, ct[j]))
                    if k < len(PM.t) - 2:
                        rr.append(PM.fav(k, x["d"], x["sd"]) >= 2 * x["sd"])
        rows.append((x["T"], c_on, r2, r1, np.mean(rr) if rr else np.nan))
    if not rows:
        return {}
    T = np.array([r[0] for r in rows])
    con = np.array([r[1] for r in rows])
    h2 = np.array([r[2] for r in rows], float)
    h1 = np.array([r[3] for r in rows], float)
    rn = np.array([r[4] for r in rows], float)
    for w in (*MAINW, "m6", "d30", "pool"):
        mk = (T >= W[w][0]) & (T < W[w][1])
        a, b = mk & con, mk & ~con
        na, nb = int(a.sum()), int(b.sum())
        cell = {"n_on": na, "n_off": nb}
        if na:
            cell.update({"hit_on": float(h2[a].mean()), "hit1_on": float(h1[a].mean()),
                         "rand_on": float(np.nanmean(rn[a])) if np.isfinite(rn[a]).any() else None})  # fmt: skip
            if cell["rand_on"] is not None:
                cell["ex_rand"] = cell["hit_on"] - cell["rand_on"]
                rr = cell["rand_on"]
                cell["p_rand"] = P.norm_p(cell["ex_rand"] / math.sqrt(rr * (1 - rr) / na)) if 0 < rr < 1 else None
        if na and nb:
            cell["hit_off"] = float(h2[b].mean())
            cell["diff"] = cell["hit_on"] - cell["hit_off"]
            cell["p_diff"] = two_prop_p(h2[a].sum(), na, h2[b].sum(), nb)
        res[w] = cell
    return res


def cell_pass(c, strict=True):
    pl = c.get("pool", {})
    if pl.get("diff") is None or pl["diff"] <= 0 or pl.get("p_diff") is None:
        return False
    big = [w for w in MAINW if c[w].get("n_on", 0) >= 10 and c[w].get("n_off", 0) >= 10]
    if len(big) < 3 or not all(c[w]["diff"] > 0 for w in big):
        return False
    if (pl.get("ex_rand") or -1) <= 0:
        return False
    return pl["p_diff"] * N_DIAG < 0.05 if strict else True


# ---------------- 규칙 ----------------
def build(S, F, EV, rule, C, ctx, lat=0, fee=1.0, slip=P.SLIP):
    comp, gate4 = rule["comps"], rule["g4"]
    tf = rule["tf"]
    t1 = S.t
    out, busy = [], -1
    for x in EV:
        i, d = x["i"], x["d"]
        if not all(C[c][0 if d == 1 else 1][i] for c in comp):
            continue
        if gate4 and ctx["g4h"][i] == -d:
            continue
        k0 = x["k0"] + lat
        if k0 >= len(t1) - 2 or k0 <= busy:
            continue
        e = S.m["o"][k0]
        lim = 0.03 if ctx["agree"][i] == d else 0.01
        if d * (e - F["c"][i]) / F["c"][i] >= lim:
            continue
        tr = S.run(k0, d, x["stop"], "X", tf, fee, slip, TRAIL[tf], 1.5)
        if tr is None:
            continue
        tr.update({"T": x["T"], "i": i})
        out.append(tr)
        busy = tr["kx"]
    return out


def rule_random(S, F, tr, rule, C, ctx, cache):
    out = np.full((len(tr), NSIM), np.nan)
    ct = F["ct"]
    tf = rule["tf"]
    for n, x in enumerate(tr):
        key = (rule["name"], x["T"], x["d"], round(x["Rp"], 6))
        if key in cache:
            out[n] = cache[key]
            continue
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        lo, hi = np.searchsorted(ct, x["T"] - 15 * DAY), np.searchsorted(ct, x["T"] + 15 * DAY)
        ok = np.ones(hi - lo, bool)
        for c in rule["comps"]:
            ok &= C[c][0 if x["d"] == 1 else 1][lo:hi]
        if rule["g4"]:
            ok &= ctx["g4h"][lo:hi] != -x["d"]
        cand = lo + np.nonzero(ok)[0]
        res = []
        if len(cand):
            for j in rng.choice(cand, NSIM):
                k0 = int(np.searchsorted(S.t, ct[j]))
                if k0 >= len(S.t) - 2 or S.t[k0] - ct[j] > 30 * MIN:
                    continue
                e = S.m["o"][k0] * (1 + x["d"] * P.SLIP)
                rr = S.run(k0, x["d"], e * (1 - x["d"] * x["Rp"]), "X", tf, trail=TRAIL[tf], be_at=1.5)
                if rr is not None:
                    res.append(rr["r"])
        v = np.array(res + [np.nan] * (NSIM - len(res)))[:NSIM]
        cache[key] = v
        out[n] = v
    return out


def ctx_for(F, Fs):
    T = F["ct"]
    g = {k: htf(Fs, k, F, Fs[k]["dir"]) for k in ("4h", "1d")}
    agree = np.where((F["dir"] == g["4h"]) & (g["4h"] == g["1d"]), g["1d"], 0)
    return {"g4h": g["4h"], "agree": agree, "T": T}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b5 = load_5m(out)
    m = P.load_1m(out)
    S = P.Sim(m, P.load_funding())
    t_end = int(min(m["t"][-1] + MIN, b5["t"][-1] + 300_000))
    Fs = {k: agg(b5, mi, t_end) for k, mi in (("15", 15), ("30", 30), ("1h", 60), ("4h", 240), ("1d", 1440))}
    del b5
    W = P.windows(t_end)
    VP = volume_profile(Fs["15"], Fs["1h"])
    print(f"loaded {time.time() - t0:.0f}s t_end {datetime.fromtimestamp(t_end / 1000, timezone.utc)}", flush=True)
    PM = Path1m(m)
    rng = np.random.default_rng(11)
    frames = {15: Fs["15"], 30: Fs["30"]}
    CC, CTX, EVS, diag = {}, {}, {}, {}
    for tf, F in frames.items():
        CTX[tf] = ctx_for(F, Fs)
        CC[tf] = components(F, Fs, VP)
        EVS[tf] = events(F, CTX[tf], m)
        diag[f"B{tf}"] = diag_cell(EVS[tf], F, None, PM, W, rng, None, t_end)
        # 켜짐 비율(설명용)
        for c in COMPS:
            on = {1: CC[tf][c][0], -1: CC[tf][c][1]}
            diag[f"{c}·{tf}"] = diag_cell(EVS[tf], F, on, PM, W, rng, c, t_end)
            on_rate = [float(np.mean(CC[tf][c][0])), float(np.mean(CC[tf][c][1]))]
            diag[f"{c}·{tf}"]["on_rate"] = on_rate
            pl = diag[f"{c}·{tf}"].get("pool", {})
            print(f"{c}·{tf}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pl.items()},
                  f"{time.time() - t0:.0f}s", flush=True)  # fmt: skip
        print(f"B{tf}", diag[f"B{tf}"].get("pool"), "events", len(EVS[tf]), flush=True)
    (out / "diag.json").write_text(json.dumps(diag, default=float))

    cells = [k for k in diag if not k.startswith("B")]
    passed = [k for k in cells if cell_pass(diag[k])]
    mode = "strict"
    if not passed:
        mode = "explore"
        loose = [k for k in cells if cell_pass(diag[k], strict=False)]
        passed = sorted(loose, key=lambda k: diag[k]["pool"]["p_diff"])[:3]
    passed = sorted(passed, key=lambda k: diag[k]["pool"]["p_diff"])[:12]
    print("cells to rules:", mode, passed, flush=True)
    rules = []
    for k in passed:
        c, tf = k.split("·")
        for g4 in (False, True):
            rules.append({"name": f"SCR{tf}+{c}{'·G4' if g4 else ''}", "tf": int(tf), "comps": [c], "g4": g4})
    for tf in (15, 30):
        tv = [k for k in passed if k.endswith(f"·{tf}") and k.startswith("T")]
        pp = [k for k in passed if k.endswith(f"·{tf}") and k.startswith("P")]
        if tv and pp:
            ct_, cp = tv[0].split("·")[0], pp[0].split("·")[0]
            for g4 in (False, True):
                rules.append({"name": f"SCR{tf}+{ct_}+{cp}{'·G4' if g4 else ''}", "tf": tf, "comps": [ct_, cp], "g4": g4})
    rules = rules[:N_RULE_MAX]
    res, trj, cache = {}, {}, {}
    for rule in rules:
        tf = rule["tf"]
        F, C, ctx, EV = frames[tf], CC[tf], CTX[tf], EVS[tf]
        tr = build(S, F, EV, rule, C, ctx)
        r = {w: P.summ(P.in_w(tr, *W[w]), *W[w]) for w in P.WINS}
        sel = [x for x in tr if x["T"] >= W["fresh"][0]]
        R = rule_random(S, F, sel, rule, C, ctx, cache)
        Ti = np.array([x["T"] for x in sel])
        rnd = {}
        for w in (*MAINW, "m6", "d30", "pool"):
            mk = (Ti >= W[w][0]) & (Ti < W[w][1])
            rnd[w] = P.rand_stats([x for x, k_ in zip(sel, mk) if k_], R[mk])
        r["rand"] = rnd
        red = {}
        for rk, kw in (("lat1m", {"lat": 1}), ("lat1bar", {"lat": tf}), ("slip+2", {"slip": P.SLIP + 0.0002}),
                       ("fee1.5", {"fee": 1.5})):  # fmt: skip
            t2 = build(S, F, EV, rule, C, ctx, **kw)
            red[rk] = {w: P.summ(P.in_w(t2, *W[w]), *W[w]) for w in ("val", "year")}
        r["red"] = red
        r["years"] = {y: P.summ(P.in_w(tr, P.ms(f"{y}-01-01"), min(P.ms(f"{y + 1}-01-01"), t_end)),
                                P.ms(f"{y}-01-01"), min(P.ms(f"{y + 1}-01-01"), t_end)) for y in range(2020, 2027)}  # fmt: skip
        res[rule["name"]] = r
        trj[rule["name"]] = {w: P.in_w(tr, *W[w]) for w in ("d30", "m6", "year", "pool")}
        print(rule["name"], f"{time.time() - t0:.0f}s", " ".join(
            f"{w}:{r[w]['n']}/{100 * r[w].get('win', 0):.0f}%/{100 * r[w].get('risk1', {}).get('ret', 0):+.0f}%"
            for w in (*MAINW, "m6", "d30")), "p", (rnd["pool"] or {}).get("p"), flush=True)  # fmt: skip
    nr = max(len(rules), 1)
    ps = {n: (res[n]["rand"]["pool"] or {}).get("p", 1.0) for n in res}
    holm, run = {}, 0.0
    for i, n in enumerate(sorted(ps, key=ps.get)):
        run = max(run, min(1.0, ps[n] * (nr - i)))
        holm[n] = run
    verdict = {}
    for n, r in res.items():
        g1 = all(r[w]["n"] and r[w]["risk1"]["ret"] > 0 and r[w]["x5_40"]["ret"] > 0 and r[w]["x5_100"]["ret"] > 0
                 and r[w]["x5_100"]["liq"] == 0 and r[w]["x5_40"]["liq"] == 0 for w in MAINW)  # fmt: skip
        g2 = holm[n] < 0.05 and all((r["rand"][w] or {}).get("ex_avg", -1) > 0 for w in MAINW)
        g3 = r["design"]["perday"] >= 0.3 and r["year"]["perday"] >= 0.3
        pl = r["pool"]
        g4 = bool(pl["n"]) and min(pl["long"], pl["short"]) >= 0.15 * pl["n"]
        g5 = all(r["red"][k][w]["n"] and r["red"][k][w]["risk1"]["ret"] > 0 for k in r["red"] for w in ("val", "year"))
        gs = [bool(g1), bool(g2), bool(g3), bool(g4), bool(g5)]
        verdict[n] = {"g": gs, "holm": holm[n], "cum": min(1.0, ps[n] * N_CUM), "pass": all(gs),
                      "near": gs[1] and sum(gs) == 4}  # fmt: skip
    (out / "result.json").write_text(json.dumps({"diag_mode": mode, "cells": passed, "rules": rules, "res": res,
                                                 "verdict": verdict, "t_end": t_end, "secs": time.time() - t0},
                                                default=float))  # fmt: skip
    (out / "trades.json").write_text(json.dumps(trj, default=float))
    print("pass:", [n for n in verdict if verdict[n]["pass"]], "near:", [n for n in verdict if verdict[n]["near"]],
          f"{time.time() - t0:.0f}s")  # fmt: skip


if __name__ == "__main__":
    main()
