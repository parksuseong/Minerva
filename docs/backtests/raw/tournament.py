"""BTC 퀀트 기법 토너먼트. 등록: 2026-11-08-btc-technique-tournament-hypotheses.md

교과서 형태·표준 값 → 살아남은 것 조합·위험 장치 → 작은 격자 수정. 장 시간 규칙 없음.
자료: 2017-09~2019-12 현물 15분봉 + 2020-01~ 선물 15분봉(펀딩). 조사 전용, 주문 없음.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/tournament-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/tournament.py --out $OUT
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scr15_timing as ST

M15, HOUR, DAY = 900_000, 3_600_000, 86_400_000
TAKER, MAKER = 0.0005, 0.0002
LEV = 5
MMR = 0.005
SPOT = Path("~/.minerva/data/btc/15m.csv").expanduser()
PREM = Path("~/.minerva/research/round5-2026-11-08/data15.npz").expanduser()
N_BEFORE = 6030
N_REG = 81


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


FUT0 = ms("2020-01-01")


# ---------------- 자료 ----------------
def load() -> dict:
    f = ST.load()
    st, so, sh, sl, sc = [], [], [], [], []
    with SPOT.open() as fh:
        for row in csv.DictReader(fh):
            ot = int(row["open_time"])
            if ot >= FUT0:
                break
            st.append(ot)
            so.append(float(row["open"]))
            sh.append(float(row["high"]))
            sl.append(float(row["low"]))
            sc.append(float(row["close"]))
    b = {"t": np.r_[np.array(st, np.int64), f["t"]]}
    for k, s in (("o", so), ("h", sh), ("l", sl), ("c", sc)):
        b[k] = np.r_[np.array(s), f[k]]
    n = len(b["t"])
    fund = np.zeros(n)
    idx = np.searchsorted(b["t"], f["ft"], side="right") - 1
    ok = idx >= 0
    np.add.at(fund, idx[ok], f["fr"][ok])
    b["fund"] = fund
    zp = np.full(n, np.nan)
    d = np.load(PREM)
    j = np.searchsorted(b["t"], d["t"])
    m = (j < n) & (b["t"][np.minimum(j, n - 1)] == d["t"])
    zp[j[m]] = d["f_zP"][m]
    b["zP"] = zp
    b["info"] = {"bars": n, "spot_bars": len(st), "fut": f["info"]}
    return b


def windows(t: np.ndarray) -> dict:
    end = int(t[-1]) + M15
    w = {"spot": (ms("2017-09-01"), ms("2020-03-01")), "fresh": (ms("2020-03-01"), ms("2021-11-01")),
         "design": (ms("2022-01-01"), ms("2024-07-01")), "val": (ms("2024-07-01"), ms("2025-11-01")),
         "year": (ms("2025-11-01"), end), "m6": (end - 183 * DAY, end), "d30": (end - 30 * DAY, end),
         "pool": (ms("2020-03-01"), ms("2025-11-01"))}  # fmt: skip
    return {k: (int(np.searchsorted(t, a)), int(np.searchsorted(t, z))) for k, (a, z) in w.items()}


# ---------------- 큰 봉 ----------------
class Frame:
    """15분봉을 큰 봉으로 묶는다. 값은 큰 봉 마감 다음 15분봉부터 쓴다."""

    def __init__(self, b: dict, period: int):
        g = b["t"] // period
        starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
        ends = np.r_[starts[1:], len(g)] - 1
        if len(ends) and int(b["t"][ends[-1]]) + M15 < (int(g[ends[-1]]) + 1) * period:
            starts, ends = starts[:-1], ends[:-1]  # 진행 중인 마지막 큰 봉 버림
        self.n15 = len(g)
        self.s, self.e = starts, ends
        self.O = b["o"][starts]
        self.H = np.maximum.reduceat(b["h"], starts)[: len(starts)]
        self.L = np.minimum.reduceat(b["l"], starts)[: len(starts)]
        self.C = b["c"][ends]

    def to15(self, v: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """큰 봉 k의 값을 e[k]+1 봉부터 다음 값이 나올 때까지."""
        out = np.full(self.n15, np.nan)
        pos = self.e + 1
        ok = pos < self.n15
        out[pos[ok]] = v[ok]
        # 앞으로 채움
        idx = np.where(~np.isnan(out), np.arange(self.n15), 0)
        np.maximum.accumulate(idx, out=idx)
        res = out[idx]
        res[: pos[0] if len(pos) else self.n15] = fill
        return res


def sma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.r_[0.0, x])
        out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def ema(x, n):
    out = np.empty(len(x))
    a = 2 / (n + 1)
    m = x[0]
    for i, v in enumerate(x):
        m = v if i == 0 else a * v + (1 - a) * m
        out[i] = m
    return out


def rstd(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1 :] = w.std(1)
    return out


def rmax(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1 :] = np.lib.stride_tricks.sliding_window_view(x, n).max(1)
    return out


def rmin(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1 :] = np.lib.stride_tricks.sliding_window_view(x, n).min(1)
    return out


def atr(H, L, C, n):
    pc = np.r_[C[0], C[:-1]]
    tr = np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc)))
    out = np.full(len(C), np.nan)
    if len(C) > n:
        a = tr[:n].mean()
        out[n - 1] = a
        for i in range(n, len(C)):
            a = (a * (n - 1) + tr[i]) / n
            out[i] = a
    return out


def rsi_w(C, n):
    out = np.full(len(C), np.nan)
    d = np.diff(C)
    if len(d) < n:
        return out
    g, ls = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = g[:n].mean(), ls[:n].mean()
    out[n] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + ls[i]) / n
        out[i + 1] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def lag(x, k=1):
    if k == 0:
        return x.astype(float)
    return np.r_[np.full(k, np.nan), x[:-k]]


# ---------------- 신호 (큰 봉 위의 목표 위치) ----------------
def state_machine(long_in, long_out, short_in, short_out):
    pos, out = 0, np.zeros(len(long_in))
    for i in range(len(long_in)):
        if pos == 1 and long_out[i]:
            pos = 0
        elif pos == -1 and short_out[i]:
            pos = 0
        if pos == 0:
            if long_in[i]:
                pos = 1
            elif short_in[i]:
                pos = -1
        out[i] = pos
    return out


def sig_tsmom(F, days_per_bar, look_days):
    k = max(1, round(look_days / days_per_bar))
    r = F.C / lag(F.C, k) - 1
    return np.where(np.isnan(r), 0, np.sign(r))


def sig_ma(F, f, s):
    d = sma(F.C, f) - sma(F.C, s)
    return np.where(np.isnan(d), 0, np.sign(d))


def sig_keltner(F, n=20, k=2.0, an=10):
    m = ema(F.C, n)
    a = atr(F.H, F.L, F.C, an)
    up, dn = m + k * a, m - k * a
    with np.errstate(invalid="ignore"):
        return state_machine(F.C > up, F.C < m, F.C < dn, F.C > m)


def sig_supertrend(F, n=10, k=3.0):
    a = atr(F.H, F.L, F.C, n)
    hl2 = (F.H + F.L) / 2
    ub, lb = hl2 + k * a, hl2 - k * a
    fu, fl = ub.copy(), lb.copy()
    d = np.zeros(len(F.C))
    cur = 1
    for i in range(len(F.C)):
        if np.isnan(a[i]):
            continue
        if i > 0 and not np.isnan(fu[i - 1]):
            fu[i] = ub[i] if (ub[i] < fu[i - 1] or F.C[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lb[i] if (lb[i] > fl[i - 1] or F.C[i - 1] < fl[i - 1]) else fl[i - 1]
            if cur == 1 and F.C[i] < fl[i]:
                cur = -1
            elif cur == -1 and F.C[i] > fu[i]:
                cur = 1
        d[i] = cur
    return d


def sig_macd(F, f=12, s=26, g=9):
    m = ema(F.C, f) - ema(F.C, s)
    return np.sign(m - ema(m, g))


def sig_bbsq(F, n=20, k=2.0, look=120, q=0.10, arm=5):
    mid = sma(F.C, n)
    sd = rstd(F.C, n)
    up, dn = mid + k * sd, mid - k * sd
    w = (up - dn) / mid
    pct = np.full(len(w), np.nan)
    for i in range(look, len(w)):
        win = w[i - look + 1 : i + 1]
        if not np.isnan(win).any():
            pct[i] = (win <= w[i]).mean()
    sq = np.nan_to_num(pct, nan=1.0) <= q
    armed = np.zeros(len(w), bool)
    for j in range(arm):
        armed |= lag(sq.astype(float), j + 1) == 1
    with np.errstate(invalid="ignore"):
        return state_machine(armed & (F.C > up), F.C < mid, armed & (F.C < dn), F.C > mid)


def sig_rsi2(F):
    r = rsi_w(F.C, 2)
    s200, s5 = sma(F.C, 200), sma(F.C, 5)
    with np.errstate(invalid="ignore"):
        return state_machine((r < 10) & (F.C > s200), F.C > s5, (r > 90) & (F.C < s200), F.C < s5)


def sig_ibs(F):
    rng = F.H - F.L
    ibs = np.where(rng > 0, (F.C - F.L) / np.where(rng > 0, rng, 1), 0.5)
    return np.where(ibs < 0.2, 1, np.where(ibs > 0.8, -1, 0)).astype(float)


def sig_bb2(F, n=20, k=2.0):
    mid = sma(F.C, n)
    sd = rstd(F.C, n)
    with np.errstate(invalid="ignore"):
        return state_machine(F.C < mid - k * sd, F.C >= mid, F.C > mid + k * sd, F.C <= mid)


def sig_z5(F, k=5, look=100, hold=5):
    r = F.C / lag(F.C, k) - 1
    z = r / rstd(np.nan_to_num(r), look)
    out = np.zeros(len(r))
    pos, left = 0, 0
    for i in range(len(r)):
        if pos and left <= 0:
            pos = 0
        if pos == 0 and not np.isnan(z[i]):
            if z[i] <= -2:
                pos, left = 1, hold
            elif z[i] >= 2:
                pos, left = -1, hold
        out[i] = pos
        left -= 1
    return out


def sig_streak(F, k=3):
    up = F.C > lag(F.C)
    dn = F.C < lag(F.C)
    d3 = np.ones(len(F.C), bool)
    u3 = np.ones(len(F.C), bool)
    for j in range(k):
        d3 &= lag(dn.astype(float), j) == 1
        u3 &= lag(up.astype(float), j) == 1
    return state_machine(d3, up, u3, dn)


# ---------------- 가격 수준 진입(15분봉 안 체결) ----------------
def level_daily_breakout(b, F, kind: str, k=0.5, nr=7):
    """LW(래리 윌리엄스), NR(nr). 큰 봉 단위로 그날 안에 수준 돌파 시 진입, 그 큰 봉 끝에 청산."""
    n = len(b["t"])
    tgt = np.zeros(n)
    pxf = np.full(n, np.nan)
    rng = F.H - F.L
    for d in range(1, len(F.s)):
        s, e = F.s[d], F.e[d]
        if kind == "LW":
            upl, dnl = F.O[d] + k * rng[d - 1], F.O[d] - k * rng[d - 1]
        else:
            if d < nr or rng[d - 1] > rng[d - nr : d].min():
                continue
            upl, dnl = F.H[d - 1], F.L[d - 1]
        pos = 0
        for i in range(s, e + 1):
            if pos == 0:
                hu, hd = b["h"][i] >= upl, b["l"][i] <= dnl
                if hu and hd:
                    if b["o"][i] >= upl:
                        hd = False
                    elif b["o"][i] <= dnl:
                        hu = False
                    else:
                        break  # 같은 봉 양쪽 — 순서를 모르니 건너뜀(보수적)
                if hu:
                    pos, pxf[i] = 1, max(upl, b["o"][i])
                elif hd:
                    pos, pxf[i] = -1, min(dnl, b["o"][i])
            tgt[i] = pos
        # e+1(다음 큰 봉 첫 15분봉) 시가에 청산 = tgt 0 (기본)
    return tgt, pxf


def turtle(b, F, ent: int, ex: int, stop_n: float = 2.0):
    n = len(b["t"])
    tgt = np.zeros(n)
    pxf = np.full(n, np.nan)
    hiE, loE = lag(rmax(F.H, ent)), lag(rmin(F.L, ent))
    hiX, loX = lag(rmax(F.H, ex)), lag(rmin(F.L, ex))
    N = lag(atr(F.H, F.L, F.C, 20))
    pos, stop = 0, 0.0
    for d in range(len(F.s)):
        if np.isnan(hiE[d]) or np.isnan(N[d]):
            continue
        for i in range(F.s[d], F.e[d] + 1):
            o, h, l = b["o"][i], b["h"][i], b["l"][i]
            if pos == 1:
                lvl = max(loX[d], stop)
                if l <= lvl:
                    pos, pxf[i] = 0, min(lvl, o)
            elif pos == -1:
                lvl = min(hiX[d], stop)
                if h >= lvl:
                    pos, pxf[i] = 0, max(lvl, o)
            if pos == 0 and np.isnan(pxf[i]):
                if h >= hiE[d] and not l <= loE[d]:
                    pos, pxf[i] = 1, max(hiE[d], o)
                    stop = pxf[i] - stop_n * N[d]
                elif l <= loE[d] and not h >= hiE[d]:
                    pos, pxf[i] = -1, min(loE[d], o)
                    stop = pxf[i] + stop_n * N[d]
            tgt[i] = pos
    return tgt, pxf


# ---------------- 캐리 ----------------
def sig_funding(b):
    n = len(b["t"])
    out = np.full(n, np.nan)
    fi = np.flatnonzero(b["fund"] != 0)
    for i in fi:
        if i + 1 < n:
            out[i + 1] = -np.sign(b["fund"][i])
    idx = np.where(~np.isnan(out), np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    res = out[idx]
    res[np.isnan(res)] = 0
    res[b["t"] < FUT0] = 0
    return res


def sig_prem(b, k=2.0):
    z = b["zP"]
    with np.errstate(invalid="ignore"):
        s = state_machine(z <= -k, z >= 0, z >= k, z <= 0)
    s[np.isnan(z)] = 0
    return lag(s, 1)  # z는 봉 마감 값 → 다음 봉부터


# ---------------- 엔진 ----------------
def run(b, tgt, a, z, lev_f, fee=TAKER, slip=0.0, pxf=None, scale=None, dd_throttle=False, band=0.2,
        delay=0, keep_curve=False):  # fmt: skip
    """목표 위치(−1~1)대로 계좌를 굴린다. lev_f = 배율 × 증거금 비율."""
    o, h, l, c, fund = b["o"], b["h"], b["l"], b["c"], b["fund"]
    T = np.nan_to_num(tgt[a - delay : z - delay] if delay else tgt[a:z])
    if scale is not None:
        T = T * np.nan_to_num(scale[a:z], nan=1.0)
    P = pxf[a:z] if (pxf is not None and not delay) else None
    E, peak, mdd, units, cur, prev = 1.0, 1.0, 0.0, 0.0, 0.0, float(o[a])
    entries, liq, trades, e_in = 0, 0, [], None
    throttle = 1.0
    curve = []
    fr = fee + slip
    for k in range(z - a):
        i = a + k
        want = T[k] * throttle
        if want != cur and (want == 0 or cur == 0 or np.sign(want) != np.sign(cur)
                            or abs(want - cur) > band * abs(cur)):  # fmt: skip
            px = float(P[k]) if (P is not None and not math.isnan(P[k])) else float(o[i])
            E += units * (px - prev)
            prev = px
            if cur != 0 and (want == 0 or np.sign(want) != np.sign(cur)) and e_in:
                trades.append(E / e_in - 1)
                e_in = None
            new = want * lev_f * E / px
            E -= abs(new - units) * px * fr
            if want != 0 and (cur == 0 or np.sign(want) != np.sign(cur)):
                entries += 1
                e_in = E
            units, cur = new, want
        if units:
            worst = l[i] if units > 0 else h[i]
            ew = E + units * (worst - prev)
            if ew <= MMR * abs(units) * worst:
                liq += 1
                E = 0.0
                mdd = 1.0
                curve.append(0.0)
                break
            mdd = max(mdd, 1 - ew / peak)
            E += units * (c[i] - prev) - units * c[i] * fund[i]
        prev = float(c[i])
        peak = max(peak, E)
        mdd = max(mdd, 1 - E / peak)
        if dd_throttle:
            throttle = 0.5 if E < 0.8 * peak else 1.0
        if keep_curve:
            curve.append(E)
    if cur != 0 and e_in and E > 0:
        trades.append(E / e_in - 1)
    days = max((z - a) * M15 / DAY, 1e-9)
    tr = np.array(trades) if trades else np.zeros(0)
    out = {"ret": E - 1, "mdd": mdd, "liq": liq, "entries": entries, "per_day": entries / days,
           "win": float((tr > 0).mean()) if len(tr) else None}  # fmt: skip
    if keep_curve:
        out["curve"] = curve
    return out


def segments(tgt, a, z):
    """일정한 위치 구간들: (시작, 길이, 위치)."""
    T = np.nan_to_num(tgt[a:z])
    ch = np.flatnonzero(np.diff(np.r_[0.0, T]) != 0)
    segs = []
    for j, s in enumerate(ch):
        e = ch[j + 1] if j + 1 < len(ch) else len(T)
        if T[s] != 0:
            segs.append((a + s, e - s, T[s]))
    return segs


def seg_ret(b, s, L, p, fee=TAKER, pxf=None):
    """구간 수익. 가격 수준 체결이면 그 가격으로(봉 시가를 쓰면 돌파 전 움직임까지 들어가 실제보다 좋아진다)."""
    o = b["o"]
    e = min(s + L, len(o) - 1)
    pin = pxf[s] if (pxf is not None and not math.isnan(pxf[s])) else o[s]
    pout = pxf[e] if (pxf is not None and not math.isnan(pxf[e])) else o[e]
    return p * (pout / pin - 1) - 2 * fee * abs(p)


def random_p(b, tgt, a, z, rvq, nsim=300, seed=7, pxf=None):
    """변동성 맞춘 무작위: 같은 길이·위치의 구간을 같은 30일 변동성 5분위 봉에서 무작위로 시작."""
    segs = segments(tgt, a, z)
    if not segs:
        return {"p": None, "n": 0}
    real = sum(math.log(max(1e-9, 1 + seg_ret(b, s, L, p, pxf=pxf))) for s, L, p in segs)
    rng = np.random.default_rng(seed)
    pools = {q: np.flatnonzero(rvq[a:z] == q) + a for q in range(5)}
    vals = np.empty(nsim)
    for k in range(nsim):
        tot = 0.0
        for s, L, p in segs:
            pool = pools.get(int(rvq[s]), None)
            pool = pool[pool < z - L] if pool is not None else None
            j = int(rng.choice(pool)) if pool is not None and len(pool) else s
            tot += math.log(max(1e-9, 1 + seg_ret(b, j, L, p)))
        vals[k] = tot
    return {"p": float((vals >= real).mean()), "n": len(segs), "real": real, "rand_med": float(np.median(vals))}


# ---------------- 조립 ----------------
def build_r1(b) -> dict:
    """1라운드 35개: 이름 → (tgt, pxf, 가격만 쓰는 일봉 규칙인가)."""
    D, H4, H1 = Frame(b, DAY), Frame(b, 4 * HOUR), Frame(b, HOUR)
    S = {}

    def put(name, F, v, price_only=False):
        S[name] = (F.to15(v, 0.0), None, price_only)

    for L in (7, 21, 84):
        put(f"TS{L}@1d", D, sig_tsmom(D, 1, L), True)
    put("TSENS@1d", D, (sig_tsmom(D, 1, 7) + sig_tsmom(D, 1, 21) + sig_tsmom(D, 1, 84)) / 3, True)
    put("MA20_50@1d", D, sig_ma(D, 20, 50), True)
    put("MA50_200@1d", D, sig_ma(D, 50, 200), True)
    put("MA20_50@4h", H4, sig_ma(H4, 20, 50))
    t, p = turtle(b, D, 20, 10)
    S["TUR20@1d"] = (t, p, True)
    t, p = turtle(b, D, 55, 20)
    S["TUR55@1d"] = (t, p, True)
    t, p = turtle(b, H4, 20, 10)
    S["TUR20@4h"] = (t, p, False)
    put("KEL@1d", D, sig_keltner(D), True)
    for nm, F, po in (("1d", D, True), ("4h", H4, False), ("1h", H1, False)):
        put(f"ST@{nm}", F, sig_supertrend(F), po)
        put(f"MACD@{nm}", F, sig_macd(F), po)
    t, p = level_daily_breakout(b, D, "LW")
    S["LW05@1d"] = (t, p, True)
    for nr in (7, 4):
        t, p = level_daily_breakout(b, D, "NR", nr=nr)
        S[f"NR{nr}@1d"] = (t, p, True)
    put("BBSQ@1d", D, sig_bbsq(D), True)
    put("BBSQ@4h", H4, sig_bbsq(H4))
    for nm, F, po in (("1d", D, True), ("4h", H4, False), ("1h", H1, False)):
        put(f"RSI2@{nm}", F, sig_rsi2(F), po)
        put(f"BB2@{nm}", F, sig_bb2(F), po)
    put("IBS@1d", D, sig_ibs(D), True)
    put("IBS@4h", H4, sig_ibs(H4))
    put("Z5@4h", H4, sig_z5(H4))
    put("Z5@1h", H1, sig_z5(H1))
    put("STREAK3@1d", D, sig_streak(D), True)
    S["FUND@8h"] = (sig_funding(b), None, False)
    S["PREM@15m"] = (sig_prem(b), None, False)
    return S


def vol_state(b, W):
    D = Frame(b, DAY)
    r = np.r_[np.nan, np.diff(np.log(D.C))]
    rv = rstd(np.nan_to_num(r), 30) * math.sqrt(365)
    rv15 = D.to15(rv)
    a, z = W["design"]
    ref = rv15[a:z]
    ref = ref[~np.isnan(ref)]
    qs = np.quantile(ref, [0.2, 0.4, 0.6, 0.8])
    rvq = np.searchsorted(qs, np.nan_to_num(rv15, nan=float(np.median(ref))))
    return rv15, rvq, float(np.median(ref)), float(np.quantile(ref, 0.9))


def evaluate(b, W, tgt, pxf, price_only, wins=("spot", "fresh", "design", "val", "year", "m6", "d30"), **kw):
    out = {}
    for w in wins:
        if w == "spot" and not price_only:
            continue
        a, z = W[w]
        out[w] = {f"m{int(f * 100)}": run(b, tgt, a, z, LEV * f, pxf=pxf, **kw) for f in (0.4, 1.0)}
    return out


def shift1(tgt, pxf):
    """조합용: 가격 수준 진입은 체결 봉 다음 봉 시가로 미뤄 미래를 보지 않게."""
    return lag(tgt, 1) if pxf is not None else tgt


def calmar(r):
    m = r["design"]["m40"]
    return (m["ret"] / max(m["mdd"], 0.05)) if m["liq"] == 0 else -9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = load()
    W = windows(b["t"])
    rv15, rvq, rv_med, rv_p90 = vol_state(b, W)
    print("data", b["info"], round(time.time() - t0, 1), flush=True)
    R1 = build_r1(b)
    assert len(R1) == 35, len(R1)
    print("built r1", round(time.time() - t0, 1), flush=True)
    res, count = {}, 0
    for name, (tgt, pxf, po) in R1.items():
        r = evaluate(b, W, tgt, pxf, po)
        a, z = W["design"]
        r["rand_design"] = random_p(b, tgt, a, z, rvq, args.nsim, pxf=pxf)
        r["price_only"] = po
        res[name] = r
        count += 1
        d = r["design"]["m40"]
        print(name, "design", round(d["ret"], 3), round(d["mdd"], 3), d["liq"], d["entries"],
              "p", r["rand_design"]["p"], round(time.time() - t0), flush=True)  # fmt: skip
    surv = [n for n, r in res.items() if r["design"]["m40"]["ret"] > 0 and r["design"]["m40"]["liq"] == 0
            and r["design"]["m40"]["entries"] >= 20
            and r["rand_design"]["p"] is not None and r["rand_design"]["p"] < 0.05]  # fmt: skip
    print("survivors", surv, flush=True)
    json.dump({"r1": res, "survivors": surv}, open(out / "r1.json", "w"), default=float)

    # ---------- 2라운드 ----------
    trend_like = {n for n in surv if n.split("@")[0].rstrip("0123456789_").rstrip("_") in
                  ("TS", "TSENS", "MA", "TUR", "KEL", "ST", "MACD", "LW", "NR", "BBSQ", "FUND", "PREM")}  # fmt: skip
    mr = [n for n in surv if n not in trend_like]
    tr = [n for n in surv if n in trend_like]
    combos = {}
    if surv:
        P = {n: shift1(R1[n][0], R1[n][1]) for n in surv}
        avg = np.mean([np.nan_to_num(P[n]) for n in surv], 0)
        vote = np.sign(np.sum([np.sign(np.nan_to_num(P[n])) for n in surv], 0))
        combos["AVG"] = avg
        combos["VOTE"] = vote
        if tr and mr:
            tsign = np.sign(np.mean([np.nan_to_num(P[n]) for n in tr], 0))
            mavg = np.mean([np.nan_to_num(P[n]) for n in mr], 0)
            combos["TFMR"] = np.where(np.sign(mavg) == tsign, mavg, 0.0)
    overlays = {
        "none": {},
        "VT": {"scale": np.minimum(1.0, rv_med / np.maximum(np.nan_to_num(rv15, nan=rv_med), 1e-6))},
        "DD": {"dd_throttle": True},
        "RV": {"scale": np.where(np.nan_to_num(rv15, nan=0) > rv_p90, 0.0, 1.0)},
    }
    r2 = {}
    for cn, tgt in combos.items():
        for on, kw in overlays.items():
            r = evaluate(b, W, tgt, None, False, **kw)
            r2[f"{cn}|{on}"] = r
            count += 1
            d = r["design"]["m40"]
            print("R2", cn, on, round(d["ret"], 3), round(d["mdd"], 3), d["liq"], round(d["per_day"], 2), flush=True)
    best2 = max(r2, key=lambda k: calmar(r2[k])) if r2 else None
    print("best r2", best2, flush=True)

    # ---------- 3라운드: 상위 3 생존자 값 격자 ----------
    top3 = sorted(surv, key=lambda n: calmar(res[n]), reverse=True)[:3]
    r3, tuned = {}, {}
    for n in top3:
        cands = {}
        for mult in (0.5, 1.0, 1.5):
            for stp in (None, 2.0):
                tg, px = variant(b, n, mult, stp)
                key = f"{n}|x{mult}|{'S2' if stp else 'S0'}"
                r = evaluate(b, W, tg, px, R1[n][2], wins=("fresh", "design", "val", "year", "m6", "d30"))
                r3[key] = r
                cands[key] = (tg, px)
                count += 1
        pick = max(cands, key=lambda k: calmar(r3[k]))
        tuned[n] = pick
        print("R3", n, "->", pick, flush=True)
    final = {}
    if best2:
        cn, on = best2.split("|")
        P = {}
        for n in surv:
            if n in tuned:
                tg, px = variant(b, n, *parse_variant(tuned[n]))
                P[n] = shift1(tg, px)
            else:
                P[n] = shift1(R1[n][0], R1[n][1])
        cand = {"AVG": np.mean([np.nan_to_num(P[n]) for n in surv], 0),
                "VOTE": np.sign(np.sum([np.sign(np.nan_to_num(P[n])) for n in surv], 0))}  # fmt: skip
        for k, tg in cand.items():
            final[k] = evaluate(b, W, tg, None, False, **overlays[on])
            count += 1
        fk = max(final, key=lambda k: calmar(final[k]))
        ftgt, fkw = cand[fk], overlays[on]
    else:
        fk, ftgt, fkw = None, None, {}
    res_final = {"combo": best2, "final": fk, "count": count}
    if fk:
        a, z = W["pool"]
        res_final["rand_pool"] = random_p(b, ftgt, a, z, rvq, 1000, 11)
        res_final["rand_by"] = {w: random_p(b, ftgt, *W[w], rvq, 500, 13) for w in ("fresh", "val", "year")}
        red = {}
        for w in ("fresh", "val", "year"):
            a, z = W[w]
            red[w] = {"delay1": run(b, ftgt, a, z, LEV * 0.4, delay=1, **fkw),
                      "delay4": run(b, ftgt, a, z, LEV * 0.4, delay=4, **fkw),
                      "slip2": run(b, ftgt, a, z, LEV * 0.4, slip=0.0002, **fkw),
                      "fee15": run(b, ftgt, a, z, LEV * 0.4, fee=TAKER * 1.5, **fkw),
                      "maker": run(b, ftgt, a, z, LEV * 0.4, fee=(MAKER + TAKER) / 2, **fkw)}  # fmt: skip
        years = {}
        for y in range(2020, 2027):
            a = int(np.searchsorted(b["t"], ms(f"{y}-01-01")))
            z = int(np.searchsorted(b["t"], ms(f"{y + 1}-01-01")))
            if z > a:
                years[y] = {f: run(b, ftgt, a, z, LEV * f, **fkw)["ret"] for f in (0.4, 1.0)}
        red["by_year"] = years
        a0 = W["fresh"][0]
        roll, step = [], 30 * 96
        for a in range(a0, len(b["t"]) - 90 * 96, step):
            roll.append(run(b, ftgt, a, a + 90 * 96, LEV * 0.4, **fkw)["ret"])
        red["roll90_pos"] = float(np.mean(np.array(roll) > 0)) if roll else None
        res_final["redteam"] = red
        np.savez_compressed(out / "final_tgt.npz", tgt=ftgt, t=b["t"],
                            scale=fkw.get("scale", np.ones(len(ftgt))))  # fmt: skip
        res_final["dd_throttle"] = bool(fkw.get("dd_throttle", False))
    json.dump({"info": b["info"], "survivors": surv, "trend": tr, "mr": mr, "r1": res, "r2": r2, "r3": r3,
               "tuned": tuned, "final_res": final, "final": res_final, "n_tests": count,
               "n_cum": N_BEFORE + count, "rv_med": rv_med, "rv_p90": rv_p90,
               "windows": {k: [int(b["t"][a]), int(b["t"][max(a, z - 1)])] for k, (a, z) in W.items()},
               "sec": time.time() - t0}, open(out / "result.json", "w"), default=float, indent=1)  # fmt: skip
    print("done", count, round(time.time() - t0), flush=True)


def parse_variant(key: str):
    _, m, s = key.split("|")
    return float(m[1:]), (2.0 if s == "S2" else None)


def variant(b, name: str, mult: float, stp):
    """생존자 값 격자: 주 기간 × mult, 손절 2×ATR(일봉 ATR14 기준, 진입가 대비) 선택."""
    base, fr = name.split("@")
    period = {"1d": DAY, "4h": 4 * HOUR, "1h": HOUR, "8h": None, "15m": None}[fr]
    F = Frame(b, period) if period else None
    m = lambda n: max(2, int(round(n * mult)))  # noqa: E731
    pxf = None
    if base.startswith("TS") and base != "TSENS":
        v = sig_tsmom(F, 1, m(int(base[2:])))
    elif base == "TSENS":
        v = (sig_tsmom(F, 1, m(7)) + sig_tsmom(F, 1, m(21)) + sig_tsmom(F, 1, m(84))) / 3
    elif base.startswith("MA"):
        f, s = base[2:].split("_")
        v = sig_ma(F, m(int(f)), m(int(s)))
    elif base.startswith("TUR"):
        ent = int(base[3:])
        t, pxf = turtle(b, F, m(ent), m(10 if ent == 20 else 20), 2.0 if stp is None else stp)
        return t, pxf
    elif base == "KEL":
        v = sig_keltner(F, m(20))
    elif base == "ST":
        v = sig_supertrend(F, m(10))
    elif base == "MACD":
        v = sig_macd(F, m(12), m(26), 9)
    elif base == "BBSQ":
        v = sig_bbsq(F, m(20))
    elif base == "RSI2":
        v = sig_rsi2(F)
    elif base == "BB2":
        v = sig_bb2(F, m(20))
    elif base == "Z5":
        v = sig_z5(F, m(5))
    elif base == "STREAK3":
        v = sig_streak(F, max(2, int(round(3 * mult))))
    elif base == "IBS":
        v = sig_ibs(F)
    elif base.startswith("LW"):
        t, pxf = level_daily_breakout(b, F, "LW", k=0.5 * mult)
        return (apply_stop(b, t, pxf, stp) if stp else t), pxf
    elif base.startswith("NR"):
        t, pxf = level_daily_breakout(b, F, "NR", nr=m(int(base[2:])))
        return (apply_stop(b, t, pxf, stp) if stp else t), pxf
    elif base == "FUND":
        return sig_funding(b), None
    elif base == "PREM":
        return sig_prem(b, 2.0 * mult), None
    else:
        raise KeyError(name)
    t = F.to15(v, 0.0)
    return (apply_stop(b, t, None, stp) if stp else t), pxf


def apply_stop(b, tgt, pxf, k):
    """진입가 대비 k × 일봉 ATR14 역행이면 그 구간을 0으로(다음 신호 변화까지 쉼)."""
    D = Frame(b, DAY)
    a14 = D.to15(atr(D.H, D.L, D.C, 14))
    out = tgt.copy()
    pos, entry, stopped = 0.0, 0.0, False
    for i in range(len(tgt)):
        want = tgt[i]
        if want != pos:
            pos, stopped = want, False
            entry = pxf[i] if (pxf is not None and not math.isnan(pxf[i])) else b["o"][i]
        if pos and not stopped and not math.isnan(a14[i]):
            lvl = entry - np.sign(pos) * k * a14[i]
            if (pos > 0 and b["l"][i] <= lvl) or (pos < 0 and b["h"][i] >= lvl):
                stopped = True
        if stopped:
            out[i] = 0.0 if i + 1 >= len(tgt) else out[i]
            if i + 1 < len(tgt):
                out[i + 1 :][: 1] = 0.0
    # 손절 뒤 같은 신호가 이어지는 동안 쉼
    pos, stopped = 0.0, False
    res = tgt.copy()
    for i in range(len(tgt)):
        if tgt[i] != pos:
            pos, stopped = tgt[i], False
        if stopped:
            res[i] = 0.0
        elif out[i] == 0.0 and tgt[i] != 0:
            stopped = True
            res[i] = tgt[i]  # 손절이 이 봉 안에서 일어남: 다음 봉 시가 청산(보수적)
    return res


if __name__ == "__main__":
    main()
