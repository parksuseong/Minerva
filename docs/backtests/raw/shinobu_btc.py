"""사용자 자동매매 코드(Shinobu, 국내 레버리지/인버스 ETF 5분봉)의 SRC 전략을 BTCUSDT 선물에 이식.

등록: docs/backtests/2026-11-08-btc-shinobu-hypotheses.md (결과 보기 전 커밋)
원 코드는 저장소 밖에서 읽기만 했고 논리를 numpy로 다시 구현했다.
- 지표: Stoch fast 5, CCI 20, RSI 5(Wilder 지수평균), scr_line
- 전략: SRC(blog_scr, 실전 기본) / V2 Normal / V2 ADX. 차트마다 가상 포지션.
- 엔진: 정방향 차트 = 롱, 역수 차트(1/가격) = 숏(원래는 인버스 ETF 차트). settle 1봉 → j+2봉 시가 체결.
  반대 차트 진입 신호면 갈아타기, 들고 있는 차트 청산 신호면 현금, 엔진 손절 1%(종가 확인).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/shinobu-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/shinobu_btc.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pine_eval import lev_path

M5, M15, H4, DAY = 300_000, 900_000, 4 * 3_600_000, 86_400_000
FEE = 0.0005
DATA = Path("~/.minerva/data/btc/futures").expanduser()
N_VAR = 13
N_TONIGHT_BEFORE = 5550


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-research"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


# ---------------- 자료 ----------------
def load5() -> dict:
    a = np.loadtxt(DATA / "klines_5m.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
    b = {"t": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": a[:, 5]}
    now = int(time.time() * 1000)
    t, rows = int(b["t"][-1]) + M5, []
    while t < now - M5:
        r = _get(f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&startTime={t}&limit=1500")
        if not r:
            break
        rows += [x for x in r if x[6] < now]
        t = r[-1][0] + M5
    if rows:
        b["t"] = np.r_[b["t"], [x[0] for x in rows]].astype(np.int64)
        for i, k in enumerate(("o", "h", "l", "c", "v"), 1):
            b[k] = np.r_[b[k], [float(x[i]) for x in rows]]
    f = np.loadtxt(DATA / "funding.csv", delimiter=",", skiprows=1, usecols=(0, 2))
    ft, fr = f[:, 0].astype(np.int64), f[:, 1]
    g = _get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&startTime={int(ft[-1]) + 1}&limit=1000")
    if g:
        ft = np.r_[ft, [x["fundingTime"] for x in g]].astype(np.int64)
        fr = np.r_[fr, [float(x["fundingRate"]) for x in g]]
    # 정산 시각이 몇 ms 어긋나 오는 경우가 있어 분 단위로 맞춘다(6차 시험에서 확인)
    b["ft"], b["fr"] = (ft + 30_000) // 60_000 * 60_000, fr
    b["info"] = {"bars5": len(b["t"]), "appended5": len(rows), "last": int(b["t"][-1]),
                 "gaps5": int((np.diff(b["t"]) != M5).sum())}  # fmt: skip
    return b


def agg(b: dict, period: int) -> dict:
    t = b["t"]
    g = t // period
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    ends = np.r_[starts[1:], len(t)]
    out = {"t": g[starts] * period, "o": b["o"][starts], "c": b["c"][ends - 1],
           "h": np.maximum.reduceat(b["h"], starts), "l": np.minimum.reduceat(b["l"], starts),
           "v": np.add.reduceat(b["v"], starts)}  # fmt: skip
    return out


# ---------------- 지표 (원 코드와 같은 정의) ----------------
def ewm(x: np.ndarray, alpha: float, minp: int) -> np.ndarray:
    """pandas ewm(alpha, adjust=False, min_periods) — 앞쪽 NaN은 건너뛴다."""
    out = np.full(len(x), np.nan)
    y, n = np.nan, 0
    for i, v in enumerate(x.tolist()):
        if v != v:
            continue
        y = v if n == 0 else (1 - alpha) * y + alpha * v
        n += 1
        if n >= minp:
            out[i] = y
    return out


def stoch_fast(h, l, c, n=5):
    out = np.full(len(c), np.nan)
    hh, ll = sliding_window_view(h, n).max(1), sliding_window_view(l, n).min(1)
    rng = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n - 1:] = np.where(rng == 0, np.nan, (c[n - 1:] - ll) / rng * 100)
    return out


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    out = np.full(len(c), np.nan)
    w = sliding_window_view(tp, n)
    m = w.mean(1)
    md = np.abs(w - m[:, None]).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n - 1:] = np.where(md == 0, np.nan, (tp[n - 1:] - m) / (0.015 * md))
    return out


def rsi5(c, n=5):
    d = np.r_[np.nan, np.diff(c)]
    g = np.where(np.isnan(d), np.nan, np.maximum(d, 0))
    lo = np.where(np.isnan(d), np.nan, np.maximum(-d, 0))
    ag, al = ewm(g, 1 / n, n), ewm(lo, 1 / n, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(al == 0, np.nan, 100 - 100 / (1 + ag / al))


def adx14(h, l, c, n=14):
    up = np.r_[np.nan, np.diff(h)]
    dn = np.r_[np.nan, -np.diff(l)]
    with np.errstate(invalid="ignore"):
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    atr = ewm(tr, 1 / n, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * ewm(pdm, 1 / n, n) / np.where(atr == 0, np.nan, atr)
        mdi = 100 * ewm(mdm, 1 / n, n) / np.where(atr == 0, np.nan, atr)
        dx = np.abs(pdi - mdi) / np.where(pdi + mdi == 0, np.nan, pdi + mdi) * 100
    return ewm(dx, 1 / n, n), np.nan_to_num(pdi), np.nan_to_num(mdi)


def ge(x, v):
    with np.errstate(invalid="ignore"):
        return x >= v


def le(x, v):
    with np.errstate(invalid="ignore"):
        return x <= v


def gt(x, v):
    with np.errstate(invalid="ignore"):
        return x > v


def shift(x, k=1, fill=np.nan):
    out = np.empty_like(x, dtype=float)
    out[:k] = fill
    out[k:] = x[:-k]
    return out


# ---------------- 전략 ----------------
@dataclass(frozen=True)
class Cfg:
    key: str
    so: float
    sb: float
    co: float
    cb: float
    ro: float
    rb: float
    prev_need: int
    cross_need: int
    close_need: int
    prev2: bool = False  # SRC: 직전 두 봉 중 하나
    sl: float = 0.0  # V2 전략 안 손절(종가)
    trail: float = 0.0
    adx: bool = False
    s_so: float = 8.0
    s_co: float = -150.0
    s_ro: float = 42.0
    s_prev: int = 1
    s_cross: int = 2
    s_scr: float = 0.10
    cooldown: int = 0
    vol_and: bool = False
    obv_k: int = 3
    obv_min: float = 0.3
    vwap_min: float = 0.0
    flex: bool = False
    flex_n: int = 1
    flex_sl: float = 0.010
    flex_tr: float = 0.012


SRC = Cfg("SRC", 20, 80, -100, 100, 30, 70, 2, 1, 3, prev2=True)
V2N = Cfg("V2N", 10, 80, -130, 150, 44, 75, 3, 2, 3, sl=0.03, trail=0.05)
V2A = replace(V2N, key="V2A", so=12, co=-125, prev_need=1, close_need=2, adx=True, cooldown=1,
              vol_and=True, flex=True)  # fmt: skip
CFGS = {"SRC": SRC, "V2N": V2N, "V2A": V2A}


def chart_ind(ch: dict) -> dict:
    h, l, c = ch["h"], ch["l"], ch["c"]
    k, cc, r = stoch_fast(h, l, c), cci(h, l, c), rsi5(c)
    scr = np.clip(((k - 50) / 50 + cc / 200 + (r - 50) / 50) / 3, -1.5, 1.5)
    return {"k": k, "cci": cc, "rsi": r, "scr": scr}


def daily_adx_prev(ch: dict) -> tuple:
    """전날 마감 일봉(UTC)의 ADX·+DI·−DI. 원 코드는 같은 날 일봉을 붙여 미래를 본다 → 바꿈(등록 6)."""
    d = agg(ch, DAY)
    a, p, m = adx14(d["h"], d["l"], d["c"])
    idx = np.searchsorted(d["t"], ch["t"] // DAY * DAY) - 1  # 전날
    ok = idx >= 0
    j = np.clip(idx, 0, None)
    return (np.where(ok, np.nan_to_num(a[j]), 0.0), np.where(ok, p[j], 0.0), np.where(ok, m[j], 0.0))


def strategy(ch: dict, ind: dict, cfg: Cfg) -> dict:
    k, cc, r, scr, c, h = ind["k"], ind["cci"], ind["rsi"], ind["scr"], ch["c"], ch["h"]
    osc = le(k, cfg.so).astype(int) + le(cc, cfg.co).astype(int) + le(r, cfg.ro).astype(int)
    obc = ge(k, cfg.sb).astype(int) + ge(cc, cfg.cb).astype(int) + ge(r, cfg.rb).astype(int)
    pk, pc, pr = shift(k), shift(cc), shift(r)
    cross = ((le(pk, cfg.so) & gt(k, cfg.so)).astype(int) + (le(pc, cfg.co) & gt(cc, cfg.co)).astype(int)
             + (le(pr, cfg.ro) & gt(r, cfg.ro)).astype(int))  # fmt: skip
    p1 = shift(osc.astype(float), 1, 0) >= cfg.prev_need
    ready = (p1 | (shift(osc.astype(float), 2, 0) >= cfg.prev_need)) if cfg.prev2 else p1
    raw_open = ready & (cross >= cfg.cross_need)
    if cfg.adx:
        dadx, _, _ = daily_adx_prev(ch)
        sosc = le(k, cfg.s_so).astype(int) + le(cc, cfg.s_co).astype(int) + le(r, cfg.s_ro).astype(int)
        scross = ((le(pk, cfg.s_so) & gt(k, cfg.s_so)).astype(int) + (le(pc, cfg.s_co) & gt(cc, cfg.s_co)).astype(int)
                  + (le(pr, cfg.s_ro) & gt(r, cfg.s_ro)).astype(int))  # fmt: skip
        strong = dadx >= 25
        strict = (shift(sosc.astype(float), 1, 0) >= cfg.s_prev) & (scross >= cfg.s_cross) & ge(scr, cfg.s_scr)
        raw_open = (~strong & raw_open) | (strong & strict)
    if cfg.vol_and:
        dirc = np.r_[0.0, np.diff(c)]
        obv = np.cumsum(np.where(dirc >= 0, ch["v"], -ch["v"]))
        slope = obv - shift(obv, cfg.obv_k, np.nan)
        slope = np.nan_to_num(slope)
        day = ch["t"] // DAY
        tp = (ch["h"] + ch["l"] + c) / 3
        starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1]
        seg = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(c)]))
        cpv = np.cumsum(tp * ch["v"])
        cv = np.cumsum(ch["v"])
        base_pv = np.r_[0.0, cpv][starts][seg]
        base_v = np.r_[0.0, cv][starts][seg]
        with np.errstate(invalid="ignore", divide="ignore"):
            vwap = (cpv - base_pv) / (cv - base_v)
        vwap = np.where(np.isfinite(vwap), vwap, c)
        raw_open = raw_open & (slope >= cfg.obv_min) & ((c / vwap - 1) >= cfg.vwap_min)
    raw_close = (shift(obc.astype(float), 1, 0) < cfg.close_need) & (obc >= cfg.close_need)
    n = len(c)
    bo, bc = np.zeros(n, bool), np.zeros(n, bool)
    inpos, ep, hi, cool = False, 0.0, 0.0, 0
    ro, rc = raw_open.tolist(), raw_close.tolist()
    cl, hl, oc, bcn = c.tolist(), h.tolist(), osc.tolist(), obc.tolist()
    use_stops = cfg.sl > 0
    for i in range(n):
        if cool > 0:
            cool -= 1
        if not inpos:
            if ro[i] and cool == 0:
                bo[i], inpos, ep, hi = True, True, cl[i], hl[i]
            continue
        if not use_stops:
            if rc[i]:
                bc[i], inpos = True, False
            continue
        hi = max(hi, hl[i])
        sl, tr = cfg.sl, cfg.trail
        if cfg.flex and max(oc[i], bcn[i]) >= cfg.flex_n:
            sl, tr = sl + cfg.flex_sl, tr + cfg.flex_tr
        slh = cl[i] <= ep * (1 - sl)
        trh = cl[i] <= hi * (1 - tr)
        if slh or trh or rc[i]:
            bc[i], inpos = True, False
            if cfg.cooldown and (slh or trh):
                cool = cfg.cooldown
    return {"bo": bo, "bc": bc, "scr": scr}


# ---------------- 큰 움직임(추격 한도 3%) ----------------
def direction(h, l, c):
    from btc_fractal import cci as cci_s
    from btc_fractal import rsi as rsi14

    k, r = cci_s(h, l, c, 20), rsi14(c, 14)
    with np.errstate(invalid="ignore"):
        return np.where((k > 0) & (r > 50), 1, np.where((k < 0) & (r < 50), -1, 0))


def dir_at(b5: dict, period: int, t_end: np.ndarray) -> np.ndarray:
    a = agg(b5, period)
    d = direction(a["h"], a["l"], a["c"])
    ends = a["t"] + period
    idx = np.searchsorted(ends, t_end, "right") - 1
    return np.where(idx >= 0, d[np.clip(idx, 0, None)], 0)


# ---------------- 엔진 ----------------
def engine(B, L, S, a, z, *, stop=0.01, as_coded=False, delay=1, fee=FEE, bo_l=None, bo_s=None):
    """판단 봉 j → j+1+delay 봉 시가 체결. 결과: 거래 목록, 빈도."""
    t, o, h, l, c = B["t"], B["o"], B["h"], B["l"], B["c"]
    BL = L["bo"] if bo_l is None else bo_l
    BS = S["bo"] if bo_s is None else bo_s
    CL, CS, SL_, SS = L["bc"], S["bc"], L["scr"], S["scr"]
    big = B["big"]
    ft, fcum = B["ft"], B["fcum"]
    trades, cnt = [], {"chase_skip": 0, "switch": 0, "stop": 0}
    pos = None
    last = z - 1

    def close(xi, px, why):
        nonlocal pos
        d, ep, e = pos["d"], pos["ep"], pos["e"]
        stop_at = xi + 1 if why == "end" else xi  # 시가 청산이면 그 봉의 나머지는 안 겪는다
        worst = d * (px / ep - 1)
        if stop_at > e:
            seg_l, seg_h = l[e:stop_at].min(), h[e:stop_at].max()
            worst = min(worst, (seg_l / ep - 1) if d == 1 else -(seg_h / ep - 1))
        t0, t1 = int(t[e]), int(t[xi]) if why != "end" else int(t[xi]) + (int(t[1]) - int(t[0]))
        i0, i1 = np.searchsorted(ft, t0, "right"), np.searchsorted(ft, t1, "right")
        fu = -d * float(fcum[i1] - fcum[i0])
        trades.append({"d": d, "j": pos["j"], "e": e, "x": xi, "ep": ep, "xp": px, "why": why,
                       "r": d * (px / ep - 1) - 2 * fee + fu, "worst": worst, "fund": fu})  # fmt: skip
        pos = None

    def open_(d, j, ei):
        nonlocal pos
        lim = 0.03 if big[j] else 0.01
        if abs(o[ei] / c[j] - 1) > lim:
            cnt["chase_skip"] += 1
            return
        pos = {"d": d, "j": j, "e": ei, "ep": o[ei]}

    for j in range(a, z):
        ei = j + 1 + delay
        if ei > last:
            break
        if pos and ei <= pos["e"]:
            continue  # 직전 주문이 아직 체결 전
        cand = []
        if BL[j]:
            cand.append((SL_[j], 1))
        if BS[j]:
            cand.append((SS[j], -1))
        chosen = None
        if pos and ((pos["d"] == 1 and BS[j]) or (pos["d"] == -1 and BL[j])):
            chosen = -pos["d"]
        elif cand:
            chosen = max(cand, key=lambda x: np.nan_to_num(x[0], nan=-9))[1]
        if pos:
            d, ep = pos["d"], pos["ep"]
            stop_hit = False
            if stop:
                if d == 1 or not as_coded:
                    ret = (c[j] / ep - 1) if d == 1 else (ep / c[j] - 1)
                else:
                    ret = c[j] / ep - 1  # 원 코드: 진입가/현재가 − 1 을 인버스 가격으로 계산
                stop_hit = ret <= -stop
            if chosen is not None and chosen != d:
                cnt["switch"] += 1
                close(ei, o[ei], "switch")
                open_(chosen, j, ei)
            elif stop_hit:
                cnt["stop"] += 1
                close(ei, o[ei], "stop")
            elif (d == 1 and CL[j]) or (d == -1 and CS[j]):
                close(ei, o[ei], "overheat")
        elif chosen is not None:
            open_(chosen, j, ei)
    if pos:
        close(last, c[last], "end")
    return trades, cnt


def summ(tr: list, cnt: dict) -> dict:
    r = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    out = {"n": len(tr), "nl": sum(x["d"] == 1 for x in tr), "ns": sum(x["d"] == -1 for x in tr),
           "wins": int((r > 0).sum()), "win": float((r > 0).mean()) if len(r) else None,
           "avg": float(r.mean()) if len(r) else None, **cnt}  # fmt: skip
    for L_ in (1, 5):
        p = lev_path(tr, L_)
        out[f"x{L_}"] = {kk: float(vv) for kk, vv in p.items()}
    return out


# ---------------- 준비 ----------------
def windows(t: np.ndarray, end: int) -> dict:
    w = {"d30": (end - 30 * DAY, end), "m6": (end - 183 * DAY, end), "year": (ms("2025-11-01"), end),
         "mid": (ms("2022-01-01"), ms("2025-11-01")), "fresh": (ms("2020-03-01"), ms("2021-11-01"))}  # fmt: skip
    return {k: (int(np.searchsorted(t, a_)), int(np.searchsorted(t, z_))) for k, (a_, z_) in w.items()}


def build(b5: dict, tf: str) -> dict:
    B = b5 if tf == "5m" else agg(b5, M15)
    B = dict(B)
    step = M5 if tf == "5m" else M15
    tend = B["t"] + step
    d15, d4, d1 = dir_at(b5, M15, tend), dir_at(b5, H4, tend), dir_at(b5, DAY, tend)
    B["big"] = (d15 != 0) & (d15 == d4) & (d15 == d1)
    B["ft"], B["fcum"] = b5["ft"], np.r_[0.0, np.cumsum(b5["fr"])]
    inv = {"t": B["t"], "o": 1 / B["o"], "c": 1 / B["c"], "h": 1 / B["l"], "l": 1 / B["h"], "v": B["v"]}
    I_L, I_S = chart_ind(B), chart_ind(inv)
    strat = {name: (strategy(B, I_L, cfg), strategy(inv, I_S, cfg)) for name, cfg in CFGS.items()}
    return {"B": B, "IL": I_L, "strat": strat, "W": windows(B["t"], int(B["t"][-1]) + step)}


VARIANTS = [(s, x, tf, False) for tf in ("15m", "5m") for s in ("SRC", "V2N", "V2A") for x in ("X0", "X1")]
VARIANTS.append(("SRC", "X1", "15m", True))
assert len(VARIANTS) == N_VAR


def vname(v) -> str:
    return f"{v[0]}|{v[1]}|{v[2]}" + ("|asis" if v[3] else "")


def run_v(T, v, a, z, **kw):
    L, S = T["strat"][v[0]]
    return engine(T["B"], L, S, a, z, stop=0.01 if v[1] == "X1" else 0.0, as_coded=v[3], **kw)


def random_p(T, v, a, z, real, nsim, seed):
    rng = np.random.default_rng(seed)
    L, S = T["strat"][v[0]]
    nL, nS = int(L["bo"][a:z].sum()), int(S["bo"][a:z].sum())
    vals = []
    for _ in range(nsim):
        rl, rs = np.zeros(len(L["bo"]), bool), np.zeros(len(L["bo"]), bool)
        idx = rng.choice(np.arange(a, z - 3), nL + nS, replace=False)
        rl[idx[:nL]] = True
        rs[idx[nL:]] = True
        tr, _ = run_v(T, v, a, z, bo_l=rl, bo_s=rs)
        vals.append(lev_path(tr, 5)["ret"])
    vals = np.array(vals)
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "nsim": nsim}


def by_year(T, tr):
    out: dict = {}
    for x in tr:
        y = datetime.fromtimestamp(int(T["B"]["t"][x["e"]]) / 1000, timezone.utc).year
        out.setdefault(y, []).append(x)
    return {y: {"n": len(v), "x5": float(lev_path(v, 5)["ret"]), "x1": float(lev_path(v, 1)["ret"])}
            for y, v in sorted(out.items())}  # fmt: skip


def jsonable(tr):
    return [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (np.integer,)) else v)
             for k, v in x.items()} for x in tr]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b5 = load5()
    print("data", b5["info"], round(time.time() - t0, 1), flush=True)
    TT = {tf: build(b5, tf) for tf in ("15m", "5m")}
    print("built", round(time.time() - t0, 1), flush=True)
    res, keep = {}, {}
    for v in VARIANTS:
        T = TT[v[2]]
        name = vname(v)
        res[name] = {}
        for w, (a, z) in T["W"].items():
            tr, cnt = run_v(T, v, a, z)
            res[name][w] = summ(tr, cnt)
            if w in ("d30", "m6"):
                keep.setdefault(name, {})[w] = jsonable(tr)
        r = res[name]
        r["gates"] = {"fresh_x5": r["fresh"]["x5"]["ret"] > 0, "mid_x5": r["mid"]["x5"]["ret"] > 0,
                      "x1_pos": r["fresh"]["x1"]["ret"] > 0 and r["mid"]["x1"]["ret"] > 0,
                      "liq0": r["fresh"]["x5"]["liq"] == 0 and r["mid"]["x5"]["liq"] == 0}  # fmt: skip
        print(name, {w: (x["n"], round(x["win"] or 0, 2), round(x["x5"]["ret"], 3)) for w, x in r.items()
                     if w != "gates"}, flush=True)  # fmt: skip
    pick = sorted(res, key=lambda n: res[n]["fresh"]["x5"]["ret"] + res[n]["mid"]["x5"]["ret"], reverse=True)[:2]
    rnd, red = {}, {}
    for name in pick:
        v = next(x for x in VARIANTS if vname(x) == name)
        T = TT[v[2]]
        rnd[name] = {}
        for w in ("fresh", "mid"):
            a, z = T["W"][w]
            rnd[name][w] = random_p(T, v, a, z, res[name][w]["x5"]["ret"], args.nsim, 7)
            print("random", name, w, rnd[name][w], flush=True)
        red[name] = {}
        for w in ("fresh", "mid", "year"):
            a, z = T["W"][w]
            red[name][w] = {"delay0": summ(*run_v(T, v, a, z, delay=0)),
                            "fee15": summ(*run_v(T, v, a, z, fee=FEE * 1.5))}  # fmt: skip
        a, z = T["W"]["fresh"][0], len(T["B"]["t"])
        red[name]["by_year"] = by_year(T, run_v(T, v, a, z)[0])
    for name, r in res.items():
        p = rnd.get(name, {}).get("mid", {}).get("p")
        r["gates"]["random_bonf"] = bool(p is not None and p * N_VAR < 0.05)
        r["pass"] = all(r["gates"].values())
    T15 = TT["15m"]
    (out / "result.json").write_text(json.dumps(
        {"info": b5["info"], "n_variants": N_VAR, "n_tonight": N_TONIGHT_BEFORE + N_VAR, "pick": pick,
         "passed": [n for n in res if res[n]["pass"]], "res": res, "random": rnd, "redteam": red,
         "windows15": {w: [int(T15["B"]["t"][a]), int(T15["B"]["t"][z - 1])] for w, (a, z) in T15["W"].items()},
         "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    (out / "trades.json").write_text(json.dumps(keep))
    for tf, T in TT.items():
        a = T["W"]["m6"][0] - 50
        B = T["B"]
        np.savez_compressed(out / f"bars_{tf}.npz", **{k: B[k][a:] for k in ("t", "o", "h", "l", "c")},
                            k=T["IL"]["k"][a:], cci=T["IL"]["cci"][a:], rsi=T["IL"]["rsi"][a:], a=a)  # fmt: skip
    print("done", pick, [n for n in res if res[n]["pass"]], round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
