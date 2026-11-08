"""다이버전스 롱/숏 신호(MPL-2.0 Pine indicator 'Divergence Long/Short Signals' 논리 이식) 15분·30분. 20개.

등록: docs/backtests/2026-11-08-btc-pine-div.md. 연구 전용(주문 없음, 바이낸스 POST 없음).
원문 Pine은 저장소에 넣지 않았다. 논리만 옮겼다(CCI·RSI 피벗 다이버전스 + 거래량 등급, 하락은 거울).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/pine-div-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/pine_div.py --out $OUT [--quick]
--quick: 최근 30일 15·ALL·X 매매와 그림만(사용자 차트 먼저).
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

MIN, DAY = 60_000, 86_400_000
TAKER, MAKER, SLIP = 0.0005, 0.0002, 0.0001
N_BONF, N_BEFORE = 20, 6500
N_CUM = N_BEFORE + N_BONF
FUT = Path("~/.minerva/data/btc/futures").expanduser()
MAINW = ("fresh", "design", "val", "year")
WINS = MAINW + ("m6", "d30", "pool")
NSIM = 100
HOLD_H = 240  # 분

TIERS = {"S": ("S",), "M": ("M",), "W": ("W",), "SM": ("S", "M"), "ALL": ("S", "M", "W")}
SPEC = {}
for tf in (15, 30):
    for t in ("S", "M", "W", "SM", "ALL"):
        SPEC[f"{tf}·{t}·X"] = {"tf": tf, "tier": t, "exit": "X", "gate": False}
for tf in (15, 30):
    for t in ("SM", "W", "ALL"):
        SPEC[f"{tf}·{t}·H"] = {"tf": tf, "tier": t, "exit": "H", "gate": False}
for tf in (15, 30):
    for t in ("SM", "W"):
        SPEC[f"{tf}·{t}·X·G"] = {"tf": tf, "tier": t, "exit": "X", "gate": True}
assert len(SPEC) == N_BONF


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def norm_p(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---------------- 자료 ----------------
def fetch_tail(out: Path, name: str, interval: str, step: int, last: int) -> np.ndarray:
    """저장 자료 뒤 빈 꼬리: 공개 GET klines(키 없음, 주문 없음). 실패하면 지난번 받은 것."""
    tail = out / f"tail_{name}.json"
    rows, now, s = [], int(time.time() * 1000), last + step
    try:
        while s < now - step:
            u = (f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval={interval}"
                 f"&startTime={s}&limit=1500")  # fmt: skip
            got = [g for g in json.loads(urllib.request.urlopen(u, timeout=20).read()) if g[6] < now]
            if not got:
                break
            rows += [[g[0], float(g[1]), float(g[2]), float(g[3]), float(g[4]), float(g[5])] for g in got]
            s = got[-1][0] + step
        tail.write_text(json.dumps(rows))
    except Exception as e:  # noqa: BLE001 - 꼬리는 없어도 시험은 돈다(적어 둔다)
        print("tail fetch failed", name, e)
        rows = json.loads(tail.read_text()) if tail.exists() else []
    return np.array(rows).reshape(-1, 6)


def load_5m(out: Path) -> dict:
    a = np.loadtxt(FUT / "klines_5m.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
    a = np.vstack([a, fetch_tail(out, "5m", "5m", 300_000, int(a[-1, 0]))])
    return {"t": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": a[:, 5]}


def load_1m(out: Path) -> dict:
    parts = [np.load(p) for p in sorted((FUT / "1m").glob("*.npz"))]
    m = {k: np.concatenate([p[k] for p in parts]) for k in ("t", "o", "h", "l", "c")}
    tl = fetch_tail(out, "1m", "1m", MIN, int(m["t"][-1]))
    for i, k in enumerate(("t", "o", "h", "l", "c")):
        m[k] = np.r_[m[k], tl[:, i].astype(m[k].dtype)]
    return m


def load_funding() -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(FUT / "funding.csv", delimiter=",", skiprows=1, usecols=(0, 2))
    return a[:, 0].astype(np.int64), a[:, 1]


def agg(b: dict, minutes: int) -> dict:
    k = b["t"] // (minutes * MIN)
    st = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    en = np.r_[st[1:] - 1, len(k) - 1]
    F = {"t": k[st] * minutes * MIN, "o": b["o"][st], "c": b["c"][en],
         "h": np.maximum.reduceat(b["h"], st), "l": np.minimum.reduceat(b["l"], st),
         "v": np.add.reduceat(b["v"], st), "n": en - st + 1, "len": minutes * MIN}  # fmt: skip
    full = F["n"] == minutes // 5
    full[-1] = full[-1] and (F["t"][-1] + F["len"] <= int(time.time() * 1000))
    for x in ("t", "o", "c", "h", "l", "v"):
        F[x] = F[x][full]
    F["ct"] = F["t"] + F["len"]  # 마감 시각
    F["rsi"], F["cci"] = rsi(F["c"]), cci(F["h"], F["l"], F["c"])
    F["vsma"] = sma(F["v"], 20)
    F["dir"] = np.where((F["cci"] > 0) & (F["rsi"] > 50), 1, np.where((F["cci"] < 0) & (F["rsi"] < 50), -1, 0))
    return F


def sma(x, n):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.r_[0.0, x])
    out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def rsi(c, n=14):
    out = np.full(len(c), np.nan)
    d = np.diff(c)
    g, ls = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = g[:n].mean(), ls[:n].mean()
    out[n] = 100 - 100 / (1 + ag / al) if al else 100.0
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + ls[i]) / n
        out[i + 1] = 100 - 100 / (1 + ag / al) if al else 100.0
    return out


def cci(h, l, c, n=20):  # noqa: E741
    tp = (h + l + c) / 3
    w = np.lib.stride_tricks.sliding_window_view(tp, n)
    m = w.mean(1)
    md = np.abs(w - m[:, None]).mean(1)
    out = np.full(len(c), np.nan)
    out[n - 1 :] = (tp[n - 1 :] - m) / (0.015 * np.where(md == 0, 1, md))
    return out


def stoch(h, l, c, n=5, s=3):  # noqa: E741
    hh = np.lib.stride_tricks.sliding_window_view(h, n).max(1)
    ll = np.lib.stride_tricks.sliding_window_view(l, n).min(1)
    k = np.full(len(c), np.nan)
    k[n - 1 :] = (c[n - 1 :] - ll) / np.maximum(hh - ll, 1e-9) * 100
    return k, sma(np.nan_to_num(k, nan=50.0), s)


def last_closed(F: dict, T: np.ndarray) -> np.ndarray:
    """시각 T에 마감돼 있던 마지막 봉 위치."""
    return np.searchsorted(F["ct"], T, "right") - 1


# ---------------- 신호 ----------------
def pivots(x: np.ndarray, low: bool, L: int = 5, R: int = 5) -> np.ndarray:
    w = np.lib.stride_tricks.sliding_window_view(x, L + R + 1)
    mid = w[:, L]
    if low:
        ok = (mid < w[:, :L].min(1)) & (mid <= w[:, L + 1 :].min(1))
    else:
        ok = (mid > w[:, :L].max(1)) & (mid >= w[:, L + 1 :].max(1))
    return np.nonzero(ok)[0] + L


def signals(F: dict) -> list:
    out = []
    for d, low in ((1, True), (-1, False)):
        x = F["l"] if low else F["h"]
        prev = None
        for p in pivots(x, low):
            cur = (x[p], F["cci"][p], F["rsi"][p], F["v"][p])
            if prev is not None and not np.isnan(cur[1]) and not np.isnan(prev[1]):
                pr = cur[0] < prev[0] if low else cur[0] > prev[0]
                dc = (cur[1] > prev[1]) if low else (cur[1] < prev[1])
                dr = (cur[2] > prev[2]) if low else (cur[2] < prev[2])
                vok = cur[3] > F["vsma"][p] * 1.2 and cur[3] > prev[3]
                tier = None
                if pr and dc and dr:
                    tier = "S" if vok else "M"
                elif pr and dc:
                    tier = "W"
                if tier:
                    out.append({"p": int(p), "c": int(p + 5), "d": d, "tier": tier, "ext": float(cur[0])})
            prev = cur
    out = [s for s in out if s["c"] < len(F["c"])]
    out.sort(key=lambda s: (s["c"], s["d"]))
    return out


# ---------------- 매매 흉내 ----------------
class Sim:
    def __init__(self, m: dict, fund: tuple):
        self.m = m
        self.t = m["t"]
        self.ft, self.fr = fund

    def first(self, a, z, f):
        s, w = a, 720
        while s < z:
            e = min(z, s + w)
            hit = f(s, e)
            if hit.any():
                return s + int(np.argmax(hit))
            s, w = e, w * 4
        return -1

    def fill_stop(self, k, d, stop, slip):
        o = self.m["o"][k]
        px = min(o, stop) if d == 1 else max(o, stop)
        return px * (1 - d * slip)

    def run(self, k0, d, stop, mode, tf, fee=1.0, slip=SLIP):
        """k0 = 진입 분(그 분 시가). 반환 dict 또는 None(진입가가 손절 밖)."""
        m = self.m
        h, l, o, c, t = m["h"], m["l"], m["o"], m["c"], self.t
        e = o[k0] * (1 + d * slip)
        R = d * (e - stop)
        if R <= 0:
            return None
        F_T, F_M = TAKER * fee, MAKER * fee
        kmax = min(len(t) - 1, k0 + (HOLD_H if mode == "H" else 14 * 1440) - 1)
        adv = (lambda s_: (lambda a, z: l[a:z] <= s_)) if d == 1 else (lambda s_: (lambda a, z: h[a:z] >= s_))
        fav = (lambda v: (lambda a, z: h[a:z] >= v)) if d == 1 else (lambda v: (lambda a, z: l[a:z] <= v))
        parts = []  # (무게, 청산가, 수수료율, 청산 분)
        ka = self.first(k0, kmax + 1, adv(stop))
        if mode == "H":
            if ka >= 0:
                parts.append((1.0, self.fill_stop(ka, d, stop, slip), F_T, ka, "stop"))
            else:
                parts.append((1.0, c[kmax] * (1 - d * slip), F_T, kmax, "time"))
        else:
            l1, l2 = e + d * R, e + 2 * d * R
            be = e * (1 + TAKER) / (1 - TAKER) if d == 1 else e * (1 - TAKER) / (1 + TAKER)
            kf = self.first(k0, kmax + 1, fav(l1))
            if ka >= 0 and (kf < 0 or ka <= kf):
                parts.append((1.0, self.fill_stop(ka, d, stop, slip), F_T, ka, "stop"))
            elif kf < 0:
                parts.append((1.0, c[kmax] * (1 - d * slip), F_T, kmax, "time"))
            else:
                kb = self.first(kf + 1, kmax + 1, adv(be))
                k2 = self.first(kf, kmax + 1, fav(l2))
                if kb >= 0 and (k2 < 0 or kb <= k2):
                    parts.append((1.0, self.fill_stop(kb, d, be, slip), F_T, kb, "be"))
                elif k2 < 0:
                    parts.append((1.0, c[kmax] * (1 - d * slip), F_T, kmax, "time"))
                else:
                    parts.append((1 / 3, l2, F_M, k2, "tp"))
                    # 나머지 2/3: 매매 봉 마감마다 직전 마감 봉 저가(숏 고가)로 손절 추적
                    a, z = k2 + 1, kmax + 1
                    if a >= z:
                        parts.append((2 / 3, c[kmax] * (1 - d * slip), F_T, kmax, "time"))
                    else:
                        tfm = tf * MIN
                        bar = t[k2:z] // tfm
                        st = np.r_[0, np.nonzero(np.diff(bar))[0] + 1]
                        ext = (np.minimum.reduceat(l[k2:z], st) if d == 1 else np.maximum.reduceat(h[k2:z], st))
                        # 봉 j의 분들에 쓰는 손절 = be와 봉 0..j-1(마감된 봉) 극값의 추적
                        if d == 1:
                            trail = np.maximum(be, np.r_[-np.inf, np.maximum.accumulate(ext)[:-1]])
                        else:
                            trail = np.minimum(be, np.r_[np.inf, np.minimum.accumulate(ext)[:-1]])
                        rep = np.diff(np.r_[st, z - k2])
                        sl = np.repeat(trail, rep)[1:]  # k2+1부터
                        hit = (l[a:z] <= sl) if d == 1 else (h[a:z] >= sl)
                        if hit.any():
                            j = int(np.argmax(hit))
                            parts.append((2 / 3, self.fill_stop(a + j, d, sl[j], slip), F_T, a + j, "trail"))
                        else:
                            parts.append((2 / 3, c[kmax] * (1 - d * slip), F_T, kmax, "time"))
        r = -TAKER * fee
        for w, px, f, _, _ in parts:
            r += w * (d * (px / e - 1) - f * px / e)
        kx = max(p[3] for p in parts)
        # 펀딩: 보유 중 정산 시각마다 −d × 비율 × 남은 무게
        a_, z_ = np.searchsorted(self.ft, t[k0], "right"), np.searchsorted(self.ft, t[kx], "right")
        for i in range(a_, z_):
            w = sum(p[0] for p in parts if t[p[3]] >= self.ft[i])
            r -= d * self.fr[i] * w
        worst = min(-R / e, min(d * (p[1] / e - 1) for p in parts)) - TAKER * fee
        return {"k0": int(k0), "kx": int(kx), "d": d, "e": float(e), "stop": float(stop), "Rp": float(R / e),
                "r": float(r), "worst": float(worst), "why": parts[-1][4] if len(parts) == 1 else "tp+" + parts[-1][4],
                "exits": [(int(p[3]), float(p[1]), float(p[0])) for p in parts]}  # fmt: skip


# ---------------- 실행 ----------------
def build_trades(S: Sim, F: dict, sigs: list, spec: dict, ctx: dict, lat=0, fee=1.0, slip=SLIP, hindsight=False) -> list:
    t1 = S.t
    tiers = TIERS[spec["tier"]]
    tf = spec["tf"]
    out, busy = [], -1
    for s in sigs:
        if s["tier"] not in tiers:
            continue
        i = s["p"] if hindsight else s["c"]
        T = int(F["ct"][i])
        k0 = int(np.searchsorted(t1, T)) + lat
        if k0 >= len(t1) - 2 or k0 <= busy or t1[k0] - T > 30 * MIN + lat * MIN:
            continue
        d = s["d"]
        if spec["gate"] and (ctx["g1h"][s["c"]] == -d or ctx["g4h"][s["c"]] == -d):
            continue
        cc = F["c"][i]
        e = S.m["o"][k0]
        lim = 0.03 if ctx["agree"][s["c"]] == d else 0.01
        if not hindsight and d * (e - cc) / cc >= lim:
            continue
        tr = S.run(k0, d, s["ext"], spec["exit"], tf, fee, slip)
        if tr is None:
            continue
        tr.update({"tier": s["tier"], "pt": int(F["ct"][s["p"]]), "pext": s["ext"], "T": T})
        out.append(tr)
        busy = tr["kx"]
    return out


def ctx_for(F: dict, Fs: dict) -> dict:
    T = F["ct"]
    g = {}
    for k in ("15", "1h", "4h", "1d"):
        j = last_closed(Fs[k], T)
        g[k] = np.where(j >= 0, Fs[k]["dir"][np.clip(j, 0, None)], 0)
    agree = np.where((g["15"] == g["4h"]) & (g["4h"] == g["1d"]), g["1d"], 0)
    return {"g1h": g["1h"], "g4h": g["4h"], "agree": agree}


def windows(t_end: int) -> dict:
    W = {"fresh": (ms("2020-03-01"), ms("2021-11-01")), "design": (ms("2022-01-01"), ms("2024-07-01")),
         "val": (ms("2024-07-01"), ms("2025-11-01")), "year": (ms("2025-11-01"), t_end),
         "m6": (t_end - 183 * DAY, t_end), "d30": (t_end - 30 * DAY, t_end), "pool": (ms("2020-03-01"), ms("2025-11-01"))}  # fmt: skip
    return W


def in_w(tr, a, z):
    return [x for x in tr if a <= x["T"] < z]


def path5(tr, frac):
    eq, peak, mdd, liq = 1.0, 1.0, 0.0, 0
    for x in tr:
        if 5 * x["worst"] <= -0.95:
            liq += 1
            pos = -1.0
        else:
            pos = max(5 * x["r"], -1.0)
            mdd = max(mdd, 1 - eq * (1 + frac * 5 * x["worst"]) / peak)
        eq *= 1 + frac * pos
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
        if eq <= 0:
            return {"ret": -1.0, "mdd": 1.0, "liq": liq}
    return {"ret": eq - 1, "mdd": mdd, "liq": liq}


def lev1(x):
    """1% 위험 크기: 손절 때 왕복 수수료 포함 1% 손실, 5배 상한."""
    per = x["Rp"] + TAKER * (2 - x["Rp"])
    return min(0.01 / per, 5.0)


def path_risk(tr):
    eq, peak, mdd = 1.0, 1.0, 0.0
    for x in tr:
        eq *= 1 + lev1(x) * x["r"]
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {"ret": eq - 1, "mdd": mdd}


def summ(tr, a, z):
    n = len(tr)
    days = max((z - a) / DAY, 1)
    if not n:
        return {"n": 0, "perday": 0.0}
    r = np.array([x["r"] for x in tr])
    return {"n": n, "perday": n / days, "win": float((r > 0).mean()), "avg": float(r.mean()),
            "avgR": float(np.mean([lev1(x) * x["r"] / 0.01 for x in tr])),
            "long": sum(x["d"] == 1 for x in tr), "short": sum(x["d"] == -1 for x in tr),
            "risk1": path_risk(tr), "x5_40": path5(tr, 0.4), "x5_100": path5(tr, 1.0),
            "Rp_med": float(np.median([x["Rp"] for x in tr]))}  # fmt: skip


def random_for(S: Sim, F: dict, tr: list, spec: dict, ctx: dict, cache: dict, nsim=NSIM) -> np.ndarray:
    """거래마다 nsim개: ±15일 같은 봉 마감 시각, 같은 방향·손절 거리·청산 틀. 반환 [거래, nsim] 수익."""
    out = np.full((len(tr), nsim), np.nan)
    ct = F["ct"]
    for i, x in enumerate(tr):
        key = (spec["tf"], spec["exit"], spec["gate"], x["T"], x["d"], round(x["Rp"], 6))
        if key in cache:
            out[i] = cache[key]
            continue
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        a, z = np.searchsorted(ct, x["T"] - 15 * DAY), np.searchsorted(ct, x["T"] + 15 * DAY)
        res, tries = [], 0
        while len(res) < nsim and tries < nsim * 5:
            tries += 1
            j = int(rng.integers(a, z))
            if spec["gate"] and (ctx["g1h"][j] == -x["d"] or ctx["g4h"][j] == -x["d"]):
                continue
            k0 = int(np.searchsorted(S.t, ct[j]))
            if k0 >= len(S.t) - 2 or S.t[k0] - ct[j] > 30 * MIN:
                continue
            e = S.m["o"][k0] * (1 + x["d"] * SLIP)
            stop = e * (1 - x["d"] * x["Rp"])
            rr = S.run(k0, x["d"], stop, spec["exit"], spec["tf"])
            if rr is not None:
                res.append(rr["r"])
        v = np.array(res + [np.nan] * (nsim - len(res)))[:nsim]
        cache[key] = v
        out[i] = v
    return out


def rand_stats(tr, R):
    if not len(tr):
        return None
    real = np.array([x["r"] for x in tr])
    sim_means = np.nanmean(R, axis=0)
    sim_win = np.nanmean(R > 0, axis=0)
    sd = sim_means.std() or 1e-12
    z = (real.mean() - sim_means.mean()) / sd
    return {"ex_avg": float(real.mean() - sim_means.mean()), "ex_win": float((real > 0).mean() - sim_win.mean()),
            "rand_win": float(sim_win.mean()), "z": float(z), "p": norm_p(z),
            "p_emp": float(((sim_means >= real.mean()).sum() + 1) / (len(sim_means) + 1))}  # fmt: skip


def diagnostics(S: Sim, Fr: dict, sigs: dict, W: dict) -> dict:
    """등급별 하루 신호 수, 앞 1/4/24시간 적중률(현실·사후) vs 시각 맞춘 무작위, 사후 간격."""
    out = {}
    a, z = W["pool"]
    days = (z - a) / DAY
    c1, t1 = S.m["c"], S.t
    rng = np.random.default_rng(7)
    for tf, sg in sigs.items():
        F = Fr[tf]
        for tier in ("S", "M", "W", "ALL"):
            ss = [s for s in sg if (tier == "ALL" or s["tier"] == tier) and a <= F["ct"][s["c"]] < z]
            res = {"n": len(ss), "perday": len(ss) / days, "long": sum(s["d"] == 1 for s in ss)}
            for name, key in (("real", "c"), ("hind", "p")):
                for hh in (1, 4, 24):
                    hit, rh = [], []
                    for s in ss:
                        T = int(F["ct"][s[key]])
                        k, kk = np.searchsorted(t1, T) - 1, np.searchsorted(t1, T + hh * 3_600_000) - 1
                        if kk >= len(t1):
                            continue
                        hit.append(s["d"] * (c1[kk] - c1[k]) > 0)
                        lo, hi = np.searchsorted(F["ct"], T - 15 * DAY), np.searchsorted(F["ct"], T + 15 * DAY)
                        for j in rng.integers(lo, hi, 20):
                            Tj = int(F["ct"][j])
                            k_, kk_ = np.searchsorted(t1, Tj) - 1, np.searchsorted(t1, Tj + hh * 3_600_000) - 1
                            if kk_ < len(t1):
                                rh.append(s["d"] * (c1[kk_] - c1[k_]) > 0)
                    hr, rr = float(np.mean(hit)) if hit else None, float(np.mean(rh)) if rh else None
                    pz = None
                    if hit and rr is not None:
                        se = math.sqrt(rr * (1 - rr) / len(hit))
                        pz = norm_p((hr - rr) / se) if se else None
                    res[f"{name}_{hh}h"] = {"hit": hr, "rand": rr, "p": pz}
            gaps = [s["d"] * (F["c"][s["c"]] - F["c"][s["p"]]) / F["c"][s["p"]] for s in ss]
            dist = [s["d"] * (F["c"][s["c"]] - s["ext"]) / F["c"][s["c"]] for s in ss]
            if ss:
                res["gap_med"], res["gap_mean"] = float(np.median(gaps)), float(np.mean(gaps))
                res["gap_q"] = [float(q) for q in np.percentile(gaps, [10, 25, 75, 90])]
                res["stopdist_med"] = float(np.median(dist))
                res["stopdist_q"] = [float(q) for q in np.percentile(dist, [10, 25, 75, 90])]
            out[f"{tf}·{tier}"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b5 = load_5m(out)
    m = load_1m(out)
    S = Sim(m, load_funding())
    t_end = int(min(m["t"][-1], b5["t"][-1] + 300_000))
    Fs = {"15": agg(b5, 15), "30": agg(b5, 30), "1h": agg(b5, 60), "4h": agg(b5, 240), "1d": agg(b5, 1440)}
    for k in Fs:
        keep = Fs[k]["ct"] <= t_end
        for x in ("t", "o", "c", "h", "l", "v", "ct", "rsi", "cci", "vsma", "dir"):
            Fs[k][x] = Fs[k][x][keep]
    del b5
    Fr = {15: Fs["15"], 30: Fs["30"]}
    sigs = {tf: signals(Fr[tf]) for tf in Fr}
    ctx = {tf: ctx_for(Fr[tf], Fs) for tf in Fr}
    W = windows(t_end)
    print(f"loaded {time.time() - t0:.0f}s; t_end {datetime.fromtimestamp(t_end / 1000, timezone.utc)}; "
          f"sigs 15m {len(sigs[15])} 30m {len(sigs[30])}", flush=True)  # fmt: skip

    if args.quick:
        spec = SPEC["15·ALL·X"]
        a, z = W["d30"]
        F = Fr[15]
        sg = [s for s in sigs[15] if F["ct"][s["c"]] >= a - DAY]
        tr = in_w(build_trades(S, F, sg, spec, ctx[15]), a, z)
        hd = in_w(build_trades(S, F, sg, spec, ctx[15], hindsight=True), a, z)
        q = {"real": summ(tr, a, z), "hind": summ(hd, a, z), "trades": tr, "hind_trades": hd,
             "sigs": [{**s, "pt": int(F["ct"][s["p"]]), "cT": int(F["ct"][s["c"]])} for s in sg if F["ct"][s["c"]] >= a]}  # fmt: skip
        (out / "quick_d30.json").write_text(json.dumps(q, default=float))
        print(json.dumps({k: q[k] for k in ("real", "hind")}, default=float, indent=1))
        return

    diag = diagnostics(S, Fr, sigs, W)
    print("diag done", f"{time.time() - t0:.0f}s", flush=True)
    res, trj, cache, hind = {}, {}, {}, {}
    for name, spec in SPEC.items():
        tf = spec["tf"]
        F = Fr[tf]
        tr = build_trades(S, F, sigs[tf], spec, ctx[tf])
        r = {w: summ(in_w(tr, *W[w]), *W[w]) for w in WINS}
        # 무작위(pool + year)
        rnd = {}
        sel = [x for x in tr if W["fresh"][0] <= x["T"]]
        R = random_for(S, F, sel, spec, ctx[tf], cache)
        Ti = np.array([x["T"] for x in sel])
        for w in (*MAINW, "pool"):
            mk = (Ti >= W[w][0]) & (Ti < W[w][1])
            rnd[w] = rand_stats([x for x, k in zip(sel, mk) if k], R[mk])
        r["rand"] = rnd
        # 레드팀
        red = {}
        for rk, kw in (("lat1m", {"lat": 1}), ("lat1bar", {"lat": tf}), ("slip+2", {"slip": SLIP + 0.0002}),
                       ("fee1.5", {"fee": 1.5})):  # fmt: skip
            t2 = build_trades(S, F, sigs[tf], spec, ctx[tf], **kw)
            red[rk] = {w: summ(in_w(t2, *W[w]), *W[w]) for w in ("val", "year")}
        r["red"] = red
        yrs = {}
        for y in range(2020, 2027):
            a_, z_ = ms(f"{y}-01-01"), min(ms(f"{y + 1}-01-01"), t_end)
            yrs[y] = summ(in_w(tr, a_, z_), a_, z_)
        r["years"] = yrs
        if spec["exit"] == "X":
            th = build_trades(S, F, sigs[tf], spec, ctx[tf], hindsight=True)
            hind[name] = {w: summ(in_w(th, *W[w]), *W[w]) for w in WINS}
        res[name] = r
        trj[name] = {w: in_w(tr, *W[w]) for w in ("d30", "fresh", "design", "val", "year")}
        print(name, f"{time.time() - t0:.0f}s", " ".join(
            f"{w}:{r[w]['n']}/{100 * r[w].get('win', 0):.0f}%/{100 * r[w].get('risk1', {}).get('ret', 0):+.0f}%"
            for w in MAINW), flush=True)  # fmt: skip
    # 판정
    ps = {n: (res[n]["rand"]["pool"] or {}).get("p", 1.0) for n in SPEC}
    order = sorted(ps, key=ps.get)
    holm, run = {}, 0.0
    for i, n in enumerate(order):
        run = max(run, min(1.0, ps[n] * (N_BONF - i)))
        holm[n] = run
    verdict = {}
    for n, r in res.items():
        g1 = all(r[w]["n"] and r[w]["risk1"]["ret"] > 0 and r[w]["x5_40"]["ret"] > 0 and r[w]["x5_100"]["ret"] > 0
                 and r[w]["x5_100"]["liq"] == 0 for w in MAINW)  # fmt: skip
        g2 = holm[n] < 0.05 and all((r["rand"][w] or {}).get("ex_avg", -1) > 0 for w in MAINW)
        g3 = r["design"]["perday"] >= 0.25 and r["year"]["perday"] >= 0.25
        pl = r["pool"]
        g4 = pl["n"] and min(pl["long"], pl["short"]) >= 0.15 * pl["n"]
        g5 = all(r["red"][k][w]["n"] and r["red"][k][w]["risk1"]["ret"] > 0 for k in r["red"] for w in ("val", "year"))
        verdict[n] = {"g1": bool(g1), "g2": bool(g2), "g3": bool(g3), "g4": bool(g4), "g5": bool(g5),
                      "holm": holm[n], "cum": min(1.0, ps[n] * N_CUM), "pass": bool(g1 and g2 and g3 and g4 and g5)}  # fmt: skip
    (out / "result.json").write_text(json.dumps({"res": res, "diag": diag, "hind": hind, "verdict": verdict,
                                                  "t_end": t_end, "secs": time.time() - t0}, default=float))  # fmt: skip
    (out / "trades.json").write_text(json.dumps(trj, default=float))
    print("pass:", [n for n in verdict if verdict[n]["pass"]], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
