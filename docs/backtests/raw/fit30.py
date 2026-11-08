"""최근 30일 맞춤 가중 점수(SCR+α) 롱/숏. 등록: 2026-11-08-btc-30d-fit-hypotheses.md

맞춤 구간 F(캐시 마지막 30일)에서 가중치·θ·진입·청산을 고르고, 같은 값을 고정해 확장 구간에서 시험한다.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/fit30-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/fit30.py --out $OUT fit
  ... fit30.py --out $OUT ext      # 확장 구간·무작위·레드팀
  ... fit30.py --out $OUT wf       # 걷기 시험(매주 다시 맞춤)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scr15_timing as X
from btc_fractal import cci, crossed, recent, rsi, stoch
from pine_eval import lev_path

M15, DAY = 900_000, 86_400_000
TAKER, MAKER = 0.0005, 0.0002
KST = timezone(timedelta(hours=9))
CACHE = Path("~/.minerva/research/round5-2026-11-08/data15.npz").expanduser()
FEATS = ("scr", "scr3", "macd", "macd_d", "dmi", "vwap", "obv", "bb", "ema20", "ema50", "prem", "flow", "hf")
LAB = {"scr": "SCR선", "scr3": "SCR 3단", "macd": "MACD 히스토", "macd_d": "MACD 기울기", "dmi": "DMI",
       "vwap": "VWAP 이격", "obv": "OBV 기울기", "bb": "볼린저 %B", "ema20": "EMA20 기울기",
       "ema50": "EMA50 기울기", "prem": "프리미엄 z", "flow": "체결 쏠림", "hf": "1h·4h 방향"}  # fmt: skip
QS = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97)  # |s| 분위수 → θ


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=KST).timestamp() * 1000)


def ema(x, n):
    a, out, m = 2 / (n + 1), np.empty(len(x)), x[0]
    for i, v in enumerate(x):
        m = v if i == 0 else m + a * (v - m)
        out[i] = m
    return out


def wilder(x, n):
    out, m = np.empty(len(x)), np.nan
    for i, v in enumerate(x):
        m = v if i == 0 or np.isnan(m) else m + (v - m) / n
        out[i] = m
    return out


def load() -> dict:
    d = np.load(CACHE)
    b = {k: d[k].astype(float) for k in ("o", "h", "l", "c", "v", "tbv", "fr", "f_zP")}
    b["t"], b["ft"] = d["t"].astype(np.int64), d["ft"].astype(np.int64)
    return b


def raw_features(b: dict) -> dict:
    h, l, c, o, v = b["h"], b["l"], b["c"], b["o"], b["v"]
    k, cc, rr = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = wilder(tr, 14)
    lx = recent(crossed(k, 20, True)) & recent(crossed(cc, -100, True)) & recent(crossed(rr, 30, True))
    sx = recent(crossed(k, 80, False)) & recent(crossed(cc, 100, False)) & recent(crossed(rr, 70, False))
    m = ema(c, 12) - ema(c, 26)
    hist = m - ema(m, 9)
    up, dn = h - np.r_[h[0], h[:-1]], np.r_[l[0], l[:-1]] - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi, mdi = 100 * wilder(pdm, 14) / atr, 100 * wilder(mdm, 14) / atr
    dx = 100 * abs(pdi - mdi) / np.where(pdi + mdi == 0, 1, pdi + mdi)
    adx = wilder(dx, 14)
    day = b["t"] // DAY  # UTC 날짜
    tp = (h + l + c) / 3
    vwap = np.empty(len(c))
    s_pv = s_v = 0.0
    for i in range(len(c)):
        if i == 0 or day[i] != day[i - 1]:
            s_pv = s_v = 0.0
        s_pv += tp[i] * v[i]
        s_v += v[i]
        vwap[i] = s_pv / s_v if s_v else c[i]
    sv = np.sign(c - pc) * v
    cs, cv = np.r_[0, np.cumsum(sv)], np.r_[0, np.cumsum(v)]
    n8 = np.arange(len(c))
    lo8 = np.maximum(n8 - 7, 0)
    obv = (cs[n8 + 1] - cs[lo8]) / np.maximum(cv[n8 + 1] - cv[lo8], 1e-9)
    cs20 = np.r_[0, np.cumsum(c)]
    cq20 = np.r_[0, np.cumsum(c * c)]
    lo20 = np.maximum(n8 - 19, 0)
    cnt = n8 + 1 - lo20
    mu = (cs20[n8 + 1] - cs20[lo20]) / cnt
    sd = np.sqrt(np.maximum((cq20[n8 + 1] - cq20[lo20]) / cnt - mu * mu, 1e-12))
    bb = (c - (mu - 2 * sd)) / (4 * sd) - 0.5
    e20, e50 = ema(c, 20), ema(c, 50)
    lag = lambda x, n: np.r_[np.full(n, x[0]), x[:-n]]  # noqa: E731
    imb = np.where(v > 0, 2 * b["tbv"] / np.maximum(v, 1e-9) - 1, 0.0)
    ci = np.r_[0, np.cumsum(imb)]
    lo4 = np.maximum(n8 - 3, 0)
    flow = (ci[n8 + 1] - ci[lo4]) / (n8 + 1 - lo4)
    d15 = X.direction(h, l, c)
    d1h, d4h, d1d = X.hf_dir(b, 3_600_000), X.hf_dir(b, 4 * 3_600_000), X.hf_dir(b, DAY)
    f = {
        "scr": ((k - 50) / 50 + cc / 200 + (rr - 50) / 50) / 3,
        "scr3": lx.astype(float) - sx.astype(float),
        "macd": hist / atr,
        "macd_d": (hist - lag(hist, 1)) / atr,
        "dmi": (pdi - mdi) / 100 * adx / 25,
        "vwap": (c / vwap - 1) / (atr / c),
        "obv": obv,
        "bb": bb,
        "ema20": (e20 - lag(e20, 4)) / atr,
        "ema50": (e50 - lag(e50, 4)) / atr,
        "prem": np.clip(np.nan_to_num(b["f_zP"]), -5, 5),
        "flow": flow,
        "hf": (d1h + d4h) / 2.0,
    }
    f = {kk: np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0) for kk, vv in f.items()}
    big = (d15 != 0) & (d15 == d4h) & (d15 == d1d)
    return {"f": f, "atr": atr, "k": k, "cci": cc, "rsi": rr, "big": big, "d15": d15, "hod": (b["t"] // M15) % 96}


def standardize(R: dict, a: int, z: int) -> tuple[np.ndarray, dict]:
    st, cols = {}, []
    for name in FEATS:
        x = R["f"][name]
        mu, sd = float(x[a:z].mean()), float(x[a:z].std()) or 1.0
        st[name] = (mu, sd)
        cols.append((x - mu) / sd)
    return np.clip(np.vstack(cols), -4, 4), st


def apply_std(R: dict, st: dict) -> np.ndarray:
    return np.clip(np.vstack([(R["f"][n] - st[n][0]) / st[n][1] for n in FEATS]), -4, 4)


# ---------------- 시뮬레이션 ----------------
def sim(b, R, s, theta, P, a, z, sig=None, delay=0, feem=1.0) -> list:
    """P: entry(M/C/L), k, exit(T1/T2/R/Z/TR), tp, sl, slk('pct'/'atr'), T."""
    o, h, l, c, t = b["o"], b["h"], b["l"], b["c"], b["t"]
    atr, big = R["atr"], R["big"]
    if sig is None:
        sig = np.where(s >= theta, 1, np.where(s <= -theta, -1, 0))
    fcum = np.r_[0.0, np.cumsum(b["fr"])]
    ft = b["ft"]
    tr, pos, pend = [], None, None
    ent, k, ex, T = P["entry"], P.get("k", 0.0), P["exit"], P["T"]

    def fund(d, t0, t1):
        i0, i1 = np.searchsorted(ft, t0, "right"), np.searchsorted(ft, t1, "right")
        return -d * (fcum[i1] - fcum[i0])

    def close(i, px, why, maker=False):
        nonlocal pos
        d, ep = pos["d"], pos["ep"]
        fee = (pos["fee"] + (MAKER if maker else TAKER)) * feem
        g = d * (px / ep - 1)
        fu = fund(d, pos["t"], int(t[i]) + M15)
        tr.append({"d": d, "e": pos["e"], "x": i, "ep": ep, "xp": px, "r": g - fee + fu,
                   "worst": pos["worst"] - pos["fee"] * feem, "why": why, "sig": pos["sig"]})  # fmt: skip
        pos = None

    def open_(i, d, px, maker, si):
        nonlocal pos
        a_ = atr[si]
        sl = P["sl"] * (px if P["slk"] == "pct" else a_)
        tp = P.get("tp", 0) * (px if P.get("tpk") == "pct" else a_)
        pos = {"d": d, "e": i, "ep": px, "t": int(t[i]), "fee": MAKER if maker else TAKER,
               "stop": px - d * sl, "tp": px + d * tp if ex in ("T1", "T2") else None,
               "best": px, "trail": a_ * 1.5, "worst": 0.0, "sig": si}  # fmt: skip

    i = a
    while i < z - 2:
        # 1) 대기 중인 진입
        if pos is None and pend is not None:
            d, si, kind, lim, until = pend
            if kind == "L":
                if (d == 1 and l[i] < lim) or (d == -1 and h[i] > lim):
                    open_(i, d, lim, True, si)
                    pend = None
                elif i >= until:
                    pend = None
            elif kind == "C" and i >= until:  # 신호 다음 봉(si+1) 종가 확인 뒤 si+2 시가
                pend = None
                if (d == 1 and c[si + 1] > c[si]) or (d == -1 and c[si + 1] < c[si]):
                    if abs(o[i] / c[si] - 1) <= (0.03 if big[si] else 0.01):
                        open_(i, d, o[i], False, si)
        # 2) 보유 중: 봉 안 손절 → 익절 → 추적
        if pos is not None:
            d = pos["d"]
            adv = (l[i] / pos["ep"] - 1) if d == 1 else (1 - h[i] / pos["ep"])
            pos["worst"] = min(pos["worst"], adv)
            hit_stop = (d == 1 and l[i] <= pos["stop"]) or (d == -1 and h[i] >= pos["stop"])
            if hit_stop:
                px = o[i] if (d == 1 and o[i] < pos["stop"]) or (d == -1 and o[i] > pos["stop"]) else pos["stop"]
                close(i, px, "stop")
            elif pos["tp"] is not None and ((d == 1 and h[i] >= pos["tp"]) or (d == -1 and l[i] <= pos["tp"])):
                close(i, pos["tp"], "tp", maker=True)
            else:
                if ex == "TR":
                    pos["best"] = max(pos["best"], h[i]) if d == 1 else min(pos["best"], l[i])
                    pos["stop"] = max(pos["stop"], pos["best"] - pos["trail"]) if d == 1 else \
                        min(pos["stop"], pos["best"] + pos["trail"])  # fmt: skip
                if i - pos["e"] + 1 >= T:
                    close(i, c[i], "time")
                elif ex == "Z" and np.sign(s[i]) == -d:
                    close(i, c[i], "zero")
        # 3) 새 신호 (봉 i 마감)
        q = sig[i - delay] if i - delay >= a else 0
        if q != 0:
            if pos is not None and ex == "R" and pos["d"] == -q:
                close(i, c[i], "rev")
            if pos is None and pend is None:
                if ent == "M":
                    if abs(o[i + 1] / c[i] - 1) <= (0.03 if big[i] else 0.01):
                        open_(i + 1, q, o[i + 1], False, i)
                elif ent == "C":
                    pend = (q, i, "C", 0.0, i + 2)
                else:
                    pend = (q, i, "L", c[i] - q * k * atr[i], i + 4)
        i += 1
    if pos is not None:
        close(z - 1, c[z - 1], "end")
    return tr


def summ(tr: list, days: float) -> dict:
    if not tr:
        return {"n": 0, "per_day": 0.0, "win": None, "long": 0, "short": 0,
                "x5": {"ret": 0.0, "mdd": 0.0, "liq": 0}, "x1": {"ret": 0.0, "mdd": 0.0, "liq": 0}}  # fmt: skip
    r = np.array([x["r"] for x in tr])
    out = {"n": len(tr), "per_day": len(tr) / days, "win": float((r > 0).mean()),
           "long": sum(x["d"] == 1 for x in tr), "short": sum(x["d"] == -1 for x in tr),
           "avg": float(r.mean()), "stops": sum(x["why"] == "stop" for x in tr)}  # fmt: skip
    for L in (1, 5):
        p = lev_path(tr, L)
        out[f"x{L}"] = {kk: float(vv) for kk, vv in p.items()}
    return out


def objective(sm: dict, days: float) -> float:
    if sm["n"] < days or sm["n"] == 0:
        return -10 + sm["n"] / days
    fl, fs = sm["long"] / sm["n"], sm["short"] / sm["n"]
    pen = 0.0
    if min(fl, fs) < 0.3:
        pen += 2 + (0.3 - min(fl, fs)) * 10
    if sm["x5"]["mdd"] > 0.25:
        pen += 2 + (sm["x5"]["mdd"] - 0.25) * 10
    if sm["x5"]["liq"] > 0:
        pen += 5
    return sm["x5"]["ret"] - pen


BASE = {"entry": "M", "exit": "R", "sl": 1.5, "slk": "atr", "T": 32}


def thetas(s, a, z):
    return [float(np.quantile(abs(s[a:z]), q)) for q in QS]


def search(b, R, Z, a, z, n_rand, rng, P=BASE, refine=True) -> list:
    days = (z - a) / 96
    res = []
    for _ in range(n_rand):
        w = np.zeros(len(FEATS))
        nz = rng.choice(len(FEATS), rng.integers(3, 13), replace=False)
        w[nz] = rng.uniform(-1, 1, len(nz))
        s = w @ Z
        for th in thetas(s, a, z):
            sm = summ(sim(b, R, s, th, P, a, z), days)
            res.append((objective(sm, days), w.copy(), th))
    res.sort(key=lambda x: -x[0])
    if not refine:
        return res[:20]
    top = []
    seen = set()
    for sc, w, th in res:
        key = tuple(np.round(w, 2))
        if key in seen:
            continue
        seen.add(key)
        top.append([sc, w, th])
        if len(top) >= 10:
            break
    for item in top:  # 좌표 미세조정
        for _ in range(2):
            for j in range(len(FEATS)):
                for dlt in (-0.25, 0.25):
                    w2 = item[1].copy()
                    w2[j] = np.clip(w2[j] + dlt, -1, 1)
                    if np.count_nonzero(w2) > 12:
                        continue
                    s = w2 @ Z
                    for th in thetas(s, a, z):
                        sm = summ(sim(b, R, s, th, P, a, z), days)
                        sc = objective(sm, days)
                        if sc > item[0]:
                            item[0], item[1], item[2] = sc, w2, th
    top.sort(key=lambda x: -x[0])
    return top + [list(x) for x in res[:10]]


def exit_grid() -> list:
    ex = [{"exit": "T1", "tp": 0.01, "tpk": "pct", "sl": 0.01, "slk": "pct"},
          {"exit": "T1", "tp": 0.01, "tpk": "pct", "sl": 1.5, "slk": "atr"}]  # fmt: skip
    ex += [{"exit": "T2", "tp": kk, "tpk": "atr", "sl": m, "slk": "atr"} for kk in (1, 2) for m in (1, 1.5)]
    ex += [{"exit": e, "sl": 1.5, "slk": "atr"} for e in ("R", "Z", "TR")]
    ents = [{"entry": "M"}, {"entry": "C"}, {"entry": "L", "k": 0.25}, {"entry": "L", "k": 0.5}]
    return [{**en, **e, "T": T} for en in ents for e in ex for T in (16, 32, 96)]


def windows(t: np.ndarray) -> dict:
    end = int(t[-1]) + M15
    f0 = end - 30 * DAY
    W = {"F": (f0, end), "P30": (f0 - 30 * DAY, f0), "D90": (end - 90 * DAY, f0), "M6": (end - 183 * DAY, f0),
         "Y1": (end - 365 * DAY, f0), "W22": (ms("2022-01-01"), ms("2025-11-01")),
         "W20": (ms("2020-03-01"), ms("2021-11-01"))}  # fmt: skip
    return {k: (int(np.searchsorted(t, x)), int(np.searchsorted(t, y))) for k, (x, y) in W.items()}


def kst(ms_: int) -> str:
    return datetime.fromtimestamp(ms_ / 1000, KST).strftime("%Y-%m-%d %H:%M")


def cmd_fit(out: Path) -> None:
    t0 = time.time()
    b = load()
    R = raw_features(b)
    W = windows(b["t"])
    a, z = W["F"]
    Z, st = standardize(R, a, z)
    rng = np.random.default_rng(20261108)
    top = search(b, R, Z, a, z, 6000, rng)
    print("stage1", round(time.time() - t0), [round(x[0], 3) for x in top[:5]], flush=True)
    days = (z - a) / 96
    best = None
    for sc0, w, th in top[:20]:
        s = w @ Z
        for P in exit_grid():
            sm = summ(sim(b, R, s, th, P, a, z), days)
            sc = objective(sm, days)
            if best is None or sc > best[0]:
                best = (sc, w, th, P, sm)
    sc, w, th, P, sm = best
    print("best", round(sc, 3), P, sm, flush=True)
    rule = {"weights": {n: round(float(x), 4) for n, x in zip(FEATS, w)}, "theta": th, "P": P,
            "std": st, "fit_window": [kst(int(b["t"][a])), kst(int(b["t"][z - 1]))], "fit": sm,
            "objective": sc, "sec": time.time() - t0}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "rule.json").write_text(json.dumps(rule, indent=1, ensure_ascii=False, default=float))


def rand_sig(b, R, tr, a, z, rng) -> np.ndarray:
    """같은 시각(15분 칸)·같은 방향 무작위 신호."""
    sig = np.zeros(len(b["t"]), int)
    hod = R["hod"]
    idx = np.arange(a, z - 3)
    by = {h_: idx[hod[idx] == h_] for h_ in range(96)}
    for x in tr:
        cand = by[hod[x["sig"]]]
        sig[rng.choice(cand)] = x["d"]
    return sig


def cmd_ext(out: Path, nsim: int) -> None:
    b = load()
    R = raw_features(b)
    W = windows(b["t"])
    rule = json.loads((out / "rule.json").read_text())
    Z = apply_std(R, rule["std"])
    w = np.array([rule["weights"][n] for n in FEATS])
    s, th, P = w @ Z, rule["theta"], rule["P"]
    res, trades = {}, {}
    rng = np.random.default_rng(7)
    for k_, (a, z) in W.items():
        days = (z - a) / 96
        tr = sim(b, R, s, th, P, a, z)
        trades[k_] = tr
        r = summ(tr, days)
        r["delay1"] = summ(sim(b, R, s, th, P, a, z, delay=1), days)["x5"]
        r["fee15"] = summ(sim(b, R, s, th, P, a, z, feem=1.5), days)["x5"]
        if k_ != "F" and tr:
            vals = []
            for _ in range(nsim if k_ in ("P30", "D90", "M6", "Y1") else max(nsim // 3, 50)):
                vals.append(lev_path(sim(b, R, s, th, P, a, z, sig=rand_sig(b, R, tr, a, z, rng)), 5)["ret"])
            vals = np.array(vals)
            r["random"] = {"p": float((vals >= r["x5"]["ret"]).mean()), "med": float(np.median(vals)),
                           "n": len(vals)}  # fmt: skip
        r["span"] = [kst(int(b["t"][a])), kst(int(b["t"][z - 1]))]
        res[k_] = r
        print(k_, r["n"], round(r["per_day"], 2), r["win"], r["x5"], r.get("random"), flush=True)
    by_year = {}
    a, z = W["W20"][0], W["F"][1]
    for x in sim(b, R, s, th, P, a, z):
        y = datetime.fromtimestamp(int(b["t"][x["e"]]) / 1000, KST).year
        by_year.setdefault(y, []).append(x)
    res["by_year"] = {y: {"n": len(v), "x5": float(lev_path(v, 5)["ret"])} for y, v in sorted(by_year.items())}
    (out / "ext.json").write_text(json.dumps(res, indent=1, default=float))
    keep = {k_: [{kk: (int(vv) if isinstance(vv, (np.integer,)) else float(vv) if isinstance(vv, (np.floating,)) else vv)
                  for kk, vv in x.items()} for x in trades[k_]] for k_ in ("F", "P30", "M6", "Y1")}  # fmt: skip
    (out / "trades.json").write_text(json.dumps(keep))


def cmd_wf(out: Path, n_rand: int) -> None:
    """매주 직전 30일로 다시 맞추고 다음 7일 매매(진입·청산은 고른 값 고정)."""
    b = load()
    R = raw_features(b)
    t = b["t"]
    rule = json.loads((out / "rule.json").read_text())
    P = rule["P"]
    end = int(t[-1]) + M15
    rng = np.random.default_rng(11)
    folds = []
    for wk in range(52, 0, -1):
        f0, f1 = end - wk * 7 * DAY - 30 * DAY, end - wk * 7 * DAY
        a, z = int(np.searchsorted(t, f0)), int(np.searchsorted(t, f1))
        z2 = int(np.searchsorted(t, f1 + 7 * DAY))
        Z, _ = standardize(R, a, z)
        top = search(b, R, Z, a, z, n_rand, rng, P=P, refine=False)
        sc, w, th = top[0]
        tr = sim(b, R, w @ Z, th, P, z, z2)
        folds.append({"start": kst(int(t[z])), "fit_obj": float(sc), "trades": tr})
        print("fold", kst(int(t[z])), len(tr), round(float(lev_path(tr, 5)["ret"]), 3), flush=True)
    res = {}
    for name, nf in (("WF6m", 26), ("WF1y", 52)):
        tr = [x for f in folds[-nf:] for x in f["trades"]]
        res[name] = summ(tr, nf * 7)
        res[name]["weekly_x5"] = [float(lev_path(f["trades"], 5)["ret"]) for f in folds[-nf:]]
    res["fold_starts"] = [f["start"] for f in folds]
    (out / "wf.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "per_day", "win", "x5")} for k, v in res.items()
                      if k.startswith("WF")}, default=float))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", choices=("fit", "ext", "wf"))
    ap.add_argument("--nsim", type=int, default=300)
    ap.add_argument("--nrand", type=int, default=800)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if a.cmd == "fit":
        cmd_fit(out)
    elif a.cmd == "ext":
        cmd_ext(out, a.nsim)
    else:
        cmd_wf(out, a.nrand)


if __name__ == "__main__":
    main()
