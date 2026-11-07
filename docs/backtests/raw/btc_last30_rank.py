"""최근 30일, 가진 BTC 기법 전부(추적기 11개 + 거울 숏 + 기록 추적 L1·S4)와 앙상블 5개 순위.

설명용 순위이고 판정 시험이 아니다. 문서: 2026-11-07-btc-last-30d-ranking.md
가격: 바이낸스 USDT-M 선물 5분봉(fapi 공개 시세로 최근 130일을 다시 받음) + 바이낸스 펀딩.
ML 신호: 추적기 모델(읽기만)을 현물 5분봉으로 다시 매긴다. 추적기 폴더에는 쓰지 않는다.
실행: PYTHONPATH=<pyml>:src:docs/backtests/raw nice python3 docs/backtests/raw/btc_last30_rank.py \
    --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import numpy as np
from btc_best_entry_vote import k2_first
from btc_ml_regime_gate import gate_g1, gate_g2
from btc_scr_learned import bars
from trend_hold_cross_market import daily_atr

KST = timezone(timedelta(hours=9))
BAR = 900_000
M5 = 300_000
DAY = 86_400_000
FEE = 0.0005
ML = (0.015, 0.015)
T75 = (0.0075, 0.015)
STOP1 = 0.01
TH1_RISK, TH1_CAP = 0.02, 3.0
LEVS = (1.0, 3.0, 10.0)
LIQ = -0.09  # 10배 교차 증거금: 이 정도 불리하면 청산 근처
HOME = Path("~/.minerva").expanduser()
TRK = HOME / "research/btc-ml-tracker"
FUT_HIST = HOME / "data/btc/futures/klines_5m.csv"
SPOT_HIST = HOME / "data/btc/5m.csv"
FAPI = "https://fapi.binance.com/fapi/v1"
SPOT_API = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1000"
TRACKER = ("A", "B", "AS", "BS", "AS4", "K2", "R1", "AS4T75", "K2B", "AS4G4", "TH1")
SINGLES = TRACKER + ("TH1S", "L1", "L1S", "S4")
ENS = ("E1", "E2", "E3", "E4", "E5")


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-research"})
    for k in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except OSError:
            if k == 3:
                raise
            time.sleep(2)


# ---- 자료 ----
def fetch_fut5(since: int, now: int) -> list:
    rows, s = [], since
    while s < now:
        k = get(f"{FAPI}/klines?symbol=BTCUSDT&interval=5m&limit=1500&startTime={s}")
        k = [x for x in k if int(x[6]) < now]
        if not k:
            break
        rows += k
        s = int(k[-1][0]) + M5
        if len(k) < 1500:
            break
    return rows


def fetch_funding(since: int) -> list:
    out, s = [], since
    while True:
        k = get(f"{FAPI}/fundingRate?symbol=BTCUSDT&limit=1000&startTime={s}")
        if not k:
            break
        out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in k]
        s = int(k[-1]["fundingTime"]) + 1
        if len(k) < 1000:
            break
    return out


def as_dict(rows: dict) -> dict:
    t = np.array(sorted(rows), dtype=float)
    v = np.array([rows[int(k)] for k in t], dtype=float)
    return {"t": t, "o": v[:, 0], "h": v[:, 1], "l": v[:, 2], "c": v[:, 3], "v": v[:, 4]}


def load_fut(out: Path, now: int, days: int = 130) -> tuple:
    """저장 이력(바이낸스 행만) + 최근 days일은 fapi로 새로 받아 덮는다."""
    cut = now - days * DAY
    cut -= cut % DAY
    rows, okx = {}, 0
    with FUT_HIST.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if x[-1] == "okx":
                okx += 1
                continue
            t = int(x[0])
            if t < cut:
                rows[t] = tuple(float(v) for v in x[1:6])
    stored = {}
    with FUT_HIST.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if int(x[0]) >= cut and x[-1] != "okx":
                stored[int(x[0])] = float(x[4])
    cache = out / "fut5m_fapi.csv"
    if cache.exists() and json.loads((out / "fetch.json").read_text())["now"] == now:
        new = list(csv.reader(cache.open()))
    else:
        new = [x[:6] for x in fetch_fut5(cut, now)]
        with cache.open("w", newline="") as f:
            csv.writer(f).writerows(new)
    diff = []
    for x in new:
        t = int(x[0])
        rows[t] = tuple(float(v) for v in x[1:6])
        if t in stored:
            diff.append(abs(float(x[4]) / stored[t] - 1))
    d = as_dict(rows)
    info = {
        "fapi_from": cut,
        "fapi_bars": len(new),
        "okx_rows_dropped": okx,
        "overlap_vs_stored": len(diff),
        "overlap_max_close_diff": float(max(diff)) if diff else None,
    }
    return d, info


def load_spot(out: Path, now: int) -> tuple:
    """현물 5분봉: 저장 이력 + 추적기 live5m(읽기만) + 그 뒤는 공개 시세."""
    rows = {}
    for p in (SPOT_HIST, TRK / "live5m.csv"):
        with p.open() as f:
            r = csv.reader(f)
            next(r)
            for x in r:
                rows[int(x[0])] = tuple(float(v) for v in x[1:6])
    last = max(rows)
    s, got = last + M5, 0
    while s < now - M5:
        k = [x for x in get(f"{SPOT_API}&startTime={s}") if int(x[6]) < now]
        if not k:
            break
        for x in k:
            rows[int(x[0])] = tuple(float(v) for v in x[1:6])
        got += len(k)
        s = int(k[-1][0]) + M5
    return as_dict(rows), {"spot_api_bars": got}


def gaps(t: np.ndarray, a_t: float, step: int = M5) -> list:
    m = t >= a_t
    dt = np.diff(t[m])
    bad = np.flatnonzero(dt != step)
    return [(float(t[m][i]), float(dt[i] / step - 1)) for i in bad]


# ---- ML 신호 (추적기 모델 그대로) ----
def ml_signals(spot5: dict, t15: np.ndarray, out: Path) -> dict:
    cache = out / "ml_sig.npz"
    if cache.exists():
        z = np.load(cache)
        if len(z["t"]) == len(t15) and (z["t"] == t15).all():
            return {k: z[k] for k in ("A", "B", "AS", "BS")}
    import btc_ml_15m as M
    import btc_ml_tracker as T

    B, X, seq, t, seq_ok = T.features(spot5)
    models, meta = T.load_models()
    seeds = T.load_seed_models(meta)
    pos = {float(v): i for i, v in enumerate(t)}
    idx = np.array([pos.get(float(v), -1) for v in t15])
    ok = (idx >= 0) & seq_ok[np.maximum(idx, 0)]
    need = idx[ok]
    Xn, Sn = T.norm(X, seq, meta["nz"])
    p = T.predict(models, Xn, Sn, need)
    pa = np.mean([p] + [T.predict(sm, Xn, Sn, need) for sm in seeds], 0)
    res = {}
    for r, pp in (("A", p), ("B", p), ("AS", pa), ("BS", pa)):
        hi, lo = meta["thr"][r]
        kk = T.RULES[r][1]
        sc = pp.mean(0)
        up, dn = (pp > M.BE).sum(0), (pp < 1 - M.BE).sum(0)
        s = np.where((sc >= hi) & (up >= kk), 1, np.where((sc <= lo) & (dn >= kk), -1, 0))
        full = np.zeros(len(t15), int)
        full[ok] = s
        res[r] = full
    np.savez(cache, t=t15, **res)
    return res


def check_vs_tracker(sig: dict, t15: np.ndarray) -> dict:
    with (TRK / "signals.csv").open() as f:
        rows = list(csv.DictReader(f))
    pos = {float(v): i for i, v in enumerate(t15)}
    out = {}
    for r in ("A", "B", "AS", "BS"):
        n = same = 0
        for x in rows:
            i = pos.get(float(x["t"]))
            if i is None or x.get(r) in (None, ""):
                continue
            n += 1
            same += int(x[r]) == int(sig[r][i])
        out[r] = {"rows": n, "same": same}
    return out


# ---- 매매 ----
def bracket(d, sig, a, b, tp, sl) -> list:
    """신호 봉 종가 진입, 목표 tp / 손절 sl. 같은 봉 둘 다면 손절, 갭이면 시가."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]  # noqa: E741
    tr, i, n = [], a, len(c)
    while i < b:
        s = int(sig[i])
        if not s:
            i += 1
            continue
        px = c[i]
        up, sp = px * (1 + tp * s), px * (1 - sl * s)
        x = None
        for j in range(i + 1, n):
            if (s > 0 and l[j] <= sp) or (s < 0 and h[j] >= sp):
                x = (j, min(o[j], sp) if s > 0 else max(o[j], sp), "손절")
                break
            if (s > 0 and h[j] >= up) or (s < 0 and l[j] <= up):
                x = (j, max(o[j], up) if s > 0 else min(o[j], up), "목표")
                break
        if x is None:
            tr.append(mk(i, s, px, n - 1, c[-1], "보유 중", True))
            break
        tr.append(mk(i, s, px, *x, False))
        i = x[0]
    return tr


def mk(i, s, px, x, xp, why, op, w=1.0) -> dict:
    return {
        "i": int(i),
        "s": int(s),
        "px": float(px),
        "x": int(x),
        "xp": float(xp),
        "why": why,
        "open": bool(op),
        "w": float(w),
    }


def r1_leg(d, reg, i, s, px):
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]  # noqa: E741
    sp = px * (1 - STOP1 * s)
    for j in range(i + 1, len(c)):
        if (s > 0 and l[j] <= sp) or (s < 0 and h[j] >= sp):
            return j, min(o[j], sp) if s > 0 else max(o[j], sp), "손절", False
        if reg[j] != s:
            if j + 1 < len(c):
                return j, o[j + 1], "방향 전환", False  # 다음 봉 시가 = 이 봉 마감
            break
    return len(c) - 1, c[-1], "보유 중", True


def r1_trades(d, reg, newday, a, b, only=0, wfn=None) -> list:
    """R1(only=0), TH1(only=1, ATR 비중), TH1S(only=-1). 손절 뒤에는 다음 일봉 마감까지 기다린다."""
    c = d["c"]
    tr, i, can = [], a, True
    while i < b:
        if newday[i]:
            can = True
        s = int(reg[i])
        if s and can and (only == 0 or s == only):
            x, xp, why, op = r1_leg(d, reg, i, s, c[i])
            tr.append(mk(i, s, c[i], x, xp, why, op, wfn(i) if wfn else 1.0))
            if op:
                break
            if why == "손절":
                can = False
            i = x if x > i else i + 1
            continue
        i += 1
    return tr


def k2_trades(d, reg, sig, a, b) -> list:
    c = d["c"]
    tr, i = [], a
    while i < b:
        s = int(reg[i])
        if s and int(sig[i]) == s:
            x, xp, why, op = r1_leg(d, reg, i, s, c[i])
            tr.append(mk(i, s, c[i], x, xp, why, op))
            if op:
                break
            i = x
            continue
        i += 1
    return tr


def l1_trades(d, a, b, side: int, look=1920, hold=96) -> list:
    """L1(side=1): 종가가 직전 20일 최고 종가를 처음 넘은 봉 → 다음 봉 시가, 96봉, 1% 손절.
    L1S(side=-1)는 거울."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]  # noqa: E741
    n = len(c)
    sig = np.zeros(n, bool)
    prev = False
    for i in range(look, n):
        w = c[i - look : i]
        now = c[i] > w.max() if side > 0 else c[i] < w.min()
        sig[i] = now and not prev
        prev = now
    tr, i = [], a
    while i < b:
        if not sig[i]:
            i += 1
            continue
        e = i + 1  # 다음 봉 시가 = i봉 마감 가격으로 기록
        px = o[e]
        sp = px * (1 - STOP1 * side)
        x = None
        for j in range(e, min(e + hold, n)):
            if (side > 0 and l[j] <= sp) or (side < 0 and h[j] >= sp):
                x = (j, min(o[j], sp) if side > 0 else max(o[j], sp), "손절")
                break
        if x is None:
            if e + hold < n:
                x = (e + hold - 1, o[e + hold], "시간")
            else:
                tr.append(mk(i, side, px, n - 1, c[-1], "보유 중", True))
                break
        tr.append(mk(i, side, px, *x, False))
        i = x[0] + 1
    return tr


def s4_trades(d15, fut5, a_t, out_end) -> list:
    """S4: 1시간봉 구조 돌파(btc_paper 정의 그대로), 뒤집기, 1% 손절. 15분봉 위치로 옮긴다."""
    from btc_fractal import resample

    from minerva.strategy.btc_paper import PBar, replay_s4

    h1 = resample(fut5, 12)
    pb = [
        PBar(int(t), o, h, lo, c)
        for t, o, h, lo, c in zip(h1["t"], h1["o"], h1["h"], h1["l"], h1["c"])
    ]
    raw = replay_s4(pb, start=int(a_t))
    t = d15["t"]
    pos = {float(v): i for i, v in enumerate(t)}
    tr = []
    for z in raw:
        i = pos[float(z.entry_t)] - 1  # 진입 봉 시가 = 앞 15분봉 마감
        if z.exit_t is None:
            tr.append(mk(i, z.side, z.entry, len(t) - 1, d15["c"][-1], "보유 중", True))
            continue
        if z.why == "stop":
            line = z.entry * (1 - z.side * STOP1)
            j0 = pos[float(z.exit_t)]
            j = j0
            for k in range(j0, min(j0 + 4, len(t))):
                if (z.side > 0 and d15["l"][k] <= line) or (z.side < 0 and d15["h"][k] >= line):
                    j = k
                    break
            tr.append(mk(i, z.side, z.entry, j, z.exit, "손절", False))
        else:
            tr.append(mk(i, z.side, z.entry, pos[float(z.exit_t)] - 1, z.exit, "뒤집기", False))
    return tr


# ---- 손익 ----
def finish(d, tr, ft, fr) -> None:
    t, h, l = d["t"], d["h"], d["l"]  # noqa: E741
    for z in tr:
        t0, t1 = t[z["i"]] + BAR, t[z["x"]] + BAR
        m = (ft > t0) & (ft <= t1)
        fund = -z["s"] * float(fr[m].sum())
        g = (z["xp"] / z["px"] - 1) * z["s"]
        lo = l[z["i"] + 1 : z["x"] + 1].min() if z["x"] > z["i"] else z["px"]
        hi = h[z["i"] + 1 : z["x"] + 1].max() if z["x"] > z["i"] else z["px"]
        z["mae"] = float((lo / z["px"] - 1) if z["s"] > 0 else -(hi / z["px"] - 1))
        z["mfe"] = float((hi / z["px"] - 1) if z["s"] > 0 else -(lo / z["px"] - 1))
        z["gross"] = float(g)
        z["fund"] = fund
        z["net"] = float(g + fund - 2 * FEE)
        z["t0"], z["t1"] = float(t0), float(t1)


def equity(d, tr, a, lev) -> tuple:
    """15분봉 종가 평가 자산(복리, 한 번에 한 포지션)과 봉 안 최악 평가."""
    c, h, l = d["c"], d["h"], d["l"]  # noqa: E741
    n = len(c)
    eq, worst = np.ones(n - a), np.ones(n - a)
    cap, k = 1.0, 0
    for idx in range(a, n):
        while k < len(tr) and not tr[k]["open"] and tr[k]["x"] < idx:
            z = tr[k]
            cap *= max(1 + lev * z["w"] * z["net"], 0.0)
            k += 1
        val = wv = cap
        if k < len(tr) and tr[k]["i"] < idx:
            z = tr[k]
            s, w = z["s"], z["w"]
            mtm = (c[idx] / z["px"] - 1) * s - FEE
            adv = ((l[idx] if s > 0 else h[idx]) / z["px"] - 1) * s - FEE
            val = cap * max(1 + lev * w * mtm, 0.0)
            wv = cap * max(1 + lev * w * adv, 0.0)
        eq[idx - a], worst[idx - a] = val, min(val, wv)
    return eq, worst


def summary(d, tr, a) -> dict:
    if not tr:
        r = {"n": 0, "long": 0, "short": 0, "open": 0, "win": None, "avg": None}
        for lev in LEVS:
            r[f"ret{int(lev)}"], r[f"mdd{int(lev)}"] = 0.0, 0.0
        r["liq10"], r["min_intrabar10"], r["max_mae"] = 0, 0.0, 0.0
        r["side"] = {}
        return r
    net = np.array([z["net"] * z["w"] for z in tr])
    r = {
        "n": len(tr),
        "long": sum(z["s"] > 0 for z in tr),
        "short": sum(z["s"] < 0 for z in tr),
        "open": sum(z["open"] for z in tr),
        "win": round(float((net > 0).mean() * 100), 1),
        "avg": round(float(net.mean() * 100), 3),
        "max_mae": round(min(z["mae"] * z["w"] for z in tr) * 100, 2),
        "liq10": sum(z["w"] * z["mae"] <= LIQ for z in tr),
        "fund": round(sum(z["fund"] * z["w"] for z in tr) * 100, 3),
    }
    for lev in LEVS:
        e, w = equity(d, tr, a, lev)
        pk = np.maximum.accumulate(e)
        r[f"ret{int(lev)}"] = round((e[-1] - 1) * 100, 2)
        r[f"mdd{int(lev)}"] = round(float((e / pk - 1).min()) * 100, 2)
        if lev == 10.0:
            r["min_intrabar10"] = round(float(w.min() - 1) * 100, 2)
    side = {}
    for s, nm in ((1, "long"), (-1, "short")):
        zz = [z for z in tr if z["s"] == s]
        if zz:
            nn = np.array([z["net"] * z["w"] for z in zz])
            side[nm] = {
                "n": len(zz),
                "win": round(float((nn > 0).mean() * 100), 1),
                "x1": round(float((np.prod(1 + nn) - 1) * 100), 2),
                "x10": round(float((np.prod(np.maximum(1 + 10 * nn, 0)) - 1) * 100), 2),
            }
    r["side"] = side
    return r


def split_equity(d, groups: list, a, lev) -> tuple:
    es = [equity(d, tr, a, lev) for tr in groups]
    return np.mean([e for e, _ in es], 0), np.mean([w for _, w in es], 0)


def split_summary(d, groups, a) -> dict:
    allz = [z for tr in groups for z in tr]
    net = np.array([z["net"] * z["w"] for z in allz]) if allz else np.zeros(0)
    r = {
        "n": len(allz),
        "long": sum(z["s"] > 0 for z in allz),
        "short": sum(z["s"] < 0 for z in allz),
        "open": sum(z["open"] for z in allz),
        "win": round(float((net > 0).mean() * 100), 1) if len(net) else None,
        "avg": round(float(net.mean() * 100), 3) if len(net) else None,
        "max_mae": round(min(z["mae"] * z["w"] for z in allz) * 100, 2) if allz else 0,
        "liq10": sum(z["w"] * z["mae"] <= LIQ for z in allz),
        "side": {},
    }
    for lev in LEVS:
        e, w = split_equity(d, groups, a, lev)
        pk = np.maximum.accumulate(e)
        r[f"ret{int(lev)}"] = round((e[-1] - 1) * 100, 2)
        r[f"mdd{int(lev)}"] = round(float((e / pk - 1).min()) * 100, 2)
        if lev == 10.0:
            r["min_intrabar10"] = round(float(w.min() - 1) * 100, 2)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--now", type=int, default=0, help="고정 시각(ms). 0이면 지금 마지막 마감 15분봉"
    )
    o_ = ap.parse_args()
    out = Path(o_.out)
    out.mkdir(parents=True, exist_ok=True)
    now = o_.now or int(time.time() * 1000) // BAR * BAR
    fut5, finfo = load_fut(out, now)
    (out / "fetch.json").write_text(json.dumps({"now": now}))
    spot5, sinfo = load_spot(out, now)
    d = bars(fut5, 0)
    t = d["t"]
    end = t[-1] + BAR
    a = int(np.searchsorted(t, end - 30 * DAY))
    a_t = t[a]
    fund = fetch_funding(int(a_t - 2 * DAY))
    ft = np.array([x[0] for x in fund], float)
    fr = np.array([x[1] for x in fund], float)
    info = {
        "start_kst": datetime.fromtimestamp(a_t / 1000, KST).isoformat(),
        "end_kst": datetime.fromtimestamp(end / 1000, KST).isoformat(),
        "bars15": int(len(t) - a),
        "fut": finfo,
        "spot": sinfo,
        "fut_gaps": gaps(fut5["t"], a_t - 60 * DAY),
        "spot_gaps": gaps(spot5["t"], a_t - 60 * DAY),
        "funding_events": int(((ft > a_t) & (ft <= end)).sum()),
        "funding_sum_pct": round(float(fr[(ft > a_t) & (ft <= end)].sum() * 100), 4),
        "buy_hold_pct": round((d["c"][-1] / d["c"][a - 1] - 1) * 100, 2),
    }
    # 신호: 15분봉 시각 t에 맞춘 현물 기반 ML
    sd = bars(spot5, 0)
    ml = ml_signals(spot5, t, out)
    info["ml_vs_tracker"] = check_vs_tracker(ml, t)
    from btc_ml_tracker import trend4h

    tr4 = trend4h(spot5, t)
    reg, newday = R.regime(fut5, d, 0)
    atr = daily_atr(fut5, d, 0)
    g4 = gate_g1(fut5, t) & gate_g2(fut5, t)
    sig = {k: ml[k].copy() for k in ("A", "B", "AS", "BS")}
    sig["AS4"] = np.where(sig["AS"] == tr4, sig["AS"], 0)
    sig["AS4G4"] = np.where(g4, sig["AS4"], 0)
    sig["K2B"] = k2_first(reg, sig["AS4"] == 1, sig["AS4"] == -1, a)
    info["signal_counts"] = {k: int((v[a:] != 0).sum()) for k, v in sig.items()}
    info["spot_last"] = float(sd["t"][-1])
    b = len(t) - 1
    a_day = a + int(np.flatnonzero(newday[a:])[0])

    def wth(i):
        v = atr[i]
        return float(min(TH1_CAP, TH1_RISK / v)) if v > 0 else 0.0

    T = {}
    for k in ("A", "B", "AS", "BS", "AS4", "AS4G4", "K2B"):
        T[k] = bracket(d, sig[k], a, b, *ML)
    T["AS4T75"] = bracket(d, sig["AS4"], a, b, *T75)
    T["K2"] = k2_trades(d, reg, sig["AS4"], a, b)
    T["R1"] = r1_trades(d, reg, newday, a_day, b)
    T["TH1"] = r1_trades(d, reg, newday, a_day, b, 1, wth)
    T["TH1S"] = r1_trades(d, reg, newday, a_day, b, -1, wth)
    T["L1"] = l1_trades(d, a, b, 1)
    T["L1S"] = l1_trades(d, a, b, -1)
    T["S4"] = s4_trades(d, fut5, a_t, end)
    # 앙상블
    cnt_up = sum((sig[k] > 0).astype(int) for k in ("A", "B", "AS", "BS", "AS4"))
    cnt_dn = sum((sig[k] < 0).astype(int) for k in ("A", "B", "AS", "BS", "AS4"))
    e1 = np.where(cnt_up >= 3, 1, np.where(cnt_dn >= 3, -1, 0))
    T["E1"] = bracket(d, e1, a, b, *ML)
    T["E2"] = bracket(d, np.where(sig["B"] == reg, sig["B"], 0), a, b, *ML)
    T["E3"] = bracket(d, np.where(sig["AS"] == reg, sig["AS"], 0), a, b, *ML)
    for tr in T.values():
        finish(d, tr, ft, fr)
    rep = {"info": info}
    for k in SINGLES + ("E1", "E2", "E3"):
        rep[k] = summary(d, T[k], a)
    top3 = sorted(SINGLES, key=lambda k: -rep[k]["ret10"])[:3]
    rep["E4"] = split_summary(d, [T[k] for k in TRACKER], a)
    rep["E5"] = split_summary(d, [T[k] for k in top3], a)
    rep["E5"]["members"] = top3
    rep["info"]["a"], rep["info"]["now"] = a, now
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "trades.json").write_text(json.dumps(T, ensure_ascii=False))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    rows = sorted(SINGLES + ENS, key=lambda k: -rep[k]["ret10"])
    for k in rows:
        r = rep[k]
        print(
            f"{k:7s} n={r['n']:3d} L{r['long']}/S{r['short']} open{r['open']} win={r['win']} "
            f"avg={r['avg']} 10x={r['ret10']} mdd10={r['mdd10']} 1x={r['ret1']} "
            f"mdd1={r['mdd1']} mae={r['max_mae']} liq={r['liq10']} "
            f"minIB10={r['min_intrabar10']}"
        )


if __name__ == "__main__":
    main()
