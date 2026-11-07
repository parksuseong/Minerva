"""비가격 진입 신호 시험용 자료: 선물 5분봉(테이커 포함) → 15분봉, metrics(OI·롱숏), 프리미엄 지수, 펀딩.

문서: docs/backtests/2026-11-08-btc-nonprice-hypotheses.md
- 선물 5분봉: 저장 이력(`klines_5m.csv`)의 바이낸스 행(OKX 행 버림) + 빈 곳·끝부분은 fapi 공개 시세.
- metrics: `nonprice_fetch.py`가 체크섬을 확인해 받은 일별 zip. create_time은 5분 구간의 시작이라
  create_time + 5분 (+ lag) ≤ 판단 시각인 행만 쓴다. OI ≤ 0, 비율 ≤ 0 은 결측.
- 프리미엄 지수 5분봉: 저장 이력 + fapi premiumIndexKlines 끝부분. 봉 시작 + 5분 ≤ 판단 시각.
- 펀딩: funding.csv + fapi.
- 15분봉 i의 판단 시각은 tc = t_i + 15분. 모든 특징은 tc에 알 수 있는 값만 쓴다.
실행: PYTHONPATH=<pyml>:docs/backtests/raw python3 docs/backtests/raw/nonprice_data.py --out <폴더>
     [--lag-min 5]  # metrics를 한 칸 더 늦춘 강건성 자료(data15_lag.npz)
"""

from __future__ import annotations

import argparse
import calendar
import csv
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from pine_data import FAPI, get

M5 = 300_000
M15 = 900_000
H1 = 3_600_000
DAY = 86_400_000
T0 = 1_635_724_800_000  # 2021-11-01 UTC
HOME = Path("~/.minerva").expanduser()
FUT_HIST = HOME / "data/btc/futures/klines_5m.csv"
PREM_HIST = HOME / "data/btc/futures/premium_5m.csv"
FUND_HIST = HOME / "data/btc/futures/funding.csv"
MET_DIR = HOME / "data/btc/futures/metrics"
ZWIN = 2880  # 30일 × 96봉
ZMIN = 2000


def fetch(path: str, since: int, until: int, cols: int) -> list:
    rows, s = [], since
    now = int(time.time() * 1000)
    while s < until:
        k = get(
            f"{FAPI}/{path}?symbol=BTCUSDT&interval=5m&limit=1500&startTime={s}&endTime={until - 1}"
        )
        k = [x for x in k if int(x[6]) < now]
        if not k:
            break
        rows += [(int(x[0]), *(float(x[c]) for c in range(1, cols))) for x in k]
        s = int(k[-1][0]) + M5
        time.sleep(0.3)
        if len(k) < 1500:
            break
    return rows


def gaps(ts: list, start: int) -> list:
    t = np.array(sorted(x for x in ts if x >= start), dtype=np.int64)
    d = np.diff(t)
    return [(int(t[i]) + M5, int(t[i + 1])) for i in np.flatnonzero(d != M5)]


def load_fut() -> tuple:
    rows: dict = {}
    okx = 0
    with FUT_HIST.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            t = int(x[0])
            if t < T0:
                continue
            if x[-1] == "okx":
                okx += 1
                continue
            # o h l c v trades taker_buy_vol
            rows[t] = (float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]),
                       float(x[8]), float(x[9]))  # fmt: skip
    n_hist = len(rows)
    now = int(time.time() * 1000)

    def kl(a, b):
        out = []
        for x in fetch("klines", a, b, 12):
            out.append((x[0], (x[1], x[2], x[3], x[4], x[5], x[8], x[9])))
        return out

    new = kl(max(rows) + M5, now)
    for t, v in new:
        rows[t] = v
    g0 = gaps(list(rows), T0)
    filled = 0
    for a, b in g0:
        for t, v in kl(a, b):
            rows[t] = v
            filled += 1
    g1 = gaps(list(rows), T0)
    ts = np.array(sorted(rows), dtype=np.int64)
    v = np.array([rows[int(t)] for t in ts])
    info = {"fut_hist_rows": n_hist, "okx_dropped": okx, "fut_new": len(new),
            "fut_gaps_before": len(g0), "fut_filled": filled,
            "fut_gaps_after": [[a, b, (b - a) // M5] for a, b in g1]}  # fmt: skip
    return ts, v, info


def load_prem() -> tuple:
    rows: dict = {}
    with PREM_HIST.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            t = int(x[0])
            if t >= T0:
                rows[t] = float(x[4])
    n = len(rows)
    now = int(time.time() * 1000)
    new = fetch("premiumIndexKlines", max(rows) + M5, now, 5)
    for x in new:
        rows[x[0]] = x[4]
    g0 = gaps(list(rows), T0)
    for a, b in g0:
        for x in fetch("premiumIndexKlines", a, b, 5):
            rows[x[0]] = x[4]
    g1 = gaps(list(rows), T0)
    ts = np.array(sorted(rows), dtype=np.int64)
    return ts, np.array([rows[int(t)] for t in ts]), {
        "prem_hist_rows": n, "prem_new": len(new), "prem_gaps_before": len(g0),
        "prem_gaps_after": [[a, b, (b - a) // M5] for a, b in g1]}  # fmt: skip


def load_fund() -> tuple:
    fund = {}
    with FUND_HIST.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            fund[int(x[0])] = float(x[2])
    s = max(fund) + 1
    while True:
        k = get(f"{FAPI}/fundingRate?symbol=BTCUSDT&limit=1000&startTime={s}")
        if not k:
            break
        for x in k:
            fund[int(x["fundingTime"])] = float(x["fundingRate"])
        s = int(k[-1]["fundingTime"]) + 1
        if len(k) < 1000:
            break
    ft = np.array(sorted(fund), dtype=np.int64)
    return ft, np.array([fund[int(t)] for t in ft])


def parse_ts(s: str) -> int:
    s = s.strip()
    if s.isdigit():
        return int(s)
    return calendar.timegm(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S")) * 1000


def load_metrics() -> tuple:
    files = sorted(MET_DIR.glob("BTCUSDT-metrics-*.zip"))
    rows: dict = {}
    for f in files:
        with zipfile.ZipFile(f) as z:
            name = z.namelist()[0]
            txt = z.read(name).decode()
        for x in csv.reader(io.StringIO(txt)):
            if not x or not x[0] or x[0].startswith("create_time"):
                continue
            vals = []
            for c in (2, 4, 5, 6, 7):
                try:
                    v = float(x[c])
                except (ValueError, IndexError):
                    v = float("nan")
                vals.append(v if v > 0 else float("nan"))
            rows[parse_ts(x[0])] = vals
    mt = np.array(sorted(rows), dtype=np.int64)
    mv = np.array([rows[int(t)] for t in mt])
    # 열: oi, top_count, top_sum(포지션), crowd_count, taker_ls
    return mt, mv, {"metrics_files": len(files), "metrics_rows": len(mt),
                    "metrics_first": int(mt[0]), "metrics_last": int(mt[-1]),
                    "metrics_nan_oi": int(np.isnan(mv[:, 0]).sum())}  # fmt: skip


def asof(src_t: np.ndarray, src_v: np.ndarray, q: np.ndarray, stale: int) -> np.ndarray:
    """q 시각에 src_t ≤ q 인 마지막 값. 그 값이 stale ms보다 오래됐으면 결측."""
    k = np.searchsorted(src_t, q, side="right") - 1
    out = np.full(q.shape, np.nan)
    ok = k >= 0
    kk = k[ok]
    v = src_v[kk].astype(float)
    v[(q[ok] - src_t[kk]) > stale] = np.nan
    out[ok] = v
    return out


def roll_z(x: np.ndarray, w: int = ZWIN, mn: int = ZMIN) -> np.ndarray:
    """직전 w봉(현재 봉 제외)의 평균·표준편차로 정규화. 결측은 빼고 센다."""
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    c0 = np.r_[0, np.cumsum(ok)]
    c1 = np.r_[0.0, np.cumsum(v)]
    c2 = np.r_[0.0, np.cumsum(v * v)]
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(i - w, 0)
    cnt = c0[i] - c0[lo]
    s1 = c1[i] - c1[lo]
    s2 = c2[i] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / cnt
        sd = np.sqrt(np.maximum(s2 / cnt - m * m, 0))
        z = (x - m) / sd
    z[cnt < mn] = np.nan
    return z


def lagged(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


def build(out: Path, lag_min: int = 0) -> dict:
    ts, v, info = load_fut()
    pt, pc, pinfo = load_prem()
    ft, fr = load_fund()
    mt, mv, minfo = load_metrics()
    info.update(pinfo)
    info.update(minfo)
    # 5분봉 → 15분봉
    g = ts - ts % M15
    keys, idx = np.unique(g, return_index=True)
    ends = np.r_[idx[1:], len(ts)]
    full = (ends - idx) == 3
    o = v[idx, 0]
    c = v[ends - 1, 3]
    h = np.maximum.reduceat(v[:, 1], idx)
    lo = np.minimum.reduceat(v[:, 2], idx)
    vol = np.add.reduceat(v[:, 4], idx)
    tbv = np.add.reduceat(v[:, 6], idx)
    if not full[-1]:
        keys, o, h, lo, c, vol, tbv, full = (a[:-1] for a in (keys, o, h, lo, c, vol, tbv, full))
    tc = keys + M15
    n = len(keys)
    F: dict = {}
    F["r1"] = c / lagged(c, 4) - 1
    F["r4"] = c / lagged(c, 16) - 1
    # metrics: create_time + 5분 + lag ≤ tc
    q = tc - M5 - lag_min * 60_000
    stale = 20 * 60_000
    met = {
        nm: asof(mt, mv[:, j], q, stale)
        for j, nm in enumerate(("oi", "topc", "tops", "crowd", "tkls"))
    }
    met_1h = asof(mt, mv[:, 0], q - H1, stale)
    met_4h = asof(mt, mv[:, 0], q - 4 * H1, stale)
    F["oi"] = met["oi"]
    F["oi1"] = np.log(met["oi"] / met_1h)
    F["oi4"] = np.log(met["oi"] / met_4h)
    F["zOI4"] = roll_z(F["oi4"])
    lnT = np.log(met["tops"])
    lnC = np.log(met["crowd"])
    lnT4 = np.log(asof(mt, mv[:, 2], q - 4 * H1, stale))
    lnC4 = np.log(asof(mt, mv[:, 3], q - 4 * H1, stale))
    F["div"] = (lnT - lnT4) - (lnC - lnC4)
    F["zD"] = roll_z(F["div"])
    F["lnC"] = lnC
    F["zC"] = roll_z(lnC)
    # 테이커 (선물 5분봉, 마감된 것만 = 이 15분봉까지)
    with np.errstate(invalid="ignore", divide="ignore"):
        cs_t = np.r_[0.0, np.cumsum(tbv)]
        cs_v = np.r_[0.0, np.cumsum(vol)]
        i = np.arange(n)
        F["tb1"] = (cs_t[i + 1] - cs_t[np.maximum(i - 3, 0)]) / (
            cs_v[i + 1] - cs_v[np.maximum(i - 3, 0)]
        )
        F["tb4"] = (cs_t[i + 1] - cs_t[np.maximum(i - 15, 0)]) / (
            cs_v[i + 1] - cs_v[np.maximum(i - 15, 0)]
        )
    F["tb1"][:3] = np.nan
    F["tb4"][:15] = np.nan
    # 프리미엄 지수: 5분봉 시작 + 5분 ≤ tc → 마지막 봉 시작 ≤ tc − 5분. 정규 5분 격자에 놓는다.
    grid0 = pt[0]
    gi = ((pt - grid0) // M5).astype(np.int64)
    pg = np.full(int(gi[-1]) + 1, np.nan)
    pg[gi] = pc
    okp = ~np.isnan(pg)
    cs_p = np.r_[0.0, np.cumsum(np.where(okp, pg, 0))]
    cs_n = np.r_[0, np.cumsum(okp)]
    last = (tc - M5 - grid0) // M5  # 쓸 수 있는 마지막 5분봉 번호
    last = np.clip(last, -1, len(pg) - 1)

    def pmean(k: int, need: int) -> np.ndarray:
        a = np.clip(last - k + 1, 0, None)
        s = cs_p[last + 1] - cs_p[a]
        cnt = cs_n[last + 1] - cs_n[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / cnt
        m[cnt < need] = np.nan
        m[last < 0] = np.nan
        return m

    F["prem1"] = pmean(12, 10)
    F["prem8"] = pmean(96, 80)
    F["zP"] = roll_z(F["prem1"])
    F["dprem"] = F["prem1"] - lagged(F["prem1"], 4)
    # 정산 펀딩
    F["f_last"] = asof(ft, fr, tc, 9 * H1)
    np.savez(out / ("data15.npz" if lag_min == 0 else f"data15_lag{lag_min}.npz"),
             t=keys, o=o, h=h, l=lo, c=c, v=vol, tbv=tbv, full=full, ft=ft, fr=fr,
             **{"f_" + k: val for k, val in F.items()})  # fmt: skip
    d15 = np.diff(keys)
    info.update({
        "built_utc_ms": int(time.time() * 1000), "lag_min": lag_min, "bars15": n,
        "first15": int(keys[0]), "last15": int(keys[-1]), "missing15": int((d15 != M15).sum()),
        "partial15": int((~full).sum()), "funding_last": int(ft[-1]),
        "prem_last": int(pt[-1]), "nan_frac": {k: float(np.isnan(val[n // 20:]).mean())
                                               for k, val in F.items()},
    })  # fmt: skip
    (out / ("data.json" if lag_min == 0 else f"data_lag{lag_min}.json")).write_text(
        json.dumps(info, indent=1)
    )
    return info


def load(out: Path, lag_min: int = 0) -> dict:
    z = np.load(out / ("data15.npz" if lag_min == 0 else f"data15_lag{lag_min}.npz"))
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lag-min", type=int, default=0)
    a = ap.parse_args()
    p = Path(a.out).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    info = build(p, a.lag_min)
    print(json.dumps({k: v for k, v in info.items() if k != "fut_gaps_after"}, indent=1))
    print("fut_gaps_after", info["fut_gaps_after"][:20])
