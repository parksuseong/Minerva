"""144번: 일봉 추세 보유 변형 T0~T5의 시장 간 견고성. 등록: 2026-11-07-trend-hold-cross-market-hypotheses.md

실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python docs/backtests/raw/trend_hold_cross_market.py --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import numpy as np
from btc_fractal import load5, resample
from btc_scr_learned import bars

M5 = 300_000
BAR = 900_000
H = 3_600_000
DAY = 86_400_000
FEE = 0.0005
STOP = 0.01
CAP = 3.0
RISK = 0.01
VARS = {
    "T0": dict(size="1x", stop=True, short=True),
    "T1": dict(size="1x", stop=True, short=False),
    "T2": dict(size="atr", stop=True, short=True),
    "T3": dict(size="atr", stop=True, short=False),
    "T4": dict(size="atr", stop=False, short=True),
    "T5": dict(size="atr", stop=False, short=False),
}
HOME = Path("~/.minerva/data").expanduser()
MARKETS = {
    "M1": dict(k5=HOME / "btc/futures/klines_5m.csv", fund=HOME / "btc/futures/funding.csv",
               start="2020-01-01", end=None),
    "M2": dict(k5=HOME / "btc/5m.csv", fund=None, start="2017-09-01", end="2020-01-01"),
    "M3": dict(k5=HOME / "eth/futures/klines_5m.csv", fund=HOME / "eth/futures/funding.csv",
               start="2020-01-01", end=None),
}


def ms(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000


def load_funding(path):
    if path is None:
        return np.array([0.0]), np.array([0.0])
    t, r = [], []
    with Path(path).open() as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            t.append(float(row[0]))
            r.append(float(row[2]))
    o = np.argsort(t)
    return np.array(t)[o], np.array(r)[o]


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    for k in range(n, len(x)):
        out[k] = out[k - 1] + (x[k] - out[k - 1]) / n
    return out


def daily_atr(d5: dict, d: dict, shift_h: int) -> np.ndarray:
    """15분봉 마감 때 이미 마감된 일봉 ATR14 ÷ 종가 (129번 daily_atr_adx와 같은 정렬)."""
    sh = dict(d5)
    sh["t"] = d5["t"] - shift_h * H
    u = resample(sh, 288)
    h, l, c = u["h"], u["l"], u["c"]  # noqa: E741
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    atr = wilder(tr, 14) / c
    close = u["t"] + shift_h * H + 288 * M5
    j = np.searchsorted(close, d["t"] + BAR, side="right") - 1
    out = np.full(len(d["t"]), np.nan)
    out[j >= 0] = atr[j[j >= 0]]
    return out


class Mkt:
    def __init__(self, name: str):
        cfg = MARKETS[name]
        d5 = load5(cfg["k5"])
        keep = d5["t"] >= ms(cfg["start"]) - 70 * DAY
        if cfg["end"]:
            keep &= d5["t"] < ms(cfg["end"])
        self.d5 = {k: v[keep] for k, v in d5.items()}
        self.d = bars(self.d5, 0)
        self.n = len(self.d["t"])
        self.ft, self.fr = load_funding(cfg["fund"])
        self.cumf = np.r_[0.0, np.cumsum(self.fr)]
        self.start = int(np.searchsorted(self.d["t"], max(ms(cfg["start"]), self.d["t"][0] + 60 * DAY)))
        self.t5 = self.d5["t"]

    def px5(self, t15_next: float, fallback: float) -> tuple:
        """다음 15분봉 시작 + 5분의 5분봉 시가와 그 5분봉 인덱스."""
        j = int(np.searchsorted(self.t5, t15_next + M5))
        if j < len(self.t5) and self.t5[j] == t15_next + M5:
            return float(self.d5["o"][j]), j
        return fallback, -1

    def fund_between(self, t0: float, t1: float) -> float:
        a = np.searchsorted(self.ft, t0, side="right")
        b = np.searchsorted(self.ft, t1, side="right")
        return float(self.cumf[b] - self.cumf[a])


def run(m: Mkt, reg: np.ndarray, newday: np.ndarray, atr: np.ndarray, var: dict) -> tuple:
    d = m.d
    t, o, h, l, c = d["t"], d["o"], d["h"], d["l"], d["c"]  # noqa: E741
    n = m.n
    eq = np.full(n, np.nan)
    inpos = np.zeros(n, bool)
    E = 1.0
    eq[: m.start] = 1.0
    trades = []
    i, can = m.start, True
    last = m.start
    while i < n - 3:
        if newday[i]:
            can = True
        s = int(reg[i])
        if s < 0 and not var["short"]:
            s = 0
        if not (s and can):
            i += 1
            continue
        if var["size"] == "1x":
            w = 1.0
        else:
            a = atr[i]
            if not (a > 0):
                i += 1
                continue
            w = min(CAP, RISK / a)
        e = i + 1
        px, j5 = m.px5(t[e], o[e])
        t_in = t[e] + M5
        E0 = E
        N = w * E0
        stop = px * (1 - s * STOP) if var["stop"] else None
        x = None
        # 진입 봉: 진입 뒤 5분봉 2개만 본다
        if stop is not None:
            if j5 >= 0:
                for jj in (j5, j5 + 1):
                    if jj < len(m.t5) and m.t5[jj] < t[e] + BAR:
                        if (m.d5["l"][jj] <= stop) if s > 0 else (m.d5["h"][jj] >= stop):
                            x = (e, stop, m.t5[jj] + M5, True)
                            break
            elif (l[e] <= stop) if s > 0 else (h[e] >= stop):
                x = (e, stop, t[e] + BAR, True)
        b = e
        while x is None and b < n - 2:
            if b > e and stop is not None and ((l[b] <= stop) if s > 0 else (h[b] >= stop)):
                fill = min(o[b], stop) if s > 0 else max(o[b], stop)
                x = (b, fill, t[b] + BAR, True)
                break
            rb = int(reg[b])
            if rb < 0 and not var["short"]:
                rb = 0
            if rb != s:
                xp, _ = m.px5(t[b + 1], o[b + 1])
                x = (b + 1, xp, t[b + 1] + M5, False)
                break
            b += 1
        if x is None:
            break
        xb, xp, t_out, stopped = x
        # 보유 구간 시가 평가 (b 봉 마감)
        seg = np.arange(e, xb)
        if len(seg):
            a0 = np.searchsorted(m.ft, t_in, side="right")
            ak = np.searchsorted(m.ft, t[seg] + BAR, side="right")
            fpaid = N * s * (m.cumf[ak] - m.cumf[a0])
            eq[seg] = E0 - N * FEE + N * s * (c[seg] / px - 1) - fpaid
            inpos[seg] = True
        fund_all = N * s * m.fund_between(t_in, t_out)
        g = N * s * (xp / px - 1)
        E = E0 - N * FEE + g - fund_all - N * (xp / px) * FEE
        eq[last:e] = np.where(np.isnan(eq[last:e]), E0, eq[last:e])
        eq[xb] = E
        trades.append((int(e), s, float((E - E0) / E0), int(xb), bool(stopped), float(w), float(t[e])))
        last = xb + 1
        if stopped:
            can = False
            i = xb
        else:
            i = xb - 1
        continue
    eq[last:] = np.where(np.isnan(eq[last:]), E, eq[last:])
    # 앞쪽 빈칸 채우기
    for k in range(1, n):
        if np.isnan(eq[k]):
            eq[k] = eq[k - 1]
    return eq, inpos, trades


def daily(m: Mkt, eq: np.ndarray) -> tuple:
    t = m.d["t"]
    key = ((t + BAR) // DAY).astype(np.int64)
    last = np.flatnonzero(np.r_[key[1:] != key[:-1], True])
    return key[last] * DAY, eq[last]


def stats(dt: np.ndarray, de: np.ndarray, t0: float, t1: float | None = None) -> dict:
    m = dt >= t0
    if t1 is not None:
        m &= dt < t1
    x, e = dt[m], de[m]
    if len(e) < 30:
        return {}
    e = e / e[0]
    yrs = (x[-1] - x[0]) / (365.25 * DAY)
    cagr = e[-1] ** (1 / yrs) - 1 if e[-1] > 0 else -1.0
    mdd = float((e / np.maximum.accumulate(e) - 1).min())
    return {"cagr": round(cagr * 100, 2), "mdd": round(mdd * 100, 2),
            "ratio": round(cagr / abs(mdd), 3) if mdd < 0 else None, "ret": round((e[-1] - 1) * 100, 1)}


def tstats(tr: list, sel=lambda z: True) -> dict:
    g = [z[2] for z in tr if sel(z)]
    if not g:
        return {"n": 0}
    g = np.array(g)
    return {"n": len(g), "avg": round(float(g.mean() * 100), 3), "win": round(float((g > 0).mean() * 100), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    keep_tr = {}
    keep_eq = {}
    for mk in MARKETS:
        m = Mkt(mk)
        t = m.d["t"]
        t0 = float(t[m.start])
        bh = m.d["c"] / m.d["c"][m.start]
        bh[: m.start] = 1.0
        bdt, bde = daily(m, bh)
        rep[mk] = {"span": [datetime.fromtimestamp(t0 / 1000, timezone.utc).strftime("%Y-%m-%d"),
                            datetime.fromtimestamp(t[-1] / 1000, timezone.utc).strftime("%Y-%m-%d")],
                   "BH": {"all": stats(bdt, bde, t0), "2024~": stats(bdt, bde, ms("2024-01-01"))}}
        keep_eq[(mk, "BH")] = (bdt, bde)
        for sh in (0, 8, 16):
            reg, newday = R.regime(m.d5, m.d, sh)
            atr = daily_atr(m.d5, m.d, sh)
            for vn, var in VARS.items():
                eq, inpos, tr = run(m, reg, newday, atr, var)
                dt, de = daily(m, eq)
                key = vn if sh == 0 else f"{vn}@{sh}h"
                r = {"all": stats(dt, de, t0), "2024~": stats(dt, de, ms("2024-01-01")),
                     "trades": tstats(tr), "long": tstats(tr, lambda z: z[1] > 0),
                     "short": tstats(tr, lambda z: z[1] < 0),
                     "in_mkt": round(float(inpos[m.start:].mean() * 100), 1),
                     "avg_w": round(float(np.mean([z[5] for z in tr])), 3) if tr else None}
                if sh == 0:
                    yrs = sorted({datetime.fromtimestamp(x / 1000, timezone.utc).year for x in dt if x >= t0})
                    r["years"] = {y: stats(dt, de, max(t0, ms(f"{y}-01-01")), ms(f"{y + 1}-01-01")).get("ret")
                                  for y in yrs}
                    keep_tr[(mk, vn)] = tr
                    keep_eq[(mk, vn)] = (dt, de)
                rep[mk][key] = r
                print(mk, key, r["all"], r["2024~"], r["trades"], flush=True)
    # 판정
    verdict = {}
    for vn in VARS:
        c1 = all(rep[mk][vn]["all"].get("cagr", -1) > 0 and (rep[mk][vn]["all"].get("ratio") or -9)
                 > (rep[mk]["BH"]["all"].get("ratio") or -9) for mk in MARKETS)
        c2 = sum((rep[mk][vn]["all"].get("ratio") or -9) > (rep[mk]["T0"]["all"].get("ratio") or -9)
                 for mk in MARKETS) >= 2
        c3 = all(rep[mk][vn]["2024~"].get("cagr", -1) > 0 for mk in ("M1", "M3"))
        verdict[vn] = {"1": bool(c1), "2": bool(c2) if vn != "T0" else None, "3": bool(c3),
                       "pass": bool(c1 and c2 and c3) if vn != "T0" else None}
    rep["verdict"] = verdict
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    np.savez(out / "series.npz", **{f"{a}|{b}|t": v[0] for (a, b), v in keep_eq.items()},
             **{f"{a}|{b}|e": v[1] for (a, b), v in keep_eq.items()})
    (out / "trades.json").write_text(json.dumps({f"{a}|{b}": v for (a, b), v in keep_tr.items()}))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
