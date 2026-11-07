"""124번: R1 + ML(BS, AS4) 묶음, 15분봉 평가·일별 낙폭. 등록: 2026-11-07-btc-strategy-combo-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_strategy_combo.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import btc_fut_daily_regime as R
import btc_ml_edge_cost as C
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_scr_learned import bars

DAY = 86_400_000
RES = Path("~/.minerva/research/btc-2026-11-04").expanduser()
FEE_R1 = 0.0005


def ts(y: int, m: int = 1, d: int = 1) -> float:
    return datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000


def ml_trades(P: dict, t: np.ndarray, li, si) -> list:
    """rest_sim과 같은 규칙. (체결 봉, 진입가, 방향, 청산 봉, 청산가, 수수료 합)."""
    o, h, l, c = P["o"], P["h"], P["l"], P["c"]  # noqa: E741
    mk, tk, sp = C.MAKER, C.TAKER, C.SLIP
    out, i, n = [], 0, len(t)
    while i < n - 2:
        s = 1 if li[i] else (-1 if si[i] else 0)
        if not s:
            i += 1
            continue
        lim = c[i] * (1 - C.OFF * s)
        j = None
        for b in range(i + 1, min(i + 1 + C.REST, n)):
            if (l[b] <= lim * (1 - C.THRU)) if s > 0 else (h[b] >= lim * (1 + C.THRU)):
                j = b
                break
        if j is None:
            i += 1
            continue
        px = lim
        up, dn = px * (1 + C.TP), px * (1 - C.SL)
        res, b = None, j
        while b < n:
            first = b == j
            if s > 0:
                if l[b] <= dn:
                    res = ((min(o[b], dn) if not first else dn) * (1 - sp), mk + tk)
                elif not first and h[b] >= up * (1 + C.THRU):
                    res = (up, 2 * mk)
            else:
                if h[b] >= up:
                    res = ((max(o[b], up) if not first else up) * (1 + sp), mk + tk)
                elif not first and l[b] <= dn * (1 - C.THRU):
                    res = (dn, 2 * mk)
            if res:
                break
            b += 1
        if res is None:
            break
        out.append((j, px, s, b, res[0], res[1]))
        i = b
    return out


def r1_trades(d: dict, d5: dict) -> list:
    """R1을 현재 자료로 다시 돌려 (진입 봉, 진입가, 방향, 청산 봉, 청산가, 수수료 합)으로 바꾼다."""
    fund = load_funding()
    reg, newday = R.regime(d5, d, 0)
    t = d["t"]
    start = int(np.searchsorted(t, t[0] + 60 * DAY))
    out = []
    for i, side, _g, x, _t, stopped, _ in R.run(d, reg, newday, start, True, fund):
        e = i + 1
        px = d["o"][e]
        if stopped:
            sp = px * (1 - R.STOP * side)
            ex = min(d["o"][x], sp) if side > 0 else max(d["o"][x], sp)
            out.append((e, px, side, x, ex, 2 * FEE_R1))
        else:
            # 청산은 x 봉 시가: x−1 봉까지 보유하고 x−1 봉 마감 뒤 o[x]에 나간다
            out.append((e, px, side, x - 1, d["o"][x], 2 * FEE_R1))
    return out


def sleeve_returns(trs: list, c: np.ndarray, lo: np.ndarray, hi: np.ndarray, t: np.ndarray,
                   lev: float, ft, fr) -> tuple:
    """봉별 레그 수익률. 거래 안에서는 진입 때 정한 명목으로 평가한다."""
    n = len(t)
    ret = np.zeros(n)
    liq = 0
    for e, px, s, x, ex, fee in trs:
        if e >= n or x >= n:
            continue
        path = c[e:x + 1] / px - 1
        path[-1] = ex / px - 1
        worst = (lo[e:x + 1] / px - 1) if s > 0 else (hi[e:x + 1] / px - 1)
        # 펀딩: 체결 뒤 정산마다 명목에 대해 낸다(롱은 양수 펀딩을 낸다)
        m = (ft > t[e]) & (ft <= t[x])
        cf = np.zeros(x - e + 1)
        if m.any():
            k = np.searchsorted(t[e:x + 1], ft[m], side="left")
            np.add.at(cf, np.minimum(k, len(cf) - 1), fr[m])
        cf = np.cumsum(cf)
        v = 1 + lev * (s * path - s * cf)
        v -= lev * fee / 2
        v[-1] -= lev * fee / 2
        bad = np.flatnonzero(1 + lev * s * worst <= 0)
        if bad.size:
            liq += 1
            v[bad[0]:] = 0.0
        prev = np.r_[1.0, v[:-1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(prev > 0, v / prev - 1, 0.0)
        ret[e:x + 1] = r
    return ret, liq


def daily(ret: np.ndarray, t: np.ndarray) -> tuple:
    day = (t // DAY).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    g = np.log1p(np.maximum(ret, -1 + 1e-12))
    cs = np.r_[0.0, np.cumsum(g)]
    dr = np.expm1(cs[edges[1:]] - cs[edges[:-1]])
    return day[edges[:-1]] * DAY, dr


def combo(dr: dict, w: dict, dt: np.ndarray) -> np.ndarray:
    """매월 첫날 비중을 다시 맞춘다. 레그 자금은 격리."""
    month = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).strftime("%Y%m") for x in dt])
    out = np.zeros(len(dt))
    edges = np.flatnonzero(np.r_[True, month[1:] != month[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        val = sum(w[k] * np.cumprod(1 + dr[k][a:b]) for k in w)
        prev = np.r_[1.0, val[:-1]]
        out[a:b] = val / prev - 1
    return out


def stats(r: np.ndarray, dt: np.ndarray) -> dict:
    if not len(r):
        return {}
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    years = len(r) / 365.0
    month = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).strftime("%Y%m") for x in dt])
    mret = [float(np.prod(1 + r[month == m]) - 1) for m in dict.fromkeys(month)]
    sd = r.std()
    cagr = eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1.0
    mdd = float((eq / peak[1:] - 1).min())
    return {"cagr": round(cagr * 100, 1), "mdd": round(mdd * 100, 1),
            "ratio": round(cagr / abs(mdd), 2) if mdd < 0 else None,
            "total": round((eq[-1] - 1) * 100, 1), "worst_month": round(min(mret) * 100, 1),
            "pos_months": round(float(np.mean([m > 0 for m in mret])) * 100, 0),
            "sharpe": round(float(r.mean() / sd * np.sqrt(365)), 2) if sd > 0 else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    d = bars(d5, 0)
    ft, fr = load_funding()
    z = np.load(RES / "ml15/pred.npz")
    tp = z["t"]
    spot = {k: z[k] for k in "ohlc"}
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    pm = np.mean(ps, 0)
    P, ok = futures_on(tp, spot)
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp)
    li, si = rule_masks(pm, tp, 0.95, 5)
    bs = ml_trades(P, tp, li, si)
    li, si = rule_masks(pm, tp, 0.90, 4)
    as4 = ml_trades(P, tp, li & (tr4 > 0), si & (tr4 < 0))
    # 공통 격자: 예측 격자(15분) 2022-01-01 ~ 예측 끝
    a0 = int(np.searchsorted(tp, ts(2022)))
    t = tp[a0:]
    c, lo, hi = P["c"][a0:], P["l"][a0:], P["h"][a0:]
    shift = lambda trs: [(e - a0, px, s, x - a0, ex, f) for e, px, s, x, ex, f in trs if e >= a0]  # noqa: E731
    # R1 거래를 예측 격자로 옮긴다(시각이 같은 봉)
    r1 = []
    for e, px, s, x, ex, f in r1_trades(d, d5):
        je, jx = np.searchsorted(t, d["t"][e]), np.searchsorted(t, d["t"][x])
        if je < len(t) and jx < len(t) and t[je] == d["t"][e] and t[jx] == d["t"][x] and d["t"][e] >= t[0]:
            r1.append((int(je), px, s, int(jx), ex, f))
    trades = {"R1": r1, "BS": shift(bs), "AS4": shift(as4)}
    lev = {"R1": 1.0, "BS": 2.0, "AS4": 2.0}
    dr, dr1, liq = {}, {}, {}
    for k, trs in trades.items():
        rr, liq[k] = sleeve_returns(trs, c, lo, hi, t, lev[k], ft, fr)
        dt, dr[k] = daily(rr, t)
        dr1[k] = daily(sleeve_returns(trs, c, lo, hi, t, 1.0, ft, fr)[0], t)[1]
    combos = {"C1": {"R1": 1.0}, "C2": {"R1": 0.5, "BS": 0.5}, "C3": {"R1": 0.5, "AS4": 0.5},
              "C4": {"R1": 1 / 3, "BS": 1 / 3, "AS4": 1 / 3}}
    periods = {"2022~23": (ts(2022), ts(2024)), "2024~": (ts(2024), t[-1] + 1),
               "1y": (t[-1] - 365 * DAY, t[-1] + 1)}
    rep: dict = {"end": datetime.fromtimestamp(t[-1] / 1000, timezone.utc).isoformat(),
                 "futures_missing_bars": int((~ok[a0:]).sum()), "liquidations": liq,
                 "trades": {k: len(v) for k, v in trades.items()}}
    series = {}
    for name, w in combos.items():
        cr = combo(dr, w, dt)
        series[name] = cr
        rep[name] = {}
        for pn, (p0, p1) in periods.items():
            m = (dt >= p0) & (dt < p1)
            rep[name][pn] = stats(cr[m], dt[m])
    for k in dr:
        rep[f"leg {k} ({lev[k]:g}x)"] = {pn: stats(dr[k][(dt >= p0) & (dt < p1)], dt[(dt >= p0) & (dt < p1)])
                                         for pn, (p0, p1) in periods.items()}
    ks = list(dr1)
    rep["corr_1x"] = {f"{x}-{y}": round(float(np.corrcoef(dr1[x], dr1[y])[0, 1]), 3)
                      for i, x in enumerate(ks) for y in ks[i + 1:]}
    c1 = rep["C1"]
    for name in ("C2", "C3", "C4"):
        r = rep[name]
        crit = {p: bool(r[p]["cagr"] > 0 and r[p]["mdd"] > -25 and (r[p]["ratio"] or -9) >= (c1[p]["ratio"] or -9))
                for p in ("2022~23", "2024~")}
        r["pass"] = all(crit.values())
        r["crit"] = crit
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    np.savez(out / "series.npz", dt=dt, **{f"c_{k}": v for k, v in series.items()},
             **{f"leg_{k}": v for k, v in dr.items()})
    (out / "trades.json").write_text(json.dumps(
        {k: [[float(t[e]), float(px), s, float(t[x]), float(ex)] for e, px, s, x, ex, _ in v]
         for k, v in trades.items()}))
    print(json.dumps({k: v for k, v in rep.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
