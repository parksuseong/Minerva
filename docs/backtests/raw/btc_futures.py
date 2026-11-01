"""펀딩비·미결제약정 가설. 사전 등록: docs/backtests/2026-11-01-btc-futures-data-hypotheses.md

numpy, scikit-learn 필요. 실행: PYTHONPATH=src:<경로>:docs/backtests/raw python <이 파일>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import btc_regress as br
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from minerva.strategy import btc_bt as bt
from minerva.strategy.btc_data import OUT

FUT = OUT / "futures"
STOP, COST = 0.01, 0.0005
SHIFTS = 300
WIN = 720


def ms(s: str) -> int:
    return int(
        datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000
    )


def load_futures(path: Path, keys):
    t, cols = [], {k: [] for k in keys}
    with path.open() as f:
        for r in csv.DictReader(f):
            v = r.get("calc_time") or r.get("create_time")
            t.append(int(v) if v.isdigit() else ms(v))
            for k in keys:
                cols[k].append(float(r[k]) if r[k] else np.nan)
    order = np.argsort(t)
    return np.array(t)[order], {k: np.array(v)[order] for k, v in cols.items()}


def asof(t_bar, t_src, x):
    i = np.searchsorted(t_src, t_bar, side="right") - 1
    out = np.where(i >= 0, x[np.maximum(i, 0)], np.nan)
    return out


def z(x, n=WIN):
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    a, b, c = (np.r_[0, np.cumsum(q)] for q in (v, v**2, ok.astype(float)))
    cnt = c[n:] - c[:-n]
    m = (a[n:] - a[:-n]) / np.maximum(cnt, 1)
    var = (b[n:] - b[:-n]) / np.maximum(cnt, 1) - m**2
    out = np.full(len(x), np.nan)
    out[n - 1 :] = (x[n - 1 :] - m) / np.sqrt(np.maximum(var, 1e-18))
    return out


def run(arr, entries, ft, fr, lo, hi, stop=STOP, cost=COST, e_slip=0.0, s_slip=0.0):
    o, h, l, t = arr  # noqa: E741
    out, free = [], -1
    for i, side, hold in entries:
        e = i + 1
        if e <= free or e < lo or e + hold >= hi:
            continue
        x = e + hold
        px0 = o[e]
        fill = px0 * (1 + side * e_slip)
        px, last = o[x], x - 1
        if stop is not None:
            line = px0 * (1 - side * stop)
            hit = l[e:x] <= line if side == 1 else h[e:x] >= line
            if hit.any():
                last = e + int(hit.argmax())
                bad = min(o[last], line) if side == 1 else max(o[last], line)
                px = bad * (1 - side * s_slip)
        a, b = np.searchsorted(ft, [t[e], t[last] + 3_600_000])
        fund = float(fr[a:b].sum()) * side
        r = (side * (px / fill - 1) + 1) * (1 - cost) - (px / fill) * cost - 1 - fund
        out.append((last, float(r), i, side))
        free = last
    return out


def stats(tr, years, days):
    if not tr:
        return {"n": 0, "final": 1.0, "cagr": 0.0, "mdd": 0.0, "ratio": 0.0}
    r = np.array([x[1] for x in tr])
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    mdd = float((eq / peak - 1).min())
    cagr = eq[-1] ** (bt.YEAR_DAYS / days) - 1
    by: dict = {}
    for x in tr:
        y = int(years[x[0]])
        by[y] = by.get(y, 1.0) * (1 + x[1])
    return {
        "n": len(r),
        "win": float((r > 0).mean()),
        "final": float(eq[-1]),
        "cagr": float(cagr),
        "mdd": mdd,
        "ratio": float(cagr / abs(mdd)) if mdd else 0.0,
        "top5": float(np.prod(np.sort(1 + r)[:-5])) if len(r) > 5 else 1.0,
        "years": {y: round(v - 1, 3) for y, v in sorted(by.items())},
    }


def month_shift(entries, months, rng):
    """같은 달 안에서 위치만 옮긴다."""
    out = []
    for i, s, h in entries:
        a, b = months[i]
        out.append((a + rng.randrange(max(b - a - 1, 1)), s, h))
    return sorted(out)


def edges(mask):
    return mask & ~np.r_[False, mask[:-1]]


def main() -> None:
    bars = bt.load("1h")
    with (OUT / "1h.csv").open() as f:
        t = np.array([int(r["open_time"]) for r in csv.DictReader(f)])
    o, h, l, c = [np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low", "close")]  # noqa: E741
    n = len(c)
    ft, f = load_futures(FUT / "funding.csv", ["last_funding_rate"])
    fr = f["last_funding_rate"]
    mt, m = load_futures(
        FUT / "metrics.csv",
        [
            "sum_open_interest_value",
            "count_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ],
    )
    tc = t + 3_600_000  # 봉 마감 시각
    fund = asof(tc, ft, fr)
    oi = asof(tc, mt, m["sum_open_interest_value"])
    ls = asof(tc, mt, m["count_long_short_ratio"])
    top = asof(tc, mt, m["sum_toptrader_long_short_ratio"])
    tak = asof(tc, mt, m["sum_taker_long_short_vol_ratio"])
    d24 = np.r_[np.full(24, np.nan), oi[24:] / oi[:-24] - 1]
    d7 = np.r_[np.full(168, np.nan), oi[168:] / oi[:-168] - 1]
    r24 = np.r_[np.full(24, 0.0), c[24:] / c[:-24] - 1]
    zf, zo, zo7, zl, zt, zk = z(fund), z(d24), z(d7), z(ls), z(top), z(tak)
    avgf = np.convolve(np.nan_to_num(fund), np.ones(WIN) / WIN, "full")[:n]
    avgf[:WIN] = np.nan

    def sig(lo_mask, hi_mask, hold):
        a = [(int(i), 1, hold) for i in np.nonzero(edges(lo_mask))[0]]
        b = [(int(i), -1, hold) for i in np.nonzero(edges(hi_mask))[0]]
        return sorted(a + b)

    never = np.zeros(n, bool)
    rules = {
        "F1": sig(zf <= -2, zf >= 2, 24),
        "F2": sig(avgf < 0, never, 168),
        "O1": sig((zo >= 2) & (r24 > 0), (zo >= 2) & (r24 < 0), 24),
        "O2": sig((zo <= -2) & (r24 < 0), (zo <= -2) & (r24 > 0), 24),
        "C1": sig((zo >= 2) & (fund < 0), (zo >= 2) & (zf >= 2), 24),
        "R1": sig(zl <= -2, zl >= 2, 24),
    }
    years = np.array([b.day.year for b in bars])
    ym = np.array([b.day.year * 100 + b.day.month for b in bars])
    s22 = next(i for i, b in enumerate(bars) if b.day >= datetime(2022, 2, 1).date())
    s25 = int(np.argmax(years == 2025))
    days = (bars[-1].day - bars[s22].day).days
    d25 = (bars[-1].day - bars[s25].day).days
    print("hold", [round(x, 3) for x in bt.hold(bars, s22, n)], flush=True)
    months = {}
    for v in np.unique(ym):
        idx = np.nonzero(ym == v)[0]
        for i in idx:
            months[int(i)] = (int(idx[0]), int(idx[-1]) + 1)
    arr = (o, h, l, t)
    rng = random.Random(7)

    def report(name, ent):
        ent = [x for x in ent if x[0] >= s22]
        a = stats(run(arr, ent, ft, fr, s22, n), years, days)
        if not a["n"]:
            print(name, "거래 없음", flush=True)
            return
        c10 = stats(run(arr, ent, ft, fr, s22, n, cost=0.001), years, days)
        sl = stats(run(arr, ent, ft, fr, s22, n, e_slip=0.0002, s_slip=0.001), years, days)
        rc = stats(run(arr, [x for x in ent if x[0] >= s25], ft, fr, s25, n), years, d25)
        fr_ = stats(run(arr, ent, ft, fr, s22, n, stop=None), years, days)
        vals = [
            stats(run(arr, month_shift(ent, months, rng), ft, fr, s22, n), years, days)["final"]
            for _ in range(SHIFTS)
        ]
        p = (sum(v >= a["final"] for v in vals) + 1) / (SHIFTS + 1)
        print(
            f"{name}: n {a['n']} win {a['win'] * 100:.1f} cagr {a['cagr'] * 100:+.1f} mdd {a['mdd'] * 100:.1f} "
            f"ratio {a['ratio']:.2f} cost10 {c10['cagr'] * 100:+.1f} slip {sl['cagr'] * 100:+.1f} "
            f"y25 {rc['final']:.3f} n25 {rc['n']} top5 {a['top5']:.2f} p {p:.3f} "
            f"nostop {fr_['cagr'] * 100:+.1f}/{fr_['mdd'] * 100:.1f} years {a['years']}",
            flush=True,
        )

    for name, ent in rules.items():
        report(name, ent)
    tr = np.r_[0.0, np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])])]
    atr = np.convolve(tr, np.ones(14) / 14, "full")[:n] / c
    X = np.c_[fund, zf, avgf, zo, zo7, zl, zt, zk, r24, atr]
    ok = np.isfinite(X).all(axis=1)
    br.MAX_BARS, br.STOP = 500, STOP
    tp = 0.02
    res = {s: br.outcomes(o, h, l, tp, s) for s in (1, -1)}
    probs = {1: np.zeros(n), -1: np.zeros(n)}
    cuts = {}
    for side in (1, -1):
        y = res[side][0].astype(float)
        for yr in range(2023, 2027):
            z0 = int(np.argmax(years == yr))
            trn = np.zeros(n, bool)
            trn[: z0 - br.MAX_BARS] = True
            trn &= ok & (y >= 0)
            te = np.nonzero((years == yr) & ok)[0]
            sc = StandardScaler().fit(X[trn])
            a, b = np.clip(sc.transform(X[trn]), -6, 6), np.clip(sc.transform(X[te]), -6, 6)
            g = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=7,
                min_samples_leaf=200,
                early_stopping=False,
                random_state=7,
            ).fit(a, y[trn].astype(int))
            lg = LogisticRegression(C=0.1, max_iter=300).fit(a, y[trn].astype(int))
            pe = (g.predict_proba(b)[:, 1] + lg.predict_proba(b)[:, 1]) / 2
            ins = (g.predict_proba(a)[:, 1] + lg.predict_proba(a)[:, 1]) / 2
            cuts[(side, yr)] = float(np.quantile(ins, 0.98))
            probs[side][te] = pe
            mask = y[te] >= 0
            print(
                f" M side{side} {yr} AUC {roc_auc_score(y[te][mask], pe[mask]):.4f} base {y[te][mask].mean():.3f} fire {(pe >= cuts[(side, yr)]).mean() * 100:.1f}%",
                flush=True,
            )
    ent = []
    for i in np.nonzero(ok & (years >= 2023))[0][:-1]:
        yr = int(years[i])
        pl, ps = probs[1][i], probs[-1][i]
        hl, hs = pl >= cuts[(1, yr)], ps >= cuts[(-1, yr)]
        if hl and (not hs or pl >= ps):
            ent.append((int(i), 1, tp))
        elif hs:
            ent.append((int(i), -1, tp))
    keyed = {(tp, s): res[s] for s in (1, -1)}
    s23 = int(np.argmax(years == 2023))
    dd = (bars[-1].day - bars[s23].day).days
    trades = br.trade(ent, keyed, COST, o)
    got = br.score(trades, years, dd)
    print(
        "M", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in got.items()}, flush=True
    )
    print(
        " cost10",
        br.score(br.trade(ent, keyed, 0.001, o), years, dd).get("cagr"),
        "2025+",
        br.score(br.trade([x for x in ent if x[0] >= s25], keyed, COST, o), years, d25).get(
            "final"
        ),
        flush=True,
    )
    _ = math


if __name__ == "__main__":
    main()
