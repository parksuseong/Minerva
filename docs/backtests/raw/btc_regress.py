"""BTCUSDT 15분봉 회귀 진입 (사전 등록: docs/backtests/2026-11-01-btc-regress-hypotheses.md).

numpy가 필요하다. 서버 패키지에 넣지 않으므로 별도 경로에 받아 쓴다.
실행: PYTHONPATH=src:<numpy 경로> python docs/backtests/raw/btc_regress.py <결과.json>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone

import numpy as np

from minerva.strategy import btc_bt as bt
from minerva.strategy import btc_features as bf
from minerva.strategy.btc_data import OUT
from minerva.strategy.indicators import cci, rsi, stoch_fast_k

STOP = 0.01
TARGETS = (0.003, 0.005, 0.0075, 0.01)
COST = 0.0005
MAX_BARS = 2000
YEARS = range(2020, 2027)
TOP = 0.99
BAD = (datetime(2018, 2, 8).date(), datetime(2018, 2, 11).date())


def outcomes(o, h, l, tp: float, side: int):
    """봉 i 신호, i+1 시가 진입. (win: 1/0/-1(시간 초과·끝), exit 봉, exit 가격)."""
    n = len(o)
    win = np.full(n, -1, dtype=np.int8)
    out = np.full(n, n - 1, dtype=np.int64)
    px = np.zeros(n)
    for i in range(n - 1):
        e = o[i + 1]
        up, dn = (e * (1 + tp), e * (1 - STOP)) if side == 1 else (e * (1 + STOP), e * (1 - tp))
        a, step = i + 1, 64
        end = min(i + 1 + MAX_BARS, n)
        while a < end:
            z = min(a + step, end)
            hit = (h[a:z] >= up) | (l[a:z] <= dn)
            if hit.any():
                j = a + int(hit.argmax())
                high, low = h[j] >= up, l[j] <= dn
                out[i] = j
                if side == 1:
                    good = high and not low
                    px[i] = max(o[j], up) if good else min(o[j], dn)
                else:
                    good = low and not high
                    px[i] = min(o[j], dn) if good else max(o[j], up)
                win[i] = int(good)
                break
            a, step = z, step * 2
    return win, out, px


def fit(x: np.ndarray, y: np.ndarray, lam: float = 1.0, rounds: int = 10) -> np.ndarray:
    """L2 로지스틱 회귀 (IRLS). x 첫 열은 절편."""
    beta = np.zeros(x.shape[1])
    pen = np.eye(x.shape[1]) * lam
    pen[0, 0] = 0
    for _ in range(rounds):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-6)
        grad = x.T @ (y - p) - pen @ beta
        beta = beta + np.linalg.solve((x.T * w) @ x + pen, grad)
    return beta


def predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))


def trade(signals, win_exit, cost: float, opens):
    """signals: (봉, 쪽, 익절) 시간순. 포지션이 없을 때만 진입. 거래별 (exit 봉, 수익, 승)."""
    out, free = [], -1
    for i, side, tp in signals:
        if i <= free:
            continue
        win, ex, px = win_exit[(tp, side)]
        if win[i] == -1:
            continue
        e = opens[i + 1]
        r = (side * (px[i] / e - 1) + 1) * (1 - cost) - (px[i] / e) * cost
        out.append((int(ex[i]), float(r - 1), int(r > 1)))
        free = int(ex[i])
    return out


def score(trades, years_of, days: int) -> dict:
    if not trades:
        return {"trades": 0}
    eq, peak, mdd = 1.0, 1.0, 0.0
    yr: dict[int, float] = {}
    for ex, r, _ in trades:
        eq *= 1 + r
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
        y = int(years_of[ex])
        yr[y] = yr.get(y, 1.0) * (1 + r)
    rets = sorted((r for _, r, _ in trades), reverse=True)
    rest = float(np.prod([1 + r for r in rets[5:]]))
    cagr = eq ** (bt.YEAR_DAYS / days) - 1
    return {
        "trades": len(trades),
        "win": sum(w for *_, w in trades) / len(trades),
        "final": eq,
        "cagr": cagr,
        "mdd": mdd,
        "ratio": cagr / abs(mdd) if mdd else 0.0,
        "no_top5": rest,
        "avg": float(np.mean(rets)),
        "years": {y: v - 1 for y, v in sorted(yr.items())},
    }


def main(path: str) -> None:
    bars = bt.load("15m", OUT)
    with (OUT / "15m.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(bars)
    taker = [float(r["taker_buy_base"]) for r in rows]
    ts = np.array([int(r["open_time"]) for r in rows])
    closes = [b.close for b in bars]
    feat = {k: v[0] for k, v in bf.features(bars, taker, OUT).items()}
    feat["Stoch"] = stoch_fast_k(bars, 5)
    feat["CCI"] = cci(bars, 30)
    feat["RSI"] = rsi(closes, 14)
    c = np.array(closes)
    for n in (4, 16, 96):
        r = np.full(len(c), np.nan)
        r[n:] = c[n:] / c[:-n] - 1
        feat[f"수익률{n}"] = r
    hour = np.array(
        [datetime.fromtimestamp(t / 1000, timezone.utc).hour for t in ts], dtype=float
    )
    feat["sin"] = np.sin(hour / 24 * 2 * np.pi)
    feat["cos"] = np.cos(hour / 24 * 2 * np.pi)
    names = list(feat)
    X = np.array(
        [[np.nan if v is None else v for v in feat[k]] for k in names], dtype=float
    ).T
    ok = np.isfinite(X).all(axis=1)
    days_of = [b.day for b in bars]
    years_of = np.array([d.year for d in days_of])
    bad = np.array([BAD[0] <= d <= BAD[1] for d in days_of])
    o = np.array([b.open for b in bars])
    h = np.array([b.high for b in bars])
    l = np.array([b.low for b in bars])  # noqa: E741
    print("입력 준비 끝", X.shape, flush=True)
    res = {tp: {s: outcomes(o, h, l, tp, s) for s in (1, -1)} for tp in TARGETS}
    print("결과 계산 끝", flush=True)
    keyed = {(tp, s): res[tp][s] for tp in TARGETS for s in (1, -1)}
    start = int(np.argmax(years_of == YEARS[0]))
    days = (days_of[-1] - days_of[start]).days
    report: dict = {"names": names, "variants": {}}
    for tp in TARGETS:
        be = (STOP + 2 * COST) / (STOP + tp)  # 손익 분기 승률
        fixed = max(0.70, be + 0.02)
        probs = {("P", s): np.zeros(len(o)) for s in (1, -1)}
        cuts = {}
        coef = {}
        base = {}
        for side in (1, -1):
            y = res[tp][side][0].astype(float)
            for yr in YEARS:
                t0 = int(np.argmax(years_of == yr))
                test = years_of == yr
                train = np.zeros(len(o), bool)
                train[: t0 - MAX_BARS] = True
                train &= ok & ~bad & (y >= 0)
                mu, sd = X[train].mean(0), X[train].std(0) + 1e-12
                design = lambda m: np.c_[  # noqa: E731
                    np.ones(int(m.sum())), np.clip((X[m] - mu) / sd, -5, 5)
                ]
                beta = fit(design(train), y[train])
                inner = predict(design(train), beta)
                cuts[(side, yr)] = float(np.quantile(inner, TOP))
                mask = test & ok
                probs[("P", side)][mask] = predict(design(mask), beta)
                coef[f"{side}/{yr}"] = [round(float(v), 3) for v in beta[1:]]
                base[f"{side}/{yr}"] = float(y[test & (y >= 0)].mean())
        for name in ("P", "Q"):
            for cost in (COST, 0.001):
                sig = []
                for i in np.nonzero(ok & (years_of >= YEARS[0]))[0][:-1]:
                    yr = int(years_of[i])
                    pl, ps = probs[("P", 1)][i], probs[("P", -1)][i]
                    tl = fixed if name == "P" else cuts[(1, yr)]
                    ts_ = fixed if name == "P" else cuts[(-1, yr)]
                    hit_l, hit_s = pl >= tl, ps >= ts_
                    if hit_l and (not hit_s or pl >= ps):
                        sig.append((int(i), 1, tp))
                    elif hit_s:
                        sig.append((int(i), -1, tp))
                got = score(trade(sig, keyed, cost, o), years_of, days)
                got["signals"] = len(sig)
                got["breakeven"] = be
                got["threshold"] = fixed if name == "P" else "top1%"
                report["variants"][f"{name}/{tp}/{cost}"] = got
                print(f"{name}/{tp}/{cost}", {k: v for k, v in got.items()}, flush=True)
        report[f"coef/{tp}"] = coef
        report[f"base/{tp}"] = base
    with open(path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
