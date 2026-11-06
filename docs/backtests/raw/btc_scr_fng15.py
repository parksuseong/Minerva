"""15분봉 SCR 3단 + 크립토 공포·탐욕 지수 문. 등록: 2026-11-06-btc-scr15-fng-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:docs/backtests/raw python docs/backtests/raw/btc_scr_fng15.py \
    --out <폴더> [--offset 5] [--lag 24] [--futures] [--perm]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import btc_scr_model as M
from btc_fractal import load5, resample, scr3
from btc_scr_gate15 import BAR, load_funding
from btc_swing_patterns import SPLIT

GATES = ("none", "W", "D", "FEAR", "TREND")
DAY = 86_400_000
H = 3_600_000


def load_fng(path: Path) -> tuple:
    rows = sorted((int(x["timestamp"]) * 1000, float(x["value"])) for x in json.loads(path.read_text())["data"])
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def gate_series(ft: np.ndarray, fv: np.ndarray, lag_h: int) -> dict:
    """문마다 (쓸 수 있는 시각, 롱 허용, 숏 허용). 시각 순으로 정렬."""
    avail = ft + lag_h * H
    out = {}
    out["FEAR"] = (avail, fv <= 40, fv >= 60)
    out["TREND"] = (avail, fv > 50, fv < 50)
    # 일별 5·10일선 (값 개수 기준, 빈 날은 건너뜀)
    c = np.cumsum(np.r_[0.0, fv])
    def ma(n: int) -> np.ndarray:
        r = np.full(len(fv), np.nan)
        r[n - 1:] = (c[n:] - c[:-n]) / n
        return r

    m5, m10 = ma(5), ma(10)
    ok = ~np.isnan(m10)
    out["D"] = (avail, ok & (m5 > m10), ok & (m5 < m10))
    # 주봉(월~일) 평균, 마감된 주만: 그 주 마지막 값이 쓸 수 있게 된 시각부터
    wk = [datetime.fromtimestamp(t / 1000, timezone.utc).isocalendar()[:2] for t in ft]
    keys, first = np.unique(np.array([y * 100 + w for y, w in wk]), return_index=True)
    order = np.argsort(first)
    first = first[order]
    last = np.r_[first[1:], len(ft)] - 1
    wv = np.array([fv[a:b + 1].mean() for a, b in zip(first, last)])
    wt = avail[last]
    cw = np.cumsum(np.r_[0.0, wv])
    w5 = np.r_[np.full(4, np.nan), (cw[5:] - cw[:-5]) / 5]
    w10 = np.r_[np.full(9, np.nan), (cw[10:] - cw[:-10]) / 10]
    okw = ~np.isnan(w10)
    # 마지막 주가 진행 중이면 아직 마감 전이므로 뺀다
    now_week = datetime.fromtimestamp(ft[-1] / 1000, timezone.utc).weekday() != 6
    sl = slice(0, len(wv) - 1) if now_week else slice(0, len(wv))
    out["W"] = (wt[sl], (okw & (w5 > w10))[sl], (okw & (w5 < w10))[sl])
    return out


def at(series: tuple, t_close: np.ndarray, side: np.ndarray) -> np.ndarray:
    avail, lo, sh = series
    j = np.searchsorted(avail, t_close, side="right") - 1
    jj = np.maximum(j, 0)
    return (j >= 0) & np.where(side > 0, lo[jj], sh[jj])


def summ(ts: list, cost: float = 0.0005) -> dict:
    if not ts:
        return {"n": 0}
    net = np.array([x[2] for x in ts]) - 2 * cost
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    eq3 = np.cumprod(1 + np.maximum(3 * net, -1))
    peak3 = np.maximum.accumulate(np.r_[1.0, eq3])
    s = np.sort(net)
    return {"n": len(ts), "win": round(float((net > 0).mean() * 100), 1),
            "ev": round(float(net.mean() * 100), 3), "ret": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
            "ret3": round(float((eq3[-1] - 1) * 100), 1),
            "mdd3": round(float((eq3 / peak3[1:] - 1).min() * 100), 1),
            "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None,
            "long": int(sum(1 for x in ts if x[1] > 0))}


def take(sigs: list, allow: dict) -> list:
    busy, tr = -1, []
    for k, (i, side, g, x, t) in enumerate(sigs):
        if i <= busy or not allow[k]:
            continue
        tr.append((i, side, g, x, t))
        busy = x
    return tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--lag", type=int, default=1, help="지수 값을 쓰기 시작하는 시각(값 시각 + 시간)")
    ap.add_argument("--futures", action="store_true")
    ap.add_argument("--perm", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / ("futures/klines_5m.csv" if args.futures else "5m.csv"))
    fund = load_funding(base / "futures/funding.csv") if args.futures else None
    ft, fv = load_fng(base / "fng.json")
    gs = gate_series(ft, fv, args.lag)
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, 3)
    d["t"] = d["t"] + args.offset * 60_000
    n = len(d["c"])
    on_l, on_s, *_ = scr3(d)
    t_start = ft[0] + 14 * DAY  # D문 10일선이 차기 전 신호는 빼서 모든 문을 같은 기간으로 맞춘다
    sigs = []
    for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
        for i in np.flatnonzero(on):
            if i < 6000 or i >= n - 2 or d["t"][i] < t_start:
                continue
            res = M.label(d, i, side, opp)
            if res is None:
                continue
            g, x = res
            if fund is not None:
                t0, t1 = d["t"][i + 1], d["t"][x]
                m = (fund[0] > t0) & (fund[0] <= t1)
                g -= side * float(fund[1][m].sum())
            sigs.append((int(i), side, float(g), int(x), float(d["t"][i])))
    sigs.sort()
    tc = np.array([s[4] for s in sigs]) + BAR
    sd = np.array([s[1] for s in sigs])
    allow = {"none": np.ones(len(sigs), bool)}
    for gname in GATES[1:]:
        allow[gname] = at(gs[gname], tc, sd)
    end = d["t"][-1] + BAR
    cut = {"1y": end - 365 * DAY, "30d": end - 30 * DAY}
    rep: dict = {"offset": args.offset, "lag": args.lag, "futures": args.futures,
                 "first_sig": datetime.fromtimestamp(sigs[0][4] / 1000, timezone.utc).isoformat(),
                 "end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat()}
    trades = {}
    for gname in GATES:
        tr = take(sigs, allow[gname])
        trades[gname] = tr
        r = {"all": summ(tr), "all_c10": summ(tr, 0.001),
             "old": summ([z for z in tr if z[4] < SPLIT]), "new": summ([z for z in tr if z[4] >= SPLIT])}
        for k, c in cut.items():
            r[k] = summ([z for z in tr if z[4] >= c])
        r["trades"] = [[z[0], z[1], round(z[2], 5), z[4], z[3]] for z in tr]
        rep[gname] = r
    if args.perm:  # none 거래에서 같은 수를 무작위로 뽑아 평균 순손익 비교
        rng = np.random.default_rng(7)
        base_g = np.array([z[2] for z in trades["none"]])
        for gname in GATES[1:]:
            k = len(trades[gname])
            if k < 2:
                continue
            actual = float(np.mean([z[2] for z in trades[gname]]))
            draws = np.array([base_g[rng.choice(len(base_g), k, replace=False)].mean() for _ in range(2000)])
            rep[gname]["perm_p"] = round(float((draws >= actual).mean()), 4)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("first", rep["first_sig"], "end", rep["end"])
    for gname in GATES:
        r = rep[gname]
        print(gname, r["all"], "c10", r["all_c10"].get("ev"), "old", r["old"].get("n"), r["old"].get("ev"),
              "new", r["new"].get("n"), r["new"].get("ev"), "1y", r["1y"], r.get("perm_p", ""), flush=True)


if __name__ == "__main__":
    main()
