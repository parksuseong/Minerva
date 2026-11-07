"""132번: S1W(30분, 완화 SCR3, 익절 3%·손절 1.5%) 표본 밖 확인. 등록: 2026-11-07-btc-s1w-holdout-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_s1w_holdout.py <출력 폴더>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from btc_fractal import load5, resample
from btc_scr_gate15 import load_funding
from btc_scr_week_opt import COST, M5, ms, signals, simulate, summ

EX = ("TP", 0.03, 0.015)
HOLD = (ms("2020-01-15 00:00"), ms("2024-01-01 00:00"))
SPOT = (ms("2017-09-15 00:00"), ms("2020-01-01 00:00"))


def frame30(d5: dict, offset_min: int = 0) -> dict:
    sh = dict(d5)
    sh["t"] = d5["t"] - offset_min * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + offset_min * 60_000
    return d


def span(d, w):
    t = d["t"]
    return int(np.searchsorted(t, w[0])), int(np.searchsorted(t, w[1]))


def run_a(d, L, S, r, w, fund):
    i0, i1 = span(d, w)
    return simulate(d, L, S, r, None, EX, i0, i1, fund)


def run_b(d, d5, L, S, w, fund):
    """현실 체결: 30분봉 마감 + 5분에 시작하는 5분봉 시가에 진입, 이후 5분봉으로 손절·익절."""
    tp, sl = EX[1], EX[2]
    t5, o5, h5, l5 = d5["t"], d5["o"], d5["h"], d5["l"]
    i0, i1 = span(d, w)
    end5 = int(np.searchsorted(t5, w[1]))
    sig = np.where(L, 1, np.where(S, -1, 0))
    out, busy_t = [], -1.0
    for i in np.flatnonzero(sig[i0:i1]) + i0:
        close_t = d["t"][i] + 6 * M5
        if close_t <= busy_t:
            continue
        k = int(np.searchsorted(t5, close_t + M5))
        if k >= end5 or t5[k] != close_t + M5:
            continue
        sd = int(sig[i])
        px = o5[k]
        stop, tgt = px * (1 - sl * sd), px * (1 + tp * sd)
        xk, xp = None, None
        for j in range(k, end5):
            if (sd > 0 and l5[j] <= stop) or (sd < 0 and h5[j] >= stop):
                xp = (min(o5[j], stop) if sd > 0 else max(o5[j], stop)) if j > k else stop
                xk = j
                break
            if (sd > 0 and h5[j] >= tgt) or (sd < 0 and l5[j] <= tgt):
                xp = max(o5[j], tgt) if sd > 0 else min(o5[j], tgt)
                xk = j
                break
        if xk is None:
            xk = end5 - 1
            xp = d5["c"][xk]
        g = (xp / px - 1) * sd
        m = (fund[0] > t5[k]) & (fund[0] <= t5[xk]) if fund is not None else None
        if m is not None:
            g -= sd * float(fund[1][m].sum())
        out.append((g - 2 * COST, float(t5[k]), float(t5[xk]), sd, i, xk, float(px), float(xp)))
        busy_t = t5[xk] + M5
    return out


def per(tr):
    return float(np.mean([z[0] for z in tr])) if tr else 0.0


def drop5(tr):
    g = sorted(z[0] for z in tr)
    return round(float(np.mean(g[:-5])) * 100, 3) if len(g) > 5 else None


def circ(d, L, S, r, w, fund, n, rng):
    i0, i1 = span(d, w)
    seg = slice(i0, i1)
    vals = []
    for _ in range(n):
        k = int(rng.integers(96, (i1 - i0) - 96))
        L2, S2 = L.copy(), S.copy()
        L2[seg], S2[seg] = np.roll(L[seg], k), np.roll(S[seg], k)
        vals.append(per(simulate(d, L2, S2, r, None, EX, i0, i1, fund)))
    return np.array(vals)


def random_entries(d, L, S, r, w, fund, n, rng):
    """시험 구간의 실제 롱·숏 신호 개수만큼 무작위 봉에 신호를 놓는다(한 번에 하나 규칙은 simulate가 지킨다)."""
    i0, i1 = span(d, w)
    nl, ns = int(L[i0:i1].sum()), int(S[i0:i1].sum())
    vals = []
    for _ in range(n):
        L2 = np.zeros(len(d["t"]), bool)
        S2 = np.zeros(len(d["t"]), bool)
        L2[rng.choice(np.arange(i0, i1 - 1), nl, replace=False)] = True
        S2[rng.choice(np.arange(i0, i1 - 1), ns, replace=False)] = True
        S2 &= ~L2
        vals.append(per(simulate(d, L2, S2, r, None, EX, i0, i1, fund)))
    return np.array(vals)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    base = Path("~/.minerva/data/btc").expanduser()
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    rng = np.random.default_rng(7)
    rep: dict = {}
    d = frame30(d5)
    L, S, s, cc, r = signals(d, "REL")
    ta = run_a(d, L, S, r, HOLD, fund)
    tb = run_b(d, d5, L, S, HOLD, fund)
    d15 = frame30(d5, 15)
    L15, S15, _, _, r15 = signals(d15, "REL")
    t15 = run_a(d15, L15, S15, r15, HOLD, fund)
    cn = circ(d, L, S, r, HOLD, fund, 200, rng)
    rn = random_entries(d, L, S, r, HOLD, fund, 200, rng)
    act = per(ta)
    rep["hold_a"] = summ(ta)
    rep["hold_b"] = summ(tb)
    rep["long_a"] = summ([z for z in ta if z[3] > 0])
    rep["short_a"] = summ([z for z in ta if z[3] < 0])
    rep["drop5_a"] = drop5(ta)
    rep["offset15_a"] = summ(t15)
    rep["circ"] = {"null_mean": round(float(cn.mean()) * 100, 3), "p": round(float((cn >= act).mean()), 4)}
    rep["random"] = {"null_mean": round(float(rn.mean()) * 100, 3), "p": round(float((rn >= act).mean()), 4)}
    rep["years_a"] = {y: summ([z for z in ta if ms(f"{y}-01-01 00:00") <= z[1] < ms(f"{y + 1}-01-01 00:00")])
                      for y in range(2020, 2024)}
    crit = {
        "1_a_pos": act > 0,
        "2_b_pos": per(tb) > 0,
        "3_long_short": per([z for z in ta if z[3] > 0]) >= 0 and per([z for z in ta if z[3] < 0]) >= 0,
        "4_n100": len(ta) >= 100,
        "5_drop5": (drop5(ta) or -1) > 0,
        "6_offset15": per(t15) > 0,
        "7_circ": rep["circ"]["p"] < 0.05,
        "8_random": act > rn.mean() and rep["random"]["p"] < 0.05,
    }
    rep["criteria"] = {k: bool(v) for k, v in crit.items()}
    rep["pass"] = bool(all(crit.values()))
    # 참고: 현물 2017-09~2019-12, 펀딩 없음
    sp = load5(base / "5m.csv")
    ds = frame30(sp)
    Ls, Ss, _, _, rs = signals(ds, "REL")
    rep["spot_a"] = summ(run_a(ds, Ls, Ss, rs, SPOT, (np.array([]), np.array([]))))
    trades = {"a": [list(z) for z in ta], "b": [list(z) for z in tb]}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "trades.json").write_text(json.dumps(trades))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
