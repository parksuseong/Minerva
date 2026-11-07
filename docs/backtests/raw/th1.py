"""145번 TH1: 일봉 방향 롱만, ATR 위험 크기, 1% 손절. 등록: 2026-11-07-th1-final-spec.md

144번 엔진(trend_hold_cross_market.py)에서 고친 것: 끝에 열린 포지션 평가, 펀딩을 정산 시각 명목으로,
손절 미끄러짐 0.05%, 15분봉 평가 낙폭.
실행: PYTHONPATH=<pylib>:src:docs/backtests/raw python docs/backtests/raw/th1.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import numpy as np
from trend_hold_cross_market import BAR, DAY, MARKETS, M5, Mkt, daily_atr, ms

FEE = 0.0005
SLIP = 0.0005
STOP = 0.01
CAP = 3.0
RISKS = (0.015, 0.02)
KRW = 1_000_000
KST = timezone(timedelta(hours=9))


def fund_pay(m: Mkt, qty: float, t0: float, t1: float) -> tuple:
    """(t0, t1] 정산마다 qty × 정산 시각 가격 × 비율. (정산 시각 배열, 누적 지급액)."""
    a = np.searchsorted(m.ft, t0, side="right")
    b = np.searchsorted(m.ft, t1, side="right")
    ft, fr = m.ft[a:b], m.fr[a:b]
    if len(ft) == 0:
        return ft, np.zeros(0)
    j = np.clip(np.searchsorted(m.t5, ft), 0, len(m.t5) - 1)
    px = m.d5["o"][j]
    return ft, np.cumsum(qty * px * fr)


def run(m: Mkt, reg: np.ndarray, newday: np.ndarray, atr: np.ndarray, risk: float) -> tuple:
    d = m.d
    t, o, l, c = d["t"], d["o"], d["l"], d["c"]  # noqa: E741
    n = m.n
    eq = np.full(n, np.nan)
    inpos = np.zeros(n, bool)
    E = 1.0
    eq[: m.start] = 1.0
    trades = []
    i, can, last = m.start, True, m.start
    open_pos = None
    while i < n - 3:
        if newday[i]:
            can = True
        if not (int(reg[i]) == 1 and can):
            i += 1
            continue
        a = atr[i]
        if not (a > 0):
            i += 1
            continue
        w = min(CAP, risk / a)
        e = i + 1
        px, j5 = m.px5(t[e], o[e])
        t_in = t[e] + M5
        E0 = E
        N = w * E0
        qty = N / px
        stop = px * (1 - STOP)
        x = None
        if j5 >= 0:
            for jj in (j5, j5 + 1):
                if jj < len(m.t5) and m.t5[jj] < t[e] + BAR and m.d5["l"][jj] <= stop:
                    x = (e, stop * (1 - SLIP), m.t5[jj] + M5, True)
                    break
        elif l[e] <= stop:
            x = (e, stop * (1 - SLIP), t[e] + BAR, True)
        b = e
        while x is None and b < n - 2:
            if b > e and l[b] <= stop:
                x = (b, min(o[b], stop) * (1 - SLIP), t[b] + BAR, True)
                break
            if int(reg[b]) != 1:
                xp, _ = m.px5(t[b + 1], o[b + 1])
                x = (b + 1, xp, t[b + 1] + M5, False)
                break
            b += 1
        if x is None:  # 끝까지 보유 중: 마지막 종가로 평가
            xb, xp, t_out, stopped = n - 1, float(c[-1]), float(t[-1] + BAR), None
        else:
            xb, xp, t_out, stopped = x
        ft, cum = fund_pay(m, qty, t_in, t_out)
        seg = np.arange(e, xb)
        if len(seg):
            k = np.searchsorted(ft, t[seg] + BAR, side="right")
            paid = np.where(k > 0, cum[np.maximum(k - 1, 0)], 0.0)
            eq[seg] = E0 - N * FEE + qty * (c[seg] - px) - paid
            inpos[seg] = True
        fund_all = float(cum[-1]) if len(cum) else 0.0
        exit_fee = 0.0 if stopped is None else qty * xp * FEE
        E = E0 - N * FEE + qty * (xp - px) - fund_all - exit_fee
        eq[last:e] = np.where(np.isnan(eq[last:e]), E0, eq[last:e])
        eq[xb] = E
        rec = dict(e=int(e), x=int(xb), t_in=float(t_in), t_out=float(t_out), px=float(px), xp=float(xp),
                   ret=float((E - E0) / E0), pnl=float(E - E0), eq0=float(E0), w=float(w),
                   stopped=stopped, open=stopped is None)
        if stopped is None:
            open_pos = rec
            break
        trades.append(rec)
        last = xb + 1
        if stopped:
            can = False
            i = xb
        else:
            i = xb - 1
    if open_pos is None:
        eq[last:] = np.where(np.isnan(eq[last:]), E, eq[last:])
    for k in range(1, n):
        if np.isnan(eq[k]):
            eq[k] = eq[k - 1]
    return eq, inpos, trades, open_pos


def daily(m: Mkt, eq: np.ndarray) -> tuple:
    t = m.d["t"]
    key = ((t + BAR) // DAY).astype(np.int64)
    last = np.flatnonzero(np.r_[key[1:] != key[:-1], True])
    return key[last] * DAY, eq[last]


def stats(dt: np.ndarray, de: np.ndarray, t0: float, t1: float | None = None) -> dict:
    msk = dt >= t0
    if t1 is not None:
        msk &= dt < t1
    x, e = dt[msk], de[msk]
    if len(e) < 30:
        return {}
    e = e / e[0]
    yrs = (x[-1] - x[0]) / (365.25 * DAY)
    cagr = e[-1] ** (1 / yrs) - 1 if e[-1] > 0 else -1.0
    dd = e / np.maximum.accumulate(e) - 1
    mdd = float(dd.min())
    # 가장 긴 회복: 고점에서 다시 고점까지 걸린 일수
    peak_i, longest = 0, 0
    for k in range(1, len(e)):
        if e[k] >= e[peak_i]:
            longest = max(longest, k - peak_i)
            peak_i = k
    longest = max(longest, len(e) - 1 - peak_i)
    return {"cagr": round(cagr * 100, 2), "mdd": round(mdd * 100, 2),
            "ratio": round(cagr / abs(mdd), 3) if mdd < 0 else None, "ret": round((e[-1] - 1) * 100, 1),
            "recovery_days": int(longest)}


def mdd15(eq: np.ndarray, start: int) -> float:
    e = eq[start:] / eq[start]
    return round(float((e / np.maximum.accumulate(e) - 1).min()) * 100, 2)


def won(m: Mkt, eq: np.ndarray, trades: list, open_pos, dt: np.ndarray, de: np.ndarray) -> dict:
    e = de[dt >= m.d["t"][m.start]]
    x = dt[dt >= m.d["t"][m.start]]
    v = e / e[0] * KRW
    peak = np.maximum.accumulate(v)
    dd_won = float((v - peak).min())
    mon = np.array([datetime.fromtimestamp(z / 1000, timezone.utc).strftime("%Y-%m") for z in x])
    ends = np.flatnonzero(np.r_[mon[1:] != mon[:-1], True])
    mv = v[ends]
    mchg = np.diff(np.r_[KRW, mv])
    worst_m = int(np.argmin(mchg))
    scale = KRW / float(eq[m.start])
    loss = [z["pnl"] * scale for z in trades if z["pnl"] < 0]
    loss_pct = [z["ret"] for z in trades if z["ret"] < 0]
    return {"final": round(float(v[-1])), "worst_dd_won": round(dd_won), "worst_month": mon[ends][worst_m],
            "worst_month_won": round(float(mchg[worst_m])),
            "median_loss_won": round(float(np.median(loss))) if loss else None,
            "median_loss_pct_of_equity": round(float(np.median(loss_pct)) * 100, 3) if loss_pct else None,
            "loss_now_won": round(float(np.median(loss_pct)) * float(v[-1])) if loss_pct else None,
            "open": None if open_pos is None else {
                "since": datetime.fromtimestamp(open_pos["t_in"] / 1000, KST).strftime("%Y-%m-%d %H:%M"),
                "px": round(open_pos["px"], 1), "now": round(open_pos["xp"], 1),
                "ret_pct": round(open_pos["ret"] * 100, 2), "w": round(open_pos["w"], 3)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    keep = {}
    for mk in MARKETS:
        m = Mkt(mk)
        t = m.d["t"]
        t0 = float(t[m.start])
        reg, newday = R.regime(m.d5, m.d, 0)
        atr = daily_atr(m.d5, m.d, 0)
        bh = m.d["c"] / m.d["c"][m.start]
        bh[: m.start] = 1.0
        bdt, bde = daily(m, bh)
        rep[mk] = {"span": [datetime.fromtimestamp(t0 / 1000, timezone.utc).strftime("%Y-%m-%d"),
                            datetime.fromtimestamp(t[-1] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")],
                   "BH": {"all": stats(bdt, bde, t0), "2024~": stats(bdt, bde, ms("2024-01-01"))}}
        keep[(mk, "BH")] = (bdt, bde)
        for r in RISKS:
            eq, inpos, tr, op = run(m, reg, newday, atr, r)
            dt, de = daily(m, eq)
            g = np.array([z["ret"] for z in tr]) if tr else np.zeros(0)
            yrs = sorted({datetime.fromtimestamp(z / 1000, timezone.utc).year for z in dt if z >= t0})
            row = {"all": stats(dt, de, t0), "2024~": stats(dt, de, ms("2024-01-01")),
                   "mdd15": mdd15(eq, m.start), "trades": len(tr),
                   "win": round(float((g > 0).mean() * 100), 1) if len(g) else None,
                   "avg_trade_pct": round(float(g.mean() * 100), 3) if len(g) else None,
                   "stopped_pct": round(float(np.mean([z["stopped"] for z in tr]) * 100), 1) if tr else None,
                   "avg_notional": round(float(np.mean([z["w"] for z in tr])), 3) if tr else None,
                   "in_mkt": round(float(inpos[m.start:].mean() * 100), 1),
                   "years": {y: stats(dt, de, max(t0, ms(f"{y}-01-01")), ms(f"{y + 1}-01-01")).get("ret")
                             for y in yrs}}
            if mk == "M1":
                row["krw"] = won(m, eq, tr, op, dt, de)
            row["open"] = None if op is None else {"since": op["t_in"], "px": op["px"], "ret": op["ret"]}
            rep[mk][f"r{r * 100:g}"] = row
            keep[(mk, f"r{r * 100:g}")] = (dt, de)
            if mk == "M1":
                (out / f"trades_M1_r{r * 100:g}.json").write_text(json.dumps(tr + ([op] if op else [])))
            print(mk, r, row["all"], row["2024~"], row["mdd15"], row["trades"], flush=True)
        if mk == "M1":
            last_close = datetime.fromtimestamp((t[-1] + BAR) / 1000, KST)
            nxt = (last_close + timedelta(days=1)).replace(hour=9, minute=0) if last_close.hour >= 9 else \
                last_close.replace(hour=9, minute=0)
            rep["now"] = {"last_bar_close_kst": last_close.strftime("%Y-%m-%d %H:%M"),
                          "daily_direction": int(reg[-1]), "th1_long_now": int(reg[-1]) == 1,
                          "next_decision_kst": nxt.strftime("%Y-%m-%d %H:%M")}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    np.savez(out / "series.npz", **{f"{a}|{b}|t": v[0] for (a, b), v in keep.items()},
             **{f"{a}|{b}|e": v[1] for (a, b), v in keep.items()})
    print(json.dumps(rep.get("now"), ensure_ascii=False))


if __name__ == "__main__":
    main()
