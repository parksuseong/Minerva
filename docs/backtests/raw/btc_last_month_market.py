"""최근 한 달, 추적 중인 7개 기법을 '시장가로 원하는 가격에 체결' 가정으로 다시 계산한다.

설명용 점검이고 판정 시험이 아니다. 문서: 2026-11-07-btc-last-month-market-fill.md
실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_last_month_market.py \
    --out <폴더> [--start 2026-10-07]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import numpy as np
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_scr_learned import bars

KST = timezone(timedelta(hours=9))
BAR = 900_000
ML_R, R1_R = 0.015, 0.01
FEE = 0.0005
RES = Path("~/.minerva/research/btc-2026-11-04").expanduser()
TRK = Path("~/.minerva/research/btc-ml-tracker").expanduser()
FUT5 = Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()
RULES = {
    "A": ("one", 0.90, 4),
    "B": ("one", 0.95, 5),
    "AS": ("mean", 0.90, 4),
    "BS": ("mean", 0.95, 5),
    "AS4": ("mean", 0.90, 4),
}


def load_spot5() -> dict:
    """현물 5분봉(4시간 추세용): 저장 자료 + 추적기의 live5m.csv."""
    rows = {}
    for p in (Path("~/.minerva/data/btc/5m.csv").expanduser(), TRK / "live5m.csv"):
        with p.open() as f:
            r = csv.reader(f)
            next(r)
            for x in r:
                rows[int(x[0])] = (float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))
    t = np.array(sorted(rows), dtype=float)
    v = np.array([rows[int(k)] for k in t])
    return {"t": t, "o": v[:, 0], "h": v[:, 1], "l": v[:, 2], "c": v[:, 3], "v": v[:, 4]}


def signals(t15: np.ndarray) -> tuple:
    """규칙별 신호(+1/−1/0)를 선물 15분봉 시각 t15에 맞춘다.

    11/03 13:15 UTC까지는 백테스트 예측(pred.npz, 씨드 5벌), 그 뒤는 추적기 signals.csv(읽기만)."""
    z = np.load(RES / "ml15/pred.npz")
    tp = z["t"]
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    pm = np.mean(ps, 0)
    tr4_all = trend4h(load_spot5(), t15)
    out = {k: np.zeros(len(t15), int) for k in RULES}
    src = {"backtest": {}, "tracker": {}}
    j = np.searchsorted(t15, tp)
    okb = (j < len(t15)) & (t15[np.minimum(j, len(t15) - 1)] == tp)
    for k, (which, q, agree) in RULES.items():
        li, si = rule_masks(z["probs"] if which == "one" else pm, tp, q, agree)
        s = np.where(li, 1, np.where(si, -1, 0))
        out[k][j[okb]] = s[okb]
    last_bt = float(tp[-1])
    with (TRK / "signals.csv").open() as f:
        rows = list(csv.DictReader(f))
    tt = np.array([float(r["t"]) for r in rows])
    jt = np.searchsorted(t15, tt)
    okt = (jt < len(t15)) & (t15[np.minimum(jt, len(t15) - 1)] == tt) & (tt > last_bt)
    for k in RULES:
        col = "AS" if k == "AS4" else k
        s = np.array([int(r[col]) for r in rows])
        out[k][jt[okt]] = s[okt]
    out["AS4"] = np.where(out["AS4"] == tr4_all, out["AS4"], 0)
    for k in RULES:
        src["backtest"][k] = int((out[k][t15 <= last_bt] != 0).sum())
        src["tracker"][k] = int((out[k][t15 > last_bt] != 0).sum())
    seam = {
        "backtest_last": last_bt,
        "tracker_first": float(tt.min()),
        "gap_bars": int(((t15 > last_bt) & (t15 < tt.min())).sum()),
        "tracker_rows_used": int(okt.sum()),
    }
    return out, src, seam


def funding(ft, fr, t0, t1, s) -> float:
    m = (ft > t0) & (ft <= t1)
    return -s * float(fr[m].sum())


def ml_trades(d, sig, a, b) -> list:
    """시장가 진입 = 신호 봉 종가. ±1.5%, 같은 봉 둘 다 닿으면 손절. 한 번에 한 포지션."""
    o, h, l, c, t = d["o"], d["h"], d["l"], d["c"], d["t"]
    tr, i = [], a
    while i < b:
        s = int(sig[i])
        if not s:
            i += 1
            continue
        px = c[i]
        tp_, sp = px * (1 + ML_R * s), px * (1 - ML_R * s)
        x = None
        for j in range(i + 1, len(c)):
            if (s > 0 and l[j] <= sp) or (s < 0 and h[j] >= sp):
                fill = min(o[j], sp) if s > 0 else max(o[j], sp)
                x = (j, fill, "손절", t[j] + BAR)
                break
            if (s > 0 and h[j] >= tp_) or (s < 0 and l[j] <= tp_):
                fill = max(o[j], tp_) if s > 0 else min(o[j], tp_)
                x = (j, fill, "익절", t[j] + BAR)
                break
        if x is None:
            tr.append(
                {
                    "i": i,
                    "s": s,
                    "px": px,
                    "x": len(c) - 1,
                    "xp": c[-1],
                    "why": "보유 중",
                    "t0": t[i] + BAR,
                    "t1": t[-1] + BAR,
                    "open": True,
                }
            )
            break
        tr.append(
            {
                "i": i,
                "s": s,
                "px": px,
                "x": x[0],
                "xp": x[1],
                "why": x[2],
                "t0": t[i] + BAR,
                "t1": x[3],
                "open": False,
            }
        )
        i = x[0]
    return tr


def r1_leg(d, reg, i, s, px):
    """R1식 관리: 1% 손절(갭이면 시가), 방향이 바뀐 봉 다음 시가에 청산."""
    o, h, l, c, t = d["o"], d["h"], d["l"], d["c"], d["t"]
    sp = px * (1 - R1_R * s)
    for j in range(i + 1, len(c)):
        if (s > 0 and l[j] <= sp) or (s < 0 and h[j] >= sp):
            fill = min(o[j], sp) if s > 0 else max(o[j], sp)
            return j, fill, "손절", t[j] + BAR, False
        if reg[j] != s:
            if j + 1 < len(c):
                return j + 1, o[j + 1], "방향 전환", t[j + 1], False
            break
    return len(c) - 1, c[-1], "보유 중", t[-1] + BAR, True


def r1_trades(d, reg, newday, a, b) -> list:
    """일봉 마감 봉 종가에 시장가 진입. 손절 뒤에는 다음 일봉 마감까지 기다린다."""
    c, t = d["c"], d["t"]
    tr, i = [], a
    can = True
    while i < b:
        if newday[i]:
            can = True
        s = int(reg[i])
        if s and can:
            x, xp, why, t1, op = r1_leg(d, reg, i, s, c[i])
            tr.append(
                {
                    "i": i,
                    "s": s,
                    "px": c[i],
                    "x": x,
                    "xp": xp,
                    "why": why,
                    "t0": t[i] + BAR,
                    "t1": t1,
                    "open": op,
                }
            )
            if op:
                break
            if why == "손절":
                can, i = False, x
            else:
                i = x - 1 if x - 1 > i else i + 1
            continue
        i += 1
    return tr


def k2_trades(d, reg, sig, a, b) -> list:
    """R1 방향과 같은 AS4 신호의 봉 종가에 시장가 진입, R1식 청산."""
    c, t = d["c"], d["t"]
    tr, i = [], a
    while i < b:
        s = int(reg[i])
        if s and int(sig[i]) == s:
            x, xp, why, t1, op = r1_leg(d, reg, i, s, c[i])
            tr.append(
                {
                    "i": i,
                    "s": s,
                    "px": c[i],
                    "x": x,
                    "xp": xp,
                    "why": why,
                    "t0": t[i] + BAR,
                    "t1": t1,
                    "open": op,
                }
            )
            if op:
                break
            i = x
            continue
        i += 1
    return tr


def finish(tr, ft, fr, cost: bool) -> None:
    for z in tr:
        g = (z["xp"] / z["px"] - 1) * z["s"]
        f = funding(ft, fr, z["t0"], z["t1"], z["s"]) if cost else 0.0
        z["gross"] = g
        z["net"] = g + f - (2 * FEE if cost else 0.0)


def equity(d, tr, a, lev: float, cost: bool) -> np.ndarray:
    """15분봉 평가 자산(복리, 한 번에 한 포지션)."""
    c = d["c"]
    n = len(c)
    eq = np.ones(n - a)
    cap = 1.0
    k = 0
    for idx in range(a, n):
        while k < len(tr) and tr[k]["x"] < idx and not tr[k]["open"]:
            z = tr[k]
            cap *= max(1 + lev * z["net"], 0.0)
            k += 1
        val = cap
        if k < len(tr) and tr[k]["i"] <= idx:
            z = tr[k]
            mtm = (c[idx] / z["px"] - 1) * z["s"] - (FEE if cost else 0.0)
            val = cap * max(1 + lev * mtm, 0.0)
        eq[idx - a] = val
    return eq


def summary(d, tr, a, days, cost) -> dict:
    done = [z for z in tr if not z["open"]]
    allz = tr
    if not allz:
        return {"n": 0}
    net = np.array([z["net"] for z in allz])
    r = {}
    for lev in (1.0, 3.0):
        e = equity(d, tr, a, lev, cost)
        pk = np.maximum.accumulate(e)
        r[f"ret{int(lev)}"] = round((e[-1] - 1) * 100, 2)
        r[f"mdd{int(lev)}"] = round(float((e / pk - 1).min()) * 100, 2)
    return {
        "n": len(allz),
        "closed": len(done),
        "open": len(allz) - len(done),
        "long": sum(z["s"] > 0 for z in allz),
        "short": sum(z["s"] < 0 for z in allz),
        "win": round(float((net > 0).mean() * 100), 1),
        "avg": round(float(net.mean() * 100), 3),
        "best": round(float(net.max() * 100), 2),
        "worst": round(float(net.min() * 100), 2),
        "per_day": round(len(allz) / days, 2),
        "hold_h": round(float(np.mean([(z["t1"] - z["t0"]) / 3_600_000 for z in allz])), 1),
        **r,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2026-10-07")
    a_ = ap.parse_args()
    out = Path(a_.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(FUT5)
    d = bars(d5, 0)
    t = d["t"]
    ft, fr = load_funding()
    reg, newday = R.regime(d5, d, 0)
    sig, src, seam = signals(t)
    t0 = datetime.strptime(a_.start, "%Y-%m-%d").replace(tzinfo=KST).timestamp() * 1000
    a = int(np.searchsorted(t, t0))
    b = len(t) - 1
    days = (t[-1] + BAR - t[a]) / 86_400_000
    rep = {
        "start_kst": datetime.fromtimestamp(t[a] / 1000, KST).isoformat(),
        "end_kst": datetime.fromtimestamp((t[-1] + BAR) / 1000, KST).isoformat(),
        "days": round(days, 2),
        "sources": src,
        "seam": seam,
        "buy_hold_pct": round((d["c"][-1] / d["c"][a - 1] - 1) * 100, 2),
    }
    a_r1 = a + int(np.flatnonzero(newday[a:])[0])  # 첫 일봉 마감부터
    trades = {}
    for name in ("A", "B", "AS", "BS", "AS4", "K2", "R1"):
        if name == "R1":
            tr = r1_trades(d, reg, newday, a_r1, b)
        elif name == "K2":
            tr = k2_trades(d, reg, sig["AS4"], a, b)
        else:
            tr = ml_trades(d, sig[name], a, b)
        res = {}
        for cost in (True, False):
            finish(tr, ft, fr, cost)
            res["cost" if cost else "free"] = summary(d, tr, a, days, cost)
        finish(tr, ft, fr, True)
        rep[name] = res
        trades[name] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, float))
                    else int(v)
                    if isinstance(v, (np.integer,))
                    else v
                )
                for k, v in z.items()
            }
            for z in tr
        ]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "trades.json").write_text(json.dumps(trades, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
