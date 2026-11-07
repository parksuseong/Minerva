"""142번: 펀딩 캐리(현물 롱 + 무기한 숏). 등록: 2026-11-07-btc-funding-carry-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_funding_carry.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from btc_fractal import load5
from btc_scr_gate15 import load_funding

M5 = 300_000
DAY = 86_400_000
FUT_FEE = 0.0005
MAINT = 0.004


def ystart(y: int) -> float:
    return datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def month_key(t: float) -> int:
    d = datetime.fromtimestamp(t / 1000, timezone.utc)
    return d.year * 12 + d.month


class Px:
    """5분봉 시가·종가·고가를 시각으로 찾는다."""

    def __init__(self, d: dict):
        self.t, self.o, self.c, self.h = d["t"], d["o"], d["c"], d["h"]

    def at_open(self, t: float) -> float:
        i = int(np.searchsorted(self.t, t))
        return float(self.o[min(i, len(self.o) - 1)])

    def close_before(self, t: float) -> float:
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        return float(self.c[max(i, 0)])

    def max_high(self, t0: float, t1: float) -> float:
        a, b = int(np.searchsorted(self.t, t0)), int(np.searchsorted(self.t, t1))
        return float(self.h[a:b].max()) if b > a else -np.inf


def simulate(variant: str, ft, fr, spot: Px, fut: Px, spot_fee: float, spot_share: float, t_end: float) -> dict:
    """자본 1.0에서 시작. spot_share = 현물 비중(1배 0.5, 2배 0.67)."""
    cash = 1.0
    pos = None  # dict(q, s0, f0, margin, spot_val)
    eq_days, eq_vals = [], []
    switches = rebal = liq = 0
    hedged_ms = 0.0
    fund_got = []
    log = []
    last_month = None
    if variant == "C0":
        enter_thr, mean_in, exit_last, mean_out = -9.0, -9.0, -9.0, -9.0
    elif variant == "C1":
        enter_thr, mean_in, exit_last, mean_out = 0.0001, 0.00005, 0.0, 0.0
    else:
        enter_thr, mean_in, exit_last, mean_out = 0.0002, 0.00005, 0.0, 0.00005

    def equity(p, t):
        if p is None:
            return cash
        s, f = spot.close_before(t), fut.close_before(t)
        return cash + p["q"] * s + p["margin"] + p["q"] * (p["f0"] - f) + p["fund"]

    def open_pos(t, cap):
        nonlocal cash
        s, f = spot.at_open(t + M5), fut.at_open(t + M5)
        spot_cap = cap * spot_share
        margin = cap - spot_cap
        q = spot_cap / (s * (1 + spot_fee))
        fee_f = q * f * FUT_FEE
        cash -= cap
        return {"q": q, "s0": s, "f0": f, "margin": margin - fee_f, "fund": 0.0, "t0": t + M5}

    def close_pos(p, t):
        s, f = spot.at_open(t + M5), fut.at_open(t + M5)
        val = p["q"] * s * (1 - spot_fee) + p["margin"] + p["q"] * (p["f0"] - f) - p["q"] * f * FUT_FEE + p["fund"]
        return max(val, 0.0)

    day_t = float(np.floor(ft[0] / DAY) * DAY + DAY)
    k0 = 3
    for k in range(k0, len(ft)):
        t = float(ft[k])
        if t > t_end:
            break
        while day_t <= t:
            eq_days.append(day_t)
            eq_vals.append(equity(pos, day_t))
            day_t += DAY
        if pos is not None:
            # 이 정산 직전까지 강제청산 확인 (선물 고가)
            hi = fut.max_high(pos["tchk"], t)
            loss = pos["q"] * (hi - pos["f0"])
            if loss >= pos["margin"] + pos["fund"] - MAINT * pos["q"] * hi:
                # 강제청산: 선물 증거금(받은 펀딩 포함)을 모두 잃고, 현물은 그 시각에 판다
                liq += 1
                cash += pos["q"] * spot.close_before(t) * (1 - spot_fee)
                log.append({"t": t, "ev": "liq"})
                pos = None
            else:
                pos["tchk"] = t
        if pos is not None:
            f_now = fut.close_before(t)
            got = fr[k] * pos["q"] * f_now
            pos["fund"] += got
            fund_got.append(fr[k])
            hedged_ms += t - pos["tlast"]
            pos["tlast"] = t
        last = fr[k]
        mean3 = float(np.mean(fr[k - 2:k + 1]))
        if pos is None:
            if last >= enter_thr and mean3 >= mean_in:
                cap = cash
                pos = open_pos(t, cap)
                pos["tchk"] = pos["tlast"] = t + M5
                switches += 1
                log.append({"t": t, "ev": "in"})
        else:
            if variant != "C0" and (last < exit_last or mean3 < mean_out):
                cash += close_pos(pos, t)
                pos = None
                switches += 1
                log.append({"t": t, "ev": "out"})
            else:
                f_now = fut.close_before(t)
                unreal = pos["q"] * (f_now - pos["f0"])
                mk = month_key(t)
                if unreal > 0.6 * pos["margin"] or (last_month is not None and mk != last_month):
                    cap = close_pos(pos, t)
                    cash += cap
                    pos = open_pos(t, cash)
                    pos["tchk"] = pos["tlast"] = t + M5
                    rebal += 1
        last_month = month_key(t)
    while day_t <= t_end:
        eq_days.append(day_t)
        eq_vals.append(equity(pos, day_t))
        day_t += DAY
    return {"days": np.array(eq_days), "eq": np.array(eq_vals), "switches": switches, "rebal": rebal,
            "liq": liq, "hedged_ms": hedged_ms, "fund": np.array(fund_got), "log": log}


def stats(days, eq, t0, t1) -> dict:
    m = (days >= t0) & (days < t1)
    if m.sum() < 2:
        return {}
    e = eq[m]
    ret = e[-1] / e[0] - 1
    yrs = (days[m][-1] - days[m][0]) / (365.25 * DAY)
    cagr = (e[-1] / e[0]) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    peak = np.maximum.accumulate(e)
    mdd = float((e / peak - 1).min())
    return {"ret": round(ret * 100, 2), "cagr": round(cagr * 100, 2), "mdd": round(mdd * 100, 2),
            "ratio": round(cagr / abs(mdd), 2) if mdd < 0 else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.data)
    fut = Px(load5(base / "futures/klines_5m.csv"))
    spot = Px(load5(base / "5m.csv"))
    ft, fr = load_funding(base / "futures/funding.csv")
    t_end = min(fut.t[-1], spot.t[-1])
    keep = ft >= max(fut.t[0], spot.t[0]) + DAY
    ft, fr = ft[keep], fr[keep]
    rep: dict = {"funding": {"n": int(len(fr)), "pos_share": round(float((fr > 0).mean() * 100), 1),
                             "mean_pct": round(float(fr.mean() * 100), 4),
                             "sign_flips": int((np.sign(fr[1:]) != np.sign(fr[:-1])).sum())}}
    # 가장 나빴던 연속 음수 구간 (누적 합)
    worst, cur, cur_n, worst_n = 0.0, 0.0, 0, 0
    for r in fr:
        if r < 0:
            cur += r
            cur_n += 1
            if cur < worst:
                worst, worst_n = cur, cur_n
        else:
            cur, cur_n = 0.0, 0
    rep["funding"]["worst_negative_run_pct"] = round(worst * 100, 4)
    rep["funding"]["worst_negative_run_n"] = worst_n
    rep["funding"]["by_year_mean_pct"] = {y: round(float(fr[(ft >= ystart(y)) & (ft < ystart(y + 1))].mean() * 100), 4)
                                          for y in range(2020, 2027)}
    series = {}
    for v in ("C0", "C1", "C2"):
        for lev, share in (("1x", 0.5), ("2x", 0.67)):
            for fee_name, sf in (("vip0", 0.001), ("bnb", 0.00075)):
                r = simulate(v, ft, fr, spot, fut, sf, share, t_end)
                key = f"{v}|{lev}|{fee_name}"
                years = {y: stats(r["days"], r["eq"], ystart(y), ystart(y + 1)) for y in range(2020, 2027)}
                span = r["days"][-1] - r["days"][0]
                rep[key] = {
                    "all": stats(r["days"], r["eq"], 0, 1e15),
                    "2024~": stats(r["days"], r["eq"], ystart(2024), 1e15),
                    "years": years,
                    "switches": r["switches"], "rebalances": r["rebal"], "liquidations": r["liq"],
                    "hedged_pct": round(r["hedged_ms"] / span * 100, 1),
                    "avg_funding_pct": round(float(r["fund"].mean() * 100), 4) if len(r["fund"]) else None,
                }
                if lev == "1x" and fee_name == "vip0":
                    series[v] = (r["days"], r["eq"], r["log"])
                print(key, rep[key]["all"], rep[key]["2024~"], "sw", r["switches"], "liq", r["liq"], flush=True)
    # 판정 (1배, VIP0 기준)
    verdict = {}
    for v in ("C1", "C2"):
        x = rep[f"{v}|1x|vip0"]
        c1 = all((x["years"][y].get("ret", -1) >= 0) for y in range(2021, 2027))
        c2 = (x["2024~"].get("cagr", -1) >= 5)
        c3 = (x["all"].get("mdd", -100) > -5)
        verdict[v] = {"1_every_year_ge0": c1, "2_cagr24_ge5": c2, "3_mdd_gt_-5": c3, "pass": c1 and c2 and c3}
    rep["verdict"] = verdict
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    np.savez(out / "series.npz", ft=ft, fr=fr,
             **{f"{v}_days": s[0] for v, s in series.items()}, **{f"{v}_eq": s[1] for v, s in series.items()})
    (out / "logs.json").write_text(json.dumps({v: s[2] for v, s in series.items()}))
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
