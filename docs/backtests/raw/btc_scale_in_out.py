"""127번: 분할 진입·분할 청산 (R1, ML BS·AS4). 등록: 2026-11-07-btc-scale-in-out-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_scale_in_out.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import btc_ml_edge_cost as C
import numpy as np
from btc_fractal import load5, rsi
from btc_fut_rsi_swing import bd_signals
from btc_futures_compare import load_funding
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_scr_learned import bars
from btc_strategy_combo import RES, daily, stats, ts

DAY = 86_400_000
R1_FEE = 0.0005
R1_R, ML_R = 0.01, 0.015


def swing_arrays(h, l, c):  # noqa: E741
    """봉마다 그 봉 마감에 새로 확정된 스윙 저점·고점 가격과 HL·LH 여부."""
    n = len(c)
    lo_px, hi_px = np.full(n, np.nan), np.full(n, np.nan)
    hl, lh = np.zeros(n, bool), np.zeros(n, bool)
    prev_lo = prev_hi = None
    for i, s, px, _ in bd_signals(h, l, rsi(c, 7)):
        if s > 0:
            lo_px[i] = px
            hl[i] = prev_lo is not None and px > prev_lo
            prev_lo = px
        else:
            hi_px[i] = px
            lh[i] = prev_hi is not None and px < prev_hi
            prev_hi = px
    return lo_px, hi_px, hl, lh


class Cost:
    def __init__(self, ml: bool, mult: float):
        if ml:
            self.entry = self.limit = C.MAKER * mult
            self.taker, self.slip = C.TAKER * mult, C.SLIP * mult
            self.thru = C.THRU
        else:
            self.entry = self.limit = self.taker = R1_FEE * mult
            self.slip, self.thru = 0.0, 0.0


def run_group(A, e, s, px0, plan, cost, sw, reg=None):
    """한 묶음. 반환: (조각 목록, 끝 봉, 손절 여부) 또는 None.

    조각 = (진입 봉, 진입가, 비중, 청산 봉, 청산가, 진입 수수료율, 청산 수수료율).
    """
    o, h, l = A["o"], A["h"], A["l"]  # noqa: E741
    n = len(o)
    R_ = plan["R"]
    N = plan["N"]
    w0 = plan["adds"][0][1] if plan["adds"] else 1.0
    open_ = [[e, px0, w0 * N, cost.entry]]  # [진입 봉, 진입가, 비중, 진입 수수료율]
    done = []
    pend = [(px0 * (1 + s * k * R_), w * N) for k, w in plan["adds"][1:]]
    stop = px0 * (1 - s * R_)
    trimmed = False
    pb_pend = plan.get("pb", False)
    pb_at = None
    lo_px, hi_px, hl, lh = sw

    def close_all(b, px, fee):
        for eb, ep, w, ef in open_:
            done.append((eb, ep, w, b, px, ef, fee))
        open_.clear()

    def avg_entry():
        tw = sum(x[2] for x in open_)
        return sum(x[1] * x[2] for x in open_) / tw

    for b in range(e, n - 1):
        if pb_at == b:
            open_.append([b, o[b], 0.5 * N, cost.taker])
            pb_at = None
        first = b == e
        stop_hit = (l[b] <= stop) if s > 0 else (h[b] >= stop)
        # 추가 진입(역지정가). 같은 봉에 손절도 닿으면 추가가 먼저 체결된 것으로 본다(불리)
        still = []
        for p, w in pend:
            if (h[b] >= p) if s > 0 else (l[b] <= p):
                fill = p if first else (max(o[b], p) if s > 0 else min(o[b], p))
                open_.append([b, fill * (1 + s * cost.slip), w, cost.taker])
            else:
                still.append((p, w))
        added = len(still) < len(pend)
        pend = still
        if plan.get("be_after_adds") and added and not pend:
            be = avg_entry()
            stop = max(stop, be) if s > 0 else min(stop, be)
            stop_hit = stop_hit or ((l[b] <= stop) if s > 0 else (h[b] >= stop))
        if stop_hit:
            fill = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
            close_all(b, fill * (1 - s * cost.slip), cost.taker)
            return done, b, True
        if not first:
            if plan.get("tp") is not None:
                T = px0 * (1 + s * plan["tp"] * R_)
                if (h[b] >= T * (1 + cost.thru)) if s > 0 else (l[b] <= T * (1 - cost.thru)):
                    close_all(b, T, cost.limit)
                    return done, b, False
            if plan.get("trim") is not None and not trimmed:
                lvl, frac = plan["trim"]
                T = px0 * (1 + s * lvl * R_)
                if (h[b] >= T * (1 + cost.thru)) if s > 0 else (l[b] <= T * (1 - cost.thru)):
                    keep = []
                    for eb, ep, w, ef in open_:
                        done.append((eb, ep, w * frac, b, T, ef, cost.limit))
                        keep.append([eb, ep, w * (1 - frac), ef])
                    open_[:] = keep
                    trimmed = True
                    pend = []  # 덜어낸 뒤에는 더 사지 않는다
                    be = avg_entry()
                    stop = max(stop, be) if s > 0 else min(stop, be)
        if trimmed and plan.get("trail") and b > e:
            p = lo_px[b] if s > 0 else hi_px[b]
            if np.isfinite(p):
                stop = max(stop, p) if s > 0 else min(stop, p)
        if pb_pend and b > e and (hl[b] if s > 0 else lh[b]):
            pb_pend = False
            pb_at = b + 1
        if reg is not None and reg[b] != s:
            close_all(b, o[b + 1], cost.taker)
            return done, b, False
        if plan.get("max_bars") and b - e >= plan["max_bars"]:
            close_all(b, o[b + 1] * (1 - s * cost.slip), cost.taker)
            return done, b, False
    return None


def r1_groups(A, reg, newday, start, plan, cost, sw):
    n = len(A["o"])
    out, i, can = [], start, True
    while i < n - 2:
        if newday[i]:
            can = True
        side = int(reg[i])
        if side and can:
            res = run_group(A, i + 1, side, A["o"][i + 1], plan, cost, sw, reg)
            if res is None:
                break
            segs, end, stopped = res
            out.append((i + 1, side, segs, end))
            if stopped:
                can = False
            i = end
            continue
        i += 1
    return out


def ml_groups(A, li, si, plan, cost, sw):
    o, h, l, c = A["o"], A["h"], A["l"], A["c"]  # noqa: E741
    n = len(o)
    out, i = [], 0
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
        res = run_group(A, j, s, lim, plan, cost, sw)
        if res is None:
            break
        segs, end, _ = res
        out.append((j, s, segs, end))
        i = end
    return out


def path_returns(groups, A, lev, ft, fr):
    """봉별 계좌 수익률. 묶음 안에서는 진입 때 정한 명목으로 평가한다."""
    t, c, lo, hi = A["t"], A["c"], A["l"], A["h"]
    n = len(t)
    ret = np.zeros(n)
    pnl_g, liq = [], 0
    for e, s, segs, end in groups:
        span = np.arange(e, end + 1)
        V = np.ones(len(span))
        W = np.ones(len(span))
        for eb, ep, w, xb, xp, ef, xf in segs:
            a, b = eb - e, xb - e
            mark = np.r_[c[eb:xb], xp] / ep - 1
            worst = ((lo if s > 0 else hi)[eb:xb + 1] / ep - 1)
            m = (ft > t[eb]) & (ft <= t[xb])
            cf = np.zeros(b - a + 1)
            if m.any():
                k = np.searchsorted(t[eb:xb + 1], ft[m], side="left")
                np.add.at(cf, np.minimum(k, len(cf) - 1), fr[m])
            cf = np.cumsum(cf)
            seg = lev * w * (s * mark - s * cf)
            seg_w = lev * w * (s * worst - s * cf)
            V[a:b + 1] += seg
            W[a:b + 1] += seg_w
            V[b + 1:] += seg[-1]
            W[b + 1:] += seg[-1]
            # 수수료: 진입분은 진입 봉부터, 청산분은 청산 봉부터
            V[a:] -= lev * w * ef
            W[a:] -= lev * w * ef
            V[b:] -= lev * w * xf
            W[b:] -= lev * w * xf
        bad = np.flatnonzero(W <= 0)
        if bad.size:
            liq += 1
            V[bad[0]:] = 0.0
        prev = np.r_[1.0, V[:-1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret[span] = np.where(prev > 0, V / prev - 1, 0.0)
        pnl_g.append((e, s, float(V[-1] - 1) / lev))
    return ret, pnl_g, liq


PLANS = {
    "R1": {
        "base": {"R": R1_R, "N": 1.0, "adds": []},
        "SI": {"R": R1_R, "N": 1.2, "adds": [(0, 1 / 3), (0.5, 1 / 3), (1.0, 1 / 3)], "be_after_adds": True},
        "SO": {"R": R1_R, "N": 1.0, "adds": [], "trim": (1.0, 0.5)},
        "SIO": {"R": R1_R, "N": 1.2, "adds": [(0, 1 / 3), (0.5, 1 / 3), (1.0, 1 / 3)], "be_after_adds": True,
                "trim": (2.0, 0.5)},
        "PB": {"R": R1_R, "N": 1.0, "adds": [(0, 0.5)], "pb": True},
    },
    "ML": {
        "base": {"R": ML_R, "N": 1.0, "adds": [], "tp": 1.0},
        "SI": {"R": ML_R, "N": 1 / 1.5, "adds": [(0, 1 / 3), (0.5, 1 / 3), (1.0, 1 / 3)], "tp": 2.0},
        "SO": {"R": ML_R, "N": 1.0, "adds": [], "trim": (1.0, 0.5), "trail": True, "max_bars": 672},
        "SIO": {"R": ML_R, "N": 1 / 1.5, "adds": [(0, 1 / 3), (0.5, 1 / 3), (1.0, 1 / 3)], "trim": (2.0, 0.5),
                "trail": True, "max_bars": 672},
    },
}


def group_stats(pg, Rb, t0, t1, tt):
    x = [(s, p) for e, s, p in pg if t0 <= tt[e] < t1]
    if not x:
        return {"n": 0}
    p = np.array([v for _, v in x])
    sd = np.array([s for s, _ in x])
    f = lambda m: round(float(p[m].mean() * 100), 3) if m.any() else None  # noqa: E731
    return {"n": len(p), "win": round(float((p > 0).mean() * 100), 1), "avg_pct": f(np.ones(len(p), bool)),
            "avg_R": round(float(p.mean() / Rb), 3), "long_pct": f(sd > 0), "short_pct": f(sd < 0),
            "n_long": int((sd > 0).sum())}


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
    P["t"] = tp
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp)
    end = float(tp[-1])
    # R1은 선물 15분봉 격자, 예측 끝에서 자른다
    keep = d["t"] <= end
    A1 = {k: d[k][keep] for k in ("o", "h", "l", "c", "t")}
    d5k = {k: v[d5["t"] <= end + 900_000] for k, v in d5.items()}
    reg, newday = R.regime(d5k, A1, 0)
    start = int(np.searchsorted(A1["t"], A1["t"][0] + 60 * DAY))
    sw1 = swing_arrays(A1["h"], A1["l"], A1["c"])
    swm = swing_arrays(P["h"], P["l"], P["c"])
    li_b, si_b = rule_masks(pm, tp, 0.95, 5)
    li_a, si_a = rule_masks(pm, tp, 0.90, 4)
    masks = {"BS": (li_b, si_b), "AS4": (li_a & (tr4 > 0), si_a & (tr4 < 0))}
    periods = {"2022~23": (ts(2022), ts(2024)), "2024~": (ts(2024), end + 1)}
    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(),
                 "futures_missing_bars_2022~": int((~ok[tp >= ts(2022)]).sum())}
    series, groups_keep = {}, {}
    for rule in ("R1", "BS", "AS4"):
        fam = "R1" if rule == "R1" else "ML"
        A = A1 if rule == "R1" else P
        sw = sw1 if rule == "R1" else swm
        Rb = R1_R if rule == "R1" else ML_R
        for mult in (1.0, 2.0):
            cost = Cost(fam == "ML", mult)
            for var, plan in PLANS[fam].items():
                if rule == "R1":
                    g = r1_groups(A, reg, newday, start, plan, cost, sw)
                else:
                    g = ml_groups(A, *masks[rule], plan, cost, sw)
                key = f"{rule}-{var}" + ("" if mult == 1 else "-c2")
                row = {"groups": len(g)}
                for lev in (1.0, 3.0):
                    rr, pg, liq = path_returns(g, A, lev, ft, fr)
                    dt, dr = daily(rr, A["t"])
                    for pn, (p0, p1) in periods.items():
                        m = (dt >= p0) & (dt < p1)
                        row[f"{pn} {lev:g}x"] = stats(dr[m], dt[m])
                    row[f"liq {lev:g}x"] = liq
                    if lev == 1.0:
                        for pn, (p0, p1) in periods.items():
                            row[f"{pn} trades"] = group_stats(pg, Rb, p0, p1, A["t"])
                        if mult == 1.0:
                            series[key] = (dt, dr)
                            groups_keep[key] = g
                if fam == "R1" and var == "PB" and mult == 1.0:
                    risks = []
                    for e, s, segs, _ in g:
                        stp = A["o"][e] * (1 - s * R1_R)
                        risks.append(sum(w * abs(ep - stp) / ep for _, ep, w, *_ in segs[:1]) +
                                     sum(w * max(0.0, s * (ep - stp)) / ep for _, ep, w, *_ in segs[1:]))
                    row["pb_avg_risk_R"] = round(float(np.mean(risks)) / R1_R, 3) if risks else None
                rep[key] = row
    # 판정
    for rule in ("R1", "BS", "AS4"):
        fam = "R1" if rule == "R1" else "ML"
        for var in PLANS[fam]:
            if var == "base":
                continue
            v, b = rep[f"{rule}-{var}"], rep[f"{rule}-base"]
            v2, b2 = rep[f"{rule}-{var}-c2"], rep[f"{rule}-base-c2"]

            def better(x, y, pn):
                sx, sy = x[f"{pn} 1x"], y[f"{pn} 1x"]
                return bool(sx.get("cagr", -1) > 0 and (sx.get("ratio") or -9) > (sy.get("ratio") or -9))

            c1 = all(better(v, b, pn) for pn in periods)
            tr = v["2024~ trades"]
            c2 = bool((tr.get("long_pct") or 0) >= 0 and (tr.get("short_pct") or 0) >= 0)
            c3 = all(better(v2, b2, pn) for pn in periods)
            v["crit"] = {"1": c1, "2": c2, "3": c3}
            v["pass"] = c1 and c2 and c3
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    np.savez(out / "series.npz", **{f"{k}__dt": v[0] for k, v in series.items()},
             **{f"{k}__dr": v[1] for k, v in series.items()})
    (out / "groups.json").write_text(json.dumps(
        {k: [[int(e), int(s), [[int(x[0]), float(x[1]), float(x[2]), int(x[3]), float(x[4])] for x in segs], int(en)]
             for e, s, segs, en in g] for k, g in groups_keep.items()}))
    for k, v in rep.items():
        if isinstance(v, dict):
            print(k, {kk: vv for kk, vv in v.items() if kk in ("groups", "2022~23 1x", "2024~ 1x", "pass")},
                  flush=True)


if __name__ == "__main__":
    main()
