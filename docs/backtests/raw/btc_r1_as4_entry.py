"""128번: R1 일봉 방향 + AS4 타점 조합. 등록: 2026-11-07-btc-r1-as4-entry-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_r1_as4_entry.py --out <폴더> \
    [--null 200]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import btc_ml_edge_cost as C
import numpy as np
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_futures_reprice import futures_on
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_scale_in_out import path_returns
from btc_strategy_combo import RES, daily, stats, ts

DAY = 86_400_000
M5 = 300_000
ML_R, R1_R = 0.015, 0.01
WAIT = 96  # K3: 24시간


class Model:
    """체결 모형. delay: 신호 봉 마감 뒤 첫 5분봉 체결 금지. thru: 지정가 관통 폭."""

    def __init__(self, delay: bool, thru: float, mult: float = 1.0):
        self.delay, self.thru = delay, thru
        self.maker, self.taker, self.slip = C.MAKER * mult, C.TAKER * mult, C.SLIP * mult
        self.r1_fee = 0.0005 * mult


class Grid:
    def __init__(self, A: dict, f5: dict):
        self.A = A
        self.n = len(A["t"])
        j = np.searchsorted(f5["t"], A["t"])
        self.j5 = j
        self.o5, self.h5, self.l5, self.t5 = f5["o"], f5["h"], f5["l"], f5["t"]

    def sub_ok(self, b: int) -> bool:
        j = self.j5[b]
        return j + 2 < len(self.t5) and self.t5[j + 1] == self.A["t"][b] + M5 and \
            self.t5[j + 2] == self.A["t"][b] + 2 * M5

    def first_hl(self, b: int, model: Model) -> tuple:
        """진입 봉의 고가·저가. 지연이면 5~15분 구간만."""
        if model.delay and self.sub_ok(b):
            j = self.j5[b]
            return float(max(self.h5[j + 1], self.h5[j + 2])), float(min(self.l5[j + 1], self.l5[j + 2]))
        return float(self.A["h"][b]), float(self.A["l"][b])

    def market_px(self, b: int, model: Model) -> float:
        if model.delay and self.sub_ok(b):
            return float(self.o5[self.j5[b] + 1])
        return float(self.A["o"][b])


def lim_fill(G: Grid, i: int, s: int, model: Model):
    """신호 봉 i 종가 −0.1%(숏 +0.1%) 지정가, 2봉. 반환 (체결 봉, 지정가) 또는 None."""
    A = G.A
    lim = float(A["c"][i]) * (1 - C.OFF * s)
    for b in range(i + 1, min(i + 1 + C.REST, G.n)):
        if b == i + 1:
            hi, lo = G.first_hl(b, model)
        else:
            hi, lo = float(A["h"][b]), float(A["l"][b])
        if (lo <= lim * (1 - model.thru)) if s > 0 else (hi >= lim * (1 + model.thru)):
            return b, lim
    return None


def bracket(G: Grid, e: int, s: int, px: float, model: Model, delayed_first: bool):
    """AS4 청산: ±1.5%, 같은 봉이면 손절 먼저, 진입 봉에서는 익절 안 봄."""
    A = G.A
    o, h, l = A["o"], A["h"], A["l"]  # noqa: E741
    stop, tp = px * (1 - s * ML_R), px * (1 + s * ML_R)
    for b in range(e, G.n - 1):
        first = b == e
        hi, lo = G.first_hl(b, model) if (first and delayed_first) else (h[b], l[b])
        if (lo <= stop) if s > 0 else (hi >= stop):
            f = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
            return b, f * (1 - s * model.slip), model.taker + 0.0, True
        if not first and ((hi >= tp * (1 + model.thru)) if s > 0 else (lo <= tp * (1 - model.thru))):
            return b, tp, model.maker, False
    return None


def r1_manage(G: Grid, e: int, s: int, px: float, reg, model: Model, delayed_first: bool, fee_exit: float,
              slip: float):
    """R1 청산: 진입가 대비 1% 손절, 일봉 방향이 바뀌면 다음 시가."""
    A = G.A
    o, h, l = A["o"], A["h"], A["l"]  # noqa: E741
    stop = px * (1 - s * R1_R)
    for b in range(e, G.n - 1):
        first = b == e
        hi, lo = G.first_hl(b, model) if (first and delayed_first) else (h[b], l[b])
        if (lo <= stop) if s > 0 else (hi >= stop):
            f = stop if first else (min(o[b], stop) if s > 0 else max(o[b], stop))
            return b, f * (1 - s * slip), fee_exit, True
        if reg[b] != s:
            return b, o[b + 1] * (1 - s * slip), fee_exit, False
    return None


def grp(e, s, px, ef, x):
    b, xp, xf, stopped = x
    return (e, s, [(e, px, 1.0, b, float(xp), ef, xf)], b), stopped


def run_r1(G, reg, newday, i0, model):
    out, i, can = [], i0, True
    while i < G.n - 2:
        if newday[i]:
            can = True
        s = int(reg[i])
        if s and can:
            e = i + 1
            px = G.market_px(e, model)
            x = r1_manage(G, e, s, px, reg, model, model.delay, model.r1_fee, 0.0)
            if x is None:
                break
            g, stopped = grp(e, s, px, model.r1_fee, x)
            out.append(g)
            if stopped:
                can = False
            i = x[0]
            continue
        i += 1
    return out


def run_ml(G, Lm, Sm, i0, model):
    out, i = [], i0
    while i < G.n - 2:
        s = 1 if Lm[i] else (-1 if Sm[i] else 0)
        if not s:
            i += 1
            continue
        f = lim_fill(G, i, s, model)
        if f is None:
            i += 1
            continue
        j, px = f
        x = bracket(G, j, s, px, model, model.delay and j == i + 1)
        if x is None:
            break
        g, _ = grp(j, s, px, model.maker, x)
        out.append(g)
        i = x[0]
    return out


def run_k(G, reg, Lm, Sm, i0, model, fallback: bool):
    out, i, wait0 = [], i0, None
    exit_fee = model.taker
    while i < G.n - 2:
        d = int(reg[i])
        if d == 0:
            wait0 = None
            i += 1
            continue
        if wait0 is None or reg[i - 1] != d:
            wait0 = i
        e = px = ef = None
        delayed_first = False
        if Lm[i] if d > 0 else Sm[i]:
            f = lim_fill(G, i, d, model)
            if f is not None:
                e, px = f
                ef = model.maker
                delayed_first = model.delay and e == i + 1
        if e is None and fallback and i - wait0 >= WAIT:
            e = i + 1
            px = G.market_px(e, model) * (1 + d * model.slip)
            ef = model.taker
            delayed_first = model.delay
        if e is None:
            i += 1
            continue
        x = r1_manage(G, e, d, px, reg, model, delayed_first, exit_fee, model.slip)
        if x is None:
            break
        g, stopped = grp(e, d, px, ef, x)
        out.append(g)
        i = x[0]
        wait0 = i if stopped else None
    return out


def net_mean(groups, t, t0) -> float:
    v = [s * (segs[0][4] / segs[0][1] - 1) - segs[0][5] - segs[0][6] for e, s, segs, _ in groups if t[e] >= t0]
    return float(np.mean(v)) if v else 0.0


def evaluate(groups, G, ft, fr, periods, levs=(1.0, 3.0)) -> dict:
    A = G.A
    row = {"groups": len(groups)}
    for lev in levs:
        rr, pg, liq = path_returns(groups, A, lev, ft, fr)
        dt, dr = daily(rr, A["t"])
        for pn, (p0, p1) in periods.items():
            m = (dt >= p0) & (dt < p1)
            row[f"{pn} {lev:g}x"] = stats(dr[m], dt[m])
        row[f"liq {lev:g}x"] = liq
        if lev == 1.0:
            row["_series"] = (dt, dr)
            for pn, (p0, p1) in periods.items():
                x = [(s, p) for e, s, p in pg if p0 <= A["t"][e] < p1]
                if not x:
                    row[f"{pn} trades"] = {"n": 0}
                    continue
                p = np.array([q for _, q in x])
                sd = np.array([s for s, _ in x])
                row[f"{pn} trades"] = {
                    "n": len(p), "win": round(float((p > 0).mean() * 100), 1),
                    "avg_pct": round(float(p.mean() * 100), 3),
                    "long_pct": round(float(p[sd > 0].mean() * 100), 3) if (sd > 0).any() else None,
                    "short_pct": round(float(p[sd < 0].mean() * 100), 3) if (sd < 0).any() else None,
                    "n_long": int((sd > 0).sum())}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--null", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f5 = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    ft, fr = load_funding()
    z = np.load(RES / "ml15/pred.npz")
    tp = z["t"]
    spot = {k: z[k] for k in "ohlc"}
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    pm = np.mean(ps, 0)
    P, ok = futures_on(tp, spot)
    P["t"] = tp
    end = float(tp[-1])
    f5k = {k: v[f5["t"] <= end + 900_000] for k, v in f5.items()}
    reg, newday = R.regime(f5k, P, 0)
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp)
    G = Grid(P, f5k)
    i0 = int(np.searchsorted(tp, ts(2023)))
    periods = {"2023": (ts(2023), ts(2024)), "2024~": (ts(2024), end + 1), "1y": (end - 365 * DAY, end + 1)}

    def as4(probs):
        li, si = rule_masks(probs, tp, 0.90, 4)
        return li & (tr4 > 0), si & (tr4 < 0)

    seeds = {"mean": as4(pm), **{f"s{k}": as4(ps[k]) for k in range(5)}}
    models = {"a": Model(False, C.THRU), "b": Model(True, C.THRU), "c": Model(False, 0.0005),
              "bc": Model(True, 0.0005), "bc_c15": Model(True, 0.0005, 1.5)}

    def run(var, model, masks):
        Lm, Sm = masks
        if var == "R1":
            return run_r1(G, reg, newday, i0, model)
        if var == "AS4":
            return run_ml(G, Lm, Sm, i0, model)
        if var == "K1":
            return run_ml(G, Lm & (reg > 0), Sm & (reg < 0), i0, model)
        return run_k(G, reg, Lm, Sm, i0, model, var == "K3")

    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(),
                 "futures_missing_bars_2023~": int((~ok[tp >= ts(2023)]).sum()),
                 "sub5m_missing_2023~": int(sum(1 for b in range(i0, G.n) if not G.sub_ok(b)))}
    series, keep = {}, {}
    VARS = ("R1", "AS4", "K1", "K2", "K3")
    for mk, model in models.items():
        for var in VARS:
            sd_list = ["mean"] if mk != "bc" else list(seeds)
            if var == "R1":
                sd_list = ["mean"]
            for sk in sd_list:
                g = run(var, model, seeds[sk])
                levs = (1.0, 3.0) if (sk == "mean" and mk in ("a", "bc")) else (1.0,)
                row = evaluate(g, G, ft, fr, periods, levs)
                s_ = row.pop("_series")
                key = f"{var}|{mk}|{sk}"
                rep[key] = row
                if sk == "mean" and mk in ("a", "bc"):
                    series[key] = s_
                    keep[key] = g
                print(key, row["groups"], {p: row[f"{p} 1x"].get("cagr") for p in periods},
                      {p: row[f"{p} 1x"].get("mdd") for p in periods}, flush=True)
    # 무작위 대조: AS4 신호 시각만 원형 이동, 모형 bc, 씨드 평균
    Lm, Sm = seeds["mean"]
    rng = np.random.default_rng(7)
    # 레드팀 128: 신호가 있는 2023~ 구간 안에서만 원형으로 민다(전체 배열을 밀면 신호가 시험 밖으로 빠진다)
    i0 = int(np.searchsorted(G.t, ts(2023)))
    span = G.n - i0
    shifts = rng.integers(96 * 7, span - 96 * 7, size=a.null)
    for var in ("K1", "K2", "K3"):
        act = net_mean(keep[f"{var}|bc|mean"], tp, ts(2023))
        null = []
        for k in shifts:
            Lr, Sr = Lm.copy(), Sm.copy()
            Lr[i0:], Sr[i0:] = np.roll(Lm[i0:], int(k)), np.roll(Sm[i0:], int(k))
            g = run(var, models["bc"], (Lr, Sr))
            null.append(net_mean(g, tp, ts(2023)))
        null = np.array(null)
        rep[f"null|{var}"] = {"actual_net_pct": round(act * 100, 4), "null_mean_pct": round(float(null.mean()) * 100, 4),
                              "p": round(float((null >= act).mean()), 4)}
        print("null", var, rep[f"null|{var}"], flush=True)
    # 판정
    for var in ("K1", "K2", "K3"):
        v = rep[f"{var}|bc|mean"]
        c1 = all((v[f"{p} 1x"].get("cagr") or -1) > 0 for p in ("2023", "2024~"))
        c2 = all((v[f"{p} 1x"].get("ratio") or -9) >= max(rep["R1|bc|mean"][f"{p} 1x"].get("ratio") or -9,
                                                          rep["AS4|bc|mean"][f"{p} 1x"].get("ratio") or -9)
                 for p in ("2023", "2024~"))
        tr = v["2024~ trades"]
        c3 = (tr.get("long_pct") or 0) >= 0 and (tr.get("short_pct") or 0) >= 0
        pos = sum(1 for k in range(5) if (rep[f"{var}|bc|s{k}"]["2024~ 1x"].get("cagr") or -1) > 0 and
                  (rep[f"{var}|bc|s{k}"]["1y 1x"].get("cagr") or -1) > 0)
        c4 = pos >= 3
        cc = rep[f"{var}|bc_c15|mean"]
        c5 = all((cc[f"{p} 1x"].get("cagr") or -1) > 0 for p in ("2023", "2024~"))
        c6 = rep[f"null|{var}"]["p"] < 0.0167
        rep[f"crit|{var}"] = {"1": c1, "2": c2, "3": bool(c3), "4": c4, "seeds_pos": pos, "5": c5, "6": bool(c6),
                              "pass": bool(c1 and c2 and c3 and c4 and c5 and c6)}
        print("crit", var, rep[f"crit|{var}"], flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    np.savez(out / "series.npz", **{f"{k}__dt": v[0] for k, v in series.items()},
             **{f"{k}__dr": v[1] for k, v in series.items()})
    (out / "groups.json").write_text(json.dumps(
        {k: [[int(e), int(s), float(segs[0][1]), int(segs[0][3]), float(segs[0][4])] for e, s, segs, _ in g]
         for k, g in keep.items()}))


if __name__ == "__main__":
    main()
