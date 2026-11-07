"""140번: ETH↔BTC 선행(L1)·상대강도 문(L2, AS4)·괴리(L3). 등록: 2026-11-07-btc-eth-leadlag-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_eth_leadlag.py --out <폴더> \
    [--null 1000]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample
from btc_futures_compare import load_funding
from btc_ml_bracket_grid import RULES, run_ml
from btc_ml_futures_reprice import futures_on
from btc_ml_regime_gate import on_15m
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, evaluate
from btc_strategy_combo import RES, ts

M5 = 300_000
H1 = 3_600_000
BAR = 900_000
TP = SL = 0.015
MAXH = 12 * H1
DELAY = 5 * 60_000
COST = 0.0005
W = 12 * 12 + 2  # 12시간 = 5분봉 144개


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    a = 2 / (n + 1)
    m = np.nan
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        m = v if not np.isfinite(m) else a * v + (1 - a) * m
        out[i] = m
    return out


def align(a: dict, b: dict) -> tuple:
    common, ia, ib = np.intersect1d(a["t"], b["t"], return_indices=True)
    return {k: v[ia] for k, v in a.items()}, {k: v[ib] for k, v in b.items()}


def zret(c: np.ndarray, n: int = 720) -> np.ndarray:
    r = np.r_[np.nan, np.diff(np.log(c))]
    z = np.full(len(r), np.nan)
    cs = np.nancumsum(np.r_[0.0, np.nan_to_num(r)])
    cs2 = np.nancumsum(np.r_[0.0, np.nan_to_num(r) ** 2])
    for i in range(n + 1, len(r)):
        m = (cs[i] - cs[i - n]) / n  # 직전 n개(현재 봉 제외): r[i-n..i-1]
        v = (cs2[i] - cs2[i - n]) / n - m * m
        if v > 0:
            z[i] = (r[i] - m) / np.sqrt(v)
    return z


def outcomes(f5: dict, entry_t: np.ndarray, side: int, ft, fr, cost: float) -> tuple:
    """각 결정 시각에 side로 들어갔을 때 (순손익, 청산 시각). 진입은 결정+5분 뒤 첫 5분봉 시가."""
    t5, o, h, lo = f5["t"], f5["o"], f5["h"], f5["l"]
    n5 = len(t5)
    j = np.searchsorted(t5, entry_t + DELAY)
    ok = j + W < n5
    net = np.full(len(entry_t), np.nan)
    xt = np.full(len(entry_t), np.nan)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return net, xt
    jj = j[idx]
    px = o[jj]
    off = np.arange(W)
    M = jj[:, None] + off[None, :]
    HH, LL, OO, TT = h[M], lo[M], o[M], t5[M]
    tgt = px * (1 + TP * side)
    stp = px * (1 - SL * side)
    if side > 0:
        hit_s = LL <= stp[:, None]
        hit_t = HH >= tgt[:, None]
    else:
        hit_s = HH >= stp[:, None]
        hit_t = LL <= tgt[:, None]
    tend = TT >= (TT[:, :1] + MAXH)
    big = W + 5
    fs = np.where(hit_s.any(1), hit_s.argmax(1), big)
    ftg = np.where(hit_t.any(1), hit_t.argmax(1), big)
    fe = np.where(tend.any(1), tend.argmax(1), big)
    first = np.minimum(np.minimum(fs, ftg), fe)
    rows = np.arange(len(idx))
    g = np.full(len(idx), np.nan)
    stop_first = (fs <= ftg) & (fs <= fe)  # 같은 봉이면 손절 먼저
    tgt_first = (~stop_first) & (ftg <= fe)
    time_first = ~(stop_first | tgt_first)
    so = OO[rows, np.minimum(fs, W - 1)]
    if side > 0:
        fill_s = np.minimum(so, stp)
    else:
        fill_s = np.maximum(so, stp)
    g[stop_first] = (fill_s[stop_first] / px[stop_first] - 1) * side
    g[tgt_first] = TP
    eo = OO[rows, np.minimum(fe, W - 1)]
    g[time_first] = (eo[time_first] / px[time_first] - 1) * side
    ex_t = TT[rows, np.minimum(first, W - 1)]
    t_in = t5[jj]
    # 펀딩: (진입, 청산] 사이 정산
    cf = np.r_[0.0, np.cumsum(fr)]
    a = np.searchsorted(ft, t_in, side="right")
    b = np.searchsorted(ft, ex_t, side="right")
    fund = cf[b] - cf[a]
    net[idx] = g - side * fund - 2 * cost
    xt[idx] = ex_t
    return net, xt


def walk(
    on_l: np.ndarray, on_s: np.ndarray, dec_t: np.ndarray, outL, outS, i0: int, i1: int
) -> list:
    """포지션 하나. (결정 index, 방향, 순손익)."""
    nl, xl = outL
    ns, xs = outS
    tr, busy = [], -np.inf
    for i in range(i0, i1):
        if dec_t[i] < busy:
            continue
        s = 1 if on_l[i] else (-1 if on_s[i] else 0)
        if not s:
            continue
        n, x = (nl[i], xl[i]) if s > 0 else (ns[i], xs[i])
        if not np.isfinite(n):
            continue
        tr.append((i, s, float(n)))
        busy = x
    return tr


def summ(tr: list) -> dict:
    if not tr:
        return {"n": 0}
    g = np.array([x[2] for x in tr])
    lo = [x[2] for x in tr if x[1] > 0]
    sh = [x[2] for x in tr if x[1] < 0]
    eq = np.cumprod(1 + g)
    pk = np.maximum.accumulate(np.r_[1.0, eq])
    eq3 = np.cumprod(1 + np.maximum(3 * g, -1))
    pk3 = np.maximum.accumulate(np.r_[1.0, eq3])
    return {
        "n": len(tr),
        "win": round(float((g > 0).mean() * 100), 1),
        "avg_pct": round(float(g.mean() * 100), 4),
        "long_n": len(lo),
        "long_pct": round(float(np.mean(lo) * 100), 4) if lo else None,
        "short_n": len(sh),
        "short_pct": round(float(np.mean(sh) * 100), 4) if sh else None,
        "total1x_pct": round(float((eq[-1] - 1) * 100), 2),
        "mdd1x_pct": round(float((eq / pk[1:] - 1).min() * 100), 2),
        "total3x_pct": round(float((eq3[-1] - 1) * 100), 2),
        "mdd3x_pct": round(float((eq3 / pk3[1:] - 1).min() * 100), 2),
    }


def ratio(row, p):
    x = row.get(f"{p} 1x", {}).get("ratio")
    return x if x is not None else -9.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--null", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fb = load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser())
    fe = load5(Path("~/.minerva/data/eth/futures/klines_5m.csv").expanduser())
    ft, fr = load_funding()
    bh, eh = align(resample(fb, 12), resample(fe, 12))
    dec = bh["t"] + H1  # 1시간봉 마감 = 결정 시각
    n = len(dec)
    zb, ze = zret(bh["c"]), zret(eh["c"])
    with np.errstate(invalid="ignore"):
        l1_l = (ze >= 2) & (zb < 0.5)
        l1_s = (ze <= -2) & (zb > -0.5)
    # L3: 24시간 신고가·신저가 괴리
    l3_l, l3_s = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(24, n):
        bhi = bh["h"][i - 24 : i].max()
        blo = bh["l"][i - 24 : i].min()
        ehi = eh["h"][i - 23 : i + 1].max()
        elo = eh["l"][i - 23 : i + 1].min()
        if bh["h"][i] >= bhi and eh["c"][i] <= ehi * 0.99:
            l3_s[i] = True
        if bh["l"][i] <= blo and eh["c"][i] >= elo * 1.01:
            l3_l[i] = True
    rep: dict = {
        "data_end_utc": datetime.fromtimestamp(dec[-1] / 1000, timezone.utc).isoformat(),
        "hours": int(n),
    }
    hi0, hi1 = int(np.searchsorted(dec, ts(2024))), n
    di0, di1 = int(np.searchsorted(dec, ts(2020) + 31 * 86_400_000)), hi0
    res_trades = {}
    oL = outcomes(fb, dec, 1, ft, fr, COST)
    oS = outcomes(fb, dec, -1, ft, fr, COST)
    oL15 = (oL[0] - COST, oL[1])  # 비용 1.5배: 왕복 0.10% → 0.15%
    oS15 = (oS[0] - COST, oS[1])
    print("outcomes", round(time.time() - t0), flush=True)
    for name, (Lf, Sf) in {"L1": (l1_l, l1_s), "L3": (l3_l, l3_s)}.items():
        d_tr = walk(Lf, Sf, dec, oL, oS, di0, di1)
        h_tr = walk(Lf, Sf, dec, oL, oS, hi0, hi1)
        h15 = walk(Lf, Sf, dec, oL15, oS15, hi0, hi1)
        row = {
            "disc": summ(d_tr),
            "hold": summ(h_tr),
            "hold_c15": summ(h15),
            "signals_hold": {"long": int(Lf[hi0:].sum()), "short": int(Sf[hi0:].sum())},
        }
        actual = row["hold"].get("avg_pct", -99)
        rng = np.random.default_rng(7)
        span = hi1 - hi0
        null = []
        for k in rng.integers(24 * 7, span - 24 * 7, size=a.null):
            Lr, Sr = Lf.copy(), Sf.copy()
            Lr[hi0:], Sr[hi0:] = np.roll(Lf[hi0:], int(k)), np.roll(Sf[hi0:], int(k))
            s_ = summ(walk(Lr, Sr, dec, oL, oS, hi0, hi1))
            null.append(s_.get("avg_pct", 0.0))
        null = np.array(null)
        row["null_mean_pct"] = round(float(null.mean()), 4)
        row["p"] = round(float((null >= actual).mean()), 4)
        rep[name] = row
        res_trades[name] = {"hold": [[int(i), s, round(g, 6)] for i, s, g in h_tr]}
        print(name, row["disc"], row["hold"], row["p"], round(time.time() - t0), flush=True)
    # L2: ETH/BTC 비율 문 + AS4
    bd, ed = align(resample(fb, 288), resample(fe, 288))
    rt = ed["c"] / bd["c"]
    em = ema(rt, 20)
    sl5 = np.r_[np.full(5, np.nan), rt[5:] - rt[:-5]]
    with np.errstate(invalid="ignore"):
        up_d = (rt > em) & (sl5 > 0)
        dn_d = (rt < em) & (sl5 < 0)
    dclose = bd["t"] + 288 * M5
    z = np.load(RES / "ml15/pred.npz")
    tp_t = z["t"]
    spot = {k: z[k] for k in "ohlc"}
    ps = [z["probs"]] + [np.load(RES / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    pm = np.mean(ps, 0)
    P, _ = futures_on(tp_t, spot)
    P["t"] = tp_t
    end = float(tp_t[-1])
    f5k = {k: v[fb["t"] <= end + BAR] for k, v in fb.items()}
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), tp_t)
    G = Grid(P, f5k)
    i0 = int(np.searchsorted(tp_t, ts(2023)))
    up = on_15m(dclose, up_d.astype(float), tp_t) > 0.5
    dn = on_15m(dclose, dn_d.astype(float), tp_t) > 0.5
    per2 = {"2023": (ts(2023), ts(2024)), "2024~": (ts(2024), end + 1)}

    def masks(probs):
        q, k, _, _ = RULES["AS4"]
        li, si = rule_masks(probs, tp_t, q, k)
        return li & (tr4 > 0), si & (tr4 < 0)

    model, model15 = Model(True, 0.0005), Model(True, 0.0005, 1.5)
    Lm, Sm = masks(pm)
    base = evaluate(run_ml(G, Lm, Sm, i0, model, 0.015, "1.5", None), G, ft, fr, per2, (1.0, 3.0))
    base.pop("_series")
    gated = evaluate(
        run_ml(G, Lm & up, Sm & dn, i0, model, 0.015, "1.5", None), G, ft, fr, per2, (1.0, 3.0)
    )
    gated.pop("_series")
    g15 = evaluate(
        run_ml(G, Lm & up, Sm & dn, i0, model15, 0.015, "1.5", None), G, ft, fr, per2, (1.0,)
    )
    g15.pop("_series")
    seeds = []
    for k in range(5):
        Ls, Ss = masks(ps[k])
        rb = evaluate(
            run_ml(G, Ls, Ss, i0, model, 0.015, "1.5", None),
            G,
            ft,
            fr,
            {"2024~": per2["2024~"]},
            (1.0,),
        )
        rg = evaluate(
            run_ml(G, Ls & up, Ss & dn, i0, model, 0.015, "1.5", None),
            G,
            ft,
            fr,
            {"2024~": per2["2024~"]},
            (1.0,),
        )
        seeds.append((ratio(rb, "2024~"), ratio(rg, "2024~")))
    span = G.n - i0
    rng = np.random.default_rng(7)
    actual = ratio(gated, "2024~")
    null = []
    for k in rng.integers(96 * 7, span - 96 * 7, size=a.null):
        ur, dr = up.copy(), dn.copy()
        ur[i0:], dr[i0:] = np.roll(up[i0:], int(k)), np.roll(dn[i0:], int(k))
        rr = evaluate(
            run_ml(G, Lm & ur, Sm & dr, i0, model, 0.015, "1.5", None),
            G,
            ft,
            fr,
            {"2024~": per2["2024~"]},
            (1.0,),
        )
        null.append(ratio(rr, "2024~"))
    null = np.array(null)
    rep["L2"] = {
        "ungated": base,
        "gated": gated,
        "gated_c15": {p: g15[f"{p} 1x"] for p in per2},
        "seed_ratio_2024_base_vs_gated": seeds,
        "null_mean_ratio": round(float(null.mean()), 3),
        "p": round(float((null >= actual).mean()), 4),
        "gate_on_2023~": {
            "up": round(float(up[i0:].mean() * 100), 1),
            "down": round(float(dn[i0:].mean() * 100), 1),
        },
    }
    print("L2", rep["L2"]["p"], round(time.time() - t0), flush=True)
    # 판정
    ps_ = sorted([(rep[k]["p"], k) for k in ("L1", "L2", "L3")])
    holm = {k: p < 0.05 / (3 - r) for r, (p, k) in enumerate(ps_)}
    # Holm: 앞 순위가 떨어지면 뒤도 떨어진다
    stop = False
    for _, k in ps_:
        if stop or not holm[k]:
            stop = True
            holm[k] = False
    crit = {}
    for k in ("L1", "L3"):
        h = rep[k]["hold"]
        c1 = h.get("n", 0) >= 30 and (h.get("avg_pct") or -1) > 0
        c2 = (h.get("long_pct") or 0) >= 0 and (h.get("short_pct") or 0) >= 0
        c3 = (rep[k]["hold_c15"].get("avg_pct") or -1) > 0
        crit[k] = {
            "1": c1,
            "2": c2,
            "3": c3,
            "4_holm": holm[k],
            "pass": bool(c1 and c2 and c3 and holm[k]),
        }
    g = rep["L2"]["gated"]
    tr24 = g.get("2024~ trades") or {}
    c1 = (tr24.get("n") or 0) >= 30 and (tr24.get("avg_pct") or -1) > 0
    c2 = (tr24.get("long_pct") or 0) >= 0 and (tr24.get("short_pct") or 0) >= 0
    c3 = (rep["L2"]["gated_c15"]["2024~"].get("total") or -1) > 0
    c5 = all(ratio(g, p) > ratio(base, p) for p in per2) and sum(y > x for x, y in seeds) >= 3
    crit["L2"] = {
        "1": c1,
        "2": c2,
        "3": c3,
        "4_holm": holm["L2"],
        "5": c5,
        "pass": bool(c1 and c2 and c3 and holm["L2"] and c5),
    }
    rep["crit"] = crit
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (out / "trades.json").write_text(json.dumps(res_trades))
    np.savez(
        out / "series.npz",
        dec=dec,
        l1_l=l1_l,
        l1_s=l1_s,
        l3_l=l3_l,
        l3_s=l3_s,
        dclose=dclose,
        ratio=rt,
        ema=em,
        up_d=up_d,
        dn_d=dn_d,
    )
    print(json.dumps(crit), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
