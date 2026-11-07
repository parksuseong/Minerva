"""141번: 4시간 RSI 기울기 + 4시간 매수 비율(RT2) 현물 2017~2019 표본 밖 확인.

등록: docs/backtests/2026-11-07-btc-rsi-taker-holdout-hypotheses.md
실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_rsi_taker_holdout.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import btc_indicator_library as L
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sw

M5, B15, DAY = L.M5, L.B15, L.DAY
FIX = {"rsi_hi": 9.486742973327637, "rsi_lo": -9.663559913635254,
       "tak_hi": 0.5193184614181519, "tak_lo": 0.47848936915397644}  # 139번 2020-03~2022 선물 q90/q10
ROLL = 2190  # 4시간봉 약 1년


def outcomes(f5: dict, t15: np.ndarray, fund) -> tuple:
    """139번 outcomes와 같은 체결(신호 마감+5분 5분봉 시가, ±1.5%, 같은 봉 손절, 최대 2일)."""
    t5, o5, h5, l5, c5 = f5["t"], f5["o"], f5["h"], f5["l"], f5["c"]
    if fund is not None:
        ft, fr = fund
        cf = np.r_[0.0, np.cumsum(fr)]
    n = len(t15)
    res = {s: (np.full(n, np.nan), np.full(n, -1, np.int64)) for s in (1, -1)}
    ent = np.searchsorted(t5, t15 + B15 + M5)
    okt = (ent < len(t5)) & (t5[np.minimum(ent, len(t5) - 1)] == t15 + B15 + M5)
    R, H = L.R, L.HOLD5
    for i in np.flatnonzero(okt):
        e = ent[i]
        if e + 2 >= len(t5):
            continue
        px = o5[e]
        hi, lo = h5[e:e + H], l5[e:e + H]
        for s in (1, -1):
            tp, sl = px * (1 + R * s), px * (1 - R * s)
            a, b = (hi >= tp, lo <= sl) if s > 0 else (lo <= tp, hi >= sl)
            ia = a.argmax() if a.any() else 10**6
            ib = b.argmax() if b.any() else 10**6
            if ib <= ia and ib < 10**6:
                k = ib
                fill = min(o5[e + k], sl) if s > 0 else max(o5[e + k], sl)
                if k == 0:
                    fill = sl
            elif ia < 10**6:
                k, fill = ia, tp
            else:
                k = len(hi) - 1
                fill = c5[e + k]
            x = e + k
            g = (fill / px - 1) * s
            if fund is not None:
                fa = np.searchsorted(ft, t5[e], side="right")
                fb = np.searchsorted(ft, t5[x] + M5, side="right")
                g -= s * (cf[fb] - cf[fa])
            res[s][0][i] = g - 2 * L.FEE
            res[s][1][i] = x
    return res[1][0], res[1][1], res[-1][0], res[-1][1], t5


def roll_q(x: np.ndarray, q: float) -> np.ndarray:
    """각 봉보다 앞선 ROLL개 값의 분위수(그 봉은 빼고)."""
    out = np.full(len(x), np.nan)
    if len(x) <= ROLL:
        return out
    w = sw(x[:-1], ROLL)  # w[k] = x[k : k+ROLL] → 봉 k+ROLL에 쓴다
    with np.errstate(all="ignore"):
        for s0 in range(0, len(w), 1000):
            out[ROLL + s0:ROLL + s0 + len(w[s0:s0 + 1000])] = np.nanquantile(w[s0:s0 + 1000], q, axis=1)
    return out


def signals(f5: dict) -> dict:
    d15 = L.resample(f5, 3)
    t15 = d15["t"]
    d4 = L.resample(f5, 48)
    names, cols = L.frame_features(d4)
    rsl, tak = cols[names.index("rsi_sl")], cols[names.index("taker")]
    cts = d4["t"] + 48 * M5
    j = np.searchsorted(cts, t15 + B15, side="right") - 1
    ok = j >= 0
    jj = np.maximum(j, 0)

    def a15(x):
        y = np.full(len(t15), np.nan)
        y[ok] = x[jj[ok]]
        return y

    R, T = a15(rsl), a15(tak)
    thr = {"F": {k: np.full(len(t15), v) for k, v in FIX.items()},
           "R": {"rsi_hi": a15(roll_q(rsl, 0.9)), "rsi_lo": a15(roll_q(rsl, 0.1)),
                 "tak_hi": a15(roll_q(tak, 0.9)), "tak_lo": a15(roll_q(tak, 0.1))}}
    out = {"t15": t15, "d15": d15, "taker15": T, "rsl15": R}
    for v, th in thr.items():
        with np.errstate(invalid="ignore"):
            ml = (R >= th["rsi_hi"]) & (T >= th["tak_hi"])
            ms = (R <= th["rsi_lo"]) & (T <= th["tak_lo"])
        out[v] = (ml & ~np.r_[False, ml[:-1]], ms & ~np.r_[False, ms[:-1]])
        out[v + "_on"] = (float(np.nanmean(ml)), float(np.nanmean(ms)))
        out[v + "_thr"] = th
    return out


def boot0(tr, t15, n=10000, seed=7, extra=0.0) -> float:
    """진입 월 블록 부트스트랩, 평균 ≤ 0 비율."""
    if not tr:
        return 1.0
    g = np.array([z[2] for z in tr]) - extra
    mon = np.array([int(t15[z[0]] // (DAY * 30.4375)) for z in tr])
    u = np.unique(mon)
    groups = [g[mon == m] for m in u]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        s = np.concatenate([groups[k] for k in pick])
        hits += s.mean() <= 0
    return hits / n


def evaluate(S, O, a, b, null=True) -> dict:
    t15 = S["t15"]
    ln, lx, sn, sx, t5 = O
    rep = {}
    for v in ("F", "R"):
        el, es = S[v]
        tr = L.walk(el, es, ln, sn, lx, sx, t5, t15, a, b)
        yrs = (t15[b - 1] - t15[a]) / (365.25 * DAY)
        r = {"base": L.stats(tr, t15, 1.0, yrs), "x3": L.stats(tr, t15, 3.0, yrs),
             "cost15": L.stats(L.walk(el, es, ln, sn, lx, sx, t5, t15, a, b, 0.0005), t15, 1.0, yrs)}
        if null and tr:
            nl = L.circ_null(el, es, ln, sn, lx, sx, t5, t15, a, b, n=1000)
            act = np.mean([z[2] for z in tr])
            r["shift_mean"] = round(float(nl.mean() * 100), 3)
            r["p_shift"] = round(float((np.sum(nl >= act) + 1) / 1001), 4)
            r["p_boot"] = round(boot0(tr, t15), 4)
            r["p_boot_cost15"] = round(boot0(tr, t15, extra=0.0005), 4)
        r["years"] = {}
        y0 = int(np.datetime64(int(t15[a]), "ms").astype("datetime64[Y]").astype(int)) + 1970
        y1 = int(np.datetime64(int(t15[b - 1]), "ms").astype("datetime64[Y]").astype(int)) + 1970
        for Y in range(y0, y1 + 1):
            ya, yb = max(int(np.searchsorted(t15, L.ts(Y))), a), min(int(np.searchsorted(t15, L.ts(Y + 1))), b)
            if yb > ya:
                r["years"][Y] = L.stats(L.walk(el, es, ln, sn, lx, sx, t5, t15, ya, yb), t15)
        r["trades"] = [(int(z[0]), int(z[1]), round(z[2], 6), z[3]) for z in tr]
        rep[v] = r
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.data)
    rep = {}
    # 판정: 현물 2017-09-01 ~ 2019-12-31
    sp = L.load_fut5(base / "5m.csv")
    S = signals(sp)
    t15 = S["t15"]
    O = outcomes(sp, t15, None)
    i0, i1 = int(np.searchsorted(t15, np.datetime64("2017-09-01", "ms").astype(np.int64))), \
        int(np.searchsorted(t15, L.ts(2020)))
    rep["spot_holdout"] = evaluate(S, O, i0, i1)
    rep["spot_on"] = {v: S[v + "_on"] for v in ("F", "R")}
    with np.errstate(invalid="ignore"):
        tk = S["taker15"][i0:i1]
        rep["spot_taker_share_ge_fix_q90"] = round(float(np.nanmean(tk >= FIX["tak_hi"])), 4)
        rep["spot_taker_share_le_fix_q10"] = round(float(np.nanmean(tk <= FIX["tak_lo"])), 4)
        rep["spot_taker_quantiles"] = [round(float(x), 5) for x in np.nanquantile(tk, (0.1, 0.5, 0.9))]
    np.savez_compressed(out / "spot.npz", t15=t15, taker=S["taker15"], rsl=S["rsl15"],
                        **{f"{v}_{k}": S[v + "_thr"][k] for v in ("F", "R") for k in FIX})
    print(json.dumps({k: rep["spot_holdout"][k]["base"] for k in ("F", "R")}), flush=True)
    # 참고: 선물(이미 본 자료)
    fu = L.load_fut5(base / "futures/klines_5m.csv")
    SF = signals(fu)
    tf = SF["t15"]
    OF = outcomes(fu, tf, L.load_funding(base / "futures/funding.csv"))
    j0 = int(np.searchsorted(tf, L.ts(2020) + 60 * DAY))
    j1 = int(np.searchsorted(tf, L.ts(2024)))
    j2 = len(tf) - 2
    rep["fut_2020_2023"] = evaluate(SF, OF, j0, j1, null=False)
    rep["fut_2024_"] = evaluate(SF, OF, j1, j2, null=False)
    for k in ("fut_2020_2023", "fut_2024_"):
        for v in ("F", "R"):
            rep[k][v].pop("trades")
    (out / "result.json").write_text(json.dumps(rep, default=float, ensure_ascii=False, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
