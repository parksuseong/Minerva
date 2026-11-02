"""오늘의 패턴·SCR·이평·볼린저·다이버전스·추세·시간대를 합쳐 진입마다 승리확률을 낸다.

등록: 2026-11-02-btc-combined-prob-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_combined_prob.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_book2 as K
import btc_pattern_census as P
import btc_pattern_context as C
import btc_swing_patterns as S
from btc_fractal import MIN5, load5, resample, rsi
from btc_scr_ma import fit_logit
from btc_zone_trigger import ema

SIZES = (0.02, 0.04)
KINDS = ("DB", "DBP", "TB", "IHS", "GART", "BAT", "CRAB", "BFLY", "ABCD", "3DRV", "GP", "ATRI")
TRAIN_END = S.SPLIT
MIN_SEL = 100


def side_code(code: int, side: int) -> tuple:
    scr, ma, bb = code // 16, (code // 4) % 4, code % 4
    if side < 0:
        scr, ma, bb = 4 - scr, 3 - ma, 3 - bb
    return scr, ma, bb


def features(d: dict, st: np.ndarray, r30: np.ndarray, trend: np.ndarray, vol_hi: np.ndarray,
             piv: dict, name: str, at: int, side: int, th: float, rr: float) -> list:
    sig = at - 1  # 진입 전에 마감된 봉
    f = [1.0 if name == k else 0.0 for k in KINDS]
    f.append(1.0 if side > 0 else 0.0)
    f.append(1.0 if th > 0.03 else 0.0)
    scr, ma, bb = side_code(int(st[sig]), side) if st[sig] >= 0 else (2, 1, 1)
    f += [1.0 if scr == k else 0.0 for k in range(5)]
    f += [1.0 if ma == k else 0.0 for k in range(4)]
    f += [1.0 if bb == k else 0.0 for k in range(4)]
    # 다이버전스(30분봉 RSI): 직전 확정 피벗 저점(숏은 고점)보다 최근 6봉 극값이 더 나가고 RSI는 덜 나감
    pv = [p for p in piv[th] if p[3] <= sig and p[2] == (-1 if side > 0 else 1)]
    div = 0.0
    if pv:
        pi, pp = pv[-1][0], pv[-1][1]
        w0 = max(sig - 5, 0)
        if side > 0:
            j = w0 + int(np.argmin(d["l"][w0:sig + 1]))
            div = float(d["l"][j] < pp and r30[j] > r30[pi])
        else:
            j = w0 + int(np.argmax(d["h"][w0:sig + 1]))
            div = float(d["h"][j] > pp and r30[j] < r30[pi])
    f.append(div)
    tr = trend[sig] * side  # +1 같은 방향, −1 반대, 0 중립
    f += [float(tr > 0), float(tr < 0)]
    hour = int(((d["t"][sig] // 3_600_000) + 9) % 24)
    f += [float(hour in (22, 23, 0, 1)), float(hour in (9, 10))]
    f.append(float(vol_hi[sig]))
    f.append(float(np.log(max(rr, 0.05))))
    return f


FEATS = (list(KINDS) + ["롱", "큰패턴(4%)"] + [f"SCR:{x}" for x in C.SCR] +
         [f"이평:{x}" for x in C.MA] + [f"볼린저:{x}" for x in C.BB] +
         ["다이버전스", "4h추세 같은방향", "4h추세 반대", "시간 22~01시", "시간 09~10시",
          "변동성 큼", "log 손익비"])


def build(d5: dict, offset: int) -> tuple:
    sh = dict(d5)
    sh["t"] = d5["t"] - offset * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + offset * 60_000
    S.CAP = K.CAP = 48
    n = len(d["c"])
    st = C.states(d)
    r30 = rsi(d["c"])
    h4 = resample(d5, 48)
    e20, e50 = ema(h4["c"], 20), ema(h4["c"], 50)
    t4 = np.where((h4["c"] > e20) & (e20 > e50), 1, np.where((h4["c"] < e20) & (e20 < e50), -1, 0))
    j4 = np.searchsorted(h4["t"] + 4 * 3_600_000, d["t"] + 1_800_000, side="right") - 1
    trend = np.where(j4 >= 0, t4[np.maximum(j4, 0)], 0)
    lr = np.r_[0.0, np.diff(np.log(d["c"]))]
    vol = np.full(n, np.nan)
    vol[47:] = np.lib.stride_tricks.sliding_window_view(lr, 48).std(1)
    med = np.full(n, np.nan)
    for i in range(1440, n, 48):
        med[i:i + 48] = np.nanmedian(vol[i - 1440:i])
    vol_hi = np.nan_to_num(vol > med).astype(bool)
    piv = {th: S.zigzag(d["h"], d["l"], th) for th in SIZES}
    X, y, g, meta = [], [], [], []
    for th in SIZES:
        for side in (1, -1):
            for name, at, gross, why, rr in P.rows_for(d, side, th):
                if at < 2000 or at >= n:
                    continue
                X.append(features(d, st, r30, trend, vol_hi, piv, name, at, side, th, rr))
                net = gross - 2 * S.COST[0]
                y.append(float(net > S.WIN))
                g.append(gross)
                meta.append((name, side, th, int(at), float(d["t"][at]), why, float(rr)))
    return d, np.array(X), np.array(y), np.array(g), meta


def predict(fit: tuple, x: np.ndarray) -> np.ndarray:
    w, mu, s = fit
    z = np.c_[np.ones(len(x)), (x - mu) / s]
    return 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))


def auc(y, p):
    o = np.argsort(p)
    rk = np.empty(len(p))
    rk[o] = np.arange(1, len(p) + 1)
    pos = y == 1
    a, b = pos.sum(), (~pos).sum()
    return float((rk[pos].sum() - a * (a + 1) / 2) / (a * b)) if a and b else float("nan")


def dedup(idx: np.ndarray, at: np.ndarray, score: np.ndarray) -> np.ndarray:
    """같은 봉에서 여러 후보가 겹치면 점수 가장 높은 하나만, 보유 중(48봉)에는 새 진입 없음."""
    order = np.argsort(at[idx], kind="stable")
    idx = idx[order]
    keep, busy = [], -1
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and at[idx[j + 1]] == at[idx[i]]:
            j += 1
        grp = idx[i:j + 1]
        best = grp[np.argmax(score[grp])]
        if at[best] > busy:
            keep.append(best)
            busy = at[best] + 48
        i = j + 1
    return np.array(keep, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    d, X, y, g, meta = build(d5, args.offset)
    t = np.array([m[4] for m in meta])
    at = np.array([m[3] for m in meta])
    rr = np.array([m[6] for m in meta])
    tr, te = t < TRAIN_END, t >= TRAIN_END
    fit = fit_logit(X[tr], y[tr], lam=10.0)
    p = predict(fit, X)
    # 기대 손익 점수: p×(계획 이익) − (1−p)×(계획 손실)을 손절 거리 1 기준으로
    score = p * rr - (1 - p)
    # 문턱은 학습 구간에서만: 겹침 제거 뒤 100회 이상 중 비용 뒤 평균이 가장 큰 승리확률 문턱
    best = None
    for q in np.arange(0.30, 0.71, 0.02):
        idx = dedup(np.flatnonzero(tr & (p >= q) & (score > 0)), at, score)
        if len(idx) < MIN_SEL:
            continue
        ev = float((g[idx] - 2 * S.COST[0]).mean())
        if best is None or ev > best[1]:
            best = (round(float(q), 2), ev, len(idx))
    rep: dict = {"offset": args.offset, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                 "base_win_train": round(float(y[tr].mean()) * 100, 1),
                 "base_win_test": round(float(y[te].mean()) * 100, 1),
                 "auc_train": round(auc(y[tr], p[tr]), 3), "auc_test": round(auc(y[te], p[te]), 3),
                 "threshold": best}
    rng = np.random.default_rng(20261102)
    boots = [auc(y[te][b], p[te][b]) for b in (rng.integers(0, te.sum(), te.sum()) for _ in range(500))]
    rep["auc_test_ci"] = [round(float(np.nanpercentile(boots, 2.5)), 3),
                          round(float(np.nanpercentile(boots, 97.5)), 3)]
    bins = []
    for lo, hi in ((0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)):
        k = te & (p >= lo) & (p < hi)
        bins.append([lo, hi, int(k.sum()), round(float(y[k].mean()) * 100, 1) if k.any() else None,
                     round(float(p[k].mean()) * 100, 1) if k.any() else None])
    rep["calibration_test"] = bins
    w = fit[0][1:] / fit[2]
    rep["coef"] = sorted([[FEATS[i], round(float(fit[0][1 + i]), 3)] for i in range(len(FEATS))],
                         key=lambda x: -abs(x[1]))
    if best:
        q = best[0]
        for nm, mask in (("train", tr), ("test", te)):
            idx = dedup(np.flatnonzero(mask & (p >= q) & (score > 0)), at, score)
            for cost, tag in ((S.COST[0], "c05"), (S.COST[1], "c10")):
                net = g[idx] - 2 * cost
                srt = np.sort(net)
                eq = np.cumprod(1 + net)
                rep[f"sel_{nm}_{tag}"] = {
                    "n": int(len(idx)), "win": round(float((net > S.WIN).mean() * 100), 1),
                    "p_mean": round(float(p[idx].mean() * 100), 1),
                    "ev": round(float(net.mean() * 100), 3),
                    "ret": round(float((eq[-1] - 1) * 100), 1) if len(eq) else 0.0,
                    "mdd": round(float((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() * 100), 1)
                    if len(eq) else 0.0,
                    "drop5": round(float(srt[:-5].mean() * 100), 3) if len(srt) > 5 else None,
                }
            if nm == "test":
                rep["test_trades"] = [
                    {"name": meta[i][0], "side": meta[i][1], "th": meta[i][2], "at": meta[i][3],
                     "t": meta[i][4], "why": meta[i][5], "p": round(float(p[i]), 3),
                     "gross": round(float(g[i]), 5), "rr": round(float(rr[i]), 2)} for i in idx]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("coef", "test_trades")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
