"""흐름 + 캔들 패턴의 경우의 수와 캔들 이미지 유사 사례. 봉 크기를 넘어 확률이 같은지(프랙탈) 본다.

등록: 2026-11-02-btc-candle-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_candle.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import MIN5, load5, resample, win_view

FRAMES = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "4h": 48, "1d": 288}
H = 48
COST = (0.0005, 0.001)
WIN = 0.001
TEST_DAYS = 183
LAST_DAYS = 30
MIN_CELL = 100
TAKE = 0.70
IMG = 20
KNN = 200
KNN_POOL = 150_000


def frame_table(d: dict) -> dict:
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    n = len(c)
    lr = np.r_[0.0, np.diff(np.log(c))]
    vol = np.full(n, np.nan)
    vol[H - 1:] = win_view(lr, H).std(1)
    body = np.abs(c - o) / (c * vol)
    bsz = np.where(body < 0.5, 0, np.where(body < 1.5, 1, 2))
    rng = np.where(h > l, h - l, np.nan)
    uw = ((h - np.maximum(o, c)) / rng > 0.4).astype(int)
    lw = ((np.minimum(o, c) - l) / rng > 0.4).astype(int)
    sym = (c >= o).astype(int) * 12 + bsz * 4 + uw * 2 + lw
    lc = np.log(c)
    z = (lc - np.r_[np.full(H, np.nan), lc[:-H]]) / (vol * np.sqrt(H))
    ctx = np.where(z > 1, 2, np.where(z < -1, 0, 1))
    ok = np.isfinite(z) & np.isfinite(vol) & np.isfinite(rng)
    codes = {}
    s1 = np.r_[0, sym[:-1]]
    s2 = np.r_[0, 0, sym[:-2]]
    codes[1] = ctx * 24 ** 3 + sym
    codes[2] = ctx * 24 ** 3 + 24 ** 2 * 0 + sym + 24 * s1 + 1_000_000
    codes[3] = ctx * 24 ** 3 + sym + 24 * s1 + 576 * s2 + 2_000_000
    e = np.arange(n) + 1
    x = e + H
    valid = ok & (x < n)
    r = np.full(n, np.nan)
    r[valid] = d["o"][x[valid]] / d["o"][e[valid]] - 1
    return {"codes": codes, "r": r, "valid": valid, "t": d["t"], "x": x, "vol": vol, "ctx": ctx}


def image(d: dict, vol: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """최근 IMG봉 시가·고가·저가·종가를 마지막 종가 대비, 변동성으로 나눈 그림."""
    lo, lh, ll, lc = (np.log(d[k]) for k in "ohlc")
    out = np.empty((len(rows), IMG * 4), dtype=np.float32)
    for j, arr in enumerate((lo, lh, ll, lc)):
        w = win_view(arr, IMG)[rows - IMG + 1]
        out[:, j * IMG:(j + 1) * IMG] = (w - lc[rows, None]) / (vol[rows, None] * np.sqrt(IMG))
    return out


def wins(r: np.ndarray, cost: float = COST[0]) -> tuple:
    return (r - 2 * cost > WIN).astype(float), (-r - 2 * cost > WIN).astype(float)


def cell_stats(codes: np.ndarray, r: np.ndarray) -> dict:
    lw, sw = wins(r)
    u, inv, cnt = np.unique(codes, return_inverse=True, return_counts=True)
    lsum = np.bincount(inv, lw)
    ssum = np.bincount(inv, sw)
    return {"code": u, "n": cnt, "pl": lsum / cnt, "ps": ssum / cnt}


def consistency(a: dict, b: dict, base_a: float, base_b: float, rng, min_n: int = 200) -> dict:
    common, ia, ib = np.intersect1d(a["code"], b["code"], return_indices=True)
    k = (a["n"][ia] >= min_n) & (b["n"][ib] >= min_n)
    if k.sum() < 10:
        return {"cells": int(k.sum())}
    xa = a["pl"][ia][k] - base_a
    xb = b["pl"][ib][k] - base_b
    real = float(np.corrcoef(xa, xb)[0, 1])
    null = [float(np.corrcoef(xa, rng.permutation(xb))[0, 1]) for _ in range(1000)]
    return {"cells": int(k.sum()), "corr": round(real, 3),
            "p": round((1 + sum(v >= real for v in null)) / 1001, 4)}


def sim(o: np.ndarray, rows: np.ndarray, side: np.ndarray, a: int, b: int, cost: float) -> dict:
    """겹치지 않게: 신호 다음 봉 시가 진입, 48봉 뒤 시가 청산, 그동안 새 신호 무시."""
    out = []
    busy = -1
    for t, s in zip(rows, side):
        if t < a or t <= busy:
            continue
        e, x = t + 1, t + 1 + H
        if x >= b:
            break
        out.append((o[x] / o[e] - 1) * s)
        busy = x - 1
    if not out:
        return {"n": 0}
    net = np.array(out) - 2 * cost
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {"n": len(net), "win": round(float((net > WIN).mean() * 100), 1),
            "ret": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
            "worst": round(float(net.min() * 100), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    end = d5["t"][-1] + MIN5
    test_start = end - TEST_DAYS * 86_400_000
    last_start = end - LAST_DAYS * 86_400_000
    rng = np.random.default_rng(20261102)
    tabs, bars = {}, {}
    for name, f in FRAMES.items():
        d = resample(d5, f)
        tb = frame_table(d)
        tb["train"] = tb["valid"] & (np.where(tb["valid"], d["t"][np.minimum(tb["x"], len(d["t"]) - 1)], np.inf) < test_start)
        tabs[name] = tb
        if name in ("15m", "30m", "1h", "4h", "1d"):
            bars[name] = d
        print(name, len(d["t"]), "train", int(tb["train"].sum()), flush=True)
    del d5
    rep: dict = {"end": end, "rows": {k: int(v["train"].sum()) for k, v in tabs.items()}}
    base = {}
    for k, tb in tabs.items():
        lw, sw = wins(tb["r"][tb["train"]])
        base[k] = (float(lw.mean()), float(sw.mean()))
    rep["base"] = {k: [round(v[0] * 100, 1), round(v[1] * 100, 1)] for k, v in base.items()}

    # 1. 프랙탈 일치: 같은 흐름·패턴의 롱 승률(봉별 기본값을 뺀 값)이 봉끼리 같은 방향인가
    rep["consistency"] = {}
    for kk in (1, 2, 3):
        cs = {nm: cell_stats(tb["codes"][kk][tb["train"]], tb["r"][tb["train"]]) for nm, tb in tabs.items()}
        rep["consistency"][kk] = {
            nm: consistency(cs["30m"], cs[nm], base["30m"][0], base[nm][0], rng)
            for nm in FRAMES if nm != "30m"
        }
        print("consistency", kk, rep["consistency"][kk], flush=True)

    # 2. 경우의 수로 고른 70% 이상 칸이 최근 6개월 30분봉에서 맞는가
    t30, d30 = tabs["30m"], bars["30m"]
    n30 = len(d30["t"])
    test_rows = np.flatnonzero(t30["valid"] & (d30["t"] + 1_800_000 >= test_start))
    a6 = int(test_rows[0]) if len(test_rows) else n30
    a1 = int(np.searchsorted(d30["t"], last_start))
    rep["test_rows"] = int(len(test_rows))
    rep["cells"] = {}
    for src in ("M30", "ALL"):
        names = ["30m"] if src == "M30" else list(FRAMES)
        for kk in (1, 2, 3):
            codes = np.concatenate([tabs[nm]["codes"][kk][tabs[nm]["train"]] for nm in names])
            rr = np.concatenate([tabs[nm]["r"][tabs[nm]["train"]] for nm in names])
            cs = cell_stats(codes, rr)
            pick_l = (cs["n"] >= MIN_CELL) & (cs["pl"] >= TAKE)
            pick_s = (cs["n"] >= MIN_CELL) & (cs["ps"] >= TAKE)
            elig = cs["n"] >= MIN_CELL
            best = np.maximum(cs["pl"], cs["ps"])[elig]
            res = {"eligible": int(elig.sum()), "picked_long": int(pick_l.sum()),
                   "picked_short": int(pick_s.sum()),
                   "best_train_rate": round(float(best.max()) * 100, 1) if len(best) else None,
                   "share_ge_60": round(float((best >= 0.6).mean()) * 100, 2) if len(best) else None}
            lmap = set(cs["code"][pick_l].tolist())
            smap = set(cs["code"][pick_s].tolist())
            tc = t30["codes"][kk][test_rows]
            side = np.array([1 if v in lmap else (-1 if v in smap else 0) for v in tc])
            hit = side != 0
            if hit.any():
                lw, sw = wins(t30["r"][test_rows][hit])
                won = np.where(side[hit] > 0, lw, sw)
                res["test_occ"] = int(hit.sum())
                res["test_occ_win"] = round(float(won.mean()) * 100, 1)
                rows_h, side_h = test_rows[hit], side[hit]
                res["trade6m"] = {str(c): sim(d30["o"], rows_h, side_h, a6, n30, c) for c in COST}
                res["trade_before30"] = sim(d30["o"], rows_h, side_h, a6, a1, COST[0])
                res["trade30d"] = sim(d30["o"], rows_h, side_h, a1, n30, COST[0])
            rep["cells"][f"{src}_{kk}"] = res
            print("cells", src, kk, res, flush=True)

    # 3. 캔들 이미지 유사 사례(kNN)
    rep["knn"] = {}
    vol30 = t30["vol"]
    xt = image(d30, vol30, test_rows)
    lw_t, sw_t = wins(t30["r"][test_rows])
    for src in ("M30", "ALL"):
        names = ["30m"] if src == "M30" else ["15m", "30m", "1h", "4h", "1d"]
        pool_x, pool_l, pool_s = [], [], []
        for nm in names:
            tb, d = tabs[nm], bars[nm]
            rows = np.flatnonzero(tb["train"] & (np.arange(len(d["t"])) >= max(IMG, H)))
            pool_x.append(rows)
        sizes = [len(r_) for r_ in pool_x]
        total = sum(sizes)
        take = min(KNN_POOL, total)
        xs, ls_, ss_ = [], [], []
        for nm, rows in zip(names, pool_x):
            k_ = int(round(take * len(rows) / total))
            pick = np.sort(rng.choice(rows, size=min(k_, len(rows)), replace=False))
            xs.append(image(bars[nm], tabs[nm]["vol"], pick))
            lw, sw = wins(tabs[nm]["r"][pick])
            ls_.append(lw)
            ss_.append(sw)
        X = np.nan_to_num(np.vstack(xs))
        L, S = np.concatenate(ls_), np.concatenate(ss_)
        x2 = (X ** 2).sum(1)
        pl = np.empty(len(xt))
        ps = np.empty(len(xt))
        xq = np.nan_to_num(xt)
        for i in range(0, len(xq), 512):
            q = xq[i:i + 512]
            dd = x2[None, :] - 2 * q @ X.T
            nn = np.argpartition(dd, KNN, axis=1)[:, :KNN]
            pl[i:i + 512] = L[nn].mean(1)
            ps[i:i + 512] = S[nn].mean(1)
        side = np.where(pl >= TAKE, 1, np.where(ps >= TAKE, -1, 0))
        hit = side != 0
        res = {"pool": int(len(X)), "max_pl": round(float(pl.max()) * 100, 1),
               "max_ps": round(float(ps.max()) * 100, 1), "n_ge_take": int(hit.sum())}
        # 확률 구간별 실제 승률 (예측력 확인용, 매매 결과 아님)
        for lo_, hi_ in ((0, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 1.01)):
            k = (pl >= lo_) & (pl < hi_)
            res[f"long_bin_{lo_}"] = [int(k.sum()), round(float(lw_t[k].mean()) * 100, 1) if k.any() else None]
        if hit.any():
            rows_h, side_h = test_rows[hit], side[hit]
            res["trade6m"] = {str(c): sim(d30["o"], rows_h, side_h, a6, n30, c) for c in COST}
            res["trade_before30"] = sim(d30["o"], rows_h, side_h, a6, a1, COST[0])
        rep["knn"][src] = res
        print("knn", src, res, flush=True)
    rep["test_base"] = [round(float(lw_t.mean()) * 100, 1), round(float(sw_t.mean()) * 100, 1)]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
