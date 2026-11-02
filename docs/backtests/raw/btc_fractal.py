"""SCR 3단 신호의 하루 승패를 모양·이평·SCR·상위봉 특징으로 학습. 여러 봉을 합쳐 30분봉에 옮겨 본다.

등록: 2026-11-02-btc-fractal-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_fractal.py --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MIN5 = 300_000
FRAMES = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "4h": 48, "1d": 288}
HORIZON = 48
COST = (0.0005, 0.001)
WIN = 0.001
TEST_DAYS = 183
LAST_DAYS = 30
SHAPE_K = (1, 2, 3, 5, 8, 13, 21, 34, 48, 96, 192, 384)
MA_N = (10, 20, 60, 120, 240, 480, 960)
UP = 8  # 상위봉 = 같은 봉 8개 묶음 (30분봉이면 4시간봉)
P_TAKE = 0.7


def load5(path: Path) -> dict:
    cols = {k: [] for k in ("t", "o", "h", "l", "c", "v")}
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            cols["t"].append(int(row[0]))
            cols["o"].append(float(row[1]))
            cols["h"].append(float(row[2]))
            cols["l"].append(float(row[3]))
            cols["c"].append(float(row[4]))
            cols["v"].append(float(row[5]))
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def resample(d: dict, factor: int, base_ms: int = MIN5) -> dict:
    """시각 기준 묶음, 봉이 다 찬 묶음만."""
    if factor == 1:
        return d
    span = base_ms * factor
    key = (d["t"] // span).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    st, en = edges[:-1], edges[1:]
    full = (en - st) == factor
    st, en = st[full], en[full]
    return {
        "t": key[st].astype(float) * span,
        "o": d["o"][st],
        "h": _agg(d["h"], st, en, np.max),
        "l": _agg(d["l"], st, en, np.min),
        "c": d["c"][en - 1],
        "v": _agg(d["v"], st, en, np.sum),
    }


def _agg(x: np.ndarray, st: np.ndarray, en: np.ndarray, fn) -> np.ndarray:
    if not len(st):
        return np.array([])
    red = {np.max: np.maximum, np.min: np.minimum, np.sum: np.add}[fn]
    out = red.reduceat(x, st)
    # reduceat은 다음 시작까지 묶으므로, 빈칸 뒤 묶음을 바로잡는다
    bad = np.flatnonzero(np.r_[st[1:] != en[:-1], en[-1] != len(x)])
    for i in bad:
        out[i] = fn(x[st[i]:en[i]])
    return out


def win_view(x: np.ndarray, n: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, n)


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        s = np.cumsum(np.r_[0.0, x])
        out[n - 1:] = (s[n:] - s[:-n]) / n
    return out


def stoch(h, l, c, n=5):
    out = np.full(len(c), np.nan)
    hh = win_view(h, n).max(1)
    ll = win_view(l, n).min(1)
    rng = hh - ll
    v = np.where(rng == 0, 50.0, (c[n - 1:] - ll) / np.where(rng == 0, 1, rng) * 100)
    out[n - 1:] = v
    return out


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    out = np.full(len(c), np.nan)
    w = win_view(tp, n)
    m = w.mean(1)
    md = np.abs(w - m[:, None]).mean(1)
    out[n - 1:] = np.where(md == 0, 0.0, (tp[n - 1:] - m) / (0.015 * np.where(md == 0, 1, md)))
    return out


def rsi(c, n=14):
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    g, ls = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = g[:n].mean(), ls[:n].mean()
    res = []
    res.append(100.0 if al == 0 else 100 - 100 / (1 + ag / al))
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + ls[i]) / n
        res.append(100.0 if al == 0 else 100 - 100 / (1 + ag / al))
    out[n:] = res
    return out


def crossed(x: np.ndarray, line: float, up: bool) -> np.ndarray:
    p = np.r_[np.nan, x[:-1]]
    with np.errstate(invalid="ignore"):
        return (p <= line) & (line < x) if up else (p >= line) & (line > x)


def recent(x: np.ndarray, w: int = 3) -> np.ndarray:
    out = x.copy()
    for j in range(1, w):
        out |= np.r_[np.zeros(j, bool), x[:-j]]
    return out


def scr3(d: dict) -> tuple:
    s, cc, r = stoch(d["h"], d["l"], d["c"]), cci(d["h"], d["l"], d["c"]), rsi(d["c"])
    lv_l = recent(crossed(s, 20, True)) & recent(crossed(cc, -100, True)) & recent(crossed(r, 30, True))
    lv_s = recent(crossed(s, 80, False)) & recent(crossed(cc, 100, False)) & recent(crossed(r, 70, False))
    on_l = lv_l & ~np.r_[False, lv_l[:-1]]
    on_s = lv_s & ~np.r_[False, lv_s[:-1]]
    return on_l, on_s, s, cc, r


def features(d: dict, s, cc, r, idx: np.ndarray, side: np.ndarray, frame_ms: int) -> np.ndarray:
    """봉 크기와 무관하게 맞춘 특징. 방향 특징은 방향 부호를 곱한다."""
    c, h, l, o, v = d["c"], d["h"], d["l"], d["o"], d["v"]
    lr = np.r_[0.0, np.diff(np.log(c))]
    vol = np.full(len(c), np.nan)
    vol[47:] = win_view(lr, 48).std(1)
    vol = np.where(vol > 0, vol, np.nan)
    cols = []
    lc = np.log(c)
    for k in SHAPE_K:
        past = np.r_[np.full(k, np.nan), lc[:-k]]
        cols.append((lc - past) / (vol * np.sqrt(k)) * 1)
    for n in MA_N:
        cols.append((lc - np.log(sma(c, n))) / (vol * np.sqrt(n)))
    for n in (20, 60):
        m = np.log(sma(c, n))
        cols.append((m - np.r_[np.full(10, np.nan), m[:-10]]) / vol)
    hh = np.r_[np.full(47, np.nan), win_view(h, 48).max(1)]
    ll = np.r_[np.full(47, np.nan), win_view(l, 48).min(1)]
    cols.append((c - ll) / np.where(hh > ll, hh - ll, np.nan) * 2 - 1)
    cols += [s / 50 - 1, np.clip(cc / 100, -5, 5), r / 50 - 1]
    rng = np.where(h > l, h - l, np.nan)
    cols.append((c - o) / rng)
    cols.append((c - l) / rng * 2 - 1)
    sided = np.column_stack(cols)
    va = np.r_[np.full(47, np.nan), win_view(v, 48).mean(1)]
    nonside = [np.log((v + 1e-9) / (va + 1e-9)), np.log(vol * np.sqrt(HORIZON))]
    up = higher(d, frame_ms)
    sided = np.column_stack([sided, up])
    x = np.column_stack([sided[idx] * side[:, None], np.column_stack(nonside)[idx]])
    return x


def higher(d: dict, frame_ms: int) -> np.ndarray:
    """같은 봉 UP개를 묶은 상위봉의, 이미 마감된 마지막 봉의 SCR 값과 모양."""
    span = frame_ms * UP
    key = (d["t"] // span).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    st, en = edges[:-1], edges[1:]
    hd = {"o": d["o"][st], "c": d["c"][en - 1],
          "h": np.array([d["h"][a:b].max() for a, b in zip(st, en)]),
          "l": np.array([d["l"][a:b].min() for a, b in zip(st, en)])}
    hs, hc, hr = stoch(hd["h"], hd["l"], hd["c"]), cci(hd["h"], hd["l"], hd["c"]), rsi(hd["c"])
    lc = np.log(hd["c"])
    lr = np.r_[0.0, np.diff(lc)]
    hv = np.full(len(lc), np.nan)
    if len(lc) > 20:
        hv[19:] = win_view(lr, 20).std(1)
    hv = np.where(hv > 0, hv, np.nan)
    shp = [(lc - np.r_[np.full(k, np.nan), lc[:-k]]) / (hv * np.sqrt(k)) for k in (1, 3, 8)]
    ma = (lc - np.log(sma(hd["c"], 20))) / (hv * np.sqrt(20))
    hf = np.column_stack([hs / 50 - 1, np.clip(hc / 100, -5, 5), hr / 50 - 1, ma] + shp)
    close_t = (key[st] + 1) * span  # 묶음 마감 시각
    bar_close = d["t"] + frame_ms
    j = np.searchsorted(close_t, bar_close, side="right") - 1
    out = np.full((len(d["t"]), hf.shape[1]), np.nan)
    ok = j >= 0
    out[ok] = hf[j[ok]]
    return out


def labels(d: dict, on_l, on_s) -> tuple:
    """신호마다 다음 봉 시가 진입, 반대 원신호(다음 봉 시가) 또는 48봉 뒤 시가 청산. 손절 없음."""
    o, n = d["o"], len(d["o"])
    sig = np.flatnonzero(on_l | on_s)
    side = np.where(on_l[sig], 1, -1)
    opp_l, opp_s = np.flatnonzero(on_s), np.flatnonzero(on_l)
    idx, sd, gr, ex, why = [], [], [], [], []
    for i, s_ in zip(sig, side):
        e = i + 1
        if e >= n:
            continue
        arr = opp_l if s_ > 0 else opp_s
        k = np.searchsorted(arr, e)
        j = arr[k] + 1 if k < len(arr) else n
        cap = e + HORIZON
        x = min(j, cap)
        if x >= n:
            continue
        idx.append(i)
        sd.append(s_)
        gr.append((o[x] / o[e] - 1) * s_)
        ex.append(x)
        why.append(0 if j <= cap else 1)
    return (np.array(idx), np.array(sd), np.array(gr), np.array(ex), np.array(why))


def fit_logit(x, y, lam=3.0, steps=40):
    mu, s = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-9
    z = np.c_[np.ones(len(x)), np.nan_to_num((x - mu) / s)]
    w = np.zeros(z.shape[1])
    reg = np.eye(z.shape[1]) * lam
    reg[0, 0] = 0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        g = z.T @ (p - y) + reg @ w
        hs = (z.T * (p * (1 - p) + 1e-6)) @ z + reg
        w -= np.linalg.solve(hs, g)
    return ("logit", w, mu, s)


def fit_knn(x, y, k=100):
    mu, s = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-9
    return ("knn", np.nan_to_num((x - mu) / s), y, mu, s, k)


def predict(m, x):
    if m[0] == "logit":
        _, w, mu, s = m
        z = np.c_[np.ones(len(x)), np.nan_to_num((x - mu) / s)]
        return 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
    _, xt, yt, mu, s, k = m
    z = np.nan_to_num((x - mu) / s)
    out = np.empty(len(z))
    for i in range(0, len(z), 256):
        dd = ((z[i:i + 256, None, :] - xt[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(dd, k, axis=1)[:, :k]
        out[i:i + 256] = yt[nn].mean(1)
    return out


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def trade(d, idx, sd, p, a, b, cost, take=P_TAKE):
    """포지션 하나. 확률 take 이상만 진입, 반대 원신호 청산(그 신호도 take 이상이면 뒤집기), 48봉 상한."""
    o, n = d["o"], len(d["o"])
    order = np.argsort(idx)
    idx, sd, p = idx[order], sd[order], p[order]
    pos = None
    out = []
    for i, s_, pp in zip(idx, sd, p):
        if i < a or i >= b - 1:
            continue
        if pos is not None:
            e, ps = pos
            if i + 1 >= e + HORIZON:  # 상한이 먼저 왔다
                x = e + HORIZON
                out.append((e, x, ps, (o[x] / o[e] - 1) * ps))
                pos = None
            elif s_ != ps:
                x = i + 1
                out.append((e, x, ps, (o[x] / o[e] - 1) * ps))
                pos = None
        if pos is None and pp >= take and i + 1 < n:
            pos = (i + 1, s_)
    if pos is not None:
        e, ps = pos
        x = min(e + HORIZON, n - 1)
        out.append((e, x, ps, (o[x] / o[e] - 1) * ps))
    if not out:
        return {"n": 0}
    net = np.array([t[3] for t in out]) - 2 * cost
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {"n": len(out), "win": round(float((net > WIN).mean() * 100), 1),
            "ret": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
            "worst": round(float(net.min() * 100), 2), "long": int(sum(t[2] > 0 for t in out))}


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
    sets = {}
    frames30 = None
    for name, f in FRAMES.items():
        d = resample(d5, f)
        on_l, on_s, s, cc, r = scr3(d)
        idx, sd, gr, ex, why = labels(d, on_l, on_s)
        x = features(d, s, cc, r, idx, sd, MIN5 * f)
        y = ((gr - 2 * COST[0]) > WIN).astype(float)
        t_sig = d["t"][idx] + MIN5 * f
        t_end = d["t"][ex]
        sets[name] = {"x": x, "y": y, "t": t_sig, "tend": t_end, "idx": idx, "sd": sd,
                      "timeout": float(why.mean()) if len(why) else 0.0}
        if name == "30m":
            frames30 = d
        print(name, len(d["t"]), "signals", len(idx), "win", round(float(y.mean()) * 100, 1),
              flush=True)
        del d
    del d5
    t30 = sets["30m"]
    test = t30["t"] >= test_start
    rep: dict = {"end": end, "counts": {k: int(len(v["y"])) for k, v in sets.items()},
                 "base_win": {k: round(float(v["y"].mean()) * 100, 1) for k, v in sets.items()},
                 "timeout_share": {k: round(v["timeout"] * 100, 1) for k, v in sets.items()}}

    def train_on(names):
        xs, ys = [], []
        for nm in names:
            v = sets[nm]
            k = v["tend"] < test_start
            xs.append(v["x"][k])
            ys.append(v["y"][k])
        return np.vstack(xs), np.concatenate(ys)

    groups = {"M30": ["30m"], "ALL": list(FRAMES), "OTHER": [k for k in FRAMES if k != "30m"]}
    yt, xt = t30["y"][test], t30["x"][test]
    last = t30["t"][test] >= last_start
    base_rate = float(train_on(["30m"])[1].mean())
    rep["test"] = {"n": int(test.sum()), "win": round(float(yt.mean()) * 100, 1),
                   "base_ll": round(logloss(yt, np.full(len(yt), base_rate)), 4)}
    d30 = frames30
    n30 = len(d30["t"])
    a6 = int(np.searchsorted(d30["t"], test_start))
    a1 = int(np.searchsorted(d30["t"], last_start))
    rep["models"] = {}
    rng = np.random.default_rng(20261102)
    for gname, names in groups.items():
        xtr, ytr = train_on(names)
        for kind in ("logit", "knn"):
            m = fit_logit(xtr, ytr) if kind == "logit" else fit_knn(xtr, ytr)
            p = predict(m, xt)
            boots = []
            for _ in range(500):
                b_ = rng.integers(0, len(yt), len(yt))
                boots.append(auc(yt[b_], p[b_]))
            res = {
                "train_n": int(len(ytr)),
                "auc": round(auc(yt, p), 3),
                "auc_ci": [round(float(np.nanpercentile(boots, 2.5)), 3),
                           round(float(np.nanpercentile(boots, 97.5)), 3)],
                "ll": round(logloss(yt, p), 4),
                "p_ge_07": int((p >= P_TAKE).sum()),
                "bins": [],
            }
            for lo_, hi_ in ((0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.01)):
                k = (p >= lo_) & (p < hi_)
                res["bins"].append([lo_, int(k.sum()), round(float(yt[k].mean()) * 100, 1) if k.any() else None])
            idx_t, sd_t = t30["idx"][test], t30["sd"][test]
            res["trade6m"] = {str(c): trade(d30, idx_t, sd_t, p, a6, n30, c) for c in COST}
            res["trade_before30"] = trade(d30, idx_t, sd_t, p, a6, a1, COST[0])
            res["trade30d"] = trade(d30, idx_t, sd_t, p, a1, n30, COST[0])
            res["last30_auc"] = round(auc(yt[last], p[last]), 3)
            rep["models"][f"{gname}_{kind}"] = res
            print(gname, kind, res["auc"], res["auc_ci"], res["ll"], res["p_ge_07"], flush=True)
    rep["all_signals_trade6m"] = trade(d30, t30["idx"][test], t30["sd"][test],
                                       np.ones(int(test.sum())), a6, n30, COST[0])
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
