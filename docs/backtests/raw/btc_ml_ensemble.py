"""비트코인 5모델 다수결 + 신뢰도 진입. 등록: 2026-11-03-btc-ml-ensemble-hypotheses.md

1단계(predict): 해마다 다시 학습해 30분봉마다 3분류 확률을 낸다 → <out>/pred.npz
2단계(trade): 확률로 매매·판정 → <out>/result.json
실행: PYTHONPATH=<numpy·sklearn·xgboost·torch 경로>:src:docs/backtests/raw \
      python docs/backtests/raw/btc_ml_ensemble.py --out <폴더> [--stage predict|trade|all]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import xgboost  # noqa: E402,F401  btc_trend_detect가 가짜 모듈을 넣기 전에 진짜를 먼저 읽는다
from btc_fractal import cci, load5, resample, rsi, sma, stoch  # noqa: E402
from btc_trend_detect import hold_state, structure  # noqa: E402

M30 = 1_800_000
D = 86_400_000
H = 24
SEQ = 64
YEARS = (2022, 2023, 2024, 2025, 2026)
CS = (0.45, 0.55, 0.65)
COST = 0.0005
STOP = 0.03
MODELS = ("rf", "mlp", "xgb", "lstm", "cnn")
ts = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731


def scr_cols(d: dict) -> np.ndarray:
    return np.stack([stoch(d["h"], d["l"], d["c"]) / 100, cci(d["h"], d["l"], d["c"]) / 200,
                     rsi(d["c"]) / 100], 1)


def asof(t_close_src: np.ndarray, vals: np.ndarray, t_need: np.ndarray) -> np.ndarray:
    j = np.searchsorted(t_close_src, t_need, side="right") - 1
    out = vals[np.maximum(j, 0)].astype(float)
    out[j < 0] = np.nan
    return out


def roll(x: np.ndarray, n: int, fn) -> np.ndarray:
    """평균·표준편차는 누적합으로 계산한다(창 배열을 만들면 메모리가 수 GB가 된다)."""
    out = np.full(len(x), np.nan)
    if fn in (np.mean, np.std):
        c1 = np.cumsum(np.r_[0.0, x])
        c2 = np.cumsum(np.r_[0.0, x * x])
        m = (c1[n:] - c1[:-n]) / n
        out[n - 1 :] = m if fn is np.mean else np.sqrt(np.maximum((c2[n:] - c2[:-n]) / n - m * m, 0))
        return out
    w = np.lib.stride_tricks.sliding_window_view(x, n)
    out[n - 1 :] = fn(w, axis=1)
    return out


def build(d5: dict) -> dict:
    d = resample(d5, 6)
    c, h, l, v, t = d["c"], d["h"], d["l"], d["v"], d["t"]  # noqa: E741
    tc = t + M30
    lc = np.log(c)
    cols, names = [], []

    def add(x, name):
        cols.append(x)
        names.append(name)

    for k in (1, 2, 4, 8, 16, 48, 96, 336):
        add(np.r_[np.full(k, np.nan), lc[k:] - lc[:-k]], f"r{k}")
    r1 = np.r_[np.nan, np.diff(lc)]
    for n in (48, 336):
        hh, ll = roll(h, n, np.max), roll(l, n, np.min)
        add((c - ll) / np.where(hh > ll, hh - ll, np.nan), f"pos{n}")
        add(np.log(roll(np.nan_to_num(r1), n, np.std) + 1e-6), f"vol{n}")
        add(c / sma(c, n) - 1, f"ma{n}")
    lv = np.log(np.maximum(v, 1e-9))
    vz = (lv - roll(lv, 336, np.mean)) / (roll(lv, 336, np.std) + 1e-9)
    add(vz, "vz")
    s30 = scr_cols(d)
    for i, nm in enumerate(("s", "cci", "rsi")):
        add(s30[:, i], f"{nm}30")
    for fac, tag in ((48, "4h"), (288, "1d")):
        f = resample(d5, fac)
        sc = scr_cols(f)
        for i, nm in enumerate(("s", "cci", "rsi")):
            add(asof(f["t"] + fac * 300_000, sc[:, i], tc), f"{nm}{tag}")
    dd = resample(d5, 288)
    dclose = dd["t"] + D
    add(asof(dclose, dd["c"] / sma(dd["c"], 60) - 1, tc), "dma60")
    sh = dict(d5)
    sh["t"] = d5["t"] - 4 * D
    wk = resample(sh, 288 * 7)
    wk["t"] = wk["t"] + 4 * D
    from btc_fractal import scr3

    wl, ws, *_ = scr3(wk)
    t4 = asof(wk["t"] + 7 * D, np.array(hold_state(wl, ws), float), tc)
    t5 = asof(dclose, np.array(structure(dd, 5), float), tc)
    add(t4, "t4")
    add(t5, "t5")
    add(np.where(t4 == t5, t4, 0.0), "t8")
    X = np.stack(cols, 1).astype(np.float32)
    rng = np.where(c > 0, (h - l) / c, np.nan)
    seq = np.stack([r1, rng, vz, s30[:, 0], s30[:, 1], s30[:, 2]], 1).astype(np.float32)
    f = np.r_[lc[H:] - lc[:-H], np.full(H, np.nan)]
    past = np.r_[np.full(H, np.nan), lc[H:] - lc[:-H]]
    thr = 0.5 * roll(np.nan_to_num(past), 1440, np.std)
    y = np.full(len(c), -1, np.int64)
    ok = np.isfinite(f) & np.isfinite(thr) & (thr > 0)
    y[ok & (f > thr)] = 2
    y[ok & (f < -thr)] = 0
    y[ok & (np.abs(f) <= thr)] = 1
    return {"d": d, "X": X, "seq": seq, "y": y, "names": names}


# ---- 모델 ----
def fit_predict_tab(name: str, Xtr, ytr, Xte, seed: int, ncls: int = 3) -> np.ndarray:
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        m = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=50,
                                   n_jobs=2, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        m = MLPClassifier((64, 32), early_stopping=True, max_iter=200, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)
    import xgboost as xgb

    prm = {"max_depth": 5, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "nthread": 2,
           "seed": seed}
    prm.update({"objective": "multi:softprob", "num_class": ncls} if ncls > 2
               else {"objective": "binary:logistic"})
    bst = xgb.train(prm, xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)
    p = bst.predict(xgb.DMatrix(Xte))
    return p if ncls > 2 else np.stack([1 - p, p], 1)


def fit_predict_seq(kind: str, seqn, Xn, y, tr_idx, te_idx, seed: int, ncls: int = 3) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    nf, nt = seqn.shape[1], Xn.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "lstm":
                self.body = nn.LSTM(nf, 32, batch_first=True)
            else:
                self.body = nn.Sequential(nn.Conv1d(nf, 32, 5, padding=2), nn.ReLU(),
                                          nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
                                          nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Linear(32 + nt, 32), nn.ReLU(), nn.Linear(32, ncls))

        def forward(self, s, x):
            if kind == "lstm":
                _, (hn, _) = self.body(s)
                z = hn[-1]
            else:
                z = self.body(s.transpose(1, 2)).squeeze(-1)
            return self.head(torch.cat([z, x], 1))

    S = torch.from_numpy(seqn)
    Xt = torch.from_numpy(Xn)
    off = torch.arange(-SEQ + 1, 1)

    def batch(idx):
        ii = torch.from_numpy(idx)
        return S[ii[:, None] + off[None, :]], Xt[ii]

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    yt = torch.from_numpy(y)
    for _ in range(4):
        perm = rng.permutation(tr_idx)
        for k in range(0, len(perm), 256):
            b = perm[k : k + 256]
            s, x = batch(b)
            opt.zero_grad()
            loss = lossf(net(s, x), yt[torch.from_numpy(b)])
            loss.backward()
            opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(te_idx), 2048):
            s, x = batch(te_idx[k : k + 2048])
            out.append(torch.softmax(net(s, x), 1).numpy())
    return np.concatenate(out)


def predict(out: Path, d5: dict) -> None:
    B = build(d5)
    X, seq, y, t = B["X"], B["seq"], B["y"], B["d"]["t"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.zeros(n, bool)
    vv = np.convolve(valid.astype(int), np.ones(SEQ, int), "full")[:n] == SEQ
    seq_ok[:] = vv
    probs = np.full((len(MODELS), n, 3), np.nan, np.float32)
    log = {}
    for Y in YEARS:
        tr = np.flatnonzero((t >= ts(2018)) & (t < ts(Y) - H * M30) & seq_ok & (y >= 0))
        te = np.flatnonzero((t >= ts(Y)) & (t < ts(Y + 1)) & seq_ok)
        if not len(te):
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        Xn[~np.isfinite(Xn)] = 0
        smu, ssd = seq[tr].mean(0), seq[tr].std(0) + 1e-6
        Sn = ((seq - smu) / ssd).astype(np.float32)
        Sn[~np.isfinite(Sn)] = 0
        tr2 = tr[::2]
        for mi, name in enumerate(MODELS):
            t0 = time.time()
            if name in ("rf", "mlp", "xgb"):
                p = fit_predict_tab(name, Xn[tr2], y[tr2], Xn[te], seed=Y)
            else:
                p = fit_predict_seq(name, Sn, Xn, y, tr2, te, seed=Y)
            probs[mi, te] = p
            log[f"{Y}-{name}"] = round(time.time() - t0, 1)
            print(Y, name, "train", len(tr2), "test", len(te), "sec", log[f"{Y}-{name}"], flush=True)
    np.savez_compressed(out / "pred.npz", probs=probs, y=y, t=t, o=B["d"]["o"], h=B["d"]["h"],
                        l=B["d"]["l"], c=B["d"]["c"])
    (out / "train_log.json").write_text(json.dumps(log, indent=1))


# ---- 매매 ----
def positions(probs: np.ndarray, c: float) -> np.ndarray:
    votes = probs.argmax(2)
    up = (votes == 2).sum(0)
    conf = probs[:, :, 2].mean(0)
    known = np.isfinite(probs).all((0, 2))
    want = np.zeros(probs.shape[1], np.int8)
    hold = False
    for i in range(len(want)):
        if not known[i]:
            hold = False
        elif hold:
            hold = up[i] >= 3
        else:
            hold = up[i] >= 3 and conf[i] >= c
        want[i] = hold
    return want


def trade(P: dict, want: np.ndarray, i0: int, i1: int) -> list:
    o, h, l = P["o"], P["h"], P["l"]  # noqa: E741
    out, inpos, px, line, ent = [], False, 0.0, 0.0, 0
    for i in range(max(i0, 1), i1):
        sig = want[i - 1]
        if inpos:
            if not sig:
                out.append((ent, o[i] / px - 1))
                inpos = False
            elif l[i] <= line:
                out.append((ent, min(o[i], line) / px - 1))
                inpos = False
                continue
            else:
                continue
        if sig and not inpos:
            inpos, px, ent = True, o[i], i
            line = px * (1 - STOP)
            if l[i] <= line:
                out.append((ent, line / px - 1))
                inpos = False
    if inpos:
        out.append((ent, P["c"][i1 - 1] / px - 1))
    return out


def summ(tr: list) -> dict:
    if not tr:
        return {"n": 0, "x1": 0.0, "mdd": 0.0}
    net = np.array([g for _, g in tr]) - 2 * COST
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {"n": len(tr), "win": round(float((net > 0.001).mean() * 100), 1),
            "ev": round(float(net.mean() * 100), 3), "x1": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1)}


def hold_stats(P: dict, i0: int, i1: int) -> dict:
    c = P["c"][i0:i1]
    eq = c / P["o"][i0]
    return {"x1": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / np.maximum.accumulate(eq) - 1).min() * 100), 1)}


def trade_stage(out: Path) -> None:
    z = np.load(out / "pred.npz")
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    probs, y, t = z["probs"], z["y"], z["t"]
    n = len(t)
    idx = {Y: int(np.searchsorted(t, ts(Y))) for Y in (*YEARS, 2027)}
    sel0, sel1, te0 = idx[2022], idx[2024], idx[2024]
    te1 = n
    rep: dict = {"classification": {}}
    known = np.isfinite(probs).all((0, 2))
    votes = probs.argmax(2)
    up = (votes == 2).sum(0)
    dn = (votes == 0).sum(0)
    maj = np.where(up >= 3, 2, np.where(dn >= 3, 0, 1))
    for Y in YEARS:
        m = known & (y >= 0) & (t >= ts(Y)) & (t < ts(Y + 1))
        if not m.any():
            continue
        base = np.bincount(y[m], minlength=3)
        row = {"n": int(m.sum()), "label_mix": base.tolist(),
               "most_common": round(float(base.max() / m.sum() * 100), 1),
               "vote_acc": round(float((maj[m] == y[m]).mean() * 100), 1)}
        for mi, name in enumerate(MODELS):
            row[name] = round(float((votes[mi][m] == y[m]).mean() * 100), 1)
        rep["classification"][Y] = row
        print("acc", Y, row, flush=True)
    sel = {}
    for c in CS:
        w = positions(probs, c)
        sel[c] = summ(trade(P, w, sel0, sel1))
        print("select 2022~23 c", c, sel[c], flush=True)
    best = max(CS, key=lambda c: sel[c]["x1"])
    w = positions(probs, best)
    test = summ(trade(P, w, te0, te1))
    hold = hold_stats(P, te0, te1)
    rng = np.random.default_rng(61)
    rand = []
    seg = w[te0:te1].copy()
    for _ in range(200):
        k = int(rng.integers(48 * 30, len(seg) - 48 * 30))
        w2 = w.copy()
        w2[te0:te1] = np.roll(seg, k)
        rand.append(summ(trade(P, w2, te0, te1))["x1"])
    rand = np.array(rand)
    p = float((rand >= test["x1"]).mean())
    calmar = test["x1"] / max(abs(test["mdd"]), 1e-9)
    hcal = hold["x1"] / max(abs(hold["mdd"]), 1e-9)
    acc_ok = all(r["vote_acc"] > r["most_common"] for Y, r in rep["classification"].items()
                 if Y >= 2024)
    crit = [test["x1"] > 0 and test.get("ev", -1) > 0, calmar > hcal, p < 0.05, acc_ok]
    exposure = float(w[te0:te1].mean() * 100)
    rep.update(select=sel, c=best, test=test, hold=hold, p_random=p,
               rand_median=float(np.median(rand)), exposure_pct=round(exposure, 1), criteria=crit)
    print("PICK c", best, "test", test, "hold", hold, "p", p, "exposure", round(exposure, 1),
          "crit", crit, flush=True)
    (out / "result.json").write_text(json.dumps(rep, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("predict", "all"):
        predict(out, load5(Path("~/.minerva/data/btc/5m.csv").expanduser()))
    if a.stage in ("trade", "all"):
        trade_stage(out)


if __name__ == "__main__":
    main()
