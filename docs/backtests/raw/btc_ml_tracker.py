"""15분봉 ML 걸어두는 지정가 규칙 2개를 앞으로의 자료로 기록만 한다 (주문 없음).

규칙(2026-11-04 사용자 승인, 문서 2026-11-04-btc-ml-rest-confirm·strict-hypotheses.md):
- A: 5개 모델 평균 점수가 2025년 상위 10% 이상 + 4개 이상이 같은 방향 (V1)
- B: 평균 점수 2025년 상위 5% 이상 + 5개 모두 같은 방향 (S3)
숏은 하위 10%·5%와 반대 방향 동의. 체결·청산은 btc_ml_edge_cost.rest_sim과 같다.

모델은 백테스트의 2026년 모델과 같은 자료·씨드로 한 번 학습해 저장한다(train).
step은 15분마다: 현물 5분봉을 받아 붙이고, 새로 마감된 15분봉에 점수를 매기고,
추적 시작 뒤의 신호로 모의 체결을 다시 계산해 새 사건만 텔레그램으로 알린다.

실행: PYTHONPATH=<pyml>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ml_tracker.py train|step|report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
import urllib.request
from pathlib import Path

import btc_ml_15m as M
import btc_ml_ensemble as E
import numpy as np
from btc_fractal import cci, load5, rsi, stoch
from btc_ml_barrier import barrier_labels
from btc_ml_edge_cost import MAKER, OFF, REST, SL, SLIP, TAKER, THRU, TP, mean_score

HOME = Path("~/.minerva").expanduser()
BASE5 = HOME / "data/btc/5m.csv"
ST = Path(os.environ.get("MINERVA_MLTRACK_DIR", HOME / "research/btc-ml-tracker"))
PRED = HOME / "research/btc-2026-11-04/ml15/pred.npz"
SEEDS_OUT = HOME / "research/btc-2026-11-04/seeds"
URL = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1000&startTime={s}"
BAR15 = 900_000
YEAR = 2026
RULES = {"A": (0.90, 4), "B": (0.95, 5), "AS": (0.90, 4), "BS": (0.95, 5)}
NAMES = {
    "A": "A(4개 동의·상위10%)",
    "B": "B(5개 동의·상위5%)",
    "AS": "A평균(씨드5벌·4개 동의·상위10%)",
    "BS": "B평균(씨드5벌·5개 동의·상위5%)",
}
# AS·BS는 같은 규칙을 씨드 5벌(Y, Y+1000·k) 평균 확률로 계산한다(2026-11-05 사용자 승인,
# 문서 2026-11-04-btc-ml-seeds-hypotheses.md). 확률 칸 접두어: 씨드 Y는 p_, 평균은 pa_.
PFX = {"A": "p_", "B": "p_", "AS": "pa_", "BS": "pa_"}
SEED_K = (1, 2, 3, 4)
NOTE = {
    "A": "씨드 시험에서 운으로 판정됨: 기대값 0으로 봐요",
    "B": "씨드 시험: 작은 우위가 있을 수도, 증명은 안 됨",
    "AS": "씨드 5벌 평균 모델. 11/05 00시대부터 기록, 앞으로의 기록으로만 판정",
    "BS": "씨드 5벌 평균 모델. 11/05 00시대부터 기록, 앞으로의 기록으로만 판정",
}


# ---- 자료 ----
def live_path() -> Path:
    return ST / "live5m.csv"


def load_all() -> dict:
    d = load5(BASE5)
    lp = live_path()
    if lp.exists():
        e = load5(lp)
        keep = e["t"] > d["t"][-1]
        d = {k: np.r_[d[k], e[k][keep]] for k in d}
    # 15분 경계에서 끝나게 자른다(부분 봉 제외는 resample이 한다)
    return d


def fetch_new() -> int:
    d = load_all()
    last = int(d["t"][-1])
    now = int(time.time() * 1000)
    rows, s = [], last + 300_000
    while s < now - 300_000:
        req = urllib.request.Request(URL.format(s=s), headers={"User-Agent": "minerva-tracker"})
        with urllib.request.urlopen(req, timeout=20) as r:
            k = json.load(r)
        k = [x for x in k if int(x[6]) < now]  # 마감된 봉만
        if not k:
            break
        rows += k
        s = int(k[-1][0]) + 300_000
        if len(k) < 999:
            break
    lp = live_path()
    new = not lp.exists()
    with lp.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["open_time", "open", "high", "low", "close", "volume", "close_time"])
        for x in rows:
            w.writerow(x[:7])
    return len(rows)


# ---- 모델 ----
def seq_net(kind: str, nf: int, nt: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "lstm":
                self.body = nn.LSTM(nf, 32, batch_first=True)
            else:
                self.body = nn.Sequential(
                    nn.Conv1d(nf, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
            self.head = nn.Sequential(nn.Linear(32 + nt, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, s, x):
            if kind == "lstm":
                _, (hn, _) = self.body(s)
                z = hn[-1]
            else:
                z = self.body(s.transpose(1, 2)).squeeze(-1)
            return self.head(torch.cat([z, x], 1))

    return Net()


def fit_seq(kind, Sn, Xn, y, tr, seed):
    """E.fit_predict_seq와 같은 순서(씨드 → 망 생성 → 4회 반복)로 학습하고 망을 돌려준다."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    net = seq_net(kind, Sn.shape[1], Xn.shape[1])
    S, Xt, yt = torch.from_numpy(Sn), torch.from_numpy(Xn), torch.from_numpy(y)
    off = torch.arange(-E.SEQ + 1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for _ in range(4):
        perm = rng.permutation(tr)
        for k in range(0, len(perm), 256):
            b = perm[k : k + 256]
            ii = torch.from_numpy(b)
            opt.zero_grad()
            loss = lossf(net(S[ii[:, None] + off[None, :]], Xt[ii]), yt[ii])
            loss.backward()
            opt.step()
    net.eval()
    return net


def fit_tab(name, Xtr, ytr, seed):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        m = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50, n_jobs=2, random_state=seed
        )
        return m.fit(Xtr, ytr)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier((64, 32), early_stopping=True, max_iter=200, random_state=seed).fit(
            Xtr, ytr
        )
    import xgboost as xgb

    prm = {
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 2,
        "seed": seed,
        "objective": "binary:logistic",
    }
    return xgb.train(prm, xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)


def features(d5):
    B = E.build(d5, M.FACTOR)
    X, seq, t = B["X"], B["seq"], B["d"]["t"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    return B, X, seq, t, seq_ok


def norm(X, seq, nz):
    Xn = ((X - nz["mu"]) / nz["sd"]).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0
    Sn = ((seq - nz["smu"]) / nz["ssd"]).astype(np.float32)
    Sn[~np.isfinite(Sn)] = 0
    return Xn, Sn


def predict(models, Xn, Sn, idx):
    import torch
    import xgboost as xgb

    idx = np.asarray(idx)
    out = np.full((len(E.MODELS), len(idx)), np.nan, np.float32)
    off = torch.arange(-E.SEQ + 1, 1)
    S, Xt = torch.from_numpy(Sn), torch.from_numpy(Xn)
    for mi, name in enumerate(E.MODELS):
        m = models[name]
        if name in ("rf", "mlp"):
            out[mi] = m.predict_proba(Xn[idx])[:, 1]
        elif name == "xgb":
            out[mi] = m.predict(xgb.DMatrix(Xn[idx]))
        else:
            res = []
            with torch.no_grad():
                for k in range(0, len(idx), 2048):
                    ii = torch.from_numpy(idx[k : k + 2048])
                    res.append(
                        torch.softmax(m(S[ii[:, None] + off[None, :]], Xt[ii]), 1)[:, 1].numpy()
                    )
            out[mi] = np.concatenate(res)
    return out


def load_models():
    import torch

    meta = pickle.loads((ST / "models.pkl").read_bytes())
    models = meta["tab"]
    for k in ("lstm", "cnn"):
        net = seq_net(k, meta["nf"], meta["nt"])
        net.load_state_dict(torch.load(ST / f"{k}.pt", weights_only=True))
        net.eval()
        models[k] = net
    sm = ST / "seeds_meta.json"
    if sm.exists():
        meta["thr"].update({r: tuple(v) for r, v in json.loads(sm.read_text())["thr"].items()})
    return models, meta


def seed_dir(k: int) -> Path:
    return ST / f"seed{k}"


def load_seed_models(meta) -> list:
    """씨드 k=1~4 모델 묶음. 없으면 빈 목록(평균 규칙은 건너뛴다)."""
    import torch

    out = []
    for k in SEED_K:
        dd = seed_dir(k)
        if not (dd / "tab.pkl").exists():
            return []
        models = pickle.loads((dd / "tab.pkl").read_bytes())
        for kind in ("lstm", "cnn"):
            net = seq_net(kind, meta["nf"], meta["nt"])
            net.load_state_dict(torch.load(dd / f"{kind}.pt", weights_only=True))
            net.eval()
            models[kind] = net
        out.append(models)
    return out


def train_seeds() -> None:
    """씨드 Y+1000·k로 같은 자료를 다시 학습하고, 걸어가기 예측과 같은지 확인한다."""
    import torch

    meta = pickle.loads((ST / "models.pkl").read_bytes())
    d5 = load5(BASE5)
    B, X, seq, t, seq_ok = features(d5)
    y = barrier_labels(B["d"], M.TP, M.SL)
    base = (t >= E.ts(2018)) & (t < E.ts(YEAR) - M.EMBARGO * M.BAR) & seq_ok
    Xn, Sn = norm(X, seq, meta["nz"])
    tr = np.flatnonzero(base & (y >= 0))[::3]
    te = np.flatnonzero((t >= E.ts(YEAR)) & seq_ok)
    rep = {}
    for k in SEED_K:
        dd = seed_dir(k)
        dd.mkdir(parents=True, exist_ok=True)
        seed = YEAR + 1000 * k
        tab = {}
        for name in E.MODELS:
            if name in ("rf", "mlp", "xgb"):
                tab[name] = fit_tab(name, Xn[tr], y[tr], seed)
            else:
                net = fit_seq(name, Sn, Xn, y, tr, seed)
                torch.save(net.state_dict(), dd / f"{name}.pt")
        (dd / "tab.pkl").write_bytes(pickle.dumps(tab))
        print("seed", k, "trained", flush=True)
    seeds = load_seed_models(meta)
    for k, models in zip(SEED_K, seeds):
        p = predict(models, Xn, Sn, te)
        z = np.load(SEEDS_OUT / f"pred_s{k}.npz")
        zi = np.searchsorted(z["t"], t[te])
        ok = zi < len(z["t"])
        ref = z["probs"][:, zi[ok]]
        for mi, name in enumerate(E.MODELS):
            a, b = p[mi][ok], ref[mi]
            m = np.isfinite(a) & np.isfinite(b)
            rep[f"s{k} {name}"] = round(float(np.abs(a[m] - b[m]).max()), 4)
    # 평균 규칙 문턱: 2025년 씨드 5벌 평균 점수의 분위수(원래 규칙과 같은 방식)
    z0 = np.load(PRED)
    ps = [z0["probs"]] + [np.load(SEEDS_OUT / f"pred_s{k}.npz")["probs"] for k in SEED_K]
    sc = mean_score(np.mean(ps, 0))
    yr = (z0["t"] >= E.ts(2025)) & (z0["t"] < E.ts(2026)) & np.isfinite(sc)
    thr = {
        r: (float(np.quantile(sc[yr], q)), float(np.quantile(sc[yr], 1 - q)))
        for r, (q, _) in RULES.items()
        if PFX[r] == "pa_"
    }
    rep["thr"] = thr
    (ST / "seeds_meta.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


def train() -> None:
    import torch

    ST.mkdir(parents=True, exist_ok=True)
    d5 = load5(BASE5)
    B, X, seq, t, seq_ok = features(d5)
    y = barrier_labels(B["d"], M.TP, M.SL)
    base = (t >= E.ts(2018)) & (t < E.ts(YEAR) - M.EMBARGO * M.BAR) & seq_ok
    i0 = np.flatnonzero(base)
    nz = {
        "mu": X[i0].mean(0),
        "sd": X[i0].std(0) + 1e-6,
        "smu": seq[i0].mean(0),
        "ssd": seq[i0].std(0) + 1e-6,
    }
    Xn, Sn = norm(X, seq, nz)
    tr = np.flatnonzero(base & (y >= 0))[::3]
    tab = {}
    for name in E.MODELS:
        t0 = time.time()
        if name in ("rf", "mlp", "xgb"):
            tab[name] = fit_tab(name, Xn[tr], y[tr], YEAR)
        else:
            net = fit_seq(name, Sn, Xn, y, tr, YEAR)
            torch.save(net.state_dict(), ST / f"{name}.pt")
        print(name, "train", len(tr), "sec", round(time.time() - t0, 1), flush=True)
    # 문턱: 백테스트와 같이 2025년 평균 점수의 분위수
    z = np.load(PRED)
    sc = mean_score(z["probs"])
    yr = (z["t"] >= E.ts(2025)) & (z["t"] < E.ts(2026)) & np.isfinite(sc)
    thr = {
        k: (float(np.quantile(sc[yr], q)), float(np.quantile(sc[yr], 1 - q)))
        for k, (q, _) in RULES.items()
    }
    meta = {
        "tab": tab,
        "nz": nz,
        "nf": seq.shape[1],
        "nt": X.shape[1],
        "thr": thr,
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "n_train": int(len(tr)),
    }
    (ST / "models.pkl").write_bytes(pickle.dumps(meta))
    # 확인: 2026년 백테스트 예측과 비교
    models, _ = load_models()
    te = np.flatnonzero((t >= E.ts(YEAR)) & seq_ok)
    p = predict(models, Xn, Sn, te)
    zi = np.searchsorted(z["t"], t[te])
    ok = zi < len(z["t"])
    ref = z["probs"][:, zi[ok]]
    rep = {"trained": meta["trained"], "thr": thr, "n_train": meta["n_train"]}
    for mi, name in enumerate(E.MODELS):
        a, b = p[mi][ok], ref[mi]
        m = np.isfinite(a) & np.isfinite(b)
        rep[name] = {
            "corr": round(float(np.corrcoef(a[m], b[m])[0, 1]), 4),
            "maxdiff": round(float(np.abs(a[m] - b[m]).max()), 4),
        }
    print(json.dumps(rep, indent=1), flush=True)
    (ST / "train_check.json").write_text(json.dumps(rep, indent=1))


# ---- 신호와 모의 체결 ----
def sig_path() -> Path:
    return ST / "signals.csv"


def read_signals() -> dict:
    p = sig_path()
    if not p.exists():
        return {}
    with p.open() as f:
        return {int(r["t"]): r for r in csv.DictReader(f)}


SIG_COLS = (
    ["t", "c", "score", "up", "dn", "A", "B"]
    + [f"p_{m}" for m in E.MODELS]
    + ["sa", "ua", "da", "AS", "BS"]
    + [f"pa_{m}" for m in E.MODELS]
)
LABEL = {"rf": "랜덤포레스트", "mlp": "MLP", "xgb": "XGBoost", "lstm": "LSTM", "cnn": "CNN"}


def model_probs(sigs: dict, t_sig, pfx: str = "p_") -> dict:
    """신호 봉에서 모델마다 낸 '+1.5% 먼저' 확률 (pa_ = 씨드 5벌 평균)."""
    r = sigs.get(int(t_sig), {})
    return {m: float(r[f"{pfx}{m}"]) for m in E.MODELS if r.get(f"{pfx}{m}") not in (None, "")}


def probs_text(pr: dict, side: int) -> str:
    if not pr:
        return ""
    be = M.BE if side > 0 else 1 - M.BE
    mark = (lambda v: v > be) if side > 0 else (lambda v: v < be)
    return "\n모델별 +1.5% 먼저 닿을 확률(롱은 0.533 넘으면, 숏은 0.467 밑이면 ✓): " + " · ".join(
        f"{LABEL[m]} {v:.2f}{'✓' if mark(v) else '✗'}" for m, v in pr.items()
    )


def paper(bars, sig, rule):
    """rest_sim과 같은 체결·청산. 끝까지 안 끝난 주문·포지션도 돌려준다."""
    t, o, h, l, c = (bars[k] for k in "tohlc")  # noqa: E741
    n = len(t)
    out, i, state = [], 0, None
    while i < n:
        s = sig[i]
        if not s:
            i += 1
            continue
        lim = c[i] * (1 - OFF * s)
        if i + 1 >= n:
            state = {"kind": "order", "side": s, "lim": lim, "t_sig": t[i]}
            break
        j = None
        for b in range(i + 1, min(i + 1 + REST, n)):
            if (l[b] <= lim * (1 - THRU)) if s > 0 else (h[b] >= lim * (1 + THRU)):
                j = b
                break
        if j is None:
            if i + REST >= n:  # 아직 기다리는 중
                state = {"kind": "order", "side": s, "lim": lim, "t_sig": t[i]}
                break
            out.append({"t_sig": t[i], "side": s, "lim": lim, "filled": False})
            i += 1
            continue
        px = lim
        up, dn = px * (1 + TP), px * (1 - SL)
        res, b = None, j
        while b < n:
            first = b == j
            if s > 0:
                if l[b] <= dn:
                    res = (
                        (min(o[b], dn) if not first else dn) * (1 - SLIP) / px - 1 - MAKER - TAKER,
                        0,
                    )
                elif not first and h[b] >= up * (1 + THRU):
                    res = (TP - 2 * MAKER, 1)
            else:
                if h[b] >= up:
                    res = (
                        -((max(o[b], up) if not first else up) * (1 + SLIP) / px - 1)
                        - MAKER
                        - TAKER,
                        0,
                    )
                elif not first and l[b] <= dn * (1 - THRU):
                    res = (SL - 2 * MAKER, 1)
            if res:
                break
            b += 1
        if res is None:
            state = {
                "kind": "pos",
                "side": s,
                "px": px,
                "t_sig": t[i],
                "t_fill": t[j],
                "tp": up if s > 0 else dn,
                "sl": dn if s > 0 else up,
                "last": c[-1],
            }
            break
        out.append(
            {
                "t_sig": t[i],
                "side": s,
                "lim": lim,
                "filled": True,
                "t_fill": t[j],
                "t_exit": t[b],
                "net1": res[0],
                "win": res[1],
            }
        )
        i = b
    return out, state


def kst(ms) -> str:
    return time.strftime("%m/%d %H:%M", time.gmtime(ms / 1000 + 9 * 3600))


def side_kr(s):
    return "롱" if s > 0 else "숏"


def step(notify: bool = True) -> dict:
    from minerva import notify as N

    got = fetch_new()
    d5 = load_all()
    B, X, seq, t, seq_ok = features(d5)
    models, meta = load_models()
    st_path = ST / "state.json"
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    if "start" not in state:
        state = {"start": float(t[-1]) + BAR15, "sent": []}  # 다음 봉부터 기록
    seeds = load_seed_models(meta) if "AS" in meta["thr"] else []
    if seeds and "start_s" not in state:
        state["start_s"] = float(t[-1]) + BAR15  # 평균 규칙은 추가한 다음 봉부터 기록
    sigs = read_signals()
    if sigs and (
        "pa_rf" not in next(iter(sigs.values()))
        or (seeds and any(not r.get("sa") for r in sigs.values()))
    ):
        # 칸이 모자란 옛 파일: 같은 모델로 처음부터 다시 매긴다(씨드 Y 결과는 같다)
        sig_path().replace(ST / f"signals.old{int(time.time())}.csv")
        sigs = {}
    need = np.flatnonzero((t >= state["start"]) & seq_ok)
    need = [i for i in need if int(t[i]) not in sigs]
    if need:
        Xn, Sn = norm(X, seq, meta["nz"])
        p = predict(models, Xn, Sn, need)
        pa = np.mean([p] + [predict(sm, Xn, Sn, need) for sm in seeds], 0) if seeds else None
        new = not sig_path().exists()
        with sig_path().open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(SIG_COLS)
            def flags(pp, k, rules):
                v = pp[:, k]
                sck = float(v.mean())
                up, dn = int((v > M.BE).sum()), int((v < 1 - M.BE).sum())
                out = []
                for r in rules:
                    hi, lo = meta["thr"][r]
                    kk = RULES[r][1]
                    out.append(1 if (sck >= hi and up >= kk) else (-1 if (sck <= lo and dn >= kk) else 0))
                return [round(sck, 5), up, dn], out

            for k, i in enumerate(need):
                head, fl = flags(p, k, ("A", "B"))
                row = [int(t[i]), B["d"]["c"][i], *head, *fl]
                row += [round(float(p[mi, k]), 4) for mi in range(len(E.MODELS))]
                if pa is not None:
                    head, fl = flags(pa, k, ("AS", "BS"))
                    row += [*head, *fl] + [round(float(pa[mi, k]), 4) for mi in range(len(E.MODELS))]
                else:
                    row += [""] * (5 + len(E.MODELS))
                w.writerow(row)
        sigs = read_signals()
    m = t >= state["start"]
    bars = {
        "t": t[m],
        "o": B["d"]["o"][m],
        "h": B["d"]["h"][m],
        "l": B["d"]["l"][m],
        "c": B["d"]["c"][m],
    }
    summary = {"bars": int(m.sum()), "fetched": got, "last": kst(t[-1] + BAR15) if len(t) else ""}
    msgs = []
    trades = {}
    sent = set(state["sent"])
    for r in RULES:
        s0 = state["start"] if PFX[r] == "p_" else state.get("start_s", float("inf"))
        sig = np.array(
            [int(sigs.get(int(x), {}).get(r) or 0) if x >= s0 else 0 for x in bars["t"]]
        )
        done, open_ = paper(bars, sig, r)
        trades[r] = done
        fills = [x for x in done if x["filled"]]
        net = np.array([x["net1"] for x in fills]) if fills else np.zeros(0)
        summary[r] = {
            "signals": int((sig != 0).sum()),
            "fills": len(fills),
            "wins": int(sum(x["win"] for x in fills)),
            "x1": round(float((np.prod(1 + net) - 1) * 100), 2),
            "x3": round(float((np.prod(1 + np.maximum(3 * net, -1)) - 1) * 100), 2),
            "open": open_,
        }
        for x in done:
            key = f"{r}:{int(x['t_sig'])}:{'x' if x['filled'] else 'n'}"
            if key in sent:
                continue
            sent.add(key)
            if x["filled"]:
                msgs.append(
                    f"[BTC 기록] {NAMES[r]} {side_kr(x['side'])} 끝: {kst(x['t_fill'])} {x['lim']:,.1f} 체결 → "
                    f"{kst(x['t_exit'])} {'목표 +1.5%' if x['win'] else '손절 −1.5%'}, "
                    f"수수료 뺀 1배 {x['net1'] * 100:+.2f}% / 3배 {x['net1'] * 300:+.2f}%"
                    + probs_text(model_probs(sigs, x["t_sig"], PFX[r]), x["side"])
                )
            else:
                msgs.append(
                    f"[BTC 기록] {NAMES[r]} {side_kr(x['side'])} 신호 {kst(x['t_sig'] + BAR15)}, "
                    f"{x['lim']:,.1f} 지정가 30분 안 닿아 취소"
                )
        if open_:
            key = f"{r}:{int(open_['t_sig'])}:{open_['kind']}"
            if key not in sent:
                sent.add(key)
                if open_["kind"] == "order":
                    msgs.append(
                        f"[BTC 기록] {NAMES[r]} {side_kr(open_['side'])} 신호 {kst(open_['t_sig'] + BAR15)}: "
                        f"{open_['lim']:,.1f}에 지정가 (30분 유효, 주문 안 함)"
                        + probs_text(model_probs(sigs, open_["t_sig"], PFX[r]), open_["side"])
                    )
                else:
                    msgs.append(
                        f"[BTC 기록] {NAMES[r]} {side_kr(open_['side'])} 체결 {kst(open_['t_fill'])} "
                        f"{open_['px']:,.1f}, 목표 {open_['tp']:,.1f} / 손절 {open_['sl']:,.1f}"
                        + probs_text(model_probs(sigs, open_["t_sig"], PFX[r]), open_["side"])
                    )
    state["sent"] = sorted(sent)
    st_path.write_text(json.dumps(state))
    (ST / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    write_view(
        summary,
        trades,
        sigs,
        meta,
        state,
        chart_data(
            B,
            X,
            seq,
            t,
            seq_ok,
            models,
            meta,
            trades,
            state["start"],
            {r: summary[r]["open"] for r in RULES},
        ),
    )
    if notify:
        for s in msgs:
            N.send(s)
    print(json.dumps(summary, default=float), *msgs, sep="\n", flush=True)
    return summary


BACKTEST = {  # 2026-11-04 문서의 2024-01~ 결과 (3배, 수수료 포함, 펀딩 포함)
    "A": {"per_year": 267, "win": 55.1, "x3": 236.4, "mdd": -61.6, "y2023": 7.5},
    "B": {"per_year": 142, "win": None, "x3": 60.0, "mdd": -37.0, "y2023": 64.8},
    # 씨드 5벌 평균 걸어가기, 이미 본 자료라 증거 아님 (2026-11-05 seeds 문서)
    "AS": {"per_year": 279, "win": 53.1, "x3": -13.4, "mdd": -80.4, "y2023": -7.3},
    "BS": {"per_year": 151, "win": 54.9, "x3": 85.5, "mdd": -55.3, "y2023": 21.2},
}


CHART_BARS = 288  # 3일


def chart_data(B, X, seq, t, seq_ok, models, meta, trades, start, opens=None) -> dict:
    """화면 차트용: 최근 3일 15분봉, Stoch·CCI·RSI, 모델 점수, 기록 시작 뒤 진입·청산 표시.

    시각은 KST로 옮긴 초(차트 라이브러리는 UTC로 그리므로 9시간을 더한다).
    기록 시작 전 봉의 점수는 같은 모델로 계산한 참고값이다(신호·체결 기록에는 안 쓴다).
    """
    d = B["d"]
    n = len(t)
    a = max(0, n - CHART_BARS)
    sec = lambda ms: int(ms // 1000) + 9 * 3600  # noqa: E731
    k = stoch(d["h"], d["l"], d["c"])
    cc = cci(d["h"], d["l"], d["c"])
    rs = rsi(d["c"])
    idx = np.array([i for i in range(a, n) if seq_ok[i]])
    score = {}
    if len(idx):
        Xn, Sn = norm(X, seq, meta["nz"])
        p = predict(models, Xn, Sn, idx).mean(0)
        score = {int(i): round(float(v), 4) for i, v in zip(idx, p)}
    rnd = lambda v: None if not np.isfinite(v) else round(float(v), 2)  # noqa: E731
    bars = [
        [
            sec(t[i]),
            d["o"][i],
            d["h"][i],
            d["l"][i],
            d["c"][i],
            rnd(k[i]),
            rnd(cc[i]),
            rnd(rs[i]),
            score.get(i),
        ]
        for i in range(a, n)
    ]
    marks = []
    for r, done in trades.items():
        for x in done:
            if not x["filled"]:
                continue
            marks.append(
                {
                    "rule": r,
                    "side": x["side"],
                    "kind": "open",
                    "time": sec(x["t_fill"]),
                    "price": x["lim"],
                }
            )
            marks.append(
                {
                    "rule": r,
                    "side": x["side"],
                    "kind": "close",
                    "time": sec(x["t_exit"]),
                    "win": bool(x["win"]),
                    "net3": round(x["net1"] * 300, 2),
                }
            )
    for r, o in (opens or {}).items():  # 아직 안 끝난 포지션도 진입 표시를 단다
        if o and o["kind"] == "pos":
            marks.append(
                {
                    "rule": r,
                    "side": o["side"],
                    "kind": "open",
                    "time": sec(o["t_fill"]),
                    "price": o["px"],
                }
            )
    return {"bars": bars, "marks": marks, "thr": meta["thr"], "start": sec(start)}


def write_view(summary, trades, sigs, meta, state, chart=None) -> None:
    """웹 화면(비트코인 탭)이 읽는 파일. 웹은 이 파일만 읽고 계산하지 않는다."""
    rows = [sigs[k] for k in sorted(sigs)][-96:]
    recent = [
        {
            "at": kst(int(r["t"]) + BAR15),
            "close": float(r["c"]),
            "score": float(r["score"]),
            "up": int(r["up"]),
            "dn": int(r["dn"]),
            "A": int(r["A"]),
            "B": int(r["B"]),
            "models": {m: float(r[f"p_{m}"]) for m in E.MODELS if r.get(f"p_{m}")},
        }
        for r in reversed(rows)
    ]

    def trade_row(x, pfx="p_"):
        row = {
            "signal_at": kst(x["t_sig"] + BAR15),
            "side": side_kr(x["side"]),
            "limit": x["lim"],
            "models": model_probs(sigs, x["t_sig"], pfx),
        }
        if x["filled"]:
            row.update(
                filled_at=kst(x["t_fill"]),
                exit_at=kst(x["t_exit"]),
                result="목표" if x["win"] else "손절",
                net1=round(x["net1"] * 100, 2),
                net3=round(x["net1"] * 300, 2),
            )
        else:
            row["result"] = "미체결 취소"
        return row

    def open_row(o, pfx="p_"):
        if not o:
            return None
        row = {
            "kind": "지정가 대기" if o["kind"] == "order" else "보유 중",
            "side": side_kr(o["side"]),
            "signal_at": kst(o["t_sig"] + BAR15),
            "models": model_probs(sigs, o["t_sig"], pfx),
        }
        if o["kind"] == "order":
            row["limit"] = o["lim"]
        else:
            row.update(
                filled_at=kst(o["t_fill"]),
                price=o["px"],
                target=o["tp"],
                stop=o["sl"],
                last=o["last"],
                pnl1=round((o["last"] / o["px"] - 1) * o["side"] * 100, 2),
            )
        return row

    rules = []
    for r, (q, k) in RULES.items():
        if r not in meta["thr"]:
            continue
        s = summary[r]
        rules.append(
            {
                "id": r,
                "name": NAMES[r],
                "top": round((1 - q) * 100),
                "agree": k,
                "long_above": round(meta["thr"][r][0], 4),
                "short_below": round(meta["thr"][r][1], 4),
                "signals": s["signals"],
                "fills": s["fills"],
                "wins": s["wins"],
                "x1": s["x1"],
                "x3": s["x3"],
                "open": open_row(s["open"], PFX[r]),
                "note": NOTE[r],
                "seed_avg": PFX[r] == "pa_",
                "backtest": BACKTEST[r],
                "trades": [trade_row(x, PFX[r]) for x in reversed(trades[r])][:50],
            }
        )
    view = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 9 * 3600)),
        "start": kst(state["start"]),
        "last_bar": summary["last"],
        "bars": summary["bars"],
        "be": round(M.BE, 4),
        "models": list(E.MODELS),
        "model_names": LABEL,
        "trained": meta.get("trained"),
        "rules": rules,
        "recent": recent,
        "chart": chart,
    }
    tmp = ST / "view.json.tmp"
    tmp.write_text(json.dumps(view, ensure_ascii=False, default=float))
    tmp.replace(ST / "view.json")


def report() -> None:
    s = json.loads((ST / "summary.json").read_text())
    print(json.dumps(s, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "train-seeds", "step", "report"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    {"train": train, "train-seeds": train_seeds, "step": lambda: step(not a.quiet), "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
