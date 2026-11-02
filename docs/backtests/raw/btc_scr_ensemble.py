"""SCR 3단 승리확률 앙상블. 등록: 2026-11-02-btc-scr-ensemble-hypotheses.md

btc_scr_model의 표본과 특징을 그대로 쓴다.
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_ensemble.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_scr_model as M
from btc_fractal import load5
from btc_scr_ma import fit_logit

# 특징 열 위치(btc_scr_model.FEATS)
H4_SAME, H4_AGAINST, D1_SAME, D1_AGAINST = (M.FEATS.index(x) for x in
                                             ("4h 같은방향", "4h 반대", "일봉 같은방향", "일봉 반대"))


def naive_bayes(xtr, ytr, xq, bins=5):
    """연속 특징은 학습 분위로 나누고, 특징별 승률 로그 비를 더한다(라플라스 보정)."""
    prior = ytr.mean()
    score = np.full(len(xq), np.log(prior / (1 - prior)))
    for j in range(xtr.shape[1]):
        col = xtr[:, j]
        if len(np.unique(col)) <= 2:
            edges = np.array([0.5])
        else:
            edges = np.unique(np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1]))
        btr, bq = np.digitize(col, edges), np.digitize(xq[:, j], edges)
        for b in np.unique(btr):
            k = btr == b
            pb = (ytr[k].sum() + 1) / (k.sum() + 2)
            score[bq == b] += np.log(pb / (1 - pb)) - np.log(prior / (1 - prior))
    return 1 / (1 + np.exp(-score))


def stumps(xtr, ytr, xq, rounds=150, lr=0.1):
    """깊이 1 나무 부스팅(로지스틱 손실). 분위 9개 후보 경계."""
    f_tr = np.full(len(ytr), np.log(ytr.mean() / (1 - ytr.mean())))
    f_q = np.full(len(xq), f_tr[0])
    cuts = [np.unique(np.quantile(xtr[:, j], np.linspace(0.1, 0.9, 9))) for j in range(xtr.shape[1])]
    for _ in range(rounds):
        p = 1 / (1 + np.exp(-f_tr))
        grad = ytr - p
        hess = p * (1 - p) + 1e-6
        best = None
        for j in range(xtr.shape[1]):
            for c in cuts[j]:
                left = xtr[:, j] <= c
                gl, hl = grad[left].sum(), hess[left].sum()
                gr, hr = grad[~left].sum(), hess[~left].sum()
                gain = gl * gl / (hl + 1) + gr * gr / (hr + 1)
                if best is None or gain > best[0]:
                    best = (gain, j, c, gl / (hl + 1), gr / (hr + 1))
        _, j, c, vl, vr = best
        f_tr += lr * np.where(xtr[:, j] <= c, vl, vr)
        f_q += lr * np.where(xq[:, j] <= c, vl, vr)
    return 1 / (1 + np.exp(-f_q))


def members(xtr, ytr, xq) -> dict:
    out = {}
    for lam in (3.0, 30.0, 300.0):
        out[f"logit{int(lam)}"] = M.predict_logit(fit_logit(xtr, ytr, lam=lam), xq)
    for k in (30, 60, 120):
        out[f"knn{k}"] = M.knn(xtr, ytr, xq, k=k)
    out["nb"] = naive_bayes(xtr, ytr, xq)
    out["gbm"] = stumps(xtr, ytr, xq)
    return out


def pct_of_train(p_tr: np.ndarray, p_q: np.ndarray) -> np.ndarray:
    """학습 구간 예측 분포에서의 백분위. 시험 신호끼리 순위를 매기지 않아 미래를 보지 않는다."""
    srt = np.sort(p_tr)
    return np.searchsorted(srt, p_q, side="right") / len(srt)


def both(xtr, ytr, xq) -> dict:
    allp = members(xtr, ytr, np.vstack([xtr, xq]))
    return {k: (v[:len(xtr)], v[len(xtr):]) for k, v in allp.items()}


def ensemble(xtr, ytr, xq, regime_tr, regime_q) -> tuple:
    """전체 모델 + 큰 봉 국면(일봉·4h가 같은 방향 / 그 밖)별 전용 모델의 백분위 평균."""
    m_all = both(xtr, ytr, xq)
    scores = [pct_of_train(a, b) for a, b in m_all.values()]
    reg = {}
    for g in (True, False):
        kt, kq = regime_tr == g, regime_q == g
        if kt.sum() < 80 or kq.sum() == 0:
            continue
        for name, (a, b) in both(xtr[kt], ytr[kt], xq[kq]).items():
            reg.setdefault(name, np.full(len(xq), np.nan))[kq] = pct_of_train(a, b)
    for v in reg.values():
        if np.isfinite(v).all():
            scores.append(v)
    return np.mean(scores, axis=0), {k: b for k, (a, b) in m_all.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    d, X, y, g, meta = M.build(d5, args.offset)
    t = np.array([m[2] for m in meta])
    tr, va, te = t < M.TRAIN_END, (t >= M.TRAIN_END) & (t < M.VAL_END), t >= M.VAL_END
    trva = tr | va
    aligned = (X[:, H4_SAME] > 0) & (X[:, D1_SAME] > 0)
    against = (X[:, H4_AGAINST] > 0) | (X[:, D1_AGAINST] > 0)
    rep: dict = {"offset": args.offset, "n": [int(tr.sum()), int(va.sum()), int(te.sum())]}
    # 검증: 학습만으로 맞추고 2023에서 각 모델·앙상블 AUC, 문턱(순위 분위) 고르기
    s_va, m_va = ensemble(X[tr], y[tr], X[va], aligned[tr], aligned[va])
    rep["val_auc"] = {k: round(M.auc(y[va], v), 3) for k, v in m_va.items()}
    rep["val_auc"]["ENS"] = round(M.auc(y[va], s_va), 3)
    sv = np.zeros(len(y))
    sv[va] = s_va
    thr = None
    for gate in ("none", "no_against"):
        mask = va if gate == "none" else (va & ~against)
        for q in np.arange(0.0, 0.81, 0.1):
            st, _ = M.trade(meta, g, sv, mask, q, M.COST[0])
            if st["n"] >= 30 and (thr is None or st["ev"] > thr[2]):
                thr = (gate, round(float(q), 1), st["ev"], st["n"])
    rep["chosen"] = thr
    # 시험: 학습+검증으로 다시 맞춤
    s_te, m_te = ensemble(X[trva], y[trva], X[te], aligned[trva], aligned[te])
    rep["test_auc"] = {k: round(M.auc(y[te], v), 3) for k, v in m_te.items()}
    rep["test_auc"]["ENS"] = round(M.auc(y[te], s_te), 3)
    rep["test_auc_ens_dayboot"] = M.day_boot(y[te], s_te, t[te])
    s = np.zeros(len(y))
    s[te] = s_te
    rep["filter_only_test"] = M.trade(meta, g, np.ones(len(y)), te & ~against, 0.0, M.COST[0])[0]
    rep["scr_only_test"] = M.trade(meta, g, np.ones(len(y)), te, 0.0, M.COST[0])[0]
    rep["quintile_test"] = []
    qs = np.quantile(s_te, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(s_te, qs)
    for k in range(5):
        rep["quintile_test"].append([int((b == k).sum()), round(float(y[te][b == k].mean()) * 100, 1)])
    if thr:
        mask = te if thr[0] == "none" else (te & ~against)
        rep["ens_test"], sel = M.trade(meta, g, s, mask, thr[1], M.COST[0])
        rep["ens_test_c10"] = M.trade(meta, g, s, mask, thr[1], M.COST[1])[0]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
