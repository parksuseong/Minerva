"""작은 사전 등록 ML: 구성요소 사건 봉에서 "목표 먼저"(순손익 > 0)를 LightGBM으로 예측.

문서: docs/backtests/2026-11-08-btc-tv-components-hypotheses.md 의 ML 절.
- 사건: 단일 신호 세트(C1~C10, H5L) 중 하나라도 켜진 봉 × 그 방향. 같은 봉에 두 방향이면 둘 다 사건.
- 표지: 모양 X3(1×ATR / 1.5×ATR / 96봉), X4(+0.6% / −1.0% / 96봉), X5(구조 손절 1R / 96봉)로
  그 한 거래의 순손익(수수료 포함) > 0 이면 1. 표지 창이 설계 끝(2024-07-01)을 넘는 사건은 학습에서 뺀다.
- 학습: 설계 사건만. 문턱: 설계 안 시간 순 5겹(앞뒤 1일 제외) 표본 밖 예측의 80%·90% 분위수.
  최종 모형은 설계 전부로 다시 학습. 하이퍼파라미터는 고정(아래 PARAMS), 고르지 않는다.
- 신호: 예측 ≥ 문턱인 사건 봉. 같은 봉에 두 방향 다 넘으면 버린다.
- 변형: 3 모양 × 2 문턱 = 6. 매매 청산은 표지와 같은 모양.
"""

from __future__ import annotations

import math

import numpy as np
from tvc_exits import R_MAX, R_MIN, label_one, make_native

PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=1, verbose=-1, n_jobs=2)  # fmt: skip
SHAPES = ("X3", "X4", "X5")
QS = (80, 90)
FEE = 0.0005
DAY = 86_400_000


def events(comp: dict) -> tuple:
    n = len(comp["atr"])
    js, ds = [], []
    for nm in comp["single"]:
        s = comp["sets"][nm]["sig"]
        for d in (1, -1):
            idx = np.flatnonzero(s == d)
            js.append(idx)
            ds.append(np.full(len(idx), d, np.int8))
    j = np.concatenate(js)
    d = np.concatenate(ds)
    key = j.astype(np.int64) * 2 + (d > 0)
    _, u = np.unique(key, return_index=True)
    j, d = j[u], d[u]
    o = np.argsort(j, kind="stable")
    j, d = j[o], d[o]
    ok = j < n - 2
    return j[ok], d[ok]


def features(comp: dict, j: np.ndarray, d: np.ndarray) -> np.ndarray:
    F = comp["feat"]
    cols = []
    for k in sorted(F):
        x = F[k][j]
        cols.append(x * d if k in comp["signed"] else x)
    cols.append(d.astype(float))
    return np.column_stack(cols)


def labels(b: dict, comp: dict, j: np.ndarray, d: np.ndarray, shape: str) -> np.ndarray:
    o, h, lo, c = (b[k] for k in ("o", "h", "l", "c"))
    atr, st_l, st_s = comp["atr"], comp["st_l"], comp["st_s"]
    y = np.full(len(j), np.nan)
    for i, (jj, dd) in enumerate(zip(j.tolist(), d.tolist())):
        if jj + 1 >= len(o):
            continue
        ep = o[jj + 1]
        if shape == "X3":
            tp, sl = atr[jj] / ep, 1.5 * atr[jj] / ep
        elif shape == "X4":
            tp, sl = 0.006, 0.010
        else:
            lvl = st_l[jj] if dd == 1 else st_s[jj]
            dist = (ep - lvl) * dd / ep if not math.isnan(lvl) else R_MAX
            sl = min(max(dist, R_MIN), R_MAX)
            tp = sl
        r = label_one(o, h, lo, c, jj, dd, tp, sl, 96, FEE)
        y[i] = 1.0 if r > 0 else 0.0
    return y


def build(b: dict, W: dict, comp: dict) -> tuple:
    import lightgbm as lgb

    t = b["t"]
    j, d = events(comp)
    X = features(comp, j, d)
    a, z = W["design"]
    # 표지 창(최대 97봉)이 설계 끝을 넘지 않는 사건만 학습
    tr_m = (j >= a) & (j + 98 < z)
    out, info = {}, {"events": int(len(j)), "train_events": int(tr_m.sum()),
                     "n_features": int(X.shape[1])}  # fmt: skip
    n = len(t)
    for shape in SHAPES:
        y = labels(b, comp, j, d, shape)
        m = tr_m & ~np.isnan(y)
        Xd, yd, jd = X[m], y[m], j[m]
        # 시간 순 5겹, 경계 앞뒤 1일 제외
        folds = np.array_split(np.arange(len(jd)), 5)
        oof = np.full(len(jd), np.nan)
        for f in folds:
            lo_t, hi_t = t[jd[f[0]]] - DAY, t[jd[f[-1]]] + DAY
            trn = (t[jd] < lo_t) | (t[jd] > hi_t)
            mdl = lgb.LGBMClassifier(**PARAMS).fit(Xd[trn], yd[trn])
            oof[f] = mdl.predict_proba(Xd[f])[:, 1]
        mdl = lgb.LGBMClassifier(**PARAMS).fit(Xd, yd)
        p = mdl.predict_proba(X)[:, 1]
        auc = _auc(yd, oof)
        info[shape] = {"base_rate_design": float(yd.mean()), "oof_auc": auc, "thr": {}}
        for q in QS:
            thr = float(np.quantile(oof, q / 100))
            sel = p >= thr
            sig = np.zeros(n, np.int8)
            jl = j[sel & (d == 1)]
            js_ = j[sel & (d == -1)]
            sig[jl] = 1
            both = np.intersect1d(jl, js_)
            sig[js_] = -1
            sig[both] = 0
            info[shape]["thr"][str(q)] = {
                "thr": thr, "oof_win_above": float(yd[oof >= thr].mean()),
            }  # fmt: skip
            nat = make_native(shape, comp["atr"], comp["st_l"], comp["st_s"])
            out[f"ML{shape}q{q}|{shape}"] = {"sig": sig, "native": nat}
        imp = mdl.booster_.feature_importance("gain")
        names = sorted(comp["feat"]) + ["dir"]
        top = np.argsort(imp)[::-1][:8]
        info[shape]["top_features"] = [(names[i], float(imp[i])) for i in top]
    return out, info


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
