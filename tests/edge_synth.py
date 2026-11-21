"""연구 특징 비교 시험용 가짜 1시간봉(고정 난수). fixtures/edge_features_parity.npz는 이 자료에
연구 코드(8차 build.symbol_feats)를 돌린 값이다. 자료를 바꾸면 그 파일도 다시 만들어야 한다."""

import numpy as np

NH = 1800  # 75일: 일봉 50일선·상장 60일 유효 조건이 다 걸리게


def synth(name: str):
    rng = np.random.default_rng({"A": 11, "B": 12}[name])
    r = rng.normal(0, 0.01, NH)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, NH)))
    low = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, NH)))
    q = rng.lognormal(15, 0.5, NH)
    if name == "B":  # 늦게 상장 + 중간 빈 시간
        for a in (o, h, low, c, q):
            a[:30] = np.nan
            a[900:905] = np.nan
    fh = np.arange(0, NH, 8, dtype=np.int64)
    fr = rng.normal(1e-4, 2e-4, len(fh))
    return o, h, low, c, q, fh, fr
