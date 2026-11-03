"""고점·저점을 특정하는 지표 조합 찾기. 등록: 2026-11-03-btc-swing-extreme-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_swing_extreme.py --out <폴더> [--offset 15] [--pick <앞 실행 result.json>]
--pick을 주면 그 파일에서 고른 조합을 그대로 쓰고 다시 고르지 않는다(15분 이동 확인용).
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import resample, scr3, win_view
from btc_scr_litfilter import asof, bars30, load5x, load_fut, roll_mean, roll_z

M5 = 300_000
UP, DOWN = 0.01, 0.03
COST = 0.0005
BREAKEVEN = (UP + 2 * COST) / (UP + DOWN)
TRAIN0 = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
TEST0 = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
DAY = 86_400_000
MIN_EP = 100
TOP = 5
SEED = 7


def hf(d5: dict, factor: int) -> dict:
    h = resample(d5, factor)
    on_l, on_s, s, cc, r = scr3(h)
    h.update(on_l=on_l, on_s=on_s, s=s, cc=cc, r=r, close_t=h["t"] + factor * M5)
    return h


def to30(h: dict, key: str, t_close: np.ndarray, back: int = 0) -> np.ndarray:
    j = np.searchsorted(h["close_t"], t_close, side="right") - 1 - back
    v = h[key][np.maximum(j, 0)].astype(float)
    return np.where(j >= 0, v, np.nan)


def pct_b(c: np.ndarray, n: int = 20) -> np.ndarray:
    out = np.full(len(c), np.nan)
    w = win_view(c, n)
    m, sd = w.mean(1), w.std(1)
    out[n - 1 :] = (c[n - 1 :] - (m - 2 * sd)) / np.where(sd > 0, 4 * sd, np.nan)
    return out


def rmax(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[n - 1 :] = np.nanmax(win_view(x, n), 1)
    return out


def rmin(x: np.ndarray, n: int) -> np.ndarray:
    return -rmax(-x, n)


def divergence(h, l, r, side, n=48):  # noqa: E741
    if side < 0:
        return (h >= rmax(h, n) * (1 - 0.003)) & (r <= rmax(r, n) - 5)
    return (l <= rmin(l, n) * (1 + 0.003)) & (r >= rmin(r, n) + 5)


def ema(x: np.ndarray, n: int) -> np.ndarray:
    k, out, cur = 2 / (n + 1), np.empty(len(x)), x[0]
    for i, v in enumerate(x):
        cur = v * k + cur * (1 - k)
        out[i] = cur
    return out


def conditions(d5: dict, d: dict, fut: dict, train: np.ndarray) -> dict:
    """{side: {번호: bool 배열}}. side −1은 고점(숏), +1은 저점(롱)."""
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]  # noqa: E741
    tc = d["t"] + 6 * M5
    on_l, on_s, s, cc, r = scr3(d)
    h1, h4, dd = hf(d5, 12), hf(d5, 48), hf(d5, 288)
    for x in (h1, h4):
        x["pb"] = pct_b(x["c"])
    h4["e50"] = ema(h4["c"], 50)
    h4["e50p"] = np.r_[np.full(10, np.nan), h4["e50"][:-10]]
    h1_sig = {1: to30(h1, "on_l", tc) > 0, -1: to30(h1, "on_s", tc) > 0}
    h4_sig = {sd: (to30(h4, k, tc) > 0) | (to30(h4, k, tc, 1) > 0)
              for sd, k in ((1, "on_l"), (-1, "on_s"))}
    s1, r1, s4, r4 = (to30(h1, "s", tc), to30(h1, "r", tc), to30(h4, "s", tc), to30(h4, "r", tc))
    pb30, pb1, pb4 = pct_b(c), to30(h1, "pb", tc), to30(h4, "pb", tc)
    # 1시간봉 다이버전스는 1시간봉에서 계산해 옮긴다
    for sd in (1, -1):
        h1[f"dv{sd}"] = divergence(h1["h"], h1["l"], h1["r"], sd).astype(float)
    hi96, lo96 = rmax(h, 96), rmin(l, 96)
    pos = (c - lo96) / np.where(hi96 > lo96, hi96 - lo96, np.nan)
    lr = np.r_[0.0, np.diff(np.log(c))]
    r4b = np.r_[np.full(4, np.nan), np.log(c[4:] / c[:-4])]
    z4 = r4b / np.sqrt(roll_mean(lr * lr, 48 * 7) * 4)
    r48 = np.r_[np.full(48, np.nan), c[48:] / c[:-48] - 1]
    vz = roll_z(np.log(np.maximum(v, 1e-9)), 48 * 7)
    rng = np.where(h > l, h - l, np.nan)
    up_w = (h - np.maximum(o, c)) / rng
    dn_w = (np.minimum(o, c) - l) / rng
    v4 = np.convolve(v, np.ones(4), "full")[: len(v)]
    tb4 = np.convolve(d["tb"], np.ones(4), "full")[: len(v)]
    imb4 = np.where(v4 > 0, tb4 / np.where(v4 > 0, v4, 1) * 2 - 1, 0.0)
    fr = asof(tc, fut["ft"], fut["fr"])
    frp = np.full(len(fr), np.nan)
    for i in np.flatnonzero(np.isfinite(fr)):
        w = fr[max(0, i - 48 * 90) : i + 1 : 16]
        frp[i] = (w < fr[i]).mean()
    oi = asof(tc, fut["t"], fut["oi"])
    oi8 = asof(tc - 8 * 6 * M5, fut["t"], fut["oi"])
    with np.errstate(all="ignore"):
        doi = np.log(oi / np.where(oi8 > 0, oi8, np.nan))
    q = lambda x, p: np.nanquantile(x[train], p)  # noqa: E731
    imb_hi, imb_lo, doi_hi = q(imb4, 0.8), q(imb4, 0.2), q(doi, 0.8)
    dcc, drr = to30(dd, "cc", tc), to30(dd, "r", tc)
    e50, e50p = to30(h4, "e50", tc), to30(h4, "e50p", tc)
    hi48, lo48 = rmax(h, 48), rmin(l, 48)
    out = {}
    with np.errstate(invalid="ignore"):
        for sd in (-1, 1):
            top = sd < 0
            ge = (lambda x, a, b: x >= a) if top else (lambda x, a, b: x <= b)  # noqa: E731
            cond = {
                1: ge(s, 80, 20), 2: ge(cc, 100, -100), 3: ge(r, 70, 30),
                4: on_s if top else on_l, 5: h1_sig[sd], 6: h4_sig[sd],
                7: ge(s1, 80, 20), 8: ge(r1, 70, 30), 9: ge(s4, 80, 20), 10: ge(r4, 65, 35),
                11: divergence(h, l, r, sd), 12: to30(h1, f"dv{sd}", tc) > 0,
                13: ge(pb30, 1, 0), 14: ge(pb1, 1, 0), 15: ge(pb4, 1, 0),
                16: ge(pos, 0.9, 0.1), 17: (h >= hi96) if top else (l <= lo96),
                18: ge(z4, 2, -2), 19: ge(r48, 0.03, -0.03), 20: vz >= 2,
                21: (up_w if top else dn_w) >= 0.5,
                22: ((c < o) & (h >= hi48)) if top else ((c > o) & (l <= lo48)),
                23: (imb4 >= imb_hi) if top else (imb4 <= imb_lo),
                24: ge(frp, 0.8, 0.2), 25: doi >= doi_hi,
                26: ((dcc < 0) & (drr < 50)) if top else ((dcc > 0) & (drr > 50)),
                27: (e50 < e50p) if top else (e50 > e50p),
                28: ((dcc > 0) & (drr > 50)) if top else ((dcc < 0) & (drr < 50)),
            }
            out[sd] = {k: np.nan_to_num(np.asarray(m, dtype=float)) > 0 for k, m in cond.items()}
    return out


def labels(d: dict, side: int) -> np.ndarray:
    """다음 시가 진입 뒤 유리한 3%가 불리한 1%보다 먼저면 1, 아니면 0, 끝까지 모르면 −1."""
    o, h, l = d["o"], d["h"], d["l"]  # noqa: E741
    n = len(o)
    out = np.full(n, -1, dtype=np.int8)
    for i in range(n - 1):
        px = o[i + 1]
        if side < 0:
            good, bad = px * (1 - DOWN), px * (1 + UP)
        else:
            good, bad = px * (1 + DOWN), px * (1 - UP)
        for j in range(i + 1, n):
            b = h[j] >= bad if side < 0 else l[j] <= bad
            g = l[j] <= good if side < 0 else h[j] >= good
            if b:
                out[i] = 0
                break
            if g:
                out[i] = 1
                break
    return out


def starts(m: np.ndarray) -> np.ndarray:
    return m & ~np.r_[False, m[:-1]]


def wilson_lo(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def precision(st: np.ndarray, lab: np.ndarray, mask: np.ndarray) -> tuple:
    sel = st & mask & (lab >= 0)
    n = int(sel.sum())
    k = int(lab[sel].sum())
    return k, n


def trade(d: dict, sig_s: np.ndarray, sig_l: np.ndarray, stop: float, i0: int) -> list:
    """반대 신호에서 뒤집기, 손절. 신호 봉 마감 → 다음 시가."""
    o, h, l, t = d["o"], d["h"], d["l"], d["t"]  # noqa: E741
    tr, side, px, ei, mae = [], 0, 0.0, 0, 0.0
    for j in range(i0 + 1, len(o)):
        want = -1 if sig_s[j - 1] else (1 if sig_l[j - 1] else 0)
        if side and want == -side:
            g = (o[j] / px - 1) * side
            tr.append((float(t[ei]), g, max(mae, -g), side))
            side = 0
        if not side and want:
            side, px, ei, mae = want, o[j], j, 0.0
        if side:
            line = px * (1 - stop * side)
            if l[j] <= line if side > 0 else h[j] >= line:
                fill = min(o[j], line) if side > 0 else max(o[j], line)
                g = (fill / px - 1) * side
                tr.append((float(t[ei]), g, max(mae, -g), side))
                side = 0
            else:
                mae = max(mae, (px - l[j]) / px if side > 0 else (h[j] - px) / px)
    return tr


def summ(ts: list) -> dict:
    if not ts:
        return {"n": 0}
    g = np.array([x[1] for x in ts])
    mae = np.array([x[2] for x in ts])
    net1 = g - 2 * COST
    net3 = np.where(mae >= 0.32, -1.0, 3 * g - 6 * COST)
    wins, losses = net1[net1 > 0.001], net1[net1 <= 0.001]
    out = {"n": len(ts), "win": round(float((net1 > 0.001).mean() * 100), 1),
           "ev": round(float(net1.mean() * 100), 3),
           "avg_win": round(float(wins.mean() * 100), 2) if len(wins) else None,
           "avg_loss": round(float(losses.mean() * 100), 2) if len(losses) else None,
           "worst": round(float(net1.min() * 100), 2),
           "long": int(sum(1 for x in ts if x[3] > 0))}
    for k, net in (("x1", net1), ("x3", net3)):
        eq = np.cumprod(1 + np.maximum(net, -1.0))
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        out[k] = round(float((eq[-1] - 1) * 100), 1)
        out[k + "_mdd"] = round(float((eq / peak[1:] - 1).min() * 100), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pick")
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5x(root / "5m.csv")
    d = bars30(d5, a.offset)
    t = d["t"]
    train = (t >= TRAIN0) & (t < TEST0)
    test = t >= TEST0
    conds = conditions(d5, d, load_fut(root), train)
    labs = {sd: labels(d, sd) for sd in (-1, 1)}
    rng = np.random.default_rng(SEED)
    rep: dict = {"offset": a.offset, "breakeven": BREAKEVEN, "base": {}}
    for sd in (-1, 1):
        rep["base"][str(sd)] = {p: precision(np.ones(len(t), bool), labs[sd], m)
                                for p, m in (("train", train), ("test", test))}
    picked = json.loads(Path(a.pick).read_text())["picked"] if a.pick else None
    rep["picked"] = {}
    for sd in (-1, 1):
        cs = conds[sd]
        if picked is None:
            rows = []
            for k in (1, 2, 3):
                for combo in itertools.combinations(sorted(cs), k):
                    m = cs[combo[0]].copy()
                    for x in combo[1:]:
                        m &= cs[x]
                    hit, n = precision(starts(m), labs[sd], train)
                    if n >= MIN_EP:
                        rows.append((wilson_lo(hit, n), hit / n, n, list(combo)))
            rows.sort(key=lambda r: -r[0])
            rep[f"searched{sd}"] = len(rows)
            rep[f"top20_train{sd}"] = rows[:20]
            chosen = [r[3] for r in rows[:TOP]]
        else:
            chosen = picked[str(sd)]
        rep["picked"][str(sd)] = chosen
        res = []
        for combo in chosen:
            m = cs[combo[0]].copy()
            for x in combo[1:]:
                m &= cs[x]
            st = starts(m)
            ktr, ntr = precision(st, labs[sd], train)
            kte, nte = precision(st, labs[sd], test)
            pool = np.flatnonzero(test & (labs[sd] >= 0))
            draws = np.array([labs[sd][rng.choice(pool, nte, replace=False)].mean()
                              for _ in range(1000)]) if nte else np.array([1.0])
            res.append({"combo": combo, "train": [ktr, ntr, round(ktr / max(ntr, 1) * 100, 1)],
                        "test": [kte, nte, round(kte / max(nte, 1) * 100, 1)],
                        "p_random": float((draws >= kte / max(nte, 1)).mean())})
        rep[f"eval{sd}"] = res
    # 매매: 쪽마다 고른 조합 × 반대쪽 1위 조합
    i0 = int(np.searchsorted(t, TEST0))
    end = t[-1] + 6 * M5
    cuts = {"test": TEST0, "1y": end - 365 * DAY, "6m": end - 183 * DAY, "30d": end - 30 * DAY}
    mk = {}
    for sd in (-1, 1):
        for combo in rep["picked"][str(sd)]:
            m = conds[sd][combo[0]].copy()
            for x in combo[1:]:
                m &= conds[sd][x]
            mk[(sd, tuple(combo))] = starts(m)
    best = {sd: tuple(rep["picked"][str(sd)][0]) for sd in (-1, 1)}
    rep["trades"] = {}
    for sd in (-1, 1):
        for combo in rep["picked"][str(sd)]:
            mine, other = mk[(sd, tuple(combo))], mk[(-sd, best[-sd])]
            sig_s, sig_l = (mine, other) if sd < 0 else (other, mine)
            for stop in (0.01, 0.03):
                tr = trade(d, sig_s, sig_l, stop, i0)
                key = f"{'S' if sd < 0 else 'L'}{combo}|SL{stop * 100:g}%"
                rep["trades"][key] = {k: summ([z for z in tr if z[0] >= c]) for k, c in cuts.items()}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    print("base", rep["base"], "breakeven", round(BREAKEVEN * 100, 1))
    for sd in (-1, 1):
        print("side", sd, "searched", rep.get(f"searched{sd}"))
        for e in rep[f"eval{sd}"]:
            print(" ", e)
    for k, v in rep["trades"].items():
        print(k, {p: (v[p].get("n"), v[p].get("win"), v[p].get("ev"), v[p].get("x1")) for p in cuts})


if __name__ == "__main__":
    main()
