"""147번: 블로그 비트코인 단타. SCR 진입(V1 추세 1단, V3 3단) + SMA20 회귀 익절, 1% 손절.
등록: 2026-11-07-btc-blog-scalp-ma20-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_blog_scalp_ma20.py \
    --out <폴더> [--perm 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from btc_fractal import cci, crossed, load5, resample, rsi, scr3, stoch
from btc_fut_blog_combo import dmi, sma
from btc_scr_gate15 import load_funding
from btc_scr_learned import ystart

M5 = 300_000
DAY = 86_400_000
STOP = 0.01
TAKER, MAKER, SLIP, THRU = 0.0005, 0.0002, 0.0005, 0.0005
FRAMES = {"15m": (3, (0, 5, 10)), "1h": (12, (0, 20, 40))}


def frame_bars(d5: dict, k: int, off_min: int) -> dict:
    sh = dict(d5)
    sh["t"] = d5["t"] - off_min * 60_000
    d = resample(sh, k)
    d["t"] = d["t"] + off_min * 60_000
    return d


def gates(d: dict) -> dict:
    """변형별 (롱 트리거, 숏 트리거, 롱 문, 숏 문). 문은 트리거를 뺀 조건."""
    h, l, c = d["h"], d["l"], d["c"]
    s = stoch(h, l, c, 5)
    m = {n: sma(c, n) for n in (5, 10, 20, 60, 120)}
    pdi, mdi, _ = dmi(d)
    t3l, t3s, *_ = scr3(d)
    with np.errstate(invalid="ignore"):
        below, above = c < m[20], c > m[20]
        short_hi = np.maximum(np.maximum(m[5], m[10]), m[20])
        short_lo = np.minimum(np.minimum(m[5], m[10]), m[20])
        rev = short_hi < np.minimum(m[60], m[120])
        fwd = short_lo > np.maximum(m[60], m[120])
        g1l = (c > m[60]) & (pdi > mdi) & below
        g1s = (c < m[60]) & (mdi > pdi) & above
        g3l = ~rev & below & np.isfinite(m[120])
        g3s = ~fwd & above & np.isfinite(m[120])
    return {"V1": (crossed(s, 20, True) & g1l, crossed(s, 80, False) & g1s, g1l, g1s),
            "V3": (t3l & g3l, t3s & g3s, g3l, g3s), "m20": m[20]}


class Sim:
    """5분봉 체결. 프레임 봉 i 마감 5분 뒤 진입, 마지막 마감 프레임 봉의 SMA20 지정가 익절, 1% 손절."""

    def __init__(self, d5, d, k, m20, fund):
        self.d5, self.fund = d5, fund
        span = k * M5
        t5 = d5["t"]
        f = np.searchsorted(d["t"] + span, t5, side="right") - 1  # 5분봉 시작 시점에 마감된 프레임 봉
        self.lvl = np.where(f >= 0, m20[np.clip(f, 0, None)], np.nan)
        self.entry = np.searchsorted(t5, d["t"] + span + M5)  # 신호 봉 마감 + 5분

    def run(self, i: int, side: int):
        o, h, l, t = self.d5["o"], self.d5["h"], self.d5["l"], self.d5["t"]
        e = int(self.entry[i])
        n = len(o)
        if e >= n:
            return None
        px = o[e]
        sp = px * (1 - STOP * side)
        lv = self.lvl
        for j in range(e, n):
            if side > 0 and l[j] <= sp:
                fill, fee = min(o[j], sp) * (1 - SLIP), TAKER
            elif side < 0 and h[j] >= sp:
                fill, fee = max(o[j], sp) * (1 + SLIP), TAKER
            else:
                L = lv[j]
                if not np.isfinite(L):
                    continue
                if side > 0 and o[j] >= L:
                    fill, fee = o[j], TAKER
                elif side < 0 and o[j] <= L:
                    fill, fee = o[j], TAKER
                elif side > 0 and h[j] >= L * (1 + THRU):
                    fill, fee = L, MAKER
                elif side < 0 and l[j] <= L * (1 - THRU):
                    fill, fee = L, MAKER
                else:
                    continue
                g = (fill / px - 1) * side - TAKER - fee
                return self._fund(g, e, j, side), j, False
            g = (fill / px - 1) * side - TAKER - fee
            return self._fund(g, e, j, side), j, True
        return None

    def _fund(self, g, e, x, side):
        if self.fund is None:
            return g
        t = self.d5["t"]
        m = (self.fund[0] > t[e]) & (self.fund[0] <= t[x])
        return g - side * float(self.fund[1][m].sum())


def summ(tr: list, extra: float = 0.0) -> dict:
    if not tr:
        return {"n": 0}
    net = np.array([z[2] for z in tr]) - 2 * extra
    eq3 = np.cumprod(1 + np.maximum(3 * net, -1))
    pk3 = np.maximum.accumulate(np.r_[1.0, eq3])
    s = np.sort(net)
    w = net[net > 0]
    return {"n": len(tr), "win": round(float((net > 0).mean() * 100), 1), "ev": round(float(net.mean() * 100), 3),
            "avg_win": round(float(w.mean() * 100), 3) if len(w) else None,
            "avg_loss": round(float(net[net <= 0].mean() * 100), 3) if (net <= 0).any() else None,
            "ret3": round(float((eq3[-1] - 1) * 100), 1), "mdd3": round(float((eq3 / pk3[1:] - 1).min() * 100), 1),
            "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None,
            "stopped": round(float(np.mean([z[4] for z in tr]) * 100), 1),
            "long": int(sum(1 for z in tr if z[1] > 0)), "short": int(sum(1 for z in tr if z[1] < 0))}


def walk(d, sim, lon, son, first):
    sigs = sorted([(int(i), 1) for i in np.flatnonzero(lon)] + [(int(i), -1) for i in np.flatnonzero(son)])
    busy, tr = -1, []
    for i, side in sigs:
        if i < first or sim.entry[i] <= busy:
            continue
        r = sim.run(i, side)
        if r is None:
            continue
        g, x, stopped = r
        tr.append((i, side, float(g), int(x), bool(stopped), float(d["t"][i])))
        busy = x
    return tr


def report(tr: list) -> dict:
    y24 = ystart(2024)
    old, new = [z for z in tr if z[5] < y24], [z for z in tr if z[5] >= y24]
    return {"all": summ(tr), "old": summ(old), "new": summ(new), "cost2": summ(tr, 0.0005),
            "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
            "new_long": summ([z for z in new if z[1] > 0]), "new_short": summ([z for z in new if z[1] < 0])}


def null_test(d, sim, gl, gs, tr, first, draws, seed=7):
    """문 조건 봉에서 구간·방향별 같은 수를 뽑아 같은 청산으로 거래당 평균 비교."""
    y24 = ystart(2024)
    t = d["t"]
    rng = np.random.default_rng(seed)
    pools, need = {}, {}
    for per, msk in (("old", t < y24), ("new", t >= y24)):
        for side, g in ((1, gl), (-1, gs)):
            idx = np.flatnonzero(g & msk & (np.arange(len(t)) >= first))
            cnt = sum(1 for z in tr if z[1] == side and ((z[5] < y24) == (per == "old")))
            if cnt == 0:
                continue
            sub = rng.choice(idx, size=min(len(idx), 6000), replace=False) if len(idx) > 6000 else idx
            vals = []
            for i in sub:
                r = sim.run(int(i), side)
                if r is not None:
                    vals.append((r[0], r[0] > 0))
            pools[(per, side)] = np.array(vals)
            need[(per, side)] = cnt
    actual = float(np.mean([z[2] for z in tr]))
    tot = sum(need.values())
    means, wins = np.empty(draws), np.empty(draws)
    for b in range(draws):
        sm = wn = 0.0
        for key, cnt in need.items():
            v = pools[key]
            pick = v[rng.integers(0, len(v), size=cnt)]
            sm += pick[:, 0].sum()
            wn += pick[:, 1].sum()
        means[b], wins[b] = sm / tot, wn / tot
    return {"p": round(float((means >= actual).mean()), 4), "null_ev": round(float(means.mean() * 100), 3),
            "null_win": round(float(wins.mean() * 100), 1), "pool": {f"{k[0]}{k[1]:+d}": len(v) for k, v in pools.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    rep: dict = {}
    for fname, (k, offs) in FRAMES.items():
        for off in offs:
            d = frame_bars(d5, k, off)
            first = int(np.searchsorted(d["t"], d["t"][0] + 60 * DAY))
            G = gates(d)
            sim = Sim(d5, d, k, G["m20"], fund)
            for v in ("V1", "V3"):
                lon, son, gl, gs = G[v]
                tr = walk(d, sim, lon, son, first)
                key = f"{v}_{fname}_off{off}"
                r = report(tr)
                if off == 0:
                    r["null"] = null_test(d, sim, gl, gs, tr, first, args.perm)
                    r["trades"] = [[z[0], z[1], round(z[2], 5), z[3], z[4]] for z in tr]
                rep[key] = r
                print(key, json.dumps(r["all"]), r.get("null"), flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
