"""148번: 블로그 "모든 지표 상승" 8개 동시 점등(C8)과 에너지 응축(C8Q). 청산 E1(+0.5%/−1%), E2(스토캐 꺾임).
등록: 2026-11-07-btc-blog-confluence-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_blog_confluence.py \
    --out <폴더> [--perm 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from btc_blog_scalp_ma20 import FRAMES, M5, MAKER, SLIP, STOP, TAKER, THRU, frame_bars, null_test, report, walk
from btc_fractal import cci, crossed, load5, rsi, stoch, win_view
from btc_fut_blog_combo import dmi, ema, sma
from btc_scr_gate15 import load_funding

DAY = 86_400_000
TARGET = 0.005


def roll_sum(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    out[n - 1:] = c[n:] - c[:-n]
    return out


def signals(d: dict, k: int) -> dict:
    h, l, c, v = d["h"], d["l"], d["c"], d["v"]
    s, cc, r = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    pdi, mdi, _ = dmi(d)
    tp = (h + l + c) / 3
    mf = tp * v
    dtp = np.r_[0.0, np.diff(tp)]
    pos, neg = roll_sum(np.where(dtp > 0, mf, 0.0), 14), roll_sum(np.where(dtp < 0, mf, 0.0), 14)
    with np.errstate(invalid="ignore", divide="ignore"):
        mfi = 100 * pos / (pos + neg)
    obv = np.cumsum(np.sign(np.r_[0.0, np.diff(c)]) * v)
    obv_m, vol_m = sma(obv, 20), sma(v, 20)
    m20 = sma(c, 20)
    sd = np.full(len(c), np.nan)
    w = win_view(c, 20)
    sd[19:] = w.std(axis=1)
    bw = 4 * sd / m20
    W = 30 * 288 // k
    q = np.full(len(c), np.nan)
    for i in range(W + 20, len(c)):  # 직전 봉까지 30일의 하위 20%
        q[i] = np.nanpercentile(bw[i - W:i], 20)
    with np.errstate(invalid="ignore"):
        allL = (s > 50) & (cc > 0) & (r > 50) & (macd > sig) & (pdi > mdi) & (mfi > 50) & (obv > obv_m) & (v > vol_m)
        allS = (s < 50) & (cc < 0) & (r < 50) & (macd < sig) & (mdi > pdi) & (mfi < 50) & (obv < obv_m) & (v > vol_m)
        squeeze = np.r_[False, bw[:-1] <= q[1:]]
    c8l = allL & ~np.r_[False, allL[:-1]]
    c8s = allS & ~np.r_[False, allS[:-1]]
    return {"C8": (c8l, c8s), "C8Q": (c8l & squeeze, c8s & squeeze),
            "exL": crossed(s, 80, False), "exS": crossed(s, 20, True)}


class Sim2:
    def __init__(self, d5, d, k, fund, mode, exL, exS):
        self.d5, self.fund, self.mode, self.dc = d5, fund, mode, d["c"]
        span = k * M5
        self.entry = np.searchsorted(d5["t"], d["t"] + span + M5)
        n5 = len(d5["t"])
        self.exec = {}
        for side, ex in ((1, exL), (-1, exS)):
            a = np.full(n5, -1)
            idx = np.flatnonzero(ex)
            e = self.entry[idx]
            ok = e < n5
            a[e[ok]] = idx[ok]
            self.exec[side] = a

    def run(self, i: int, side: int):
        o, h, l = self.d5["o"], self.d5["h"], self.d5["l"]
        e = int(self.entry[i])
        n = len(o)
        if e >= n:
            return None
        px = o[e]
        sp = px * (1 - STOP * side)
        tg = px * (1 + TARGET * side)
        ex = self.exec[side]
        for j in range(e, n):
            if self.mode == "E2" and j > e and ex[j] >= 0 and (self.dc[ex[j]] - px) * side > 0:
                return self._done((o[j] / px - 1) * side - 2 * TAKER, e, j, side, False)
            if side > 0 and l[j] <= sp:
                return self._done((min(o[j], sp) * (1 - SLIP) / px - 1) - 2 * TAKER, e, j, side, True)
            if side < 0 and h[j] >= sp:
                return self._done(-(max(o[j], sp) * (1 + SLIP) / px - 1) - 2 * TAKER, e, j, side, True)
            if self.mode == "E1":
                if (o[j] - tg) * side >= 0 and j > e:
                    return self._done((o[j] / px - 1) * side - 2 * TAKER, e, j, side, False)
                if (side > 0 and h[j] >= tg * (1 + THRU)) or (side < 0 and l[j] <= tg * (1 - THRU)):
                    return self._done(TARGET - TAKER - MAKER, e, j, side, False)
        return None

    def _done(self, g, e, x, side, stopped):
        if self.fund is not None:
            t = self.d5["t"]
            m = (self.fund[0] > t[e]) & (self.fund[0] <= t[x])
            g -= side * float(self.fund[1][m].sum())
        return g, x, stopped


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
            S = signals(d, k)
            allm = np.ones(len(d["c"]), bool)
            for mode in ("E1", "E2"):
                sim = Sim2(d5, d, k, fund, mode, S["exL"], S["exS"])
                for v in ("C8", "C8Q"):
                    lon, son = S[v]
                    tr = walk(d, sim, lon, son, first)
                    key = f"{v}_{mode}_{fname}_off{off}"
                    r = report(tr)
                    if off == 0:
                        r["null"] = null_test(d, sim, allm, allm, tr, first, args.perm)
                        r["trades"] = [[z[0], z[1], round(z[2], 5), z[3], z[4]] for z in tr]
                    rep[key] = r
                    print(key, json.dumps(r["all"]), r.get("null"), flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
