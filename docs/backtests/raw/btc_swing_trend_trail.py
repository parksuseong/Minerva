"""123번: R1 일봉 방향 안에서 확정 HL(LH) 진입, 새 HL(LH)마다 손절을 끌어올린다.
등록: 2026-11-07-btc-swing-trend-trail-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_swing_trend_trail.py \
    --out <폴더> [--day-shift 8] [--circ 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import btc_fut_daily_regime as R
from btc_fractal import load5, resample, rsi
from btc_fut_rsi_swing import bd_signals
from btc_scr_entry_stop import summ, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, ystart

DAY = 86_400_000
BAR = 900_000
STOP = 0.01
BUF = 0.001


def pivots(d: dict, frame: int) -> list:
    """확정 피벗을 15분봉 인덱스로: (알게 되는 15분봉, 종류 +1=저점/−1=고점, 가격, 직전 같은 종류보다 높음?)."""
    u = d if frame == 1 else resample(d, frame, base_ms=BAR)
    sig = bd_signals(u["h"], u["l"], rsi(u["c"], 7))
    known = u["t"] + frame * BAR  # 그 봉 마감 시각
    j = np.searchsorted(d["t"] + BAR, known, side="left")  # 마감 시각이 같은 15분봉
    out, prev = [], {1: None, -1: None}
    for (i, s, px, _), jj in zip(sig, j[[z[0] for z in sig]] if sig else []):
        kind = 1 if s > 0 else -1  # s=+1: 저점 확정, s=−1: 고점 확정
        higher = prev[kind] is not None and px > prev[kind]
        lower = prev[kind] is not None and px < prev[kind]
        prev[kind] = px
        out.append((int(jj), kind, float(px), higher, lower))
    return out


def run(d: dict, reg: np.ndarray, piv: list, start: int, fund, cost_side: float = 0.0005) -> list:
    """한 번에 포지션 하나. 진입은 HL(롱)/LH(숏) 확정 봉의 다음 시가."""
    o, h, l, t = d["o"], d["h"], d["l"], d["t"]
    n = len(o)
    by_bar: dict = {}
    for jj, kind, px, hi, lo in piv:
        by_bar.setdefault(jj, []).append((kind, px, hi, lo))
    tr = []
    pos = 0
    for i in range(start, n - 2):
        if pos == 0:
            for kind, px, hi, lo in by_bar.get(i, []):
                side = 1 if (kind == 1 and hi) else (-1 if (kind == -1 and lo) else 0)
                if side and reg[i] == side:
                    e = i + 1
                    entry = o[e]
                    sp = px * (1 - BUF) if side > 0 else px * (1 + BUF)
                    lim = entry * (1 - STOP * side)
                    sp = max(sp, lim) if side > 0 else min(sp, lim)
                    if (side > 0 and sp >= entry) or (side < 0 and sp <= entry):
                        continue  # 이미 피벗 밖에서 시작: 진입 안 함
                    pos, ent_i, ent_px, stop = side, i, entry, sp
                    break
            continue
        # 보유 중: 봉 i에서 손절 확인 → 마감 후 방향/새 피벗 반영
        side = pos
        hit = (side > 0 and l[i] <= stop) or (side < 0 and h[i] >= stop)
        if hit and i >= ent_i + 1:
            fill = min(o[i], stop) if side > 0 else max(o[i], stop)
            g = (fill / ent_px - 1) * side
            tr.append((ent_i, side, float(with_funding(g, d, ent_i, i, side, fund)), i, float(t[ent_i]), True, 0))
            pos = 0
            continue
        if reg[i] != side:
            g = (o[i + 1] / ent_px - 1) * side
            tr.append((ent_i, side, float(with_funding(g, d, ent_i, i + 1, side, fund)), i + 1, float(t[ent_i]),
                       False, 0))
            pos = 0
            continue
        for kind, px, hi, lo in by_bar.get(i, []):
            if side > 0 and kind == 1 and hi:
                stop = max(stop, px * (1 - BUF))
            if side < 0 and kind == -1 and lo:
                stop = min(stop, px * (1 + BUF))
    return tr


def legs(d: dict, frame: int) -> np.ndarray:
    """사후 지그재그 다리(+1 상승 다리, −1 하락 다리), 15분봉 기준. 닮음 측정용."""
    u = d if frame == 1 else resample(d, frame, base_ms=BAR)
    sig = bd_signals(u["h"], u["l"], rsi(u["c"], 7))
    ext = sorted({(z[3], -z[1]) for z in sig})  # (극점 봉, +1=고점에서 끝나는 상승 다리 끝)
    lab = np.zeros(len(d["t"]), int)
    pos = np.searchsorted(d["t"], u["t"][[e for e, _ in ext]]) if ext else []
    for k in range(1, len(ext)):
        a, b = pos[k - 1], pos[k]
        lab[a:b] = 1 if ext[k][1] == 1 else -1
    return lab


def report(tr: list, n_days_old: float, n_days_new: float) -> dict:
    old = [z for z in tr if z[4] < ystart(2024)]
    new = [z for z in tr if z[4] >= ystart(2024)]
    g = sorted(z[2] for z in tr)
    return {"all": summ(tr), "old": summ(old), "new": summ(new),
            "new_c10": summ(new, 0.001),
            "new_long": summ([z for z in new if z[1] > 0]), "new_short": summ([z for z in new if z[1] < 0]),
            "drop5": round((float(np.mean(g[:-5])) - 0.001) * 100, 3) if len(g) > 5 else None,
            "per_day": round(len(tr) / (n_days_old + n_days_new), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--day-shift", type=int, default=0)
    ap.add_argument("--circ", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    d = bars(d5, 0)
    t = d["t"]
    reg, _ = R.regime(d5, d, args.day_shift)
    start = int(np.searchsorted(t, t[0] + 60 * DAY))
    days_old = (ystart(2024) - t[start]) / DAY
    days_new = (t[-1] - ystart(2024)) / DAY
    rep: dict = {"day_shift": args.day_shift}
    for name, frame in (("P15", 1), ("P1h", 4)):
        piv = pivots(d, frame)
        tr = run(d, reg, piv, start, fund)
        r = report(tr, days_old, days_new)
        r["trades"] = [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]
        if args.day_shift == 0:
            lab = legs(d, frame)
            pos = np.zeros(len(t), int)
            for z in tr:
                pos[z[0] + 1:z[3]] = z[1]
            m = np.arange(len(t)) >= start
            up, dn = m & (lab == 1), m & (lab == -1)
            r["resemble_up_long"] = round(float((pos[up] == 1).mean() * 100), 1)
            r["resemble_dn_short"] = round(float((pos[dn] == -1).mean() * 100), 1)
            r["flat_pct"] = round(float((pos[m] == 0).mean() * 100), 1)
        if args.circ and tr:
            # 방향은 두고 피벗 시각만 원형으로 민다
            rng = np.random.default_rng(7)
            actual = summ(tr)["ev"]
            null = []
            nb = len(t)
            for _ in range(args.circ):
                k = int(rng.integers(30 * 96, nb - 30 * 96))
                # 민 피벗의 가격은 원래 종가 대비 비율을 유지해 새 시각 종가에 붙인다
                sh2 = []
                for jj0, kind, px, hi, lo in piv:
                    jj = (jj0 + k) % nb
                    sh2.append((jj, kind, float(d["c"][jj] * px / d["c"][jj0]), hi, lo))
                tt = run(d, reg, sorted(sh2), start, fund)
                null.append(summ(tt).get("ev", 0.0))
            null = np.array(null)
            r["circ_null_ev"] = round(float(null.mean()), 3)
            r["circ_p"] = round(float((null >= actual).mean()), 4)
        rep[name] = r
    _, newday = R.regime(d5, d, args.day_shift)
    r1 = R.run(d, reg, newday, start, True, fund)
    rep["R1"] = {"old": summ([z for z in r1 if z[4] < ystart(2024)]),
                 "new": summ([z for z in r1 if z[4] >= ystart(2024)]), "all": summ(r1)}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
