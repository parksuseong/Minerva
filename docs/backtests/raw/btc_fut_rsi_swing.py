"""15분봉 RSI 7 스윙 구조(HL·LH) 진입. 등록: 2026-11-07-btc-fut15-rsi-swing-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_fut_rsi_swing.py \
    --out <폴더> [--offset 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_fut_sr_pullback as SP
from btc_fractal import load5, rsi
from btc_fut_trend_hold import upper_dir
from btc_scr_entry_stop import summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, ystart

DAY = 86_400_000


def swings(h, l, r):
    """(확정 봉, 가격, 종류: 'HL'/'LL'/'LH'/'HH')."""
    out = []
    lo_zone = hi_zone = False
    ext = None
    prev_lo = prev_hi = None
    for i in range(1, len(r)):
        if np.isnan(r[i]):
            continue
        if r[i] < 30:
            ext = l[i] if not lo_zone else min(ext, l[i])
            lo_zone = True
        elif lo_zone:  # 30 위로 올라온 봉: 저점 확정
            lo_zone = False
            out.append((i, ext, "HL" if prev_lo is not None and ext > prev_lo else "LL"))
            prev_lo = ext
        if r[i] > 70:
            ext_h = h[i] if not hi_zone else max(ext_h, h[i])
            hi_zone = True
        elif hi_zone:
            hi_zone = False
            out.append((i, ext_h, "LH" if prev_hi is not None and ext_h < prev_hi else "HH"))
            prev_hi = ext_h
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    d = bars(d5, args.offset)
    t = d["t"]
    n = len(t)
    sw = swings(d["h"], d["l"], rsi(d["c"], 7))
    reg = upper_dir(d5, d, 288)
    days_old = (ystart(2024) - t[6000]) / DAY
    days_new = (t[-1] - ystart(2024)) / DAY
    rep: dict = {"offset": args.offset, "swings": len(sw)}
    for tp in (0.01, 0.02):
        SP.TP = tp
        for gn in ("G0", "GD"):
            cands = []
            for i, px, kind in sw:
                if kind not in ("HL", "LH") or i < 6000 or i >= n - 2:
                    continue
                side = 1 if kind == "HL" else -1
                if gn == "GD" and reg[i] != side:
                    continue
                stop = px * (1 - 0.001) if side > 0 else px * (1 + 0.001)
                res = SP.trade(d, int(i), side, stop)
                if res is None:
                    continue
                g, x, stopped, dist = res
                g = with_funding(g, d, int(i), x, side, fund)
                cands.append((int(i), side, float(g), int(x), float(t[i]), stopped, dist))
            cands.sort()
            tr = walk(cands)
            old = [z for z in tr if z[4] < ystart(2024)]
            new = [z for z in tr if z[4] >= ystart(2024)]
            rep[f"P{int(tp*100)}-{gn}"] = {
                "all": summ(tr), "old": summ(old), "new": summ(new),
                "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
                "per_day_old": round(len(old) / days_old, 2), "per_day_new": round(len(new) / days_new, 2),
                "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))




def bd_signals(h, l, r):
    """BalintDavid RSI Swing Indicator 상태 기계. (봉, +1/−1, 확정된 스윙 가격, 스윙 극점 봉)."""
    laststate, hh, ll = 0, l[0], h[0]
    hh_i = ll_i = 0
    out = []
    for i in range(len(r)):
        if np.isnan(r[i]):
            continue
        ob, os_ = r[i] >= 70, r[i] <= 30
        if laststate == 2 and ob:
            out.append((i, 1, ll, ll_i))
            hh, hh_i = h[i], i
        if laststate == 1 and os_:
            out.append((i, -1, hh, hh_i))
            ll, ll_i = l[i], i
        if ob:
            if h[i] >= hh:
                hh, hh_i = h[i], i
            laststate = 1
        if os_:
            if l[i] <= ll:
                ll, ll_i = l[i], i
            laststate = 2
        if laststate == 1 and h[i] >= hh:
            hh, hh_i = h[i], i
        if laststate == 2 and l[i] <= ll:
            ll, ll_i = l[i], i
    return out


def bd_run(d, sig, stop_kind, gate, fund):
    o, h, l, t = d["o"], d["h"], d["l"], d["t"]
    n = len(o)
    nxt = {}
    sig = [s for s in sig if s[0] >= 6000 and s[0] < n - 2]
    tr = []
    for k, (i, side, sw, _) in enumerate(sig):
        if tr and i < tr[-1][3]:
            continue
        if gate is not None and gate[i] != side:
            continue
        e = i + 1
        px = o[e]
        stop = sw * (1 - 0.001 * side) if stop_kind == "s" else px * (1 - 0.01 * side)
        if (side > 0 and stop >= px) or (side < 0 and stop <= px):
            continue
        opp = next((s[0] for s in sig[k + 1:] if s[1] != side), None)
        end = n - 2 if opp is None else opp + 1
        res = None
        for j in range(e, end):
            if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
                fill = min(o[j], stop) if side > 0 else max(o[j], stop)
                res = ((fill / px - 1) * side, j, True)
                break
        if res is None:
            if opp is None:
                continue
            res = ((o[end] / px - 1) * side, end, False)
        g = with_funding(res[0], d, i, res[1], side, fund)
        tr.append((i, side, float(g), int(res[1]), float(t[i]), res[2], abs(stop / px - 1)))
    return tr


def bd_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    d = bars(d5, args.offset)
    t = d["t"]
    sig = bd_signals(d["h"], d["l"], rsi(d["c"], 7))
    reg = upper_dir(d5, d, 288)
    days_old = (ystart(2024) - t[6000]) / DAY
    days_new = (t[-1] - ystart(2024)) / DAY
    rep: dict = {"offset": args.offset, "signals": len(sig)}
    for sk in ("s", "1"):
        for gn, g in (("G0", None), ("GD", reg)):
            tr = bd_run(d, sig, sk, g, fund)
            old = [z for z in tr if z[4] < ystart(2024)]
            new = [z for z in tr if z[4] >= ystart(2024)]
            rep[f"SW{sk}-{gn}"] = {
                "all": summ(tr), "old": summ(old), "new": summ(new),
                "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
                "per_day_old": round(len(old) / days_old, 2), "per_day_new": round(len(new) / days_new, 2),
                "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
    # 사후 표시 기준: 표시된 저점에서 사서 다음 표시된 고점에서 판다(실제로는 불가능)
    s6 = [s for s in sig if s[0] >= 6000]
    hind = []
    for a, b in zip(s6, s6[1:]):
        if a[1] > 0 and b[1] < 0:
            hind.append((b[2] / a[2] - 1, a[0]))
        elif a[1] < 0 and b[1] > 0:
            hind.append((-(b[2] / a[2] - 1), a[0]))
    hn = np.array([x[0] for x in hind]) - 0.001
    rep["hindsight"] = {"n": len(hn), "win": round(float((hn > 0).mean() * 100), 1), "ev": round(float(hn.mean() * 100), 3)}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import sys
    if "--bd" in sys.argv:
        sys.argv.remove("--bd")
        bd_main()
    else:
        main()
