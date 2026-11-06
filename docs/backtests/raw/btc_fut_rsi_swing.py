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


if __name__ == "__main__":
    main()
