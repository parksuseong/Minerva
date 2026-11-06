"""1시간·4시간봉 SCR 3단 재시험. 등록: 2026-11-06-btc-scr-1h4h-recheck-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_frame.py \
    --out <폴더> --frame 12|48 [--offset <분>] [--futures] [--perm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample, scr3
from btc_scr_entry_stop import label, summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_swing_patterns import SPLIT

STOPS = (0.01, 0.03)
WARM_MIN = 6000 * 15  # 15분봉 6,000봉과 같은 준비 기간(분)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--frame", type=int, required=True, help="5분봉 몇 개를 묶을지 (12=1시간, 48=4시간)")
    ap.add_argument("--offset", type=int, default=0, help="봉 시작 이동(분)")
    ap.add_argument("--futures", action="store_true")
    ap.add_argument("--perm", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / ("futures/klines_5m.csv" if args.futures else "5m.csv"))
    fund = load_funding(base / "futures/funding.csv") if args.futures else None
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, args.frame)
    d["t"] = d["t"] + args.offset * 60_000
    n = len(d["c"])
    warm = WARM_MIN // (5 * args.frame)
    on_l, on_s, *_ = scr3(d)
    rep: dict = {"frame": args.frame, "offset": args.offset, "futures": args.futures}
    for stop in STOPS:
        cands = []
        for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
            for i in np.flatnonzero(on):
                if i < warm or i >= n - 2:
                    continue
                res = label(d, int(i), side, opp, d["o"][i + 1] * (1 - stop * side))
                if res is None:
                    continue
                g, x, dist, stopped = res
                g = with_funding(g, d, int(i), x, side, fund)
                cands.append((int(i), side, float(g), int(x), float(d["t"][i]), stopped, dist))
        cands.sort()
        tr = walk(cands)
        name = f"S{round(stop * 100)}"
        r = {"all": summ(tr), "all_c10": summ(tr, 0.001), "old": summ([z for z in tr if z[4] < SPLIT]),
             "new": summ([z for z in tr if z[4] >= SPLIT]), "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
        r["pass_1245"] = bool(r["old"].get("n", 0) >= 30 and r["new"].get("n", 0) >= 30 and r["old"]["ev"] > 0
                              and r["new"]["ev"] > 0 and r["all_c10"]["ev"] > 0 and (r["all"]["drop5"] or -1) > 0)
        if args.perm and r["pass_1245"]:
            rng = np.random.default_rng(7)
            nl = sum(1 for z in tr if z[1] > 0)
            actual = float(np.mean([z[2] for z in tr]))
            pool = np.arange(warm, n - 2)
            cache: dict = {}

            def outcome(i: int, side: int):
                if (i, side) not in cache:
                    res = label(d, i, side, on_s if side > 0 else on_l, d["o"][i + 1] * (1 - stop * side))
                    cache[(i, side)] = None if res is None else (with_funding(res[0], d, i, res[1], side, fund), res[1])
                return cache[(i, side)]

            hits = 0
            for _ in range(2000):
                picks = sorted([(int(k), 1) for k in rng.choice(pool, nl, replace=False)]
                               + [(int(k), -1) for k in rng.choice(pool, len(tr) - nl, replace=False)])
                busy, gs = -1, []
                for i, side in picks:
                    if i <= busy:
                        continue
                    oc = outcome(i, side)
                    if oc is None:
                        continue
                    gs.append(oc[0])
                    busy = oc[1]
                if gs and np.mean(gs) >= actual:
                    hits += 1
            r["perm_p"] = round(hits / 2000, 4)
        rep[name] = r
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
