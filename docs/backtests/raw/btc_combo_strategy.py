"""오늘 조합을 묶은 전략: 패턴 12종 + 숏 돌파 거래량 확인 + 해마다 과거로만 칸 고르기.

등록: 2026-11-02-btc-combo-strategy-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_combo_strategy.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import btc_book2 as K
import btc_pattern_census as P
import btc_swing_patterns as S
from btc_fractal import load5, resample

SIZES = (0.02, 0.04)
BREAK = ("DB", "DBP", "TB", "IHS", "ATRI")
MIN_N = 30
COST = 0.0005


def year_of(ms: float) -> int:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).year


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + args.offset * 60_000
    S.CAP = K.CAP = 48
    v = d["v"]
    cs = np.cumsum(np.r_[0.0, v])

    def vmean(a: int, b: int) -> float:
        a = max(a, 0)
        return (cs[b] - cs[a]) / max(b - a, 1)

    cands = []
    for th in SIZES:
        for side in (1, -1):
            for name, at, g, why, rr in P.rows_for(d, side, th):
                if at < 200 or at >= len(v) - 1:
                    continue
                if side < 0 and name in BREAK:
                    sig = at - 1
                    if not (v[sig] >= 1.5 * vmean(sig - 20, sig) and
                            vmean(sig - 40, sig) < vmean(sig - 80, sig - 40)):
                        continue
                cell = f"{name}_{'L' if side > 0 else 'S'}_{int(th * 100)}"
                cands.append({"cell": cell, "at": int(at), "t": float(d["t"][at]),
                              "y": year_of(d["t"][at]), "g": float(g), "why": why, "rr": float(rr)})
    cands.sort(key=lambda c: c["at"])
    years = sorted({c["y"] for c in cands})
    trades, picks = [], {}
    for yr in years:
        if yr < 2020:
            continue
        hist = [c for c in cands if c["y"] < yr]
        stats = {}
        for c in hist:
            stats.setdefault(c["cell"], []).append(c["g"] - 2 * COST)
        ok = {k: float(np.mean(x)) for k, x in stats.items() if len(x) >= MIN_N and np.mean(x) > 0}
        picks[yr] = sorted(ok.items(), key=lambda kv: -kv[1])
        busy = -1
        cur = [c for c in cands if c["y"] == yr and c["cell"] in ok]
        i = 0
        while i < len(cur):
            j = i
            while j + 1 < len(cur) and cur[j + 1]["at"] == cur[i]["at"]:
                j += 1
            grp = cur[i:j + 1]
            best = max(grp, key=lambda c: ok[c["cell"]])
            if best["at"] > busy:
                trades.append({**best, "ev_hist": ok[best["cell"]]})
                busy = best["at"] + 48
            i = j + 1

    def summ(ts: list, cost: float) -> dict:
        if not ts:
            return {"n": 0}
        net = np.array([x["g"] for x in ts]) - 2 * cost
        eq = np.cumprod(1 + net)
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        s = np.sort(net)
        return {"n": len(ts), "win": round(float((net > S.WIN).mean() * 100), 1),
                "ev": round(float(net.mean() * 100), 3), "ret": round(float((eq[-1] - 1) * 100), 1),
                "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
                "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None}

    rep = {"offset": args.offset, "n_cands": len(cands),
           "picks": {str(k): [[a, round(b * 100, 3)] for a, b in v_] for k, v_ in picks.items()},
           "all_c05": summ(trades, COST), "all_c10": summ(trades, 0.001),
           "by_year": {str(y): summ([x for x in trades if x["y"] == y], COST) for y in years if y >= 2020},
           "since2024": summ([x for x in trades if x["y"] >= 2024], COST),
           "since2024_c10": summ([x for x in trades if x["y"] >= 2024], 0.001),
           "hold_btc": None}
    rep["trades"] = trades
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v_ for k, v_ in rep.items() if k not in ("trades", "picks")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
