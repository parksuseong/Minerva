"""116번: R1 숏에만 주봉(·월봉) 방향 정렬 문. 등록: 2026-11-07-btc-r1-short-htf-gate-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_fut_r1_short_gate.py \
    --out <폴더> [--day-shift 8] [--circ 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import btc_fut_daily_regime as R
from btc_fractal import load5
from btc_scr_entry_stop import summ
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, direction, ystart

H = 3_600_000
DAY = 86_400_000


def monthly(d5: dict, d: dict, shift_h: int) -> np.ndarray:
    """마감된 달력 월봉 방향을 15분봉 마감 시각에 맞춘다. 월 경계는 UTC 0시 + shift."""
    ts = (d5["t"] - shift_h * H).astype("datetime64[ms]")
    m = ts.astype("datetime64[M]")
    edges = np.flatnonzero(np.r_[True, m[1:] != m[:-1], True])
    st, en = edges[:-1], edges[1:]
    hh, ll = np.maximum.reduceat(d5["h"], st), np.minimum.reduceat(d5["l"], st)
    st, en, hh, ll = st[:-1], en[:-1], hh[:-1], ll[:-1]  # 마지막(진행 중) 달은 버린다
    u = {"o": d5["o"][st], "h": hh, "l": ll, "c": d5["c"][en - 1]}
    nxt = (m[st] + 1).astype("datetime64[ms]").astype(np.int64) + shift_h * H
    dr = direction(u)
    j = np.searchsorted(nxt, d["t"] + 900_000, side="right") - 1
    out = np.zeros(len(d["t"]), int)
    out[j >= 0] = dr[j[j >= 0]]
    return out


def gated(reg: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return np.where(reg > 0, 1, np.where((reg < 0) & (gate < 0), -1, 0))


def block(tr: list) -> dict:
    return {"all": summ(tr), "old": summ([z for z in tr if z[4] < ystart(2024)]),
            "new": summ([z for z in tr if z[4] >= ystart(2024)])}


def drop_top(tr: list, k: int = 5) -> float:
    g = sorted(z[2] for z in tr)
    g = g[:-k] if len(g) > k else []
    return round((float(np.mean(g)) - 0.001) * 100, 3) if g else 0.0


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
    sh = args.day_shift
    reg, newday = R.regime(d5, d, sh)
    wk, _ = R.regime(d5, d, 96 + sh, 2016)  # 1970-01-01은 목요일, +4일 = 월요일 시작
    mo = monthly(d5, d, sh)
    start = int(np.searchsorted(t, t[0] + 60 * DAY))
    gates = {"R1": np.full(len(t), -1), "W": wk, "WM": np.where((wk < 0) & (mo < 0), -1, 0)}
    rep: dict = {"day_shift": sh}
    for name, g in gates.items():
        tr = R.run(d, gated(reg, g), newday, start, True, fund)
        sh_tr = [z for z in tr if z[1] < 0]
        r = {"total": block(tr), "long": block([z for z in tr if z[1] > 0]), "short": block(sh_tr),
             "short_drop5": drop_top(sh_tr),
             "short_years": {y: summ([z for z in sh_tr if ystart(y) <= z[4] < ystart(y + 1)])
                             for y in range(2020, 2027)},
             "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
        if args.circ and name != "R1" and sh_tr:
            rng = np.random.default_rng(7)
            actual = summ(sh_tr)["ev"]
            null = []
            for _ in range(args.circ):
                k = int(rng.integers(60 * 96, len(t) - 60 * 96))
                tt = R.run(d, gated(reg, np.roll(g, k)), newday, start, True, fund)
                null.append(summ([z for z in tt if z[1] < 0]).get("ev", 0.0))
            null = np.array(null)
            r["circ_null_short_ev"] = round(float(null.mean()), 3)
            r["circ_p"] = round(float((null >= actual).mean()), 4)
        rep[name] = r
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
