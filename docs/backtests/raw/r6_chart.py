"""6차 탐색 차트: 구간별 매매(가격 + Stoch·CCI·RSI + ATR%)와 자산 곡선(10배·3배·1배).

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 10배 순손익.
자산 곡선은 15분 종가 평가(나눠 익절은 마지막 청산가로 근사).
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 r6_chart.py --out <폴더> 이름:구간 ...
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nonprice_chart as NC
import pine_chart as C
import pine_eval as E
import pine_ports as P
import r6_eval as V

LAB = {"fresh": "fresh 2020-03~2021-10", "val": "validation 2024-07~2025-10",
       "seen": "seen 2025-11~", "design": "design 2022-01~2024-06",
       "spot": "spot 2017-09~2020-02"}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("items", nargs="+", help="이름:구간 (예: TAWfail|XA:val)")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b, s, W, Ws, S, Ss, _ = V.setup()
    rows = []
    for it in a_.items:
        key, _, w = it.rpartition(":")
        src, spec, (a, z) = (s, Ss, Ws["spot"]) if w == "spot" else (b, S, W[w])
        tr = V.sim(V.bars_of(src), spec[key], a, z)
        sm = V.summ(tr)
        src = dict(src)
        src["f__atr"] = P.atr(src["h"], src["l"], src["c"], 14) / src["c"] * 100
        NC.DRIVER[key] = ("_atr", "ATR14 % of price")
        t0 = datetime.fromtimestamp(src["t"][a] / 1000, C.KST)
        t1 = datetime.fromtimestamp((src["t"][z - 1] + E.M15) / 1000, C.KST)
        when = f"{t0:%Y-%m-%d} ~ {t1:%Y-%m-%d} KST"
        f = out / f"chart_{key.replace('|', '_')}_{w}.png"
        NC.trade_chart(src, key, "", tr, sm, a, z, 4 * 3_600_000, f,
                       f"{key} - {LAB[w]} {when}, 4h candles, fills on 15m", [])  # fmt: skip
        print(f, sm["n"], sm["win"], sm["x2"]["ret"], sm["x3"]["ret"], sm["x10"]["ret"])
        if w != "spot":
            rows.append((f"{key} {LAB[w]}", tr, a, z))
    if rows:
        C.equity_chart(b, rows, out / "equity.png",
                       "Equity (mark-to-market on 15m closes), 10x / 3x / 1x vs buy & hold",
                       log=True)  # fmt: skip
        print(out / "equity.png")


if __name__ == "__main__":
    main()
