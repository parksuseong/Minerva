"""5차 탐색 차트: 구간별 매매(가격 + Stoch·CCI·RSI + 프리미엄 z)와 자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 10배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 r5_chart.py --out <폴더> 이름:구간 ...
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
import r5_eval as V

LAB = {"fresh": "fresh 2020-03~2021-10 (untouched)", "val": "validation 2024-07~2025-10",
       "seen": "seen 2025-11~", "design": "design 2022-01~2024-06"}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("items", nargs="+", help="이름:구간 (예: K2|X2:fresh)")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b, W, sets, comp, B = V.setup(out)
    b["f__drv"] = comp["zP"]
    rows = []
    for it in a_.items:
        key, _, w = it.rpartition(":")
        a, z = W[w]
        tr = V.run_one(B, b, comp, sets, key, a, z)
        s = V.summ(tr)
        ms = 4 * 3_600_000
        when = (f"{datetime.fromtimestamp(b['t'][a] / 1000, C.KST):%Y-%m-%d} ~ "
                f"{datetime.fromtimestamp((b['t'][z - 1] + E.M15) / 1000, C.KST):%Y-%m-%d} KST")  # fmt: skip
        NC.DRIVER[key] = ("_drv", "premium 1h z (30d)")
        f = out / f"chart_{key.replace('|', '_')}_{w}.png"
        NC.trade_chart(b, key, "", tr, s, a, z, ms, f,
                       f"{key} - {LAB[w]} {when}, 4h candles, fills on 15m",
                       [V.K2_LO, V.K2_HI])  # fmt: skip
        print(f)
        rows.append((f"{key} {LAB[w]}", tr, a, z))
    C.equity_chart(b, rows, out / "equity.png",
                   "Equity (mark-to-market on 15m closes), 10x / 3x / 1x vs buy & hold", log=True)
    print(out / "equity.png")


if __name__ == "__main__":
    main()
