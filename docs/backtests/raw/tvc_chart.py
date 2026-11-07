"""트레이딩뷰 구성요소 시험 차트: 검증·최근 30일 매매(가격 + Stoch·CCI·RSI + 동인)와 자산 곡선.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 10배 순손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 tvc_chart.py --out <폴더> "C1|X3" ...
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nonprice_chart as NC
import numpy as np
import pine_chart as C
import pine_eval as E
import pine_tv_eval as T
import tvc_eval as V

ZP_SETS = {"C1", "C1L", "K1", "K2", "K3", "K4", "K5", "K6", "K7"}


def driver(b: dict, comp: dict, g: dict, sname: str) -> tuple:
    if sname in ZP_SETS or sname.startswith("ML"):
        q = g["thresholds"]["zP"]
        lines = [q["0.5"], q["99.5"]] if sname == "C1" else [q["2"], q["98"]]
        return b["f_zP"], "premium 1h z (30d)", lines
    if sname in ("C6", "K8"):
        return comp["feat"]["z15"], "VWAP z 15m", [-2.0, 2.0] if sname == "C6" else [-1.5, 1.5]
    if sname == "C7":
        return comp["feat"]["cvdz"], "CVD z", [-2.0, 2.0]
    return None, "", []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("items", nargs="+", help="세트|청산[:tag]")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b, W, g, S, comp = V.load_strats(out)
    B = V.bars(b)
    rows = []
    for it in a_.items:
        key, _, tag = it.partition(":")
        sname = key.split("|")[0]
        feat, dlab, lines = driver(b, comp, g, sname)
        for w, ms in (("val", 4 * 3_600_000), ("last30", 3_600_000)):
            a, z = W[w]
            tr = T.simulate(B, S[key], ("N",), a, z)
            s = V.summ(tr)
            when = (
                f"{datetime.fromtimestamp(b['t'][a] / 1000, C.KST):%Y-%m-%d} ~ "
                f"{datetime.fromtimestamp((b['t'][z - 1] + E.M15) / 1000, C.KST):%Y-%m-%d %H:%M} KST"
            )
            lab = "validation" if w == "val" else "last 30 days (inside seen period)"
            f = out / f"chart_{key.replace('|', '_')}_{w}.png"
            fe = feat if feat is not None else np.full(len(b["t"]), np.nan)
            NC.DRIVER[key] = ("_drv", dlab or "(no non-price driver)")
            b["f__drv"] = fe
            NC.trade_chart(b, key, "", tr, s, a, z, ms, f,
                           f"{key} {tag} - {lab} {when}, {ms // 3_600_000}h candles, fills on 15m",
                           lines)  # fmt: skip
            print(f)
        for w in ("val", "seen"):
            a, z = W[w]
            rows.append((f"{key} {tag} {w}", T.simulate(B, S[key], ("N",), a, z), a, z))
    C.equity_chart(b, rows, out / "equity.png",
                   "Equity (mark-to-market on 15m closes), 10x / 3x / 1x vs buy & hold; "
                   "validation 2024-07~2025-10 and seen 2025-11~", log=True)  # fmt: skip
    print(out / "equity.png")


if __name__ == "__main__":
    main()
