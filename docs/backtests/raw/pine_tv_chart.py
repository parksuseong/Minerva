"""트레이딩뷰 21개 시험 차트: pine_chart의 그리기 함수를 그대로 쓰고 매매만 pine_tv_eval로.

표시: 롱 원, 숏 삼각형. 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강. 숫자는 거래당 10배 손익.
실행: PYTHONPATH=<pyml>:<pylib>:docs/backtests/raw python3 pine_tv_chart.py --out <폴더> \
      --prefix grid "T22H:N:(tag)" "oE2|G3|K1/2:(tag)" [--long 이름...]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pine_chart as C
import pine_eval as E
import pine_tv_eval as V


def split_item(it: str) -> tuple:
    if "|" in it:
        core, _, tag = it.partition(":(")
        name, g, v = core.split("|")
        return core, f"{name}|{g}", v, "(" + tag if tag else ""
    parts = it.split(":", 2)
    return f"{parts[0]}:{parts[1]}", parts[0], parts[1], parts[2] if len(parts) > 2 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="grid")
    ap.add_argument("items", nargs="+")
    ap.add_argument("--long", action="store_true", help="장기 자산 곡선도")
    a_ = ap.parse_args()
    out = Path(a_.out).expanduser()
    b, S, gates, B, _ = V.load_all(out, old=True)
    W = V.windows(b)
    ha, hz = W["holdout"]
    rows, lrows = [], []
    for it in a_.items:
        core, label, var, tag = split_item(it)
        safe = label.replace("|", "_")
        for w in ("holdout", "last30"):
            a, z = W[w]
            tr = V.run_item(B, S, gates, core, a, z)
            s = E.summarize(tr)
            f = out / f"{a_.prefix}_{safe}_{w}.png"
            when = (
                f"{datetime.fromtimestamp(b['t'][a] / 1000, C.KST):%Y-%m-%d %H:%M} ~ "
                f"{datetime.fromtimestamp((b['t'][z - 1] + E.M15) / 1000, C.KST):%m-%d %H:%M} KST"
            )
            C.trade_chart(
                b,
                label,
                var,
                tr,
                s,
                a,
                z,
                f,
                f"{label} {tag} - {w} {when}, 1h candles, fills on 15m",
                ha - 4000,
            )
            if w == "holdout":
                rows.append((f"{label} [{var}] {tag} holdout", tr, a, z))
            print(f)
        if a_.long:
            la, lz = W["long"]
            lrows.append(
                (
                    f"{label} [{var}] 2020-08 ~ 2025-11 (log; 10x hits 0 at first liquidation)",
                    V.run_item(B, S, gates, core, la, lz),
                    la,
                    lz,
                )
            )
    C.equity_chart(
        b,
        rows,
        out / f"{a_.prefix}_equity_holdout.png",
        "Holdout equity (mark-to-market on 15m closes), 10x / 3x / 1x vs buy & hold",
    )
    print(out / f"{a_.prefix}_equity_holdout.png")
    if lrows:
        C.equity_chart(
            b, lrows, out / f"{a_.prefix}_equity_long.png", "Long-window robustness", log=True
        )
        print(out / f"{a_.prefix}_equity_long.png")


if __name__ == "__main__":
    main()
