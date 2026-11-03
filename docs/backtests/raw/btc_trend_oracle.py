"""추세를 미리 알면 30분봉 SCR 승률이 얼마나 오르나 (상한 확인용, 매매 규칙 아님).

추세 라벨은 btc_xgb_trend.labeling(앞으로 60일 가격으로 스윙을 정함)이라 미래를 쓴다.
실시간 대조는 전날 일봉 종가가 60일선 위면 상승, 아래면 하락.
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_trend_oracle.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5
from btc_scr_trend import simulate, stats
from btc_scr_wait_fast import on_grid
from btc_xgb_trend import labeling, load_daily, sma

D = 86_400_000
ts = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731


def main() -> None:
    root = Path("~/.minerva/data/btc").expanduser()
    d5 = load5(root / "5m.csv")
    dd = load_daily(root / "1d.csv")
    lab, p0, p1 = labeling(dd["c"])
    trend = np.zeros(len(dd["t"]))
    trend[p0:p1] = np.where(lab[p0:p1] == 0, 1, np.where(lab[p0:p1] == 1, -1, 0))
    ma = sma(dd["c"], 60)
    rt = np.where(np.isnan(ma), 0, np.where(dd["c"] > ma, 1, -1))
    day = np.searchsorted(dd["t"], d5["t"], side="right") - 1
    oracle = trend[np.maximum(day, 0)]
    live = np.r_[0, rt][np.maximum(day, 0)]  # 전날 마감 값
    el, es = on_grid(d5, 6, 0)
    t = d5["t"]
    last = dd["t"][p1 - 1]
    for name, a, b in (("2018~21", ts(2018), ts(2022)), ("2022~", ts(2022), last)):
        j0, j1 = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
        for st in (0.01, 0.03):
            for fname, f in (("all", None), ("hindsight", oracle), ("ma60_live", live)):
                ent = (el, es) if f is None else (el & (f > 0), es & (f < 0))
                r = stats(simulate(d5, ent, (el, es), "X1", st, None, j0, j1))
                print(name, f"SL{st * 100:g}%", fname,
                      {k: r.get(k) for k in ("n", "win", "ev", "x3", "x3_mdd", "x1")}, flush=True)


if __name__ == "__main__":
    main()
