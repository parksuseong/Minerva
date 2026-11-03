"""1시간봉 SCR 3단 고점·저점 적중, 2018~2021 확인. 등록: 2026-11-03-btc-h1-scr-top-holdout.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_h1_scr_holdout.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import btc_swing_extreme as X
from btc_fractal import load5, resample, scr3

A = datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
B = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def main() -> None:
    d = resample(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), 12)
    on_l, on_s, *_ = scr3(d)
    t = d["t"]
    yr = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).year for x in t])
    rng = np.random.default_rng(11)
    for name, sd, on in (("short", -1, on_s), ("long", 1, on_l)):
        lab = X.labels(d, sd)
        for label, m in (("2018~21", (t >= A) & (t < B)), ("2022~", t >= B)):
            ok = m & (lab >= 0)
            base = lab[ok].mean()
            sel = ok & on
            k, n = int(lab[sel].sum()), int(sel.sum())
            pool = np.flatnonzero(ok)
            draws = np.array([lab[rng.choice(pool, n, replace=False)].mean() for _ in range(1000)])
            p = float((draws >= k / n).mean())
            years = {}
            for y in sorted(set(yr[sel])):
                s2, b2 = sel & (yr == y), ok & (yr == y)
                years[int(y)] = (int(s2.sum()), round(lab[s2].mean() * 100, 1),
                                 round(lab[b2].mean() * 100, 1))
            print(name, label, f"hit {k}/{n} = {k / n * 100:.1f}%", f"base {base * 100:.1f}%",
                  f"p {p:.3f}", years)


if __name__ == "__main__":
    main()
