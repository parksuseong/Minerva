"""5분봉 SCR 3단. 등록: 2026-11-03-btc-scr-5m-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_5m.py --out <폴더>
"""

from __future__ import annotations

import btc_majors

btc_majors.RULES = {"R5a": (1, 0, "X1", 0.01), "R5b": (1, 0, "X1", 0.03)}

if __name__ == "__main__":
    btc_majors.main()
