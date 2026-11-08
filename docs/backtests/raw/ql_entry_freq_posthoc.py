"""사후 점검(등록 밖 1개): 등록 최종안은 B1u(범위 길이 합집합) 때문에 2024-25·최근 1년이 무너졌다.
받아들인 B 중 B2(같은 날 손절 뒤 재진입)만 P2에 붙인 R2를 같은 잣대(무작위·레드팀·연도별)로 본다.
사후라 판정에 쓰지 않고 참고로만 적는다."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ql_entry_freq as Q
import ql_upgrade as U


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    C = Q.setup()
    b, gates, m, fund_fn, W, vt, vol_fn, he, S = (
        C[k] for k in ("b", "gates", "m", "fund_fn", "W", "vt", "vol_fn", "hour_ema", "S")
    )
    plans = {"R2": [(0, S["NY60"], "A0", True)], "P2": [(0, S["NY60"], "A0", False)]}
    res: dict = {}
    keep_all: dict = {}
    for name, plan in plans.items():
        r, keep = {}, {}
        for w in U.SHOW:
            tr = Q.run(m, b, gates, plan, fund_fn, *W[w], he)
            f = U.sizes(tr, "S1", vol_fn, vt) if tr else None
            r[w] = U.summ(tr, *W[w], f)
            keep[w] = (tr, f)
        rnd = {}
        for w in ("pool",) + U.MAIN:
            tr, f = keep[w]
            for x in tr:
                x["_z"] = next(s["z"] for s in S[x["kind"]] if s["st"] == x["st"])
                x["_sd"] = abs(x["ep"] - x["stop"]) / x["ep"]
                x["_td"] = abs(x["tp"] - x["ep"]) / x["ep"]
                x["_fin"] = U.MAKER
            rnd[w] = Q.random_p(m, tr, fund_fn, S, f, 300 if w == "pool" else 200, 7)
        red: dict = {}
        for tag, kw in (
            ("lat1", {"lat": 1}),
            ("lat5", {"lat": 5}),
            ("slip2bp", {"slip": 0.0002}),
            ("fee15", {"fee_mult": 1.5}),
        ):
            red[tag] = {}
            for w in U.MAIN:
                tr = Q.run(m, b, gates, plan, fund_fn, *W[w], he, **kw)
                f = U.sizes(tr, "S1", vol_fn, vt) if tr else None
                red[tag][w] = U.path(tr, 1.0, f)
        tr = Q.run(m, b, gates, plan, fund_fn, W["fresh"][0], W["year"][1], he)
        f = U.sizes(tr, "S1", vol_fn, vt)
        yrs: dict = {}
        for i, x in enumerate(tr):
            yrs.setdefault(datetime.fromtimestamp(x["t_in"] / 1000, timezone.utc).year, []).append(
                i
            )
        red["by_year"] = {
            y: {"n": len(ix), **U.path([tr[i] for i in ix], 1.0, f[ix])}
            for y, ix in sorted(yrs.items())
        }
        rolls, s0 = [], W["fresh"][0]
        while s0 + 90 * U.DAY <= W["year"][1]:
            ix = [i for i, x in enumerate(tr) if s0 <= x["t_in"] < s0 + 90 * U.DAY]
            rolls.append(U.path([tr[i] for i in ix], 1.0, f[ix])["ret"] if ix else 0.0)
            s0 += 30 * U.DAY
        rolls = np.array(rolls)
        red["rolling90"] = {
            "pos_share": float((rolls > 0).mean()),
            "median": float(np.median(rolls)),
            "worst": float(rolls.min()),
        }
        nre = sum(1 for x in keep["pool"][0] if x["k"] > 0) - len(
            {x["st"] for x in keep["pool"][0]}
        )
        res[name] = {"res": r, "random": rnd, "redteam": red, "reentries_pool": int(nre)}
        keep_all[name] = {
            w: [{k: v for k, v in x.items() if not k.startswith("_")} for x in keep[w][0]]
            for w in ("d30", "m6", "year")
        }
        print(name, {w: (r[w]["n"], round(r[w]["per_day"], 2), round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f1.0"]["mdd"], 2),
                         round(r[w]["x5_f0.4"]["ret"], 2)) for w in U.SHOW}, flush=True)  # fmt: skip
        print(
            name,
            "random",
            {w: round(v["p"], 3) for w, v in rnd.items()},
            "re-entries",
            nre,
            flush=True,
        )
        print(name, "red", {t: {w: round(v["ret"], 2) for w, v in d.items()} for t, d in red.items() if t not in ("by_year", "rolling90")},
              {y: round(v["ret"], 2) for y, v in red["by_year"].items()}, red["rolling90"], flush=True)  # fmt: skip
    (out / "posthoc.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "posthoc_trades.json").write_text(json.dumps(keep_all, default=float))


if __name__ == "__main__":
    main()
