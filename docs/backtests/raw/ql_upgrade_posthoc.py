"""사후 점검(등록 밖, 2개): 등록 최종안이 추가 우위 A1 때문에 무너져, 본선만으로
P1 = E0m·Gany·X5(익절 지정가)·S0, P2 = E0m·Gany·X6(뉴욕 16시 청산 + 익절 지정가)·S1 을 같은 잣대로 본다.
결과는 사후라 판정에 쓰지 않고 참고로만 적는다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import daily_wave as DW
import numpy as np
import ql_portfolio as QP
import ql_upgrade as U
import regime_lines as RL


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b, s4 = RL.load_all(DW.R5)
    F = DW.features(b)
    QP.add_gates(b, s4, F)
    gates = {k: F["gate"][k] for k in U.GATES4}
    m = U.load_1m(int(b["t"][-1]) + U.M15)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    res = json.loads((out / "result.json").read_text())
    W = {k: tuple(v) for k, v in res["windows"].items()}
    vt = res["pick"]["vtarget"]
    dc_i = np.r_[np.nonzero(np.diff(b["t"] // U.DAY))[0], len(b["t"]) - 1]
    dc, dday = b["c"][dc_i], (b["t"] // U.DAY)[dc_i]
    dr = np.r_[np.nan, dc[1:] / dc[:-1] - 1]
    rv = np.array([np.nanstd(dr[max(0, i - 20) : i]) if i >= 5 else np.nan for i in range(len(dr))])

    def vol_fn(t_in):
        i = min(max(int(np.searchsorted(dday, t_in // U.DAY)), 0), len(rv) - 1)
        return float(rv[i]) if np.isfinite(rv[i]) else 0.0

    sess = U.sessions(m, b, gates, "NY")
    sess_by = {(s["kind"], s["st"]): s for s in sess}
    lists = [(0, sess, "E0m", "Gany")]
    out_all, keep_all = {}, {}
    for name, X, S in (("P1", "X5", "S0"), ("P2", "X6", "S1")):
        cfg = {"X": X}
        r, keep = {}, {}
        for w in U.SHOW:
            tr = U.run(m, b, gates, lists, cfg, fund_fn, *W[w])
            f = U.sizes(tr, S, vol_fn, vt) if tr else None
            r[w] = U.summ(tr, *W[w], f)
            keep[w] = (tr, f)
        rnd = {}
        for w in ("pool",) + U.MAIN:
            tr, f = keep[w]
            for x in tr:
                x["_e"] = "E0m"
            U.annotate(tr, m, sess_by)
            rnd[w] = U.random_p(m, b, gates, tr, cfg, fund_fn, sess, *W[w], 300 if w == "pool" else 200, 7, f)
        red = {}
        for tag, kw in (("slip2bp", {"slip": 0.0002}), ("fee15", {"fee_mult": 1.5}), ("lat1", {"lat": 1})):
            red[tag] = {}
            for w in U.MAIN:
                tr = U.run(m, b, gates, lists, cfg, fund_fn, *W[w], **kw)
                f = U.sizes(tr, S, vol_fn, vt) if tr else None
                red[tag][w] = U.path(tr, 1.0, f)
        a_ms, z_ms = W["fresh"][0], W["year"][1]
        tr = U.run(m, b, gates, lists, cfg, fund_fn, a_ms, z_ms)
        f = U.sizes(tr, S, vol_fn, vt)
        from datetime import datetime, timezone

        yrs: dict = {}
        for i, x in enumerate(tr):
            yrs.setdefault(datetime.fromtimestamp(x["t_in"] / 1000, timezone.utc).year, []).append(i)
        red["by_year"] = {y: U.path([tr[i] for i in ix], 1.0, f[ix]) for y, ix in sorted(yrs.items())}
        out_all[name] = {"X": X, "S": S, "res": r, "random": rnd, "redteam": red}
        keep_all[name] = {w: [{k: v for k, v in x.items() if not k.startswith("_")} for x in keep[w][0]]
                          for w in ("d30", "m6", "year")}  # fmt: skip
        print(name, {w: (r[w]["n"], round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f1.0"]["mdd"], 2))
                     for w in U.SHOW}, flush=True)  # fmt: skip
        print(name, "random", {w: round(v["p"], 3) for w, v in rnd.items()}, flush=True)
        print(name, "red", {t_: {w: round(v["ret"], 2) for w, v in d.items()} for t_, d in red.items()}, flush=True)
    (out / "posthoc.json").write_text(json.dumps(out_all, indent=1, default=float))
    (out / "posthoc_trades.json").write_text(json.dumps(keep_all, default=float))


if __name__ == "__main__":
    main()
