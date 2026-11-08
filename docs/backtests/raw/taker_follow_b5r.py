"""등록 밖 덧붙임: deep B5를 그대로 뒤집은 규칙(B5R) — 4시간 테이커 쏠림 + 1시간 가격이 반대(흡수처럼 보임) → 쏠린 쪽으로.

등록 18개에는 없다(등록 P는 가격 조건을 뺀 단순형). 결과를 본 뒤 추가했으므로 본페로니 6,451로 세고 "사후"로 표시한다.
롱: tbS4 ≥ 설계 q95 그리고 r1 ≤ 0, 숏: tbS4 ≤ 설계 q05 그리고 r1 ≥ 0. 청산 H24·NS(P와 같음), 같은 실행기·무작위·레드팀.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import daily_wave as DW
import deep as DP
import numpy as np
import ql_upgrade as U
import rr_asym as A
import taker_follow as TF
from r5_data import load

N_CUM = TF.N_CUM + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    t0 = time.time()
    b = load(DW.R5)
    t_end = int(b["t"][-1]) + U.M15
    m = U.load_1m(t_end)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    sim = DP.Sim(m, b, fund_fn)
    F = TF.build(b)
    W = F["W"]
    W["dv"] = (W["design"][0], W["val"][1])
    t, c = b["t"], b["c"]
    dz = (t >= W["design"][0]) & (t < W["design"][1])
    with np.errstate(all="ignore"):
        s4 = DP.rmean(b["tbv"].astype(float), 16) / DP.rmean(b["v"].astype(float), 16)
    r1 = np.r_[np.full(4, np.nan), c[4:] / c[:-4] - 1]
    q95, q05 = float(np.nanquantile(s4[dz], 0.95)), float(np.nanquantile(s4[dz], 0.05))
    with np.errstate(invalid="ignore"):
        L, S = (s4 >= q95) & (r1 <= 0), (s4 <= q05) & (r1 >= 0)
    u = L & ~np.r_[False, L[:-1]]
    dn = S & ~np.r_[False, S[:-1]]
    F["sig"]["B5R"] = F["sig"]["B5R|PF"] = (u, dn)  # diag1이 |PF 키도 찾는다(같은 값)
    F["Z"]["B5R"] = F["Z"]["T4h"]
    TF.SPEC["B5R|H24|NS"] = ("B5R", "H24", "NS", False)
    TF.FEATS = (*TF.FEATS, "B5R")
    periods = {w: W[w] for w in TF.MAINW + ("m6", "d30")}
    for y in range(2020, 2027):
        periods[str(y)] = (max(A.ms(f"{y}-01-01"), A.ms("2020-03-01")), min(A.ms(f"{y + 1}-01-01"), t_end))
    fw = TF.fwd_table(sim)
    d1 = {k: v for k, v in TF.diag1(sim, F, fw, periods, np.random.default_rng(11)).items() if k.startswith("B5R|") and "|PF" not in k}
    for k, r in d1.items():
        print("D1", k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in r.items()}, flush=True)
    del fw
    run = TF.Runner(sim, F)
    fn = "B5R|H24|NS"
    tr = run.run(fn)
    res = {w: DP.summ(DP.slice_w(tr, *W[w]), *W[w]) for w in TF.SHOWW}
    for w in TF.MAINW + ("m6", "d30"):
        s = res[w]
        print(fn, w, s["n"], round(s["per_day"], 2), s["nl"], s["ns"], s["win"] and round(s["win"], 2),
              s["avg1"] and round(s["avg1"] * 100, 3), round(s["x5_f1.0"]["ret"], 3), round(s["x5_f0.4"]["ret"], 3),
              s["x5_f1.0"]["liq"], flush=True)  # fmt: skip
    dv = DP.slice_w(tr, *W["dv"])
    rnd = TF.random_match(sim, dv, *W["dv"], args.nsim, 7)
    rnd["p_bonf"] = min(1.0, rnd["p_norm"] * N_CUM)
    print("rnd", rnd, flush=True)
    red = {}
    for nm, kw in (("lat1", {"lat": 1}), ("lat5", {"lat": 5}), ("lat15", {"lat": 15}),
                   ("slip+0.02", {"slip": DP.SLIP + 0.0002}), ("fee1.5", {"fee_mult": 1.5})):  # fmt: skip
        tr2 = run.run(fn, **kw)
        red[nm] = {w: DP.summ(DP.slice_w(tr2, *W[w]), *W[w]) for w in TF.MAINW}
        print("red", nm, {w: round(red[nm][w]["x5_f1.0"]["ret"], 2) for w in red[nm]}, flush=True)
    years = {}
    for y in range(2020, 2027):
        years[y] = DP.summ(DP.slice_w(tr, *periods[str(y)]), *periods[str(y)])
    print("years", {y: (v["n"], round(v["x5_f1.0"]["ret"], 2), round(v["x5_f0.4"]["ret"], 2)) for y, v in years.items()})
    json.dump({"th": {"q95": q95, "q05": q05}, "D1": d1, "res": res, "rnd": rnd, "red": red, "years": years,
               "n_cum": N_CUM}, (out / "b5r_result.json").open("w"), default=float)  # fmt: skip
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
