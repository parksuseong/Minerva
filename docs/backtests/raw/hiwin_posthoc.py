"""사후 점검(변형 아님): E4 무작위를 'GT가 같은 방향인 15분봉'에서만 뽑아 다시 잰다.

등록 무작위는 ±15일 아무 봉이라 GT가 반대인 봉이 섞이고, E4 무효화(GT 반전 청산)가 그 봉에서 바로 걸려
무작위 승률이 낮아진다(승률 초과 부풀림). 문을 맞춘 무작위로 그 효과를 뺀다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import daily_wave as DW
import hiwin as H
import numpy as np
import ql_upgrade as U
import regime_lines as RL
import rr_asym as RA


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b, _ = RL.load_all(DW.R5)
    F = DW.features(b)
    t_end = int(b["t"][-1]) + H.M15
    m = U.load_1m(t_end)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    sim = H.Sim(m, b, F, fund_fn)
    W = RA.windows(t_end)
    sigs = H.build_signals(b, F)
    sigs["NRg"], sigs["NR0"] = H.nr4_signals(b, F, m)
    res = json.loads((out / "result.json").read_text())
    kq = res["kq"]
    gt = F["gate"]["GT"]
    t = b["t"]
    last_ok = int(np.searchsorted(t, int(t[-1]) - H.HOLD * H.MIN - H.DAY))
    rng = np.random.default_rng(11)
    outd = {}
    for name in ("R4a|E4", "R1|E4", "DC|E4", "NRg|E4"):
        s, e = name.split("|")
        tr = sim.run(sigs[s], W["fresh"][0], t_end, e, kq[s])
        R = np.full((len(tr), 200), np.nan)
        for a, x in enumerate(tr):
            c = x["i15"]
            lo, hi = max(c - 15 * 96, 200), min(c + 15 * 96, last_ok)
            pool = np.arange(lo, hi)
            pool = pool[gt[pool] == x["d"]]
            if not len(pool):
                continue
            for j in range(200):
                k = sim.kmin(int(rng.choice(pool)))
                if k >= 0:
                    i = int(rng.choice(pool))
                    k = sim.kmin(i)
                    if k >= 0:
                        R[a, j] = sim.trade(k, x["d"], x["S"], x["T"], True, i)["r"]
        tin = np.array([x["t_in"] for x in tr])
        outd[name] = {}
        for w in H.MAINW + ("pool",):
            sel = (tin >= W[w][0]) & (tin < W[w][1])
            outd[name][w] = H.rnd_stats([x for x, z in zip(tr, sel) if z], R[sel])
        print(name, {w: (round(v["excess"], 3), round(v["p_win"], 4), round(v["p_mean"], 3))
                     for w, v in outd[name].items()}, flush=True)  # fmt: skip
    json.dump(outd, (out / "posthoc_gate_random.json").open("w"))


if __name__ == "__main__":
    main()
