"""다이버전스 실시간 사용: 추가 등록 C 대조(피벗 전체) + V1~V4 1시간 다이버전스 규칙.

등록: docs/backtests/2026-11-08-btc-div-usable.md "추가 등록". 연구 전용(주문 없음, 바이낸스 POST 없음).
실행(div_usable.py와 같은 PYTHONPATH, 메모리 상한):
  ... python3 docs/backtests/raw/div_usable_rules.py --out ~/.minerva/research/div-usable-2026-11-08
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import div_usable as DU
import numpy as np
import pine_div as P

DAY, HOUR = P.DAY, 3_600_000
N_RULES, N_CUM = 4, 6536 + 131
MAINW = P.MAINW
EXIT = {"exit": "X", "gate": False, "tier": "ALL", "be": 1.5}
SPEC = {
    "V1·1hR2·direct": {**EXIT, "tf": 60, "R": 2, "trig": None, "trail": 240},
    "V2·1hR3·direct": {**EXIT, "tf": 60, "R": 3, "trig": None, "trail": 240},
    "V3·1hR2·15turn": {**EXIT, "tf": 15, "R": 2, "trig": "turn", "trail": 60},
    "V4·1hR3·15turn": {**EXIT, "tf": 15, "R": 3, "trig": "turn", "trail": 60},
}


def plain_pivots(F, R):
    out = []
    for d, low in ((1, True), (-1, False)):
        x = F["l"] if low else F["h"]
        for p in DU.pivots(x, low, 5, R):
            if p + R < len(F["c"]):
                out.append({"p": int(p), "c": int(p + R), "d": d, "tier": "S", "ext": float(x[p])})
    out.sort(key=lambda s: (s["c"], s["d"]))
    return out


def two_prop(h1, n1, h2, n2):
    pp = (h1 * n1 + h2 * n2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return P.norm_p((h1 - h2) / se) if se else None


def turn_triggers(F1, F15, sigs):
    """1시간 설정 확정 뒤 12시간 안 첫 15분 방향 전환. 그 사이 피벗 극값 이탈이면 무효."""
    out = []
    ct, dr = F15["ct"], F15["dir"]
    for s in sigs:
        Tc, d = F1["ct"][s["c"]], s["d"]
        a, z = np.searchsorted(ct, Tc, "right"), np.searchsorted(ct, Tc + 12 * HOUR, "right")
        for i in range(max(a, 1), z):
            if (d == 1 and F15["l"][i] <= s["ext"]) or (d == -1 and F15["h"][i] >= s["ext"]):
                break
            if dr[i] == d and dr[i - 1] != d:
                out.append({"p": int(i), "c": int(i), "d": d, "tier": s["tier"], "ext": s["ext"],
                            "setup_T": int(Tc)})  # fmt: skip
                break
    out.sort(key=lambda s: (s["c"], s["d"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    t0 = time.time()
    b5 = P.load_5m(out)
    m = P.load_1m(out)
    S = P.Sim(m, P.load_funding())
    t_end = int(min(m["t"][-1], b5["t"][-1] + 300_000))
    Fs = {"15": P.agg(b5, 15), "30": P.agg(b5, 30), "1h": P.agg(b5, 60), "4h": P.agg(b5, 240), "1d": P.agg(b5, 1440)}
    del b5
    for k in Fs:
        keep = Fs[k]["ct"] <= t_end
        for x in ("t", "o", "c", "h", "l", "v", "ct", "rsi", "cci", "vsma", "dir"):
            Fs[k][x] = Fs[k][x][keep]
    Fs["1d"]["t6d"] = P.t6d_daily(Fs["1d"])
    for k in ("15", "1h"):
        P.ribbon(Fs[k])
    W = P.windows(t_end)
    C = DU.Ctx(m)
    rng = np.random.default_rng(20261109)
    res = {"t_end": t_end}

    # C 대조
    ctl = {}
    for fk, R, metric, hz in (("15", 2, "hit_1h", (1,)), ("30", 2, "hit_4h", (4,)),
                              ("1h", 2, "tb_0.01", (4,)), ("1h", 3, "tb_0.01", (4,))):  # fmt: skip
        F = Fs[fk]
        span = 15 * DAY
        dv = DU.frame_diag(C, F, DU.signals(F, R), W, span, hz, rng, t_end)[metric]
        pv = DU.frame_diag(C, F, plain_pivots(F, R), W, span, hz, rng, t_end)[metric]
        row = {}
        for w in (*MAINW, "pool"):
            a_, b_ = dv[w], pv[w]
            row[w] = {"div": a_.get("hit"), "piv": b_.get("hit"), "rand": a_.get("rand"), "n_div": a_["n"],
                      "n_piv": b_["n"], "p": two_prop(a_["hit"], a_["n"], b_["hit"], b_["n"]) if a_["n"] else None}  # fmt: skip
        ctl[f"{fk}|R{R}|{metric}"] = row
        pl = row["pool"]
        print("ctl", fk, R, metric, f"div {pl['div']:.3f} piv {pl['piv']:.3f} rand {pl['rand']:.3f} p {pl['p']:.2g}",
              f"{time.time() - t0:.0f}s", flush=True)  # fmt: skip
    res["control"] = ctl

    # V 규칙
    ctx15 = P.ctx_for(Fs["15"], Fs)
    ctx1h = P.ctx_for(Fs["1h"], Fs)
    rules, trj, cache = {}, {}, {}
    for name, spec in SPEC.items():
        s1 = DU.signals(Fs["1h"], spec["R"])
        if spec["trig"]:
            F, ctx, sg = Fs["15"], ctx15, turn_triggers(Fs["1h"], Fs["15"], s1)
        else:
            F, ctx, sg = Fs["1h"], ctx1h, s1
        tr = P.build_trades(S, F, sg, spec, ctx)
        r = {w: P.summ(P.in_w(tr, *W[w]), *W[w]) for w in P.WINS}
        sel = [x for x in tr if W["fresh"][0] <= x["T"]]
        Rm = P.random_for(S, F, sel, spec, ctx, cache)
        Ti = np.array([x["T"] for x in sel])
        r["rand"] = {}
        for w in (*MAINW, "pool"):
            mk = (Ti >= W[w][0]) & (Ti < W[w][1])
            r["rand"][w] = P.rand_stats([x for x, k in zip(sel, mk) if k], Rm[mk])
        red = {}
        for rk, kw in (("lat1m", {"lat": 1}), ("lat1bar", {"lat": spec["tf"]}), ("slip+2", {"slip": P.SLIP + 0.0002}),
                       ("fee1.5", {"fee": 1.5})):  # fmt: skip
            t2 = P.build_trades(S, F, sg, spec, ctx, **kw)
            red[rk] = {w: P.summ(P.in_w(t2, *W[w]), *W[w]) for w in ("val", "year")}
        r["red"] = red
        r["years"] = {}
        for y in range(2020, 2027):
            a_, z_ = P.ms(f"{y}-01-01"), min(P.ms(f"{y + 1}-01-01"), t_end)
            r["years"][y] = P.summ(P.in_w(tr, a_, z_), a_, z_)
        rules[name] = r
        trj[name] = {w: P.in_w(tr, *W[w]) for w in ("d30", "fresh", "design", "val", "year")}
        trj[name]["sigs_d30"] = [{"pt": int(F["ct"][s["p"]]), "d": s["d"], "ext": s["ext"]} for s in sg
                                 if F["ct"][s["c"]] >= W["d30"][0]]  # fmt: skip
        print(name, f"{time.time() - t0:.0f}s", " ".join(
            f"{w}:{r[w]['n']}/{100 * r[w].get('win', 0):.0f}%/{100 * r[w].get('risk1', {}).get('ret', 0):+.0f}%"
            for w in MAINW), flush=True)  # fmt: skip
    ps = {n: (rules[n]["rand"]["pool"] or {}).get("p", 1.0) for n in SPEC}
    order = sorted(ps, key=ps.get)
    holm, run = {}, 0.0
    for i, n in enumerate(order):
        run = max(run, min(1.0, ps[n] * (N_RULES - i)))
        holm[n] = run
    verdict = {}
    for n, r in rules.items():
        g1 = all(r[w]["n"] and r[w]["risk1"]["ret"] > 0 and r[w]["x5_40"]["ret"] > 0 and r[w]["x5_100"]["ret"] > 0
                 and r[w]["x5_100"]["liq"] == 0 for w in MAINW)  # fmt: skip
        g2 = holm[n] < 0.05 and all((r["rand"][w] or {}).get("ex_avg", -1) > 0 for w in MAINW)
        g3 = r["design"]["perday"] >= 0.25 and r["year"]["perday"] >= 0.25
        pl = r["pool"]
        g4 = bool(pl["n"]) and min(pl["long"], pl["short"]) >= 0.15 * pl["n"]
        g5 = all(r["red"][k][w]["n"] and r["red"][k][w]["risk1"]["ret"] > 0 for k in r["red"] for w in ("val", "year"))
        verdict[n] = {"g1": bool(g1), "g2": bool(g2), "g3": bool(g3), "g4": bool(g4), "g5": bool(g5), "holm": holm[n],
                      "cum": min(1.0, ps[n] * N_CUM), "pass": bool(g1 and g2 and g3 and g4 and g5)}  # fmt: skip
    res.update({"rules": rules, "verdict": verdict, "spec": SPEC, "secs": time.time() - t0})
    (out / "rules.json").write_text(json.dumps(res, default=float))
    (out / "trades.json").write_text(json.dumps(trj, default=float))
    print("pass:", [n for n in verdict if verdict[n]["pass"]], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
