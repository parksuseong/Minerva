"""SCR 실험실 — 기준선 고정(사용자 기본값), 지표 길이만 바꾸기. 묶음 2(rsi 30/80, 30/70) × 길이 800 × 청산 4.

등록: docs/backtests/2026-11-08-btc-scr-sar-len.md. 연구 전용(주문 없음, 바이낸스 POST 없음, 공개 GET klines 꼬리만).
신호·비용·손절 모의는 scr_sar.py / scr_sar_stop.py와 같다. 청산 N = 손절 없는 뒤집기.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/scr-sar-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pylib:docs/backtests/raw python3 docs/backtests/raw/scr_sar_len.py --out $OUT
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pine_div as P
import scr_sar as S
import scr_sar_stop as X

ST_N = (3, 5, 7, 9, 12, 14, 18, 21, 25, 30)
CC_N = (7, 10, 14, 20, 26, 30, 40, 50)
RS_N = (7, 9, 11, 14, 18, 21, 25, 28, 35, 42)
EXITS = ("N", "F2", "F3", "A3")
FAM = {"L80": 80, "L70": 70}
N_CUM = 68961 + 2 * 800 * 4 + 2 + 1
X.N_CUM = N_CUM
DAY, MIN = S.DAY, S.MIN


def params(fam, sn, cn, rn):
    return {"stoch_n": sn, "stoch_lo": 20, "stoch_hi": 80, "cci_n": cn, "cci_lo": -100, "cci_hi": 100,
            "rsi_n": rn, "rsi_lo": 30, "rsi_hi": FAM[fam], "w": 3, "need": 3}  # fmt: skip


def key(p, ex):
    return f"S{p['stoch_n']} C{p['cci_n']} R{p['rsi_n']} (rsi 30/{p['rsi_hi']}) {ex}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    t0 = time.time()
    m = P.load_1m(out)
    now = int(m["t"][-1]) + MIN
    ft, fr = P.load_funding()
    F = S.bars(m, 15, now)
    F4, Fd = S.bars(m, 240, now), S.bars(m, 1440, now)
    t_end = int(F["ct"][-1])
    ind = S.Ind(F)
    stops = X.Stops(F)
    sim = X.SimStop(m, F, F4, Fd, ft, fr, ind, stops=stops)
    book = S.Book(F, ft, fr)
    rng = np.random.default_rng(202611083)
    ms = S.ms
    W = {"d7": (t_end - 7 * DAY, t_end), "d14": (t_end - 14 * DAY, t_end), "d30": (t_end - 30 * DAY, t_end),
         "m6": (t_end - 183 * DAY, t_end), "y1": (t_end - 365 * DAY, t_end), "full": (ms("2020-03-01"), t_end),
         "m6x": (t_end - 183 * DAY, t_end - 30 * DAY), "y1x": (t_end - 365 * DAY, t_end - 30 * DAY),
         "p2025": (ms("2020-03-01"), ms("2026-01-01"))}  # fmt: skip
    YRS = [f"y{y}" for y in range(2020, 2027)]
    for y in range(2020, 2027):
        W[f"y{y}"] = (max(ms(f"{y}-01-01"), ms("2020-03-01")), min(ms(f"{y + 1}-01-01"), t_end))
    ALLW = ["d7", "d14", "d30", "m6", "y1", "full", "m6x", "y1x", "p2025"] + YRS
    print("end", S.ds(t_end), "n_cum", N_CUM, flush=True)

    def bar(T):
        return int(np.searchsorted(F["t"], T))

    def report(p, ex, wins, red=False, rand=False):
        sig = ind.sig(p)
        o = {"key": key(p, ex), "params": p, "exit": ex}
        for wn in wins:
            a, z = W[wn]
            tr, b = sim.run_stop(sig, a, z, ex, p["w"])
            o[wn] = X.summ(tr, a, z, b)
            o[wn]["lab4h"] = S.lab4h(ind, F, p, a, z)
        if red:
            a, z = W["y1x"]
            o["red_y1x"] = {"lat1m": X.summ(sim.run_stop(sig, a, z, ex, p["w"], lat=1)[0], a, z),
                            "fee15_slip2": X.summ(sim.run_stop(sig, a, z, ex, p["w"], fee=S.TAKER * 1.5,
                                                               slip=0.0002)[0], a, z)}  # fmt: skip
        if rand:
            for wn in ("d30", "y1x", "p2025"):
                a, z = W[wn]
                o[f"rand_{wn}"] = X.rand_p(book, stops, ex, p["w"], sig, bar(a), bar(z), rng)
        return o

    res = {"t_end_s": S.ds(t_end), "n_cum": N_CUM, "fam": {}}
    mondays = np.arange(ms("2022-01-03"), t_end, 7 * DAY)
    FW = ["d30", "m6x", "y1x", "p2025"]
    nb = len(F["t"])
    for fam in FAM:
        combos = [params(fam, *q) for q in itertools.product(ST_N, CC_N, RS_N)]
        NG = len(combos) * len(EXITS)
        fixed = {wn: {k: np.full(NG, np.nan, np.float32) for k in ("n", "win", "ret5", "liq5", "ret1")} for wn in FW}
        TR = {tw: {k: np.zeros((NG, len(mondays)), np.float32) for k in ("n", "w", "s5", "liq")} for tw in (30, 90)}
        nsig = np.zeros((len(combos), 4), np.int32)  # 전체 롱·숏 신호, 30일 롱·숏
        a30 = bar(W["d30"][0])
        for ci, p in enumerate(combos):
            lo, sh = ind.sides(p)
            nsig[ci] = (lo.sum(), sh.sum(), lo[a30:].sum(), sh[a30:].sum())
            sig = ind.sig(p)
            sb = np.nonzero(sig)[0]
            sd = sig[sb]
            for xi, ex in enumerate(EXITS):
                gi = ci * len(EXITS) + xi
                t = X.chain(book, stops, ex, p["w"], sb, sd, nb)
                if t is None:
                    continue
                for wn in FW:
                    s = X.win_stats(t, *W[wn])
                    for k in fixed[wn]:
                        fixed[wn][k][gi] = s.get(k, 0) if s["n"] else (np.nan if k == "win" else 0)
                r, d = t["r"], t["d"]
                liq = (t["adv"] >= S.liq_thr(5, d)).astype(np.int32)
                lg = np.where(liq > 0, 0.0, np.log(np.maximum(1 + 5 * r, 1e-12)))
                Pw, Pl, Pq = (np.r_[0, np.cumsum(x)] for x in ((r > 0).astype(np.int32), lg, liq))
                for tw in (30, 90):
                    lo_ = np.searchsorted(t["Te"], mondays - tw * DAY)
                    hi_ = np.maximum(np.searchsorted(t["Tx"], mondays, "right"), lo_)
                    TR[tw]["n"][gi] = hi_ - lo_
                    TR[tw]["w"][gi] = Pw[hi_] - Pw[lo_]
                    TR[tw]["s5"][gi] = Pl[hi_] - Pl[lo_]
                    TR[tw]["liq"][gi] = Pq[hi_] - Pq[lo_]
            if len(ind.rec) > 200:
                ind.rec.clear()
        print(fam, "grid", f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(out / f"grid_len_{fam}.npz", nsig=nsig,
                            **{f"{wn}_{k}": v for wn in fixed for k, v in fixed[wn].items()})  # fmt: skip

        def gk(gi, combos=combos):
            return combos[gi // len(EXITS)], EXITS[gi % len(EXITS)]

        R = {"signals": {"full_long_med": float(np.median(nsig[:, 0])), "full_short_med": float(np.median(nsig[:, 1])),
                         "d30_long_med": float(np.median(nsig[:, 2])), "d30_short_med": float(np.median(nsig[:, 3])),
                         "d30_short_zero": int((nsig[:, 3] == 0).sum()), "full_short_lt_long_half": int(
                             (nsig[:, 1] < 0.5 * nsig[:, 0]).sum())}}  # fmt: skip
        f30 = fixed["d30"]
        win30 = np.nan_to_num(f30["win"])
        c70 = (f30["n"] >= 10) & (win30 >= 0.7) & (f30["liq5"] == 0)
        R["grid"] = {"n": NG, "n_30d_win70": int(c70.sum()),
                     "by_exit": {ex: int(c70[i::len(EXITS)].sum()) for i, ex in enumerate(EXITS)}}  # fmt: skip
        for wn in ("m6x", "y1x", "p2025"):
            g = fixed[wn]
            R["grid"][wn] = {"win70": int(((np.nan_to_num(g["win"]) >= 0.7) & c70).sum()),
                             "ret5_pos_liq0": int(((g["ret5"] > 0) & (g["liq5"] == 0) & c70).sum()),
                             "win_med": float(np.nanmedian(g["win"][c70])) if c70.any() else None,
                             "ret1_med": float(np.median(g["ret1"][c70])) if c70.any() else None}  # fmt: skip
        surv = c70 & (fixed["m6x"]["ret5"] > 0) & (fixed["m6x"]["liq5"] == 0) & (fixed["y1x"]["ret5"] > 0) \
            & (fixed["y1x"]["liq5"] == 0)  # fmt: skip
        R["grid"]["survive_m6x_y1x"] = int(surv.sum())
        s5 = np.log(np.maximum(1 + f30["ret5"].astype(float), 1e-12))
        best, ok70 = X.pick(f30["n"], win30, np.nan_to_num(s5, nan=-1e9), f30["liq5"], 10)
        R["best"] = {"gi": best, "ok70": ok70}
        base = params(fam, 9, 20, 21)
        R["base"] = [report(base, ex, ALLW) for ex in EXITS]
        order = np.argsort(-np.where(c70, f30["ret5"], -np.inf))
        top = [int(i) for i in order[:10] if c70[i]]
        R["top30"] = [report(*gk(i), ["d7", "d14", "d30", "m6x", "y1x", "p2025"] + YRS, red=(i == best),
                             rand=(i == best)) for i in top]  # fmt: skip
        R["survivors"] = [report(*gk(i), ["d30", "m6x", "y1x", "p2025"] + YRS, red=True, rand=True)
                          for i in np.flatnonzero(surv)[:15].tolist()]  # fmt: skip
        if best is not None and best not in top:
            R["best_report"] = report(*gk(best), ["d7", "d14", "d30", "m6x", "y1x", "p2025"] + YRS, red=True, rand=True)
        # 이웃과 열지도
        if best is not None:
            bp, bx = gk(best)
            xi = EXITS.index(bx)
            nbrs = []
            for ax, vals in (("stoch_n", ST_N), ("cci_n", CC_N), ("rsi_n", RS_N)):
                k = vals.index(bp[ax])
                for kk in (k - 1, k + 1):
                    if 0 <= kk < len(vals):
                        q = dict(bp, **{ax: vals[kk]})
                        gi = combos.index(q) * len(EXITS) + xi
                        nbrs.append({"key": key(q, bx), **{f"{wn}_{k2}": float(fixed[wn][k2][gi]) for wn in FW
                                     for k2 in ("n", "win", "ret5", "liq5", "ret1")}})  # fmt: skip
            R["neighbors"] = nbrs
            heat = {}
            for cn in sorted({20, bp["cci_n"]}):
                for ex in EXITS:
                    xi2 = EXITS.index(ex)
                    tab = []
                    for sn in ST_N:
                        row = []
                        for rn in RS_N:
                            gi = combos.index(params(fam, sn, cn, rn)) * len(EXITS) + xi2
                            row.append([float(fixed["d30"]["n"][gi]), float(fixed["d30"]["win"][gi]),
                                        float(fixed["y1x"]["win"][gi]), float(fixed["y1x"]["ret5"][gi]),
                                        float(fixed["p2025"]["ret1"][gi])])  # fmt: skip
                        tab.append(row)
                    heat[f"C{cn}_{ex}"] = tab
            R["heat"] = heat
        # 걷기 전진
        R["wf"] = {}
        for tw, nmin in ((30, 10), (90, 30)):
            trs, picks, n70 = [], [], 0
            for wi, mo in enumerate(mondays.tolist()):
                n = TR[tw]["n"][:, wi]
                win = np.where(n > 0, TR[tw]["w"][:, wi] / np.maximum(n, 1), 0)
                gi, ok = X.pick(n, win, TR[tw]["s5"][:, wi], TR[tw]["liq"][:, wi], nmin)
                if gi is None:
                    continue
                n70 += ok
                p, ex = gk(gi)
                picks.append(key(p, ex))
                tr, _ = sim.run_stop(ind.sig(p), mo, min(mo + 7 * DAY, t_end), ex, p["w"])
                trs += tr
                if len(ind.rec) > 200:
                    ind.rec.clear()
            o = {"picked70": n70, "picked": len(picks), "weeks": len(mondays), "distinct": len(set(picks)),
                 "exits": {ex: sum(x.endswith(" " + ex) for x in picks) for ex in EXITS}}  # fmt: skip
            for nm, a in (("from2022", ms("2022-01-03")), ("from2024_07", ms("2024-07-01"))):
                sub = [x for x in trs if x["Te"] >= a]
                o[nm] = X.summ(sub, a, t_end)
            R["wf"][tw] = o
        res["fam"][fam] = R
        print(fam, "done", f"{time.time() - t0:.0f}s", flush=True)
        (out / "result_len.json").write_text(json.dumps(res, default=float))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
