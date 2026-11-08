"""SCR 실험실 신호 + 롱↔숏 뒤집기 + 거래소 손절(STOP-MARKET), 15분봉. 격자 7,776 × 손절 7 + 걷기 전진.

등록: docs/backtests/2026-11-08-btc-scr-sar-stop.md. 연구 전용(주문 없음, 바이낸스 POST 없음, 공개 GET klines 꼬리만).
신호·자료·비용은 scr_sar.py와 같다. 손절 뒤에는 다음 신호(어느 방향이든)까지 무포지션.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/scr-sar-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pylib:docs/backtests/raw python3 docs/backtests/raw/scr_sar_stop.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pine_div as P
import scr_sar as S

TAKER, SSLIP, MMR = S.TAKER, 0.0002, S.MMR
STOPS = ("F1", "F1.5", "F2", "F3", "A2", "A3", "ST")
N_NEW = S.N_GRID * len(STOPS) + 2 + 2 + 1 + 1
N_CUM = 14523 + N_NEW
DAY, MIN = S.DAY, S.MIN


def atr(F, n=14):
    pc = np.r_[F["c"][0], F["c"][:-1]]
    tr = np.maximum(F["h"] - F["l"], np.maximum(np.abs(F["h"] - pc), np.abs(F["l"] - pc)))
    out = np.empty_like(tr)
    out[0] = tr[0]
    a = tr.tolist()
    o = out.tolist()
    for i in range(1, len(a)):
        o[i] = o[i - 1] + (a[i] - o[i - 1]) / n
    return np.array(o)


class Stops:
    def __init__(self, F):
        self.atr = atr(F)
        self.ll = {n: np.r_[np.full(n - 1, np.nan), np.lib.stride_tricks.sliding_window_view(F["l"], n).min(1)]
                   for n in (3, 5)}  # fmt: skip
        self.hh = {n: np.r_[np.full(n - 1, np.nan), np.lib.stride_tricks.sliding_window_view(F["h"], n).max(1)]
                   for n in (3, 5)}  # fmt: skip

    def dist(self, kind, s, d, pe, w):
        """진입 때 손절 거리(비율). s = 신호 봉, d = 방향, pe = 진입가."""
        if kind == "N":  # 손절 없음: 닿을 수 없는 거리
            return np.full(np.shape(pe), 10.0)
        if kind[0] == "F":
            return np.full(np.shape(pe), float(kind[1:]) / 100)
        if kind[0] == "A":
            return float(kind[1:]) * self.atr[s] / pe
        n = max(w, 3)
        lv = np.where(d > 0, self.ll[n][s] * (1 - 0.0005), self.hh[n][s] * (1 + 0.0005))
        dd = np.where(d > 0, 1 - lv / pe, lv / pe - 1)
        return np.maximum(np.nan_to_num(dd, nan=0.0015), 0.0015)


# ---------------- 15분봉 근사 사슬 (격자·무작위·걷기 전진 학습) ----------------
def chain(book, stops, kind, w, sb, sd, z_bar, fee=TAKER):
    """신호 봉 sb(정렬)·방향 sd. 뒤집기 구간마다 손절 → 같은 구간 다음 신호에 재진입. 거래 dict(정렬)."""
    F, n = book.F, book.n
    m = sb + 1 < z_bar
    sb, sd = sb[m], sd[m]
    if len(sb) == 0:
        return None
    keep = np.r_[True, sd[1:] != sd[:-1]]
    seg = np.cumsum(keep) - 1
    fbar = sb[keep]
    X = np.r_[fbar[1:] + 1, z_bar]
    sdir = sd[keep]
    act_s, act_g = fbar.copy(), np.arange(len(fbar))
    out = []
    while len(act_s):
        e = act_s + 1
        x = X[act_g]
        d = sdir[act_g]
        pe = F["o"][e]
        dist = stops.dist(kind, act_s, d, pe, w)
        stp = pe * (1 - d * dist)
        L = x - e
        st = np.r_[0, np.cumsum(L)[:-1]]
        rep = np.repeat(np.arange(len(e)), L)
        pos = np.arange(L.sum()) - st[rep] + e[rep]
        dr = d[rep]
        hit = np.where(dr > 0, F["l"][np.minimum(pos, n - 1)] <= stp[rep], F["h"][np.minimum(pos, n - 1)] >= stp[rep])
        hits = np.flatnonzero(hit)
        j = np.searchsorted(hits, st)
        stopped = (j < len(hits)) & (hits[np.minimum(j, max(len(hits) - 1, 0))] < st + L) if len(hits) else np.zeros(len(e), bool)
        h = np.where(stopped, pos[hits[np.minimum(j, max(len(hits) - 1, 0))]] if len(hits) else 0, x)
        # 체결가
        oh = F["o"][np.minimum(h, n - 1)]
        gap = np.where(d > 0, oh <= stp, oh >= stp)
        sfill = np.where(gap, oh, stp) * (1 - d * SSLIP)
        nfill = np.where(x < n, F["o"][np.minimum(x, n - 1)], F["c"][n - 1])
        px = np.where(stopped, sfill, nfill)
        xb = np.where(stopped, h, x)
        fund = book.FP[np.minimum(xb, n - 1) + 1] - book.FP[e + 1]
        rr = px / pe
        r = d * (rr - 1) - fee * (1 + rr) - d * fund / pe
        lo = np.minimum.reduceat(F["l"][np.minimum(pos, n - 1)], st)
        hi = np.maximum.reduceat(F["h"][np.minimum(pos, n - 1)], st)
        adv = np.where(stopped, np.maximum(-d * (rr - 1), 0), np.where(d > 0, 1 - lo / pe, hi / pe - 1))
        Te = F["t"][e]
        Tx = np.where(xb < n, F["t"][np.minimum(xb, n - 1)], F["ct"][-1])
        out.append((e, Te, Tx, d, r, adv, dist, stopped))
        # 재진입: 손절 봉 이후(그 봉 마감 신호 포함) 같은 구간의 다음 신호
        jn = np.searchsorted(sb, h[stopped])
        g = act_g[stopped]
        ok = jn < len(sb)
        ok[ok] &= seg[jn[ok]] == g[ok]
        act_s, act_g = sb[jn[ok]], g[ok]
    cat = [np.concatenate([o[k] for o in out]) for k in range(8)]
    o = np.argsort(cat[0], kind="stable")
    return dict(zip(("e", "Te", "Tx", "d", "r", "adv", "dist", "stopped"), (c[o] for c in cat)))


# ---------------- 1분봉 정밀 모의 ----------------
class SimStop(S.Sim):
    def __init__(self, *a, stops=None, **k):
        super().__init__(*a, **k)
        self.stops = stops

    def scan(self, pos, pe, stp, je, j, be, tp, R):
        """[je, j) 1분봉에서 손절/BE/목표. (분 번호, 체결가, 종류) 또는 None."""
        m = self.m
        if j <= je:
            return None
        lo, hi, op = m["l"][je:j], m["h"][je:j], m["o"][je:j]
        adverse = (lo <= stp) if pos > 0 else (hi >= stp)
        k0 = int(np.argmax(adverse)) if adverse.any() else None
        cands = []
        if k0 is not None:
            cands.append((k0, 0, stp, "stop"))
        if tp:
            tpp = pe * (1 + pos * 2 * R)
            fav = (hi >= tpp) if pos > 0 else (lo <= tpp)
            if fav.any():
                cands.append((int(np.argmax(fav)), 1, tpp, "tp"))
        if be:
            t1 = pe * (1 + pos * R)
            fav = (hi >= t1) if pos > 0 else (lo <= t1)
            if fav.any():
                k1 = int(np.argmax(fav))
                if k0 is None or k1 < k0:
                    after = (lo[k1 + 1 :] <= pe) if pos > 0 else (hi[k1 + 1 :] >= pe)
                    cands = [c for c in cands if c[3] == "tp"]
                    if after.any():
                        cands.append((k1 + 1 + int(np.argmax(after)), 0, pe, "be"))
        if not cands:
            return None
        k, _, lvl, kind = min(cands)
        if kind == "tp":
            return je + k, lvl, kind
        o = op[k]
        p = o if ((o <= lvl) if pos > 0 else (o >= lvl)) else lvl
        return je + k, p * (1 - pos * SSLIP), kind

    def run_stop(self, sig, a, z, kind, w, lat=0, fee=TAKER, slip=0.0, chase=True, be=False, tp=False):
        F, m = self.F, self.m
        ia, iz = np.searchsorted(F["ct"], a), np.searchsorted(F["ct"], z)
        idx = np.nonzero(sig[ia:iz])[0] + ia
        jz = np.searchsorted(m["t"], z)
        tr, binds = [], 0
        st = {"pos": 0}
        cost = fee + slip

        def close(j, p, T, how):
            pos, pe, je = st["pos"], st["pe"], st["je"]
            seg = slice(je, max(j, je + 1))
            wv = m["l"][seg].min() if pos > 0 else m["h"][seg].max()
            adv = 1 - wv / pe if pos > 0 else wv / pe - 1
            if how in ("stop", "be"):
                adv = max(0.0, -pos * (p / pe - 1))
            k0, k1 = np.searchsorted(self.ft, st["Te"], "right"), np.searchsorted(self.ft, T, "right")
            fund = float((self.fr[k0:k1] * self.fp[k0:k1]).sum())
            rr = p / pe
            tr.append({"Te": int(st["Te"]), "Tx": int(T), "d": pos, "pe": pe, "px": p, "stop": st["stp"],
                       "dist": st["R"], "how": how, "r": pos * (rr - 1) - cost * (1 + rr) - pos * fund / pe,
                       "adv": adv})  # fmt: skip
            st["pos"] = 0

        def check(j):
            if st["pos"]:
                h = self.scan(st["pos"], st["pe"], st["stp"], st["je"], j, be, tp, st["R"])
                if h:
                    close(h[0], h[1], int(m["t"][h[0]]), h[2])

        for i in idx.tolist():
            d = int(sig[i])
            T = int(F["ct"][i]) + lat * MIN
            if T >= z:
                break
            j, p = self.px(T)
            check(j)
            if d == st["pos"]:
                continue
            if st["pos"]:
                close(j, p, T, "flip")
            lim = 0.03 if self.agree[i] == d else 0.01
            if chase and d * (p / F["c"][i] - 1) >= lim:
                binds += 1
                continue
            R = float(self.stops.dist(kind, np.array([i]), np.array([d]), np.array([p]), w)[0])
            st.update(pos=d, pe=p, je=j, Te=T, R=R, stp=p * (1 - d * R))
        check(jz)
        if st["pos"]:
            close(jz, m["c"][jz - 1], z, "end")
        return tr, binds


def risk1(tr, fee=TAKER):
    eq, peak, mdd, liq = 1.0, 1.0, 0.0, 0
    for x in tr:
        L = min(0.01 / (x["dist"] + 2 * fee), 5.0)
        if x["adv"] >= float(S.liq_thr(L, x["d"], fee)):
            liq += 1
            return {"ret": -1.0, "mdd": 1.0, "liq": liq}
        mdd = max(mdd, 1 - eq * max(0.0, 1 - L * (x["adv"] + fee)) / peak)
        eq *= 1 + L * x["r"]
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {"ret": eq - 1, "mdd": mdd, "liq": liq}


def summ(tr, a, z, binds=0):
    o = S.summ(tr, a, z, binds)
    if tr:
        o["stopped"] = sum(x["how"] in ("stop", "be") for x in tr)
        o["risk1"] = risk1(tr)
        o["dist_med"] = float(np.median([x["dist"] for x in tr]))
    return o


def win_stats(t, a, z):
    lo, hi = np.searchsorted(t["Te"], a), np.searchsorted(t["Tx"], z, "right")
    if hi <= lo:
        return {"n": 0}
    r, adv, d = t["r"][lo:hi], t["adv"][lo:hi], t["d"][lo:hi]
    p5 = S.path(r, adv, d, 5)
    return {"n": int(hi - lo), "win": float((r > 0).mean()), "mean": float(r.mean()), "ret5": p5["ret"],
            "liq5": p5["liq"], "ret1": float(np.prod(1 + r) - 1)}  # fmt: skip


def rand_p(book, stops, kind, w, sig, a_bar, z_bar, rng, nsim=1000):
    sb = np.nonzero(sig[a_bar:z_bar - 1])[0] + a_bar
    sd = sig[sb]
    t = chain(book, stops, kind, w, sb, sd, z_bar)
    if t is None or len(t["r"]) < 3:
        return None
    act = float(t["r"].mean())
    span = z_bar - 1 - a_bar
    sims = np.empty(nsim)
    for k in range(nsim):
        sh = (sb - a_bar + rng.integers(0, span)) % span + a_bar
        o = np.argsort(sh, kind="stable")
        sh, dd = sh[o], sd[o]
        u = np.r_[True, sh[1:] != sh[:-1]]
        tt = chain(book, stops, kind, w, sh[u], dd[u], z_bar)
        sims[k] = tt["r"].mean() if tt is not None else 0.0
    zsc = (act - sims.mean()) / (sims.std() + 1e-12)
    p_norm = 0.5 * math.erfc(zsc / math.sqrt(2))
    return {"act": act, "rand_mean": float(sims.mean()), "p_emp": float((1 + (sims >= act).sum()) / (1 + nsim)),
            "z": float(zsc), "p_norm": p_norm, "p_bonf": min(1.0, p_norm * N_CUM)}  # fmt: skip


def pick(n, win, s5, lq, nmin):
    ok = (n >= nmin) & (lq == 0)
    for need70 in (True, False):
        c = ok & (win >= 0.7) if need70 else ok
        if c.any():
            return int(np.argmax(np.where(c, s5 + n * 1e-9, -np.inf))), need70
    return None, False


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
    stops = Stops(F)
    sim = SimStop(m, F, F4, Fd, ft, fr, ind, stops=stops)
    book = S.Book(F, ft, fr)
    rng = np.random.default_rng(202611082)
    ms = S.ms
    W = {"d7": (t_end - 7 * DAY, t_end), "d14": (t_end - 14 * DAY, t_end), "d30": (t_end - 30 * DAY, t_end),
         "m6": (t_end - 183 * DAY, t_end), "y1": (t_end - 365 * DAY, t_end), "full": (ms("2020-03-01"), t_end),
         "m6x": (t_end - 183 * DAY, t_end - 30 * DAY), "y1x": (t_end - 365 * DAY, t_end - 30 * DAY),
         "p2025": (ms("2020-03-01"), ms("2026-01-01"))}  # fmt: skip
    for y in range(2020, 2027):
        W[f"y{y}"] = (max(ms(f"{y}-01-01"), ms("2020-03-01")), min(ms(f"{y + 1}-01-01"), t_end))
    YRS = [f"y{y}" for y in range(2020, 2027)]
    print("end", S.ds(t_end), "n_cum", N_CUM, flush=True)

    def bar(T):
        return int(np.searchsorted(F["t"], T))

    def report(p, kind, wins, red=False, rand=False, be=False, tp=False):
        sig = ind.sig(p)
        o = {"key": S.key(p) + " " + kind + (" BE" if be else "") + (" TP2R" if tp else ""), "params": p, "stop": kind}
        for wn in wins:
            a, z = W[wn]
            tr, b = sim.run_stop(sig, a, z, kind, p["w"], be=be, tp=tp)
            o[wn] = summ(tr, a, z, b)
        if red:
            for wn in ("d30", "y1x"):
                a, z = W[wn]
                o[f"red_{wn}"] = {"lat1m": summ(sim.run_stop(sig, a, z, kind, p["w"], lat=1, be=be, tp=tp)[0], a, z),
                                  "fee15_slip2": summ(sim.run_stop(sig, a, z, kind, p["w"], fee=TAKER * 1.5,
                                                                   slip=0.0002, be=be, tp=tp)[0], a, z)}  # fmt: skip
        if rand:
            for wn in ("d30", "y1x", "p2025"):
                a, z = W[wn]
                o[f"rand_{wn}"] = rand_p(book, stops, kind, p["w"], sig, bar(a), bar(z), rng)
        return o

    prev = json.loads((out / "result.json").read_text())
    named = [("base", S.BASE), ("rsi70", {**S.BASE, "rsi_hi": 70})] + [(f"top{i + 1}", t["params"])
                                                                     for i, t in enumerate(prev["top30"])]  # fmt: skip
    res = {"t_end": t_end, "t_end_s": S.ds(t_end), "n_cum": N_CUM, "named": {}}
    for nm, p in named:
        wins = ["d7", "d14", "d30", "m6", "y1", "full", "m6x", "y1x", "p2025"] + YRS if nm in ("base", "rsi70", "top1") \
            else ["d7", "d14", "d30", "m6x", "y1x", "p2025"] + YRS  # fmt: skip
        res["named"][nm] = [report(p, k, wins) for k in STOPS]
        print("named", nm, f"{time.time() - t0:.0f}s", flush=True)
    (out / "result_stop.json").write_text(json.dumps(res, default=float))

    # ---- 격자 7,776 × 7 ----
    combos = list(S.grid())
    NG = len(combos) * len(STOPS)
    mondays = np.arange(ms("2022-01-03"), t_end, 7 * DAY)
    FW = ["d30", "m6x", "y1x", "p2025"]
    fixed = {wn: {k: np.zeros(NG, np.float32) for k in ("n", "win", "ret5", "liq5", "mean", "ret1")} for wn in FW}
    TR = {tw: {"n": np.zeros((NG, len(mondays)), np.int16), "w": np.zeros((NG, len(mondays)), np.int16),
               "s5": np.zeros((NG, len(mondays)), np.float32), "liq": np.zeros((NG, len(mondays)), np.int8)}
          for tw in (30, 90)}  # fmt: skip
    nb = len(F["t"])
    for ci, p in enumerate(combos):
        sig = ind.sig(p)
        sb = np.nonzero(sig)[0]
        sd = sig[sb]
        for si, kind in enumerate(STOPS):
            gi = ci * len(STOPS) + si
            t = chain(book, stops, kind, p["w"], sb, sd, nb)
            if t is None:
                continue
            for wn in FW:
                s = win_stats(t, *W[wn])
                for k in fixed[wn]:
                    fixed[wn][k][gi] = s.get(k, 0) if s["n"] else (np.nan if k == "win" else 0)
            r, d = t["r"], t["d"]
            liq = (t["adv"] >= S.liq_thr(5, d)).astype(np.int32)
            lg = np.where(liq > 0, 0.0, np.log(np.maximum(1 + 5 * r, 1e-12)))
            Pw, Pl, Pq = (np.r_[0, np.cumsum(x)] for x in ((r > 0).astype(np.int32), lg, liq))
            for tw in (30, 90):
                lo = np.searchsorted(t["Te"], mondays - tw * DAY)
                hi = np.maximum(np.searchsorted(t["Tx"], mondays, "right"), lo)
                TR[tw]["n"][gi] = np.minimum(hi - lo, 32000)
                TR[tw]["w"][gi] = np.minimum(Pw[hi] - Pw[lo], 32000)
                TR[tw]["s5"][gi] = Pl[hi] - Pl[lo]
                TR[tw]["liq"][gi] = np.minimum(Pq[hi] - Pq[lo], 100)
        if ci % 250 == 0:
            print("grid", ci, f"{time.time() - t0:.0f}s", flush=True)
        if len(ind.rec) > 200:
            ind.rec.clear()
    np.savez_compressed(out / "grid_stop_fixed.npz", **{f"{wn}_{k}": v for wn in fixed for k, v in fixed[wn].items()})

    def gkey(gi):
        return combos[gi // len(STOPS)], STOPS[gi % len(STOPS)]

    f30 = fixed["d30"]
    win30 = np.nan_to_num(f30["win"])
    s5 = np.log(np.maximum(1 + f30["ret5"].astype(float), 1e-12))
    best, ok70 = pick(f30["n"], win30, s5, f30["liq5"], 10)
    c70 = (f30["n"] >= 10) & (win30 >= 0.7) & (f30["liq5"] == 0)
    res["grid"] = {"n": NG, "n_30d_win70": int(c70.sum()),
                   "by_stop_30d_win70": {k: int(c70[i::len(STOPS)].sum()) for i, k in enumerate(STOPS)}}  # fmt: skip
    for wn in ("m6x", "y1x", "p2025"):
        g = fixed[wn]
        res["grid"][f"win70_{wn}"] = {"ret5_pos_liq0": int(((g["ret5"] > 0) & (g["liq5"] == 0) & c70).sum()),
                                      "liq5_0": int(((g["liq5"] == 0) & c70).sum()),
                                      "win70": int(((np.nan_to_num(g["win"]) >= 0.7) & c70).sum()),
                                      "win_med": float(np.nanmedian(g["win"][c70])) if c70.any() else None,
                                      "ret1_med": float(np.median(g["ret1"][c70])) if c70.any() else None}  # fmt: skip
    # 통과 기준 2·3을 격자 근사로 만족하는 수(정밀 모의 대상)
    surv = c70 & (fixed["m6x"]["ret5"] > 0) & (fixed["m6x"]["liq5"] == 0) & (fixed["y1x"]["ret5"] > 0) \
        & (fixed["y1x"]["liq5"] == 0)  # fmt: skip
    res["grid"]["survive_m6x_y1x"] = int(surv.sum())
    res["grid"]["survive_list"] = [S.key(gkey(i)[0]) + " " + gkey(i)[1] for i in np.flatnonzero(surv)[:50]]
    order = np.argsort(-np.where(c70, f30["ret5"], -np.inf))
    top = [int(i) for i in order[:10] if c70[i]]
    res["top30"] = []
    for i in top:
        p, kind = gkey(i)
        res["top30"].append(report(p, kind, ["d7", "d14", "d30", "m6x", "y1x", "p2025"] + YRS,
                                   red=(i == top[0]), rand=(i == top[0])))  # fmt: skip
    # 살아남은 것(격자 근사) 정밀 확인, 최대 20개
    res["survivors"] = []
    for i in np.flatnonzero(surv)[:20].tolist():
        p, kind = gkey(i)
        res["survivors"].append(report(p, kind, ["d30", "m6x", "y1x", "p2025"] + YRS, red=True, rand=True))
    # 덧붙임: 1위 BE·TP2R, 다른 손절 종류의 최고에 BE
    if top:
        p, kind = gkey(top[0])
        res["addon"] = [report(p, kind, ["d30", "m6x", "y1x", "p2025"] + YRS, be=True),
                        report(p, kind, ["d30", "m6x", "y1x", "p2025"] + YRS, tp=True)]  # fmt: skip
        other = [i for i in top if gkey(i)[1] != kind]
        if other:
            p2, k2 = gkey(other[0])
            res["addon"].append(report(p2, k2, ["d30", "m6x", "y1x", "p2025"] + YRS, be=True))
    print("top done", f"{time.time() - t0:.0f}s", flush=True)
    (out / "result_stop.json").write_text(json.dumps(res, default=float))

    # ---- 이웃 ----
    if best is not None:
        bp, bk = gkey(best)
        nb_list = []
        ci = best // len(STOPS)
        for si in range(len(STOPS)):
            gi = ci * len(STOPS) + si
            nb_list.append({"key": S.key(bp) + " " + STOPS[si], **{f"{wn}_{k}": float(fixed[wn][k][gi]) for wn in FW
                            for k in ("n", "win", "ret5", "liq5")}})  # fmt: skip
        axes = {"stoch_n": S.G_ST_N, "stoch_l": S.G_ST_L, "cci_n": S.G_CC_N, "cci_l": S.G_CC_L, "rsi_n": S.G_RS_N,
                "rsi_l": S.G_RS_L, "w": S.G_W, "need": S.G_NEED}  # fmt: skip
        cur = {"stoch_n": bp["stoch_n"], "stoch_l": (bp["stoch_lo"], bp["stoch_hi"]), "cci_n": bp["cci_n"],
               "cci_l": (bp["cci_lo"], bp["cci_hi"]), "rsi_n": bp["rsi_n"], "rsi_l": (bp["rsi_lo"], bp["rsi_hi"]),
               "w": bp["w"], "need": bp["need"]}  # fmt: skip
        for ax, vals in axes.items():
            k = vals.index(cur[ax])
            for kk in (k - 1, k + 1):
                if 0 <= kk < len(vals):
                    q = dict(cur, **{ax: vals[kk]})
                    qp = {"stoch_n": q["stoch_n"], "stoch_lo": q["stoch_l"][0], "stoch_hi": q["stoch_l"][1],
                          "cci_n": q["cci_n"], "cci_lo": q["cci_l"][0], "cci_hi": q["cci_l"][1], "rsi_n": q["rsi_n"],
                          "rsi_lo": q["rsi_l"][0], "rsi_hi": q["rsi_l"][1], "w": q["w"], "need": q["need"]}  # fmt: skip
                    gi = combos.index(qp) * len(STOPS) + STOPS.index(bk)
                    nb_list.append({"key": S.key(qp) + " " + bk, **{f"{wn}_{k2}": float(fixed[wn][k2][gi])
                                    for wn in FW for k2 in ("n", "win", "ret5", "liq5")}})  # fmt: skip
        res["neighbors"] = nb_list

    # ---- 걷기 전진 ----
    res["wf"] = {}
    for tw, nmin in ((30, 10), (90, 30)):
        trs, picks, n70 = [], [], 0
        for wi, mo in enumerate(mondays.tolist()):
            n = TR[tw]["n"][:, wi].astype(float)
            win = np.where(n > 0, TR[tw]["w"][:, wi] / np.maximum(n, 1), 0)
            gi, ok = pick(n, win, TR[tw]["s5"][:, wi], TR[tw]["liq"][:, wi], nmin)
            if gi is None:
                continue
            n70 += ok
            p, kind = gkey(gi)
            picks.append(S.key(p) + " " + kind)
            tr, _ = sim.run_stop(ind.sig(p), mo, min(mo + 7 * DAY, t_end), kind, p["w"])
            trs += tr
            if len(ind.rec) > 200:
                ind.rec.clear()
        o = {"picked70": n70, "weeks": len(mondays), "distinct": len(set(picks)),
             "stop_kinds": {k: sum(x.endswith(" " + k) for x in picks) for k in STOPS}}  # fmt: skip
        for nm, a in (("from2022", ms("2022-01-03")), ("from2024_07", ms("2024-07-01"))):
            sub = [x for x in trs if x["Te"] >= a]
            o[nm] = summ(sub, a, t_end)
            o[nm]["years"] = {}
            for y in range(2022, 2027):
                yy = [x for x in sub if ms(f"{y}-01-01") <= x["Te"] < ms(f"{y + 1}-01-01")]
                if yy:
                    s = summ(yy, max(a, ms(f"{y}-01-01")), min(ms(f"{y + 1}-01-01"), t_end))
                    o[nm]["years"][y] = {k: s[k] for k in ("n", "win", "x1", "x5", "risk1")}
        res["wf"][tw] = o
        print("wf", tw, f"{time.time() - t0:.0f}s", flush=True)
    (out / "result_stop.json").write_text(json.dumps(res, indent=1, default=float))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
