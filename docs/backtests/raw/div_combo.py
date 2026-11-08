"""다이버전스 조합: H1 거르개(NR·DC·SCR 기본 규칙 × 거부/확인 × 6/12시간), H2 방향 × 크기, H3 여러 봉 일치. 18개.

등록: docs/backtests/2026-11-08-btc-div-combo.md. 연구 전용(주문 없음, 바이낸스 POST 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/div-combo-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/div_combo.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import daily_wave as DW
import div_usable as DU
import leads_ens as LE
import numpy as np
import pine_div as P
import ql_upgrade as U
import regime_lines as RL
import rr_asym as A
import rr_asym2 as A2
import scr15_timing as SC

DAY, HOUR, MIN, M15 = P.DAY, 3_600_000, P.MIN, 900_000
N_STUDY, N_CUM = 18, 6667 + 18
MAINW = P.MAINW
WINS = P.WINS
NBOOT, NPERM, NR = 5000, 10000, 20
XS = (0.01, 0.02)


# ---------------- 다이버전스 사건 ----------------
def events(Fs: dict, frames: tuple) -> dict:
    """frame → (T 확정 마감 시각, d, ext) 배열, 세 등급 전부, R=2."""
    out = {}
    for k in frames:
        F = Fs[k]
        sg = DU.signals(F, 2)
        out[k] = (np.array([F["ct"][s["c"]] for s in sg], np.int64), np.array([s["d"] for s in sg]),
                  np.array([s["ext"] for s in sg]))  # fmt: skip
    return out


def has_event(ev, t, d, hours):
    """t − hours < T ≤ t 에 방향 d 사건이 있는가(ev = 여러 봉 합친 (T, d))."""
    T, D = ev
    a, z = np.searchsorted(T, t - hours * HOUR, "right"), np.searchsorted(T, t, "right")
    return bool(np.any(D[a:z] == d))


def merge(evs: dict, keys) -> tuple:
    T = np.concatenate([evs[k][0] for k in keys])
    D = np.concatenate([evs[k][1] for k in keys])
    o = np.argsort(T, kind="stable")
    return T[o], D[o]


def allow_fn(ev, kind, hours):
    if kind == "base":
        return lambda t, d: True
    if kind == "V":
        return lambda t, d: not has_event(ev, t, -d, hours)
    return lambda t, d: has_event(ev, t, d, hours)


# ---------------- 기본 규칙 셋 ----------------
class NRRunner(LE.Runner):
    """leads_ens N4D·GT. allow(t_dec, d)로 recs를 거른다. 익절 k는 기본 규칙 값으로 고정."""

    allow = staticmethod(lambda t, d: True)

    def recs(self, v):
        base = super().recs(v)
        t1 = self.sim.m["t"]
        return [r for r in base if self.allow(int(t1[r[3]]) if r[3] is not None else int(self.sim.b["t"][r[0]]) + M15, r[1])]


def nr_trades(R, v, a, z, allow, **kw):
    R.allow = allow
    tr = R.run(v, a, z, **kw)
    return [{"T": x["t_in"], "d": x["d"], "r": x["r"], "worst": x["worst"], "Rp": x["S"], "e": x["ep"],
             "stop": x["ep"] * (1 - x["d"] * x["S"]), "exits": [(x["t_out"] - MIN, x["xp"], 1.0)], "why": x["why"],
             "tdec": x["t_in"]} for x in tr]  # fmt: skip


def dc_trades(sim2, sig, b, a, z, allow, **kw):
    i, d = sig
    tdec = b["t"][i] + M15
    keep = np.array([allow(int(t), int(dd)) for t, dd in zip(tdec, d)], bool)
    tr = sim2.run2((i[keep], d[keep]), a, z, "S1", 3, "BE", 5760, **kw)
    return [{"T": x["t_in"], "d": x["d"], "r": x["r"], "worst": x["worst"], "Rp": x["S"], "e": x["ep"],
             "stop": x["ep"] * (1 - x["d"] * x["S"]), "exits": [(x["t_out"] - MIN, x["xp"], 1.0)], "why": x["why"],
             "tdec": int(b["t"][x["i15"]]) + M15} for x in tr]  # fmt: skip


def scr_trades(b, F, a_ms, z_ms, allow, delay=0, fee=SC.FEE, slip=0.0):
    """scr15_timing B0+S1 그대로, 거르개는 새 진입만 막는다(반대 신호 청산은 그대로). slip은 레드팀용."""
    t, o, h, l, c = b["t"], b["o"], b["h"], b["l"], b["c"]
    a, z = int(np.searchsorted(t, a_ms)), int(np.searchsorted(t, z_ms - M15))
    L, S, big = F["L"], F["S"], F["big"]
    ft, fcum = b["ft"], F["fcum"]
    out, pos = [], None

    def close(i_bar, px, t_exit):
        nonlocal pos
        d, ep = pos["d"], pos["ep"]
        px = px * (1 - d * slip)
        i0, i1 = np.searchsorted(ft, pos["t"], "right"), np.searchsorted(ft, t_exit, "right")
        fu = -d * float(fcum[i1] - fcum[i0])
        r = d * (px / ep - 1) - 2 * fee + fu
        out.append({"T": pos["t"], "d": d, "r": r, "worst": min(pos["worst"], d * (px / ep - 1)), "Rp": 0.01,
                    "e": ep, "stop": pos["stop"], "exits": [(t_exit - MIN, px, 1.0)], "tdec": pos["tdec"]})  # fmt: skip
        pos = None

    last = z - 1
    for i in range(a, z):
        if pos and i >= pos["e"]:
            d, ep, sp = pos["d"], pos["ep"], pos["stop"]
            if (l[i] <= sp) if d == 1 else (h[i] >= sp):
                gap = (o[i] < sp) if d == 1 else (o[i] > sp)
                close(i, o[i] if gap else sp, int(t[i]) + M15)
            else:
                pos["worst"] = min(pos["worst"], (l[i] / ep - 1) if d == 1 else -(h[i] / ep - 1))
        if i >= last - delay:
            continue
        sig = 1 if (L[i] and not S[i]) else (-1 if (S[i] and not L[i]) else 0)
        if not sig:
            continue
        ei = i + 1 + delay
        if pos and pos["d"] == -sig:
            close(ei, o[ei], int(t[ei]))
        if pos is not None:
            continue
        tdec = int(t[i]) + M15
        if not allow(tdec, sig):
            continue
        ep = o[ei] * (1 + sig * slip)
        lim = 0.03 if big[i] else 0.01
        if abs(ep / c[i] - 1) > lim:
            continue
        pos = {"d": sig, "e": ei, "ep": ep, "t": int(t[ei]), "stop": ep * (1 - sig * 0.01), "worst": 0.0, "tdec": tdec}
    if pos:
        close(last, c[last], int(t[last]) + M15)
    return [x for x in out if x["T"] < z_ms]


# ---------------- 통계 ----------------
def boot_diff(a, b, rng):
    if len(a) < 2 or len(b) < 2:
        return None
    ia = rng.integers(0, len(a), (NBOOT, len(a)))
    ib = rng.integers(0, len(b), (NBOOT, len(b)))
    dm = a[ia].mean(1) - b[ib].mean(1)
    dw = (a[ia] > 0).mean(1) - (b[ib] > 0).mean(1)
    return {"d_mean": float(a.mean() - b.mean()), "ci_mean": [float(np.quantile(dm, 0.025)), float(np.quantile(dm, 0.975))],
            "d_win": float((a > 0).mean() - (b > 0).mean()),
            "ci_win": [float(np.quantile(dw, 0.025)), float(np.quantile(dw, 0.975))]}  # fmt: skip


def perm_p(r, keep, rng):
    k = int(keep.sum())
    if k < 2 or k > len(r) - 2:
        return None
    obs = r[keep].mean() - r[~keep].mean()
    tot = r.sum()
    cnt = 0
    for _ in range(NPERM // 1000):
        idx = np.argsort(rng.random((1000, len(r))), 1)[:, :k]
        s = r[idx].sum(1)
        d = s / k - (tot - s) / (len(r) - k)
        cnt += int((d >= obs).sum())
    return (cnt + 1) / (NPERM + 1)


def filter_effect(base, keep, W, rng):
    T = np.array([x["T"] for x in base])
    r = np.array([x["r"] for x in base])
    out = {}
    for w in (*MAINW, "pool"):
        mk = (T >= W[w][0]) & (T < W[w][1])
        rk, rr = r[mk & keep], r[mk & ~keep]
        e = {"n_keep": int(len(rk)), "n_drop": int(len(rr)),
             "keep_mean": float(rk.mean()) if len(rk) else None, "drop_mean": float(rr.mean()) if len(rr) else None,
             "keep_win": float((rk > 0).mean()) if len(rk) else None, "drop_win": float((rr > 0).mean()) if len(rr) else None,
             "base_mean": float(r[mk].mean()) if mk.any() else None, "base_win": float((r[mk] > 0).mean()) if mk.any() else None}  # fmt: skip
        e["kd"] = boot_diff(rk, rr, rng)
        e["kb"] = boot_diff(rk, r[mk], rng)
        if w == "pool":
            e["p_perm"] = perm_p(r[mk], keep[mk], rng)
        out[w] = e
    return out


def summ_all(tr, W, t_end):
    r = {w: P.summ(P.in_w(tr, *W[w]), *W[w]) for w in WINS}
    r["years"] = {}
    for y in range(2020, 2027):
        a_, z_ = max(P.ms(f"{y}-01-01"), W["fresh"][0]), min(P.ms(f"{y + 1}-01-01"), t_end)
        r["years"][y] = P.summ(P.in_w(tr, a_, z_), a_, z_)
    return r


# ---------------- 진단(목표 먼저 vs 손절) ----------------
def tb_diag(C, sigs, cand_t, cand_ok, W, rng, t_end):
    """sigs: [(T, d, ext)]. 무작위 = ±15일 cand_t 중 cand_ok, 같은 방향·같은 손절 거리, NR개."""
    ss = [s for s in sigs if s[0] < t_end - 14 * DAY]
    T = np.array([s[0] for s in ss], np.int64)
    d = np.array([s[1] for s in ss])
    ext = np.array([s[2] for s in ss])
    k0 = np.searchsorted(C.t, T)
    e = C.o[np.clip(k0, 0, len(C.t) - 1)]
    sd = d * (e - ext) / e
    ok = (sd > 0) & (k0 < len(C.t) - 2)
    fav = np.zeros(len(T))
    rf = np.full((len(T), NR), np.nan)
    pool_t = cand_t[cand_ok & (cand_t < t_end - 14 * DAY)]
    for i in np.nonzero(ok)[0]:
        fav[i] = C.path(int(k0[i]), int(d[i]), float(sd[i]))[0]
        lo, hi = np.searchsorted(pool_t, T[i] - 15 * DAY), np.searchsorted(pool_t, T[i] + 15 * DAY)
        if hi <= lo:
            continue
        for j, tr in enumerate(pool_t[rng.integers(lo, hi, NR)]):
            rf[i, j] = C.path(int(np.searchsorted(C.t, tr)), int(d[i]), float(sd[i]))[0]
    res = {"n": int(ok.sum()), "stop_med": float(np.median(sd[ok])) if ok.any() else None,
           "mfe_med": float(np.median(fav[ok])) if ok.any() else None,
           "rand_mfe_med": float(np.nanmedian(rf[ok])) if ok.any() else None}  # fmt: skip
    for X in XS:
        real = (fav[ok] >= X).astype(float)
        rnd = np.where(np.isnan(rf[ok]), np.nan, (rf[ok] >= X).astype(float))
        res[f"tb_{X}"] = DU.by_window(W, T[ok], real, rnd)
    return res


def diag_gate(cell):
    pl = cell.get("pool", {})
    if not pl.get("n") or pl.get("p") is None:
        return False
    big = [w for w in MAINW if cell[w].get("n", 0) >= 10]
    return pl["p"] * N_STUDY < 0.05 and len(big) >= 3 and all(cell[w]["ex"] > 0 for w in big) and pl["ex"] >= 0.03


# ---------------- 규칙(H2R·H3R) ----------------
_gate_orig = P.gate_ok


def gate_ok(spec, ctx, i, d):
    if spec["gate"] == "FLAG":
        return bool(ctx["flag"][i])
    return _gate_orig(spec, ctx, i, d)


P.gate_ok = gate_ok


def conv_pd(tr, t1):
    return [{**x, "T": int(x["T"]), "e": x["e"], "exits": [(int(t1[k]), px, w) for k, px, w in x["exits"]]} for x in tr]


def run_rule(S, F, sg, spec, ctx, W, t_end, cache):
    tr = P.build_trades(S, F, sg, spec, ctx)
    r = summ_all(tr, W, t_end)
    sel = [x for x in tr if W["fresh"][0] <= x["T"]]
    Rm = P.random_for(S, F, sel, spec, ctx, cache)
    Ti = np.array([x["T"] for x in sel])
    r["rand"] = {}
    for w in (*MAINW, "pool"):
        mk = (Ti >= W[w][0]) & (Ti < W[w][1]) if len(Ti) else np.zeros(0, bool)
        r["rand"][w] = P.rand_stats([x for x, k in zip(sel, mk) if k], Rm[mk]) if len(Ti) else None
    red = {}
    for rk, kw in (("lat1m", {"lat": 1}), ("lat1bar", {"lat": spec["tf"]}), ("slip+2", {"slip": P.SLIP + 0.0002}),
                   ("fee1.5", {"fee": 1.5})):  # fmt: skip
        t2 = P.build_trades(S, F, sg, spec, ctx, **kw)
        red[rk] = {w: P.summ(P.in_w(t2, *W[w]), *W[w]) for w in ("val", "year")}
    r["red"] = red
    return r, tr


def rule_verdict(r, p_holm):
    g1 = all(r[w]["n"] and r[w]["risk1"]["ret"] > 0 and r[w]["x5_40"]["ret"] > 0 and r[w]["x5_100"]["ret"] > 0
             and r[w]["x5_100"]["liq"] == 0 for w in MAINW)  # fmt: skip
    g2 = p_holm < 0.05 and all((r["rand"][w] or {}).get("ex_avg", -1) > 0 for w in MAINW)
    g3 = r["design"]["perday"] >= 0.03 and r["year"]["perday"] >= 0.03
    pl = r["pool"]
    g4 = bool(pl["n"]) and min(pl["long"], pl["short"]) >= 0.15 * pl["n"]
    g5 = all(r["red"][k][w]["n"] and r["red"][k][w]["risk1"]["ret"] > 0 for k in r["red"] for w in ("val", "year"))
    return {"g1": bool(g1), "g2": bool(g2), "g3": bool(g3), "g4": bool(g4), "g5": bool(g5), "holm": p_holm,
            "pass": bool(g1 and g2 and g3 and g4 and g5)}  # fmt: skip


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda z: z[1])
    run, out = 0.0, {}
    for i, (n, p) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * p))
        out[n] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(20261108)

    # 기본 규칙 자료(원래 코드 그대로)
    b, _ = RL.load_all(DW.R5)
    F = DW.features(b)
    t_end = int(b["t"][-1]) + M15
    m = U.load_1m(t_end)
    t_end = min(t_end, int(m["t"][-1]) + MIN)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    W = P.windows(t_end)
    # 다이버전스 사건(5분봉 → 15분·30분·1시간 등)
    b5 = P.load_5m(out)
    Fs = {"15": P.agg(b5, 15), "30": P.agg(b5, 30), "1h": P.agg(b5, 60), "4h": P.agg(b5, 240), "1d": P.agg(b5, 1440)}
    del b5
    for k in Fs:
        keep = Fs[k]["ct"] <= t_end
        for x in ("t", "o", "c", "h", "l", "v", "ct", "rsi", "cci", "vsma", "dir"):
            Fs[k][x] = Fs[k][x][keep]
    Fs["1d"]["t6d"] = P.t6d_daily(Fs["1d"])
    for k in ("15", "1h"):
        P.ribbon(Fs[k])
    evs = events(Fs, ("15", "30", "1h"))
    ev_h1 = merge(evs, ("30", "1h"))
    print("loaded", {k: len(v[0]) for k, v in evs.items()}, round(time.time() - t0), flush=True)
    res = {"t_end": t_end, "n_study": N_STUDY, "n_cum": N_CUM}
    trj = {}

    # ---------------- H1 ----------------
    gt = F["gate"]["GT"]
    gates = {"GT": gt, "ZP": gt, "GZ": gt}  # ZP는 안 쓴다(N4D·GT만), 자리만
    simN = LE.Sim(m, b, F, gates, fund_fn)
    sigsN = {"N4D": LE.nr_signals(b, m, 1440, 4)}
    RN = NRRunner(simN, sigsN, np.zeros(len(b["t"]), bool))
    vN = LE.SPEC["N4D·GT"]
    RN.kq_leg = {"N4D": RN.mfe_k(vN, "N4D", *W["design"])}
    sim2 = A2.Sim(m, b, fund_fn, F["atr_d"] / b["c"])
    sigD = A.signals(b, F)["DC24h|GT"]
    FS = SC.features(b)
    fams = {
        "NR": lambda allow, **kw: nr_trades(RN, vN, W["fresh"][0], t_end, allow, **kw),
        "DC": lambda allow, **kw: dc_trades(sim2, sigD, b, W["fresh"][0], t_end, allow, **kw),
        "SCR": lambda allow, **kw: scr_trades(b, FS, W["fresh"][0], t_end, allow, **kw),
    }
    red_kw = {"NR": {"lat1m": {"lat": 1}, "lat1bar": {"lat": 15}, "slip+2": {"slip": LE.SLIP + 0.0002}, "fee1.5": {"fee_mult": 1.5}},
              "DC": {"lat1m": {"lat": 1}, "lat1bar": {"lat": 15}, "slip+2": {"slip": A2.SLIP + 0.0002}, "fee1.5": {"fee_mult": 1.5}},
              "SCR": {"lat1bar": {"delay": 1}, "slip+2": {"slip": 0.0002}, "fee1.5": {"fee": SC.FEE * 1.5}}}  # fmt: skip
    h1, h1p = {}, {}
    for fam, fn in fams.items():
        base = fn(allow_fn(None, "base", 0))
        res[f"{fam}|base"] = summ_all(base, W, t_end)
        trj[f"{fam}|base"] = P.in_w(base, *W["d30"])
        print(fam, "base", " ".join(f"{w}:{res[f'{fam}|base'][w]['n']}/{res[f'{fam}|base'][w].get('avg', 0) * 100:+.3f}"
                                    for w in MAINW), round(time.time() - t0), flush=True)  # fmt: skip
        for kind in ("V", "C"):
            for N in (6, 12):
                name = f"{fam}|{kind}{N}"
                allow = allow_fn(ev_h1, kind, N)
                keep = np.array([allow(x["tdec"], x["d"]) for x in base], bool)
                eff = filter_effect(base, keep, W, rng)
                tr = fn(allow)
                r = summ_all(tr, W, t_end)
                r["effect"] = eff
                r["red"] = {}
                for rk, kw in red_kw[fam].items():
                    t2 = fn(allow, **kw)
                    r["red"][rk] = {w: P.summ(P.in_w(t2, *W[w]), *W[w]) for w in ("val", "year")}
                h1[name] = r
                h1p[name] = eff["pool"]["p_perm"] if eff["pool"]["p_perm"] is not None else 1.0
                trj[name] = P.in_w(tr, *W["d30"])
                pe = eff["pool"]
                print(name, f"keep {pe['n_keep']} drop {pe['n_drop']} dmean {100 * (pe['kd'] or {}).get('d_mean', 0):+.3f}%",
                      f"p {h1p[name]:.3f}", " ".join(f"{w}:{r[w]['n']}/{100 * r[w].get('risk1', {}).get('ret', 0):+.0f}%"
                                                     for w in MAINW), round(time.time() - t0), flush=True)  # fmt: skip
    hh = holm(h1p)
    for name, r in h1.items():
        eff = r["effect"]
        big = [w for w in MAINW if eff[w]["n_keep"] >= 5 and eff[w]["n_drop"] >= 5]
        g1 = hh[name] < 0.05 and len(big) >= 3 and all(eff[w]["kd"]["d_mean"] > 0 for w in big)
        g2 = all(r[w]["n"] and r[w]["risk1"]["ret"] > 0 and r[w]["x5_40"]["ret"] > 0 and r[w]["x5_100"]["ret"] > 0
                 and r[w]["x5_100"]["liq"] == 0 for w in MAINW)  # fmt: skip
        pl = r["pool"]
        g3 = bool(pl["n"]) and min(pl["long"], pl["short"]) >= 0.15 * pl["n"]
        g4 = all(r["red"][k][w]["n"] and r["red"][k][w]["risk1"]["ret"] > 0 for k in r["red"] for w in ("val", "year"))
        r["verdict"] = {"g1": bool(g1), "g2": bool(g2), "g3": bool(g3), "g4": bool(g4), "holm": hh[name],
                        "cum": min(1.0, h1p[name] * N_CUM), "pass": bool(g1 and g2 and g3 and g4)}  # fmt: skip
    res["h1"] = h1
    del simN, RN, sim2, FS

    # ---------------- 큰 움직임 표시 ----------------
    ad = LE.all_down(b).astype(np.int64)
    ad_t = b["t"] + M15
    cs = np.r_[0, np.cumsum(ad)]

    def flag24(T):
        T = np.asarray(T, np.int64)
        return (cs[np.searchsorted(ad_t, T, "right")] - cs[np.searchsorted(ad_t, T - 24 * HOUR, "left")]) > 0

    del b, F
    C = DU.Ctx(m)
    S = P.Sim(m, P.load_funding())
    t1 = m["t"]

    # ---------------- H2 ----------------
    F1 = Fs["1h"]
    s1 = DU.signals(F1, 2)
    fl = flag24(F1["ct"])
    sig_on = [(int(F1["ct"][s["c"]]), s["d"], s["ext"]) for s in s1 if fl[s["c"]]]
    sig_off = [(int(F1["ct"][s["c"]]), s["d"], s["ext"]) for s in s1 if not fl[s["c"]]]
    res["h2_flag_share"] = float(fl.mean())
    res["h2_diag"] = tb_diag(C, sig_on, F1["ct"], fl, W, rng, t_end)
    res["h2_diag_off"] = tb_diag(C, sig_off, F1["ct"], ~fl, W, rng, t_end)  # 설명용
    print("h2 diag", {X: res["h2_diag"][f"tb_{X}"]["pool"] for X in XS}, round(time.time() - t0), flush=True)
    ctx1 = P.ctx_for(F1, Fs)
    ctx1["flag"] = fl
    spec2 = {"exit": "X", "gate": "FLAG", "tier": "ALL", "be": 1.5, "tf": 60, "R": 2, "trail": 240}
    sg2 = [s for s in s1 if fl[s["c"]]]
    cache = {}
    rules, rtr = {}, {}
    rules["H2R"], tr2 = run_rule(S, F1, sg2, spec2, ctx1, W, t_end, cache)
    rtr["H2R"] = conv_pd(tr2, t1)
    print("H2R", " ".join(f"{w}:{rules['H2R'][w]['n']}/{100 * rules['H2R'][w].get('risk1', {}).get('ret', 0):+.0f}%"
                          for w in MAINW), round(time.time() - t0), flush=True)  # fmt: skip

    # ---------------- H3 ----------------
    F15 = Fs["15"]
    allev = sorted((int(T), int(d), float(e), k) for k, (Ts, Ds, Es) in evs.items() for T, d, e in zip(Ts, Ds, Es))
    h3, last = [], {1: -10**15, -1: -10**15}
    for j, (T, d, e, k) in enumerate(allev):
        win = [x for x in allev[max(0, j - 400) : j + 1] if x[1] == d and T - 2 * HOUR <= x[0] <= T]
        win += [x for x in allev[j + 1 : j + 6] if x[0] == T and x[1] == d]
        if len({x[3] for x in win}) < 2 or last[d] >= T - 2 * HOUR:
            continue
        last[d] = T
        ext = min(x[2] for x in win) if d == 1 else max(x[2] for x in win)
        h3.append((T, d, ext))
    res["h3_n"] = len(h3)
    res["h3_diag"] = tb_diag(C, h3, F15["ct"], np.ones(len(F15["ct"]), bool), W, rng, t_end)
    print("h3 diag", {X: res["h3_diag"][f"tb_{X}"]["pool"] for X in XS}, round(time.time() - t0), flush=True)
    dpass = {f"H2|{X}": diag_gate(res["h2_diag"][f"tb_{X}"]) for X in XS}
    dpass |= {f"H3|{X}": diag_gate(res["h3_diag"][f"tb_{X}"]) for X in XS}
    res["diag_pass"] = dpass
    if dpass["H3|0.01"] or dpass["H3|0.02"]:
        ix = {int(t): i for i, t in enumerate(F15["ct"])}
        sg3 = [{"p": ix[T], "c": ix[T], "d": d, "tier": "S", "ext": e} for T, d, e in h3 if T in ix]
        ctx15 = P.ctx_for(F15, Fs)
        spec3 = {"exit": "X", "gate": False, "tier": "ALL", "be": 1.5, "tf": 15, "R": 2, "trail": 60}
        rules["H3R"], tr3 = run_rule(S, F15, sg3, spec3, ctx15, W, t_end, cache)
        rtr["H3R"] = conv_pd(tr3, t1)
    ps = {n: (r["rand"]["pool"] or {}).get("p", 1.0) for n, r in rules.items()}
    # 등록: Holm(2). H3R을 계산하지 않으면 H2R은 ×2(센 수 그대로).
    hr = {n: min(1.0, ps[n] * 2) for n in ps} if len(ps) == 1 else holm(ps)
    for n, r in rules.items():
        r["verdict"] = rule_verdict(r, hr[n]) | {"cum": min(1.0, ps[n] * N_CUM)}
    res["rules"] = rules
    trj.update({n: P.in_w(tr, *W["d30"]) for n, tr in rtr.items()})
    trj["_full"] = {n: tr for n, tr in rtr.items()}
    res["secs"] = time.time() - t0
    (out / "result.json").write_text(json.dumps(res, default=float))
    (out / "trades.json").write_text(json.dumps(trj, default=float))
    print("done", {n: r["verdict"]["pass"] for n, r in h1.items()}, {n: r["verdict"] for n, r in rules.items()},
          dpass, round(time.time() - t0), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
