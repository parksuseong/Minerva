"""BTC 테이커 흐름 따라가기: 4시간 테이커 매수 비중 z가 쏠린 쪽으로 진입. 18개 변형 + 진단 + 1분봉 체결 + 판정.

등록: docs/backtests/2026-11-08-btc-taker-follow.md. 연구 전용(주문 없음).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/taker-follow-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/taker_follow.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import daily_wave as DW
import deep as DP
import nonprice_data as ND
import numpy as np
import ql_upgrade as U
import rr_asym as A
from dmi15_eval import frame
from htf_live import direction, to15
from r5_data import load

MIN, M15, DAY = U.MIN, U.M15, U.DAY
TAKER, TICK, SLIP = U.TAKER, U.TICK, DP.SLIP
N_BEFORE, N_THIS = 6432, 18
N_CUM = N_BEFORE + N_THIS
RATE = 1.5  # 설계 구간 새 넘김(롱+숏)/일
MAINW = ("fresh", "design", "val", "year")
SHOWW = MAINW + ("m6", "d30", "dv")
HOLDS = {"H4": 240, "H12": 720, "H24": 1440, "FLIP": 2880}
FEATS = ("T2h", "T4h", "T8h", "M4h", "MOM4h")
METRIC = ("M4h",)
# 이름: (특징, 청산, 손절, PF)
SPEC = {
    "P_T4h|H24|NS": ("T4h", "H24", "NS", False),
    "T2h|H24|NS": ("T2h", "H24", "NS", False),
    "T8h|H24|NS": ("T8h", "H24", "NS", False),
    "T4h|H4|NS": ("T4h", "H4", "NS", False),
    "T4h|H12|NS": ("T4h", "H12", "NS", False),
    "T4h|FLIP|NS": ("T4h", "FLIP", "NS", False),
    "T4h|H24|ATR": ("T4h", "H24", "ATR", False),
    "T4h|H24|NS|PF": ("T4h", "H24", "NS", True),
    "T4h|H24|ATR|PF": ("T4h", "H24", "ATR", True),
    "T8h|H24|ATR": ("T8h", "H24", "ATR", False),
    "T2h|H12|NS": ("T2h", "H12", "NS", False),
    "T8h|FLIP|NS": ("T8h", "FLIP", "NS", False),
    "T4h|FLIP|ATR|PF": ("T4h", "FLIP", "ATR", True),
    "M4h|H24|NS": ("M4h", "H24", "NS", False),
    "M4h|H24|ATR": ("M4h", "H24", "ATR", False),
    "M4h|FLIP|NS": ("M4h", "FLIP", "NS", False),
    "MOM4h|H24|NS": ("MOM4h", "H24", "NS", False),
    "MOM4h|H24|ATR": ("MOM4h", "H24", "ATR", False),
}
CONTROL = ("MOM4h|H24|NS", "MOM4h|H24|ATR")
PRIMARY = "P_T4h|H24|NS"
assert len(SPEC) == N_THIS


# ---------------- 특징 ----------------
def metrics_share(b) -> np.ndarray:
    """metrics 5분 테이커 롱숏 비율 → 매수 비중, 최근 48개(4시간) 평균, create_time + 5분 ≤ 판단 시각."""
    mt, mv, info = ND.load_metrics()
    rho = mv[:, 4]
    sh = rho / (1 + rho)
    ok = np.isfinite(sh)
    mt, sh = mt[ok], sh[ok]
    c = np.r_[0.0, np.cumsum(sh)]
    mean48 = np.full(len(sh), np.nan)
    mean48[47:] = (c[48:] - c[:-48]) / 48
    span = np.full(len(sh), np.inf)
    span[47:] = mt[47:] - mt[:-47]
    mean48[span > 4 * 3_600_000 + 10 * MIN] = np.nan  # 빈 곳이 있으면 버림
    q = b["t"] + M15 - ND.M5
    return ND.asof(mt, mean48, q, 20 * MIN), info


def crossings(z: np.ndarray, th: float):
    with np.errstate(invalid="ignore"):
        up, dn = z > th, z < -th
    upf = up & ~np.r_[False, up[:-1]]
    dnf = dn & ~np.r_[False, dn[:-1]]
    return upf, dnf


def calib(z: np.ndarray, dz: np.ndarray, days: float) -> float:
    lo, hi = 0.2, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        u, d = crossings(z, mid)
        rate = (u[dz].sum() + d[dz].sum()) / days
        lo, hi = (mid, hi) if rate > RATE else (lo, mid)
    return (lo + hi) / 2


def build(b) -> dict:
    n = len(b["c"])
    t, h, lo, c = b["t"], b["h"], b["l"], b["c"]
    v, tb = b["v"].astype(float), b["tbv"].astype(float)
    W = A.windows(int(t[-1]) + M15)
    dz = (t >= W["design"][0]) & (t < W["design"][1])
    days = (W["design"][1] - W["design"][0]) / DAY
    Z = {}
    for nm, N in (("T2h", 8), ("T4h", 16), ("T8h", 32)):
        with np.errstate(all="ignore"):
            s = DP.rmean(tb, N) / DP.rmean(v, N)
        Z[nm] = ND.roll_z(s)
    msh, minfo = metrics_share(b)
    Z["M4h"] = ND.roll_z(msh)
    lr4 = np.r_[np.full(16, np.nan), np.log(c[16:] / c[:-16])]
    lr24 = np.r_[np.full(96, np.nan), np.log(c[96:] / c[:-96])]
    Z["MOM4h"] = ND.roll_z(lr4)
    zR24 = ND.roll_z(lr24)
    th = {k: calib(Z[k], dz, days) for k in FEATS}
    sig = {}
    for k in FEATS:
        u, d = crossings(Z[k], th[k])
        sig[k] = (u, d)
        with np.errstate(invalid="ignore"):
            sig[k + "|PF"] = (u & (lr4 > 0), d & (lr4 < 0))
    F1, F4, D = frame(b, 60), frame(b, 240), frame(b, 1440)
    pc = np.r_[F1["c"][0], F1["c"][:-1]]
    trr = np.maximum(F1["h"] - F1["l"], np.maximum(abs(F1["h"] - pc), abs(F1["l"] - pc)))
    atr = to15(F1, DP.rmean(trr, 14), n)
    stop_atr = np.clip(3 * atr / c, 0.005, 0.05)
    d15 = direction(h, lo, c).astype(float)
    d4 = np.nan_to_num(to15(F4, direction(F4["h"], F4["l"], F4["c"]).astype(float), n), nan=0.0)
    d1 = np.nan_to_num(to15(D, direction(D["h"], D["l"], D["c"]).astype(float), n), nan=0.0)
    chase_l = np.where((d15 == 1) & (d4 == 1) & (d1 == 1), 0.03, 0.01)
    chase_s = np.where((d15 == -1) & (d4 == -1) & (d1 == -1), 0.03, 0.01)
    return {"Z": Z, "zR24": zR24, "lr4": lr4, "th": th, "sig": sig, "stop_atr": stop_atr,
            "chase": (chase_l, chase_s), "F4": F4, "minfo": minfo, "W": W}  # fmt: skip


def next_flip(z: np.ndarray):
    """i 다음(> i) z<0 인 첫 봉, z>0 인 첫 봉."""
    n = len(z)
    nneg, npos = np.full(n, n), np.full(n, n)
    with np.errstate(invalid="ignore"):
        neg, pos = z < 0, z > 0
    a, bb = n, n
    for i in range(n - 1, -1, -1):
        nneg[i], npos[i] = a, bb
        if neg[i]:
            a = i
        if pos[i]:
            bb = i
    return nneg, npos


# ---------------- 실행기 ----------------
class Runner:
    def __init__(self, sim: DP.Sim, F: dict):
        self.sim, self.F = sim, F
        self.flip = {k: next_flip(F["Z"][k]) for k in FEATS}

    def one(self, i, d, k, feat, hold_nm, stop_nm, fee_mult, slip, lat, cap=None):
        sim = self.sim
        ep = float(sim.m["o"][k]) + d * TICK
        sp = float(self.F["stop_atr"][i]) if stop_nm == "ATR" else 0.9
        if not np.isfinite(sp):
            return None
        if hold_nm == "FLIP":
            nneg, npos = self.flip[feat]
            j = int(nneg[i] if d == 1 else npos[i])
            if j < len(sim.b["t"]):
                kj = sim.kmin(j, lat)
                hold = min(kj - 1 - k, HOLDS["FLIP"]) if kj > 0 else HOLDS["FLIP"]
            else:
                hold = HOLDS["FLIP"]
        else:
            hold = HOLDS[hold_nm]
        if cap is not None:
            hold = min(hold, cap)
        hold = max(hold, 0)
        x = sim.exit_(k, d, ep, ep * (1 - d * sp), "T", hold, TAKER * fee_mult + slip, fee_mult, slip)
        x["i15"], x["_sp"], x["_hold"] = int(i), sp, int(hold)
        return x

    def run(self, name, lat=0, fee_mult=1.0, slip=SLIP):
        feat, hold_nm, stop_nm, pf = SPEC[name]
        u, dn = self.F["sig"][feat + ("|PF" if pf else "")]
        idx = np.nonzero(u | dn)[0]
        t, c = self.sim.b["t"], self.sim.b["c"]
        a0 = A.ms("2020-03-01")
        out, cur = [], None
        for i in idx:
            if t[i] < a0:
                continue
            d = 1 if u[i] else -1
            k = self.sim.kmin(int(i), lat)
            if k < 0:
                continue
            if cur is not None and k <= cur["x"]:
                if d == cur["d"]:
                    continue
                # 반대 신호: 앞 분 종가에 청산하고 뒤집는다(체결 가능할 때만)
            ep = float(self.sim.m["o"][k]) + d * TICK
            lim = self.F["chase"][0 if d == 1 else 1][i]
            if abs(ep / float(c[i]) - 1) >= lim:
                continue
            if cur is not None and k <= cur["x"]:
                y = self.one(cur["i15"], cur["d"], cur["k"], feat, hold_nm, stop_nm, fee_mult, slip, lat,
                             cap=k - 1 - cur["k"])  # fmt: skip
                out[-1] = y
            x = self.one(int(i), d, k, feat, hold_nm, stop_nm, fee_mult, slip, lat)
            if x is None:
                continue
            out.append(x)
            cur = x
        return out


# ---------------- 무작위 ----------------
def random_match(sim, tr, a, z, nsim, seed):
    rng = np.random.default_rng(seed)
    pool = DP.slot_pool(sim.b, a, z, (HOLDS["FLIP"] + 60) * MIN + M15)
    t = sim.b["t"]
    spec = [(int(t[x["i15"]] % DAY), x["d"], x["_sp"], x["_hold"]) for x in tr]
    vals = []
    for _ in range(nsim):
        rs = []
        for s, d, sp, hold in spec:
            for _try in range(10):
                k = sim.kmin(int(rng.choice(pool[s])))
                if k < 0:
                    continue
                ep = float(sim.m["o"][k]) + d * TICK
                x = sim.exit_(k, d, ep, ep * (1 - d * sp), "T", hold, TAKER + SLIP, 1.0, SLIP)
                rs.append(x["r"])
                break
        vals.append(np.mean(rs))
    real = float(np.mean([x["r"] for x in tr]))
    zz, pz = DP.zp(real, vals)
    return {"real": real, "rnd_mean": float(np.mean(vals)), "rnd_sd": float(np.std(vals)), "z": zz, "p_norm": pz,
            "p_bonf": min(1.0, pz * N_CUM), "p_emp": float((np.sum(np.array(vals) >= real) + 1) / (nsim + 1)),
            "n": len(tr)}  # fmt: skip


# ---------------- 진단 ----------------
def fwd_table(sim) -> tuple:
    """모든 15분봉 i: 다음 분 시가 대비 1·4·12·24시간 앞 수익. 못 구하면 NaN."""
    b, m = sim.b, sim.m
    ts = b["t"] + M15
    k = np.searchsorted(m["t"], ts)
    ok = (k < sim.n1 - 1441) & (np.abs(m["t"][np.minimum(k, sim.n1 - 1)] - ts) <= 5 * MIN)
    k = np.where(ok, k, 0)
    o = m["o"][k]
    out = {}
    for hh in (60, 240, 720, 1440):
        r = m["c"][k + hh - 1] / o - 1
        out[hh] = np.where(ok, r, np.nan)
    return out


def diag1(sim, F, fw, periods, rng) -> dict:
    t = sim.b["t"]
    res = {}
    for key in [*FEATS, *(f + "|PF" for f in FEATS)]:
        u, dn = F["sig"][key]
        for pn, (a, z) in periods.items():
            sel = (t >= a) & (t + DAY + M15 < z)
            ii = np.nonzero((u | dn) & sel)[0]
            ii = ii[np.isfinite(fw[1440][ii])]
            if len(ii) < 10:
                continue
            dd = np.where(u[ii], 1.0, -1.0)
            pool = DP.slot_pool(sim.b, a, z, DAY + 2 * M15)
            rec = {"n": int(len(ii)), "per_day": len(ii) / ((z - a) / DAY), "nl": int((dd == 1).sum()),
                   "ns": int((dd == -1).sum())}  # fmt: skip
            # 신호당 무작위 20개(같은 UTC 칸) 평균
            rnd_idx = np.array([rng.choice(pool[int(t[i] % DAY)], 20) for i in ii])
            for hh, nm in ((60, "1h"), (240, "4h"), (720, "12h"), (1440, "24h")):
                s = dd * fw[hh][ii]
                q = dd * np.nanmean(fw[hh][rnd_idx], axis=1)
                diff = s - q
                diff = diff[np.isfinite(diff)]
                se = diff.std() / math.sqrt(len(diff))
                rec[f"sig{nm}"] = float(np.nanmean(s))
                rec[f"rnd{nm}"] = float(np.nanmean(q))
                rec[f"t{nm}"] = float(diff.mean() / se) if se > 0 else 0.0
            res[f"{key}|{pn}"] = rec
    return res


def nw_ols(y, X, lag=6):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    iX = np.linalg.inv(XtX)
    V = iX @ S @ iX
    return beta, beta / np.sqrt(np.diag(V))


def diag2(sim, F, fw, periods) -> dict:
    t = sim.b["t"]
    idx = F["F4"]["last15"]
    y_all = np.log1p(fw[1440])
    out = {}
    for feat in ("T4h", "M4h", "T2h", "T8h"):
        for pn, (a, z) in periods.items():
            ii = idx[(t[idx] >= a) & (t[idx] + DAY + M15 < z)]
            y = y_all[ii]
            xs = [F["Z"][feat][ii], F["Z"]["MOM4h"][ii], F["zR24"][ii]]
            ok = np.isfinite(y) & np.all([np.isfinite(x) for x in xs], axis=0)
            if ok.sum() < 100:
                continue
            X = np.column_stack([np.ones(ok.sum())] + [x[ok] for x in xs])
            beta, tt = nw_ols(y[ok], X)
            X0 = np.column_stack([np.ones(ok.sum()), xs[0][ok]])
            b0, t0 = nw_ols(y[ok], X0)
            out[f"{feat}|{pn}"] = {"n": int(ok.sum()), "b_feat": float(beta[1]), "t_feat": float(tt[1]),
                                   "t_mom4h": float(tt[2]), "t_mom24h": float(tt[3]), "t_feat_alone": float(t0[1]),
                                   "corr_mom4h": float(np.corrcoef(xs[0][ok], xs[1][ok])[0, 1])}  # fmt: skip
    return out


def diag3(F) -> dict:
    """P 신호(zT_4h)와 MOM 신호가 ±1시간 안에서 같은 방향으로 겹치는 비율."""
    u, d = F["sig"]["T4h"]
    mu, md_ = F["sig"]["MOM4h"]
    out = {}
    for nm, a, b2 in (("long", u, mu), ("short", d, md_)):
        near = b2.copy()
        for j in range(1, 5):
            near[j:] |= b2[:-j]
            near[:-j] |= b2[j:]
        out[nm] = float(near[a].mean()) if a.sum() else None
    return out


# ---------------- 실행 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = load(DW.R5)
    t_end = int(b["t"][-1]) + M15
    m = U.load_1m(t_end)
    ft, fr = b["ft"].astype(np.int64), b["fr"].astype(float)

    def fund_fn(t_in, t_out, d):
        a_, z_ = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        return -d * float(fr[a_:z_].sum())

    sim = DP.Sim(m, b, fund_fn)
    F = build(b)
    W = F["W"]
    W["dv"] = (W["design"][0], W["val"][1])
    print("loaded", len(m["t"]), {k: round(v, 3) for k, v in F["th"].items()}, F["minfo"], round(time.time() - t0),
          flush=True)  # fmt: skip
    periods = {w: W[w] for w in MAINW + ("m6", "d30")}
    for y in range(2020, 2027):
        periods[str(y)] = (max(A.ms(f"{y}-01-01"), A.ms("2020-03-01")), min(A.ms(f"{y + 1}-01-01"), t_end))
    rng = np.random.default_rng(11)
    fw = fwd_table(sim)
    d1 = diag1(sim, F, fw, periods, rng)
    for k, r in d1.items():
        print("D1", k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in r.items()}, flush=True)
    d2 = diag2(sim, F, fw, periods)
    for k, r in d2.items():
        print("D2", k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in r.items()}, flush=True)
    d3 = diag3(F)
    print("D3 overlap", d3, flush=True)
    json.dump({"D1": d1, "D2": d2, "D3": d3, "th": F["th"]}, (out / "diag.json").open("w"), default=float)
    del fw

    run = Runner(sim, F)
    res, trades = {}, {}

    def evaluate(tr):
        return {w: DP.summ(DP.slice_w(tr, *W[w]), *W[w]) for w in SHOWW}

    for name in SPEC:
        tr = run.run(name)
        trades[name] = tr
        res[name] = evaluate(tr)
        r = res[name]
        print(name, {w: (r[w]["n"], round(r[w]["per_day"], 2), r[w]["win"] and round(r[w]["win"], 2),
                         r[w]["avg1"] and round(r[w]["avg1"] * 100, 3),
                         round(r[w]["x5_f1.0"]["ret"], 2), round(r[w]["x5_f0.4"]["ret"], 2), r[w]["x5_f1.0"]["liq"])
                     for w in MAINW + ("d30",)}, round(time.time() - t0), flush=True)  # fmt: skip

    cands = [n for n in SPEC if n not in CONTROL]
    ok = [n for n in cands if res[n]["design"]["per_day"] >= 0.8] or cands
    best = max(ok, key=lambda n: res[n]["design"]["avg1"] or -1)
    finals = [PRIMARY] + ([best] if best != PRIMARY else [])
    print("FINALS", finals, flush=True)
    detail = {}
    for fn in finals:
        feat, hold_nm, stop_nm, _pf = SPEC[fn]
        dv = DP.slice_w(trades[fn], *W["dv"])
        rnd = random_match(sim, dv, *W["dv"], args.nsim, 7) if len(dv) >= 5 else None
        print("rnd", fn, rnd, round(time.time() - t0), flush=True)
        red = {}
        for nm, kw in (("lat1", {"lat": 1}), ("lat5", {"lat": 5}), ("lat15", {"lat": 15}),
                       ("slip+0.02", {"slip": SLIP + 0.0002}), ("fee1.5", {"fee_mult": 1.5})):  # fmt: skip
            tr = run.run(fn, **kw)
            red[nm] = {w: DP.summ(DP.slice_w(tr, *W[w]), *W[w]) for w in MAINW + ("m6", "d30")}
            print("red", fn, nm, {w: round(red[nm][w]["x5_f1.0"]["ret"], 2) for w in red[nm]}, flush=True)
        years = {}
        for y in range(2020, 2027):
            a_, z_ = periods[str(y)]
            years[y] = DP.summ(DP.slice_w(trades[fn], a_, z_), a_, z_)
        print("years", fn, {y: (v["n"], round(v["x5_f1.0"]["ret"], 2), round(v["x5_f0.4"]["ret"], 2))
                            for y, v in years.items()}, flush=True)  # fmt: skip
        fx = res[fn]
        mw = [w for w in MAINW if not (feat in METRIC and w == "fresh")]
        g1 = all(fx[w]["per_day"] >= 0.8 for w in ("design", "val", "year"))
        g2 = all(fx[w][f"x5_f{f}"]["ret"] > 0 for w in mw for f in (1.0, 0.4))
        g3 = all(fx[w]["x5_f1.0"]["liq"] == 0 for w in mw)
        g4 = rnd is not None and rnd["real"] > rnd["rnd_mean"] and rnd["p_norm"] * N_CUM < 0.05
        g5 = all(red[k][w]["x5_f1.0"]["ret"] > 0 for k in red for w in ("val", "year"))
        dfeat = "M4h" if feat in METRIC else feat
        tv = [d2[f"{dfeat}|{w}"]["t_feat"] for w in mw if f"{dfeat}|{w}" in d2]
        g6a = sum(x >= 2 for x in tv) >= 3 and all(x > -2 for x in tv)
        ctrl = "MOM4h|H24|ATR" if stop_nm == "ATR" else "MOM4h|H24|NS"
        g6b = (fx["dv"]["avg1"] or -1) > (res[ctrl]["dv"]["avg1"] or -1)
        verdict = {"g1_freq": g1, "g2_all_windows": g2, "g3_no_liq": g3, "g4_random_bonf": g4, "g5_red": g5,
                   "g6a_partial_t": g6a, "g6b_beats_mom": g6b, "t_feat": tv, "ctrl": ctrl,
                   "pass": bool(g1 and g2 and g3 and g4 and g5 and g6a and g6b)}  # fmt: skip
        print("VERDICT", fn, verdict, flush=True)
        detail[fn] = {"rnd": rnd, "red": red, "years": years, "verdict": verdict}

    n_pos = {w: sum(res[n][w]["x5_f1.0"]["ret"] > 0 for n in SPEC) for w in MAINW}
    json.dump({"windows": W, "spec": SPEC, "res": res, "finals": finals, "detail": detail, "n_pos": n_pos,
               "th": F["th"], "n_this": N_THIS, "n_cum": N_CUM}, (out / "result.json").open("w"), default=float)  # fmt: skip
    json.dump({fn: {w: DP.slice_w(trades[fn], *W[w]) for w in SHOWW} for fn in set(finals) | set(CONTROL)},
              (out / "trades.json").open("w"))  # fmt: skip
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
