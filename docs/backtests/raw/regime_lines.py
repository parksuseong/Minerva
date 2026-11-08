"""큰 구간 선(상승·하락·횡보) 비율 닮은꼴. 등록: 2026-11-08-btc-regime-lines-hypotheses.md

4시간 로그 종가를 큰 구간 직선(지그재그 Z5·Z8·Z12, 선 맞추기 L1.5·L3)으로 나누고, 마지막 k다리의 높이·기간 비율이
닮은 과거 사건 20개가 그 다음 구간에서 어디로 갔는지로 방향을 정한다. 구간 끝은 확인된 뒤에만 쓴다.
매매: R(큰 구간 직접) 20개 + G(단타 방향 문: 뉴욕 돌파·15분 SCR 확인) 20개 = 40개.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/regime-lines-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/regime_lines.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import daily_wave as DW
import numpy as np
from pine_eval import lev_path
from r5_data import load

M15, DAY, H4 = 900_000, 86_400_000, 14_400_000
SPOT_CSV = Path("~/.minerva/data/btc/15m.csv").expanduser()
SEGS = {"Z5": ("z", 0.05), "Z8": ("z", 0.08), "Z12": ("z", 0.12), "L1.5": ("l", 1.5), "L3": ("l", 3.0)}
KS = (3, 5)
K_NB, MIN_LIB = 20, 10
Q = 0.6
N_VAR = 40
N_TONIGHT_BEFORE = 5859
MAIN = ("fresh", "design", "val", "year")


# ---------------- 자료 ----------------
def load_all(r5: Path) -> tuple[dict, dict]:
    """선물 15분봉 b(매매용)와 4시간봉 s(현물 2017-09~ + 선물). s["i15"] = 4시간봉 마지막 15분봉의 b 위치(선물 전 -1)."""
    b = load(r5)
    raw = np.genfromtxt(SPOT_CSV, delimiter=",", skip_header=1, usecols=(0, 2, 3, 4))
    raw = raw[raw[:, 0] < b["t"][0]]
    t = np.r_[raw[:, 0].astype(np.int64), b["t"].astype(np.int64)]
    h, lo, c = np.r_[raw[:, 1], b["h"]], np.r_[raw[:, 2], b["l"]], np.r_[raw[:, 3], b["c"]]
    k4 = t // H4
    starts = np.r_[0, np.nonzero(np.diff(k4))[0] + 1]
    ends = np.r_[starts[1:] - 1, len(t) - 1]
    s = {"t": k4[starts] * H4, "c": c[ends], "h": np.maximum.reduceat(h, starts),
         "l": np.minimum.reduceat(lo, starts)}  # fmt: skip
    nsp = len(raw)
    s["i15"] = np.where(ends >= nsp, ends - nsp, -1)
    s["x"] = np.log(s["c"])
    # 일봉 ATR%(14), 전날까지 마감된 일봉 값만
    kd = t // DAY
    ds = np.r_[0, np.nonzero(np.diff(kd))[0] + 1]
    de = np.r_[ds[1:] - 1, len(t) - 1]
    dh, dl, dc = np.maximum.reduceat(h, ds), np.minimum.reduceat(lo, ds), c[de]
    pc = np.r_[dc[0], dc[:-1]]
    tr = np.maximum(dh - dl, np.maximum(np.abs(dh - pc), np.abs(dl - pc))) / pc
    atr = np.empty_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / 14
    dday = kd[ds]
    pos = np.searchsorted(dday, s["t"] // DAY) - 1  # 전날
    s["atrd"] = atr[np.clip(pos, 0, None)]
    return b, s


# ---------------- 구간 나누기 ----------------
def seg_zigzag(x: np.ndarray, th: float) -> list:
    """(극점 위치, 확정 위치) 목록. 종가 기준, 극점에서 log(1+th) 되돌리면 확정."""
    d = np.log1p(th)
    out, mode, hi, lo = [], 0, 0, 0
    for i in range(1, len(x)):
        if mode == 0:
            hi = i if x[i] > x[hi] else hi
            lo = i if x[i] < x[lo] else lo
            if x[i] <= x[hi] - d:
                out.append((hi, i))
                mode, lo = -1, i
            elif x[i] >= x[lo] + d:
                out.append((lo, i))
                mode, hi = 1, i
        elif mode == 1:
            if x[i] > x[hi]:
                hi = i
            elif x[i] <= x[hi] - d:
                out.append((hi, i))
                mode, lo = -1, i
        else:
            if x[i] < x[lo]:
                lo = i
            elif x[i] >= x[lo] + d:
                out.append((lo, i))
                mode, hi = 1, i
    return out


def seg_lines(x: np.ndarray, atrd: np.ndarray, k: float) -> list:
    """선 맞추기: 마지막 꺾임점→지금 직선에서 k×일봉 ATR%(로그)보다 먼 점이 생기면 가장 먼 점을 꺾임점으로 확정."""
    out, s = [(0, 0)], 0
    for i in range(2, len(x)):
        while i - s >= 2:
            seg = x[s : i + 1]
            ln = np.linspace(seg[0], seg[-1], len(seg))
            dev = np.abs(seg - ln)
            j = int(dev.argmax())
            if dev[j] <= k * np.log1p(atrd[i]):
                break
            s = s + j
            out.append((s, i))
    return out


def events(s: dict, name: str) -> dict:
    kind, p = SEGS[name]
    x = s["x"]
    ev = seg_zigzag(x, p) if kind == "z" else seg_lines(x, s["atrd"], p)
    # 극점 순서가 어긋난 것(같은 위치 반복) 정리
    clean = []
    for pv, cf in ev:
        if clean and pv <= clean[-1][0]:
            continue
        clean.append((pv, cf))
    P = np.array([e[0] for e in clean])
    C = np.array([e[1] for e in clean])
    m = np.r_[np.nan, np.diff(x[P])]
    dur = np.r_[np.nan, np.diff(P)] / 6.0  # 일
    z = np.abs(m) / (np.log1p(s["atrd"][P]) * np.sqrt(np.maximum(dur, 1 / 6)))
    lab = np.where(z < 1, 0, np.sign(m)).astype(int)  # 1 상승, -1 하락, 0 횡보
    lab[0] = 0
    n = len(P)
    # 결과: 확정 시점 이후에 끝나는 첫 구간 끝까지
    out_end = np.full(n, -1)
    for e in range(n):
        m_ = e + 1
        while m_ < n and P[m_] <= C[e]:
            m_ += 1
        out_end[e] = m_ if m_ < n else -1
    g = np.where(out_end >= 0, x[P[np.maximum(out_end, 0)]] - x[C], np.nan)
    avail = np.where(out_end >= 0, C[np.maximum(out_end, 0)], 10**9)  # 결과가 확정되는 4시간봉 위치
    hold = np.where(out_end >= 0, (P[np.maximum(out_end, 0)] - C) * 16, 0)  # 15분봉 수
    return {"P": P, "C": C, "m": m, "dur": dur, "lab": lab, "g": g, "avail": avail, "hold": hold}


def feats(E: dict, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """사건별 특징(다리 k개 × [이름표 3, 높이비, 부호이동/평균, 0.5×기간비] + 진행 다리 2)과 이름표 열쇠."""
    n = len(E["P"])
    m, dur, lab = E["m"], E["dur"], E["lab"]
    F = np.full((n, k, 6), np.nan)
    prog = np.full((n, 2), np.nan)
    key = np.full((n, k + 1), 9, dtype=int)
    for e in range(k + 1, n):
        mm, dd, ll = m[e - k + 1 : e + 1], dur[e - k + 1 : e + 1], lab[e - k + 1 : e + 1]
        mp, dp = m[e - k : e], dur[e - k : e]
        sc = np.mean(np.abs(mm)) + 1e-9
        F[e, :, 0] = ll == 1
        F[e, :, 1] = ll == -1
        F[e, :, 2] = ll == 0
        F[e, :, 3] = np.log((np.abs(mm) + 1e-6) / (np.abs(mp) + 1e-6))
        F[e, :, 4] = mm / sc
        F[e, :, 5] = 0.5 * np.log((dd + 1e-3) / (dp + 1e-3))
        pm = x[E["C"][e]] - x[E["P"][e]]
        prog[e] = (pm / sc, np.log((abs(pm) + 1e-6) / (abs(mm[-1]) + 1e-6)))
        key[e, :k] = ll
        key[e, k] = int(np.sign(pm))
    return F, prog, key


def dist_E(F, prog, key, e, lib) -> np.ndarray:
    a = np.c_[F[lib][:, :, 3:].reshape(len(lib), -1), prog[lib]]
    q = np.r_[F[e][:, 3:].ravel(), prog[e]]
    d = np.sqrt(((a - q) ** 2).sum(1))
    mis = (key[lib] != key[e]).sum(1)
    if (mis == 0).sum() >= MIN_LIB:
        return np.where(mis == 0, d, np.inf)
    return d + mis


def dist_D(F, prog, key, e, lib) -> np.ndarray:
    """다리 단위 DTW(띠 ±1), 비용 = 특징 차(높이비·이동·기간) + 이름표 다르면 1. 진행 다리는 끝에 더한다."""
    A = F[lib][:, :, 3:]
    Bq = F[e][:, 3:]
    k = A.shape[1]
    cost = np.sqrt(((A[:, :, None, :] - Bq[None, None, :, :]) ** 2).sum(-1))
    cost += (F[lib][:, :, None, :3].argmax(-1) != F[e][None, None, :, :3].argmax(-1)).astype(float)
    D = np.full((len(lib), k + 1, k + 1), np.inf)
    D[:, 0, 0] = 0
    for i in range(1, k + 1):
        for j in range(max(1, i - 1), min(k, i + 1) + 1):
            D[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    pd = np.sqrt(((prog[lib] - prog[e]) ** 2).sum(1)) + (key[lib][:, -1] != key[e][-1])
    return D[:, k, k] + pd


def predict(s: dict, name: str, k: int, sim: str) -> dict:
    E = events(s, name)
    F, prog, key = feats(E, s["x"], k)
    n = len(E["P"])
    pr = {"p": np.full(n, np.nan), "med": np.full(n, np.nan), "tp_l": np.full(n, np.nan),
          "tp_s": np.full(n, np.nan), "hold": np.full(n, np.nan), "nb": [None] * n}  # fmt: skip
    ok = ~np.isnan(F[:, 0, 0]) & ~np.isnan(E["g"])
    for e in range(k + 1, n):
        if np.isnan(F[e, 0, 0]):
            continue
        lib = np.nonzero(ok & (E["avail"] < E["C"][e]))[0]
        if len(lib) < MIN_LIB:
            continue
        d = dist_E(F, prog, key, e, lib) if sim == "E" else dist_D(F, prog, key, e, lib)
        o = np.argsort(d)[:K_NB]
        o = o[np.isfinite(d[o])]
        if len(o) < MIN_LIB:
            continue
        nb = lib[o]
        g = E["g"][nb]
        pr["p"][e] = (g > 0).mean()
        pr["med"][e] = np.median(g)
        up, dn = g[g > 0], g[g < 0]
        pr["tp_l"][e] = np.median(up) if len(up) else np.nan
        pr["tp_s"][e] = -np.median(dn) if len(dn) else np.nan
        pr["hold"][e] = np.median(E["hold"][nb])
        pr["nb"][e] = nb.tolist()
    return {"E": E, "pr": pr}


# ---------------- 매매 후보 ----------------
def direction_of(pr: dict, e: int) -> int:
    p, med = pr["p"][e], pr["med"][e]
    if np.isnan(p):
        return 0
    if p >= Q and med > 0.01:
        return 1
    if p <= 1 - Q and med < -0.01:
        return -1
    return 0


def regime_cands(b: dict, s: dict, R: dict) -> tuple[list, np.ndarray]:
    E, pr = R["E"], R["pr"]
    x = s["x"]
    c = b["c"]
    n15 = len(c)
    cands, gate = [], np.zeros(n15)
    evs = [(e, int(s["i15"][E["C"][e]])) for e in range(len(E["P"])) if s["i15"][E["C"][e]] >= 0]
    for idx, (e, i) in enumerate(evs):
        d = direction_of(pr, e)
        nxt = evs[idx + 1][1] if idx + 1 < len(evs) else n15
        if d:
            gate[i + 1 : nxt + 1] = d
        if not d or i >= n15 - 2:
            continue
        tpd = pr["tp_l"][e] if d == 1 else pr["tp_s"][e]
        tpd = float(np.clip(tpd if np.isfinite(tpd) else 0.01, 0.01, 0.15))
        piv = x[E["P"][e]]
        adverse = (piv < x[E["C"][e]]) if d == 1 else (piv > x[E["C"][e]])
        sd = float(np.clip(abs(x[E["C"][e]] - piv), 0.01, 0.04)) if adverse else 0.04
        hold = int(min(max(pr["hold"][e], 16), 30 * 96))
        cands.append(DW.cand(i, d, c[i] * np.exp(-d * sd), c[i] * np.exp(d * tpd), min(i + hold, n15 - 1)))
    return cands, gate


# ---------------- 평가 ----------------
def windows(t) -> dict:
    W = DW.windows(t)
    end = int(t[-1]) + M15
    W["prev30"] = (int(np.searchsorted(t, end - 60 * DAY)), int(np.searchsorted(t, end - 30 * DAY)))
    W["mid"] = (W["design"][0], W["val"][1])
    return W


def summ(tr, a, z, b) -> dict:
    return DW.summ(tr, a, z, b)


def flip_random(b, F, cands, a, z, real_tr, nsim, seed) -> dict:
    """방향만 섞은 무작위: 같은 신호 시점·익절/손절 거리, 방향은 동전."""
    rng = np.random.default_rng(seed)
    c = b["c"]
    vals = []
    for _ in range(nsim):
        rc = []
        for x in cands:
            if rng.random() < 0.5:
                rc.append(x)
            else:
                ci = c[x["i"]]
                rc.append(DW.cand(x["i"], -x["d"], ci * ci / x["stop"], ci * ci / x["tp"], x["tmax"]))
        tr, _ = DW.simulate(b, F, rc, "G0", "FA", a, z)
        vals.append(lev_path(tr, DW.LEV)["ret"])
    vals = np.array(vals)
    real = lev_path(real_tr, DW.LEV)["ret"]
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "real": float(real), "nsim": nsim}


def count_regimes(s: dict) -> dict:
    out = {}
    for name in SEGS:
        E = events(s, name)
        lab, dur = E["lab"][1:], E["dur"][1:]
        tot = dur.sum()
        out[name] = {"legs": int(len(lab)),
                     "count_share": {k: float((lab == v).mean()) for k, v in (("up", 1), ("down", -1), ("side", 0))},
                     "time_share": {k: float(dur[lab == v].sum() / tot) for k, v in (("up", 1), ("down", -1), ("side", 0))},
                     "median_days": float(np.median(dur)),
                     "median_move": float(np.median(np.abs(np.expm1(E["m"][1:]))))}  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=100)
    ap.add_argument("--nsim-big", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b, s = load_all(DW.R5)
    F = DW.features(b)
    C = DW.all_cands(b, F)
    W = windows(b["t"])
    print("data", len(b["t"]), len(s["t"]), round(time.time() - t0, 1), flush=True)
    counts = count_regimes(s)
    print("counts", json.dumps(counts), flush=True)
    res, preds, keep, calib = {}, {}, {}, []
    for name in SEGS:
        for k in KS:
            for sim in ("E", "D"):
                R = predict(s, name, k, sim)
                preds[f"{name}|k{k}|{sim}"] = R
                E, pr = R["E"], R["pr"]
                for e in range(len(E["P"])):
                    tt = int(s["t"][E["C"][e]])
                    if np.isfinite(pr["p"][e]) and np.isfinite(E["g"][e]) and DW.ms("2020-03-01") <= tt < DW.ms("2025-11-01"):
                        calib.append((f"{name}|k{k}|{sim}", float(pr["p"][e]), float(E["g"][e] > 0)))
                cands, gate = regime_cands(b, s, R)
                variants = [(f"R|{name}|k{k}|{sim}", cands, "G0")]
                if sim == "E":
                    F["gate"][f"RG_{name}_{k}"] = gate
                    variants += [(f"G|{name}|k{k}|{en}", C[en], f"RG_{name}_{k}") for en in ("S-NY", "P8")]
                for vn, cc, g in variants:
                    res[vn] = {"cands": len(cc)}
                    for w, (a, z) in W.items():
                        tr, cnt = DW.simulate(b, F, cc, g, "FA", a, z)
                        res[vn][w] = summ(tr, a, z, b) | {"cnt": cnt}
                        if w in ("d30", "year", "m6", "mid", "fresh", "pool"):
                            keep.setdefault(vn, {})[w] = tr
                    a, z = W["pool"]
                    trm, _ = DW.simulate(b, F, cc, g, "FB", a, z)
                    res[vn]["pool_maker"] = summ(trm, a, z, b)
                    rd = {}
                    for w in ("fresh", "design", "val", "year"):
                        a, z = W[w]
                        rd[w] = {"delay1": summ(DW.simulate(b, F, cc, g, "FA", a, z, delay=1)[0], a, z, b)["x5"],
                                 "fee15": summ(DW.simulate(b, F, cc, g, "FA", a, z, fee_mult=1.5)[0], a, z, b)["x5"]}  # fmt: skip
                    res[vn]["redteam"] = rd
                    r = res[vn]
                    r["pos_all"] = all(r[w]["x5"]["ret"] > 0 for w in MAIN)
                    r["liq0"] = all(r[w]["x5"]["liq"] == 0 for w in MAIN)
                    r["mdd_ok"] = all(r[w]["x5"]["mdd"] <= 0.40 for w in MAIN)
                    print(vn, {w: (r[w]["n"], round(r[w]["x5"]["ret"], 3)) for w in MAIN + ("d30",)},
                          "pd", round(r["year"]["per_day"], 2), round(time.time() - t0), flush=True)  # fmt: skip
    # 무작위(시각 맞춤, 합친 표본) — 모두 100번, 네 구간 모두 플러스면 크게
    allc = {}
    for name in SEGS:
        for k in KS:
            for sim in ("E", "D"):
                cands, _ = regime_cands(b, s, preds[f"{name}|k{k}|{sim}"])
                allc[f"R|{name}|k{k}|{sim}"] = (cands, "G0")
                if sim == "E":
                    for en in ("S-NY", "P8"):
                        allc[f"G|{name}|k{k}|{en}"] = (C[en], f"RG_{name}_{k}")
    a, z = W["pool"]
    for vn, (cc, g) in allc.items():
        tr = keep[vn]["pool"]
        if not tr:
            res[vn]["random"] = None
            continue
        ns = args.nsim_big if res[vn]["pos_all"] else args.nsim
        # 시각 맞춘 무작위는 신호가 문을 통과한 거래만 대상으로 한다(진짜 거래와 같은 개수·시각·거리)
        res[vn]["random"] = DW.random_p(b, F, cc, g, "FA", a, z, tr, ns, 31)
        if vn.startswith("R|"):
            res[vn]["random_flip"] = flip_random(b, F, cc, a, z, tr, min(ns, 300), 37)
        p = res[vn]["random"]["p"]
        res[vn]["bonf"] = min(1.0, p * N_VAR) if p is not None else None
        res[vn]["pass"] = bool(res[vn]["pos_all"] and res[vn]["liq0"] and res[vn]["mdd_ok"] and res[vn]["bonf"] < 0.05)
        print("random", vn, res[vn]["random"], res[vn].get("random_flip"), flush=True)
    by_year = {}
    for vn in res:
        tr = keep[vn].get("pool", []) + [x for x in keep[vn].get("year", [])]
        yrs: dict = {}
        for x in tr:
            y = int(np.datetime64(int(b["t"][x["e"]]), "ms").astype("datetime64[Y]").astype(int) + 1970)
            yrs.setdefault(y, []).append(x)
        by_year[vn] = {y: {"n": len(v), "x5": float(lev_path(v, 5)["ret"])} for y, v in sorted(yrs.items())}
    # 지금 사건(가장 최근) 예측
    now = {}
    for key_, R in preds.items():
        E, pr = R["E"], R["pr"]
        e = len(E["P"]) - 1
        now[key_] = {"event_t": int(s["t"][E["C"][e]]), "pivot_t": int(s["t"][E["P"][e]]),
                     "p_up": None if np.isnan(pr["p"][e]) else float(pr["p"][e]),
                     "med": None if np.isnan(pr["med"][e]) else float(pr["med"][e]), "dir": direction_of(pr, e),
                     "labels": E["lab"][-6:].tolist(), "nb": pr["nb"][e]}  # fmt: skip
    save = {"n_variants": N_VAR, "n_tonight": N_TONIGHT_BEFORE + N_VAR, "counts": counts,
            "passed": [v for v in res if res[v].get("pass")], "res": res, "by_year": by_year, "now": now,
            "calib": calib, "sec": time.time() - t0}  # fmt: skip
    (out / "result.json").write_text(json.dumps(save, indent=1, default=float))
    trk = {vn: {w: tr for w, tr in d.items() if w in ("d30", "m6", "mid", "fresh", "year")} for vn, d in keep.items()}
    (out / "trades.json").write_text(json.dumps(trk, default=float))
    print("passed", save["passed"], "sec", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
