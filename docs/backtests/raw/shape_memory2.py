"""모양 기억 2: 시간 늘림·줄임(TW), 큰 봉 경로(HF), 기간 지운 지그재그(ZN), 어느 봉이든(U).

등록: docs/backtests/2026-11-08-btc-shape-memory-2-hypotheses.md (결과 전 커밋 df2393e)
기억은 결과 확정 시각이 결정 시점보다 앞선 모양만 쓴다.

실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/shape-memory2-2026-11-08
  P="systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
     PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3"
  $P docs/backtests/raw/shape_memory2.py --out $OUT votes   # 이웃 투표(캐시)
  $P docs/backtests/raw/shape_memory2.py --out $OUT eval    # 매매·판정·무작위·레드팀·보정 곡선
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scr15_timing as X
import shape_memory as SM
from pine_eval import lev_path

M15, DAY = X.M15, X.DAY
TAKER, MAKER = 0.0005, 0.0002
K = 50
QS = (0.6, 0.7)
FRAMES = {"1h": (4, 4), "4h": (16, 6), "1d": (96, 3)}  # 15분봉 몇 개 묶음, 결과 지평(그 봉 수)
TW_SCALES = (16, 32, 64, 96, 192, 288, 480)
TW_PTS = 64
TW_LIB_STRIDE = 8
TW_QW = (96, 192)
TW_Q_STRIDE = 4
DTW_BAND = 6
DTW_PRE = 200
N_TESTS = 40
N_TONIGHT_BEFORE = 5819
Q_START = X.ms("2020-03-01")


# ---------------- 큰 봉 ----------------
def frame_bars(s: dict, f: int) -> dict:
    """15분봉 f개를 UTC 경계로 묶는다. 다 찬(f개) 봉만, end = 그 봉의 마지막 15분봉 위치."""
    t = s["t"]
    key = t // (f * M15)
    cut = np.flatnonzero(np.diff(key)) + 1
    st = np.r_[0, cut]
    en = np.r_[cut, len(t)] - 1
    full = (en - st + 1) == f
    st, en = st[full], en[full]
    return {"t": key[st] * f * M15, "end": en, "o": s["o"][st], "c": s["c"][en],
            "h": np.maximum.reduceat(s["h"], np.r_[0, cut])[full],
            "l": np.minimum.reduceat(s["l"], np.r_[0, cut])[full]}  # fmt: skip


def knn_frame(V: np.ndarray, c: np.ndarray, hz: int, q0: int, valid0: int, stride: int = 1, chunk: int = 256):
    """프레임 봉 단위 이웃 투표. 기억 j는 결과 봉 j+hz가 결정 봉보다 앞설 때만."""
    n = len(c)
    fut = np.full(n, np.nan)
    fut[: n - hz] = c[hz:] / c[: n - hz] - 1
    lib = np.arange(valid0, n - hz, stride)
    lib = lib[np.linalg.norm(V[lib], axis=1) > 0.5]
    ready = lib + hz
    LV, Ly, Lr = V[lib], (fut[lib] > 0).astype(np.float32), fut[lib].astype(np.float32)
    p = np.full(n, np.nan, np.float32)
    r = np.full(n, np.nan, np.float32)
    for s0 in range(q0, n, chunk):
        s1 = min(s0 + chunk, n)
        m = int(np.searchsorted(ready, s0, "left"))  # ready < s0
        if m < K:
            continue
        sims = V[s0:s1] @ LV[:m].T
        top = np.argpartition(-sims, K - 1, axis=1)[:, :K]
        w = np.clip(np.take_along_axis(sims, top, 1), 0, None) + 1e-6
        p[s0:s1] = (w * Ly[top]).sum(1) / w.sum(1)
        r[s0:s1] = (w * Lr[top]).sum(1) / w.sum(1)
        bad = np.linalg.norm(V[s0:s1], axis=1) < 0.5
        p[s0:s1][bad] = np.nan
        r[s0:s1][bad] = np.nan
    return p, r, fut


def zn_vecs(h, l, c, A, k: float = 1.5, legs: int = 6) -> np.ndarray:
    """지그재그 마지막 6다리의 높이 비율만(기간 성분 없음). 인과적: 극점에서 k×ATR 되돌면 확정."""
    n = len(c)
    piv_p: list = []
    trend, lo_i, hi_i = 0, 0, 0
    out = np.zeros((n, legs), np.float32)
    for i in range(1, n):
        th = k * A[i]
        if trend >= 0 and h[i] > h[hi_i]:
            hi_i = i
        if trend <= 0 and l[i] < l[lo_i]:
            lo_i = i
        if trend == 0:
            if h[i] - l[lo_i] >= th and lo_i < i:
                piv_p.append(l[lo_i])
                trend, hi_i = 1, i
            elif h[hi_i] - l[i] >= th and hi_i < i:
                piv_p.append(h[hi_i])
                trend, lo_i = -1, i
        elif trend == 1 and h[hi_i] - l[i] >= th and hi_i < i:
            piv_p.append(h[hi_i])
            trend, lo_i = -1, i
        elif trend == -1 and h[i] - l[lo_i] >= th and lo_i < i:
            piv_p.append(l[lo_i])
            trend, hi_i = 1, i
        if len(piv_p) >= legs:
            dr = np.diff(np.log(np.r_[piv_p[-legs:], c[i]]))
            m_r = np.abs(dr).mean()
            if m_r > 0:
                out[i] = dr / m_r
    return SM.unit(out, center=False).astype(np.float32)


# ---------------- 시간 늘림·줄임 ----------------
def tw_vecs(logc: np.ndarray, idx: np.ndarray, W: int) -> np.ndarray:
    """idx에서 끝나는 W봉 로그 경로를 64점으로 다시 뽑아 평균 빼고 단위 길이."""
    pos = np.linspace(0, W - 1, TW_PTS)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, W - 1)
    fr = (pos - lo).astype(np.float32)
    out = np.empty((len(idx), TW_PTS), np.float32)
    for a in range(0, len(idx), 4096):
        ii = idx[a:a + 4096]
        base = ii[:, None] - (W - 1)
        v = logc[base + lo] * (1 - fr) + logc[base + hi] * fr
        out[a:a + 4096] = SM.unit(v)
    return out


def tw_library(logc: np.ndarray):
    vecs, ys, rs, ready = [], [], [], []
    n = len(logc)
    for W in TW_SCALES:
        h = W // 4
        idx = np.arange(W - 1, n - h, TW_LIB_STRIDE)
        V = tw_vecs(logc, idx, W)
        ok = np.linalg.norm(V, axis=1) > 0.5
        idx, V = idx[ok], V[ok]
        win = np.lib.stride_tricks.sliding_window_view(logc, W)[idx - (W - 1)]
        rng = win.max(1) - win.min(1)
        ret = logc[idx + h] - logc[idx]
        vecs.append(V), ys.append((ret > 0).astype(np.float32))
        rs.append((ret / np.where(rng > 0, rng, 1)).astype(np.float32)), ready.append(idx + h)
        del win
    ready = np.concatenate(ready)
    o = np.argsort(ready, kind="stable")
    return np.concatenate(vecs)[o], np.concatenate(ys)[o], np.concatenate(rs)[o], ready[o]


def dtw_dist(q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """q (Q,64), C (Q,M,64) → (Q,M) 띠 DTW 거리(제곱 차이 합)."""
    Q, M, P = C.shape
    inf = np.float32(1e9)
    prev = np.full((Q, M, P), inf, np.float32)
    for i in range(P):
        cur = np.full((Q, M, P), inf, np.float32)
        j0, j1 = max(0, i - DTW_BAND), min(P, i + DTW_BAND + 1)
        for j in range(j0, j1):
            cost = (q[:, i, None] - C[:, :, j]) ** 2
            if i == 0 and j == 0:
                best = 0.0
            else:
                cands = []
                if i > 0:
                    cands.append(prev[:, :, j])
                    if j > 0:
                        cands.append(prev[:, :, j - 1])
                if j > 0:
                    cands.append(cur[:, :, j - 1])
                best = np.minimum.reduce(cands)
            cur[:, :, j] = cost + best
        prev = cur
    return prev[:, :, P - 1]


def tw_votes(logc: np.ndarray, lib, QW: int, q0: int, chunk: int = 32):
    LV, Ly, Lr, ready = lib
    n = len(logc)
    qidx = np.arange(max(q0, QW - 1), n, TW_Q_STRIDE)
    out = {m: (np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)) for m in ("C", "D")}
    t0 = time.time()
    for a in range(0, len(qidx), chunk):
        ii = qidx[a:a + chunk]
        m = int(np.searchsorted(ready, ii[0], "left"))
        if m < DTW_PRE:
            continue
        qv = tw_vecs(logc, ii, QW)
        sims = qv @ LV[:m].T
        top = np.argpartition(-sims, DTW_PRE - 1, axis=1)[:, :DTW_PRE]
        ts = np.take_along_axis(sims, top, 1)
        o = np.argsort(-ts, axis=1)
        top, ts = np.take_along_axis(top, o, 1), np.take_along_axis(ts, o, 1)
        w = np.clip(ts[:, :K], 0, None) + 1e-6
        out["C"][0][ii] = (w * Ly[top[:, :K]]).sum(1) / w.sum(1)
        out["C"][1][ii] = (w * Lr[top[:, :K]]).sum(1) / w.sum(1)
        d = dtw_dist(qv, LV[top])
        o2 = np.argsort(d, axis=1)[:, :K]
        tk = np.take_along_axis(top, o2, 1)
        w2 = 1.0 / (np.take_along_axis(d, o2, 1) + 1e-3)
        out["D"][0][ii] = (w2 * Ly[tk]).sum(1) / w2.sum(1)
        out["D"][1][ii] = (w2 * Lr[tk]).sum(1) / w2.sum(1)
        if a % (chunk * 200) == 0:
            print("tw", QW, a, len(qidx), round(time.time() - t0), flush=True)
    return out


def cmd_votes(out: Path, only: str | None) -> None:
    b, s, fut0 = SM.load_all()
    q0 = int(np.searchsorted(s["t"], Q_START))
    A15 = SM.atr(s["h"], s["l"], s["c"])
    save = {"t": s["t"]}
    f = out / "votes2.npz"
    if f.exists():
        save = dict(np.load(f))
    jobs = ["HF", "ZN15", "ZN4h", "TW96", "TW192"]
    for job in jobs:
        if only and job != only:
            continue
        if any(k.startswith(job + "_") for k in save):
            print("skip", job, flush=True)
            continue
        t0 = time.time()
        if job == "HF":
            for fr, (fac, hz) in FRAMES.items():
                F_ = frame_bars(s, fac)
                V = SM.path_vecs(F_["c"], 32)
                fq0 = int(np.searchsorted(F_["t"], Q_START))
                p, r, fut = knn_frame(V, F_["c"], hz, fq0, 32)
                P_, R_, O_ = (np.full(len(s["t"]), np.nan, np.float32) for _ in range(3))
                P_[F_["end"]], R_[F_["end"]], O_[F_["end"]] = p, r, fut
                save[f"HF_{fr}_p"], save[f"HF_{fr}_r"], save[f"HF_{fr}_o"] = P_, R_, O_
                print("HF", fr, len(F_["c"]), round(time.time() - t0), flush=True)
        elif job == "ZN15":
            V = zn_vecs(s["h"], s["l"], s["c"], A15)
            p, r, fut = knn_frame(V, s["c"], 8, q0, 200, stride=4)
            save["ZN15_p"], save["ZN15_r"], save["ZN15_o"] = p, r, fut.astype(np.float32)
        elif job == "ZN4h":
            fac, hz = FRAMES["4h"]
            F_ = frame_bars(s, fac)
            A4 = SM.atr(F_["h"], F_["l"], F_["c"])
            V = zn_vecs(F_["h"], F_["l"], F_["c"], A4)
            fq0 = int(np.searchsorted(F_["t"], Q_START))
            p, r, fut = knn_frame(V, F_["c"], hz, fq0, 60)
            P_, R_, O_ = (np.full(len(s["t"]), np.nan, np.float32) for _ in range(3))
            P_[F_["end"]], R_[F_["end"]], O_[F_["end"]] = p, r, fut
            save["ZN4h_p"], save["ZN4h_r"], save["ZN4h_o"] = P_, R_, O_
        else:
            QW = int(job[2:])
            logc = np.log(s["c"]).astype(np.float64)
            lib = tw_library(logc)
            print("tw lib", len(lib[0]), flush=True)
            v = tw_votes(logc, lib, QW, q0)
            n = len(logc)
            o = np.full(n, np.nan, np.float32)
            h = QW // 4
            o[: n - h] = np.exp(logc[h:] - logc[: n - h]) - 1
            for m_, (p, r) in v.items():
                save[f"{job}{m_}_p"], save[f"{job}{m_}_r"] = p, r
            save[f"{job}_o"] = o
            del lib
        np.savez_compressed(f, **save)
        print(job, "done", round(time.time() - t0), flush=True)


# ---------------- 매매 ----------------
def on_b(save: dict, b: dict, key: str) -> np.ndarray:
    st = save["t"]
    a0 = int(np.searchsorted(st, b["t"][0]))
    m = min(len(st) - a0, len(b["t"]))
    out = np.full(len(b["t"]), np.nan, np.float32)
    out[:m] = save[key][a0:a0 + m]
    return out


SRC = {  # 이름 → (투표 키 앞부분, 지평 15분봉 수)
    "TW96C": ("TW96C", 24), "TW96D": ("TW96D", 24), "TW192C": ("TW192C", 48), "TW192D": ("TW192D", 48),
    "HF1h": ("HF_1h", 16), "HF4h": ("HF_4h", 96), "HF1d": ("HF_1d", 288),
    "ZN15": ("ZN15", 8), "ZN4h": ("ZN4h", 96),
}  # fmt: skip


def rule_signals(save, b, name: str, q: float):
    """(L, S, H배열). U = HF 1h·4h·1d 중 아무거나(방향이 엇갈리면 건너뜀, 같으면 큰 봉 지평)."""
    n = len(b["t"])
    if name == "U":
        L, S, Hs = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, int)
        for fr in ("HF1h", "HF4h", "HF1d"):
            l_, s_, h_ = rule_signals(save, b, fr, q)
            Hs = np.where((l_ | s_), np.maximum(Hs, h_), Hs)
            L |= l_
            S |= s_
        both = L & S
        return L & ~both, S & ~both, Hs
    pre, hz = SRC[name]
    p, r = on_b(save, b, pre + "_p"), on_b(save, b, pre + "_r")
    with np.errstate(invalid="ignore"):
        L = np.nan_to_num((p >= q) & (r > 0)).astype(bool)
        S = np.nan_to_num((p <= 1 - q) & (r < 0)).astype(bool)
    return L, S, np.full(n, hz, int)


def sim2(b, F, A, L, S, Hs, a, z, maker=False, delay=0, fee_mult=1.0):
    t, o, h, l, c = b["t"], b["o"], b["h"], b["l"], b["c"]
    ft, fcum, big = b["ft"], F["fcum"], F["big"]
    fin = TAKER * fee_mult
    fen = (MAKER if maker else TAKER) * fee_mult
    trades, cnt = [], {"chase_skip": 0, "nofill": 0}
    i = a
    while i < z - 2:
        d = 1 if L[i] else (-1 if S[i] else 0)
        if not d:
            i += 1
            continue
        sig, hz = i, int(Hs[i])
        if maker:
            e, ep = None, c[sig]
            for j in range(sig + 1 + delay, min(sig + 5 + delay, z - 1)):
                if (d == 1 and l[j] < ep - 0.1) or (d == -1 and h[j] > ep + 0.1):
                    e = j
                    break
            if e is None:
                cnt["nofill"] += 1
                i += 1
                continue
            first_open = False
        else:
            e = sig + 1 + delay
            if e >= z - 1:
                break
            ep = o[e]
            if abs(ep / c[sig] - 1) > (0.03 if big[sig] else 0.01):
                cnt["chase_skip"] += 1
                i += 1
                continue
            first_open = True
        dist = A[sig] * np.sqrt(hz)
        tp, sp = ep + d * dist, ep - d * dist
        worst, x, xp, why = 0.0, None, None, None
        for j in range(e, min(e + hz, z)):
            hit_s = (l[j] <= sp) if d == 1 else (h[j] >= sp)
            hit_t = (h[j] >= tp) if d == 1 else (l[j] <= tp)
            if j == e and not first_open:
                hit_t = False
            worst = min(worst, (l[j] / ep - 1) if d == 1 else -(h[j] / ep - 1))
            if hit_s:
                gap = (o[j] <= sp) if d == 1 else (o[j] >= sp)
                x, xp, why = j, (o[j] if gap and j > e else sp), "stop"
                break
            if hit_t:
                x, xp, why = j, tp, "tp"
                break
        if x is None:
            x = min(e + hz, z) - 1
            xp, why = c[x], "time"
        i0, i1 = np.searchsorted(ft, int(t[e]), "right"), np.searchsorted(ft, int(t[x]) + M15, "right")
        fu = -d * float(fcum[i1] - fcum[i0])
        r = d * (xp / ep - 1) - fen - fin + fu
        trades.append({"d": d, "j": sig, "e": e, "x": x, "ep": float(ep), "xp": float(xp), "why": why,
                       "r": float(r), "worst": float(min(worst, d * (xp / ep - 1))), "fund": fu, "hz": hz})  # fmt: skip
        i = x + 1
    return trades, cnt


def random_p(b, F, A, tr, a, z, real, nsim, seed, maker=False) -> dict:
    rng = np.random.default_rng(seed)
    t = b["t"]
    slot = ((t // M15) % 96).astype(int)
    by_slot = {s_: np.flatnonzero(slot[a:z - 300] == s_) + a for s_ in range(96)}
    vals = []
    for _ in range(nsim):
        Lr, Sr, Hr = np.zeros(len(t), bool), np.zeros(len(t), bool), np.zeros(len(t), int)
        for x in tr:
            k = int(rng.choice(by_slot[int(slot[x["j"]])]))
            (Lr if x["d"] == 1 else Sr)[k] = True
            Hr[k] = x["hz"]
        rt, _ = sim2(b, F, A, Lr, Sr, Hr, a, z, maker=maker)
        vals.append(lev_path(rt, 5)["ret"])
    vals = np.array(vals)
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "nsim": nsim}


RULES = [f"{n}|q{int(q * 100)}|{fee}" for n in ("TW96C", "TW96D", "TW192C", "TW192D", "HF1h", "HF4h", "HF1d",
                                              "ZN15", "ZN4h", "U") for q in QS for fee in ("T", "M")]  # fmt: skip
assert len(RULES) == N_TESTS
BUSY_FAMILY = ("ZN15", "U")


def calibration(save, s_t) -> dict:
    """2020-03~2025-10 결정 시점의 이웃 오를 확률 칸별 실제 오른 비율."""
    a = int(np.searchsorted(s_t, Q_START))
    z = int(np.searchsorted(s_t, X.ms("2025-11-01")))
    bins = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0001]
    srcs = {"15m (N32)": ("N32", None), "1h": ("HF_1h_p", "HF_1h_o"), "4h": ("HF_4h_p", "HF_4h_o"),
            "1d": ("HF_1d_p", "HF_1d_o"), "TW 1d corr": ("TW96C_p", "TW96_o"), "TW 1d DTW": ("TW96D_p", "TW96_o"),
            "zigzag 15m": ("ZN15_p", "ZN15_o"), "zigzag 4h": ("ZN4h_p", "ZN4h_o")}  # fmt: skip
    out = {}
    for name, (pk, ok) in srcs.items():
        if pk == "N32":
            d = np.load(Path("~/.minerva/research/shape-memory-2026-11-08/votes_N32.npz").expanduser())
            st, p = d["t"], d["p50"]
            c = None
            o = np.full(len(st), np.nan)
            _, s, _ = SM.load_all()
            cc = s["c"]
            assert len(cc) >= len(st)
            o[: len(st) - SM.H] = cc[SM.H:len(st)] / cc[: len(st) - SM.H] - 1
            aa, zz = int(np.searchsorted(st, Q_START)), int(np.searchsorted(st, X.ms("2025-11-01")))
            p, o = p[aa:zz], o[aa:zz]
            del c
        else:
            if pk not in save:
                continue
            p, o = save[pk][a:z], save[ok][a:z]
        m = np.isfinite(p) & np.isfinite(o)
        p, up = p[m], (o[m] > 0)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (p >= lo) & (p < hi)
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(sel.sum()), "pred": float(p[sel].mean()) if sel.any() else None,
                         "up": float(up[sel].mean()) if sel.any() else None})  # fmt: skip
        conf = np.abs(p - 0.5) >= 0.1
        hit = ((p > 0.5) == up)[conf]
        out[name] = {"bins": rows, "n": int(len(p)), "base_up": float(up.mean()),
                     "hit_confident": float(hit.mean()) if len(hit) else None, "n_confident": int(conf.sum()),
                     "corr": float(np.corrcoef(p, up)[0, 1]) if len(p) > 2 else None}  # fmt: skip
        print("calib", name, out[name]["hit_confident"], out[name]["n_confident"], out[name]["corr"], flush=True)
    return out


def cmd_eval(out: Path, nsim: int) -> None:
    t0 = time.time()
    b = X.load()
    F = X.features(b)
    A = SM.atr(b["h"], b["l"], b["c"])
    W = SM.windows(b["t"])
    save = dict(np.load(out / "votes2.npz"))
    res = {}
    for key in RULES:
        name, qs, fee = key.split("|")
        L, S, Hs = rule_signals(save, b, name, int(qs[1:]) / 100)
        res[key] = {}
        for w, (a, z) in W.items():
            days = (b["t"][z - 1] - b["t"][a]) / DAY
            tr, cnt = sim2(b, F, A, L, S, Hs, a, z, maker=fee == "M")
            res[key][w] = SM.summ(tr, cnt, days)
        print(key, {w: (r["n"], round(r["x5"]["ret"], 3)) for w, r in res[key].items()}, flush=True)
    main_w = ("fresh", "mid", "y1")
    for key, r in res.items():
        r["gate"] = {
            "x5_pos": all(r[w]["x5"]["ret"] > 0 for w in main_w),
            "liq0": all(r[w]["x5"]["liq"] == 0 for w in main_w),
            "mdd40": all(r[w]["x5"]["mdd"] <= 0.40 for w in main_w),
        }
        if key.split("|")[0] in BUSY_FAMILY:
            r["gate"]["per_day1"] = all((r[w]["per_day"] or 0) >= 1 for w in main_w)
    score = lambda k_: res[k_]["mid"]["x5"]["ret"] + res[k_]["fresh"]["x5"]["ret"] + res[k_]["y1"]["x5"]["ret"]  # noqa: E731
    best = max(res, key=score)
    busy = [k_ for k_ in res if all((res[k_][w]["per_day"] or 0) >= 1 for w in main_w)]
    best_busy = max(busy, key=score) if busy else None
    best_tw = max([k_ for k_ in res if k_.startswith("TW")], key=score)
    best_hf = max([k_ for k_ in res if k_.startswith("HF")], key=score)
    rnd, red = {}, {}
    for key in dict.fromkeys([best, best_busy, best_tw, best_hf]):
        if key is None:
            continue
        name, qs, fee = key.split("|")
        L, S, Hs = rule_signals(save, b, name, int(qs[1:]) / 100)
        rnd[key], red[key] = {}, {}
        for w in main_w:
            a, z = W[w]
            days = (b["t"][z - 1] - b["t"][a]) / DAY
            tr, _ = sim2(b, F, A, L, S, Hs, a, z, maker=fee == "M")
            rnd[key][w] = random_p(b, F, A, tr, a, z, res[key][w]["x5"]["ret"], nsim, 13, maker=fee == "M") if tr else None
            red[key][w] = {"delay1": SM.summ(*sim2(b, F, A, L, S, Hs, a, z, maker=fee == "M", delay=1), days),
                           "fee15": SM.summ(*sim2(b, F, A, L, S, Hs, a, z, maker=fee == "M", fee_mult=1.5), days)}  # fmt: skip
            print("random/red", key, w, rnd[key][w], flush=True)
        a, z = W["fresh"][0], W["y1"][1]
        tr, _ = sim2(b, F, A, L, S, Hs, a, z, maker=fee == "M")
        yrs: dict = {}
        for x in tr:
            yrs.setdefault(datetime.fromtimestamp(int(b["t"][x["e"]]) / 1000, timezone.utc).year, []).append(x)
        red[key]["by_year"] = {y: {"n": len(v_), "x5": float(lev_path(v_, 5)["ret"])} for y, v_ in sorted(yrs.items())}
    for key, r in res.items():
        p = (rnd.get(key, {}).get("mid") or {}).get("p")
        r["gate"]["random_bonf"] = bool(p is not None and p * N_TESTS < 0.05)
        r["gate"]["red"] = bool(key in red and red[key]["mid"]["delay1"]["x5"]["ret"] > 0
                                and red[key]["mid"]["fee15"]["x5"]["ret"] > 0)  # fmt: skip
        r["pass"] = all(r["gate"].values())
    cal = calibration(save, save["t"])
    (out / "result.json").write_text(json.dumps(
        {"windows": {w: [int(b["t"][a]), int(b["t"][z - 1])] for w, (a, z) in W.items()}, "n_tests": N_TESTS,
         "n_tonight": N_TONIGHT_BEFORE + N_TESTS, "best": best, "best_busy": best_busy, "best_tw": best_tw,
         "best_hf": best_hf, "passed": [k_ for k_, r in res.items() if r["pass"]], "res": res, "random": rnd,
         "redteam": red, "calibration": cal, "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    print("best", best, "busy", best_busy, "tw", best_tw, "hf", best_hf,
          "passed", [k_ for k_, r in res.items() if r["pass"]], flush=True)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=300)
    ap.add_argument("--only")
    ap.add_argument("cmd", choices=("votes", "eval"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "votes":
        cmd_votes(out, a.only)
    else:
        cmd_eval(out, a.nsim)


if __name__ == "__main__":
    main()
