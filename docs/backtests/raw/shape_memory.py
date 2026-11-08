"""모양 기억(최근접 유사 모양) 시험. 등록: 2026-11-08-btc-shape-memory-hypotheses.md

지금 15분봉 모양과 가장 닮은 과거 모양 K개를 찾아, 그 뒤 8봉 동안 오른 비율로 방향을 정한다.
기억은 결정 시점보다 앞서 결과까지 확정된 모양만 쓴다(미래 정보 차단).

실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/shape-memory-2026-11-08
  P="systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
     PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3"
  $P docs/backtests/raw/shape_memory.py --out $OUT knn     # 모양별 이웃 투표 계산(캐시)
  $P docs/backtests/raw/shape_memory.py --out $OUT eval    # 매매·판정·무작위·레드팀
  $P docs/backtests/raw/shape_memory.py --out $OUT count   # "모양이 몇 개냐"
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scr15_timing as X
from btc_fractal import cci, rsi, stoch
from numpy.lib.stride_tricks import sliding_window_view
from pine_eval import lev_path

M15, DAY = X.M15, X.DAY
H = 8  # 예측 지평(봉)
LIB_STRIDE = 4  # 기억은 1시간마다 하나
CHUNK = 256
KS = (20, 50, 200)
QS = (0.6, 0.7)
SHAPES = ("N16", "N32", "N64", "R32", "Z6")
TAKER, MAKER = 0.0005, 0.0002
TP = 0.01
N_RULES = len(SHAPES) * len(KS) * len(QS)  # 30
N_TESTS = N_RULES * 2  # 수수료 방식 2개 → 60
N_TONIGHT_BEFORE = 5759
SPOT_CSV = Path("~/.minerva/data/btc/15m.csv").expanduser()
Q_START = X.ms("2020-03-01")


# ---------------- 자료 ----------------
def load_all() -> tuple[dict, dict, int]:
    """선물 b(매매용)와 현물 앞부분을 붙인 모양용 배열 s. fut0 = s에서 선물이 시작하는 위치."""
    b = X.load()
    raw = np.genfromtxt(SPOT_CSV, delimiter=",", skip_header=1, usecols=(0, 1, 2, 3, 4))
    raw = raw[raw[:, 0] < b["t"][0]]
    s = {"t": np.r_[raw[:, 0].astype(np.int64), b["t"]]}
    for i, k in enumerate(("o", "h", "l", "c"), 1):
        s[k] = np.r_[raw[:, i], b[k]]
    return b, s, len(raw)


def atr(h, l, c, n=14) -> np.ndarray:
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.empty_like(tr)
    out[0] = tr[0]
    a = 1 / n
    for i in range(1, len(tr)):
        out[i] = out[i - 1] + a * (tr[i] - out[i - 1])
    return out


# ---------------- 모양 ----------------
def unit(m: np.ndarray, center: bool = True) -> np.ndarray:
    if center:
        m = m - m.mean(1, keepdims=True)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.where(n > 1e-12, m / np.where(n > 1e-12, n, 1), 0.0)


def path_vecs(c: np.ndarray, n: int) -> np.ndarray:
    """i번째 행 = 봉 i에서 끝나는 n봉 로그 경로(평균 빼고 단위 길이). 앞 n−1봉은 0."""
    w = sliding_window_view(np.log(c), n)
    out = np.zeros((len(c), n), np.float32)
    out[n - 1:] = unit(w)
    return out


def rich_vecs(s: dict, A: np.ndarray) -> np.ndarray:
    h, l, c = s["h"], s["l"], s["c"]
    p = path_vecs(c, 32)
    rng = (h - l) / np.where(A > 0, A, 1)
    rv = np.zeros((len(c), 32), np.float32)
    rv[31:] = unit(sliding_window_view(rng, 32))
    k, cc, rr = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    blocks = []
    for x, f in ((k, lambda v: v / 100 - 0.5), (cc, lambda v: np.clip(v / 200, -1.5, 1.5)), (rr, lambda v: (v - 50) / 50)):
        v = np.nan_to_num(f(x))
        m = np.zeros((len(c), 8), np.float32)
        m[7:] = sliding_window_view(v, 8) / np.sqrt(8)
        blocks.append(m)
    out = np.concatenate([p, rv] + blocks, axis=1)
    return unit(out, center=False).astype(np.float32)


def zigzag_vecs(s: dict, A: np.ndarray, k: float = 1.5, legs: int = 6) -> np.ndarray:
    """인과적 지그재그: 극점에서 k×ATR 되돌면 그 봉에 확정. 마지막 다리는 확정 극점 → 지금 종가."""
    h, l, c = s["h"], s["l"], s["c"]
    n = len(c)
    piv_i, piv_p = [], []
    trend = 0
    lo_i = hi_i = 0
    out = np.zeros((n, 2 * legs), np.float32)
    for i in range(1, n):
        th = k * A[i]
        if trend >= 0:
            if h[i] > h[hi_i]:
                hi_i = i
        if trend <= 0:
            if l[i] < l[lo_i]:
                lo_i = i
        if trend == 0:
            if h[i] - l[lo_i] >= th and lo_i < i:
                piv_i.append(lo_i), piv_p.append(l[lo_i])
                trend, hi_i = 1, i
            elif h[hi_i] - l[i] >= th and hi_i < i:
                piv_i.append(hi_i), piv_p.append(h[hi_i])
                trend, lo_i = -1, i
        elif trend == 1 and h[hi_i] - l[i] >= th and hi_i < i:
            piv_i.append(hi_i), piv_p.append(h[hi_i])
            trend, lo_i = -1, i
        elif trend == -1 and h[i] - l[lo_i] >= th and lo_i < i:
            piv_i.append(lo_i), piv_p.append(l[lo_i])
            trend, hi_i = 1, i
        if len(piv_i) >= legs:
            pi = np.r_[piv_i[-legs:], i]
            pp = np.log(np.r_[piv_p[-legs:], c[i]])
            dr = np.diff(pp)
            dt = np.diff(pi).astype(float)
            m_r = np.abs(dr).mean()
            m_t = dt.mean()
            if m_r > 0 and m_t > 0:
                out[i] = np.r_[dr / m_r, 0.5 * dt / m_t]
    return unit(out, center=False).astype(np.float32)


def shape_vecs(name: str, s: dict, A: np.ndarray) -> np.ndarray:
    if name.startswith("N"):
        return path_vecs(s["c"], int(name[1:]))
    if name == "R32":
        return rich_vecs(s, A)
    return zigzag_vecs(s, A)


# ---------------- 이웃 투표 ----------------
def knn_votes(V: np.ndarray, c: np.ndarray, q0: int, valid0: int) -> dict:
    """q0부터 끝까지 매 봉에서 K개 이웃의 (오를 확률 p, 평균 수익 r). 기억은 묶음 시작 − H − 1 이전."""
    n = len(c)
    fut = np.full(n, np.nan)
    fut[: n - H] = c[H:] / c[: n - H] - 1
    lib = np.arange(valid0, n - H, LIB_STRIDE)
    lib = lib[np.linalg.norm(V[lib], axis=1) > 0.5]
    LV, Ly, Lr = V[lib], (fut[lib] > 0).astype(np.float32), fut[lib].astype(np.float32)
    out = {f"p{k}": np.full(n, np.nan, np.float32) for k in KS} | {f"r{k}": np.full(n, np.nan, np.float32) for k in KS}
    kmax = max(KS)
    for s0 in range(q0, n, CHUNK):
        s1 = min(s0 + CHUNK, n)
        m = int(np.searchsorted(lib, s0 - H - 1, "right"))  # 결과가 확정된 기억만
        if m < kmax:
            continue
        sims = V[s0:s1] @ LV[:m].T
        top = np.argpartition(-sims, kmax - 1, axis=1)[:, :kmax]
        ts = np.take_along_axis(sims, top, 1)
        order = np.argsort(-ts, axis=1)
        top, ts = np.take_along_axis(top, order, 1), np.take_along_axis(ts, order, 1)
        w = np.clip(ts, 0, None) + 1e-6
        for k in KS:
            ww = w[:, :k]
            out[f"p{k}"][s0:s1] = (ww * Ly[top[:, :k]]).sum(1) / ww.sum(1)
            out[f"r{k}"][s0:s1] = (ww * Lr[top[:, :k]]).sum(1) / ww.sum(1)
        bad = np.linalg.norm(V[s0:s1], axis=1) < 0.5
        for key in out:
            out[key][s0:s1][bad] = np.nan
    return out


def cmd_knn(out: Path) -> None:
    b, s, fut0 = load_all()
    A = atr(s["h"], s["l"], s["c"])
    q0 = int(np.searchsorted(s["t"], Q_START))
    for name in SHAPES:
        f = out / f"votes_{name}.npz"
        if f.exists():
            print("skip", name, flush=True)
            continue
        t0 = time.time()
        V = shape_vecs(name, s, A)
        valid0 = {"N16": 16, "N32": 32, "N64": 64, "R32": 40}.get(name, 200)
        v = knn_votes(V, s["c"], q0, valid0)
        np.savez_compressed(f, t=s["t"], **v)
        print(name, "done", round(time.time() - t0, 1), flush=True)
        del V


# ---------------- 매매 ----------------
def windows(t: np.ndarray) -> dict:
    end = int(t[-1]) + M15
    w = {"d30": (end - 30 * DAY, end), "p30": (end - 60 * DAY, end - 30 * DAY), "m6": (end - 183 * DAY, end),
         "y1": (X.ms("2025-11-01"), end), "mid": (X.ms("2022-01-01"), X.ms("2025-11-01")),
         "fresh": (X.ms("2020-03-01"), X.ms("2021-11-01"))}  # fmt: skip
    return {k: (int(np.searchsorted(t, a)), int(np.searchsorted(t, z))) for k, (a, z) in w.items()}


def signals(v: dict, k: int, q: float, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    p, r = v[f"p{k}"][sl], v[f"r{k}"][sl]
    with np.errstate(invalid="ignore"):
        L = (p >= q) & (r > 0)
        S = (p <= 1 - q) & (r < 0)
    return np.nan_to_num(L).astype(bool), np.nan_to_num(S).astype(bool)


def sim(b: dict, F: dict, A: np.ndarray, L, S, a: int, z: int, maker: bool = False, delay: int = 0,
        fee_mult: float = 1.0) -> tuple[list, dict]:  # fmt: skip
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
        sig = i
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
            lim = 0.03 if big[sig] else 0.01
            if abs(ep / c[sig] - 1) > lim:
                cnt["chase_skip"] += 1
                i += 1
                continue
            first_open = True
        tp, sp = ep * (1 + d * TP), ep - d * A[sig]
        worst, x, xp, why = 0.0, None, None, None
        for j in range(e, min(e + H, z)):
            lo_r, hi_r = (l[j] / ep - 1), (h[j] / ep - 1)
            hit_s = (l[j] <= sp) if d == 1 else (h[j] >= sp)
            hit_t = (h[j] >= tp) if d == 1 else (l[j] <= tp)
            if j == e and not first_open:  # 지정가 체결 봉: 체결 뒤 움직임만 알 수 없어 손절만 본다
                hit_t = False
            worst = min(worst, d * (lo_r if d == 1 else hi_r))
            if hit_s:
                gap = (o[j] <= sp) if d == 1 else (o[j] >= sp)
                x, xp, why = j, (o[j] if gap and j > e else sp), "stop"
                break
            if hit_t:
                x, xp, why = j, tp, "tp"
                break
        if x is None:
            x = min(e + H, z) - 1
            xp, why = c[x], "time"
        t_in, t_out = int(t[e]), int(t[x]) + M15
        i0, i1 = np.searchsorted(ft, t_in, "right"), np.searchsorted(ft, t_out, "right")
        fu = -d * float(fcum[i1] - fcum[i0])
        r = d * (xp / ep - 1) - fen - fin + fu
        trades.append({"d": d, "j": sig, "e": e, "x": x, "ep": float(ep), "xp": float(xp), "why": why,
                       "r": float(r), "worst": float(min(worst, d * (xp / ep - 1))), "fund": fu})  # fmt: skip
        i = x + 1
    return trades, cnt


def summ(tr: list, cnt: dict, days: float) -> dict:
    r = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    out = {"n": len(tr), "per_day": len(tr) / days if days else None, "win": float((r > 0).mean()) if len(r) else None,
           "long_share": float(np.mean([x["d"] == 1 for x in tr])) if tr else None,
           "avg": float(r.mean()) if len(r) else None, **cnt}  # fmt: skip
    for L_ in (1, 5):
        out[f"x{L_}"] = {k: float(v) for k, v in lev_path(tr, L_).items()}
    return out


def random_p(b, F, A, tr, a, z, real, nsim, seed, maker=False) -> dict:
    """같은 시각대(하루 중 같은 15분 칸)·같은 롱/숏 수로 무작위 날짜에 들어간 것과 비교."""
    rng = np.random.default_rng(seed)
    t = b["t"]
    slot = ((t // M15) % 96).astype(int)
    by_slot = {s_: np.flatnonzero(slot[a:z - H - 2] == s_) + a for s_ in range(96)}
    vals = []
    for _ in range(nsim):
        Lr, Sr = np.zeros(len(t), bool), np.zeros(len(t), bool)
        for x in tr:
            cand = by_slot[int(slot[x["j"]])]
            k = int(rng.choice(cand))
            (Lr if x["d"] == 1 else Sr)[k] = True
        rt, _ = sim(b, F, A, Lr, Sr, a, z, maker=maker)
        vals.append(lev_path(rt, 5)["ret"])
    vals = np.array(vals)
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "nsim": nsim}


def cmd_eval(out: Path, nsim: int) -> None:
    t0 = time.time()
    b = X.load()
    F = X.features(b)
    A = atr(b["h"], b["l"], b["c"])
    W = windows(b["t"])
    res, votes = {}, {}
    for name in SHAPES:
        d = np.load(out / f"votes_{name}.npz")
        st = d["t"]
        a0 = int(np.searchsorted(st, b["t"][0]))
        assert np.array_equal(st[a0:a0 + len(b["t"])], b["t"][: len(st) - a0]), "시각 불일치"
        v = {k: np.full(len(b["t"]), np.nan, np.float32) for k in d.files if k != "t"}
        m = min(len(st) - a0, len(b["t"]))
        for k in v:
            v[k][:m] = d[k][a0:a0 + m]
        votes[name] = v
    for name in SHAPES:
        for k in KS:
            for q in QS:
                L, S = signals(votes[name], k, q, slice(None))
                for fee in ("T", "M"):
                    key = f"{name}|K{k}|q{int(q * 100)}|{fee}"
                    res[key] = {}
                    for w, (a, z) in W.items():
                        days = (b["t"][z - 1] - b["t"][a]) / DAY
                        tr, cnt = sim(b, F, A, L, S, a, z, maker=fee == "M")
                        res[key][w] = summ(tr, cnt, days)
                print(key, {w: (r["n"], round(r["x5"]["ret"], 3)) for w, r in res[key].items()}, flush=True)
    for key, r in res.items():
        r["gate"] = {
            "x5_pos": all(r[w]["x5"]["ret"] > 0 for w in ("fresh", "mid", "y1")),
            "liq0": all(r[w]["x5"]["liq"] == 0 for w in ("fresh", "mid", "y1")),
            "mdd40": all(r[w]["x5"]["mdd"] <= 0.40 for w in ("fresh", "mid", "y1")),
            "per_day1": all((r[w]["per_day"] or 0) >= 1 for w in ("fresh", "mid", "y1")),
        }
    best = max(res, key=lambda k_: res[k_]["mid"]["x5"]["ret"] + res[k_]["fresh"]["x5"]["ret"])
    busy = [k_ for k_ in res if all((res[k_][w]["per_day"] or 0) >= 1 for w in ("fresh", "mid", "y1"))]
    best_busy = max(busy, key=lambda k_: res[k_]["mid"]["x5"]["ret"] + res[k_]["fresh"]["x5"]["ret"]) if busy else None
    rnd, red = {}, {}
    for key in dict.fromkeys([best, best_busy]):
        if key is None:
            continue
        name, ks, qs, fee = key.split("|")
        L, S = signals(votes[name], int(ks[1:]), int(qs[1:]) / 100, slice(None))
        rnd[key], red[key] = {}, {}
        for w in ("fresh", "mid", "y1"):
            a, z = W[w]
            tr, _ = sim(b, F, A, L, S, a, z, maker=fee == "M")
            rnd[key][w] = random_p(b, F, A, tr, a, z, res[key][w]["x5"]["ret"], nsim, 11, maker=fee == "M")
            days = (b["t"][z - 1] - b["t"][a]) / DAY
            red[key][w] = {"delay1": summ(*sim(b, F, A, L, S, a, z, maker=fee == "M", delay=1), days),
                           "fee15": summ(*sim(b, F, A, L, S, a, z, maker=fee == "M", fee_mult=1.5), days)}  # fmt: skip
            print("random/red", key, w, rnd[key][w], flush=True)
        a, z = W["fresh"][0], W["y1"][1]
        tr, _ = sim(b, F, A, L, S, a, z, maker=fee == "M")
        yrs: dict = {}
        for x in tr:
            yrs.setdefault(datetime.fromtimestamp(int(b["t"][x["e"]]) / 1000, timezone.utc).year, []).append(x)
        red[key]["by_year"] = {y: {"n": len(v_), "x5": float(lev_path(v_, 5)["ret"])} for y, v_ in sorted(yrs.items())}
    for key, r in res.items():
        p = rnd.get(key, {}).get("mid", {}).get("p")
        r["gate"]["random_bonf"] = bool(p is not None and p * N_TESTS < 0.05)
        r["gate"]["red"] = bool(key in red and red[key]["mid"]["delay1"]["x5"]["ret"] > 0
                                and red[key]["mid"]["fee15"]["x5"]["ret"] > 0)  # fmt: skip
        r["pass"] = all(r["gate"].values())
    (out / "result.json").write_text(json.dumps(
        {"windows": {w: [int(b["t"][a]), int(b["t"][z - 1])] for w, (a, z) in W.items()}, "n_tests": N_TESTS,
         "n_tonight": N_TONIGHT_BEFORE + N_TESTS, "best": best, "best_busy": best_busy,
         "passed": [k_ for k_, r in res.items() if r["pass"]], "res": res, "random": rnd, "redteam": red,
         "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    print("best", best, "best_busy", best_busy, "passed", [k_ for k_, r in res.items() if r["pass"]], flush=True)


def cmd_count(out: Path) -> None:
    """같은 모양(상관 ≥ 문턱)끼리 묶어 몇 묶음이 나오나. 표본 2만 개, 순차 대표 묶음."""
    _, s, _ = load_all()
    rng = np.random.default_rng(3)
    res = {}
    for n in (16, 32, 64):
        V = path_vecs(s["c"], n)
        idx = rng.choice(np.arange(n, len(V)), 20000, replace=False)
        X_ = V[idx]
        for th in (0.8, 0.9):
            reps = np.zeros((0, n), np.float32)
            for i in range(0, len(X_), 500):
                blk = X_[i:i + 500]
                for v in blk:
                    if len(reps) == 0 or float((reps @ v).max()) < th:
                        reps = np.vstack([reps, v[None]])
            res[f"N{n}@{th}"] = int(len(reps))
            print(n, th, len(reps), flush=True)
    (out / "count.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=300)
    ap.add_argument("cmd", choices=("knn", "eval", "count"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    {"knn": lambda: cmd_knn(out), "eval": lambda: cmd_eval(out, a.nsim), "count": lambda: cmd_count(out)}[a.cmd]()


if __name__ == "__main__":
    main()
