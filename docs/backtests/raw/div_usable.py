"""다이버전스 실시간 사용 가능성 진단(D1 확정 지연, D2 상위 봉, D2b 체제 전환, D3 목표 먼저 vs 손절). 123개.

등록: docs/backtests/2026-11-08-btc-div-usable.md. 연구 전용(주문 없음, 바이낸스 POST 없음).
신호 논리는 pine_div.signals와 같고 오른쪽 봉 수 R만 바꾼다.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/div-usable-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/div_usable.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pine_div as P

DAY, HOUR, MIN = P.DAY, 3_600_000, P.MIN
N_DIAG, N_CUM = 123, 6536 + 123
MAINW = ("fresh", "design", "val", "year")
NR = 20
FRAMES = {"15": (15, (1, 2, 3, 5), (1, 4, 24)), "30": (30, (1, 2, 3, 5), (1, 4, 24)),
          "1h": (60, (2, 3, 5), (4, 24, 72, 168)), "4h": (240, (2, 3, 5), (4, 24, 72, 168)),
          "1d": (1440, (2, 3, 5), (4, 24, 72, 168))}  # fmt: skip
XS = (0.005, 0.01, 0.02)


def pivots(x, low, L, R):
    w = np.lib.stride_tricks.sliding_window_view(x, L + R + 1)
    mid = w[:, L]
    if low:
        ok = (mid < w[:, :L].min(1)) & (mid <= w[:, L + 1 :].min(1))
    else:
        ok = (mid > w[:, :L].max(1)) & (mid >= w[:, L + 1 :].max(1))
    return np.nonzero(ok)[0] + L


def signals(F, R):
    """pine_div.signals와 같고 확정 = 피벗 + R. 등급 S/M/W 모두."""
    out = []
    for d, low in ((1, True), (-1, False)):
        x = F["l"] if low else F["h"]
        prev = None
        for p in pivots(x, low, 5, R):
            cur = (x[p], F["cci"][p], F["rsi"][p], F["v"][p])
            if prev is not None and not np.isnan(cur[1]) and not np.isnan(prev[1]):
                pr = cur[0] < prev[0] if low else cur[0] > prev[0]
                dc = (cur[1] > prev[1]) if low else (cur[1] < prev[1])
                dr = (cur[2] > prev[2]) if low else (cur[2] < prev[2])
                vok = cur[3] > F["vsma"][p] * 1.2 and cur[3] > prev[3]
                tier = ("S" if vok else "M") if (pr and dc and dr) else ("W" if (pr and dc) else None)
                if tier and p + R < len(F["c"]):
                    out.append({"p": int(p), "c": int(p + R), "d": d, "tier": tier, "ext": float(cur[0])})
            prev = cur
    out.sort(key=lambda s: (s["c"], s["d"]))
    return out


def zp(hit, rnd, n):
    if n == 0 or rnd is None or rnd <= 0 or rnd >= 1:
        return None
    return P.norm_p((hit - rnd) / math.sqrt(rnd * (1 - rnd) / n))


def by_window(W, T, real, rand):
    """real: [n] 0/1, rand: [n, NR] 0/1(nan 허용). 구간별 적중·무작위·p."""
    out = {}
    for w in (*MAINW, "pool"):
        mk = (T >= W[w][0]) & (T < W[w][1])
        n = int(mk.sum())
        if n == 0:
            out[w] = {"n": 0}
            continue
        h, r = float(np.mean(real[mk])), float(np.nanmean(rand[mk]))
        out[w] = {"n": n, "hit": h, "rand": r, "ex": h - r, "p": zp(h, r, n)}
    return out


class Ctx:
    def __init__(self, m):
        self.t, self.o, self.h, self.l, self.c = m["t"], m["o"], m["h"], m["l"], m["c"]

    def px(self, T):
        return self.c[np.clip(np.searchsorted(self.t, T) - 1, 0, len(self.t) - 1)]

    def path(self, k0, d, sdist, cap=14 * 1440):
        """k0 분 시가 진입, 손절 거리 sdist. 반환 (손절 전 최대 유리 이동 비율, 손절 여부). 같은 분은 손절 먼저."""
        e = self.o[k0]
        stop = e * (1 - d * sdist)
        z = min(len(self.t), k0 + cap)
        s, w, ks = k0, 1440, -1
        while s < z:
            q = min(z, s + w)
            hit = (self.l[s:q] <= stop) if d == 1 else (self.h[s:q] >= stop)
            if hit.any():
                ks = s + int(np.argmax(hit))
                break
            s, w = q, w * 4
        end = ks if ks >= 0 else z
        if end <= k0:
            return 0.0, True
        fav = (self.h[k0:end].max() / e - 1) if d == 1 else (1 - self.l[k0:end].min() / e)
        return float(fav), ks >= 0


def rand_idx(rng, ct, T, span):
    lo, hi = np.searchsorted(ct, T - span), np.searchsorted(ct, T + span)
    return rng.integers(lo, np.maximum(hi, lo + 1), (NR,))


def frame_diag(C, F, sigs, W, span, horizons, rng, t_end):
    ct = F["ct"]
    ss = [s for s in sigs if ct[s["c"]] < t_end - max(horizons) * HOUR]
    T = np.array([ct[s["c"]] for s in ss], dtype=np.int64)
    Tp = np.array([ct[s["p"]] for s in ss], dtype=np.int64)
    d = np.array([s["d"] for s in ss])
    ext = np.array([s["ext"] for s in ss])
    RI = np.array([rand_idx(rng, ct, t, span) for t in T]).reshape(len(T), NR)
    TR = ct[RI]
    res = {"n_all": len(sigs)}
    a, z = W["pool"]
    mk = (T >= a) & (T < z)
    res["perday"] = float(mk.sum() / ((z - a) / DAY))
    pc, pp = C.px(T), C.px(Tp)
    res["gap_med"] = float(np.median((d * (pc - pp) / pp)[mk]))
    res["stop_med"] = float(np.median((d * (pc - ext) / pc)[mk]))
    for hh in horizons:
        H = hh * HOUR
        real = (d * (C.px(T + H) - pc) > 0).astype(float)
        hind = (d * (C.px(Tp + H) - pp) > 0).astype(float)
        rnd = (d[:, None] * (C.px(TR + H) - C.px(TR)) > 0).astype(float)
        res[f"hit_{hh}h"] = by_window(W, T, real, rnd)
        res[f"hind_{hh}h"] = by_window(W, T, hind, rnd)
    # D3: 확정 직후 1분봉 시가 진입, 손절 = 피벗 극값
    k0 = np.searchsorted(C.t, T)
    e = C.o[np.clip(k0, 0, len(C.t) - 1)]
    sd = d * (e - ext) / e
    ok = (sd > 0) & (k0 < len(C.t) - 2)
    fav, stp = np.zeros(len(T)), np.zeros(len(T), bool)
    rf = np.full((len(T), NR), np.nan)
    hf = np.zeros(len(T))
    for i in np.nonzero(ok)[0]:
        fav[i], stp[i] = C.path(int(k0[i]), int(d[i]), float(sd[i]))
        for j, tr in enumerate(TR[i]):
            kr = int(np.searchsorted(C.t, tr))
            if kr < len(C.t) - 2:
                rf[i, j] = C.path(kr, int(d[i]), float(sd[i]))[0]
        # 사후판: 피벗 봉 마감 직후 진입, 같은 피벗 손절
        kp = int(np.searchsorted(C.t, Tp[i]))
        ep = C.o[kp]
        sdp = d[i] * (ep - ext[i]) / ep
        hf[i] = C.path(kp, int(d[i]), float(sdp))[0] if sdp > 0 else 0.0
    res["d3_skipped"] = int((~ok).sum())
    Tk = T[ok]
    for X in XS:
        real = (fav[ok] >= X).astype(float)
        rnd = np.where(np.isnan(rf[ok]), np.nan, (rf[ok] >= X).astype(float))
        res[f"tb_{X}"] = by_window(W, Tk, real, rnd)
        res[f"tbh_{X}"] = by_window(W, Tk, (hf[ok] >= X).astype(float), rnd)
        res[f"tb_{X}"]["neither_pool"] = float(np.mean((fav[ok] < X) & ~stp[ok]))
    return res


def regime_diag(Fs, F, sigs, W, rng, t_end):
    """D2b: 신호 반대 체제에서 T6D 7일 / 4시간 방향 72시간 안에 신호 방향으로 바뀌는가."""
    ct = F["ct"]
    D, H4 = Fs["1d"], Fs["4h"]
    out = {}
    for name, G, horizon, start_ok in (
        ("t6d", D["t6d"], 7 * DAY, lambda s, d: s == -d),
        ("dir4h", H4["dir"], 72 * HOUR, lambda s, d: s != d),
    ):
        Gt = D["ct"] if name == "t6d" else H4["ct"]

        def state(T, G=G, Gt=Gt):
            j = np.searchsorted(Gt, T, "right") - 1
            return np.where(j >= 0, G[np.clip(j, 0, None)], 0)

        def flips(T, d, G=G, Gt=Gt, horizon=horizon):
            a, z = np.searchsorted(Gt, T, "right"), np.searchsorted(Gt, T + horizon, "right")
            return float(np.any(G[a:z] == d))

        cand = ct[(ct < t_end - horizon)]
        st_c = state(cand)
        Ts, real, rnd = [], [], []
        for s in sigs:
            T, d = int(ct[s["c"]]), s["d"]
            if T >= t_end - horizon or not start_ok(state(np.array([T]))[0], d):
                continue
            lo, hi = np.searchsorted(cand, T - 60 * DAY), np.searchsorted(cand, T + 60 * DAY)
            pool = lo + np.nonzero(start_ok(st_c[lo:hi], d))[0]
            if not len(pool):
                continue
            Ts.append(T)
            real.append(flips(T, d))
            rnd.append([flips(int(cand[j]), d) for j in rng.choice(pool, NR)])
        out[name] = by_window(W, np.array(Ts, dtype=np.int64), np.array(real), np.array(rnd).reshape(-1, NR))
    return out


def gate(cell):
    pl = cell.get("pool", {})
    if not pl.get("n") or pl.get("p") is None:
        return False
    big = [w for w in MAINW if cell[w].get("n", 0) >= 10]
    return (pl["p"] * N_DIAG < 0.05 and len(big) >= 3 and all(cell[w]["ex"] > 0 for w in big)
            and pl["ex"] >= 0.03)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b5 = P.load_5m(out)
    m = P.load_1m(out)
    t_end = int(min(m["t"][-1], b5["t"][-1] + 300_000))
    Fs = {k: P.agg(b5, v[0]) for k, v in FRAMES.items()}
    del b5
    for k in Fs:
        keep = Fs[k]["ct"] <= t_end
        for x in ("t", "o", "c", "h", "l", "v", "ct", "rsi", "cci", "vsma", "dir"):
            Fs[k][x] = Fs[k][x][keep]
    Fs["1d"]["t6d"] = P.t6d_daily(Fs["1d"])
    C = Ctx(m)
    W = P.windows(t_end)
    rng = np.random.default_rng(20261108)
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    diag = {"t_end": t_end}
    for k, (_, rbs, hz) in FRAMES.items():
        span = 15 * DAY if k in ("15", "30", "1h") else 60 * DAY
        for R in rbs:
            sg = signals(Fs[k], R)
            r = frame_diag(C, Fs[k], sg, W, span, hz, rng, t_end)
            if k in ("4h", "1d"):
                r["regime"] = regime_diag(Fs, Fs[k], sg, W, rng, t_end)
            diag[f"{k}|R{R}"] = r
            h = r[f"hit_{hz[0]}h"]["pool"]
            print(k, R, f"{r['perday']:.2f}/d hit{hz[0]}h {h['hit']:.3f} vs {h['rand']:.3f}",
                  f"tb1% {r['tb_0.01']['pool']['hit']:.3f} vs {r['tb_0.01']['pool']['rand']:.3f}",
                  f"{time.time() - t0:.0f}s", flush=True)  # fmt: skip
    # 진단 판정(123개)
    cells, passed = {}, []
    for key, r in diag.items():
        if key == "t_end":
            continue
        k = key.split("|")[0]
        hz = FRAMES[k][2]
        names = [f"hit_{h}h" for h in hz] + [f"tb_{x}" for x in XS]
        for nm in names:
            cells[f"{key}|{nm}"] = r[nm]
        for nm in r.get("regime", {}):
            cells[f"{key}|reg_{nm}"] = r["regime"][nm]
    assert len(cells) == N_DIAG, len(cells)
    for nm, c in cells.items():
        if gate(c):
            passed.append(nm)
    diag["gate_pass"] = passed
    diag["n_cells"] = len(cells)
    diag["secs"] = time.time() - t0
    (out / "diag.json").write_text(json.dumps(diag, default=float))
    print("gate pass:", passed, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
