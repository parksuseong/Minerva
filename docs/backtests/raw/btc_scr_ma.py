"""BTCUSDT 30분봉 SCR + 이평 조건 탐색. 등록: 2026-11-02-btc-scr-ma-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src python docs/backtests/raw/btc_scr_ma.py --out <폴더>
조합별로 진입·청산 위치와 총수익만 배열로 남긴다(메모리 제한).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from minerva.strategy.indicators import Bar, rsi, tm_states

NS = (20, 50, 60, 100, 120, 200, 240, 480, 720, 1440, 2400, 3120)
TAIL = 30_000
YEAR = 365 * 48
SLOPE = 10
STOP = 0.01
WIN = 0.001
COSTS = (0.0005, 0.001)
SIGNALS = ("S3", "R")
EXITS = ("O", "M")


def load(path: Path, offset_min: int = 0, merge: bool = False) -> dict:
    """봉 파일을 배열로. merge면 15분봉을 offset_min만큼 밀어 30분봉으로 묶는다."""
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                (int(r["open_time"]), float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]))
            )
    rows = rows[-TAIL * (2 if merge else 1) :]
    if merge:
        groups: dict = {}
        for t, o, h, l, c in rows:
            k = (t - offset_min * 60_000) // 1_800_000
            g = groups.get(k)
            if g is None:
                groups[k] = [t, o, h, l, c, 1]
            else:
                g[2], g[3], g[4], g[5] = max(g[2], h), min(g[3], l), c, g[5] + 1
        rows = [tuple(v[:5]) for _, v in sorted(groups.items()) if v[5] == 2]
    a = np.array(rows, dtype=float)
    return {"t": a[:, 0], "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4]}


def raw_signals(d: dict) -> dict:
    bars = [
        Bar(datetime.fromtimestamp(t / 1000, timezone.utc).date(), o, h, l, c, 0.0)
        for t, o, h, l, c in zip(d["t"], d["o"], d["h"], d["l"], d["c"])
    ]
    st = tm_states(bars)
    ll = np.array([s.long_level for s in st])
    sl = np.array([s.short_level for s in st])
    s3l = (ll == 3) & (np.r_[0, ll[:-1]] != 3)
    s3s = (sl == 3) & (np.r_[0, sl[:-1]] != 3)
    r = np.array([np.nan if v is None else v for v in rsi(d["c"].tolist())], dtype=float)
    p = np.r_[np.nan, r[:-1]]
    return {"S3": (s3l, s3s), "R": ((p <= 30) & (r > 30), (p >= 70) & (r < 70))}


def sma(c: np.ndarray, n: int) -> np.ndarray:
    s = np.cumsum(np.r_[0.0, c])
    out = np.full(len(c), np.nan)
    out[n - 1 :] = (s[n:] - s[:-n]) / n
    return out


def ema(c: np.ndarray, n: int) -> np.ndarray:
    k = 2 / (n + 1)
    out = np.empty(len(c))
    v = c[0]
    for i, x in enumerate(c.tolist()):
        v += k * (x - v)
        out[i] = v
    out[: n - 1] = np.nan
    return out


def all_mas(c: np.ndarray) -> dict:
    out = {}
    for n in NS:
        out["S", n] = sma(c, n)
        out["E", n] = ema(c, n)
    return out


def filters() -> list:
    out: list = [("none",)]
    for ty in "SE":
        for n in NS:
            out += [("P", ty, n), ("D", ty, n)]
        for i, f in enumerate(NS):
            for s in NS[i + 1 :]:
                out += [("X", ty, f, s), ("A", ty, f, s)]
    return out


def masks(flt: tuple, c: np.ndarray, m: dict) -> tuple:
    """(롱 허용, 숏 허용, 롱 청산, 숏 청산) 불리언 배열."""
    if flt[0] == "none":
        ok = np.ones(len(c), bool)
        no = np.zeros(len(c), bool)
        return ok, ok, no, no
    ty = flt[1]
    if flt[0] in ("P", "D"):
        ref = m[ty, flt[2]]
    else:
        f, ref = m[ty, flt[2]], m[ty, flt[3]]
    if flt[0] == "P":
        lo, sh = c > ref, c < ref
    elif flt[0] == "D":
        prev = np.r_[np.full(SLOPE, np.nan), ref[:-SLOPE]]
        lo, sh = ref > prev, ref < prev
    elif flt[0] == "X":
        lo, sh = f > ref, f < ref
    else:
        lo, sh = (c > f) & (f > ref), (c < f) & (f < ref)
    below, above = c < ref, c > ref
    return lo, sh, below & np.r_[False, above[:-1]], above & np.r_[False, below[:-1]]


def run(d: dict, sig: tuple, perm: tuple, exit_kind: str, a: int, b: int) -> tuple:
    """[a,b) 시뮬레이션. (진입 봉, 청산 봉, 방향, 총수익) 배열을 낸다.

    신호는 i봉 마감, 체결은 i+1봉 시가. 손절은 진입 봉부터 본다.
    """
    o, h, l = d["o"], d["h"], d["l"]
    rl, rs = sig
    ok_l, ok_s, ex_l, ex_s = perm
    al = np.flatnonzero(rl[a:b - 1] & ok_l[a:b - 1]) + a
    als = np.flatnonzero(rs[a:b - 1] & ok_s[a:b - 1]) + a
    xl = np.flatnonzero(rs[a:b - 1] | (ex_l[a:b - 1] if exit_kind == "M" else False)) + a
    xs = np.flatnonzero(rl[a:b - 1] | (ex_s[a:b - 1] if exit_kind == "M" else False)) + a
    ei, xi, sd, gr = [], [], [], []
    p = a  # 이 봉 마감 신호부터 본다
    pending = None
    while True:
        if pending is None:
            jl = np.searchsorted(al, p)
            js = np.searchsorted(als, p)
            il = al[jl] if jl < len(al) else b
            ix = als[js] if js < len(als) else b
            if il >= b and ix >= b:
                break
            i, side = (il, 1) if il <= ix else (ix, -1)
        else:
            i, side = pending
            pending = None
        e = int(i) + 1
        entry = o[e]
        stop = entry * (1 - STOP * side)
        arr = xl if side > 0 else xs
        j = np.searchsorted(arr, e)
        x = int(arr[j]) if j < len(arr) else None
        u = (x + 1) if x is not None else b
        hit = l[e:u] <= stop if side > 0 else h[e:u] >= stop
        k = int(hit.argmax()) if len(hit) else 0
        if len(hit) and hit[k]:
            st = e + k
            px = min(o[st], stop) if side > 0 else max(o[st], stop)
            end = st
            p = st
        elif x is None:
            px, end, p = d["c"][b - 1], b - 1, b
        else:
            end = x + 1
            px = o[end]
            p = end
            opp_raw = rs[x] if side > 0 else rl[x]
            opp_ok = ok_s[x] if side > 0 else ok_l[x]
            if opp_raw and opp_ok:
                pending = (x, -side)
        ei.append(e)
        xi.append(end)
        sd.append(side)
        gr.append((px / entry - 1) * side)
        if p >= b - 1 and pending is None:
            break
    return (
        np.array(ei, dtype=np.int32),
        np.array(xi, dtype=np.int32),
        np.array(sd, dtype=np.int8),
        np.array(gr, dtype=np.float32),
    )


def stats(tr: tuple, lo: int, hi: int, cost: float) -> dict:
    ei, _, sd, gr = tr
    k = (ei >= lo) & (ei < hi)
    net = gr[k].astype(float) - 2 * cost
    if not len(net):
        return {"n": 0, "win": 0.0, "ret": 0.0, "mdd": 0.0, "avg": 0.0, "med": 0.0, "worst": 0.0}
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {
        "n": int(len(net)),
        "win": round(float((net > WIN).mean() * 100), 1),
        "ret": round(float((eq[-1] - 1) * 100), 1),
        "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
        "avg": round(float(net.mean() * 100), 3),
        "med": round(float(np.median(net) * 100), 3),
        "worst": round(float(net.min() * 100), 2),
        "long": int((sd[k] > 0).sum()),
    }


def key(flt: tuple) -> str:
    return "_".join(map(str, flt))


def grid(d: dict, sigs: dict, perms: list, a: int, b: int) -> dict:
    out = {}
    for sn in SIGNALS:
        for fi, flt in enumerate(FILTERS):
            for ex in EXITS:
                if flt[0] == "none" and ex == "M":
                    continue
                out[sn, fi, ex] = run(d, sigs[sn], perms[fi], ex, a, b)
    return out


def best_win(res: dict, lo: int, hi: int, cost: float, minn: int) -> list:
    rows = []
    for k, tr in res.items():
        s = stats(tr, lo, hi, cost)
        if s["n"] >= minn:
            rows.append((s["win"], s["ret"], k, s))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    return rows


def recursive(res: dict, months: list) -> tuple:
    """7번째 구간부터 그 전 구간으로 승률 최고(20회+, 수익 플러스)를 골라 다음 구간에 쓴다."""
    picks = []
    pnl = {c: [] for c in COSTS}
    for m in range(6, 12):
        lo, hi = months[m]
        cut = months[0][0]
        rows = []
        for k, tr in res.items():
            ei, xi, sd, gr = tr
            sel = (ei >= cut) & (xi < lo)
            if sel.sum() < 20:
                continue
            net = gr[sel].astype(float) - 2 * COSTS[0]
            total = np.prod(1 + net) - 1
            if total <= 0:
                continue
            rows.append(((net > WIN).mean(), total, k))
        if not rows:
            picks.append(None)
            continue
        rows.sort(key=lambda r: (-r[0], -r[1]))
        k = rows[0][2]
        picks.append((k[0], key(FILTERS[k[1]]), k[2], round(rows[0][0] * 100, 1)))
        ei, _, _, gr = res[k]
        mask_ = (ei >= lo) & (ei < hi)
        for c in COSTS:
            pnl[c] += (gr[mask_].astype(float) - 2 * c).tolist()
    return picks, {
        str(c): {
            "n": len(v),
            "win": round(float((np.array(v) > WIN).mean() * 100), 1) if v else 0.0,
            "ret": round(float((np.prod(1 + np.array(v)) - 1) * 100), 1) if v else 0.0,
        }
        for c, v in pnl.items()
    }


def fit_logit(x: np.ndarray, y: np.ndarray, lam: float = 1.0, steps: int = 30) -> tuple:
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.c_[np.ones(len(x)), (x - mu) / sd]
    w = np.zeros(z.shape[1])
    reg = np.eye(z.shape[1]) * lam
    reg[0, 0] = 0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        g = z.T @ (p - y) + reg @ w
        hs = (z.T * (p * (1 - p) + 1e-6)) @ z + reg
        w -= np.linalg.solve(hs, g)
    return w, mu, sd


def regression(d: dict, sigs: dict, m: dict, months: list, base: dict, none_i: int) -> dict:
    c = d["c"]
    dist = np.column_stack([np.log(c / m["S", n]) for n in NS])
    out = {}
    for sn in SIGNALS:
        ei, xi, sd, gr = base[sn, none_i, "O"]
        pnl = {cc: [] for cc in COSTS}
        for k in range(6, 12):
            lo, hi = months[k]
            tr = xi < lo
            if tr.sum() < 40:
                continue
            xs = dist[ei[tr] - 1] * sd[tr, None]
            ys = ((gr[tr].astype(float) - 2 * COSTS[0]) > WIN).astype(float)
            if ys.min() == ys.max():
                continue
            w, mu, s = fit_logit(xs, ys)
            def prob(v):
                z = np.nan_to_num((v - mu) / s)
                return (np.c_[np.ones(len(z)), z] @ w) > 0
            perm = (prob(dist), prob(-dist), np.zeros(len(c), bool), np.zeros(len(c), bool))
            t = run(d, sigs[sn], perm, "O", lo, hi)
            for cc in COSTS:
                pnl[cc] += (t[3].astype(float) - 2 * cc).tolist()
        out[sn] = {
            str(cc): {
                "n": len(v),
                "win": round(float((np.array(v) > WIN).mean() * 100), 1) if v else 0.0,
                "ret": round(float((np.prod(1 + np.array(v)) - 1) * 100), 1) if v else 0.0,
            }
            for cc, v in pnl.items()
        }
    return out


def shuffle_test(d, sigs, perms, a, b, n_iter, real, minn):
    rng = random.Random(20261102)
    span = b - a
    beat = 0
    tops = []
    for it in range(n_iter):
        sh = rng.randrange(30 * 48, span - 30 * 48)
        rolled = []
        for q in perms:
            r = []
            for arr in q:
                c = arr.copy()
                c[a:b] = np.roll(arr[a:b], sh)
                r.append(c)
            rolled.append(tuple(r))
        res = grid(d, sigs, rolled, a, b)
        top = best_win(res, a, b, COSTS[0], minn)
        v = top[0][0] if top else 0.0
        tops.append(v)
        beat += v >= real
        if (it + 1) % 20 == 0:
            print(f"shuffle {it + 1}/{n_iter} beat={beat}", flush=True)
    return {"p": round((beat + 1) / (n_iter + 1), 4), "median": float(np.median(tops))}


FILTERS = filters()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    p.add_argument("--shuffles", type=int, default=200)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    d = load(data / "30m.csv")
    n = len(d["c"])
    a, b = n - YEAR, n
    step = YEAR // 12
    months = [(a + i * step, a + (i + 1) * step if i < 11 else b) for i in range(12)]
    cut = months[9][0]
    sigs = raw_signals(d)
    m = all_mas(d["c"])
    perms = [masks(f, d["c"], m) for f in FILTERS]
    print("filters", len(FILTERS), "bars", n, flush=True)
    res = grid(d, sigs, perms, a, b)
    none_i = 0
    rep: dict = {
        "window": [
            datetime.fromtimestamp(d["t"][a] / 1000, timezone.utc).isoformat(),
            datetime.fromtimestamp(d["t"][-1] / 1000, timezone.utc).isoformat(),
        ],
        "hold": {
            "year": round((d["c"][-1] / d["o"][a] - 1) * 100, 1),
            "last3": round((d["c"][-1] / d["o"][cut] - 1) * 100, 1),
        },
        "configs": len(res),
        "raw_counts": {s: [int(sigs[s][0][a:b].sum()), int(sigs[s][1][a:b].sum())] for s in SIGNALS},
    }
    rep["baseline"] = {
        f"{s}_O": {
            "year": stats(res[s, none_i, "O"], a, b, COSTS[0]),
            "last3": stats(res[s, none_i, "O"], cut, b, COSTS[0]),
        }
        for s in SIGNALS
    }

    def card(k):
        tr = res[k]
        base = res[k[0], none_i, "O"]
        return {
            "sig": k[0],
            "filter": key(FILTERS[k[1]]),
            "exit": k[2],
            "year": stats(tr, a, b, COSTS[0]),
            "year_c10": stats(tr, a, b, COSTS[1]),
            "first9": stats(tr, a, cut, COSTS[0]),
            "last3": stats(tr, cut, b, COSTS[0]),
            "last3_c10": stats(tr, cut, b, COSTS[1]),
            "base_last3_win": stats(base, cut, b, COSTS[0])["win"],
        }

    front = best_win(res, a, cut, COSTS[0], 30)
    year = best_win(res, a, b, COSTS[0], 30)
    rep["front_top2"] = [card(r[2]) for r in front[:2]]
    rep["year_top2"] = [card(r[2]) for r in year[:2]]
    rep["year_top10"] = [
        {"sig": r[2][0], "filter": key(FILTERS[r[2][1]]), "exit": r[2][2], **r[3]}
        for r in year[:10]
    ]
    rep["front_top_ret"] = sorted(
        (
            {"sig": r[2][0], "filter": key(FILTERS[r[2][1]]), "exit": r[2][2], **card(r[2])["last3"]}
            for r in front[:20]
        ),
        key=lambda x: -x["ret"],
    )
    wins = np.array([r[0] for r in front])
    rets = np.array([r[1] for r in year])
    rep["distribution"] = {
        "front_n": len(front),
        "front_win_p50": float(np.median(wins)) if len(wins) else None,
        "year_n": len(year),
        "year_ret_positive": int((rets > 0).sum()),
    }
    picks, rec = recursive(res, months)
    rep["recursive"] = {"picks": picks, "result": rec}
    rep["regression"] = regression(d, sigs, m, months, res, none_i)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False)[:600], flush=True)

    d2 = load(data / "15m.csv", offset_min=15, merge=True)
    sig2 = raw_signals(d2)
    m2 = all_mas(d2["c"])
    n2 = len(d2["c"])
    a2, cut2 = n2 - YEAR, n2 - YEAR + 9 * step
    rep["offset"] = []
    for r in list(front[:2]) + list(year[:2]):
        k = r[2]
        t2 = run(d2, sig2[k[0]], masks(FILTERS[k[1]], d2["c"], m2), k[2], a2, n2)
        rep["offset"].append(
            {"filter": key(FILTERS[k[1]]), "sig": k[0], "exit": k[2],
             "year": stats(t2, a2, n2, COSTS[0]), "last3": stats(t2, cut2, n2, COSTS[0])}
        )
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))

    real = year[0][0] if year else 0.0
    rep["shuffle"] = shuffle_test(d, sigs, perms, a, b, args.shuffles, real, 30)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("done", rep["shuffle"], flush=True)


if __name__ == "__main__":
    main()
