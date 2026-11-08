"""SCR 3단 돌파(15분봉 선물) 진입 시점·손절 변형. 등록: 2026-11-08-btc-scr15-timing-hypotheses.md

자료: round5 캐시(data15.npz, 선물 15분봉 + 펀딩) 뒤에 fapi 공개 시세로 지금까지 이어 붙인다.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/scr15-timing-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/scr15_timing.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from btc_fractal import cci, crossed, recent, rsi, stoch
from pine_eval import lev_path

M15, DAY = 900_000, 86_400_000
FEE = 0.0005
KST = timezone(timedelta(hours=9))
CACHE = Path("~/.minerva/research/round5-2026-11-08/data15.npz").expanduser()
ENTRIES = ("B0", "A2", "A4", "A8", "P4", "P8", "P16")
STOPS = ("S1", "C8", "C16")
VARIANTS = [(e, s, None) for e in ENTRIES for s in STOPS] + [("B0", "S1", "1h"), ("B0", "S1", "4h")]
N_VAR = len(VARIANTS)  # 23
N_TONIGHT_BEFORE = 5527


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def vname(v) -> str:
    return f"{v[0]}+{v[1]}" + (f"+D{v[2]}" if v[2] else "")


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-research"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def load() -> dict:
    d = np.load(CACHE)
    b = {k: d[k] for k in ("t", "o", "h", "l", "c", "ft", "fr")}
    now = int(time.time() * 1000)
    t, rows = int(b["t"][-1]) + M15, []
    while t < now - M15:
        r = _get(f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&startTime={t}&limit=1500")
        if not r:
            break
        rows += [x for x in r if x[6] < now]
        t = r[-1][0] + M15
    if rows:
        b["t"] = np.r_[b["t"], [x[0] for x in rows]]
        for i, k in enumerate(("o", "h", "l", "c"), 1):
            b[k] = np.r_[b[k], [float(x[i]) for x in rows]]
    f = _get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&startTime={int(b['ft'][-1]) + 1}&limit=1000")
    if f:
        b["ft"] = np.r_[b["ft"], [x["fundingTime"] for x in f]]
        b["fr"] = np.r_[b["fr"], [float(x["fundingRate"]) for x in f]]
    gaps = int((np.diff(b["t"]) != M15).sum())
    b["info"] = {"bars": len(b["t"]), "appended": len(rows), "gaps": gaps,
                 "first": int(b["t"][0]), "last": int(b["t"][-1])}  # fmt: skip
    return b


def direction(h, l, c) -> np.ndarray:
    k, r = cci(h, l, c, 20), rsi(c, 14)
    with np.errstate(invalid="ignore"):
        return np.where((k > 0) & (r > 50), 1, np.where((k < 0) & (r < 50), -1, 0))


def hf_dir(b: dict, period: int) -> np.ndarray:
    """15분봉 i 마감 시점에 끝나 있던 큰 봉의 방향."""
    t = b["t"]
    g = t // period
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    ends = np.r_[starts[1:], len(t)]
    h = np.array([b["h"][s:e].max() for s, e in zip(starts, ends)])
    l = np.array([b["l"][s:e].min() for s, e in zip(starts, ends)])
    c = b["c"][ends - 1]
    d = direction(h, l, c)
    gi = np.searchsorted(g[starts], (t + M15) // period) - 1  # 마감된 마지막 큰 봉
    out = np.where(gi >= 0, d[np.clip(gi, 0, None)], 0)
    return out


def features(b: dict) -> dict:
    h, l, c = b["h"], b["l"], b["c"]
    k, cc, rr = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    lx = recent(crossed(k, 20, True)) & recent(crossed(cc, -100, True)) & recent(crossed(rr, 30, True))
    sx = recent(crossed(k, 80, False)) & recent(crossed(cc, 100, False)) & recent(crossed(rr, 70, False))
    L = lx & ~np.r_[False, lx[:-1]]
    S = sx & ~np.r_[False, sx[:-1]]
    d15, d1h, d4h, d1d = direction(h, l, c), hf_dir(b, 3_600_000), hf_dir(b, 4 * 3_600_000), hf_dir(b, DAY)
    big = (d15 != 0) & (d15 == d4h) & (d15 == d1d)
    return {"k": k, "cci": cc, "rsi": rr, "L": L, "S": S, "d1h": d1h, "d4h": d4h, "big": big,
            "fcum": np.r_[0.0, np.cumsum(b["fr"])]}  # fmt: skip


def sim(b, F, v, a, z, L=None, S=None, delay=0, fee=FEE) -> tuple[list, dict]:
    entry, stop_kind, filt = v
    L = F["L"] if L is None else L
    S = F["S"] if S is None else S
    t, o, h, l, c = b["t"], b["o"], b["h"], b["l"], b["c"]
    k = F["k"]
    ft, fcum = b["ft"], F["fcum"]
    big = F["big"]
    fdir = F["d" + filt] if filt else None
    n_n = int(entry[1:]) if entry != "B0" else 0
    trades, cnt = [], {"chase_skip": 0, "void": 0, "filter_skip": 0}
    pos = pend = None

    def fund(d, t0, t1):
        i0, i1 = np.searchsorted(ft, t0, "right"), np.searchsorted(ft, t1, "right")
        return -d * float(fcum[i1] - fcum[i0])

    def close(i_bar, px, why, t_exit):
        nonlocal pos
        d, ep = pos["d"], pos["ep"]
        worst = min(pos["worst"], d * (px / ep - 1))
        fu = fund(d, pos["t"], t_exit)
        trades.append({"d": d, "j": pos["j"], "e": pos["e"], "x": i_bar, "ep": ep, "xp": px, "why": why,
                       "r": d * (px / ep - 1) - 2 * fee + fu, "worst": worst, "fund": fu})  # fmt: skip
        pos = None

    last = z - 1
    for i in range(a, z):
        if pos and i >= pos["e"]:
            d, ep, sp = pos["d"], pos["ep"], pos["stop"]
            hit = (l[i] <= sp) if d == 1 else (h[i] >= sp)
            if hit:
                gap = (o[i] < sp) if d == 1 else (o[i] > sp)
                close(i, o[i] if gap else sp, "stop", int(t[i]) + M15)
            else:
                pos["worst"] = min(pos["worst"], (l[i] / ep - 1) if d == 1 else -(h[i] / ep - 1))
        if i >= last - delay:
            continue
        sig = 1 if (L[i] and not S[i]) else (-1 if (S[i] and not L[i]) else 0)
        # 기다리던 진입 확인(이 봉 마감 기준)
        trig = False
        if pend and pend["j"] < i:
            j, d = pend["j"], pend["d"]
            if i - j > n_n:
                cnt["void"] += 1
                pend = None
            elif entry[0] == "A":
                trig = (c[i] > h[j]) if d == 1 else (c[i] < l[j])
            else:  # P: 50 너머 → 되돌림 → 다시 50 너머
                up = k[i] > 50 if d == 1 else k[i] < 50
                if pend["ph"] == 0 and up:
                    pend["ph"] = 1
                elif pend["ph"] == 1 and not up:
                    pend["ph"] = 2
                elif pend["ph"] == 2 and up:
                    trig = True
        if sig:
            if pos and pos["d"] == -sig:
                close(i + 1 + delay, o[i + 1 + delay], "opp", int(t[i + 1 + delay]))
            if pos is None and not (pend and pend["d"] == sig):
                if pend:
                    cnt["void"] += 1  # 반대 신호가 기다리던 신호를 지운다
                pend = {"d": sig, "j": i, "ph": 0}
                trig = entry == "B0"
        if trig and pend and pos is None:
            j, d = pend["j"], pend["d"]
            pend = None
            ei = i + 1 + delay
            ep = o[ei]
            if fdir is not None and fdir[j] == -d:
                cnt["filter_skip"] += 1
                continue
            lim = 0.03 if big[j] else 0.01
            if abs(ep / c[j] - 1) > lim:
                cnt["chase_skip"] += 1
                continue
            if stop_kind == "S1":
                dist = 0.01
            else:
                n = int(stop_kind[1:])
                ref = l[max(0, j - n + 1): j + 1].min() * 0.999 if d == 1 else h[max(0, j - n + 1): j + 1].max() * 1.001
                dist = min(max(d * (ep - ref) / ep, 0.002), 0.02)
            pos = {"d": d, "j": j, "e": ei, "ep": ep, "t": int(t[ei]), "stop": ep * (1 - d * dist), "worst": 0.0}
    if pos:
        close(last, c[last], "end", int(t[last]) + M15)
    return trades, cnt


def summ(tr: list, cnt: dict) -> dict:
    r = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    out = {"n": len(tr), "wins": int((r > 0).sum()), "win": float((r > 0).mean()) if len(r) else None,
           "stops": sum(x["why"] == "stop" for x in tr), "avg": float(r.mean()) if len(r) else None, **cnt}  # fmt: skip
    for L_ in (1, 5):
        p = lev_path(tr, L_)
        out[f"x{L_}"] = {kk: float(vv) for kk, vv in p.items()}
    return out


def windows(t: np.ndarray) -> dict:
    end = int(t[-1]) + M15
    w = {"fresh": (ms("2020-03-01"), ms("2021-11-01")), "mid": (ms("2022-01-01"), ms("2025-11-01")),
         "year": (ms("2025-11-01"), end), "d30": (end - 30 * DAY, end)}  # fmt: skip
    return {k: (int(np.searchsorted(t, a_)), int(np.searchsorted(t, z_))) for k, (a_, z_) in w.items()}


def random_p(b, F, v, a, z, real: float, nsim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nL, nS = int(F["L"][a:z].sum()), int(F["S"][a:z].sum())
    vals = []
    for _ in range(nsim):
        Lr, Sr = np.zeros(len(F["L"]), bool), np.zeros(len(F["L"]), bool)
        idx = rng.choice(np.arange(a, z - 2), nL + nS, replace=False)
        Lr[idx[:nL]] = True
        Sr[idx[nL:]] = True
        tr, cnt = sim(b, F, v, a, z, Lr, Sr)
        vals.append(lev_path(tr, 5)["ret"])
    vals = np.array(vals)
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "nsim": nsim}


def by_year(b, tr) -> dict:
    out: dict = {}
    for x in tr:
        y = datetime.fromtimestamp(int(b["t"][x["e"]]) / 1000, timezone.utc).year
        out.setdefault(y, []).append(x)
    return {y: {"n": len(v), "x5": float(lev_path(v, 5)["ret"]), "x1": float(lev_path(v, 1)["ret"])}
            for y, v in sorted(out.items())}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=300)
    out = Path(ap.parse_args().out).expanduser()
    nsim = ap.parse_args().nsim
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = load()
    F = features(b)
    W = windows(b["t"])
    print("data", b["info"], round(time.time() - t0, 1), flush=True)
    res, trades = {}, {}
    for v in VARIANTS:
        name = vname(v)
        res[name] = {}
        for w, (a, z) in W.items():
            tr, cnt = sim(b, F, v, a, z)
            res[name][w] = summ(tr, cnt)
            if w in ("d30", "year"):
                trades.setdefault(name, {})[w] = tr
        print(name, {w: (r["n"], r["x5"]["ret"]) for w, r in res[name].items()}, flush=True)
    base = res["B0+S1"]
    for r in res.values():
        r["better"] = {
            "fresh_x5_gt_base": r["fresh"]["x5"]["ret"] > base["fresh"]["x5"]["ret"],
            "mid_x5_gt_base": r["mid"]["x5"]["ret"] > base["mid"]["x5"]["ret"],
            "liq0": r["fresh"]["x5"]["liq"] == 0 and r["mid"]["x5"]["liq"] == 0,
            "x1_pos": r["fresh"]["x1"]["ret"] > 0 and r["mid"]["x1"]["ret"] > 0,
        }
    cands = [n for n in res if n != "B0+S1"]
    best = max(cands, key=lambda n: res[n]["fresh"]["x5"]["ret"] + res[n]["mid"]["x5"]["ret"])
    rnd = {}
    for name in ("B0+S1", best):
        v = next(x for x in VARIANTS if vname(x) == name)
        rnd[name] = {}
        for w in ("fresh", "mid"):
            a, z = W[w]
            rnd[name][w] = random_p(b, F, v, a, z, res[name][w]["x5"]["ret"], nsim, 7)
            print("random", name, w, rnd[name][w], flush=True)
    for name in res:
        p = rnd.get(name, {}).get("mid", {}).get("p")
        res[name]["better"]["random_bonf"] = bool(p is not None and p * N_VAR < 0.05)
        res[name]["better_all"] = all(res[name]["better"].values())
    red = {}
    for name in ("B0+S1", best):
        v = next(x for x in VARIANTS if vname(x) == name)
        red[name] = {}
        for w in ("fresh", "mid", "year"):
            a, z = W[w]
            red[name][w] = {"delay1": summ(*sim(b, F, v, a, z, delay=1)),
                            "fee15": summ(*sim(b, F, v, a, z, fee=FEE * 1.5))}  # fmt: skip
        a, z = W["fresh"][0], W["year"][1]
        red[name]["by_year"] = by_year(b, sim(b, F, v, a, z)[0])
    sig_counts = {w: {"long": int(F["L"][a:z].sum()), "short": int(F["S"][a:z].sum())} for w, (a, z) in W.items()}
    (out / "result.json").write_text(json.dumps(
        {"info": b["info"], "windows": {w: [int(b["t"][a]), int(b["t"][z - 1])] for w, (a, z) in W.items()},
         "n_variants": N_VAR, "n_tonight": N_TONIGHT_BEFORE + N_VAR, "signals": sig_counts, "best": best,
         "passed": [n for n in res if res[n].get("better_all")], "res": res, "random": rnd, "redteam": red,
         "sec": time.time() - t0}, indent=1, default=float))  # fmt: skip
    keep = {n: {w: [{kk: (float(vv) if isinstance(vv, (np.floating, float)) else int(vv) if isinstance(vv, (np.integer,)) else vv)
                     for kk, vv in x.items()} for x in tr] for w, tr in d.items()}
            for n, d in trades.items() if n in ("B0+S1", best)}  # fmt: skip
    (out / "trades.json").write_text(json.dumps({"t0": int(b["t"][0]), "trades": keep}, default=float))
    np.savez_compressed(out / "bars_recent.npz", **{k: b[k][W["year"][0] - 200:] for k in ("t", "o", "h", "l", "c")},
                        off=W["year"][0] - 200)  # fmt: skip
    print("best", best, "passed", [n for n in res if res[n].get("better_all")], round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
