"""트레이딩뷰 21개 이식 + 앙상블 4 + S70 탐색 + H-6D-T1, 직전 pine_eval 틀 그대로.

문서: docs/backtests/2026-11-08-pine-tv-hypotheses.md (사전 등록 그대로)
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/pine-tv-2026-11-08
  P="systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
     PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3"
  $P docs/backtests/raw/pine_tv_eval.py --out $OUT grid      # 설계: 32 세트 × 19 + 앙상블 4 × 18
  $P ... holdout    # 대표 표본 밖 1회, 최근 30일, 결선 장기
  $P ... s70        # 설계: 56 진입원 × 16 모양 × 4 문
  $P ... s70hold    # S70 결선 10개 표본 밖 1회
  $P ... random --items 이름:변형 ...   # 무작위·시각 맞춤 무작위
  $P ... hyp6d      # 현물 2017-09 ~ 2020-07 H-6D-T1
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pine_eval as E
import pine_ports as P
import pine_tv_ports as T
from pine_data import load

FEE = E.FEE
M15 = E.M15
DAY = E.DAY
EXITS, STOPS = E.EXITS, E.STOPS
N_TOTAL = 4265  # 사전 등록한 변형 수(본페로니)
OLD_SETS = ("1", "2m", "2D", "3m", "3D", "4", "5", "6m", "6D", "7", "8", "9", "10m", "10D")
OLD_ENS = ("E1", "E2", "E3", "E4", "E5", "E6")
NEW_ENS = ("X1", "X2", "X3", "X4")
K_TP = (0.5, 1.0, 2.0, 3.0)
K_SL = (1.0, 2.0, 3.0, 0.0)  # 0 = 손절 없음
GATES = ("G0", "G10", "G3", "GT")


# ---------------- 모의 매매(pine_eval.simulate + 자체 청산 훅 + K 모양) ----------------
def simulate(B, S: dict, var: tuple, a: int, z: int, sig=None, epx=None) -> list:
    """var = ("N",) 자체 청산, (exit, stop) 직전 묶음, ("K", k, m) S70 모양."""
    native = S.get("native") if var[0] == "N" else None
    if sig is None:
        sig = S["sig"]
        if native is not None and "sig_all" in native:
            sig = native["sig_all"]
        epx = S.get("epx")
    intrabar = epx is not None
    sig_l = sig.tolist() if hasattr(sig, "tolist") else list(sig)
    idx = (np.flatnonzero(np.asarray(sig[a:z]) != 0) + a).tolist()
    reverse = native["reverse"] if native is not None else True
    ex_kind = var[0]
    stop_kind = var[1] if (len(var) > 1 and ex_kind != "K") else "S0"
    o, h, lo, c = B.o, B.h, B.l, B.c
    atr, rsi = B.atr, B.rsi
    exl = native.get("exl") if native else None
    exs = native.get("exs") if native else None
    dyl = native.get("dyn_sl_l") if native else None
    dys = native.get("dyn_sl_s") if native else None
    h_init = native.get("init") if native else None
    h_close = native.get("on_close") if native else None
    trades: list = []
    pos = None
    i = a
    pending = None

    def open_pos(dr: int, j: int, e: int, ep: float) -> dict:
        ref = j - 1 if intrabar else j
        at = atr[ref]
        p = {
            "d": dr, "j": j, "e": e, "ep": ep, "size": 1.0, "real": 0.0, "worst": -FEE,
            "sl": None, "tps": [], "trail": False, "hx": ep, "half_t": None, "be": None,
            "tin": int(B.t[e]) + (1 if intrabar else 0),
        }  # fmt: skip
        if native is not None:
            if "sl" in native and not math.isnan(native["sl"][j]):
                p["sl"] = float(native["sl"][j])
            if "tp" in native and not math.isnan(native["tp"][j]):
                p["tps"] = [(1.0, float(native["tp"][j]))]
            if "be_risk" in native:
                p["be"] = float(native["be_risk"][j])
            if "ladder" in native:
                p["tps"] = [(f, ep * (1 + dr * k)) for f, k in native["ladder"]]
                p["sl"] = ep * (1 - dr * native["sl_pct"])
            if "daygain_risk" in native:
                p["dg"] = float(native["daygain_risk"][j])
            if h_init is not None:
                h_init(p, j, ep, dr)
        elif ex_kind == "K":
            p["tps"] = [(1.0, ep + dr * var[1] * at)]
            if var[2] > 0:
                p["sl"] = ep - dr * var[2] * at
        else:
            if ex_kind in ("T1", "T2", "T3"):
                k = int(ex_kind[1])
                p["tps"] = [(1.0, ep + dr * k * at)]
            elif ex_kind == "P":
                p["tps"] = [(0.5, ep + dr * at)]
                p["trail_k"] = 3.0
            if stop_kind == "S1":
                p["sl"] = ep - dr * 3 * at
            elif stop_kind == "S2":
                p["sl"] = (B.ll20[ref] - 0.25 * at) if dr == 1 else (B.hh20[ref] + 0.25 * at)
        return p

    def close_part(p: dict, frac: float, px: float) -> None:
        p["real"] += frac * (p["d"] * (px / p["ep"] - 1) - FEE * px / p["ep"])
        p["size"] -= frac

    def finish(p: dict, x: int, px: float, reason: str, t_out: int) -> None:
        close_part(p, p["size"], px)
        r = p["real"] - FEE
        fund = E.funding(B, p["d"], p["ep"], p["tin"], t_out, p["half_t"])
        r -= fund
        p["worst"] = min(p["worst"], r)
        trades.append(
            {"d": p["d"], "j": p["j"], "e": p["e"], "x": x, "ep": p["ep"], "xp": px, "r": r,
             "worst": p["worst"], "fund": fund, "why": reason}
        )  # fmt: skip

    while i < z:
        if pos is None:
            if pending is not None:
                dr, j = pending
                pending = None
                pos = open_pos(dr, j, i, o[i])
            else:
                k = bisect.bisect_left(idx, i)
                if k >= len(idx):
                    break
                j = idx[k]
                if intrabar:
                    if j >= z:
                        break
                    pos = open_pos(sig_l[j], j, j, float(epx[j]))
                    i = j
                else:
                    if j + 1 >= z:
                        break
                    pos = open_pos(sig_l[j], j, j + 1, o[j + 1])
                    i = j + 1
        p = pos
        dr = p["d"]
        ep = p["ep"]
        skip_intra = intrabar and i == p["e"]
        exited = False
        if not skip_intra:
            stop = p["sl"]
            if native is not None and dyl is not None:
                dyn = dyl[i] if dr == 1 else dys[i]
                if not math.isnan(dyn):
                    stop = dyn if stop is None else (max(stop, dyn) if dr == 1 else min(stop, dyn))
            if p["trail"]:
                tr_ = p["trail_px"]
                stop = tr_ if stop is None else (max(stop, tr_) if dr == 1 else min(stop, tr_))
            adverse = lo[i] if dr == 1 else h[i]
            if stop is not None:
                hit_open = o[i] <= stop if dr == 1 else o[i] >= stop
                hit = lo[i] <= stop if dr == 1 else h[i] >= stop
                if hit_open or hit:
                    px = o[i] if hit_open else stop
                    mtm = p["real"] + p["size"] * (dr * (px / ep - 1)) - FEE
                    p["worst"] = min(p["worst"], mtm)
                    finish(p, i, px, "stop", int(B.t[i]) + 1)
                    pos = None
                    exited = True
            if not exited:
                mtm = p["real"] + p["size"] * (dr * (adverse / ep - 1)) - FEE
                p["worst"] = min(p["worst"], mtm)
                while p["tps"]:
                    f, px = p["tps"][0]
                    hit_open = o[i] >= px if dr == 1 else o[i] <= px
                    hit = h[i] >= px if dr == 1 else lo[i] <= px
                    if not (hit_open or hit):
                        break
                    fill = o[i] if hit_open else px
                    p["tps"].pop(0)
                    if f >= p["size"] - 1e-9 or not p["tps"] and "trail_k" not in p:
                        finish(p, i, fill, "tp", int(B.t[i]) + 1)
                        pos = None
                        exited = True
                        break
                    close_part(p, f, fill)
                    p["half_t"] = int(B.t[i])
                    if "trail_k" in p:
                        p["trail"] = True
                        p["trail_px"] = ep
                        p["hx"] = h[i] if dr == 1 else lo[i]
        if exited:
            i = i + 1 if intrabar else i
            continue
        if p["trail"]:
            p["hx"] = max(p["hx"], h[i]) if dr == 1 else min(p["hx"], lo[i])
            tp_ = p["hx"] - dr * p["trail_k"] * atr[i]
            p["trail_px"] = max(p["trail_px"], tp_) if dr == 1 else min(p["trail_px"], tp_)
        if p["be"] is not None and p["sl"] is not None:
            if (h[i] >= ep + p["be"]) if dr == 1 else (lo[i] <= ep - p["be"]):
                p["sl"] = max(p["sl"], ep) if dr == 1 else min(p["sl"], ep)
        why = None
        if native is not None:
            if exl is not None and (exl[i] if dr == 1 else exs[i]):
                why = "cond"
            if h_close is not None:
                w2 = h_close(p, i)
                if w2 is not None:
                    why = w2
            if native.get("time_bars") and i - p["e"] >= native["time_bars"]:
                why = "time"
            if "dg" in p and native["day_first"][i]:
                rr = dr * (c[i] - ep) / p["dg"] if p["dg"] > 0 else 0
                if 0.9 < rr <= 1.5:
                    why = "daygain"
        elif ex_kind == "R" and i > p["e"]:
            if dr == 1 and rsi[i - 1] >= 70 > rsi[i]:
                why = "rsi"
            if dr == -1 and rsi[i - 1] <= 30 < rsi[i]:
                why = "rsi"
        s_now = sig_l[i] if i < z - 1 else 0
        if reverse and s_now == -dr and not intrabar:
            why = "opp"
        if why is not None:
            if i + 1 < z:
                finish(p, i + 1, o[i + 1], why, int(B.t[i + 1]))
                pos = None
                if not intrabar and s_now != 0:
                    pending = (s_now, i)
                i += 1
                continue
            break
        if intrabar and reverse and i + 1 < z and sig_l[i + 1] == -dr:
            px = float(epx[i + 1])
            finish(p, i + 1, px, "opp", int(B.t[i + 1]) + 1)
            pos = None
            i += 1
            continue
        i += 1
    if pos is not None:
        x = min(i, z - 1)
        finish(pos, x, c[x], "end", int(B.t[x]) + M15)
    return trades


# ---------------- 세트·앙상블·문 ----------------
def build_all(b: dict, old: bool = True) -> tuple:
    n = len(b["c"])
    S = T.build(b)
    d = P.daily(b)
    gd = P.s3_gauss_core(d["h"], d["l"], d["c"])
    g3 = np.where(P.daily_state_15(d, gd["up"].astype(float), n) == 1, 1,
                  np.where(P.daily_state_15(d, gd["dn"].astype(float), n) == 1, -1, 0))  # fmt: skip
    g10 = np.nan_to_num(P.daily_state_15(d, P.s10_ma(d["c"])["state"], n))
    gt = np.nan_to_num(P.daily_state_15(d, T.th1_dir(d), n))
    s6d = P.daily_to_15(d, P.s6_trend_dev(d["h"], d["l"], d["c"]), n)
    st6 = P.last_dir_state(s6d)
    gates = {"G0": None, "G10": g10, "G3": g3, "GT": gt}

    def union(names):
        ul = np.zeros(n, bool)
        us = np.zeros(n, bool)
        for k in names:
            ul |= S[k]["sig"] == 1
            us |= S[k]["sig"] == -1
        return ul & ~us, us & ~ul

    def gate(ul, us, st):
        return np.where(ul & (st == 1), 1, np.where(us & (st == -1), -1, 0)).astype(np.int8)

    X = {}
    X["X1"] = gate(*union(["T01H", "T05H", "T11H", "T14H", "T15H", "T19H", "T22H"]), g3)
    X["X2"] = gate(*union(["T12", "T21", "T22H"]), st6)
    X["X3"] = gate(*union(["T02", "T08", "T12", "T16", "T17", "T18", "T20", "T21"]), gt)
    sts = [S[k]["state"] for k in ("T01H", "T05H", "T11H", "T19H", "T14H")]
    bull = sum((np.nan_to_num(s) == 1).astype(int) for s in sts) >= 4
    bear = sum((np.nan_to_num(s) == -1).astype(int) for s in sts) >= 4
    X["X4"] = np.where(E._rise(bull), 1, np.where(E._rise(bear), -1, 0)).astype(np.int8)
    for k, v in X.items():
        S[k] = {"sig": v, "native": None}
    if old:
        So = E.build_sets(b)
        for k in OLD_SETS + OLD_ENS:
            S["o" + k] = So[k]
    return S, gates


def variants(name: str, S: dict) -> list:
    vs = [("N",)] if S[name].get("native") is not None else []
    return vs + [(e, s) for e in EXITS for s in STOPS]


def vname(v: tuple) -> str:
    if v[0] == "N":
        return "N"
    if v[0] == "K":
        return f"K{v[1]:g}/{v[2]:g}"
    return f"{v[0]}/{v[1]}"


def parse_v(s: str) -> tuple:
    if s == "N":
        return ("N",)
    if s.startswith("K"):
        k, m = s[1:].split("/")
        return ("K", float(k), float(m))
    return tuple(s.split("/"))


def gated(S: dict, name: str, gates: dict, g: str) -> dict:
    st = gates[g]
    if st is None:
        return S[name]
    s = S[name]["sig"]
    sig = np.where(np.sign(s) == st, s, 0).astype(np.int8)
    out = {"sig": sig, "native": None}
    if S[name].get("epx") is not None:
        out["epx"] = S[name]["epx"]
    return out


def windows(b: dict) -> dict:
    t = b["t"]
    w = {k: E.win_idx(t, k) for k in ("design", "holdout", "long")}
    w["last30"] = (int(np.searchsorted(t, t[-1] + M15 - 30 * DAY)), len(t))
    return w


def be_win(tr: list) -> dict:
    """1배·10배 손익분기 승률(그 변형의 평균 이익·손실로)."""
    r = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    w, lo = r[r > 0], r[r <= 0]
    if len(w) == 0 or len(lo) == 0:
        return {"be1": None, "be10": None}
    mw, ml = float(w.mean()), float(lo.mean())
    be1 = -ml / (mw - ml)
    if 1 + 10 * ml <= 0:
        return {"be1": be1, "be10": 1.0}
    gw, gl = math.log(1 + 10 * mw), math.log(1 + 10 * ml)
    return {"be1": be1, "be10": -gl / (gw - gl)}


def summ(tr: list) -> dict:
    s = E.summarize(tr)
    s.update(be_win(tr))
    return s


# ---------------- 명령 ----------------
def load_all(out: Path, old: bool = True):
    b = load(out)
    t0 = time.time()
    S, gates = build_all(b, old)
    B = E.Bars(b)
    return b, S, gates, B, time.time() - t0


def new_names(S: dict) -> list:
    return [k for k in S if k.startswith("T")] + list(NEW_ENS)


def cmd_grid(out: Path) -> None:
    b, S, gates, B, dt = load_all(out, old=False)
    a, z = windows(b)["design"]
    res: dict = {}
    count = 0
    for name in new_names(S):
        res[name] = {}
        for v in variants(name, S):
            res[name][vname(v)] = summ(simulate(B, S[name], v, a, z))
            count += 1
        print(name, len(res[name]), round(time.time() - dt, 1), flush=True)
    best = {n: max(rr, key=lambda v: E.key(rr[v])) for n, rr in res.items()}
    elig = [n for n in best if res[n][best[n]]["n"] >= 30]
    fin = sorted(elig, key=lambda n: E.key(res[n][best[n]]), reverse=True)[:2]
    sigc = {n: {"long": int((S[n]["sig"][a:z] == 1).sum()),
                "short": int((S[n]["sig"][a:z] == -1).sum())} for n in res}  # fmt: skip
    (out / "grid.json").write_text(json.dumps(
        {"variants_evaluated": count, "build_s": dt, "best": best, "finalists": fin,
         "signals_design": sigc, "res": res}, indent=1))  # fmt: skip
    print("variants", count, "finalists", fin, [best[f] for f in fin])


def cmd_holdout(out: Path) -> None:
    b, S, gates, B, _ = load_all(out, old=False)
    g = json.loads((out / "grid.json").read_text())
    W = windows(b)
    res: dict = {}
    for w in ("holdout", "last30"):
        a, z = W[w]
        res[w] = {n: summ(simulate(B, S[n], parse_v(v), a, z)) for n, v in g["best"].items()}
        res[w]["_bh"] = {"x1": float(b["c"][z - 1] / b["o"][a] - 1)}
    a, z = W["long"]
    res["long"] = {}
    for n in g["finalists"]:
        tr = simulate(B, S[n], parse_v(g["best"][n]), a, z)
        res["long"][n] = summ(tr)
        yr = {}
        for y in range(2020, 2026):
            ya = int(np.searchsorted(b["t"], E.ms(f"{y}-01-01")))
            yz = int(np.searchsorted(b["t"], E.ms(f"{y + 1}-01-01")))
            yr[str(y)] = summ([t for t in tr if ya <= t["e"] < yz])
        res["long"][n + ":years"] = yr
    res["long"]["_bh"] = {"x1": float(b["c"][z - 1] / b["o"][a] - 1)}
    (out / "holdout.json").write_text(json.dumps(res, indent=1))
    for n in g["best"]:
        s = res["holdout"][n]
        print(n, g["best"][n], s["n"], s["win"], round(s["x10"]["ret"], 3), s["x10"]["liq"])


def s70_sources(S: dict) -> list:
    return new_names(S) + ["o" + k for k in OLD_SETS + OLD_ENS]


def cmd_s70(out: Path) -> None:
    b, S, gates, B, dt = load_all(out, old=True)
    a, z = windows(b)["design"]
    rows = []
    t0 = time.time()
    srcs = s70_sources(S)
    for name in srcs:
        for g in GATES:
            Sg = gated(S, name, gates, g)
            for k in K_TP:
                for m in K_SL:
                    tr = simulate(B, Sg, ("K", k, m), a, z)
                    s = summ(tr)
                    s.pop("why", None)
                    rows.append({"src": name, "gate": g, "v": vname(("K", k, m)), **s})
        print(name, len(rows), round(time.time() - t0, 1), flush=True)
    ok = [r for r in rows if r["n"] >= 30 and (r["win"] or 0) >= 0.70
          and r["x10"]["liq"] == 0 and r["x10"]["ret"] > 0]  # fmt: skip
    ok.sort(key=lambda r: (r["x10"]["ret"], r["x3"]["ret"], r["x1"]["ret"]), reverse=True)
    fin = [f"{r['src']}|{r['gate']}|{r['v']}" for r in ok[:10]]
    (out / "s70.json").write_text(json.dumps(
        {"variants_evaluated": len(rows), "n_sources": len(srcs), "eligible": len(ok),
         "finalists": fin, "rows": rows}))  # fmt: skip
    print("s70 variants", len(rows), "eligible", len(ok), fin)


def run_item(B, S, gates, item: str, a: int, z: int) -> list:
    """item = 이름:변형 또는 이름|문|K모양."""
    if "|" in item:
        name, g, v = item.split("|")
        return simulate(B, gated(S, name, gates, g), parse_v(v), a, z)
    name, v = item.split(":", 1)
    return simulate(B, S[name], parse_v(v), a, z)


def cmd_s70hold(out: Path) -> None:
    b, S, gates, B, _ = load_all(out, old=True)
    g = json.loads((out / "s70.json").read_text())
    W = windows(b)
    res = {}
    for it in g["finalists"]:
        res[it] = {
            w: summ(run_item(B, S, gates, it, *W[w])) for w in ("design", "holdout", "last30")
        }
        h = res[it]["holdout"]
        print(it, h["n"], h["win"], round(h["x10"]["ret"], 3), h["x10"]["liq"], flush=True)
    (out / "s70hold.json").write_text(json.dumps(res, indent=1))


def random_sigs(rng, tr, a, z, t, matched: bool):
    nl = sum(x["d"] == 1 for x in tr)
    ns = sum(x["d"] == -1 for x in tr)
    sig = np.zeros(len(t), np.int8)
    if not matched:
        pick = rng.choice(np.arange(a, z - 1), nl + ns, replace=False)
    else:
        slots = (t[[x["j"] for x in tr]] % DAY) // M15
        cand = np.arange(a, z - 1)
        cs = (t[cand] % DAY) // M15
        pick = []
        for s_, cnt in zip(*np.unique(slots, return_counts=True)):
            pool = cand[cs == s_]
            pick += rng.choice(pool, min(cnt, len(pool)), replace=False).tolist()
        pick = np.array(pick)
        rng.shuffle(pick)
    sig[pick[:nl]] = 1
    sig[pick[nl:]] = -1
    return sig


def cmd_random(out: Path, items: list, nsim: int, wins: list, fname: str) -> None:
    b, S, gates, B, _ = load_all(out, old=True)
    W = windows(b)
    rng = np.random.default_rng(20261108)
    res: dict = {}
    for it in items:
        if "|" in it:
            name, g, vs = it.split("|")
            base = gated(S, name, gates, g)
        else:
            name, vs = it.split(":", 1)
            base = S[name]
        v = parse_v(vs)  # 자체 청산(N)도 같은 훅으로, 신호만 무작위
        res[it] = {}
        for w in wins:
            a, z = W[w]
            tr = simulate(B, base, v, a, z)
            real = summ(tr)
            for matched in (False, True):
                r10, r1 = [], []
                t0 = time.time()
                for _ in range(nsim):
                    sg = random_sigs(rng, tr, a, z, b["t"], matched)
                    rs = E.summarize(simulate(B, base, v, a, z, sig=sg, epx=None))
                    r10.append(rs["x10"]["ret"])
                    r1.append(rs["x1"]["ret"])
                r10a, r1a = np.array(r10), np.array(r1)
                z1 = (real["x1"]["ret"] - r1a.mean()) / (r1a.std() + 1e-12)
                z10 = (real["x10"]["ret"] - r10a.mean()) / (r10a.std() + 1e-12)
                pn1 = 0.5 * math.erfc(z1 / math.sqrt(2))
                key = f"{w}:{'matched' if matched else 'plain'}"
                res[it][key] = {
                    "n_real": real["n"], "win": real["win"], "real10": real["x10"]["ret"],
                    "real1": real["x1"]["ret"], "rand10_median": float(np.median(r10a)),
                    "rand1_mean": float(r1a.mean()), "rand1_sd": float(r1a.std()),
                    "p10": float((r10a >= real["x10"]["ret"]).mean()),
                    "p1": float((r1a >= real["x1"]["ret"]).mean()), "z1": float(z1),
                    "z10": float(z10), "p1_normal": pn1,
                    "p1_normal_bonf": min(1.0, pn1 * N_TOTAL), "sec": time.time() - t0,
                }  # fmt: skip
                print(it, key, res[it][key], flush=True)
    (out / fname).write_text(json.dumps(res, indent=1))


# ---------------- H-6D-T1: 현물 2017-09 ~ 2020-07 ----------------
SPOT5 = Path("~/.minerva/data/btc/5m.csv").expanduser()
SPOT_END = E.ms("2020-08-01")


def spot15() -> dict:
    rows = []
    with SPOT5.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            t = int(x[0])
            if t >= SPOT_END:
                break
            rows.append((t, float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])))
    v = np.array(rows)
    ts = v[:, 0].astype(np.int64)
    g = ts - ts % M15
    keys, idx = np.unique(g, return_index=True)
    ends = np.r_[idx[1:], len(ts)]
    b = {
        "t": keys,
        "o": v[idx, 1],
        "h": np.maximum.reduceat(v[:, 2], idx),
        "l": np.minimum.reduceat(v[:, 3], idx),
        "c": v[ends - 1, 4],
        "v": np.add.reduceat(v[:, 5], idx),
        "ft": np.array([SPOT_END + 10 * DAY], np.int64),
        "fr": np.array([0.0]),
    }
    gaps = np.diff(keys) != M15
    b["_gaps"] = int(gaps.sum())
    b["_gap_bars"] = int(((np.diff(keys)[gaps] // M15) - 1).sum())
    return b


def cmd_hyp6d(out: Path, nsim: int) -> None:
    b = spot15()
    n = len(b["c"])
    d = P.daily(b)
    sig = P.daily_to_15(d, P.s6_trend_dev(d["h"], d["l"], d["c"]), n)
    S6 = {"sig": sig, "native": None}
    B = E.Bars(b)
    a, z = 0, n
    tr = simulate(B, S6, ("T1", "S0"), a, z)
    real = summ(tr)
    rng = np.random.default_rng(20261108)
    res = {"bars15": n, "gaps": b["_gaps"], "gap_bars": b["_gap_bars"],
           "first": int(b["t"][0]), "last": int(b["t"][-1]),
           "daily_bars": int(len(d["c"])), "bh_x1": float(b["c"][-1] / b["o"][0] - 1),
           "first_trade": int(b["t"][tr[0]["e"]]) if tr else None,
           "real": real}  # fmt: skip
    for matched in (False, True):
        r10, r1 = [], []
        for _ in range(nsim):
            sg = random_sigs(rng, tr, max(tr[0]["j"] - 1, 0), z, b["t"], matched)
            rs = E.summarize(simulate(B, S6, ("T1", "S0"), a, z, sig=sg))
            r10.append(rs["x10"]["ret"])
            r1.append(rs["x1"]["ret"])
        r10a, r1a = np.array(r10), np.array(r1)
        res["matched" if matched else "plain"] = {
            "p10": float((r10a >= real["x10"]["ret"]).mean()),
            "p1": float((r1a >= real["x1"]["ret"]).mean()),
            "rand10_median": float(np.median(r10a)), "rand1_mean": float(r1a.mean()),
            "rand_liq_share": float((r10a <= -0.999).mean()),
        }  # fmt: skip
    yr = {}
    for y in range(2017, 2021):
        ya = int(np.searchsorted(b["t"], E.ms(f"{y}-01-01")))
        yz = int(np.searchsorted(b["t"], E.ms(f"{y + 1}-01-01")))
        yr[str(y)] = summ([t for t in tr if ya <= t["e"] < yz])
    res["years"] = yr
    (out / "hyp6d.json").write_text(json.dumps(res, indent=1))
    (out / "hyp6d_trades.json").write_text(json.dumps(tr))
    print(json.dumps({k: v for k, v in res.items() if k != "years"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", choices=["grid", "holdout", "s70", "s70hold", "random", "hyp6d"])
    ap.add_argument("--items", nargs="*", default=None)
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--wins", nargs="*", default=["design", "holdout"])
    ap.add_argument("--fname", default="random.json")
    a_ = ap.parse_args()
    o = Path(a_.out).expanduser()
    if a_.cmd == "grid":
        cmd_grid(o)
    elif a_.cmd == "holdout":
        cmd_holdout(o)
    elif a_.cmd == "s70":
        cmd_s70(o)
    elif a_.cmd == "s70hold":
        cmd_s70hold(o)
    elif a_.cmd == "random":
        cmd_random(o, a_.items, a_.nsim, a_.wins, a_.fname)
    else:
        cmd_hyp6d(o, a_.nsim)
