"""파인 스크립트 10개 신호 × 청산 묶음 × 손절, 앙상블 6개, 설계/표본 밖 분할, 무작위 기준.

문서: docs/backtests/2026-11-08-pine-10-hypotheses.md (사전 등록 그대로)
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/pine-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 \
    env PYTHONPATH=$R/pyml:docs/backtests/raw \
    python3 docs/backtests/raw/pine_eval.py --out $OUT grid      # 설계 기간 전 변형 + 대표 선택
  ... pine_eval.py --out $OUT holdout   # 대표·결선 표본 밖 1회, 최근 30일, 장기
  ... pine_eval.py --out $OUT random    # 결선 무작위 기준
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pine_ports as P
from pine_data import load

FEE = 0.0005
M15 = 900_000
DAY = 86_400_000
LEVS = (1.0, 3.0, 10.0)
LIQ = -0.95  # L × 최악 평가손익(1배) ≤ −95% 이면 청산


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


WINDOWS = {
    "design": (ms("2025-11-08"), ms("2026-08-08")),
    "holdout": (ms("2026-08-08"), None),
    "long": (ms("2020-08-01"), ms("2025-11-08")),
}
EXITS = ("T1", "T2", "T3", "P", "R", "O")
STOPS = ("S0", "S1", "S2")
F15 = ("1", "2m", "3m", "4", "5", "6m", "7", "8", "10m")
ENSEMBLES = ("E1", "E2", "E3", "E4", "E5", "E6")


# ---------------- 신호 세트 ----------------
def build_sets(b: dict) -> dict:
    n = len(b["c"])
    o, h, lo, c = b["o"], b["h"], b["l"], b["c"]
    d = P.daily(b)
    S: dict = {}
    s1 = P.s1_ob_retest(b)
    S["1"] = {"sig": s1["sig"], "native": s1["native"]}
    S["2m"] = {"sig": P.s2_divergence(o, h, lo, c), "native": None}
    S["2D"] = {
        "sig": P.daily_to_15(d, P.s2_divergence(d["o"], d["h"], d["l"], d["c"]), n),
        "native": None,
    }
    g = P.s3_gauss_core(h, lo, c)
    S["3m"] = {"sig": g["sig"], "native": P.s3_native(g)}
    gd = P.s3_gauss_core(d["h"], d["l"], d["c"])
    S["3D"] = {
        "sig": P.daily_to_15(d, gd["sig"], n),
        "native": {
            "exl": P.daily_to_15(d, gd["close_l"].astype(np.int8), n) == 1,
            "exs": P.daily_to_15(d, gd["close_s"].astype(np.int8), n) == 1,
            "dyn_sl_l": P.daily_level_next(d, gd["lband"], n),
            "dyn_sl_s": P.daily_level_next(d, gd["hband"], n),
            "reverse": False,
        },
    }
    s4 = P.s4_cryptoedge(b)
    S["4"] = {"sig": s4["sig"], "native": s4["native"]}
    s5 = P.s5_quatro(b)
    S["5"] = {"sig": s5["sig"], "native": s5["native"]}
    rev = {"reverse": True}
    s6m = P.s6_trend_dev(h, lo, c)
    S["6m"] = {"sig": s6m, "native": rev}
    S["6D"] = {
        "sig": P.daily_to_15(d, P.s6_trend_dev(d["h"], d["l"], d["c"]), n),
        "native": rev,
    }
    s7 = P.s7_session(b)
    S["7"] = {
        "sig": s7["sig"],
        "epx": s7["epx"],
        "native": dict(s7["native"], sig_all=s7["sig_all"]),
    }
    S["8"] = {"sig": P.s8_extreme(c), "native": None}
    s9 = P.s9_hash(b["ht"], b["hr"], b["t"])
    S["9"] = {"sig": s9["sig"], "native": rev}
    s10m = P.s10_ma(c)
    S["10m"] = {"sig": s10m["sig"], "native": rev}
    s10d = P.s10_ma(d["c"])
    S["10D"] = {"sig": P.daily_to_15(d, s10d["sig"], n), "native": rev}
    # 앙상블
    S.update(build_ensembles(S, b, d, g, gd, s5, s6m, s9, s10m, s10d))
    return S


def _window_any(x: np.ndarray, w: int) -> np.ndarray:
    cs = np.cumsum(np.r_[0, x.astype(int)])
    idx = np.arange(len(x))
    return (cs[idx + 1] - cs[np.maximum(idx + 1 - w, 0)]) > 0


def _rise(x: np.ndarray) -> np.ndarray:
    p = np.r_[False, x[:-1]]
    return x & ~p


def build_ensembles(S, b, d, g, gd, s5, s6m, s9, s10m, s10d) -> dict:
    n = len(b["c"])
    E: dict = {}
    sigs = {k: S[k]["sig"] for k in F15}
    nl = sum(_window_any(s == 1, 4).astype(int) for s in sigs.values())
    ns = sum(_window_any(s == -1, 4).astype(int) for s in sigs.values())
    cl, cs = (nl >= 2) & (ns == 0), (ns >= 2) & (nl == 0)
    E["E1"] = np.where(_rise(cl), 1, np.where(_rise(cs), -1, 0))
    u_l = (S["1"]["sig"] == 1) | (S["4"]["sig"] == 1) | (S["8"]["sig"] == 1)
    u_s = (S["1"]["sig"] == -1) | (S["4"]["sig"] == -1) | (S["8"]["sig"] == -1)
    ul, us = u_l & ~u_s, u_s & ~u_l
    st10 = P.daily_state_15(d, s10d["state"], n)
    E["E2"] = np.where(ul & (st10 == 1), 1, np.where(us & (st10 == -1), -1, 0))
    up3 = P.daily_state_15(d, gd["up"].astype(float), n) == 1
    dn3 = P.daily_state_15(d, gd["dn"].astype(float), n) == 1
    E["E3"] = np.where(ul & up3, 1, np.where(us & dn3, -1, 0))
    hs = s9["state"]
    E["E4"] = np.where(ul & (hs == 1), 1, np.where(us & (hs == -1), -1, 0))
    s8, s2 = S["8"]["sig"], S["2m"]["sig"]
    e5l = ((s8 == 1) & _window_any(s2 == 1, 8)) | ((s2 == 1) & _window_any(s8 == 1, 8))
    e5s = ((s8 == -1) & _window_any(s2 == -1, 8)) | ((s2 == -1) & _window_any(s8 == -1, 8))
    E["E5"] = np.where(e5l & ~e5s, 1, np.where(e5s & ~e5l, -1, 0))
    st = [
        np.nan_to_num(s10m["state"]),
        np.where(g["up"], 1, np.where(g["dn"], -1, 0)),
        s5["state"],
        P.last_dir_state(s6m),
    ]
    bull = sum((x == 1).astype(int) for x in st) >= 3
    bear = sum((x == -1).astype(int) for x in st) >= 3
    E["E6"] = np.where(_rise(bull), 1, np.where(_rise(bear), -1, 0))
    return {k: {"sig": v.astype(np.int8), "native": None} for k, v in E.items()}


# ---------------- 모의 매매 ----------------
class Bars:
    def __init__(self, b: dict):
        self.t = b["t"]
        self.o, self.h, self.l, self.c = (b[k].tolist() for k in ("o", "h", "l", "c"))
        self.atr = P.atr(b["h"], b["l"], b["c"], 14).tolist()
        self.rsi = P.rsi(b["c"], 14).tolist()
        self.ll20 = P.lowest(b["l"], 20).tolist()
        self.hh20 = P.highest(b["h"], 20).tolist()
        self.ft = b["ft"]
        self.fr = b["fr"]
        fi = np.searchsorted(self.t, b["ft"])
        fi = np.minimum(fi, len(self.t) - 1)
        self.f_open = np.asarray(b["o"])[fi]
        self.n = len(self.t)


def funding(B: Bars, d: int, ep: float, t_in: int, t_out: int, t_half: int | None) -> float:
    a = bisect.bisect_right(B.ft, t_in)
    z = bisect.bisect_right(B.ft, t_out)
    tot = 0.0
    for k in range(a, z):
        size = 0.5 if (t_half is not None and B.ft[k] > t_half) else 1.0
        tot += d * B.fr[k] * size * B.f_open[k] / ep
    return tot


def simulate(B: Bars, S: dict, var: tuple, a: int, z: int, sig=None, epx=None) -> list:
    """var = ("N",) 자체 청산 또는 (exit, stop). [a, z) 봉 안의 신호만, z−1 종가에 강제 청산."""
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
    stop_kind = var[1] if len(var) > 1 else "S0"
    o, h, lo, c = B.o, B.h, B.l, B.c
    atr, rsi = B.atr, B.rsi
    exl = native.get("exl") if native else None
    exs = native.get("exs") if native else None
    dyl = native.get("dyn_sl_l") if native else None
    dys = native.get("dyn_sl_s") if native else None
    trades: list = []
    pos = None
    i = a
    pending = None  # (dir, signal index) 다음 봉 시가 진입

    def open_pos(dr: int, j: int, e: int, ep: float) -> dict:
        ref = j - 1 if intrabar else j
        at = atr[ref]
        p = {
            "d": dr,
            "j": j,
            "e": e,
            "ep": ep,
            "size": 1.0,
            "real": 0.0,
            "worst": -FEE,
            "sl": None,
            "tps": [],
            "trail": False,
            "hx": ep,
            "half_t": None,
            "be": None,
            "reason": None,
            "tin": int(B.t[e]) + (1 if intrabar else 0),
        }
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
        fund = funding(B, p["d"], p["ep"], p["tin"], t_out, p["half_t"])
        r -= fund
        # 청산 판정: 펀딩·청산 수수료로 줄어든 증거금도 넣는다(최종 손익이 최악점보다 나쁠 수 있음)
        p["worst"] = min(p["worst"], r)
        trades.append(
            {
                "d": p["d"],
                "j": p["j"],
                "e": p["e"],
                "x": x,
                "ep": p["ep"],
                "xp": px,
                "r": r,
                "worst": p["worst"],
                "fund": fund,
                "why": reason,
            }
        )

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
        # 봉 안: 손절 먼저, 다음 목표가
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
        # 봉 마감
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


# ---------------- 성과 ----------------
def lev_path(tr: list, L: float) -> dict:
    eq, peak, mdd, liq, dead = 1.0, 1.0, 0.0, 0, False
    for t in tr:
        if L * t["worst"] <= LIQ:
            liq += 1
            if not dead:
                mdd = max(mdd, 1.0)
                eq = 0.0
                dead = True
        if dead:
            continue
        trough = eq * (1 + L * t["worst"])
        mdd = max(mdd, 1 - trough / peak)
        eq *= 1 + L * t["r"]
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {"ret": eq - 1, "mdd": mdd, "liq": liq}


def summarize(tr: list) -> dict:
    out: dict = {"n": len(tr)}
    if not tr:
        for L in LEVS:
            out[f"x{int(L)}"] = {"ret": 0.0, "mdd": 0.0, "liq": 0}
        out.update(nl=0, ns=0, win=None, avg=None, med=None, worst_r=None, mae=None)
        return out
    r = np.array([t["r"] for t in tr])
    out.update(
        nl=sum(t["d"] == 1 for t in tr),
        ns=sum(t["d"] == -1 for t in tr),
        win=float((r > 0).mean()),
        avg=float(r.mean()),
        med=float(np.median(r)),
        worst_r=float(r.min()),
        best_r=float(r.max()),
        mae=float(min(t["worst"] for t in tr)),
        fund=float(sum(t["fund"] for t in tr)),
        hold_bars=float(np.mean([t["x"] - t["e"] for t in tr])),
    )
    for L in LEVS:
        out[f"x{int(L)}"] = lev_path(tr, L)
    out["long10"] = lev_path([t for t in tr if t["d"] == 1], 10.0)["ret"]
    out["short10"] = lev_path([t for t in tr if t["d"] == -1], 10.0)["ret"]
    out["long1"] = lev_path([t for t in tr if t["d"] == 1], 1.0)["ret"]
    out["short1"] = lev_path([t for t in tr if t["d"] == -1], 1.0)["ret"]
    why: dict = {}
    for t in tr:
        why[t["why"]] = why.get(t["why"], 0) + 1
    out["why"] = why
    return out


def key(s: dict) -> tuple:
    return (s["x10"]["ret"], s["x3"]["ret"], s["x1"]["ret"])


def variants(name: str, S: dict) -> list:
    vs = []
    if S[name].get("native") is not None:
        vs.append(("N",))
    vs += [(e, s) for e in EXITS for s in STOPS]
    return vs


def win_idx(t: np.ndarray, w: str) -> tuple:
    a, z = WINDOWS[w]
    ia = int(np.searchsorted(t, a))
    iz = len(t) if z is None else int(np.searchsorted(t, z))
    return ia, iz


def vname(v: tuple) -> str:
    return "N" if v[0] == "N" else f"{v[0]}/{v[1]}"


# ---------------- 실행 ----------------
def load_all(out: Path) -> tuple:
    b = load(out)
    t0 = time.time()
    S = build_sets(b)
    B = Bars(b)
    return b, S, B, time.time() - t0


def cmd_grid(out: Path) -> None:
    b, S, B, dt = load_all(out)
    a, z = win_idx(b["t"], "design")
    res: dict = {}
    count = 0
    for name in S:
        res[name] = {}
        for v in variants(name, S):
            tr = simulate(B, S[name], v, a, z)
            res[name][vname(v)] = summarize(tr)
            count += 1
        print(name, len(res[name]), flush=True)
    best = {}
    for name, rr in res.items():
        k = max(rr, key=lambda v: key(rr[v]))
        best[name] = k
    elig = [n for n in best if res[n][best[n]]["n"] >= 30]
    fin = sorted(elig, key=lambda n: key(res[n][best[n]]), reverse=True)[:2]
    sig_counts = {
        n: {
            "long": int((S[n]["sig"][a:z] == 1).sum()),
            "short": int((S[n]["sig"][a:z] == -1).sum()),
        }
        for n in S
    }
    (out / "grid.json").write_text(
        json.dumps(
            {
                "variants_evaluated": count,
                "build_s": dt,
                "design_bars": z - a,
                "best": best,
                "finalists": fin,
                "signals_design": sig_counts,
                "res": res,
            },
            indent=1,
        )
    )
    print("variants", count, "finalists", fin, [best[f] for f in fin])


def parse_v(s: str) -> tuple:
    return ("N",) if s == "N" else tuple(s.split("/"))


def cmd_holdout(out: Path) -> None:
    b, S, B, _ = load_all(out)
    g = json.loads((out / "grid.json").read_text())
    res: dict = {}
    trades: dict = {}
    for w in ("holdout", "last30"):
        if w == "last30":
            z = len(b["t"])
            a = int(np.searchsorted(b["t"], b["t"][-1] + M15 - 30 * DAY))
        else:
            a, z = win_idx(b["t"], w)
        res[w] = {}
        for name, v in g["best"].items():
            tr = simulate(B, S[name], parse_v(v), a, z)
            res[w][name] = summarize(tr)
            if name in g["finalists"] or w == "holdout":
                trades[f"{w}:{name}"] = tr
        res[w]["_bh"] = {
            "x1": float(b["c"][z - 1] / b["o"][a] - 1),
            "from": int(b["t"][a]),
            "to": int(b["t"][z - 1]) + M15,
        }
    a, z = win_idx(b["t"], "long")
    res["long"] = {}
    for name in g["finalists"]:
        tr = simulate(B, S[name], parse_v(g["best"][name]), a, z)
        res["long"][name] = summarize(tr)
        yr: dict = {}
        for y in range(2020, 2026):
            ya = int(np.searchsorted(b["t"], ms(f"{y}-01-01")))
            yz = int(np.searchsorted(b["t"], ms(f"{y + 1}-01-01")))
            sub = [t for t in tr if ya <= t["e"] < yz]
            yr[str(y)] = summarize(sub)
        res["long"][name + ":years"] = yr
    # 설계 거래도 결선만 저장(차트·점검용)
    a, z = win_idx(b["t"], "design")
    for name in g["finalists"]:
        trades[f"design:{name}"] = simulate(B, S[name], parse_v(g["best"][name]), a, z)
    (out / "holdout.json").write_text(json.dumps(res, indent=1))
    (out / "trades.json").write_text(json.dumps(trades))
    for name in g["best"]:
        s = res["holdout"][name]
        print(name, g["best"][name], s["n"], s["win"], s["x10"], s["x1"]["ret"])


def cmd_random(out: Path, nsim: int, items=None, fname: str = "random.json") -> None:
    b, S, B, _ = load_all(out)
    g = json.loads((out / "grid.json").read_text())
    rng = np.random.default_rng(20261108)
    res: dict = {}
    todo = items or [(n, g["best"][n]) for n in g["finalists"]]
    for name, vs in todo:
        v = parse_v(vs)
        if v[0] == "N":
            raise SystemExit("자체 청산 결선은 무작위 기준 정의가 다름: 문서에 따로 적을 것")
        res[name] = {}
        for w in ("design", "holdout"):
            a, z = win_idx(b["t"], w)
            tr = simulate(B, S[name], v, a, z)
            nl = sum(t["d"] == 1 for t in tr)
            ns = sum(t["d"] == -1 for t in tr)
            real = summarize(tr)
            r10, r1, ntr = [], [], []
            t0 = time.time()
            for _ in range(nsim):
                sig = np.zeros(len(b["t"]), np.int8)
                pick = rng.choice(np.arange(a, z - 1), nl + ns, replace=False)
                sig[pick[:nl]] = 1
                sig[pick[nl:]] = -1
                rt = simulate(B, S[name], v, a, z, sig=sig, epx=None)
                ss = summarize(rt)
                r10.append(ss["x10"]["ret"])
                r1.append(ss["x1"]["ret"])
                ntr.append(ss["n"])
            r10a, r1a = np.array(r10), np.array(r1)
            p10 = float((r10a >= real["x10"]["ret"]).mean())
            p1 = float((r1a >= real["x1"]["ret"]).mean())
            z1 = (real["x1"]["ret"] - r1a.mean()) / (r1a.std() + 1e-12)
            res[name][w] = {
                "real10": real["x10"]["ret"],
                "real1": real["x1"]["ret"],
                "n_real": real["n"],
                "n_random_mean": float(np.mean(ntr)),
                "rand10_median": float(np.median(r10a)),
                "rand1_mean": float(r1a.mean()),
                "rand1_sd": float(r1a.std()),
                "p10": p10,
                "p1": p1,
                "z1": float(z1),
                "p1_normal": float(0.5 * math.erfc(z1 / math.sqrt(2))),
                "sec": time.time() - t0,
            }
            print(name, w, res[name][w], flush=True)
    (out / fname).write_text(json.dumps(res, indent=1))


def cmd_check(out: Path, items: list) -> None:
    """설계에서 이미 센 변형(예: 결선의 손절 있는 판)을 모든 기간에 다시 계산(새 선택 아님)."""
    b, S, B, _ = load_all(out)
    res: dict = {}
    for name, vs in items:
        v = parse_v(vs)
        r: dict = {}
        for w in ("design", "holdout", "last30", "long"):
            if w == "last30":
                z = len(b["t"])
                a = int(np.searchsorted(b["t"], b["t"][-1] + M15 - 30 * DAY))
            else:
                a, z = win_idx(b["t"], w)
            r[w] = summarize(simulate(B, S[name], v, a, z))
        res[f"{name}:{vs}"] = r
    (out / "check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", choices=["grid", "holdout", "random", "check"])
    ap.add_argument("--items", nargs="*", default=None, help="이름:변형 (check·random)")
    ap.add_argument("--nsim", type=int, default=2000)
    a = ap.parse_args()
    o = Path(a.out).expanduser()
    if a.cmd == "grid":
        cmd_grid(o)
    elif a.cmd == "holdout":
        cmd_holdout(o)
    elif a.cmd == "check":
        cmd_check(o, [tuple(x.split(":", 1)) for x in a.items])
    else:
        its = [tuple(x.split(":", 1)) for x in a.items] if a.items else None
        cmd_random(o, a.nsim, its, "random_extra.json" if its else "random.json")
