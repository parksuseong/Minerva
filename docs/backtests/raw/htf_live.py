"""상위 프레임(4시간·일봉·주봉) 신호 + 15분봉 체결 실전 후보. 등록: 2026-11-08-btc-htf-live-hypotheses.md

실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/htf-live-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/htf_live.py --out $OUT grid
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pine_eval as E
import pine_ports as P
import btc_fractal as BF
from btc_fractal import scr3
from dmi15_eval import dmi, frame, xover
from r5_data import load

R5 = Path("~/.minerva/research/round5-2026-11-08").expanduser()
SPOT = Path("~/.minerva/data/btc/15m.csv").expanduser()
M15, DAY = 900_000, 86_400_000
FEE = 0.0005
LEV = 5
WIN = {
    "fresh": ("2020-03-01", "2021-11-01"),
    "design": ("2022-01-01", "2024-07-01"),
    "val": ("2024-07-01", "2025-11-01"),
    "seen": ("2025-11-01", None),
}
GATE_WINS = ("fresh", "design", "val", "seen")
SIGS = ("ENA1d", "ENA4h", "TRM1d", "TRM4h", "SCR1d", "SCR4h", "SCR1w", "T6D", "G3D", "PRM1d", "DON1d")
PRICE_ONLY = tuple(s for s in SIGS if s != "PRM1d")
GATES = ("G0", "G1")
EXITS = ("XA", "XB", "XC", "XD", "XE")
PORTS = {"PF1": ("G0", "XC"), "PF2": ("G1", "XD")}
N_TOTAL = len(SIGS) * len(GATES) * len(EXITS) + len(PORTS)  # 112
N_TONIGHT = 5563 + N_TOTAL


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


# ---------------- 자료 ----------------
def load_spot() -> dict:
    t, o, h, lo, c = [], [], [], [], []
    with SPOT.open() as f:
        for row in csv.DictReader(f):
            ot = int(row["open_time"])
            if ot >= ms("2020-03-01"):
                break
            t.append(ot)
            o.append(float(row["open"]))
            h.append(float(row["high"]))
            lo.append(float(row["low"]))
            c.append(float(row["close"]))
    return {"t": np.array(t), "o": np.array(o), "h": np.array(h), "l": np.array(lo), "c": np.array(c),
            "ft": np.zeros(0, np.int64), "fr": np.zeros(0)}  # fmt: skip


def weekly(D: dict, t: np.ndarray) -> dict:
    """완결 일봉 → 주봉(월요일 00:00 UTC 시작, 7일 모두 있는 주만)."""
    day = (t[D["last15"]] // DAY).astype(np.int64)
    wk = (day + 3) // 7
    keys, idx = np.unique(wk, return_index=True)
    ends = np.r_[idx[1:], len(wk)]
    full = (ends - idx) == 7
    idx, ends = idx[full], ends[full]
    return {"o": D["o"][idx], "h": np.array([D["h"][a:z].max() for a, z in zip(idx, ends)]),
            "l": np.array([D["l"][a:z].min() for a, z in zip(idx, ends)]),
            "c": D["c"][ends - 1], "last15": D["last15"][ends - 1]}  # fmt: skip


def direction(h, lo, c) -> np.ndarray:
    cc, r = BF.cci(h, lo, c, 20), BF.rsi(c, 14)
    with np.errstate(invalid="ignore"):
        return np.where((cc > 0) & (r > 50), 1, np.where((cc < 0) & (r < 50), -1, 0)).astype(np.int8)


def to15(F: dict, v: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    """상위 봉 값(마감 시점에 확정)을 15분봉 축으로 앞으로 채운다."""
    out = np.full(n, fill, dtype=float)
    out[F["last15"]] = v
    idx = np.where(~np.isnan(out), np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    res = out[idx]
    res[: F["last15"][0]] = fill
    return res


# ---------------- 신호 ----------------
def frame_sigs(F: dict, kind: str) -> np.ndarray:
    h, lo, c = F["h"], F["l"], F["c"]
    if kind in ("ENA", "TRM"):
        pdi, mdi, adx = dmi(h, lo, c)
        if kind == "ENA":
            long_, short = xover(adx, mdi), xover(adx, pdi)
        else:
            long_, short = xover(pdi, mdi), xover(mdi, pdi)
    elif kind == "SCR":
        long_, short, *_ = scr3({"h": h, "l": lo, "c": c})
    elif kind == "T6D":
        return P.s6_trend_dev(h, lo, c).astype(np.int8)
    elif kind == "G3D":
        return P.s3_gauss_core(h, lo, c)["sig"].astype(np.int8)
    elif kind == "DON":
        hh = np.r_[np.nan, P.highest(h, 20)[:-1]]
        ll = np.r_[np.nan, P.lowest(lo, 20)[:-1]]
        with np.errstate(invalid="ignore"):
            long_, short = c > hh, c < ll
        long_ = long_ & ~np.r_[False, long_[:-1]]
        short = short & ~np.r_[False, short[:-1]]
    else:
        raise ValueError(kind)
    return np.where(long_ & ~short, 1, np.where(short & ~long_, -1, 0)).astype(np.int8)


def prem_daily(b: dict, D: dict) -> np.ndarray:
    """그날 프리미엄(1시간 평균) 평균의 직전 60일 대비 z. ±2 이상이면 되돌림 신호."""
    p = b["f_prem1"]
    starts = np.r_[0, D["last15"][:-1] + 1]
    m = np.array([np.nanmean(p[a : z + 1]) if np.isfinite(p[a : z + 1]).any() else np.nan
                  for a, z in zip(starts, D["last15"])])  # fmt: skip
    z = np.full(len(m), np.nan)
    for i in range(60, len(m)):
        w = m[i - 60 : i]
        w = w[np.isfinite(w)]
        if len(w) >= 40 and np.isfinite(m[i]):
            z[i] = (m[i] - w.mean()) / (w.std() + 1e-12)
    with np.errstate(invalid="ignore"):
        return np.where(z <= -2, 1, np.where(z >= 2, -1, 0)).astype(np.int8)


def build(b: dict, futures: bool = True) -> dict:
    n = len(b["c"])
    F4 = frame(b, 240)
    D = frame(b, 1440)
    W = weekly(D, b["t"])
    frames = {"4h": F4, "1d": D, "1w": W}
    sig_f = {"ENA1d": ("ENA", "1d"), "ENA4h": ("ENA", "4h"), "TRM1d": ("TRM", "1d"),
             "TRM4h": ("TRM", "4h"), "SCR1d": ("SCR", "1d"), "SCR4h": ("SCR", "4h"),
             "SCR1w": ("SCR", "1w"), "T6D": ("T6D", "1d"), "G3D": ("G3D", "1d"),
             "DON1d": ("DON", "1d"), "PRM1d": ("PRM", "1d")}  # fmt: skip
    atr15 = {k: to15(F, P.atr(F["h"], F["l"], F["c"], 14), n) for k, F in frames.items()}
    sma200 = to15(D, P.sma(D["c"], 200), n)
    d15 = direction(b["h"], b["l"], b["c"])
    d4 = to15(F4, direction(F4["h"], F4["l"], F4["c"]).astype(float), n, 0)
    d1 = to15(D, direction(D["h"], D["l"], D["c"]).astype(float), n, 0)
    agree = (d15 != 0) & (d15 == d4) & (d4 == d1)
    sigs, fr_of = {}, {}
    for name, (kind, tf) in sig_f.items():
        if kind == "PRM" and not futures:
            continue
        F = frames[tf]
        v = prem_daily(b, D) if kind == "PRM" else frame_sigs(F, kind)
        a = np.zeros(n, np.int8)
        a[F["last15"]] = v
        a[-1] = 0  # 다음 봉이 없으면 진입 불가
        sigs[name] = a
        fr_of[name] = tf
    return {"sigs": sigs, "tf": fr_of, "atr": atr15, "sma200": sma200, "agree": agree,
            "frames": frames}  # fmt: skip


# ---------------- 모의 매매 ----------------
def gate_sig(S: dict, b: dict, name: str, gate: str) -> np.ndarray:
    s = S["sigs"][name].copy()
    if gate == "G1":
        ma, c = S["sma200"], b["c"]
        with np.errstate(invalid="ignore"):
            s[(s == 1) & ~(c > ma)] = 0
            s[(s == -1) & ~(c < ma)] = 0
    return s


def simulate(b: dict, S: dict, sig: np.ndarray, atr: np.ndarray, ex: str, a: int, z: int,
             delay: int = 0, fee: float = FEE) -> tuple[list, int]:  # fmt: skip
    """sig[j] != 0: j 봉 마감에 신호 → j+1+delay 봉 시가 시장가. 결과: 거래 목록, 추격 한도로 건너뛴 수."""
    o, h, lo, c, t = b["o"], b["h"], b["l"], b["c"], b["t"]
    n = len(c)
    ft, frr = b["ft"], b["fr"]
    cumf = np.r_[0.0, np.cumsum(frr)]
    reverse = ex in ("XD", "XE")
    sl_pct = 0.03 if ex == "XE" else 0.02
    tlim = {"XA": 960, "XB": 960, "XC": 2880}.get(ex, 10**9)
    idx = np.flatnonzero(sig[a:z]) + a
    trades, skipped, busy_until = [], 0, -1
    k = 0
    while k < len(idx):
        j = idx[k]
        k += 1
        e = j + 1 + delay
        if j < busy_until or e >= n:
            continue
        d = int(sig[j])
        lim = 0.03 if S["agree"][j] else 0.01
        ep = o[e]
        if abs(ep / c[j] - 1) > lim:
            skipped += 1
            continue
        sl = ep * (1 - d * sl_pct)
        at = atr[j] if np.isfinite(atr[j]) else ep * 0.02
        end = min(n - 1, e + tlim, z + 2880)  # 구간 끝 뒤로 최대 30일까지 들고 있는다
        hs, ls_ = h[e : end + 1], lo[e : end + 1]
        if d == 1:
            hit_sl = ls_ <= sl
        else:
            hit_sl = hs >= sl
        if ex == "XC":
            cs = c[e : end + 1]
            best = np.maximum.accumulate(cs) if d == 1 else np.minimum.accumulate(cs)
            trail = np.r_[sl, best[:-1] - d * 2 * at]
            lvl = np.maximum(trail, sl) if d == 1 else np.minimum(trail, sl)
            hit_sl = (ls_ <= lvl) if d == 1 else (hs >= lvl)
        else:
            lvl = np.full(len(hs), sl)
        if ex in ("XA", "XB"):
            tp = ep + d * (1.0 if ex == "XA" else 2.0) * at
            hit_tp = (hs >= tp) if d == 1 else (ls_ <= tp)
        else:
            hit_tp = np.zeros(len(hs), bool)
        if reverse:
            opp = np.flatnonzero(sig[j + 1 : end + 1] == -d)
            opp_i = j + 1 + opp[0] if len(opp) else None
        else:
            opp_i = None
        i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else None
        i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else None
        cands = []
        if i_sl is not None:
            px = lvl[i_sl]
            gap = (o[e + i_sl] < px) if d == 1 else (o[e + i_sl] > px)
            cands.append((e + i_sl, 0, o[e + i_sl] if gap and i_sl > 0 else px, "stop"))
        if i_tp is not None:
            cands.append((e + i_tp, 1, tp, "tp"))
        if opp_i is not None and opp_i + 1 + delay < n:
            cands.append((opp_i + 1 + delay, 2, o[opp_i + 1 + delay], "opp"))
        if not cands:
            cands.append((end, 3, c[end], "time"))
        xi, _, xp, why = min(cands, key=lambda q: (q[0], q[1]))
        worst_px = lo[e : xi + 1].min() if d == 1 else h[e : xi + 1].max()
        worst = d * (worst_px / ep - 1) - 2 * fee
        g = d * (xp / ep - 1) - 2 * fee
        tx = t[xi] + (0 if why in ("opp",) else M15)
        f0, f1 = np.searchsorted(ft, t[e], "right"), np.searchsorted(ft, tx, "right")
        fund = -d * (cumf[f1] - cumf[f0])
        trades.append({"d": d, "j": int(j), "e": int(e), "x": int(xi), "ep": float(ep), "xp": float(xp),
                       "r": float(g + fund), "worst": float(min(worst, g + fund)), "why": why,
                       "fund": float(fund)})  # fmt: skip
        busy_until = opp_i if why == "opp" else xi  # 반대 신호 봉은 다음 반복에서 반전 진입
    return trades, skipped


def stats(tr: list, skipped: int = 0) -> dict:
    out = {"n": len(tr), "skipped": skipped}
    if not tr:
        out.update(win=None, x1={"ret": 0.0, "mdd": 0.0, "liq": 0}, x5={"ret": 0.0, "mdd": 0.0, "liq": 0},
                   worst=None, nl=0, ns=0)  # fmt: skip
        return out
    r = np.array([x["r"] for x in tr])
    out.update(win=float((r > 0).mean()), avg=float(r.mean()), worst=float(min(x["worst"] for x in tr)),
               nl=sum(x["d"] == 1 for x in tr), ns=sum(x["d"] == -1 for x in tr),
               x1={k: float(v) for k, v in E.lev_path(tr, 1.0).items()},
               x5={k: float(v) for k, v in E.lev_path(tr, float(LEV)).items()})  # fmt: skip
    return out


def win_idx(t: np.ndarray, a_: str, z_: str | None) -> tuple[int, int]:
    return int(np.searchsorted(t, ms(a_))), len(t) if z_ is None else int(np.searchsorted(t, ms(z_)))


def split(tr: list, t: np.ndarray, a: int, z: int) -> list:
    return [x for x in tr if a <= x["j"] < z]


def run_variant(b, S, name, gate, ex, a, z, **kw):
    sig = gate_sig(S, b, name, gate)
    return simulate(b, S, sig, S["atr"][S["tf"][name]], ex, a, z, **kw)


def port_stats(trs: list[list], skipped: int) -> dict:
    """소매별 5배 자산을 같은 증거금으로 평균. 낙폭은 청산 시점 기준(보유 중 평가손 미반영)."""
    if not any(trs):
        return stats([], skipped)
    ev = sorted(((x["x"], si, x) for si, tr in enumerate(trs) for x in tr), key=lambda q: (q[0], q[1]))
    eq = np.ones(len(trs))
    dead = np.zeros(len(trs), bool)
    peak, mdd, liq = 1.0, 0.0, 0
    for _, si, x in ev:
        if dead[si]:
            continue
        if LEV * x["worst"] <= E.LIQ:
            liq += 1
            dead[si] = True
            eq[si] = 0.0
        else:
            eq[si] *= 1 + LEV * x["r"]
        v = eq.mean()
        peak = max(peak, v)
        mdd = max(mdd, 1 - v / peak)
    allt = [x for tr in trs for x in tr]
    r = np.array([x["r"] for x in allt])
    eq1 = np.mean([np.prod([1 + x["r"] for x in tr]) if tr else 1.0 for tr in trs])
    return {"n": len(allt), "skipped": skipped, "win": float((r > 0).mean()), "avg": float(r.mean()),
            "worst": float(min(x["worst"] for x in allt)), "nl": int(sum(x["d"] == 1 for x in allt)),
            "ns": int(sum(x["d"] == -1 for x in allt)), "x1": {"ret": float(eq1 - 1), "mdd": None, "liq": 0},
            "x5": {"ret": float(eq.mean() - 1), "mdd": float(mdd), "liq": liq}}  # fmt: skip


def passes(r: dict) -> dict:
    out = {}
    for w in GATE_WINS:
        s = r[w]
        out[f"{w}_x5pos"] = s["x5"]["ret"] > 0
        out[f"{w}_liq0"] = s["x5"]["liq"] == 0
        out[f"{w}_mdd40"] = (s["x5"]["mdd"] or 0) <= 0.40
        out[f"{w}_n25"] = s["n"] >= 25
    return out


def all_windows(t: np.ndarray) -> dict:
    W = {k: win_idx(t, *v) for k, v in WIN.items()}
    end = int(t[-1]) + M15
    W["m6"] = (int(np.searchsorted(t, end - 183 * DAY)), len(t))
    W["d30"] = (int(np.searchsorted(t, end - 30 * DAY)), len(t))
    W["pool"] = (W["fresh"][0], W["val"][1])
    return W


def cmd_grid(out: Path) -> None:
    t0 = time.time()
    b = load(R5)
    S = build(b)
    W = all_windows(b["t"])
    bs = load_spot()
    SS = build(bs, futures=False)
    Wsp = (int(np.searchsorted(bs["t"], ms("2017-10-01"))), len(bs["t"]))
    print("built", round(time.time() - t0, 1), {k: int((v != 0).sum()) for k, v in S["sigs"].items()}, flush=True)
    res, full_tr = {}, {}
    a0 = W["fresh"][0]
    for name in SIGS:
        for g in GATES:
            for ex in EXITS:
                key = f"{name}|{g}|{ex}"
                tr, sk = run_variant(b, S, name, g, ex, a0, len(b["t"]))
                full_tr[key] = tr
                r = {}
                for w, (a, z) in W.items():
                    sub = split(tr, b["t"], a, z)
                    r[w] = stats(sub)
                r["skipped_total"] = sk
                if name in PRICE_ONLY:
                    trs, sks = run_variant(bs, SS, name, g, ex, *Wsp)
                    r["spot1720"] = stats(trs, sks)
                r["pass"] = passes(r)
                r["pass_all"] = all(r["pass"].values())
                res[key] = r
        print(name, round(time.time() - t0, 1), flush=True)
    for pf, (g, ex) in PORTS.items():
        r = {}
        for w, (a, z) in W.items():
            r[w] = port_stats([split(full_tr[f"{s}|{g}|{ex}"], b["t"], a, z) for s in SIGS], 0)
        r["pass"] = passes(r)
        r["pass_all"] = all(r["pass"].values())
        res[pf] = r
    cand = [k for k, r in res.items() if r["pass_all"]]

    def score(r):
        return (sum(r["pass"].values()), min(r[w]["x5"]["ret"] for w in GATE_WINS))

    ranked = sorted(res, key=lambda k: score(res[k]), reverse=True)
    (out / "grid.json").write_text(json.dumps(
        {"n_variants": len(res), "n_tonight": N_TONIGHT, "candidates": cand, "ranked": ranked[:20],
         "windows": {k: [int(b["t"][a]), int(b["t"][min(z, len(b["t"])) - 1])] for k, (a, z) in W.items()},
         "sec": time.time() - t0, "res": res}, indent=1, default=float))  # fmt: skip
    print("variants", len(res), "candidates", cand, flush=True)
    for k in ranked[:12]:
        r = res[k]
        print(k, sum(r["pass"].values()), {w: (r[w]["n"], round(r[w]["win"] or 0, 2), round(r[w]["x5"]["ret"], 2),
                                              round(r[w]["x5"]["mdd"] or 0, 2), r[w]["x5"]["liq"])
                                          for w in GATE_WINS + ("m6", "d30")},
              "spot", (lambda s: (s["n"], round(s["x5"]["ret"], 2)) if s else None)(r.get("spot1720")),
              flush=True)  # fmt: skip


def random_sig(rng, b, S, name, sig_real, a, z) -> np.ndarray:
    """같은 롱·숏 수를 그 신호 프레임의 마감 시각들에서 무작위로."""
    F = S["frames"][S["tf"][name]]
    cl = F["last15"][(F["last15"] >= a) & (F["last15"] < z - 1)]
    nl = int((sig_real[a:z] == 1).sum())
    ns = int((sig_real[a:z] == -1).sum())
    pick = rng.choice(cl, min(nl + ns, len(cl)), replace=False)
    s = np.zeros(len(b["c"]), np.int8)
    s[pick[:nl]] = 1
    s[pick[nl:]] = -1
    return s


def cmd_random(out: Path, items: list, nsim: int) -> None:
    b = load(R5)
    S = build(b)
    W = all_windows(b["t"])
    a, z = W["pool"]
    rng = np.random.default_rng(20261108)
    res = {}
    for key in items:
        name, g, ex = key.split("|")
        sig = gate_sig(S, b, name, g)
        atr = S["atr"][S["tf"][name]]
        real = stats(split(simulate(b, S, sig, atr, ex, a, z)[0], b["t"], a, z))
        vals = []
        for _ in range(nsim):
            rs = random_sig(rng, b, S, name, sig, a, z)
            vals.append(stats(split(simulate(b, S, rs, atr, ex, a, z)[0], b["t"], a, z))["x1"]["ret"])
        v = np.array(vals)
        zsc = (real["x1"]["ret"] - v.mean()) / (v.std() + 1e-12)
        pn = 0.5 * math.erfc(zsc / math.sqrt(2))
        res[key] = {"n": real["n"], "real1": real["x1"]["ret"], "rand1_mean": float(v.mean()),
                    "rand1_med": float(np.median(v)), "p_emp": float((v >= real["x1"]["ret"]).mean()),
                    "p_normal": pn, "p_bonf": min(1.0, pn * N_TOTAL), "nsim": nsim}  # fmt: skip
        print(key, res[key], flush=True)
    (out / "random.json").write_text(json.dumps(res, indent=1))


def cmd_check(out: Path, items: list) -> None:
    b = load(R5)
    S = build(b)
    W = all_windows(b["t"])
    res = {}
    for key in items:
        name, g, ex = key.split("|")
        r = {}
        for w in GATE_WINS:
            a, z = W[w]
            r[f"delay1_{w}"] = stats(split(run_variant(b, S, name, g, ex, a, z, delay=1)[0], b["t"], a, z))
            r[f"fee15_{w}"] = stats(split(run_variant(b, S, name, g, ex, a, z, fee=FEE * 1.5)[0], b["t"], a, z))
        a0 = W["fresh"][0]
        tr = run_variant(b, S, name, g, ex, a0, len(b["t"]))[0]
        yrs: dict = {}
        for x in tr:
            y = datetime.fromtimestamp(b["t"][x["e"]] / 1000, timezone.utc).year
            yrs.setdefault(y, []).append(x)
        r["by_year"] = {y: stats(v) for y, v in sorted(yrs.items())}
        worst = min(x["worst"] for x in tr)
        r["worst_trade_1x"] = worst
        r["margin_frac_for_10pct"] = min(1.0, 0.10 / (LEV * -worst)) if worst < 0 else 1.0
        res[key] = r
        print(key, {k: (v["n"], round(v["x5"]["ret"], 2)) for k, v in r.items() if k.startswith(("delay", "fee"))},
              flush=True)  # fmt: skip
        print("  years", {y: (v["n"], round(v["win"] or 0, 2), round(v["x5"]["ret"], 2)) for y, v in r["by_year"].items()},
              "worst", round(worst, 4), "margin", round(r["margin_frac_for_10pct"], 2), flush=True)  # fmt: skip
    (out / "check.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("grid")
    r_ = sp.add_parser("random")
    r_.add_argument("--items", nargs="+", required=True)
    r_.add_argument("--nsim", type=int, default=300)
    c_ = sp.add_parser("check")
    c_.add_argument("--items", nargs="+", required=True)
    args = ap.parse_args()
    o_ = Path(args.out).expanduser()
    o_.mkdir(parents=True, exist_ok=True)
    if args.cmd == "grid":
        cmd_grid(o_)
    elif args.cmd == "random":
        cmd_random(o_, args.items, args.nsim)
    else:
        cmd_check(o_, args.items)
