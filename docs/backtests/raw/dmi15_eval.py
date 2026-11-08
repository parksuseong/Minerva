"""사용자 DMI 규칙(PDI·MDI 교차, SCR 선행)을 BTC 15분봉에 옮긴 시험. 판정·무작위 기준·자체 점검.

문서: docs/backtests/2026-11-08-btc-dmi15-hypotheses.md (사전 등록 그대로, 변형 145개)
자료: 5차 선물 15분봉 + 펀딩(`~/.minerva/research/round5-2026-11-08`).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/dmi15-2026-11-08
  P="systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
     PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3"
  $P docs/backtests/raw/dmi15_eval.py --out $OUT grid
  $P docs/backtests/raw/dmi15_eval.py --out $OUT random --nsim 2000
  $P docs/backtests/raw/dmi15_eval.py --out $OUT check
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pine_eval as E
import pine_ports as P
import pine_tv_eval as T
import tvc_components as K
from btc_fractal import scr3
from r5_data import load
from tvc_exits import make_native

R5 = Path("~/.minerva/research/round5-2026-11-08").expanduser()
M15 = E.M15
DAY = E.DAY
WIN = {
    "fresh": (E.ms("2020-03-01"), E.ms("2021-11-01")),
    "design": (E.ms("2022-01-01"), E.ms("2024-07-01")),
    "val": (E.ms("2024-07-01"), E.ms("2025-11-01")),
    "seen": (E.ms("2025-11-01"), None),
}
TFS = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
FAMS = ("TR-A", "TR-L", "TR-M", "EN-A", "EN-L", "ES-A", "ES-L", "ES-M")
EXITS = ("X1", "X2", "X3", "XO", "XR")
GATE_K = 8
N_TOTAL = 145
N_TONIGHT = 5382 + N_TOTAL
LEVS = (1, 3, 5, 10)


# ---------------- 지표 ----------------
def dmi(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> tuple:
    """와일더 DMI. PDI·MDI·ADX."""
    up = np.r_[np.nan, h[1:] - h[:-1]]
    dn = np.r_[np.nan, lo[:-1] - lo[1:]]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = mdm[0] = np.nan
    tr = P.tr(h, lo, c)
    tr[0] = np.nan
    atr = P.rma(tr, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * P.rma(pdm, n) / atr
        mdi = 100 * P.rma(mdm, n) / atr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = P.rma(dx, n)
    return pdi, mdi, adx


def frame(b: dict, minutes: int) -> dict:
    """완결된 상위 봉만(UTC 경계). last15 = 그 봉의 마지막 15분봉 인덱스."""
    if minutes == 15:
        return {k: b[k] for k in ("o", "h", "l", "c")} | {"last15": np.arange(len(b["c"]))}
    step = minutes * 60_000
    g = b["t"] // step
    _, idx = np.unique(g, return_index=True)
    ends = np.r_[idx[1:], len(g)]
    full = (ends - idx) == minutes // 15
    idx, ends = idx[full], ends[full]
    return {"o": b["o"][idx], "h": np.array([b["h"][a:z].max() for a, z in zip(idx, ends)]),
            "l": np.array([b["l"][a:z].min() for a, z in zip(idx, ends)]),
            "c": b["c"][ends - 1], "last15": ends - 1}  # fmt: skip


def xover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pa, pb = np.r_[np.nan, a[:-1]], np.r_[np.nan, b[:-1]]
    with np.errstate(invalid="ignore"):
        return (pa <= pb) & (a > b)


def recent(x: np.ndarray, k: int) -> np.ndarray:
    out = x.copy()
    for j in range(1, k):
        out[j:] |= x[:-j]
    return out


def tf_signals(F: dict) -> dict:
    pdi, mdi, adx = dmi(F["h"], F["l"], F["c"])
    lv = np.full(len(pdi), 20.0)
    part = {"A": (adx, adx), "L": (lv, lv), "M": (mdi, pdi)}  # (PDI의 상대, MDI의 상대)
    on_l, on_s, *_ = scr3({"h": F["h"], "l": F["l"], "c": F["c"]})
    sl = recent(on_s, GATE_K)  # 위 이탈(과매수 이탈)이 8봉 안
    lg = recent(on_l, GATE_K)  # 아래 돌파(과매도 탈출)이 8봉 안
    out = {}
    for p, (pp, mp) in part.items():
        p_up, m_up = xover(pdi, pp), xover(mdi, mp)
        p_dn, m_dn = xover(pp, pdi), xover(mp, mdi)
        if p == "M":
            out["TR-M"] = K.sig_from(p_up, m_up)
            out["ES-M"] = K.sig_from(m_dn & lg, p_dn & sl)
            continue
        out[f"TR-{p}"] = K.sig_from(p_up, m_up)
        out[f"EN-{p}"] = K.sig_from(m_dn, p_dn)
        out[f"ES-{p}"] = K.sig_from(m_dn & lg, p_dn & sl)
    return out, (pdi, mdi, adx)


def build(b: dict) -> tuple:
    n = len(b["c"])
    h, lo, c = b["h"], b["l"], b["c"]
    atr = P.atr(h, lo, c, 14)
    st_l = P.lowest(lo, 8) - 0.25 * atr
    st_s = P.highest(h, 8) + 0.25 * atr
    rsi = P.rsi(c, 14)
    nz = K.nz
    xr_l, xr_s = nz(rsi, 50) >= 70, nz(rsi, 50) <= 30
    sigs, lines = {}, {}
    for tf, m in TFS.items():
        F = frame(b, m)
        s, ln = tf_signals(F)
        lines[tf] = (F["last15"], ln)
        for fam, v in s.items():
            if tf == "1d" and fam.startswith("ES"):
                continue
            a = np.zeros(n, np.int8)
            a[F["last15"]] = v
            sigs[f"{fam}@{tf}"] = a
    sets = {}
    for name, sg in sigs.items():
        for x in EXITS:
            if x in ("X1", "X2", "X3"):
                nat = make_native(x, atr, st_l, st_s)
            elif x == "XO":
                nat = {"reverse": True, "time_bars": 2880,
                       "init": lambda p, j, ep, dr: p.__setitem__("sl", ep * (1 - dr * 0.07))}  # fmt: skip
            else:
                nat = {"reverse": True, "time_bars": 96, "exl": xr_l, "exs": xr_s,
                       "init": lambda p, j, ep, dr: p.__setitem__("sl", ep * (1 - dr * 0.07))}  # fmt: skip
            sets[f"{name}|{x}"] = {"sig": sg, "native": nat}
    assert len(sets) == N_TOTAL, len(sets)
    return sets, sigs, lines


def windows(t: np.ndarray) -> dict:
    w = {}
    for k, (a, z) in WIN.items():
        w[k] = (int(np.searchsorted(t, a)), len(t) if z is None else int(np.searchsorted(t, z)))
    w["last30"] = (int(np.searchsorted(t, t[-1] + M15 - 30 * DAY)), len(t))
    return w


def summ(tr: list) -> dict:
    s = T.summ(tr)
    s.pop("why", None)
    s["x5"] = E.lev_path(tr, 5.0) if tr else {"ret": 0.0, "mdd": 0.0, "liq": 0}
    return s


def run(B, sets, key, a, z, sig=None) -> list:
    if sig is None:
        return T.simulate(B, sets[key], ("N",), a, z)
    return T.simulate(B, sets[key], ("N",), a, z, sig=sig, epx=None)


def passes(r: dict) -> dict:
    d, v, f = r["design"], r["val"], r["fresh"]
    return {
        "win70": (v["win"] or 0) >= 0.70, "x10_pos": v["x10"]["ret"] > 0,
        "liq0_val": v["x10"]["liq"] == 0, "liq0_design": d["x10"]["liq"] == 0,
        "n30": v["n"] >= 30, "design1_pos": d["x1"]["ret"] > 0,
        "fresh_ok": ((f["win"] or 0) >= 0.70 and f["x10"]["ret"] > 0
                     and f["x10"]["liq"] == 0 and f["n"] >= 20),
    }  # fmt: skip


def setup():
    b = load(R5)
    W = windows(b["t"])
    sets, sigs, lines = build(b)
    B = E.Bars({k: b[k] for k in ("t", "o", "h", "l", "c", "ft", "fr")})
    return b, W, sets, sigs, lines, B


def cmd_grid(out: Path) -> None:
    t0 = time.time()
    b, W, sets, sigs, _, B = setup()
    print("built", round(time.time() - t0, 1), flush=True)
    res = {}
    for key in sets:
        r = {w: summ(run(B, sets, key, *W[w])) for w in ("fresh", "design", "val", "seen", "last30")}
        r["pass"] = {k: bool(x) for k, x in passes(r).items()}
        r["pass_all"] = bool(all(r["pass"].values()))
        res[key] = r
    pool = [(k, r) for k, r in res.items() if r["val"]["n"] >= 30
            and r["val"]["x10"]["liq"] == 0 and r["design"]["x10"]["liq"] == 0]  # fmt: skip
    pool.sort(key=lambda x: ((x[1]["val"]["win"] or 0), x[1]["val"]["x10"]["ret"]), reverse=True)
    top_win = [k for k, _ in pool[:2]]
    pool.sort(key=lambda x: (x[1]["val"]["x10"]["ret"], x[1]["val"]["x1"]["ret"]), reverse=True)
    top_ret = [k for k, _ in pool[:2]]
    cand = [k for k, r in res.items() if r["pass_all"]]
    sigc = {s: {w: int((v[a:z] != 0).sum()) for w, (a, z) in W.items()} for s, v in sigs.items()}
    (out / "grid.json").write_text(json.dumps(
        {"variants_evaluated": len(res), "variants_tonight": N_TONIGHT, "candidates": cand,
         "top2_val_win": top_win, "top2_val_x10": top_ret, "signals": sigc,
         "sec": time.time() - t0, "res": res}, indent=1, default=float))  # fmt: skip
    print("variants", len(res), "candidates", cand, "top", top_win, top_ret, flush=True)


def cmd_random(out: Path, nsim: int, items: list | None) -> None:
    b, W, sets, _, _, B = setup()
    g = json.loads((out / "grid.json").read_text())
    if not items:
        items = list(dict.fromkeys(g["candidates"] + g["top2_val_win"] + g["top2_val_x10"]))
    rng = np.random.default_rng(20261108)
    res = {}
    for it in items:
        ow = {}
        for w in ("fresh", "design", "val"):
            a, z = W[w]
            tr = run(B, sets, it, a, z)
            real = summ(tr)
            if real["n"] == 0:
                continue
            r1, wn = [], []
            for _ in range(nsim):
                sg = T.random_sigs(rng, tr, a, z, b["t"], False)
                rs = summ(run(B, sets, it, a, z, sig=sg))
                r1.append(rs["x1"]["ret"])
                wn.append(rs["win"] or 0.0)
            r1a, wna = np.array(r1), np.array(wn)
            z1 = (real["x1"]["ret"] - r1a.mean()) / (r1a.std() + 1e-12)
            pn1 = 0.5 * math.erfc(z1 / math.sqrt(2))
            ow[w] = {"n": real["n"], "win": real["win"], "real1": real["x1"]["ret"],
                     "rand1_mean": float(r1a.mean()), "rand_win_mean": float(wna.mean()),
                     "p1": float((r1a >= real["x1"]["ret"]).mean()),
                     "pwin": float((wna >= (real["win"] or 0)).mean()),
                     "p1_normal": pn1, "p1_bonf": min(1.0, pn1 * N_TOTAL)}  # fmt: skip
        res[it] = ow
        print(it, {w: (v["n"], round(v["win"], 3), round(v["p1"], 4), round(v["p1_bonf"], 3))
                   for w, v in ow.items()}, flush=True)  # fmt: skip
    (out / "random.json").write_text(json.dumps(res, indent=1))


def cmd_check(out: Path, items: list | None) -> None:
    """레드팀 자체 점검: 1봉 지연 진입, 수수료 1.5배, 해별 1배·10배."""
    b, W, sets, _, _, B = setup()
    g = json.loads((out / "grid.json").read_text())
    if not items:
        items = list(dict.fromkeys(g["candidates"] + g["top2_val_win"] + g["top2_val_x10"]))
    res = {}
    for it in items:
        r: dict = {}
        sg = sets[it]["sig"]
        late = np.r_[np.int8(0), sg[:-1]]
        for w in ("design", "val"):
            a, z = W[w]
            r[f"delay1_{w}"] = summ(run(B, sets, it, a, z, sig=late))
            fee0 = T.FEE
            T.FEE = fee0 * 1.5
            try:
                r[f"fee15_{w}"] = summ(run(B, sets, it, a, z))
            finally:
                T.FEE = fee0
        a, z = W["fresh"][0], len(b["t"])
        tr = run(B, sets, it, a, z)
        yrs: dict = {}
        for t in tr:
            y = datetime.fromtimestamp(b["t"][t["e"]] / 1000, timezone.utc).year
            yrs.setdefault(y, []).append(t)
        r["by_year"] = {y: {"n": len(v), "win": float(np.mean([x["r"] > 0 for x in v])),
                            "x1": E.lev_path(v, 1)["ret"], "x10": E.lev_path(v, 10)["ret"]}
                        for y, v in sorted(yrs.items())}  # fmt: skip
        res[it] = r
        print(it, {k: (v["n"], round(v["win"] or 0, 3), round(v["x10"]["ret"], 3))
                   for k, v in r.items() if k != "by_year"}, flush=True)  # fmt: skip
        print("  years", {y: (v["n"], round(v["win"], 2), round(v["x1"], 3), round(v["x10"], 2))
                          for y, v in r["by_year"].items()}, flush=True)  # fmt: skip
    (out / "check.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("grid")
    r = sp.add_parser("random")
    r.add_argument("--nsim", type=int, default=2000)
    r.add_argument("--items", nargs="*")
    c = sp.add_parser("check")
    c.add_argument("--items", nargs="*")
    a = ap.parse_args()
    p = Path(a.out).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    if a.cmd == "grid":
        cmd_grid(p)
    elif a.cmd == "random":
        cmd_random(p, a.nsim, a.items)
    else:
        cmd_check(p, a.items)
