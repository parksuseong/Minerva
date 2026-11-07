"""비가격 진입 가설 11개 × 청산 29개: 설계 분위수 → 검증 판정, 이미 본 구간, 무작위 기준.

문서: docs/backtests/2026-11-08-btc-nonprice-hypotheses.md (사전 등록 그대로)
모의 매매는 pine_tv_eval.simulate(= pine_eval 묶음 + K 모양 + 자체 청산)를 그대로 쓴다.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/nonprice-2026-11-08
  P="systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
     PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3"
  $P docs/backtests/raw/nonprice_eval.py --out $OUT grid          # 설계·검증·이미 본 구간 전부
  $P docs/backtests/raw/nonprice_eval.py --out $OUT random --nsim 2000   # 판정 후보 + 검증 상위 2
  $P docs/backtests/raw/nonprice_eval.py --out $OUT robust        # 살아남은 것만
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pine_eval as E
import pine_ports as P
import pine_tv_eval as T
from nonprice_data import load

M15 = E.M15
DAY = E.DAY
N_TOTAL = 319
HYPS = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11")
K_EXTRA = ((0.5, 1.0), (0.5, 2.0), (0.5, 3.0), (0.5, 0.0), (1.0, 1.0), (1.0, 2.0),
           (2.0, 1.0), (2.0, 2.0), (3.0, 1.0), (3.0, 2.0))  # fmt: skip
WIN = {
    "design": (E.ms("2022-01-01"), E.ms("2025-01-01")),
    "val": (E.ms("2025-01-01"), E.ms("2025-11-01")),
    "seen": (E.ms("2025-11-01"), None),
}


def windows(b: dict) -> dict:
    t = b["t"]
    w = {}
    for k, (a, z) in WIN.items():
        w[k] = (int(np.searchsorted(t, a)), len(t) if z is None else int(np.searchsorted(t, z)))
    w["last30"] = (int(np.searchsorted(t, t[-1] + M15 - 30 * DAY)), len(t))
    return w


def edge(x: np.ndarray) -> np.ndarray:
    prev = np.r_[False, x[:-1]]
    return x & ~prev


def build_hyps(b: dict, W: dict, thr: dict | None = None) -> tuple:
    """가설 신호와 자체 청산. 문턱은 설계 구간 분위수(thr 주면 그것을 쓴다)."""
    a, z = W["design"]
    f = {k[2:]: b[k] for k in b if k.startswith("f_")}
    f["absr4"] = np.abs(f["r4"])
    qs = {
        "zOI4": (95, 97), "r4": (10, 90), "r1": (0.5, 50, 99.5), "oi1": (5,), "prem8": (2, 50, 98),
        "zP": (0.5, 99.5), "dprem": (0.5, 99.5), "tb1": (1, 99), "zD": (1, 99), "zC": (0.5, 99.5),
        "absr4": (30,), "tb4": (50,),
    }  # fmt: skip
    if thr is None:
        thr = {}
        for k, ps in qs.items():
            x = f[k][a:z]
            x = x[~np.isnan(x)]
            thr[k] = {str(p): float(np.quantile(x, p / 100)) for p in ps}
    Q = lambda k, p: thr[k][str(p)]  # noqa: E731
    with np.errstate(invalid="ignore"):
        g = f
        ge = lambda k, p: np.where(np.isnan(g[k]), False, g[k] >= Q(k, p))  # noqa: E731
        le = lambda k, p: np.where(np.isnan(g[k]), False, g[k] <= Q(k, p))  # noqa: E731
        c = b["c"]
        n = len(c)
        rsi = P.rsi(c, 14)
        atr = P.atr(b["h"], b["l"], c, 14)
        sl_l = c - 3 * atr
        sl_s = c + 3 * atr
        L, S = {}, {}
        L["H1"] = ge("zOI4", 95) & ge("r4", 90)
        S["H1"] = ge("zOI4", 95) & le("r4", 10)
        L["H2"] = le("r1", 0.5) & le("oi1", 5)
        S["H2"] = ge("r1", 99.5) & le("oi1", 5)
        L["H3"] = S["H2"].copy()
        S["H3"] = L["H2"].copy()
        r1 = np.nan_to_num(g["r1"])
        L["H4"] = le("prem8", 2) & (r1 > 0)
        S["H4"] = ge("prem8", 98) & (r1 < 0)
        L["H5"] = le("zP", 0.5)
        S["H5"] = ge("zP", 99.5)
        L["H6"] = ge("dprem", 99.5)
        S["H6"] = le("dprem", 0.5)
        L["H7"] = le("tb1", 1) & ge("r1", 50)
        S["H7"] = ge("tb1", 99) & le("r1", 50)
        L["H8"] = ge("zD", 99)
        S["H8"] = le("zD", 1)
        L["H9"] = le("zC", 0.5)
        S["H9"] = ge("zC", 99.5)
        coil = le("absr4", 30) & ge("zOI4", 97)
        tb4 = g["tb4"]
        L["H10"] = coil & (np.nan_to_num(tb4, nan=Q("tb4", 50)) > Q("tb4", 50))
        S["H10"] = coil & (np.nan_to_num(tb4, nan=Q("tb4", 50)) < Q("tb4", 50))
        # H11 펀딩 음전환: 정산 시각에 시작하는 15분봉 마감에 신호
        ft, fr = b["ft"], b["fr"]
        h11 = np.zeros(n, bool)
        t = b["t"]
        for k in range(9, len(ft)):
            if fr[k] < 0 <= fr[k - 1] and fr[k - 9 : k].mean() > 0:
                i = int(np.searchsorted(t, ft[k] - ft[k] % M15))
                if i < n and t[i] == ft[k] - ft[k] % M15:
                    h11[i] = True
        L["H11"] = h11
        S["H11"] = np.zeros(n, bool)
        pm = Q("prem8", 50)
        p8 = g["prem8"]
        zP, zC = g["zP"], g["zC"]
        nat_spec = {
            "H1": (16, None, None),
            "H2": (32, rsi > 50, rsi < 50),
            "H3": (16, None, None),
            "H4": (96, np.where(np.isnan(p8), False, p8 >= pm), np.where(np.isnan(p8), False, p8 <= pm)),
            "H5": (32, np.where(np.isnan(zP), False, zP >= 0), np.where(np.isnan(zP), False, zP <= 0)),
            "H6": (16, None, None),
            "H7": (16, None, None),
            "H8": (32, None, None),
            "H9": (96, np.where(np.isnan(zC), False, np.abs(zC) < 0.5),
                   np.where(np.isnan(zC), False, np.abs(zC) < 0.5)),
            "H10": (16, None, None),
            "H11": (96, None, None),
        }  # fmt: skip
    Sx = {}
    for h in HYPS:
        le_, se_ = edge(L[h]), edge(S[h])
        sig = np.where(le_ & ~se_, 1, np.where(se_ & ~le_, -1, 0)).astype(np.int8)
        tb, exl, exs = nat_spec[h]
        # 손절가는 신호 방향에 따라 init 훅에서 고른다(native["sl"]은 비워 둔다)
        nat = {"reverse": True, "sl": np.full(n, np.nan), "time_bars": tb}

        def init(p, j, ep, dr, _l=sl_l, _s=sl_s):
            p["sl"] = float(_l[j] if dr == 1 else _s[j])

        nat["init"] = init
        if exl is not None:
            nat["exl"], nat["exs"] = exl, exs
        Sx[h] = {"sig": sig, "native": nat}
    return Sx, thr


def variants() -> list:
    vs = [("N",)] + [(e, s) for e in E.EXITS for s in E.STOPS]
    return vs + [("K", k, m) for k, m in K_EXTRA]


def summ(tr: list) -> dict:
    s = T.summ(tr)
    s.pop("why", None)
    return s


def bars(b: dict) -> E.Bars:
    return E.Bars({k: b[k] for k in ("t", "o", "h", "l", "c", "ft", "fr")})


def passes(d: dict, v: dict) -> dict:
    return {
        "win70": (v["win"] or 0) >= 0.70,
        "x10_pos": v["x10"]["ret"] > 0,
        "liq0": v["x10"]["liq"] == 0,
        "n30": v["n"] >= 30,
        "design1_pos": d["x1"]["ret"] > 0,
    }


def cmd_grid(out: Path) -> None:
    b = load(out)
    W = windows(b)
    Sx, thr = build_hyps(b, W)
    B = bars(b)
    res, cnt, t0 = {}, 0, time.time()
    sigc = {}
    for h in HYPS:
        res[h] = {}
        sig = Sx[h]["sig"]
        sigc[h] = {w: {"long": int((sig[a:z] == 1).sum()), "short": int((sig[a:z] == -1).sum())}
                   for w, (a, z) in W.items()}  # fmt: skip
        for v in variants():
            r = {
                w: summ(T.simulate(B, Sx[h], v, *W[w])) for w in ("design", "val", "seen", "last30")
            }
            r["pass"] = {k: bool(x) for k, x in passes(r["design"], r["val"]).items()}
            r["pass_all5"] = bool(all(r["pass"].values()))
            res[h][T.vname(v)] = r
            cnt += 1
        print(h, sigc[h], round(time.time() - t0, 1), flush=True)
    bh = {w: float(b["c"][z - 1] / b["o"][a] - 1) for w, (a, z) in W.items()}
    cand = [f"{h}:{v}" for h in res for v, r in res[h].items() if r["pass_all5"]]
    pool = [(h, v, r) for h in res for v, r in res[h].items()
            if r["val"]["n"] >= 30 and r["val"]["x10"]["liq"] == 0]  # fmt: skip
    pool.sort(key=lambda x: (x[2]["val"]["x10"]["ret"], x[2]["val"]["x1"]["ret"]), reverse=True)
    top2 = [f"{h}:{v}" for h, v, _ in pool[:2]]
    (out / "grid.json").write_text(json.dumps(
        {"variants_evaluated": cnt, "thresholds": thr, "signals": sigc, "bh": bh,
         "candidates": cand, "top2_val": top2, "res": res}, indent=1))  # fmt: skip
    print("variants", cnt, "candidates", cand, "top2", top2)


def item(Sx, it: str) -> tuple:
    h, vs = it.split(":", 1)
    return Sx[h], T.parse_v(vs)


def cmd_random(out: Path, nsim: int, items: list | None) -> None:
    b = load(out)
    W = windows(b)
    g = json.loads((out / "grid.json").read_text())
    Sx, _ = build_hyps(b, W, g["thresholds"])
    B = bars(b)
    if not items:
        items = list(dict.fromkeys(g["candidates"] + g["top2_val"]))
    rng = np.random.default_rng(20261108)
    res = {}
    a, z = W["val"]
    for it in items:
        base, v = item(Sx, it)
        tr = T.simulate(B, base, v, a, z)
        real = summ(tr)
        r10, r1, wn = [], [], []
        t0 = time.time()
        for _ in range(nsim):
            sg = T.random_sigs(rng, tr, a, z, b["t"], False)
            rs = summ(T.simulate(B, base, v, a, z, sig=sg, epx=None))
            r10.append(rs["x10"]["ret"])
            r1.append(rs["x1"]["ret"])
            wn.append(rs["win"] or 0.0)
        r10a, r1a, wna = np.array(r10), np.array(r1), np.array(wn)
        z1 = (real["x1"]["ret"] - r1a.mean()) / (r1a.std() + 1e-12)
        zw = ((real["win"] or 0) - wna.mean()) / (wna.std() + 1e-12)
        pn1 = 0.5 * math.erfc(z1 / math.sqrt(2))
        pnw = 0.5 * math.erfc(zw / math.sqrt(2))
        res[it] = {
            "n": real["n"], "win": real["win"], "real10": real["x10"]["ret"],
            "real1": real["x1"]["ret"], "rand_win_mean": float(wna.mean()),
            "rand10_median": float(np.median(r10a)), "rand1_mean": float(r1a.mean()),
            "p10": float((r10a >= real["x10"]["ret"]).mean()),
            "p1": float((r1a >= real["x1"]["ret"]).mean()),
            "pwin": float((wna >= (real["win"] or 0)).mean()),
            "z1": float(z1), "zwin": float(zw), "p1_normal": pn1, "pwin_normal": pnw,
            "p1_bonf": min(1.0, pn1 * N_TOTAL), "pwin_bonf": min(1.0, pnw * N_TOTAL),
            "sec": time.time() - t0,
        }  # fmt: skip
        print(it, res[it], flush=True)
    (out / "random.json").write_text(json.dumps(res, indent=1))


def cmd_robust(out: Path, items: list | None) -> None:
    b = load(out)
    W = windows(b)
    g = json.loads((out / "grid.json").read_text())
    Sx, thr = build_hyps(b, W, g["thresholds"])
    B = bars(b)
    items = items or g["candidates"]
    res = {}
    for it in items:
        base, v = item(Sx, it)
        res[it] = {}
        a, z = W["val"]
        sh = dict(base)
        sh["sig"] = np.r_[0, base["sig"][:-1]].astype(np.int8)
        res[it]["delay1"] = summ(T.simulate(B, sh, v, a, z))
        fee0 = T.FEE
        T.FEE = E.FEE * 1.5
        res[it]["fee1.5"] = summ(T.simulate(B, base, v, a, z))
        T.FEE = fee0
        lagf = out / "data15_lag5.npz"
        if lagf.exists():
            bl = load(out, 5)
            Sl, _ = build_hyps(bl, windows(bl), thr)
            res[it]["metrics_lag+5m"] = summ(T.simulate(bars(bl), Sl[it.split(":")[0]], v,
                                                        *windows(bl)["val"]))  # fmt: skip
        print(it, {k: (x["n"], x["win"], round(x["x10"]["ret"], 3)) for k, x in res[it].items()})
    (out / "robust.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", choices=("grid", "random", "robust"))
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--items", nargs="*")
    a_ = ap.parse_args()
    o = Path(a_.out).expanduser()
    if a_.cmd == "grid":
        cmd_grid(o)
    elif a_.cmd == "random":
        cmd_random(o, a_.nsim, a_.items)
    else:
        cmd_robust(o, a_.items)
