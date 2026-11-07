"""트레이딩뷰 구성요소 조합 + ML: 설계(2022-01~2024-06) → 검증(2024-07~2025-10) 판정, 이미 본 구간 보고.

문서: docs/backtests/2026-11-08-btc-tv-components-hypotheses.md (사전 등록 그대로)
모의 매매는 pine_tv_eval.simulate(오늘 밤 세 시험과 같은 체결·비용·청산 판정)를 쓴다.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/tv-components-2026-11-08
  P="systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
     PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3"
  $P docs/backtests/raw/tvc_eval.py --out $OUT grid
  $P docs/backtests/raw/tvc_eval.py --out $OUT random --nsim 2000
  $P docs/backtests/raw/tvc_eval.py --out $OUT spot
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pine_eval as E
import pine_tv_eval as T
import tvc_components as K
from nonprice_data import load
from tvc_exits import EXIT_SHAPES, make_native

M15 = E.M15
DAY = E.DAY
WIN = {
    "design": (E.ms("2022-01-01"), E.ms("2024-07-01")),
    "val": (E.ms("2024-07-01"), E.ms("2025-11-01")),
    "seen": (E.ms("2025-11-01"), None),
}
N_TOTAL = 216  # 사전 등록 변형 수(21 신호 × 10 청산 + ML 6)
N_TONIGHT = 4955 + N_TOTAL


def windows(t: np.ndarray) -> dict:
    w = {}
    for k, (a, z) in WIN.items():
        w[k] = (int(np.searchsorted(t, a)), len(t) if z is None else int(np.searchsorted(t, z)))
    w["last30"] = (int(np.searchsorted(t, t[-1] + M15 - 30 * DAY)), len(t))
    return w


def bars(b: dict) -> E.Bars:
    return E.Bars({k: b[k] for k in ("t", "o", "h", "l", "c", "ft", "fr")})


def summ(tr: list) -> dict:
    s = T.summ(tr)
    s.pop("why", None)
    return s


def passes(d: dict, v: dict) -> dict:
    return {
        "win70": (v["win"] or 0) >= 0.70,
        "x10_pos": v["x10"]["ret"] > 0,
        "liq0_val": v["x10"]["liq"] == 0,
        "liq0_design": d["x10"]["liq"] == 0,
        "n30": v["n"] >= 30,
        "design1_pos": d["x1"]["ret"] > 0,
    }


def strategies(b: dict, W: dict, thr: dict | None = None) -> tuple:
    """신호 세트(단일 + 조합 + ML) × 청산 모양 → {이름: {"sig", "native"}}"""
    comp, thr = K.build(b, W["design"], thr)
    atr = comp["atr"]
    out = {}
    for sname, s in comp["sets"].items():
        for x in EXIT_SHAPES:
            nat = make_native(x, atr, comp["st_l"], comp["st_s"], s.get("own_l"), s.get("own_s"))
            out[f"{sname}|{x}"] = {"sig": s["sig"], "native": nat}
    return out, comp, thr


def run_all(b: dict, W: dict, S: dict, B) -> dict:
    res = {}
    for k, s in S.items():
        r = {w: summ(T.simulate(B, s, ("N",), *W[w])) for w in ("design", "val", "seen", "last30")}
        r["pass"] = {kk: bool(x) for kk, x in passes(r["design"], r["val"]).items()}
        r["pass_all"] = bool(all(r["pass"].values()))
        res[k] = r
    return res


def cmd_grid(out: Path) -> None:
    b = load(out)
    W = windows(b["t"])
    B = bars(b)
    t0 = time.time()
    S, comp, thr = strategies(b, W)
    print("built", len(S), round(time.time() - t0, 1), flush=True)
    import tvc_ml as ML

    mls, mlinfo = ML.build(b, W, comp)
    for k, s in mls.items():
        S[k] = s
    assert len(S) == N_TOTAL, len(S)
    res = run_all(b, W, S, B)
    sigc = {
        nm: {w: {"long": int((s["sig"][a:z] == 1).sum()), "short": int((s["sig"][a:z] == -1).sum())}
             for w, (a, z) in W.items()}
        for nm, s in list(comp["sets"].items()) + [(k.split("|")[0], v) for k, v in mls.items()]
    }  # fmt: skip
    bh = {w: float(b["c"][z - 1] / b["o"][a] - 1) for w, (a, z) in W.items()}
    cand = [k for k, r in res.items() if r["pass_all"]]
    pool = [(k, r) for k, r in res.items()
            if r["val"]["n"] >= 30 and r["val"]["x10"]["liq"] == 0
            and r["design"]["x10"]["liq"] == 0]  # fmt: skip
    pool.sort(key=lambda x: ((x[1]["val"]["win"] or 0), x[1]["val"]["x10"]["ret"]), reverse=True)
    top2_win = [k for k, _ in pool[:2]]
    pool.sort(key=lambda x: (x[1]["val"]["x10"]["ret"], x[1]["val"]["x1"]["ret"]), reverse=True)
    top2_ret = [k for k, _ in pool[:2]]
    (out / "grid.json").write_text(json.dumps(
        {"variants_evaluated": len(res), "thresholds": thr, "ml": mlinfo, "signals": sigc, "bh": bh,
         "candidates": cand, "top2_val_win": top2_win, "top2_val_x10": top2_ret,
         "sec": time.time() - t0, "res": res}, indent=1))  # fmt: skip
    print("variants", len(res), "candidates", cand, "top2", top2_win, top2_ret)


def load_strats(out: Path):
    b = load(out)
    W = windows(b["t"])
    g = json.loads((out / "grid.json").read_text())
    S, comp, _ = strategies(b, W, g["thresholds"])
    import tvc_ml as ML

    mls, _ = ML.build(b, W, comp)
    S.update(mls)
    return b, W, g, S, comp


def cmd_random(out: Path, nsim: int, items: list | None) -> None:
    b, W, g, S, _ = load_strats(out)
    B = bars(b)
    if not items:
        items = list(dict.fromkeys(g["candidates"] + g["top2_val_win"] + g["top2_val_x10"]))
    rng = np.random.default_rng(20261108)
    res = {}
    for it in items:
        base = S[it]
        out_w = {}
        for w in ("design", "val"):
            a, z = W[w]
            tr = T.simulate(B, base, ("N",), a, z)
            real = summ(tr)
            r10, r1, wn, nn = [], [], [], []
            for _ in range(nsim):
                sg = T.random_sigs(rng, tr, a, z, b["t"], False)
                rs = summ(T.simulate(B, base, ("N",), a, z, sig=sg, epx=None))
                r10.append(rs["x10"]["ret"])
                r1.append(rs["x1"]["ret"])
                wn.append(rs["win"] or 0.0)
                nn.append(rs["n"])
            r10a, r1a, wna = np.array(r10), np.array(r1), np.array(wn)
            z1 = (real["x1"]["ret"] - r1a.mean()) / (r1a.std() + 1e-12)
            zw = ((real["win"] or 0) - wna.mean()) / (wna.std() + 1e-12)
            pn1 = 0.5 * math.erfc(z1 / math.sqrt(2))
            pnw = 0.5 * math.erfc(zw / math.sqrt(2))
            # 승률 이항 근사: 무작위 평균 승률 p0, 실제 n회 중 승 k 이상일 확률
            k = round((real["win"] or 0) * real["n"])
            p0 = float(wna.mean())
            pbin = binom_sf(k, real["n"], p0)
            out_w[w] = {
                "n": real["n"], "win": real["win"], "real10": real["x10"]["ret"],
                "real1": real["x1"]["ret"], "rand_n_mean": float(np.mean(nn)),
                "rand_win_mean": p0, "rand10_median": float(np.median(r10a)),
                "rand1_mean": float(r1a.mean()),
                "p10": float((r10a >= real["x10"]["ret"]).mean()),
                "p1": float((r1a >= real["x1"]["ret"]).mean()),
                "pwin": float((wna >= (real["win"] or 0)).mean()),
                "z1": float(z1), "zwin": float(zw), "p1_normal": pn1, "pwin_normal": pnw,
                "pwin_binom": pbin,
                "p1_bonf": min(1.0, pn1 * N_TOTAL), "pwin_bonf": min(1.0, pnw * N_TOTAL),
                "pwin_binom_bonf": min(1.0, pbin * N_TOTAL),
            }  # fmt: skip
        res[it] = out_w
        print(it, {w: (v["n"], v["win"], round(v["p1"], 4), round(v["pwin"], 4),
                       round(v["p1_bonf"], 3), round(v["pwin_binom_bonf"], 3))
                   for w, v in out_w.items()}, flush=True)  # fmt: skip
    (out / "random.json").write_text(json.dumps(res, indent=1))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X ≥ k), X ~ Bin(n, p). 정확 합."""
    if n == 0:
        return 1.0
    tot = 0.0
    for i in range(k, n + 1):
        tot += math.comb(n, i) * p**i * (1 - p) ** (n - i)
    return min(1.0, tot)


def cmd_spot(out: Path, items: list | None) -> None:
    """가격만 쓰는 결선의 현물 2017-09 ~ 2020-07 표본 밖(펀딩 없음, 빈 봉 표시)."""
    g = json.loads((out / "grid.json").read_text())
    if not items:
        items = list(dict.fromkeys(g["candidates"] + g["top2_val_win"] + g["top2_val_x10"]))
    items = [it for it in items if K.price_only(it.split("|")[0])]
    if not items:
        print("가격만 쓰는 결선 없음")
        (out / "spot.json").write_text(json.dumps({"note": "no price-only finalist"}))
        return
    b = T.spot15()
    comp, _ = K.build_price_only(b)
    B = bars(b)
    res = {"gaps": b["_gaps"], "gap_bars": b["_gap_bars"],
           "bh": float(b["c"][-1] / b["o"][0] - 1)}  # fmt: skip
    a = int(np.searchsorted(b["t"], E.ms("2018-01-01")))
    for it in items:
        sname, x = it.split("|")
        s = comp["sets"][sname]
        nat = make_native(
            x, comp["atr"], comp["st_l"], comp["st_s"], s.get("own_l"), s.get("own_s")
        )
        tr = T.simulate(B, {"sig": s["sig"], "native": nat}, ("N",), a, len(b["t"]))
        res[it] = summ(tr)
        print(it, res[it]["n"], res[it]["win"], res[it]["x10"], res[it]["x1"])
    (out / "spot.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", choices=("grid", "random", "spot"))
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--items", nargs="*")
    a_ = ap.parse_args()
    o = Path(a_.out).expanduser()
    if a_.cmd == "grid":
        cmd_grid(o)
    elif a_.cmd == "random":
        cmd_random(o, a_.nsim, a_.items)
    else:
        cmd_spot(o, a_.items)
