"""차트 패턴 돌파 + 거래량 확인. 등록: 2026-11-02-btc-breakout-volume-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_breakout_volume.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_swing_patterns as S
from btc_fractal import load5, resample

SIZES = (0.02, 0.04)
KINDS = ("DB", "TB", "IHS")
VARS = ("V0", "V1", "V2", "V3")


def atri(piv: list, c: np.ndarray) -> list:
    """상승 삼각(두 번째 묶음과 같은 조건): (신호 봉, 손절, 목표)."""
    out = []
    for k in range(4, len(piv)):
        L0, H1, L1, H2, L2 = piv[k - 4:k + 1]
        if L2[2] != -1 or L0[2] != -1:
            continue
        ht = H1[1] - L0[1]
        if ht > 0 and L0[1] < L1[1] < L2[1] and abs(H2[1] - H1[1]) <= 0.15 * ht:
            neck = max(H1[1], H2[1])
            br = S.breakout(c, L2[3], neck, L2[1], 2 * max(L2[0] - L0[0], 4))
            if br is not None:
                out.append((br, L2[1], neck + ht, "ATRI"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + args.offset * 60_000
    S.CAP = 48
    v = d["v"]
    n = len(v)
    cs = np.cumsum(np.r_[0.0, v])

    def vmean(a: int, b: int) -> float:  # [a, b)
        a = max(a, 0)
        return (cs[b] - cs[a]) / max(b - a, 1)

    rows = []
    for th in SIZES:
        base = S.zigzag(d["h"], d["l"], th)
        for side in (1, -1):
            piv = base if side > 0 else [(i, -p, -k, cf) for i, p, k, cf in base]
            c = d["c"] * side
            sigs = []
            for kind in KINDS:
                sigs += S.find(piv, c, kind)
            sigs += atri(piv, c)
            for sig, stop, tgt, name in sigs:
                if sig < 100 or sig >= n - 2:
                    continue
                st, tg = (stop, tgt) if side > 0 else (-stop, -tgt)
                res = S.outcome(d, sig, side, st, tg)
                if res is None:
                    continue
                g, why, px = res
                vb = v[sig] >= 1.5 * vmean(sig - 20, sig)
                vf = vmean(sig - 40, sig) < vmean(sig - 80, sig - 40)
                flags = {"V0": True, "V1": vb, "V2": vf, "V3": vb and vf}
                for var, ok in flags.items():
                    if ok:
                        rows.append({"pat": name, "side": side, "th": th, "v": var,
                                     "t": float(d["t"][sig]), "g": float(g), "why": why})
    rep: dict = {"offset": args.offset, "table": {}}
    groups = {"ALL": ["DB", "TB", "IHS", "ATRI"], "DB": ["DB"], "TB": ["TB"], "IHS": ["IHS"],
              "ATRI": ["ATRI"]}
    for gname, pats in groups.items():
        for side in (1, -1):
            for var in VARS:
                rs = [r for r in rows if r["pat"] in pats and r["side"] == side and r["v"] == var]
                key = f"{gname}_{'L' if side > 0 else 'S'}_{var}"
                cell = {}
                for nm, sub in (("all", rs), ("old", [r for r in rs if r["t"] < S.SPLIT]),
                                ("new", [r for r in rs if r["t"] >= S.SPLIT])):
                    if not sub:
                        cell[nm] = {"n": 0}
                        continue
                    g = np.array([r["g"] for r in sub])
                    srt = np.sort(g - 0.001)
                    cell[nm] = {"n": len(sub), "win": round(float(((g - 0.001) > S.WIN).mean() * 100), 1),
                                "ev": round(float((g - 0.001).mean() * 100), 3),
                                "ev10": round(float((g - 0.002).mean() * 100), 3),
                                "drop5": round(float(srt[:-5].mean() * 100), 3) if len(srt) > 5 else None}
                rep["table"][key] = cell
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
