"""패턴 책 두 번째 묶음: 하모닉 4종, AB=CD, 쓰리 드라이브, 골든 포켓, 상승 삼각.

등록: 2026-11-02-btc-book2-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_book2.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample
from btc_swing_patterns import CAP, COST, SPLIT, THRESH, breakout, summarize, zigzag

# 이름: (B 되돌림 범위, C 되돌림 범위, CD/BC 범위, D = XA 비율, 손절 = XA 비율)
HARM = {
    "GART": ((0.588, 0.648), (0.382, 0.886), (1.13, 1.618), 0.786, 1.0),
    "BAT": ((0.382, 0.5), (0.382, 0.886), (1.618, 2.618), 0.886, 1.13),
    "CRAB": ((0.382, 0.618), (0.382, 0.886), (2.618, 3.618), 1.618, 2.0),
    "BFLY": ((0.756, 0.816), (0.382, 0.886), (1.618, 2.24), 1.27, 1.414),
}
TGT = 0.618  # 목표: AD의 0.618 되돌림


def limit_trade(d: dict, side: int, start: int, lim: int, entry: float, stop: float,
                target: float, invalid: float):
    """start 봉부터 지정가 entry 체결을 기다린다. invalid를 먼저 넘으면 취소.

    체결 봉에서 손절가에 닿으면 손절로 본다. 체결 뒤 CAP봉 안에 결판 없으면 다음 시가.
    롱 기준(side=1): 저가가 entry 이하이면 체결, 시가가 이미 아래면 시가 체결.
    """
    o, h, l = d["o"], d["h"], d["l"]
    n = len(o)
    for i in range(start, min(start + lim, n - 1)):
        if side > 0:
            if h[i] > invalid and l[i] > entry:
                return None
            if l[i] <= entry:
                px = min(o[i], entry)
                break
        else:
            if l[i] < invalid and h[i] < entry:
                return None
            if h[i] >= entry:
                px = max(o[i], entry)
                break
    else:
        return None
    if (side > 0 and px <= stop) or (side < 0 and px >= stop):
        # 시가가 이미 손절가 밖에서 체결: 손절 주문이 바로 체결가에 나간다고 본다
        return 0.0, "stop", px, i
    end = min(i + CAP, n - 1)
    for j in range(i, end):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            return (stop / px - 1) * side, "stop", px, i
        if j > i and ((side > 0 and h[j] >= target) or (side < 0 and l[j] <= target)):
            return (target / px - 1) * side, "target", px, i
    return (o[end] / px - 1) * side, "time", px, i


def next_open_trade(d: dict, side: int, sig: int, stop: float, target: float):
    o, h, l = d["o"], d["h"], d["l"]
    e = sig + 1
    px = o[e]
    if (side > 0 and (px <= stop or px >= target)) or (side < 0 and (px >= stop or px <= target)):
        return None
    end = min(e + CAP, len(o) - 1)
    for i in range(e, end):
        if (side > 0 and l[i] <= stop) or (side < 0 and h[i] >= stop):
            return (stop / px - 1) * side, "stop", px, e
        if (side > 0 and h[i] >= target) or (side < 0 and l[i] <= target):
            return (target / px - 1) * side, "target", px, e
    return (o[end] / px - 1) * side, "time", px, e


def inr(x: float, lo_hi: tuple) -> bool:
    return lo_hi[0] <= x <= lo_hi[1]


def scan(d: dict, piv: list, side: int) -> list:
    """piv는 롱 기준으로 뒤집힌 피벗(저점 −1, 고점 +1). 가격도 롱 기준(숏이면 −가격)."""
    out = []
    sgn = side
    dd = d

    def real(p):
        return p * sgn

    for k in range(len(piv)):
        p = piv
        # 하모닉: X 저 → A 고 → B 저 → C 고, D 지정가 매수
        if k >= 3 and p[k][2] == 1 and p[k - 3][2] == -1:
            X, A, B, C = p[k - 3], p[k - 2], p[k - 1], p[k]
            xa = A[1] - X[1]
            ab = A[1] - B[1]
            if xa > 0 and ab > 0:
                b_r = ab / xa
                bc = C[1] - B[1]
                c_r = bc / ab
                for name, (br, cr, cdr, dr, sr) in HARM.items():
                    if not (inr(b_r, br) and inr(c_r, cr)):
                        continue
                    Dp = A[1] - dr * xa
                    if bc <= 0 or not inr((C[1] - Dp) / bc, cdr):
                        continue
                    stop = A[1] - sr * xa
                    tgt = Dp + TGT * (A[1] - Dp)
                    lim = 2 * max(C[0] - X[0], 4)
                    out.append((name, C[3] + 1, lim, Dp, stop, tgt, C[1], "limit"))
            # AB=CD: A 고 → B 저 → C 고, D = C − AB
            Ah, Bl, Ch = p[k - 2], p[k - 1], p[k]
            ab2 = Ah[1] - Bl[1]
            if ab2 > 0 and Ah[2] == 1 and Bl[2] == -1:
                bcr = (Ch[1] - Bl[1]) / ab2
                if inr(bcr, (0.382, 0.886)):
                    Dp = Ch[1] - ab2
                    stop = Ch[1] - 1.272 * ab2
                    tgt = Dp + TGT * (Ah[1] - Dp)
                    lim = 2 * max(Ch[0] - Ah[0], 4)
                    out.append(("ABCD", Ch[3] + 1, lim, Dp, stop, tgt, Ch[1], "limit"))
        # 쓰리 드라이브: H0 → L1 → H1 → L2 → H2, 세 번째 드라이브 끝 지정가 매수
        if k >= 4 and p[k][2] == 1 and p[k - 4][2] == 1:
            H0, L1, H1, L2, H2 = p[k - 4:k + 1]
            d1 = H0[1] - L1[1]
            r1 = H1[1] - L1[1]
            if d1 > 0 and r1 > 0 and inr(r1 / d1, (0.518, 0.718)):
                d2 = H1[1] - L2[1]
                if inr(d2 / r1, (1.13, 1.414)) and L2[1] < L1[1]:
                    r2 = H2[1] - L2[1]
                    if d2 > 0 and inr(r2 / d2, (0.518, 0.718)):
                        Dp = H2[1] - 1.272 * r2
                        stop = H2[1] - 1.618 * r2
                        if Dp < L2[1]:
                            lim = 2 * max(H2[0] - L1[0], 4)
                            out.append(("3DRV", H2[3] + 1, lim, Dp, stop, H2[1], H2[1], "limit"))
        # 골든 포켓: 저점 L → 고점 H 확정 뒤 0.618 되돌림 지정가 매수, 손절 L, 목표 H
        if k >= 1 and p[k][2] == 1 and p[k - 1][2] == -1:
            L, Hh = p[k - 1], p[k]
            up = Hh[1] - L[1]
            if up > 0:
                Dp = Hh[1] - 0.618 * up
                lim = 2 * max(Hh[0] - L[0], 4)
                out.append(("GP", Hh[3] + 1, lim, Dp, L[1], Hh[1], Hh[1], "limit"))
        # 상승 삼각: L0 → H1 → L1 → H2 → L2, 고점 수평, 저점 상승, 고점 돌파 종가
        if k >= 4 and p[k][2] == -1 and p[k - 4][2] == -1:
            L0, H1, L1, H2, L2 = p[k - 4:k + 1]
            height = H1[1] - L0[1]
            if height > 0 and L0[1] < L1[1] < L2[1] and abs(H2[1] - H1[1]) <= 0.15 * height:
                neck = max(H1[1], H2[1])
                br = breakout(dd["c_side"], L2[3], neck, L2[1], 2 * max(L2[0] - L0[0], 4))
                if br is not None:
                    out.append(("ATRI", br, 0, None, L2[1], neck + height, None, "break"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = resample(load5(Path(args.data) / "5m.csv"), 6)
    n = len(d["c"])
    end = d["t"][-1] + 1_800_000
    t6, t1 = end - 183 * 86_400_000, end - 30 * 86_400_000
    rows = []
    for th in THRESH:
        base = zigzag(d["h"], d["l"], th)
        for side in (1, -1):
            piv = base if side > 0 else [(i, -p, -k, cf) for i, p, k, cf in base]
            d["c_side"] = d["c"] * side
            for name, start, lim, entry, stop, tgt, inval, how in scan(d, piv, side):
                if how == "limit":
                    st, tg, en, iv = stop * side, tgt * side, entry * side, inval * side
                    res = limit_trade(d, side, start, lim, en, st, tg, iv)
                else:
                    st, tg = stop * side, tgt * side
                    res = next_open_trade(d, side, start, st, tg)
                if res is None:
                    continue
                gross, why, px, at = res
                rr = abs(tg - px) / abs(px - st) if px != st else 0.0
                rows.append({"pat": name, "side": side, "th": th, "t": float(d["t"][at]),
                             "at": int(at), "gross": float(gross), "why": why, "rr": float(rr)})
        print("th", th, "rows", len(rows), flush=True)
    groups = {k: [k] for k in ("GART", "BAT", "CRAB", "BFLY", "ABCD", "3DRV", "GP", "ATRI")}
    groups["HARM"] = ["GART", "BAT", "CRAB", "BFLY"]
    rep: dict = {"end": end, "table": {}}
    for gname, pats in groups.items():
        for side in (1, -1):
            for th in THRESH:
                rs = [r for r in rows if r["pat"] in pats and r["side"] == side and r["th"] == th]
                key = f"{gname}_{'L' if side > 0 else 'S'}_{int(th * 100)}"
                rep["table"][key] = {
                    "all": summarize(rs, COST[0]), "all_c10": summarize(rs, COST[1]),
                    "old": summarize([r for r in rs if r["t"] < SPLIT], COST[0]),
                    "new": summarize([r for r in rs if r["t"] >= SPLIT], COST[0]),
                    "6m": summarize([r for r in rs if r["t"] >= t6], COST[0]),
                    "30d": summarize([r for r in rs if r["t"] >= t1], COST[0]),
                }
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "rows.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
