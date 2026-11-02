"""패턴 D 구간 + 5분봉 방아쇠(SCR 3단, 구조 전환) + 다이버전스 + 4시간봉 추세.

등록: 2026-11-02-btc-zone-trigger-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_zone_trigger.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_book2 as K
from btc_fractal import MIN5, load5, resample, rsi, scr3
from btc_swing_patterns import SPLIT, WIN, zigzag

SIZES = (0.02, 0.04)
WAIT = 72  # 5분봉 72개 = 6시간
CAP5 = 288  # 5분봉 하루
COST = (0.0005, 0.001)
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V6")


def ema(x: np.ndarray, n: int) -> np.ndarray:
    k = 2 / (n + 1)
    out = np.empty(len(x))
    v = x[0]
    for i, xi in enumerate(x):
        v += k * (xi - v)
        out[i] = v
    return out


def zones(piv: list) -> list:
    """K.scan과 같은 조건의 D 구간 패턴. 직전 저점(B, L2, L) 피벗을 함께 낸다."""
    out = []
    p = piv
    for k in range(len(p)):
        if k >= 3 and p[k][2] == 1 and p[k - 3][2] == -1:
            X, A, B, C = p[k - 3], p[k - 2], p[k - 1], p[k]
            xa, ab, bc = A[1] - X[1], A[1] - B[1], C[1] - B[1]
            if xa > 0 and ab > 0 and bc > 0:
                for name, (br, cr, cdr, dr, sr) in K.HARM.items():
                    if K.inr(ab / xa, br) and K.inr(bc / ab, cr):
                        dp = A[1] - dr * xa
                        if K.inr((C[1] - dp) / bc, cdr):
                            out.append(("HARM", C, B, dp, A[1] - sr * xa, dp + K.TGT * (A[1] - dp),
                                        2 * max(C[0] - X[0], 4)))
            if K.inr(bc / ab if ab > 0 else 0, (0.382, 0.886)) and ab > 0:
                dp = C[1] - ab
                out.append(("ABCD", C, B, dp, C[1] - 1.272 * ab, dp + K.TGT * (A[1] - dp),
                            2 * max(C[0] - A[0], 4)))
        if k >= 4 and p[k][2] == 1 and p[k - 4][2] == 1:
            H0, L1, H1, L2, H2 = p[k - 4:k + 1]
            d1, r1 = H0[1] - L1[1], H1[1] - L1[1]
            if d1 > 0 and r1 > 0 and K.inr(r1 / d1, (0.518, 0.718)):
                d2 = H1[1] - L2[1]
                if K.inr(d2 / r1, (1.13, 1.414)) and L2[1] < L1[1]:
                    r2 = H2[1] - L2[1]
                    if d2 > 0 and K.inr(r2 / d2, (0.518, 0.718)):
                        dp = H2[1] - 1.272 * r2
                        if dp < L2[1]:
                            out.append(("3DRV", H2, L2, dp, H2[1] - 1.618 * r2, H2[1],
                                        2 * max(H2[0] - L1[0], 4)))
        if k >= 1 and p[k][2] == 1 and p[k - 1][2] == -1:
            L, Hh = p[k - 1], p[k]
            if Hh[1] > L[1]:
                out.append(("GP", Hh, L, Hh[1] - 0.618 * (Hh[1] - L[1]), L[1], Hh[1],
                            2 * max(Hh[0] - L[0], 4)))
    return out


def run_side(d30, d5, f5, r5, scr_on, trend_bad, piv, side) -> list:
    """f5: 롱 기준으로 뒤집힌 5분봉(o,h,l,c). r5: 그 기준 RSI. scr_on: 그 기준 SCR 3단 반등."""
    t30, t5 = d30["t"], d5["t"]
    o, h, l, c = f5["o"], f5["h"], f5["l"], f5["c"]
    n5 = len(t5)
    out = []
    for name, C, B, dp, stop, tgt, lim in zones(piv):
        # 30분봉 시작 봉(C 확정 다음 봉)의 5분봉 위치부터 지정가 구간을 본다
        s30 = C[3] + 1
        if s30 >= len(t30):
            continue
        j = int(np.searchsorted(t5, t30[s30]))
        jend = min(int(np.searchsorted(t5, t30[min(s30 + lim, len(t30) - 1)])), n5 - 1)
        touch = None
        while j < jend:
            if h[j] > C[1] and l[j] > dp:
                break
            if l[j] <= dp:
                touch = j
                break
            j += 1
        if touch is None:
            continue
        # B(직전 저점)의 5분봉 위치: 그 30분봉 안의 최저가 봉
        b0 = int(np.searchsorted(t5, t30[B[0]]))
        b_idx = b0 + int(np.argmin(l[b0:b0 + 6]))
        res = {}
        # V0: D 지정가 즉시 체결
        px0 = min(o[touch], dp)
        res["V0"] = (touch, px0)
        # 방아쇠 대기
        level = h[max(touch - 6, 0):touch + 1].max()
        trig_scr = trig_brk = None
        for jj in range(touch, min(touch + WAIT, n5 - 2)):
            if l[jj] <= stop:
                break
            if trig_scr is None and scr_on[jj]:
                trig_scr = jj
            if trig_brk is None and jj > touch and c[jj] > level:
                trig_brk = jj
            if trig_scr is not None and trig_brk is not None:
                break

        def div_ok(jj: int) -> bool:
            low_j = touch + int(np.argmin(l[touch:jj + 1]))
            return l[low_j] < l[b_idx] and r5[low_j] > r5[b_idx]

        for v, trig, need_div, need_trend in (
            ("V1", trig_scr, False, False), ("V2", trig_brk, False, False),
            ("V3", trig_scr, True, False), ("V4", trig_brk, True, False),
            ("V5", trig_scr, True, True), ("V6", trig_brk, True, True),
        ):
            if trig is None or trig + 1 >= n5:
                continue
            if need_div and (name == "GP" or not div_ok(trig)):
                continue
            if need_trend and trend_bad[trig]:
                continue
            res[v] = (trig + 1, o[trig + 1])
        for v, (e, px) in res.items():
            if px >= tgt:
                continue
            if px <= stop:
                out.append((name, v, e, 0.0, "stop", 0.0))
                continue
            end = min(e + CAP5, n5 - 1)
            x, why = o[end], "time"
            for q in range(e, end):
                if l[q] <= stop:
                    x, why = stop, "stop"
                    break
                if q > e and h[q] >= tgt:
                    x, why = tgt, "target"
                    break
            # 뒤집힌 가격(숏)은 음수라 x/px − 1의 부호를 바꾸면 원래 숏 수익이 된다
            g = (x / px - 1) * side
            out.append((name, v, e, float(g), why, (tgt - px) / (px - stop)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0, help="30분봉 시작을 분 단위로 민다")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d30 = resample(sh, 6)
    d30["t"] = d30["t"] + args.offset * 60_000
    on_l, on_s, _, _, r = scr3(d5)
    h4 = resample(d5, 48)
    e20, e50 = ema(h4["c"], 20), ema(h4["c"], 50)
    down4 = (h4["c"] < e20) & (e20 < e50)
    up4 = (h4["c"] > e20) & (e20 > e50)
    j4 = np.searchsorted(h4["t"] + 4 * 3_600_000, d5["t"] + MIN5, side="right") - 1
    ok4 = j4 >= 0
    bad_long = np.zeros(len(d5["t"]), bool)
    bad_short = bad_long.copy()
    bad_long[ok4] = down4[j4[ok4]]
    bad_short[ok4] = up4[j4[ok4]]
    flip5 = {"o": -d5["o"], "h": -d5["l"], "l": -d5["h"], "c": -d5["c"]}
    rows = []
    for th in SIZES:
        base = zigzag(d30["h"], d30["l"], th)
        for side in (1, -1):
            piv = base if side > 0 else [(i, -p, -k, cf) for i, p, k, cf in base]
            f5 = d5 if side > 0 else flip5
            r5 = r if side > 0 else 100 - r
            on = on_l if side > 0 else on_s
            bad = bad_long if side > 0 else bad_short
            for name, v, e, g, why, rr in run_side(d30, d5, f5, r5, on, bad, piv, side):
                rows.append({"pat": name, "v": v, "side": side, "th": th, "t": float(d5["t"][e]),
                             "gross": g, "why": why, "rr": float(rr)})
        print("th", th, len(rows), flush=True)
    end = d5["t"][-1] + MIN5
    t6 = end - 183 * 86_400_000

    def summ(rs, cost):
        if not rs:
            return {"n": 0}
        net = np.array([x["gross"] for x in rs]) - 2 * cost
        s = np.sort(net)
        return {"n": len(rs), "win": round(float((net > WIN).mean() * 100), 1),
                "rr": round(float(np.median([x["rr"] for x in rs])), 2),
                "ev": round(float(net.mean() * 100), 3),
                "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None,
                "stop": round(float(np.mean([x["why"] == "stop" for x in rs]) * 100), 1)}

    rep: dict = {"offset": args.offset, "table": {}}
    fams = {"ZONE": ["HARM", "ABCD", "3DRV"], "HARM": ["HARM"], "ABCD": ["ABCD"], "3DRV": ["3DRV"],
            "GP": ["GP"]}
    for fam, pats in fams.items():
        for side in (1, -1):
            for v in VARIANTS:
                rs = [x for x in rows if x["pat"] in pats and x["side"] == side and x["v"] == v]
                key = f"{fam}_{'L' if side > 0 else 'S'}_{v}"
                rep["table"][key] = {
                    "all": summ(rs, COST[0]), "c10": summ(rs, COST[1]),
                    "old": summ([x for x in rs if x["t"] < SPLIT], COST[0]),
                    "new": summ([x for x in rs if x["t"] >= SPLIT], COST[0]),
                    "6m": summ([x for x in rs if x["t"] >= t6], COST[0]),
                }
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "rows.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
