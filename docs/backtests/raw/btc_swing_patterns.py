"""지그재그 스윙 구조 패턴(이중·삼중 바닥/천장, 첫 눌림, 머리어깨)의 과거 승률과 손익비.

등록: 2026-11-02-btc-swing-patterns-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_swing_patterns.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import MIN5, load5, resample

THRESH = (0.01, 0.02, 0.04, 0.08)
CAP = 48
COST = (0.0005, 0.001)
WIN = 0.001
TOL = 0.2  # 두 바닥(천장) 차이 ≤ 반등 높이의 20%
SHOULDER_TOL = 0.35
SPLIT = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def zigzag(h: np.ndarray, l: np.ndarray, th: float) -> list:
    """(극점 봉, 가격, 종류 +1 고점/−1 저점, 확정 봉). 극점에서 th만큼 되돌면 그 봉에 확정.

    다음 극점은 확정된 극점의 다음 봉부터 찾는다. 같은 봉을 고점·저점으로 거듭 찍지 않게 하려는 것이다.
    """
    piv = []
    trend = 0
    hi_i = lo_i = 0
    for i in range(1, len(h)):
        if trend == 0:
            if h[i] > h[hi_i]:
                hi_i = i
            if l[i] < l[lo_i]:
                lo_i = i
            if h[i] >= l[lo_i] * (1 + th) and lo_i < i:
                piv.append((lo_i, l[lo_i], -1, i))
                trend = 1
                hi_i = lo_i + 1 + int(np.argmax(h[lo_i + 1:i + 1]))
            elif l[i] <= h[hi_i] * (1 - th) and hi_i < i:
                piv.append((hi_i, h[hi_i], 1, i))
                trend = -1
                lo_i = hi_i + 1 + int(np.argmin(l[hi_i + 1:i + 1]))
        elif trend == 1:
            if h[i] > h[hi_i]:
                hi_i = i
            if l[i] <= h[hi_i] * (1 - th) and hi_i < i:
                piv.append((hi_i, h[hi_i], 1, i))
                trend = -1
                lo_i = hi_i + 1 + int(np.argmin(l[hi_i + 1:i + 1]))
        else:
            if l[i] < l[lo_i]:
                lo_i = i
            if h[i] >= l[lo_i] * (1 + th) and lo_i < i:
                piv.append((lo_i, l[lo_i], -1, i))
                trend = 1
                hi_i = lo_i + 1 + int(np.argmax(h[lo_i + 1:i + 1]))
    return piv


def sym(a: int, b: int) -> bool:
    """두 구간 봉 수의 비가 0.5~2 사이. 모양이 한쪽으로 늘어진 것을 뺀다."""
    return a > 0 and b > 0 and 0.5 <= a / b <= 2.0


def find(piv: list, c: np.ndarray, kind: str) -> list:
    """롱 기준 패턴을 찾는다. 숏은 가격을 뒤집어 같은 함수를 쓴다.

    결과: (신호 봉, 손절가, 목표가, 이름). 신호 봉 마감에 판단, 다음 봉 시가 진입.
    """
    sigs = []
    n = len(c)
    lows = [k for k, p in enumerate(piv) if p[2] == -1]
    for k in lows:
        p = piv
        if kind in ("DB", "DBP"):
            if k < 3:
                continue
            h0, l1, hh, l2 = p[k - 3], p[k - 2], p[k - 1], p[k]
            if not (h0[2] == 1 and l1[2] == -1 and hh[2] == 1):
                continue
            height = hh[1] - min(l1[1], l2[1])
            if height <= 0 or abs(l2[1] - l1[1]) > TOL * (hh[1] - l1[1]):
                continue
            if h0[1] - l1[1] < (hh[1] - l1[1]):  # 앞선 하락이 반등보다 커야 한다
                continue
            if not sym(hh[0] - l1[0], l2[0] - hh[0]):  # 두 바닥의 간격이 고르게
                continue
            neck, stop = hh[1], min(l1[1], l2[1])
            target = neck + height
            br = breakout(c, l2[3], neck, stop, max(l2[0] - l1[0], 4) * 2)
            if br is None:
                continue
            if kind == "DB":
                sigs.append((br, stop, target, "DB"))
            else:
                # 돌파 뒤 첫 눌림 저점이 넥라인 근처에서 확정되면 진입
                nxt = [q for q in p[k + 1:] if q[3] > br]
                if len(nxt) >= 2 and nxt[0][2] == 1 and nxt[1][2] == -1:
                    hl = nxt[1]
                    if hl[1] > stop and hl[3] < n - 1:
                        sigs.append((hl[3], hl[1], target, "DBP"))
        elif kind == "TB":
            if k < 5:
                continue
            h0, l1, h1, l2, h2, l3 = p[k - 5:k + 1]
            if not (h0[2] == 1 and l1[2] == -1):
                continue
            base = min(l1[1], l2[1], l3[1])
            neck = max(h1[1], h2[1])
            height = neck - base
            if height <= 0 or max(l1[1], l2[1], l3[1]) - base > TOL * height:
                continue
            if not sym(l2[0] - l1[0], l3[0] - l2[0]) or abs(h1[1] - h2[1]) > 0.3 * height:
                continue
            if h0[1] - l1[1] < (h1[1] - l1[1]):
                continue
            br = breakout(c, l3[3], neck, base, max(l3[0] - l1[0], 4))
            if br is not None:
                sigs.append((br, base, neck + height, "TB"))
        elif kind == "IHS":
            if k < 5:
                continue
            h0, ls, h1, hd, h2, rs = p[k - 5:k + 1]
            if not (h0[2] == 1 and ls[2] == -1):
                continue
            if not (hd[1] < ls[1] and hd[1] < rs[1]):
                continue
            span = min(h1[1], h2[1]) - hd[1]
            if span <= 0 or abs(ls[1] - rs[1]) > SHOULDER_TOL * span:
                continue
            if not sym(hd[0] - ls[0], rs[0] - hd[0]):
                continue
            if h0[1] - ls[1] < (h1[1] - ls[1]):
                continue
            slope = (h2[1] - h1[1]) / max(h2[0] - h1[0], 1)
            head_neck = h1[1] + slope * (hd[0] - h1[0])
            height = head_neck - hd[1]
            stop = rs[1]
            lim = max(rs[0] - ls[0], 4)
            for i in range(rs[3], min(rs[3] + lim, n - 1)):
                neck_i = h1[1] + slope * (i - h1[0])
                if c[i] < stop:
                    break
                if c[i] > neck_i:
                    sigs.append((i, stop, neck_i + height, "IHS"))
                    break
    return sigs


def breakout(c: np.ndarray, start: int, neck: float, stop: float, lim: int):
    for i in range(start, min(start + lim, len(c) - 1)):
        if c[i] < stop:
            return None
        if c[i] > neck:
            return i
    return None


def outcome(d: dict, sig: int, side: int, stop: float, target: float) -> tuple:
    """다음 봉 시가 진입. 같은 봉에 둘 다 닿으면 손절 먼저. CAP봉 안에 결판 없으면 그 시가."""
    o, h, l = d["o"], d["h"], d["l"]
    e = sig + 1
    px = o[e]
    if (side > 0 and (px <= stop or px >= target)) or (side < 0 and (px >= stop or px <= target)):
        return None  # 진입가가 이미 손절·목표 밖
    end = min(e + CAP, len(o) - 1)
    for i in range(e, end):
        hit_s = l[i] <= stop if side > 0 else h[i] >= stop
        hit_t = h[i] >= target if side > 0 else l[i] <= target
        if hit_s:
            return (stop / px - 1) * side, "stop", px
        if hit_t:
            return (target / px - 1) * side, "target", px
    return (o[end] / px - 1) * side, "time", px


def summarize(rows: list, cost: float) -> dict:
    if not rows:
        return {"n": 0}
    g = np.array([r["gross"] for r in rows])
    net = g - 2 * cost
    w = net > WIN
    rr = np.array([r["rr"] for r in rows])
    loss = net[net < 0]
    gain = net[net > 0]
    return {
        "n": len(rows),
        "win": round(float(w.mean() * 100), 1),
        "rr_plan_med": round(float(np.median(rr)), 2),
        "avg_gain": round(float(gain.mean() * 100), 2) if len(gain) else 0.0,
        "avg_loss": round(float(loss.mean() * 100), 2) if len(loss) else 0.0,
        "ev": round(float(net.mean() * 100), 3),
        "target_share": round(float(np.mean([r["why"] == "target" for r in rows]) * 100), 1),
        "stop_share": round(float(np.mean([r["why"] == "stop" for r in rows]) * 100), 1),
    }


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
    flip = {"o": -d["o"], "h": -d["l"], "l": -d["h"], "c": -d["c"]}
    rows = []
    for th in THRESH:
        for side, src in ((1, d), (-1, flip)):
            # 숏: 가격을 −로 뒤집으면 천장이 바닥이 된다. 지그재그 비율은 원래 가격으로 계산한다
            piv = zigzag(d["h"], d["l"], th)
            if side < 0:
                piv = [(i, -p, -k, cf) for i, p, k, cf in piv]
            for kind in ("DB", "DBP", "TB", "IHS"):
                for sig, stop, target, name in find(piv, src["c"], kind):
                    if sig >= n - 2:
                        continue
                    st, tg = (stop, target) if side > 0 else (-stop, -target)
                    res = outcome(d, sig, side, st, tg)
                    if res is None:
                        continue
                    gross, why, px = res
                    rr = abs(tg - px) / abs(px - st)
                    rows.append({"pat": name, "side": side, "th": th, "t": float(d["t"][sig]),
                                 "sig": int(sig), "gross": float(gross), "why": why, "rr": float(rr),
                                 "stop": float(st), "target": float(tg), "entry": float(px)})
        print("th", th, "rows", len(rows), flush=True)
    rep: dict = {"end": end, "bars": n, "table": {}}
    for name in ("DB", "DBP", "TB", "IHS"):
        for side in (1, -1):
            for th in THRESH:
                rs = [r for r in rows if r["pat"] == name and r["side"] == side and r["th"] == th]
                key = f"{name}_{'L' if side > 0 else 'S'}_{int(th * 100)}"
                rep["table"][key] = {
                    "all": summarize(rs, COST[0]),
                    "all_c10": summarize(rs, COST[1]),
                    "old": summarize([r for r in rs if r["t"] < SPLIT], COST[0]),
                    "new": summarize([r for r in rs if r["t"] >= SPLIT], COST[0]),
                    "6m": summarize([r for r in rs if r["t"] >= t6], COST[0]),
                    "30d": summarize([r for r in rs if r["t"] >= t1], COST[0]),
                }
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "rows.json").write_text(json.dumps(rows))
    for k, v in rep["table"].items():
        a = v["all"]
        if a["n"]:
            print(k, a["n"], a["win"], a["rr_plan_med"], a["ev"], v["all_c10"]["ev"],
                  v["old"].get("ev"), v["new"].get("ev"), v["6m"].get("n"), v["6m"].get("ev"),
                  flush=True)


if __name__ == "__main__":
    main()
