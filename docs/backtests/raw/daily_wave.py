"""하루 1~2회 매매 규칙(daily wave). 등록: 2026-11-08-btc-daily-wave-hypotheses.md

진입 7개(세션 범위 돌파 3, SCR 눌림 2, 일중 되돌림 2) × 일봉 방향 문 4 × 수수료 방식 3 = 84.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/daily-wave-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3 docs/backtests/raw/daily_wave.py --out $OUT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pine_ports as P
from btc_fractal import scr3
from dmi15_eval import dmi, frame
from htf_live import direction, to15
from pine_eval import lev_path
from r5_data import load

R5 = Path("~/.minerva/research/round5-2026-11-08").expanduser()
M15, DAY = 900_000, 86_400_000
KST = ZoneInfo("Asia/Seoul")
TAKER, MAKER = 0.0005, 0.0002
TICK = 0.1
LEV = 5
ENTRIES = ("S-AS", "S-LN", "S-NY", "P0", "P8", "O-AN", "V-AN")
GATES = ("G0", "GT", "GD", "GM")
FEES = ("FA", "FB", "FC")
N_VAR = len(ENTRIES) * len(GATES) * len(FEES)  # 84
N_TONIGHT_BEFORE = 5675  # 5,563 + htf-live 112
WIN = {"fresh": ("2020-03-01", "2021-11-01"), "design": ("2022-01-01", "2024-07-01"),
       "val": ("2024-07-01", "2025-11-01"), "year": ("2025-11-01", None)}  # fmt: skip
MAIN = ("fresh", "design", "val", "year")
SESS = {"S-AS": (timezone.utc, 0, 0), "S-LN": (ZoneInfo("Europe/London"), 8, 0),
        "S-NY": (ZoneInfo("America/New_York"), 9, 30)}  # fmt: skip


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


# ---------------- 특징 ----------------
def features(b: dict) -> dict:
    n = len(b["c"])
    t, h, lo, c, v = b["t"], b["h"], b["l"], b["c"], b["v"]
    D = frame(b, 1440)
    F4 = frame(b, 240)
    # 일봉 방향 문 (마감된 일봉만, 다음 15분봉부터 적용)
    ev = P.s6_trend_dev(D["h"], D["l"], D["c"]).astype(float)
    st = np.where(ev != 0, ev, np.nan)
    idx = np.where(~np.isnan(st), np.arange(len(st)), 0)
    np.maximum.accumulate(idx, out=idx)
    t6 = np.nan_to_num(st[idx], nan=0.0)
    hh, ll = P.highest(D["h"], 20), P.lowest(D["l"], 20)
    with np.errstate(invalid="ignore"):
        dmid = np.where(D["c"] > (hh + ll) / 2, 1.0, np.where(D["c"] < (hh + ll) / 2, -1.0, 0.0))
    pdi, mdi, _ = dmi(D["h"], D["l"], D["c"])
    with np.errstate(invalid="ignore"):
        dm = np.where(pdi > mdi, 1.0, np.where(pdi < mdi, -1.0, 0.0))
    # htf_live.to15(fill=0)은 앞으로 채우지 않는다(채움은 NaN 칸만) → NaN으로 채운 뒤 0으로 바꾼다
    ff = lambda v: np.nan_to_num(to15(D, v, n), nan=0.0)  # noqa: E731
    gate = {"G0": np.zeros(n), "GT": ff(t6), "GD": ff(dmid), "GM": ff(dm)}
    atr_d = to15(D, P.atr(D["h"], D["l"], D["c"], 14), n)
    d15 = direction(h, lo, c)
    d4 = np.nan_to_num(to15(F4, direction(F4["h"], F4["l"], F4["c"]).astype(float), n), nan=0.0)
    d1 = np.nan_to_num(to15(D, direction(D["h"], D["l"], D["c"]).astype(float), n), nan=0.0)
    agree = (d15 != 0) & (d15 == d4) & (d4 == d1)
    day = t // DAY
    first = np.r_[True, day[1:] != day[:-1]]
    day_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    last = np.r_[day[1:] != day[:-1], True]
    day_end = np.minimum.accumulate(np.where(last, np.arange(n), n)[::-1])[::-1]
    # 전일 VWAP
    tp = (h + lo + c) / 3
    ud, inv = np.unique(day, return_inverse=True)
    pv = np.bincount(inv, tp * v)
    vv = np.bincount(inv, v)
    vw = np.where(vv > 0, pv / np.where(vv > 0, vv, 1), np.nan)
    prev_vwap = np.r_[np.nan, vw[:-1]][inv]
    on_l, on_s, s, cc, r = scr3({"h": h, "l": lo, "c": c})
    return {"gate": gate, "atr_d": atr_d, "agree": agree, "day": day, "day_start": day_start,
            "day_end": day_end, "prev_vwap": prev_vwap, "scr_l": on_l, "scr_s": on_s,
            "stoch": s, "cci": cc, "rsi": r}  # fmt: skip


# ---------------- 후보(신호) ----------------
def cand(i, d, stop, tp, tmax) -> dict:
    return {"i": int(i), "d": int(d), "stop": float(stop), "tp": float(tp), "tmax": int(tmax)}


def session_cands(b, F, name) -> list:
    t, h, lo, c = b["t"], b["h"], b["l"], b["c"]
    n = len(c)
    tz, hh_, mm_ = SESS[name]
    out = []
    for dd in range(int(t[0] // DAY), int(t[-1] // DAY) + 1):
        dt = datetime.fromtimestamp(dd * 86400, timezone.utc)
        if dt.weekday() >= 5:
            continue
        st_ms = int(datetime(dt.year, dt.month, dt.day, hh_, mm_, tzinfo=tz).timestamp() * 1000)
        st_ms = (st_ms + M15 - 1) // M15 * M15  # 뉴욕 9:30 → 15분 칸에 맞춘다
        s0 = int(np.searchsorted(t, st_ms))
        if s0 + 4 + 16 >= n or int(t[s0]) != st_ms or int(t[s0 + 4]) - st_ms != 4 * M15:
            continue
        H, L = h[s0 : s0 + 4].max(), lo[s0 : s0 + 4].min()
        for m in range(s0 + 4, s0 + 20):
            if c[m] > H or c[m] < L:
                d = 1 if c[m] > H else -1
                stop = L if d == 1 else H
                out.append(cand(m, d, stop, c[m] + d * (H - L), min(m + 32, n - 1)))
                break
    return out


def scr_cands(b, F, confirm: bool) -> list:
    h, lo, c = b["h"], b["l"], b["c"]
    n = len(c)
    out = []
    sig = np.where(F["scr_l"], 1, np.where(F["scr_s"], -1, 0))
    for i in np.nonzero(sig)[0]:
        d = int(sig[i])
        j = None
        if not confirm:
            j = i
        else:
            for k in range(i + 1, min(i + 9, n - 1)):
                if (d == 1 and c[k] > h[i]) or (d == -1 and c[k] < lo[i]):
                    j = k
                    break
        if j is None or j < 8 or j >= n - 2:
            continue
        if d == 1:
            stop = max(lo[j - 7 : j + 1].min() * (1 - 0.001), c[j] * 0.98)
        else:
            stop = min(h[j - 7 : j + 1].max() * (1 + 0.001), c[j] * 1.02)
        risk = abs(c[j] - stop)
        if risk <= 0:
            continue
        out.append(cand(j, d, stop, c[j] + d * 1.5 * risk, F["day_end"][j]))
    return out


def revert_cands(b, F, anchor: str) -> list:
    o, c = b["o"], b["c"]
    n = len(c)
    out = []
    starts = np.unique(F["day_start"])
    for s0 in starts:
        e = int(F["day_end"][s0])
        a = o[s0] if anchor == "O-AN" else F["prev_vwap"][s0]
        k = F["atr_d"][s0]
        if not np.isfinite(a) or not np.isfinite(k) or k <= 0:
            continue
        seg = c[s0 : e + 1]
        for d in (1, -1):
            hit = np.nonzero((seg - a) * d <= -0.5 * k)[0]
            if len(hit):
                i = s0 + int(hit[0])
                if i < e and i < n - 2:
                    out.append(cand(i, d, c[i] - d * 0.5 * k, a, e))
    return out


def all_cands(b, F) -> dict:
    out = {}
    for name in ("S-AS", "S-LN", "S-NY"):
        out[name] = session_cands(b, F, name)
    out["P0"] = scr_cands(b, F, False)
    out["P8"] = scr_cands(b, F, True)
    out["O-AN"] = revert_cands(b, F, "O-AN")
    out["V-AN"] = revert_cands(b, F, "V-AN")
    for k in out:
        out[k].sort(key=lambda x: x["i"])
    return out


# ---------------- 모의 매매 ----------------
def simulate(b, F, cands, gate, fee, a, z, delay=0, fee_mult=1.0) -> tuple[list, dict]:
    o, h, lo, c, t = b["o"], b["h"], b["l"], b["c"], b["t"]
    ft, fr = b["ft"], b["fr"]
    g = F["gate"][gate]
    tk, mk = TAKER * fee_mult, MAKER * fee_mult
    trades, busy_until, per_day = [], -1, {}
    cnt = {"signals": 0, "gated": 0, "chase": 0, "nofill": 0, "busy": 0, "daycap": 0, "bad": 0}
    for x in cands:
        i, d = x["i"], x["d"]
        if i < a or i >= z - 2:
            continue
        cnt["signals"] += 1
        if gate != "G0" and g[i] != d:
            cnt["gated"] += 1
            continue
        if i <= busy_until:
            cnt["busy"] += 1
            continue
        day = int(F["day"][i])
        if per_day.get(day, 0) >= 2:
            cnt["daycap"] += 1
            continue
        tmax = min(x["tmax"], z - 1)
        if fee == "FA":
            e = i + 1 + delay
            if e > tmax:
                cnt["bad"] += 1
                continue
            ep = o[e]
            lim = 0.03 if F["agree"][i] else 0.01
            if abs(ep / c[i] - 1) > lim:
                cnt["chase"] += 1
                continue
            fin = tk
            chk0 = e
        else:
            lp = c[i]
            e = None
            for j in range(i + 1 + delay, min(i + 5 + delay, tmax + 1)):
                if (d == 1 and lo[j] < lp - TICK) or (d == -1 and h[j] > lp + TICK):
                    e = j
                    break
            if e is None:
                cnt["nofill"] += 1
                continue
            ep = lp
            fin = mk
            chk0 = e
        stop, tp = x["stop"], x["tp"]
        if (stop - ep) * d >= 0 or (tp - ep) * d <= 0:
            cnt["bad"] += 1
            continue
        xp = why = None
        worst = 0.0
        for j in range(chk0, tmax + 1):
            adv = (lo[j] / ep - 1) if d == 1 else (1 - h[j] / ep)
            worst = min(worst, adv)
            stop_hit = (lo[j] <= stop) if d == 1 else (h[j] >= stop)
            # 지정가 체결 봉에서는 익절을 보지 않는다(보수적)
            tp_hit = ((h[j] >= tp) if d == 1 else (lo[j] <= tp)) and not (fee != "FA" and j == e)
            if stop_hit:
                gap = (o[j] < stop) if d == 1 else (o[j] > stop)
                xp, why, xi = (o[j] if gap and j > chk0 else stop), "stop", j
                worst = min(worst, d * (xp / ep - 1))
                break
            if tp_hit:
                xp, why, xi = tp, "tp", j
                break
        if xp is None:
            xi = tmax
            xp, why = c[tmax], "time"
        fout = mk if (why == "tp" and fee == "FC") else tk
        t_in, t_out = int(t[e]), int(t[xi]) + M15
        sel = (ft > t_in) & (ft <= t_out)
        fund = -d * float(fr[sel].sum())
        r = d * (xp / ep - 1) - fin - fout + fund
        trades.append({"i": i, "e": e, "x": int(xi), "d": d, "ep": float(ep), "xp": float(xp),
                       "r": float(r), "worst": float(min(worst, r)), "why": why, "fund": fund})  # fmt: skip
        busy_until = xi
        per_day[day] = per_day.get(day, 0) + 1
    return trades, cnt


def summ(tr: list, a: int, z: int, b: dict, cnt: dict | None = None) -> dict:
    days = max((int(b["t"][z - 1]) - int(b["t"][a])) / DAY, 1)
    r = np.array([x["r"] for x in tr]) if tr else np.zeros(0)
    out = {"n": len(tr), "per_day": len(tr) / days, "win": float((r > 0).mean()) if len(r) else None,
           "avg": float(r.mean()) if len(r) else None,
           "fee_share": None, **(cnt or {})}  # fmt: skip
    for L in (1, LEV):
        p = lev_path(tr, L)
        out[f"x{L}"] = {k: float(v) for k, v in p.items()}
    return out


def windows(t) -> dict:
    end = int(t[-1]) + M15
    w = {k: (ms(a_), end if z_ is None else ms(z_)) for k, (a_, z_) in WIN.items()}
    w["m6"] = (end - 183 * DAY, end)
    w["d30"] = (end - 30 * DAY, end)
    w["pool"] = (ms("2020-03-01"), ms("2025-11-01"))
    return {k: (int(np.searchsorted(t, a_)), int(np.searchsorted(t, z_))) for k, (a_, z_) in w.items()}


def random_p(b, F, cands, gate, fee, a, z, real_tr, nsim, seed) -> dict:
    """시각 맞춘 무작위: 진짜 거래마다 같은 시각(UTC 15분 칸)·방향·익절/손절 거리·보유 한도로, 날짜만 섞는다."""
    rng = np.random.default_rng(seed)
    t = b["t"]
    c = b["c"]
    n = len(t)
    by_slot: dict = {}
    for i in range(a, z - 2):
        by_slot.setdefault(int(t[i] % DAY), []).append(i)
    by_slot = {k: np.array(v) for k, v in by_slot.items()}
    src = {x["i"]: x for x in cands}
    vals = []
    for _ in range(nsim):
        rc = []
        for tr_ in real_tr:
            x = src[tr_["i"]]
            pool = by_slot[int(t[x["i"]] % DAY)]
            j = int(rng.choice(pool))
            sd, td = (x["stop"] / c[x["i"]] - 1), (x["tp"] / c[x["i"]] - 1)
            hold = x["tmax"] - x["i"]
            rc.append(cand(j, x["d"], c[j] * (1 + sd), c[j] * (1 + td), min(j + hold, n - 1)))
        rc.sort(key=lambda y: y["i"])
        tr, _ = simulate(b, F, rc, "G0", fee, a, z)
        vals.append(lev_path(tr, LEV)["ret"])
    vals = np.array(vals)
    real = lev_path(real_tr, LEV)["ret"]
    return {"p": float((vals >= real).mean()), "rand_med": float(np.median(vals)), "real": float(real),
            "nsim": nsim}  # fmt: skip


def cmd_random(out: Path, names: list, nsim: int) -> None:
    """Bonferroni(×84)를 가를 만큼 무작위 횟수를 늘린다(200번이면 p 해상도 0.005라 부족)."""
    b = load(R5)
    F = features(b)
    C = all_cands(b, F)
    W = windows(b["t"])
    res = {}
    for name in names:
        en, g, fee = name.split("|")
        res[name] = {}
        for w in ("pool", "fresh", "design", "val", "year"):
            a, z = W[w]
            tr, _ = simulate(b, F, C[en], g, fee, a, z)
            res[name][w] = random_p(b, F, C[en], g, fee, a, z, tr, nsim if w == "pool" else 1000, 23)
            print(name, w, res[name][w], flush=True)
    (out / "random_more.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsim", type=int, default=200)
    ap.add_argument("--random", nargs="*", help="이 변형만 무작위 비교를 다시(많은 횟수)")
    args = ap.parse_args()
    if args.random:
        cmd_random(Path(args.out).expanduser(), args.random, args.nsim)
        return
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = load(R5)
    F = features(b)
    C = all_cands(b, F)
    W = windows(b["t"])
    print("built", round(time.time() - t0, 1), {k: len(v) for k, v in C.items()}, flush=True)
    res = {}
    for en in ENTRIES:
        for g in GATES:
            for fee in FEES:
                name = f"{en}|{g}|{fee}"
                res[name] = {}
                for w, (a, z) in W.items():
                    tr, cnt = simulate(b, F, C[en], g, fee, a, z)
                    res[name][w] = summ(tr, a, z, b, cnt)
                rr = res[name]
                rr["gate"] = {
                    "fee_ok": fee in ("FA", "FB"),
                    "pos_all": all(rr[w]["x5"]["ret"] > 0 for w in MAIN),
                    "liq0": all(rr[w]["x5"]["liq"] == 0 for w in MAIN),
                    "mdd40": all(rr[w]["x5"]["mdd"] <= 0.40 for w in MAIN),
                    "freq": rr["pool"]["per_day"] >= 0.7,
                }
        print(en, round(time.time() - t0, 1), flush=True)
    score = lambda n_: sum(np.log1p(max(res[n_][w]["x5"]["ret"], -0.999)) for w in MAIN)  # noqa: E731
    pre = [n_ for n_ in res if all(v for k, v in res[n_]["gate"].items())]
    ranked = sorted([n_ for n_ in res if res[n_]["pool"]["n"] >= 100], key=score, reverse=True)
    check = list(dict.fromkeys(pre + [n_ for n_ in ranked if n_.endswith("FA")][:2]
                               + [n_ for n_ in ranked if n_.endswith("FB")][:2]))  # fmt: skip
    rnd = {}
    a, z = W["pool"]
    for name in check:
        en, g, fee = name.split("|")
        tr, _ = simulate(b, F, C[en], g, fee, a, z)
        rnd[name] = random_p(b, F, C[en], g, fee, a, z, tr, args.nsim, 11)
        print("random", name, rnd[name], flush=True)
    for name in res:
        p = rnd.get(name, {}).get("p")
        res[name]["gate"]["random_bonf"] = bool(p is not None and p * N_VAR < 0.05)
        res[name]["pass"] = all(res[name]["gate"].values())
    red = {}
    for name in check:
        en, g, fee = name.split("|")
        red[name] = {}
        for w in MAIN:
            a, z = W[w]
            red[name][w] = {
                "delay1": summ(simulate(b, F, C[en], g, fee, a, z, delay=1)[0], a, z, b)["x5"],
                "fee15": summ(simulate(b, F, C[en], g, fee, a, z, fee_mult=1.5)[0], a, z, b)["x5"],
            }
        a, z = W["fresh"][0], len(b["t"])
        tr, _ = simulate(b, F, C[en], g, fee, a, z)
        yrs: dict = {}
        for x in tr:
            y = datetime.fromtimestamp(int(b["t"][x["e"]]) / 1000, timezone.utc).year
            yrs.setdefault(y, []).append(x)
        red[name]["by_year"] = {y: {"n": len(v), "x5": float(lev_path(v, LEV)["ret"])} for y, v in sorted(yrs.items())}
    passed = [n_ for n_ in res if res[n_]["pass"]]
    (out / "result.json").write_text(json.dumps(
        {"n_variants": N_VAR, "n_tonight": N_TONIGHT_BEFORE + N_VAR, "windows": {k: [int(b["t"][a]), int(b["t"][z - 1])] for k, (a, z) in W.items()},
         "cands": {k: len(v) for k, v in C.items()}, "ranked_top10": ranked[:10], "checked": check,
         "passed": passed, "res": res, "random": rnd, "redteam": red, "sec": time.time() - t0}, indent=1))  # fmt: skip
    print("passed", passed, "top", ranked[:6], round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
