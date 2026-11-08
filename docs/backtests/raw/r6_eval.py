"""6차 탐색: CME 주말 갭(CG)·상위 시간대 쓸기(SW)·시각 기준점(TA)·일봉/4시간 추세(DT).

문서: docs/backtests/2026-11-08-btc-round6-hypotheses.md (사전 등록 그대로, 변형 128개)
자료: 5차 선물 15분봉(`~/.minerva/research/round5-2026-11-08`), 현물 15분봉 2017-09~2020-02(가격만),
CME 일봉(야후 BTC=F, `~/.minerva/data/btc/cme/btc_f_1d_yahoo.json`).
모든 거래는 자체 모의 매매(`sim`): 신호 봉 마감 → 다음 봉 시가 진입, 손절 먼저, 나눠 익절, 본전,
시간 청산(진입 뒤 tmax봉째 시가). 포지션은 겹치지 않는다.
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/round6-2026-11-08
  P="systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
     PYTHONPATH=$R/pyml:$R/pylib:docs/backtests/raw python3"
  $P docs/backtests/raw/r6_eval.py --out $OUT grid
  $P docs/backtests/raw/r6_eval.py --out $OUT random --nsim 2000
  $P docs/backtests/raw/r6_eval.py --out $OUT check   # 레드팀 자체 점검(1봉 지연·수수료 1.5배·해별)
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pine_eval as E
import pine_ports as P
import pine_tv_eval as T
import tvc_components as K
from r5_data import load as load5

M15 = 900_000
H1 = 3_600_000
DAY = 86_400_000
FEE = 0.0005
DS = 0.07
LIQ = -0.95
R_MIN, R_MAX = 0.003, 0.03
HOME = Path("~/.minerva").expanduser()
R5 = HOME / "research/round5-2026-11-08"
CME = HOME / "data/btc/cme/btc_f_1d_yahoo.json"
CHI = ZoneInfo("America/Chicago")
LON = ZoneInfo("Europe/London")
NYC = ZoneInfo("America/New_York")
WIN = {
    "spot": (E.ms("2017-09-01"), E.ms("2020-03-01")),
    "fresh": (E.ms("2020-03-01"), E.ms("2021-11-01")),
    "design": (E.ms("2022-01-01"), E.ms("2024-07-01")),
    "val": (E.ms("2024-07-01"), E.ms("2025-11-01")),
    "seen": (E.ms("2025-11-01"), None),
}
N_TOTAL = 128
N_TONIGHT = 5254 + N_TOTAL
LEVS = (1, 2, 3, 10)


# ---------------- 모의 매매 ----------------
def _resolve_base(spec: dict, ep: float):
    """진입가가 정해진 뒤 목표·손절 가격. 목표가 손해 쪽이면 spec["bad"]에 따라 건너뛰거나 1R."""
    d = spec["d"]
    sl = None
    k, v = spec["sl"]
    if k == "pct":
        sl = ep * (1 - d * v)
    elif k == "abs":
        sl = v
    elif k == "clamp":
        dist = (ep - v) * d / ep if not math.isnan(v) else R_MAX
        sl = ep * (1 - d * min(max(dist, R_MIN), R_MAX))
    R = abs(ep - sl)
    tps = []
    for f, kk, vv in spec["tps"]:
        if kk == "abs":
            px = vv
            if d * (px - ep) <= 0:
                if spec.get("bad") == "skip":
                    return None
                px = ep + d * R
        elif kk == "pct":
            px = ep * (1 + d * vv)
        elif kk == "dist":
            px = ep + d * vv
        else:  # "R"
            px = ep + d * vv * R
        tps.append((f, px))
    tps.sort(key=lambda x: d * (x[1] - ep))
    return tps, sl


def sim(B: dict, specs: dict, a: int, z: int, fee: float = FEE, delay: int = 0) -> list:
    """specs: {신호 봉 j: spec}. spec = d, tps[(비율, 종류, 값)], sl(종류, 값), be, tmax, bad."""
    o, h, lo, c, t = B["o"], B["h"], B["l"], B["c"], B["t"]
    ft, fr, fo = B["ft"], B["fr"], B["f_open"]
    js = sorted(j for j in specs if a <= j < z)
    trades = []
    busy = -1
    for j in js:
        if j <= busy:
            continue
        e = j + 1 + delay
        if e >= z:
            break
        sp = specs[j]
        d = sp["d"]
        ep = o[e]
        rs = resolve(sp, ep)
        if rs is None:
            continue
        tps, sl = rs
        size, real, worst, parts, last = 1.0, 0.0, -fee, [], ep
        tmax = sp["tmax"]
        xi, why = None, None
        ii = e
        while True:
            if ii >= z:
                xi, xp, why = z - 1, c[z - 1], "end"
                break
            if ii - e >= tmax:
                xi, xp, why = ii, o[ii], "time"
                break
            hit_o = (o[ii] <= sl) if d == 1 else (o[ii] >= sl)
            hit = (lo[ii] <= sl) if d == 1 else (h[ii] >= sl)
            if hit_o or hit:
                xi, xp, why = ii, (o[ii] if hit_o else sl), "stop"
                worst = min(worst, real + size * d * (xp / ep - 1) - fee)
                break
            adverse = lo[ii] if d == 1 else h[ii]
            worst = min(worst, real + size * d * (adverse / ep - 1) - fee)
            done = False
            while tps:
                f, px = tps[0]
                ho = (o[ii] >= px) if d == 1 else (o[ii] <= px)
                hh = (h[ii] >= px) if d == 1 else (lo[ii] <= px)
                if not (ho or hh):
                    break
                fill = o[ii] if ho else px
                tps.pop(0)
                f = min(f, size)
                real += f * (d * (fill / ep - 1) - fee * fill / ep)
                size -= f
                parts.append((int(t[ii]), f))
                last = fill
                if sp.get("be"):
                    sl = max(sl, ep) if d == 1 else min(sl, ep)
                if size <= 1e-9 or not tps:
                    done = True
                    break
            if done:
                xi, xp, why = ii, None, "tp"
                break
            ii += 1
        if xp is not None and size > 1e-9:
            real += size * (d * (xp / ep - 1) - fee * xp / ep)
            parts.append((int(t[xi]), size))
        r = real - fee
        t_in = int(t[e])
        t_out = int(t[xi]) + 1
        fund = 0.0
        ka, kz = bisect.bisect_right(ft, t_in), bisect.bisect_right(ft, t_out)
        for k in range(ka, kz):
            left = 1.0 - sum(f for tt, f in parts if tt < ft[k])
            if left > 1e-9:
                fund += d * fr[k] * left * fo[k] / ep
        r -= fund
        worst = min(worst, r)
        trades.append({"d": d, "j": j, "e": e, "x": xi, "ep": ep, "xp": last if xp is None else xp,
                       "r": r, "worst": worst, "fund": fund, "why": why,
                       "shape": shape_of(sp, ep, resolve(sp, ep))})  # fmt: skip
        busy = xi
    return trades


def shape_of(sp: dict, ep: float, rs) -> dict:
    """무작위 기준용: 진입가 대비 목표·손절 거리(%)와 시간 한도."""
    tps, sl = rs
    d = sp["d"]
    return {"tps": [(f, "pct", d * (px / ep - 1)) for f, px in tps],
            "sl": ("pct", d * (1 - sl / ep)), "be": sp.get("be", False),
            "tmax": sp["tmax"]}  # fmt: skip


def lev_path(tr: list, L: float) -> dict:
    eq, peak, mdd, liq, dead = 1.0, 1.0, 0.0, 0, False
    for x in tr:
        if L * x["worst"] <= LIQ:
            liq += 1
            if not dead:
                mdd, eq, dead = 1.0, 0.0, True
        if dead:
            continue
        mdd = max(mdd, 1 - eq * (1 + L * x["worst"]) / peak)
        eq *= 1 + L * x["r"]
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {"ret": eq - 1, "mdd": mdd, "liq": liq}


def summ(tr: list) -> dict:
    out: dict = {"n": len(tr)}
    for L in LEVS:
        out[f"x{L}"] = lev_path(tr, L)
    if not tr:
        out.update(win=None, avg=None, worst_r=None, mae=None, nl=0, ns=0, be10=None)
        return out
    r = np.array([x["r"] for x in tr])
    out.update(win=float((r > 0).mean()), avg=float(r.mean()), med=float(np.median(r)),
               worst_r=float(r.min()), mae=float(min(x["worst"] for x in tr)),
               nl=int(sum(x["d"] == 1 for x in tr)), ns=int(sum(x["d"] == -1 for x in tr)),
               big_losses=int((r <= -0.02).sum()))  # fmt: skip
    out.update({k: v for k, v in T.be_win(tr).items()})
    why: dict = {}
    for x in tr:
        why[x["why"]] = why.get(x["why"], 0) + 1
    out["why"] = why
    return out


# ---------------- 자료 ----------------
def bars_of(b: dict) -> dict:
    t = b["t"]
    fi = np.minimum(np.searchsorted(t, b["ft"]), len(t) - 1)
    return {"t": t, "o": b["o"].tolist(), "h": b["h"].tolist(), "l": b["l"].tolist(),
            "c": b["c"].tolist(), "ft": b["ft"].tolist(), "fr": b["fr"].tolist(),
            "f_open": b["o"][fi].tolist()}  # fmt: skip


def load_data() -> tuple:
    b = load5(R5)
    keep = ("t", "o", "h", "l", "c", "v", "ft", "fr", "f_f_last")
    b = {k: b[k] for k in keep}
    s = T.spot15()
    s = {k: s[k] for k in ("t", "o", "h", "l", "c", "v", "ft", "fr")}
    cut = np.searchsorted(s["t"], WIN["spot"][1])
    for k in ("t", "o", "h", "l", "c", "v"):
        s[k] = s[k][:cut]
    s["f_f_last"] = np.full(cut, np.nan)
    return b, s


def windows(t: np.ndarray, spot: bool) -> dict:
    w = {}
    names = ("spot",) if spot else ("fresh", "design", "val", "seen")
    for k in names:
        a, z = WIN[k]
        w[k] = (int(np.searchsorted(t, a)), len(t) if z is None else int(np.searchsorted(t, z)))
    if not spot:
        w["last30"] = (int(np.searchsorted(t, t[-1] + M15 - 30 * DAY)), len(t))
    return w


def idx_of(t: np.ndarray, ms: int) -> int:
    k = int(np.searchsorted(t, ms))
    return k if k < len(t) and int(t[k]) == ms else -1


def utc_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def period_hl(t: np.ndarray, h, lo, o, pid: np.ndarray) -> tuple:
    """봉마다 직전 기간(바로 앞 id)의 고·저와 이번 기간 시가. 직전 기간이 비면 nan."""
    keys, idx = np.unique(pid, return_index=True)
    ends = np.r_[idx[1:], len(pid)]
    ph = np.maximum.reduceat(h, idx)
    pl = np.minimum.reduceat(lo, idx)
    po = o[idx]
    pos = np.searchsorted(keys, pid)
    n = len(t)
    prev_h = np.full(n, np.nan)
    prev_l = np.full(n, np.nan)
    ok = pos > 0
    ok_prev = ok.copy()
    ok_prev[ok] = keys[pos[ok] - 1] == pid[ok] - 1
    prev_h[ok_prev] = ph[pos[ok_prev] - 1]
    prev_l[ok_prev] = pl[pos[ok_prev] - 1]
    cur_o = po[pos]
    full_prev = np.zeros(n, bool)
    span = ends - idx
    full_prev[ok_prev] = span[pos[ok_prev] - 1] >= 0.95 * np.median(span)
    prev_h[~full_prev] = np.nan
    prev_l[~full_prev] = np.nan
    return prev_h, prev_l, cur_o, pos, idx


# ---------------- 신호 → spec ----------------
def mk(d, tps, sl, tmax, be=False, bad=None) -> dict:
    return {"d": int(d), "tps": tps, "sl": sl, "tmax": tmax, "be": be, "bad": bad}


def fam_cg(b: dict, atr: np.ndarray) -> dict:
    t, h, lo, c = b["t"], b["h"], b["l"], b["c"]
    r = json.loads(CME.read_text())["chart"]["result"][0]
    ts, q = r["timestamp"], r["indicators"]["quote"][0]
    rows = []
    for k in range(len(ts) - 1):
        d0 = datetime.fromtimestamp(ts[k], NYC).date()
        d1 = datetime.fromtimestamp(ts[k + 1], NYC).date()
        if d0.weekday() != 4 or (d1 - d0).days != 3:
            continue
        fc, mo = q["close"][k], q["open"][k + 1]
        if fc is None or mo is None:
            continue
        fri_close = utc_ms(datetime(d0.year, d0.month, d0.day, 16, 0, tzinfo=CHI))
        sun = d0 + timedelta(days=2)
        reopen = utc_ms(datetime(sun.year, sun.month, sun.day, 17, 0, tzinfo=CHI))
        rows.append((mo / fc - 1, fri_close, reopen))
    out: dict = {}
    gaps = []
    for g, fcl, ro in rows:
        jf = idx_of(t, fcl - M15)
        j = idx_of(t, ro)
        if jf < 0 or j < 0:
            continue
        F = c[jf]
        wh, wl = h[jf + 1 : j + 1].max(), lo[jf + 1 : j + 1].min()
        gaps.append((j, g, F, wh, wl))
    for th in (0.005, 0.01, 0.02):
        tag = f"{th * 100:g}"
        for ex in ("E1", "E2", "E3"):
            out[f"CG{tag}|{ex}"] = {}
        out[f"CGF{tag}|F"] = {}
        for j, g, F, wh, wl in gaps:
            if abs(g) < th:
                continue
            d = -1 if g > 0 else 1
            out[f"CG{tag}|E1"][j] = mk(d, [(1.0, "abs", F)], ("pct", DS), 480, bad="skip")
            out[f"CG{tag}|E2"][j] = mk(d, [(1.0, "half", F)], ("pct", DS), 480, bad="skip")
            st = (wh + 0.25 * atr[j]) if d == -1 else (wl - 0.25 * atr[j])
            out[f"CG{tag}|E3"][j] = mk(d, [(1.0, "abs", F)], ("clamp", st), 480, bad="skip")
            out[f"CGF{tag}|F"][j] = mk(-d, [(1.0, "pct", abs(g))], ("pct", DS), 480)
    return out, len(gaps)


def fam_sw(b: dict, atr: np.ndarray, up: np.ndarray, dn: np.ndarray) -> dict:
    t, o, h, lo, c = b["t"], b["o"], b["h"], b["l"], b["c"]
    n = len(c)
    out: dict = {}
    pids = {"PD": t // DAY, "PW": (t // DAY + 3) // 7}
    for lv, pid in pids.items():
        ph, pl, po, pos, _ = period_hl(t, h, lo, o, pid)
        mid = (ph + pl) / 2
        tmax = 96 if lv == "PD" else 192
        ev = {"C0": [], "C1": []}
        start = np.r_[True, pid[1:] != pid[:-1]]
        i = 0
        hl, ll, cl = h.tolist(), lo.tolist(), c.tolist()
        while i < n:
            if not start[i]:
                i += 1
                continue
            k = i + 1
            while k < n and not start[k]:
                k += 1
            if not math.isnan(ph[i]):
                H, L = ph[i], pl[i]
                for side in (1, -1):
                    k0 = next(
                        (m for m in range(i, k) if (ll[m] < L if side == 1 else hl[m] > H)), None
                    )
                    if k0 is None:
                        continue
                    lvl = L if side == 1 else H
                    back = [side * (cl[m] - lvl) > 0 for m in range(k0, min(k0 + 4, k))]
                    if back[0]:
                        ev["C0"].append((k0, side, k0))
                    m1 = next((k0 + q for q, ok in enumerate(back) if ok), None)
                    if m1 is not None:
                        ev["C1"].append((m1, side, k0))
            i = k
        for cf, evs in ev.items():
            for ex in ("T1", "T2", "T3", "T4"):
                for g in ("G0", "GT"):
                    key = f"SW{lv}{cf}|{ex}|{g}"
                    out[key] = {}
                    for m, d, k0 in evs:
                        if g == "GT" and not (up[m] if d == 1 else dn[m]):
                            continue
                        ext = min(ll[k0 : m + 1]) if d == 1 else max(hl[k0 : m + 1])
                        st = ext - d * 0.25 * atr[m]
                        if ex == "T1":
                            tps = [(1.0, "abs", mid[m])]
                        elif ex == "T2":
                            tps = [(1.0, "abs", po[m])]
                        elif ex == "T3":
                            tps = [(1.0, "R", 1.0)]
                        else:
                            tps = [(1 / 3, "R", 1.0), (1 / 3, "R", 2.0), (1 / 3, "R", 3.0)]
                        out[key][m] = mk(d, tps, ("clamp", st), tmax, be=ex == "T4", bad="R1")
    return out


def fam_ta(b: dict, atr: np.ndarray, design: tuple) -> tuple:
    t, h, lo, c = b["t"], b["h"], b["l"], b["c"]
    n = len(c)
    out: dict = {}
    thr = {}
    # C1 펀딩
    f = b["f_f_last"]
    has_f = not np.all(np.isnan(f))
    if has_f:
        a, z = design
        af = np.abs(f[a:z])
        thr["fund_q90"] = float(np.nanquantile(af, 0.9))
        for kind in ("pre", "post", "post4"):
            for g in ("all", "ext"):
                out[f"TAF{kind}|{g}"] = {}
        for T_ in b["ft"]:
            T_ = (int(T_) + 30_000) // 60_000 * 60_000  # 정산 시각에 몇 ms가 붙어 온다
            for kind, off, tm in (("pre", -75, 4), ("post", -15, 4), ("post4", -15, 16)):
                j = idx_of(t, T_ + off * 60_000)
                if j < 0 or math.isnan(f[j]) or f[j] == 0:
                    continue
                d = -int(np.sign(f[j])) if kind == "pre" else int(np.sign(f[j]))
                sp = mk(d, [], ("pct", DS), tm)
                out[f"TAF{kind}|all"][j] = sp
                if abs(f[j]) >= thr["fund_q90"]:
                    out[f"TAF{kind}|ext"][j] = sp
    # C2 만기
    for per in ("W", "M"):
        for mode in ("fade", "follow"):
            for ex in ("24h", "X2"):
                out[f"TAX{per}{mode}|{ex}"] = {}
    day0 = int(t[0] // DAY)
    for dd in range(day0, int(t[-1] // DAY) + 1):
        dt = datetime.fromtimestamp(dd * 86400, timezone.utc)
        if dt.weekday() != 4:
            continue
        j = idx_of(t, dd * DAY + 7 * H1 + 45 * 60_000)
        if j < 96:
            continue
        r24 = c[j] / c[j - 96] - 1
        if r24 == 0:
            continue
        monthly = (dt + timedelta(days=7)).month != dt.month
        for per in ("W", "M"):
            if per == "M" and not monthly:
                continue
            for mode in ("fade", "follow"):
                d = -int(np.sign(r24)) if mode == "fade" else int(np.sign(r24))
                out[f"TAX{per}{mode}|24h"][j] = mk(d, [], ("pct", DS), 96)
                out[f"TAX{per}{mode}|X2"][j] = mk(d, [(1.0, "dist", 0.5 * atr[j])], ("pct", DS), 96)
    # C3 주말 범위
    for mode in ("brk", "fail"):
        for ex in ("XA", "XB", "XC"):
            out[f"TAW{mode}|{ex}"] = {}
    for dd in range(day0, int(t[-1] // DAY) + 1):
        dt = datetime.fromtimestamp(dd * 86400, timezone.utc)
        if dt.weekday() != 0:
            continue
        a_ = idx_of(t, (dd - 2) * DAY)
        m0 = idx_of(t, dd * DAY)
        if a_ < 0 or m0 < 0 or m0 - a_ != 192:
            continue
        H, L = h[a_:m0].max(), lo[a_:m0].min()
        mid = (H + L) / 2
        brk = fail = None
        for m in range(m0, min(m0 + 96, n)):
            if brk is None and (c[m] > H or c[m] < L):
                brk = (m, 1 if c[m] > H else -1)
            if fail is None:
                if h[m] > H and c[m] < H:
                    fail = (m, -1, h[m])
                elif lo[m] < L and c[m] > L:
                    fail = (m, 1, lo[m])
        if brk:
            m, d = brk
            out["TAWbrk|XA"][m] = mk(d, [(1.0, "dist", atr[m])], ("pct", DS), 96)
            out["TAWbrk|XB"][m] = mk(d, [(1.0, "R", 1.0)], ("clamp", mid), 96)
            out["TAWbrk|XC"][m] = mk(d, [(1 / 3, "R", 1.0), (1 / 3, "R", 2.0), (1 / 3, "R", 3.0)],
                                     ("clamp", mid), 192, be=True)  # fmt: skip
        if fail:
            m, d, ext = fail
            st = ext - d * 0.25 * atr[m]
            out["TAWfail|XA"][m] = mk(d, [(1.0, "dist", atr[m])], ("pct", DS), 96)
            out["TAWfail|XB"][m] = mk(d, [(1.0, "R", 1.0)], ("clamp", st), 96)
            out["TAWfail|XC"][m] = mk(d, [(1 / 3, "R", 1.0), (1 / 3, "R", 2.0), (1 / 3, "R", 3.0)],
                                      ("clamp", st), 192, be=True)  # fmt: skip
    # C4 세션 시가 범위(평일만)
    sess = {"AS": (timezone.utc, 0, 0), "LN": (LON, 8, 0), "NY": (NYC, 9, 30)}
    for s_ in sess:
        for rng in (30, 60):
            for mode in ("brk", "fail"):
                for ex in ("R1", "A1"):
                    out[f"TAO{s_}{rng}{mode}|{ex}"] = {}
    for dd in range(day0, int(t[-1] // DAY) + 1):
        dt = datetime.fromtimestamp(dd * 86400, timezone.utc)
        if dt.weekday() >= 5:
            continue
        for s_, (tz, hh, mm) in sess.items():
            st_ms = utc_ms(datetime(dt.year, dt.month, dt.day, hh, mm, tzinfo=tz))
            s0 = idx_of(t, st_ms)
            if s0 < 0:
                continue
            for rng in (30, 60):
                k = rng // 15
                if s0 + k + 16 > n or int(t[s0 + k]) - int(t[s0]) != k * M15:
                    continue
                H, L = h[s0 : s0 + k].max(), lo[s0 : s0 + k].min()
                brk = fail = None
                for m in range(s0 + k, s0 + k + 16):
                    if brk is None and (c[m] > H or c[m] < L):
                        brk = (m, 1 if c[m] > H else -1)
                    if fail is None:
                        if h[m] > H and c[m] < H:
                            fail = (m, -1, h[m])
                        elif lo[m] < L and c[m] > L:
                            fail = (m, 1, lo[m])
                if brk:
                    m, d = brk
                    st = L if d == 1 else H
                    out[f"TAO{s_}{rng}brk|R1"][m] = mk(d, [(1.0, "R", 1.0)], ("clamp", st), 96)
                    out[f"TAO{s_}{rng}brk|A1"][m] = mk(
                        d, [(1.0, "dist", atr[m])], ("clamp", st), 96
                    )
                if fail:
                    m, d, ext = fail
                    st = ext - d * 0.25 * atr[m]
                    out[f"TAO{s_}{rng}fail|R1"][m] = mk(d, [(1.0, "R", 1.0)], ("clamp", st), 96)
                    out[f"TAO{s_}{rng}fail|A1"][m] = mk(
                        d, [(1.0, "dist", atr[m])], ("clamp", st), 96
                    )
    return out, thr, has_f


def frame(b: dict, minutes: int) -> dict:
    """완결된 상위 봉만. last15 = 그 봉의 마지막 15분봉 인덱스."""
    step = minutes * 60_000
    g = b["t"] // step
    keys, idx = np.unique(g, return_index=True)
    ends = np.r_[idx[1:], len(g)]
    full = (ends - idx) == minutes // 15
    idx, ends = idx[full], ends[full]
    return {"o": b["o"][idx], "h": np.array([b["h"][a:z].max() for a, z in zip(idx, ends)]),
            "l": np.array([b["l"][a:z].min() for a, z in zip(idx, ends)]),
            "c": b["c"][ends - 1], "last15": ends - 1}  # fmt: skip


def htf_sigs(b: dict, daily_src: dict | None) -> dict:
    """일봉·4시간 신호를 15분봉 마지막 봉에 붙인다. 일봉은 현물+선물 이은 일봉으로 계산(데우기)."""
    n = len(b["c"])
    D = P.daily(b)
    out = {}
    if daily_src is not None:
        # daily_src: 이전(현물) 일봉. 겹치지 않는 날만 앞에 붙인다.
        keep = daily_src["day"] < D["day"][0]
        cat = {k: np.r_[daily_src[k][keep], D[k]] for k in ("h", "l", "c")}
        off = int(keep.sum())
    else:
        cat = {k: D[k] for k in ("h", "l", "c")}
        off = 0

    def to15(sig_cat):
        s = np.asarray(sig_cat)[off:]
        return P.daily_to_15(D, s.astype(np.int8), n)

    out["6D"] = to15(P.s6_trend_dev(cat["h"], cat["l"], cat["c"]))
    out["3D"] = to15(P.s3_gauss_core(cat["h"], cat["l"], cat["c"])["sig"])
    hh = P.shift(P.highest(cat["h"], 20), 1)
    ll = P.shift(P.lowest(cat["l"], 20), 1)
    with np.errstate(invalid="ignore"):
        out["DON20"] = to15(K.sig_from(cat["c"] > hh, cat["c"] < ll))
    datr = P.atr(cat["h"], cat["l"], cat["c"], 14)[off:]
    datr15 = P.daily_state_15(D, datr, n)
    F = frame(b, 240)
    d4 = P.supertrend_dir(F["h"], F["l"], F["c"], 3.0, 10)
    s4 = np.zeros(len(d4), np.int8)
    s4[1:][(d4[1:] < 0) & (d4[:-1] > 0)] = 1
    s4[1:][(d4[1:] > 0) & (d4[:-1] < 0)] = -1
    e20, e50 = P.ema(F["c"], 20), P.ema(F["c"], 50)
    with np.errstate(invalid="ignore"):
        x4 = K.sig_from(P.cross_over(e20, e50), P.cross_under(e20, e50))
    for name, s in (("ST4", s4), ("EMA4", x4)):
        a = np.zeros(n, np.int8)
        a[F["last15"]] = s
        out[name] = a
    return out, datr15


def fam_dt(b: dict, daily_src: dict | None) -> dict:
    h, lo, c, v = b["h"], b["l"], b["c"], b["v"]
    n = len(c)
    sigs, datr = htf_sigs(b, daily_src)
    z15 = K.vwz(h, lo, c, v)
    out = {}
    exits = {
        "SO1": ([(0.5, 1.0), (0.5, 3.0)], 1.5, False, 1920),
        "SO2": ([(1 / 3, 0.5), (1 / 3, 1.0), (1 / 3, 2.0)], 1.0, True, 960),
        "SO3": ([(0.5, 0.5), (0.5, 2.0)], 2.0, True, 1920),
        "SO4": ([(1.0, 0.25)], 2.0, False, 960),
    }
    for name, s in sigs.items():
        js = np.flatnonzero(s)
        for ent in ("N", "PB"):
            ev = []
            for j in js:
                d = int(s[j])
                if ent == "N":
                    ev.append((j, d, j))
                else:
                    zz = z15[j + 1 : min(j + 97, n)]
                    ok = np.flatnonzero((zz <= -1) if d == 1 else (zz >= 1))
                    if len(ok):
                        ev.append((j + 1 + int(ok[0]), d, j))
            for ex, (lad, slk, be, tm) in exits.items():
                key = f"DT{name}{ent}|{ex}"
                out[key] = {}
                for m, d, j in ev:
                    A = datr[j]
                    if math.isnan(A):
                        continue
                    out[key][m] = mk(d, [(f, "dist", k * A) for f, k in lad],
                                     ("dist_sl", slk * A), tm, be=be)  # fmt: skip
    return out


def resolve(spec: dict, ep: float):
    """ "half"(진입가와 F의 중간)·"dist_sl"(진입가 − 거리)을 바꾼 뒤 _resolve_base."""
    d = spec["d"]
    if spec["sl"][0] == "dist_sl":
        spec = dict(spec, sl=("abs", ep - d * spec["sl"][1]))
    if spec["tps"] and spec["tps"][0][1] == "half":
        F = spec["tps"][0][2]
        if d * (F - ep) <= 0:
            return None
        spec = dict(spec, tps=[(1.0, "abs", (ep + F) / 2)])
    return _resolve_base(spec, ep)


# ---------------- 조립 ----------------
def build_all(b: dict, design: tuple | None, daily_src: dict | None = None) -> tuple:
    h, lo, c = b["h"], b["l"], b["c"]
    atr = P.atr(h, lo, c, 14)
    D = P.daily(b)
    sma50 = P.daily_state_15(D, P.sma(D["c"], 50), len(c))
    dcl = P.daily_state_15(D, D["c"], len(c))
    with np.errstate(invalid="ignore"):
        up = ~np.isnan(sma50) & (dcl > sma50)
        dn = ~np.isnan(sma50) & (dcl < sma50)
    S: dict = {}
    cg, ngap = fam_cg(b, atr)
    S.update(cg)
    S.update(fam_sw(b, atr, up, dn))
    ta, thr, _ = fam_ta(b, atr, design or (0, len(c)))
    S.update(ta)
    S.update(fam_dt(b, daily_src))
    thr["cme_weeks"] = ngap
    return S, thr


def passes(r: dict) -> dict:
    d, v, f = r["design"], r["val"], r["fresh"]
    return {
        "win70": (v["win"] or 0) >= 0.70, "x10_pos": v["x10"]["ret"] > 0,
        "liq0_val": v["x10"]["liq"] == 0, "liq0_design": d["x10"]["liq"] == 0,
        "n30": v["n"] >= 30, "design1_pos": d["x1"]["ret"] > 0,
        "fresh_ok": ((f["win"] or 0) >= 0.70 and f["x10"]["ret"] > 0 and f["x10"]["liq"] == 0
                     and f["n"] >= 20),
    }  # fmt: skip


def strip(tr: list) -> list:
    return [{k: x[k] for k in ("d", "j", "e", "x", "ep", "r", "worst", "why")} for x in tr]


def setup():
    b, s = load_data()
    W = windows(b["t"], False)
    Ws = windows(s["t"], True)
    Ds = P.daily(s)
    S, thr = build_all(b, W["design"], Ds)
    Ss, _ = build_all(s, None, None)
    return b, s, W, Ws, S, Ss, thr


def cmd_grid(out: Path) -> None:
    t0 = time.time()
    b, s, W, Ws, S, Ss, thr = setup()
    B, Bs = bars_of(b), bars_of(s)
    keys = sorted(S)
    print("built", len(keys), round(time.time() - t0, 1), flush=True)
    assert len(keys) == N_TOTAL, len(keys)
    res = {}
    for k in keys:
        r = {w: summ(sim(B, S[k], *W[w])) for w in ("fresh", "design", "val", "seen", "last30")}
        r["spot"] = summ(sim(Bs, Ss.get(k, {}), *Ws["spot"]))
        r["pass"] = {kk: bool(x) for kk, x in passes(r).items()}
        r["pass_all"] = bool(all(r["pass"].values()))
        r["sig_n"] = {w: int(sum(1 for j in S[k] if W[w][0] <= j < W[w][1])) for w in W}
        res[k] = r
    pool = [(k, r) for k, r in res.items() if r["val"]["n"] >= 30
            and r["val"]["x10"]["liq"] == 0 and r["design"]["x10"]["liq"] == 0]  # fmt: skip
    pool.sort(key=lambda x: ((x[1]["val"]["win"] or 0), x[1]["val"]["x10"]["ret"]), reverse=True)
    top_win = [k for k, _ in pool[:2]]
    pool.sort(key=lambda x: (x[1]["val"]["x10"]["ret"], x[1]["val"]["x1"]["ret"]), reverse=True)
    top_ret = [k for k, _ in pool[:2]]
    # 청산 0 조건 없이 검증 승률 상위(구조 설명용)
    allw = sorted(((k, r) for k, r in res.items() if r["val"]["n"] >= 30),
                  key=lambda x: (x[1]["val"]["win"] or 0), reverse=True)  # fmt: skip
    cand = [k for k, r in res.items() if r["pass_all"]]
    (out / "grid.json").write_text(json.dumps(
        {"variants_evaluated": len(res), "thresholds": thr, "candidates": cand,
         "top2_val_win": top_win, "top2_val_x10": top_ret,
         "top5_val_win_any": [k for k, _ in allw[:5]], "sec": time.time() - t0, "res": res},
        indent=1))  # fmt: skip
    print("candidates", cand, "top", top_win, top_ret, [k for k, _ in allw[:5]],
          round(time.time() - t0), flush=True)  # fmt: skip


def random_specs(rng, tr: list, a: int, z: int) -> dict:
    out = {}
    picks = rng.choice(np.arange(a, z - 1), len(tr), replace=False)
    for p, x in zip(picks, tr):
        sh = x["shape"]
        out[int(p)] = {"d": x["d"], "tps": sh["tps"], "sl": sh["sl"], "tmax": sh["tmax"],
                       "be": sh["be"], "bad": None}  # fmt: skip
    return out


def cmd_random(out: Path, nsim: int, items: list | None) -> None:
    b, s, W, Ws, S, Ss, thr = setup()
    B = bars_of(b)
    g = json.loads((out / "grid.json").read_text())
    if not items:
        items = list(dict.fromkeys(g["candidates"] + g["top2_val_win"] + g["top2_val_x10"]))
    rng = np.random.default_rng(20261108)
    res = {}
    for it in items:
        ow = {}
        for w in ("fresh", "design", "val"):
            a, z = W[w]
            tr = sim(B, S[it], a, z)
            real = summ(tr)
            if real["n"] == 0:
                continue
            r1, wn, r10 = [], [], []
            for _ in range(nsim):
                rs = summ(sim(B, random_specs(rng, tr, a, z), a, z))
                r1.append(rs["x1"]["ret"])
                r10.append(rs["x10"]["ret"])
                wn.append(rs["win"] or 0.0)
            r1a, wna = np.array(r1), np.array(wn)
            z1 = (real["x1"]["ret"] - r1a.mean()) / (r1a.std() + 1e-12)
            pn1 = 0.5 * math.erfc(z1 / math.sqrt(2))
            ow[w] = {"n": real["n"], "win": real["win"], "real1": real["x1"]["ret"],
                     "real10": real["x10"]["ret"], "rand1_mean": float(r1a.mean()),
                     "rand10_median": float(np.median(r10)), "rand_win_mean": float(wna.mean()),
                     "p1": float((r1a >= real["x1"]["ret"]).mean()),
                     "pwin": float((wna >= (real["win"] or 0)).mean()),
                     "p1_normal": pn1, "p1_bonf": min(1.0, pn1 * N_TOTAL)}  # fmt: skip
        res[it] = ow
        print(it, {w: (v["n"], round(v["win"], 3), round(v["p1"], 4), round(v["p1_bonf"], 3))
                   for w, v in ow.items()}, flush=True)  # fmt: skip
    (out / "random.json").write_text(json.dumps(res, indent=1))


def cmd_check(out: Path, items: list | None) -> None:
    """레드팀 자체 점검: 1봉 지연 진입, 수수료 1.5배, 해별 1배·10배."""
    b, s, W, Ws, S, Ss, thr = setup()
    B = bars_of(b)
    g = json.loads((out / "grid.json").read_text())
    if not items:
        items = list(dict.fromkeys(g["candidates"] + g["top2_val_win"] + g["top2_val_x10"]))
    res = {}
    for it in items:
        r = {}
        for w in ("fresh", "design", "val"):
            a, z = W[w]
            r[w] = {"delay1": summ(sim(B, S[it], a, z, delay=1)),
                    "fee15": summ(sim(B, S[it], a, z, fee=FEE * 1.5))}  # fmt: skip
        yr = {}
        a0, z0 = W["fresh"][0], len(b["t"])
        tr = sim(B, S[it], a0, z0)
        for x in tr:
            y = datetime.fromtimestamp(int(b["t"][x["e"]]) / 1000, timezone.utc).year
            yr.setdefault(y, []).append(x)
        r["years"] = {y: {"n": len(v), "win": summ(v)["win"], "x1": summ(v)["x1"]["ret"],
                          "x3": summ(v)["x3"]["ret"], "x10": summ(v)["x10"]["ret"],
                          "liq10": summ(v)["x10"]["liq"], "worst": summ(v)["worst_r"]}
                      for y, v in sorted(yr.items())}  # fmt: skip
        res[it] = r
        print(it, json.dumps({w: {k: (v["n"], v["win"], round(v["x10"]["ret"], 3))
                                  for k, v in r[w].items()} for w in ("fresh", "design", "val")}),
              flush=True)  # fmt: skip
    (out / "check.json").write_text(json.dumps(res, indent=1, default=str))


def trades_for(it: str, w: str):
    """차트용: (b, 거래, a, z)."""
    b, s, W, Ws, S, Ss, thr = setup()
    if w == "spot":
        return s, sim(bars_of(s), Ss[it], *Ws["spot"]), *Ws["spot"]
    return b, sim(bars_of(b), S[it], *W[w]), *W[w]


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
