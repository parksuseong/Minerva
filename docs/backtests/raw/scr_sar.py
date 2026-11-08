"""SCR 실험실 신호 + 손절 없는 롱↔숏 뒤집기(SAR), 15분봉. 격자 7,776개 + 걷기 전진.

등록: docs/backtests/2026-11-08-btc-scr-sar.md. 연구 전용(주문 없음, 바이낸스 POST 없음, 공개 GET klines 꼬리만).
신호 계산은 src/minerva/strategy/scr_lab.py(= static/scrlab_core.js)를 numpy로 옮긴 것(--check로 맞춰 본다).
실행:
  R=~/.minerva/research/btc-2026-11-04; OUT=~/.minerva/research/scr-sar-2026-11-08
  systemd-run --user --scope -p MemoryMax=1200M nice -n 10 env \
    PYTHONPATH=$R/pylib:docs/backtests/raw python3 docs/backtests/raw/scr_sar.py --out $OUT [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pine_div as P

MIN, DAY = 60_000, 86_400_000
B15 = 15 * MIN
TAKER, MMR = 0.0005, 0.004
N_GRID = 7776
N_CUM = 6745 + N_GRID + 2
LEVS = (1, 3, 5)
BASE = {"stoch_n": 9, "stoch_lo": 20, "stoch_hi": 80, "cci_n": 20, "cci_lo": -100, "cci_hi": 100,
        "rsi_n": 21, "rsi_lo": 30, "rsi_hi": 80, "w": 3, "need": 3}  # fmt: skip
G_ST_N, G_ST_L = (5, 9, 14), ((15, 85), (20, 80), (25, 75))
G_CC_N, G_CC_L = (14, 20, 30), ((-80, 80), (-100, 100), (-150, 150))
G_RS_N, G_RS_L = (9, 14, 21, 28), ((25, 75), (30, 70), (30, 80), (35, 65))
G_W, G_NEED = (1, 3, 5), (2, 3)
LAB_SCR = Path("/home/ec2-user/work/Minerva/src/minerva/strategy/scr_lab.py")


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def ds(t) -> str:
    return datetime.fromtimestamp(int(t) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------- 자료 ----------------
def bars(m: dict, minutes: int, now: int) -> dict:
    k = m["t"] // (minutes * MIN)
    st = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    en = np.r_[st[1:] - 1, len(k) - 1]
    F = {"t": k[st] * minutes * MIN, "o": m["o"][st], "c": m["c"][en],
         "h": np.maximum.reduceat(m["h"], st), "l": np.minimum.reduceat(m["l"], st)}  # fmt: skip
    ok = F["t"] + minutes * MIN <= now
    F = {x: F[x][ok] for x in F}
    F["ct"] = F["t"] + minutes * MIN
    return F


# ---------------- 지표 (scr_lab.py와 같은 정의) ----------------
def stoch(h, l, c, n):  # noqa: E741
    hh = np.lib.stride_tricks.sliding_window_view(h, n).max(1)
    ll = np.lib.stride_tricks.sliding_window_view(l, n).min(1)
    out = np.full(len(c), np.nan)
    rng = hh - ll
    out[n - 1 :] = np.where(rng == 0, 50.0, (c[n - 1 :] - ll) / np.where(rng == 0, 1, rng) * 100)
    return out


def cci(h, l, c, n):  # noqa: E741
    tp = (h + l + c) / 3
    w = np.lib.stride_tricks.sliding_window_view(tp, n)
    m = w.mean(1)
    md = np.abs(w - m[:, None]).mean(1)
    out = np.full(len(c), np.nan)
    out[n - 1 :] = np.where(md == 0, 0.0, (tp[n - 1 :] - m) / (0.015 * np.where(md == 0, 1, md)))
    return out


def rsi(c, n):
    out = np.full(len(c), np.nan)
    d = np.diff(c)
    g, ls = np.maximum(d, 0).tolist(), np.maximum(-d, 0).tolist()
    ag, al = sum(g[:n]) / n, sum(ls[:n]) / n
    o = out.tolist()
    o[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + ls[i]) / n
        o[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return np.array(o, dtype=float)


def crossed(x, line, up):
    p, v = x[:-1], x[1:]
    f = (p <= line) & (line < v) if up else (p >= line) & (line > v)
    return np.r_[False, f]


def recent(f, w):
    cs = np.cumsum(np.r_[0, f.astype(np.int32)])
    i = np.arange(len(f))
    return (cs[i + 1] - cs[np.maximum(0, i - w + 1)]) > 0


def first_on(on):
    return on & ~np.r_[False, on[:-1]]


class Ind:
    """지표·교차 캐시."""

    def __init__(self, F):
        self.F, self.ind, self.rec = F, {}, {}

    def x(self, kind, n):
        key = (kind, n)
        if key not in self.ind:
            F = self.F
            self.ind[key] = (stoch(F["h"], F["l"], F["c"], n) if kind == "s" else
                             cci(F["h"], F["l"], F["c"], n) if kind == "c" else rsi(F["c"], n))  # fmt: skip
        return self.ind[key]

    def r(self, kind, n, line, up, w):
        key = (kind, n, line, up, w)
        if key not in self.rec:
            self.rec[key] = recent(crossed(self.x(kind, n), line, up), w).astype(np.uint8)
        return self.rec[key]

    def sides(self, p):
        w, need = p["w"], p["need"]
        out = []
        for up in (True, False):
            k = "lo" if up else "hi"
            s = (self.r("s", p["stoch_n"], p[f"stoch_{k}"], up, w) + self.r("c", p["cci_n"], p[f"cci_{k}"], up, w)
                 + self.r("r", p["rsi_n"], p[f"rsi_{k}"], up, w))  # fmt: skip
            out.append(first_on(s >= need))
        return out  # long, short (첫 봉)

    def sig(self, p):
        lo, sh = self.sides(p)
        return np.where(lo & ~sh, 1, np.where(sh & ~lo, -1, 0)).astype(np.int8)


# ---------------- 15분봉 SAR (격자·걷기 전진·무작위) ----------------
class Book:
    def __init__(self, F, ft, fr):
        self.F = F
        n = len(F["t"])
        fund = np.zeros(n)
        j = np.searchsorted(F["t"], ft)
        ok = (j < n) & (F["t"][np.minimum(j, n - 1)] == ft)
        np.add.at(fund, j[ok], fr[ok] * F["o"][j[ok]])
        self.FP = np.r_[0.0, np.cumsum(fund)]
        self.n = n

    def flips(self, sig, a_bar, z_bar):
        """[a_bar, z_bar) 안의 신호 봉에서 방향이 바뀌는 것만(첫 신호 포함). 체결 봉 = 신호 봉 + 1 < z_bar."""
        idx = np.nonzero(sig[a_bar:z_bar - 1])[0] + a_bar
        if len(idx) == 0:
            return idx, idx
        d = sig[idx]
        keep = np.r_[True, d[1:] != d[:-1]]
        return idx[keep], d[keep]

    def trades(self, fb, fd, z_bar, fee=TAKER):
        """체결 = 신호 봉 다음 봉 시가. 마지막은 z_bar 시가(없으면 마지막 종가)에 청산."""
        F = self.F
        if len(fb) == 0:
            return None
        e = fb + 1
        x = np.r_[e[1:], z_bar]
        pe = F["o"][e]
        px = np.where(x < self.n, F["o"][np.minimum(x, self.n - 1)], F["c"][self.n - 1])
        fund = self.FP[np.minimum(x, self.n - 1) + 1] - self.FP[e + 1]
        rr = px / pe
        r = fd * (rr - 1) - fee * (1 + rr) - fd * fund / pe
        lo = np.minimum.reduceat(F["l"], e)
        hi = np.maximum.reduceat(F["h"], e)
        xl = min(x[-1], self.n)
        lo[-1], hi[-1] = F["l"][e[-1]:xl].min(), F["h"][e[-1]:xl].max()
        adv = np.where(fd > 0, 1 - lo / pe, hi / pe - 1)
        return {"e": e, "x": x, "d": fd, "r": r, "adv": adv, "Te": F["t"][e],
                "Tx": np.where(x < self.n, F["t"][np.minimum(x, self.n - 1)], F["ct"][-1])}  # fmt: skip


def liq_thr(L, d, fee=TAKER):
    base = 1 / L - fee - MMR
    return np.where(d > 0, base / (1 - MMR), base / (1 + MMR))


def path(r, adv, d, L, fee=TAKER):
    """L배 복리, 거래 중 최악가 포함 최대 낙폭, 청산 수(청산이면 계좌 0)."""
    liq = adv >= liq_thr(L, d, fee)
    eq, peak, mdd, dead = 1.0, 1.0, 0.0, False
    for ri, ai, li in zip(r.tolist(), adv.tolist(), liq.tolist()):
        if dead:
            break
        low = eq * max(0.0, 1 - L * (ai + fee))
        mdd = max(mdd, 1 - low / peak)
        if li:
            eq, dead = 0.0, True
            mdd = 1.0
            break
        eq *= max(0.0, 1 + L * ri)
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {"ret": eq - 1, "mdd": mdd, "liq": int(liq.sum())}


# ---------------- 정밀 모의(1분 체결, 추격 제한) ----------------
class Sim:
    def __init__(self, m, F, F4, Fd, ft, fr, ind):
        self.m, self.F, self.ft, self.fr, self.ind = m, F, ft, fr, ind
        jf = np.searchsorted(m["t"], ft)
        self.fp = m["o"][np.minimum(jf, len(m["t"]) - 1)]
        dir15 = self.dirs(F)
        d4, dd = self.dirs(F4), self.dirs(Fd)
        i4 = np.searchsorted(F4["ct"], F["ct"], "right") - 1
        idd = np.searchsorted(Fd["ct"], F["ct"], "right") - 1
        self.agree = np.where((dir15 == np.where(i4 >= 0, d4[i4], 0)) & (dir15 == np.where(idd >= 0, dd[idd], 0)),
                              dir15, 0)  # fmt: skip

    @staticmethod
    def dirs(F):
        c, r = cci(F["h"], F["l"], F["c"], 20), rsi(F["c"], 14)
        return np.where((c > 0) & (r > 50), 1, np.where((c < 0) & (r < 50), -1, 0))

    def px(self, T):
        j = np.searchsorted(self.m["t"], T)
        return j, self.m["o"][min(j, len(self.m["t"]) - 1)]

    def run(self, sig, a, z, lat=0, fee=TAKER, slip=0.0, chase=True):
        F, m = self.F, self.m
        ia, iz = np.searchsorted(F["ct"], a), np.searchsorted(F["ct"], z)
        idx = np.nonzero(sig[ia:iz])[0] + ia
        jz = np.searchsorted(m["t"], z)
        pz = m["c"][jz - 1]
        tr, pos, pe, je, Te, binds = [], 0, 0.0, 0, 0, 0
        cost = fee + slip

        def close(j, p, T):
            seg = slice(je, max(j, je + 1))
            w = m["l"][seg].min() if pos > 0 else m["h"][seg].max()
            adv = 1 - w / pe if pos > 0 else w / pe - 1
            k0, k1 = np.searchsorted(self.ft, Te, "right"), np.searchsorted(self.ft, T, "right")
            fund = float((self.fr[k0:k1] * self.fp[k0:k1]).sum())
            rr = p / pe
            tr.append({"Te": int(Te), "Tx": int(T), "d": pos, "pe": pe, "px": p,
                       "r": pos * (rr - 1) - cost * (1 + rr) - pos * fund / pe, "adv": adv})  # fmt: skip

        for i in idx.tolist():
            d = int(sig[i])
            if d == pos:
                continue
            T = int(F["ct"][i]) + lat * MIN
            if T >= z:
                break
            j, p = self.px(T)
            if pos:
                close(j, p, T)
                pos = 0
            lim = 0.03 if self.agree[i] == d else 0.01
            if chase and d * (p / F["c"][i] - 1) >= lim:
                binds += 1
                continue
            pos, pe, je, Te = d, p, j, T
        if pos:
            close(jz, pz, z)
        return tr, binds


def summ(tr, a, z, binds=0, fee=TAKER):
    days = (z - a) / DAY
    if not tr:
        return {"n": 0, "perday": 0.0, "binds": binds}
    r = np.array([x["r"] for x in tr])
    adv = np.array([x["adv"] for x in tr])
    d = np.array([x["d"] for x in tr])
    w = r > 0
    out = {"n": len(tr), "perday": len(tr) / days, "long": int((d > 0).sum()), "short": int((d < 0).sum()),
           "win": float(w.mean()), "avg_win": float(r[w].mean()) if w.any() else None,
           "avg_loss": float(r[~w].mean()) if (~w).any() else None, "mean": float(r.mean()),
           "mae_med": float(np.median(adv)), "mae_p95": float(np.quantile(adv, 0.95)), "mae_max": float(adv.max()),
           "hold_h_med": float(np.median([(x["Tx"] - x["Te"]) / 3_600_000 for x in tr])), "binds": binds}  # fmt: skip
    for L in LEVS:
        out[f"x{L}"] = path(r, adv, d, L, fee)
    return out


def lab4h(ind, F, p, a, z):
    """실험실 '4시간 뒤 맞는 방향': 롱·숏 첫 봉 신호 모두, 종가→16봉 뒤 종가."""
    lo, sh = ind.sides(p)
    c = F["c"]
    ia, iz = np.searchsorted(F["ct"], a), np.searchsorted(F["ct"], z)
    right = tot = 0
    for f, dd in ((lo, 1), (sh, -1)):
        i = np.nonzero(f[ia:iz])[0] + ia
        i = i[i + 16 < len(c)]
        mv = (c[i + 16] / c[i] - 1) * dd
        right += int((mv > 0).sum())
        tot += len(i)
    return {"right": right, "n": tot, "pct": right / tot if tot else None}


# ---------------- 무작위(원형 이동) ----------------
def rand_p(book, sig, a_bar, z_bar, rng, nsim=1000):
    fb, fd = book.flips(sig, a_bar, z_bar)
    t = book.trades(fb, fd, z_bar)
    if t is None or len(fb) < 3:
        return None
    act = float(t["r"].mean())
    span = z_bar - 1 - a_bar
    sims = np.empty(nsim)
    for k in range(nsim):
        sh = np.sort((fb - a_bar + rng.integers(0, span)) % span) + a_bar
        sh = np.unique(sh)
        d0 = rng.choice((-1, 1))
        dd = (d0 * (-1) ** np.arange(len(sh))).astype(np.int8)
        sims[k] = book.trades(sh, dd, z_bar)["r"].mean()
    p_emp = (1 + (sims >= act).sum()) / (1 + nsim)
    zsc = (act - sims.mean()) / (sims.std() + 1e-12)
    p_norm = 0.5 * math.erfc(zsc / math.sqrt(2))
    return {"act": act, "rand_mean": float(sims.mean()), "p_emp": float(p_emp), "z": float(zsc), "p_norm": p_norm,
            "p_bonf": min(1.0, p_norm * N_CUM)}  # fmt: skip


# ---------------- 격자 ----------------
def grid():
    for sn, sl, cn, cl, rn, rl, w, need in itertools.product(G_ST_N, G_ST_L, G_CC_N, G_CC_L, G_RS_N, G_RS_L, G_W, G_NEED):
        yield {"stoch_n": sn, "stoch_lo": sl[0], "stoch_hi": sl[1], "cci_n": cn, "cci_lo": cl[0], "cci_hi": cl[1],
               "rsi_n": rn, "rsi_lo": rl[0], "rsi_hi": rl[1], "w": w, "need": need}  # fmt: skip


def key(p):
    return (f"S{p['stoch_n']}:{p['stoch_lo']}/{p['stoch_hi']} C{p['cci_n']}:±{p['cci_hi']} "
            f"R{p['rsi_n']}:{p['rsi_lo']}/{p['rsi_hi']} W{p['w']} {p['need']}/3")  # fmt: skip


def win_stats(t, a, z):
    """창 안에서 시작·끝난 거래(전체 거래열에서)."""
    lo, hi = np.searchsorted(t["Te"], a), np.searchsorted(t["Tx"], z, "right")
    if hi <= lo:
        return {"n": 0}
    r, adv, d = t["r"][lo:hi], t["adv"][lo:hi], t["d"][lo:hi]
    p5 = path(r, adv, d, 5)
    return {"n": int(hi - lo), "win": float((r > 0).mean()), "mean": float(r.mean()), "ret5": p5["ret"],
            "liq5": p5["liq"], "mdd5": p5["mdd"], "ret1": float(np.prod(1 + r) - 1)}  # fmt: skip


def pick(stats, nmin):
    """선택 규칙: n≥nmin, 5배 청산 0, 승률≥70% 중 5배 수익 최대(동률이면 거래 많은 쪽). 없으면 승률 조건만 뺌."""
    n, win, s5, lq = stats
    ok = (n >= nmin) & (lq == 0)
    for need70 in (True, False):
        c = ok & (win >= 0.7) if need70 else ok
        if c.any():
            sc = np.where(c, s5 + n * 1e-9, -np.inf)
            return int(np.argmax(sc)), need70
    return None, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = P.load_1m(out)
    now = int(m["t"][-1]) + MIN
    ft, fr = P.load_funding()
    F = bars(m, 15, now)
    F4, Fd = bars(m, 240, now), bars(m, 1440, now)
    t_end = int(F["ct"][-1])
    print("1m", len(m["t"]), "15m", len(F["t"]), "end", ds(t_end), f"{time.time() - t0:.0f}s", flush=True)
    ind = Ind(F)

    if args.check:  # 실험실 파이썬과 맞춰 보기
        spec = importlib.util.spec_from_file_location("scr_lab", LAB_SCR)
        lab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lab)
        rng = np.random.default_rng(1)
        allg = list(grid())
        ps = [BASE] + [allg[i] for i in rng.choice(len(allg), 20, replace=False)]
        a = len(F["t"]) - 3000
        sub = {k: F[k][a:] for k in ("h", "l", "c")}
        bad = 0
        for p in ps:
            mine = Ind({k: sub[k] for k in sub}).sides(p)
            ref = lab.signals(sub["h"].tolist(), sub["l"].tolist(), sub["c"].tolist(), {**p, "mode": "cross"})
            same = (np.nonzero(mine[0])[0].tolist() == ref["long"]) and (np.nonzero(mine[1])[0].tolist() == ref["short"])
            bad += not same
        print("check vs scr_lab.py:", len(ps) - bad, "/", len(ps), "same")
        return

    W = {"d7": (t_end - 7 * DAY, t_end), "d14": (t_end - 14 * DAY, t_end), "d30": (t_end - 30 * DAY, t_end),
         "m6": (t_end - 183 * DAY, t_end), "y1": (t_end - 365 * DAY, t_end), "full": (ms("2020-03-01"), t_end),
         "m6x": (t_end - 183 * DAY, t_end - 30 * DAY), "y1x": (t_end - 365 * DAY, t_end - 30 * DAY),
         "p2025": (ms("2020-03-01"), ms("2026-01-01"))}  # fmt: skip
    for y in range(2020, 2027):
        W[f"y{y}"] = (max(ms(f"{y}-01-01"), ms("2020-03-01")), min(ms(f"{y + 1}-01-01"), t_end))
    sim = Sim(m, F, F4, Fd, ft, fr, ind)
    book = Book(F, ft, fr)
    rng = np.random.default_rng(20261108)
    res = {"t_end": t_end, "t_end_s": ds(t_end), "n_cum": N_CUM}

    def bar(T):
        return int(np.searchsorted(F["t"], T))

    def report(p, wins, red=True, rand=True):
        sig = ind.sig(p)
        o = {"key": key(p), "params": p}
        for wn in wins:
            a, z = W[wn]
            tr, b = sim.run(sig, a, z)
            o[wn] = summ(tr, a, z, b)
            o[wn]["lab4h"] = lab4h(ind, F, p, a, z)
        if red:
            for wn in ("d30", "y1x", "full"):
                a, z = W[wn]
                o[f"red_{wn}"] = {"lat1m": summ(*sim.run(sig, a, z, lat=1)[:1], a, z),
                                  "fee15_slip2": summ(sim.run(sig, a, z, fee=TAKER * 1.5, slip=0.0002)[0], a, z)}  # fmt: skip
        if rand:
            for wn in ("d30", "y1x", "p2025"):
                a, z = W[wn]
                o[f"rand_{wn}"] = rand_p(book, sig, bar(a), bar(z), rng)
        return o

    ALLW = ["d7", "d14", "d30", "m6", "y1", "full", "m6x", "y1x", "p2025"] + [f"y{y}" for y in range(2020, 2027)]
    res["base"] = report(BASE, ALLW)
    print("base done", f"{time.time() - t0:.0f}s", flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))

    # ---- 격자 ----
    combos = list(grid())
    assert len(combos) == N_GRID
    mondays = np.arange(ms("2022-01-03"), t_end, 7 * DAY)
    FW = ["d7", "d14", "d30", "m6x", "y1x", "p2025"]
    fixed = {wn: {k: np.zeros(N_GRID, np.float32) for k in ("n", "win", "ret5", "liq5", "mean", "ret1")} for wn in FW}
    TR = {tw: {k: np.zeros((N_GRID, len(mondays)), np.float32) for k in ("n", "win", "s5", "liq")} for tw in (30, 90)}
    nb = len(F["t"])
    for ci, p in enumerate(combos):
        sig = ind.sig(p)
        fb, fd = book.flips(sig, 0, nb)
        t = book.trades(fb, fd, nb)
        if t is None:
            continue
        for wn in FW:
            s = win_stats(t, *W[wn])
            for k in fixed[wn]:
                fixed[wn][k][ci] = s.get(k, 0) if s["n"] else (0 if k != "win" else np.nan)
        # 걷기 전진 학습 창: 접두합으로
        r, d = t["r"], t["d"]
        liq = (t["adv"] >= liq_thr(5, d)).astype(np.int32)
        lg = np.where(liq > 0, 0.0, np.log(np.maximum(1 + 5 * r, 1e-12)))
        Pw, Pl, Pq = (np.r_[0, np.cumsum(x)] for x in ((r > 0).astype(np.int32), lg, liq))
        for tw in (30, 90):
            lo = np.searchsorted(t["Te"], mondays - tw * DAY)
            hi = np.maximum(np.searchsorted(t["Tx"], mondays, "right"), lo)
            n = hi - lo
            TR[tw]["n"][ci] = n
            TR[tw]["win"][ci] = np.where(n > 0, (Pw[hi] - Pw[lo]) / np.maximum(n, 1), 0)
            TR[tw]["s5"][ci] = Pl[hi] - Pl[lo]
            TR[tw]["liq"][ci] = Pq[hi] - Pq[lo]
        if ci % 500 == 0:
            print("grid", ci, f"{time.time() - t0:.0f}s", flush=True)
        if len(ind.rec) > 200:  # 메모리: 교차 캐시는 다시 만들 수 있다
            ind.rec.clear()
    np.savez_compressed(out / "grid_fixed.npz", **{f"{wn}_{k}": v for wn in fixed for k, v in fixed[wn].items()})

    # ---- (b) 30일 선택 ----
    f30 = fixed["d30"]
    s5 = np.log(np.maximum(1 + f30["ret5"].astype(float), 1e-12))
    best, ok70 = pick((f30["n"], np.nan_to_num(f30["win"]), s5, f30["liq5"]), 10)
    c70 = (f30["n"] >= 10) & (np.nan_to_num(f30["win"]) >= 0.7)
    res["grid"] = {"n": N_GRID, "n_30d_win70": int(c70.sum()), "n_30d_win70_liq0": int((c70 & (f30["liq5"] == 0)).sum())}
    for wn in ("m6x", "y1x", "p2025"):
        g = fixed[wn]
        sel = c70 & (f30["liq5"] == 0)
        if sel.any():
            res["grid"][f"win70_{wn}"] = {
                "ret5_pos": int((g["ret5"][sel] > 0).sum()), "liq5_0": int((g["liq5"][sel] == 0).sum()),
                "win_med": float(np.nanmedian(g["win"][sel])), "ret1_med": float(np.median(g["ret1"][sel])),
                "ret5_pos_and_liq0": int(((g["ret5"][sel] > 0) & (g["liq5"][sel] == 0)).sum())}  # fmt: skip
    # 30일 승률 70%↑ 상위(5배 수익순) 10개 정밀 보고
    order = np.argsort(-np.where(c70 & (f30["liq5"] == 0), f30["ret5"], -np.inf))
    top = [int(i) for i in order[:10] if c70[i] and f30["liq5"][i] == 0]
    res["best30"] = {"idx": best, "ok70": ok70}
    res["top30"] = []
    for i in top:
        res["top30"].append(report(combos[i], ["d7", "d14", "d30", "m6x", "y1x", "p2025"]
                                   + [f"y{y}" for y in range(2020, 2027)], red=(i == best), rand=(i == best)))  # fmt: skip
    print("top done", f"{time.time() - t0:.0f}s", flush=True)

    # ---- (d) 이웃 ----
    if best is not None:
        bp = combos[best]
        axes = {"stoch_n": G_ST_N, "stoch_l": G_ST_L, "cci_n": G_CC_N, "cci_l": G_CC_L, "rsi_n": G_RS_N,
                "rsi_l": G_RS_L, "w": G_W, "need": G_NEED}  # fmt: skip
        cur = {"stoch_n": bp["stoch_n"], "stoch_l": (bp["stoch_lo"], bp["stoch_hi"]), "cci_n": bp["cci_n"],
               "cci_l": (bp["cci_lo"], bp["cci_hi"]), "rsi_n": bp["rsi_n"], "rsi_l": (bp["rsi_lo"], bp["rsi_hi"]),
               "w": bp["w"], "need": bp["need"]}  # fmt: skip
        nbrs = []
        for ax, vals in axes.items():
            k = vals.index(cur[ax])
            for kk in (k - 1, k + 1):
                if 0 <= kk < len(vals):
                    q = dict(cur, **{ax: vals[kk]})
                    qp = {"stoch_n": q["stoch_n"], "stoch_lo": q["stoch_l"][0], "stoch_hi": q["stoch_l"][1],
                          "cci_n": q["cci_n"], "cci_lo": q["cci_l"][0], "cci_hi": q["cci_l"][1], "rsi_n": q["rsi_n"],
                          "rsi_lo": q["rsi_l"][0], "rsi_hi": q["rsi_l"][1], "w": q["w"], "need": q["need"]}  # fmt: skip
                    j = combos.index(qp)
                    nbrs.append({"axis": ax, "key": key(qp), **{f"{wn}_{k2}": float(fixed[wn][k2][j]) for wn in
                                 ("d30", "m6x", "y1x", "p2025") for k2 in ("n", "win", "ret5", "liq5")}})  # fmt: skip
        res["neighbors"] = nbrs

    # ---- (c) 걷기 전진 ----
    res["wf"] = {}
    for tw, nmin in ((30, 10), (90, 30)):
        trs, picks, n70 = [], [], 0
        for wi, mo in enumerate(mondays.tolist()):
            st = (TR[tw]["n"][:, wi], TR[tw]["win"][:, wi], TR[tw]["s5"][:, wi], TR[tw]["liq"][:, wi])
            ci, ok = pick(st, nmin)
            if ci is None:
                picks.append(None)
                continue
            n70 += ok
            picks.append(key(combos[ci]))
            z = min(mo + 7 * DAY, t_end)
            tr, _ = sim.run(ind.sig(combos[ci]), mo, z)
            trs += tr
            if len(ind.rec) > 200:
                ind.rec.clear()
        o = {"picked70": n70, "weeks": len(mondays), "distinct": len(set(x for x in picks if x))}
        for nm, a in (("from2022", ms("2022-01-03")), ("from2024_07", ms("2024-07-01"))):
            sub = [x for x in trs if x["Te"] >= a]
            o[nm] = summ(sub, a, t_end)
            o[nm]["years"] = {}
            for y in range(2022, 2027):
                yy = [x for x in sub if ms(f"{y}-01-01") <= x["Te"] < ms(f"{y + 1}-01-01")]
                if yy:
                    s = summ(yy, max(a, ms(f"{y}-01-01")), min(ms(f"{y + 1}-01-01"), t_end))
                    o[nm]["years"][y] = {k: s[k] for k in ("n", "win", "x1", "x5")}
        res["wf"][tw] = o
        print("wf", tw, f"{time.time() - t0:.0f}s", flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
