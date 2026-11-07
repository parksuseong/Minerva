"""146번: TH1을 한 번도 안 본 알트코인 8개에. 등록: 2026-11-07-th1-alt-markets-hypotheses.md

실행:
  python docs/backtests/raw/th1_alts.py fetch            # data.binance.vision 5분봉·펀딩
  PYTHONPATH=<pylib>:src:docs/backtests/raw python docs/backtests/raw/th1_alts.py run --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SYMS = ("SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT")
HOME = Path.home() / ".minerva/data/alts"
BNV = "https://data.binance.vision/data/futures/um"
UA = {"User-Agent": "minerva-data/1.0"}
RISK = 0.02


def get(url: str, tries: int = 4) -> bytes | None:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (k + 1))
        except Exception:  # noqa: BLE001 - 네트워크 일시 오류는 다시 시도
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed {url}")


def vision(url: str) -> list | None:
    blob = get(url)
    if blob is None:
        return None
    want = get(url + ".CHECKSUM").decode().split()[0]
    if hashlib.sha256(blob).hexdigest() != want:
        raise ValueError(f"checksum mismatch {url}")
    z = zipfile.ZipFile(io.BytesIO(blob))
    rows = list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))))
    return [x for x in rows if x and x[0].isdigit()]


def months(a: date, b: date) -> list:
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_sym(sym: str) -> dict:
    """한 코인씩, 받은 파일의 행을 바로 디스크 CSV에 이어 쓴다(메모리 3.8GB 서버, 이전 다운로드가 메모리 부족으로 죽음).

    이어 받기: 끝난 월·일 이름을 done.txt에 적고 다시 실행하면 건너뛴다.
    """
    d = HOME / sym
    d.mkdir(parents=True, exist_ok=True)
    kp, fp, dp = d / "klines_5m.csv", d / "funding.csv", d / "done.txt"
    done = set(dp.read_text().split()) if dp.exists() else set()
    if not kp.exists():
        kp.write_text("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                      "taker_buy_volume,taker_buy_quote_volume,ignore\n")
    if not fp.exists():
        fp.write_text("calc_time,funding_interval_hours,last_funding_rate\n")
    today = datetime.now(timezone.utc).date()
    last_month = (today.replace(day=1) - timedelta(days=1))
    found = any(x.startswith("m") for x in done)
    for ym in months(date(2019, 9, 1), last_month):
        if f"m{ym}" in done or f"x{ym}" in done:
            continue
        rows = vision(f"{BNV}/monthly/klines/{sym}/5m/{sym}-5m-{ym}.zip")
        if rows is None:
            if found:
                print(sym, "missing month", ym, file=sys.stderr)
            else:
                with dp.open("a") as f:
                    f.write(f"x{ym}\n")  # 상장 전
            continue
        found = True
        with kp.open("a", newline="") as f:
            csv.writer(f).writerows(r[:12] for r in rows)
        fr = vision(f"{BNV}/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip")
        with fp.open("a", newline="") as f:
            csv.writer(f).writerows(r[:3] for r in (fr or []))
        with dp.open("a") as f:
            f.write(f"m{ym}\n")
        del rows, fr
        time.sleep(0.3)
    day = today.replace(day=1)
    while day < today:
        key = f"d{day.isoformat()}"
        if key not in done:
            rows = vision(f"{BNV}/daily/klines/{sym}/5m/{sym}-5m-{day.isoformat()}.zip")
            if rows:
                with kp.open("a", newline="") as f:
                    csv.writer(f).writerows(r[:12] for r in rows)
                with dp.open("a") as f:
                    f.write(key + "\n")
            time.sleep(0.3)
        day += timedelta(days=1)
    # 정렬·중복 점검만 (한 번에 시각 열만 읽는다)
    ts = []
    with kp.open() as f:
        next(f)
        for line in f:
            ts.append(int(line.split(",", 1)[0]))
    srt = all(a < b for a, b in zip(ts, ts[1:]))
    gaps = sum(1 for a, b in zip(ts, ts[1:]) if b - a != 300_000)
    nf = sum(1 for _ in fp.open()) - 1
    return {"sym": sym, "rows": len(ts), "first": ts[0] if ts else None, "last": ts[-1] if ts else None,
            "funding": nf, "gaps": gaps, "sorted": srt}


def do_fetch() -> None:
    res = []
    for sym in SYMS:  # 한 번에 하나씩 (메모리)
        r = fetch_sym(sym)
        r["first"] = datetime.fromtimestamp(r["first"] / 1000, timezone.utc).strftime("%Y-%m-%d")
        r["last"] = datetime.fromtimestamp(r["last"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(json.dumps(r), flush=True)
        res.append(r)
    (HOME / "coverage.json").write_text(json.dumps(res, indent=1))


# ---------------- 분석 ----------------

def do_run(out: Path) -> None:
    import numpy as np

    import btc_fut_daily_regime as R
    import th1
    import trend_hold_cross_market as T

    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    series: dict = {}

    def run_side(m, reg, newday, atr, side: int):
        """th1.run과 같은 규칙을 롱(+1) 또는 거울 숏(−1)으로. 숏은 reg를 뒤집고 가격 손익 부호를 바꾼다."""
        if side == 1:
            return th1.run(m, reg, newday, atr, RISK)
        d = m.d
        t, o, h, c = d["t"], d["o"], d["h"], d["c"]
        n = m.n
        eq = np.full(n, np.nan)
        inpos = np.zeros(n, bool)
        E = 1.0
        eq[: m.start] = 1.0
        trades = []
        i, can, last = m.start, True, m.start
        open_pos = None
        while i < n - 3:
            if newday[i]:
                can = True
            if not (int(reg[i]) == -1 and can):
                i += 1
                continue
            a = atr[i]
            if not (a > 0):
                i += 1
                continue
            w = min(th1.CAP, RISK / a)
            e = i + 1
            px, j5 = m.px5(t[e], o[e])
            t_in = t[e] + T.M5
            E0 = E
            N = w * E0
            qty = N / px
            stop = px * (1 + th1.STOP)
            x = None
            if j5 >= 0:
                for jj in (j5, j5 + 1):
                    if jj < len(m.t5) and m.t5[jj] < t[e] + T.BAR and m.d5["h"][jj] >= stop:
                        x = (e, stop * (1 + th1.SLIP), m.t5[jj] + T.M5, True)
                        break
            elif h[e] >= stop:
                x = (e, stop * (1 + th1.SLIP), t[e] + T.BAR, True)
            b = e
            while x is None and b < n - 2:
                if b > e and h[b] >= stop:
                    x = (b, max(o[b], stop) * (1 + th1.SLIP), t[b] + T.BAR, True)
                    break
                if int(reg[b]) != -1:
                    xp, _ = m.px5(t[b + 1], o[b + 1])
                    x = (b + 1, xp, t[b + 1] + T.M5, False)
                    break
                b += 1
            if x is None:
                xb, xp, t_out, stopped = n - 1, float(c[-1]), float(t[-1] + T.BAR), None
            else:
                xb, xp, t_out, stopped = x
            ft, cum = th1.fund_pay(m, qty, t_in, t_out)
            cum = -cum  # 숏은 양의 펀딩을 받는다
            seg = np.arange(e, xb)
            if len(seg):
                k = np.searchsorted(ft, t[seg] + T.BAR, side="right")
                cz = np.r_[0.0, cum]
                eq[seg] = E0 - N * th1.FEE - qty * (c[seg] - px) - cz[k]
                inpos[seg] = True
            fund_all = float(cum[-1]) if len(cum) else 0.0
            exit_fee = 0.0 if stopped is None else qty * xp * th1.FEE
            E = E0 - N * th1.FEE - qty * (xp - px) - fund_all - exit_fee
            eq[last:e] = np.where(np.isnan(eq[last:e]), E0, eq[last:e])
            eq[xb] = E
            rec = dict(e=int(e), x=int(xb), ret=float((E - E0) / E0), w=float(w), stopped=stopped)
            if stopped is None:
                open_pos = rec
                break
            trades.append(rec)
            last = xb + 1
            if stopped:
                can = False
                i = xb
            else:
                i = xb - 1
        if open_pos is None:
            eq[last:] = np.where(np.isnan(eq[last:]), E, eq[last:])
        for k in range(1, n):
            if np.isnan(eq[k]):
                eq[k] = eq[k - 1]
        return eq, inpos, trades, open_pos

    def years(dt, de, t0):
        """전년 말 자산 기준 연도별 수익 (145 레드팀 수정)."""
        out = {}
        yrs = sorted({datetime.fromtimestamp(z / 1000, timezone.utc).year for z in dt if z >= t0})
        for y in yrs:
            a, b = T.ms(f"{y}-01-01"), T.ms(f"{y + 1}-01-01")
            pre = de[(dt < a) & (dt >= t0)]
            base = pre[-1] if len(pre) else de[dt >= t0][0]
            cur = de[(dt >= a) & (dt < b)]
            if len(cur):
                out[y] = round((cur[-1] / base - 1) * 100, 1)
        return out

    def shuffle_reg(reg, rng, start):
        """롱 구간(reg==1)과 빈 구간의 길이는 그대로, 순서만 섞는다."""
        on = (reg[start:] == 1).astype(np.int8)
        edges = np.flatnonzero(np.r_[True, on[1:] != on[:-1], True])
        lens = np.diff(edges)
        vals = on[edges[:-1]]
        ones = lens[vals == 1].copy()
        zeros = lens[vals == 0].copy()
        rng.shuffle(ones)
        rng.shuffle(zeros)
        seq, a, b = [], 0, 0
        for v in vals:
            if v == 1:
                seq.append(np.ones(ones[a], np.int8))
                a += 1
            else:
                seq.append(np.zeros(zeros[b], np.int8))
                b += 1
        new = reg.copy()
        new[start:] = np.where(np.concatenate(seq) == 1, 1, 0)
        return new

    def ratio_of(m, eq):
        dt, de = th1.daily(m, eq)
        s = th1.stats(dt, de, float(m.d["t"][m.start]))
        return s.get("ratio") if s else None

    for sym in SYMS:
        T.MARKETS[sym] = dict(k5=HOME / sym / "klines_5m.csv", fund=HOME / sym / "funding.csv",
                              start="2019-09-01", end=None)
        m = T.Mkt(sym)
        t = m.d["t"]
        t0 = float(t[m.start])
        reg, newday = R.regime(m.d5, m.d, 0)
        atr = T.daily_atr(m.d5, m.d, 0)
        bh = m.d["c"] / m.d["c"][m.start]
        bh[: m.start] = 1.0
        bdt, bde = th1.daily(m, bh)
        row = {"span": [datetime.fromtimestamp(t0 / 1000, timezone.utc).strftime("%Y-%m-%d"),
                        datetime.fromtimestamp(t[-1] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")],
               "BH": {"all": th1.stats(bdt, bde, t0), "2024~": th1.stats(bdt, bde, T.ms("2024-01-01"))}}
        series[f"{sym}|BH|t"], series[f"{sym}|BH|e"] = bdt, bde
        res_side = {}
        for name, side in (("long", 1), ("short", -1)):
            eq, inpos, tr, op = run_side(m, reg, newday, atr, side)
            dt, de = th1.daily(m, eq)
            g = np.array([z["ret"] for z in tr]) if tr else np.zeros(0)
            res_side[name] = eq
            row[name] = {"all": th1.stats(dt, de, t0), "2024~": th1.stats(dt, de, T.ms("2024-01-01")),
                         "mdd15": th1.mdd15(eq, m.start), "trades": len(tr),
                         "win": round(float((g > 0).mean() * 100), 1) if len(g) else None,
                         "avg_trade_pct": round(float(g.mean() * 100), 3) if len(g) else None,
                         "avg_notional": round(float(np.mean([z["w"] for z in tr])), 3) if tr else None,
                         "in_mkt": round(float(inpos[m.start:].mean() * 100), 1),
                         "years": years(dt, de, t0), "open": op is not None}
            series[f"{sym}|{name}|t"], series[f"{sym}|{name}|e"] = dt, de
        # 롱+숏: 두 쪽 자산 수익을 반씩이 아니라 같은 자본에서 시간 순으로 겹치지 않으니(방향이 다름) 곱한다
        le, se = res_side["long"], res_side["short"]
        ls = le * se
        dt, de = th1.daily(m, ls)
        row["long_short"] = {"all": th1.stats(dt, de, t0), "2024~": th1.stats(dt, de, T.ms("2024-01-01")),
                             "mdd15": th1.mdd15(ls, m.start)}
        # 무작위 켜짐·꺼짐 대조 (롱)
        actual = row["long"]["all"].get("ratio")
        rng = np.random.default_rng(7)
        null = []
        t_null = time.time()
        for _ in range(N_NULL):
            eqn, *_ = th1.run(m, shuffle_reg(reg, rng, m.start), newday, atr, RISK)
            null.append(ratio_of(m, eqn) or -9.0)
        null = np.array(null)
        row["null"] = {"n": N_NULL, "mean": round(float(null.mean()), 3), "median": round(float(np.median(null)), 3),
                       "p": round(float((null >= actual).mean()), 4) if actual is not None else None,
                       "secs": round(time.time() - t_null, 1)}
        rep[sym] = row
        print(sym, row["span"], row["long"]["all"], row["BH"]["all"].get("ratio"), row["null"], flush=True)
    ratios = [rep[s]["long"]["all"].get("ratio") for s in SYMS]
    cagrs = [rep[s]["long"]["all"].get("cagr") for s in SYMS]
    beats = [rep[s]["long"]["all"].get("ratio", -9) > (rep[s]["BH"]["all"].get("ratio") or -9) for s in SYMS]
    med = float(np.median([r for r in ratios if r is not None]))
    crit = {"1_median_ratio_gt_0.5": bool(med > 0.5), "median_ratio": round(med, 3),
            "2_cagr_pos_ge6": bool(sum(c > 0 for c in cagrs if c is not None) >= 6),
            "n_cagr_pos": int(sum(c > 0 for c in cagrs if c is not None)),
            "3_beat_bh_ge6": bool(sum(beats) >= 6), "n_beat_bh": int(sum(beats)),
            "n_null_p_lt_0.05": int(sum(1 for s in SYMS if (rep[s]["null"]["p"] or 1) < 0.05))}
    crit["pass"] = crit["1_median_ratio_gt_0.5"] and crit["2_cagr_pos_ge6"] and crit["3_beat_bh_ge6"]
    rep["criteria"] = crit
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    np.savez(out / "series.npz", **series)
    print(json.dumps(crit))


N_NULL = 500

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fetch", "run"))
    ap.add_argument("--out", default="")
    ap.add_argument("--null", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "fetch":
        do_fetch()
    else:
        N_NULL = a.null
        do_run(Path(a.out))
