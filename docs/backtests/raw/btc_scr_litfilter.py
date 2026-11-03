"""30분봉 SCR 3단 + 문헌 지표 필터. 등록: 2026-11-03-btc-scr-litfilter-hypotheses.md

신호마다 두 결과를 잰다.
- 실전: 1% 손절, 반대 SCR 신호 청산, 보유 상한 없음. 비용 한쪽 0.05%, 순수익 > 0.1%면 승.
- 대칭: 다음 시가에서 ±1% 중 유리한 쪽이 먼저 닿으면 승(한 봉에 둘 다면 제외).
지표 구간은 학습(2022-01~2024-12) 신호 값으로 5등분, 같은 경계를 시험(2025-01~)에 쓴다.
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_litfilter.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import btc_scr_model as M
from btc_fractal import resample, scr3

MIN30 = 1_800_000
COST = 0.0005
TRAIN0 = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
TEST0 = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def load5x(path: Path) -> dict:
    cols = {k: [] for k in ("t", "o", "h", "l", "c", "v", "n", "tb")}
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            for k, j in (("t", 0), ("o", 1), ("h", 2), ("l", 3), ("c", 4), ("v", 5), ("n", 8), ("tb", 9)):
                cols[k].append(float(row[j]))
    return {k: np.array(v) for k, v in cols.items()}


def bars30(d5: dict, offset: int) -> dict:
    sh = dict(d5)
    sh["t"] = d5["t"] - offset * 60_000
    d = resample(sh, 6)
    key = (sh["t"] // (6 * 300_000)).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    st, en = edges[:-1], edges[1:]
    full = (en - st) == 6
    st, en = st[full], en[full]
    for k in ("n", "tb"):
        cs = np.r_[0.0, np.cumsum(d5[k])]
        d[k] = cs[en] - cs[st]
    d["t"] = d["t"] + offset * 60_000
    return d


def roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    ok = np.isfinite(x)
    cs = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    cc = np.r_[0.0, np.cumsum(ok)]
    out = np.full(len(x), np.nan)
    cnt = cc[n:] - cc[:-n]
    out[n - 1 :] = np.where(cnt > n // 2, (cs[n:] - cs[:-n]) / np.maximum(cnt, 1), np.nan)
    return out


def roll_z(x: np.ndarray, n: int) -> np.ndarray:
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    a, b, c = (np.r_[0, np.cumsum(q)] for q in (v, v * v, ok.astype(float)))
    cnt = np.maximum(c[n:] - c[:-n], 1)
    m = (a[n:] - a[:-n]) / cnt
    sd = np.sqrt(np.maximum((b[n:] - b[:-n]) / cnt - m * m, 1e-18))
    out = np.full(len(x), np.nan)
    out[n - 1 :] = (x[n - 1 :] - m) / sd
    return out


def asof(t_need: np.ndarray, t_src: np.ndarray, x: np.ndarray) -> np.ndarray:
    j = np.searchsorted(t_src, t_need, side="right") - 1
    return np.where(j >= 0, x[np.maximum(j, 0)], np.nan)


def load_fut(root: Path) -> dict:
    fut = root / "futures"
    t, oi, top, glob, tk = [], [], [], [], []
    with (fut / "metrics.csv").open() as f:
        for r in csv.DictReader(f):
            ts = datetime.strptime(r["create_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            t.append(ts.timestamp() * 1000)
            g = lambda k: float(r[k]) if r[k] else np.nan  # noqa: E731
            oi.append(g("sum_open_interest"))
            top.append(g("sum_toptrader_long_short_ratio"))
            glob.append(g("count_long_short_ratio"))
            tk.append(g("sum_taker_long_short_vol_ratio"))
    o = np.argsort(t)
    m = {"t": np.array(t)[o], "oi": np.array(oi)[o], "top": np.array(top)[o],
         "glob": np.array(glob)[o], "tk": np.array(tk)[o]}
    ft, fr = [], []
    with (fut / "funding.csv").open() as f:
        for r in csv.DictReader(f):
            ft.append(float(r["calc_time"]))
            fr.append(float(r["last_funding_rate"]))
    o = np.argsort(ft)
    m["ft"], m["fr"] = np.array(ft)[o], np.array(fr)[o]
    return m


def feature_table(d: dict, fut: dict) -> dict:
    """신호 봉 i가 마감된 시각까지 알려진 값만. 방향 지표는 나중에 방향 부호를 곱한다."""
    c, o, v = d["c"], d["o"], d["v"]
    close_t = d["t"] + MIN30
    imb = np.where(v > 0, d["tb"] / np.where(v > 0, v, 1) * 2 - 1, 0.0)
    v4 = np.convolve(v, np.ones(4), "full")[: len(v)]
    tb4 = np.convolve(d["tb"], np.ones(4), "full")[: len(v)]
    lr = np.r_[0.0, np.diff(np.log(c))]
    rv = np.sqrt(roll_mean(lr * lr, 48))
    day = (d["t"] // 86_400_000) * 86_400_000
    jd = np.searchsorted(d["t"], day)
    since_open = np.log(c / o[jd])
    r48 = np.r_[np.full(48, np.nan), np.log(c[48:] / c[:-48])]
    # 선물 지표: 5분 값은 create_time 시점 값이라 마감 시각까지의 최신 값을 쓴다
    mt = fut["t"]
    oi = asof(close_t, mt, fut["oi"])
    oi8 = asof(close_t - 8 * MIN30, mt, fut["oi"])
    ma_imb = imb  # 이름 유지
    with np.errstate(all="ignore"):
        r2 = np.r_[np.full(2, np.nan), np.log(c[2:] / c[:-2])]
        rz = r2 / np.sqrt(roll_mean(r2 * r2, 48 * 7))
        up = np.sign(c - o)
        imbz = roll_z(imb, 96)
        lq = np.log(np.maximum(v * c, 1e-9))
        slot = np.full(len(c), np.nan)
        for k in range(1, 8):
            prev = np.r_[np.full(48 * k, np.nan), lq[: -48 * k]]
            slot = np.where(np.isnan(slot), 0.0, slot) + np.nan_to_num(prev) if k == 1 else slot + np.nan_to_num(prev)
        av = lq - slot / 7
        fr = asof(close_t, fut["ft"], fut["fr"])
        frp = np.full(len(fr), np.nan)
        for i in np.flatnonzero(np.isfinite(fr)):
            w = fr[max(0, i - 48 * 90) : i + 1 : 16]
            frp[i] = (w < fr[i]).mean()
        vp = roll_mean(np.abs(imb), 50)
    oi8 = np.where(oi8 > 0, oi8, np.nan)
    return {
        # (값, 방향 부호를 곱하는가)
        "F1 체결 불균형(1봉)": (ma_imb, True),
        "F2 체결 불균형(4봉)": (np.where(v4 > 0, tb4 / np.where(v4 > 0, v4, 1) * 2 - 1, 0.0), True),
        "F3 거래량 이상(z, 48봉)": (roll_z(np.log(np.maximum(v, 1e-9)), 48 * 7), False),
        "F4 평균 체결 크기(z)": (roll_z(np.log(np.maximum(v / np.maximum(d["n"], 1), 1e-12)), 48 * 7), False),
        "F5 실현 변동성(48봉)": (rv / roll_mean(rv, 48 * 30), False),
        "F6 당일 수익(UTC 00시부터)": (since_open, True),
        "F7 직전 24시간 수익": (r48, True),
        "F8 시간대(UTC 4시간 묶음)": (((d["t"] // 3_600_000) % 24 // 4).astype(float), False),
        "F9 펀딩비": (asof(close_t, fut["ft"], fut["fr"]), True),
        "F10 미결제약정 4시간 변화": (np.log(oi / oi8), False),
        "F11 전체 롱숏 계정 비율(z)": (roll_z(np.log(asof(close_t, mt, fut["glob"])), 48 * 30), True),
        "F12 상위 트레이더 포지션 비율(z)": (roll_z(np.log(asof(close_t, mt, fut["top"])), 48 * 30), True),
        "F13 선물 테이커 매수/매도 비율": (np.log(asof(close_t, mt, fut["tk"])), True),
        "P1 직전 1시간 수익 z(되돌림)": (rz, True),
        "P2 체결 불균형 z × 봉 방향": (imbz * (up != 0) * (np.sign(imbz) == up), True),
        "P3 같은 시각 대비 이상 거래대금": (av, False),
        "P4 펀딩비 90일 백분위": (frp, True),
        "P7 VPIN 근사(50봉)": (vp, False),
    }


def touch(d: dict, i: int, side: int) -> int | None:
    o, h, l = d["o"], d["h"], d["l"]  # noqa: E741
    e = i + 1
    px = o[e]
    up, dn = px * 1.01, px * 0.99
    for j in range(e, min(e + 5000, len(o))):
        hi, lo = h[j] >= up, l[j] <= dn
        if hi and lo:
            return None
        if hi or lo:
            return int(hi == (side > 0))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = bars30(load5x(root / "5m.csv"), a.offset)
    fut = load_fut(root)
    n = len(d["c"])
    on_l, on_s, *_ = scr3(d)
    feats = feature_table(d, fut)
    rows = []
    for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
        for i in np.flatnonzero(on):
            if d["t"][i] < TRAIN0 or i >= n - 2:
                continue
            res = M.label(d, i, side, opp)
            if res is None:
                continue
            g, x = res
            rows.append({"i": int(i), "t": float(d["t"][i]), "side": side, "net": float(g) - 2 * COST,
                         "exit": int(x), "touch": touch(d, i, side),
                         "f": {k: (float(val[i]) * (side if signed else 1)) for k, (val, signed) in feats.items()}})
    rows.sort(key=lambda r: r["i"])
    # 실전 승률은 포지션이 없을 때만 진입한 거래로 센다
    busy = -1
    for r in rows:
        r["taken"] = r["i"] > busy
        if r["taken"]:
            busy = r["exit"]
    end = d["t"][-1] + MIN30

    def stat(rs):
        tk = [r for r in rs if r["taken"]]
        net = np.array([r["net"] for r in tk])
        tch = [r["touch"] for r in rs if r["touch"] is not None]
        return {"n": len(tk), "win": round(float((net > 0.001).mean() * 100), 1) if len(net) else None,
                "ev": round(float(net.mean() * 100), 3) if len(net) else None,
                "tn": len(tch), "touch": round(float(np.mean(tch) * 100), 1) if tch else None}

    periods = {"train": lambda t: t < TEST0, "test": lambda t: t >= TEST0,
               "6m": lambda t: t >= end - 183 * 86_400_000, "30d": lambda t: t >= end - 30 * 86_400_000}
    rep = {"offset": a.offset, "base": {p: stat([r for r in rows if f(r["t"])]) for p, f in periods.items()}}
    for name in feats:
        vals = [r["f"][name] for r in rows if r["t"] < TEST0 and np.isfinite(r["f"][name])]
        if len(vals) < 50:
            rep[name] = {"skip": len(vals)}
            continue
        if name.startswith("F8"):
            cuts = [0.5, 1.5, 2.5, 3.5, 4.5]
        else:
            cuts = list(np.quantile(vals, [0.2, 0.4, 0.6, 0.8]))
        ent = {"cuts": [round(float(x), 6) for x in cuts]}
        for b in range(len(cuts) + 1):
            sel = [r for r in rows if np.isfinite(r["f"][name]) and int(np.sum(r["f"][name] > np.array(cuts))) == b]
            # 필터 안에서 다시 포지션 점유를 센다
            busy, kept = -1, []
            for r in sel:
                if r["i"] > busy:
                    kept.append(dict(r, taken=True))
                    busy = r["exit"]
            ent[str(b)] = {p: stat([r for r in kept if f(r["t"])]) for p, f in periods.items()}
        rep[name] = ent
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep["base"], ensure_ascii=False))
    for name in feats:
        e = rep[name]
        if "skip" in e:
            print(name, "skip", e["skip"])
            continue
        line = []
        for b in range(len(e["cuts"]) + 1):
            s = e[str(b)]
            line.append(f"{b}: tr {s['train']['win']}%/{s['train']['touch']}% ({s['train']['n']}) "
                        f"te {s['test']['win']}%/{s['test']['touch']}% ({s['test']['n']})")
        print(name, " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
