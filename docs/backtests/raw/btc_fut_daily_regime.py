"""일봉 추세 방향 보유, 15분봉 시장가 체결. 등록: 2026-11-07-btc-fut-daily-regime-hold-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_fut_daily_regime.py \
    --out <폴더> [--day-shift 8] [--perm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample
from btc_scr_entry_stop import summ, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, direction, ystart

M5 = 300_000
BAR = 900_000
DAY = 86_400_000
STOP = 0.01


def regime(d5: dict, d: dict, shift_h: int, frame: int = 288, ma: int = 0):
    sh = dict(d5)
    sh["t"] = d5["t"] - shift_h * 3_600_000
    u = resample(sh, frame)
    close = u["t"] + shift_h * 3_600_000 + frame * M5
    j = np.searchsorted(close, d["t"] + BAR, side="right") - 1
    reg = np.zeros(len(d["t"]), int)
    dr = direction(u)
    if ma:  # 111번: 롱은 종가 > ma일선, 숏은 종가 < ma일선일 때만
        c = u["c"]
        cs = np.cumsum(np.r_[0.0, c])
        sma = np.full(len(c), np.nan)
        sma[ma - 1:] = (cs[ma:] - cs[:-ma]) / ma
        with np.errstate(invalid="ignore"):
            dr = np.where((dr == 1) & (c > sma), 1, np.where((dr == -1) & (c < sma), -1, 0))
    reg[j >= 0] = dr[j[j >= 0]]
    newday = np.r_[False, j[1:] != j[:-1]]
    return reg, newday


def leg(d: dict, e: int, side: int, reg: np.ndarray, stop: bool):
    """e 시가 진입. (총손익, 청산 봉, 손절 여부, 방향 끝 봉)."""
    o, h, l = d["o"], d["h"], d["l"]
    px = o[e]
    sp = px * (1 - STOP * side)
    for j in range(e, len(o) - 1):
        if stop and ((side > 0 and l[j] <= sp) or (side < 0 and h[j] >= sp)):
            fill = min(o[j], sp) if side > 0 else max(o[j], sp)
            return (fill / px - 1) * side, j, True
        if reg[j] != side:
            return (o[j + 1] / px - 1) * side, j + 1, False
    return None


def run(d: dict, reg: np.ndarray, newday: np.ndarray, start: int, stop: bool, fund) -> list:
    n = len(d["o"])
    tr, i = [], start
    can = True  # 손절 뒤에는 다음 일봉 마감까지 기다린다
    while i < n - 2:
        if newday[i]:
            can = True
        side = int(reg[i])
        if side and can:
            res = leg(d, i + 1, side, reg, stop)
            if res is None:
                break
            g, x, stopped = res
            g = with_funding(g, d, i, x, side, fund)
            tr.append((i, side, float(g), int(x), float(d["t"][i]), stopped, STOP))
            if stopped:
                can = False
                i = x  # x 봉 마감에 새 일봉이 마감되면 그때 다시 볼 수 있다
            else:
                i = x - 1  # x-1 봉 마감에 방향이 바뀌었다. 그 봉에서 새 방향 진입을 본다
            continue
        i += 1
    return tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--day-shift", type=int, default=0)
    ap.add_argument("--perm", action="store_true")
    ap.add_argument("--spot", action="store_true", help="108번: 현물 5분봉, 펀딩 없음")
    ap.add_argument("--end", default="", help="YYYY-MM-DD 전까지만")
    ap.add_argument("--frame", type=int, default=288, help="상위 봉 크기(5분봉 개수): 48=4시간, 2016=주")
    ap.add_argument("--ma200", action="store_true", help="111번: 200일선 문")
    ap.add_argument("--warm-days", type=int, default=60)
    ap.add_argument("--circ", type=int, default=0, help="방향 시계열 원형 이동 대조 횟수")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / ("5m.csv" if args.spot else "futures/klines_5m.csv"))
    fund = None if args.spot else load_funding(base / "futures/funding.csv")
    if args.end:
        from datetime import datetime, timezone
        cut = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        keep = d5["t"] < cut
        d5 = {k: v[keep] for k, v in d5.items()}
    d = bars(d5, 0)
    t = d["t"]
    reg, newday = regime(d5, d, args.day_shift, args.frame, 200 if args.ma200 else 0)
    start = int(np.searchsorted(t, t[0] + args.warm_days * DAY))
    rep: dict = {"day_shift": args.day_shift, "frame": args.frame}
    bh = (d["o"][-1] / d["o"][start + 1] - 1) * 100
    rep["buy_hold_pct"] = round(float(bh), 1)
    for name, stop in (("R0", False), ("R1", True)):
        tr = run(d, reg, newday, start, stop, fund)
        r = {"all": summ(tr), "all_c10": summ(tr, 0.001),
             "old": summ([z for z in tr if z[4] < ystart(2024)]), "new": summ([z for z in tr if z[4] >= ystart(2024)]),
             "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
             "years": {y: summ([z for z in tr if ystart(y) <= z[4] < ystart(y + 1)]) for y in range(2020, 2027)},
             "hold_d": round(float(np.mean([(z[3] - z[0] - 1) / 96 for z in tr])), 2),
             "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
        if args.perm and tr:
            # 같은 진입 시각에서 방향만 무작위. 반대 방향은 같은 시각부터 그 방향의 반대 규칙으로 다시 계산한다.
            alt = []
            for z in tr:
                i, side = z[0], z[1]
                # 반대 방향 보유: 원래 보유가 끝나는 봉까지(같은 구간), 손절은 같은 규칙
                fake = np.where(np.arange(len(t)) < z[3] - 1, -side, 0) if not z[5] else None
                if z[5]:
                    # 손절 거래: 원래 방향 구간 끝을 찾는다
                    endr = leg(d, i + 1, side, reg, False)
                    fake = np.where(np.arange(len(t)) < endr[1] - 1, -side, 0)
                q = leg(d, i + 1, -side, fake, stop)
                alt.append(with_funding(q[0], d, i, q[1], -side, fund) if q else 0.0)
            g = np.array([z[2] for z in tr])
            a = np.array(alt)
            actual = g.mean()
            rng = np.random.default_rng(7)
            hits = sum(1 for _ in range(2000) if np.where(rng.random(len(g)) < 0.5, g, a).mean() >= actual)
            r["perm_p"] = round(hits / 2000, 4)
        if args.circ and tr:
            rng = np.random.default_rng(7)
            nl = len(reg)
            null = []
            for _ in range(args.circ):
                k = int(rng.integers(30 * 96, nl - 30 * 96))
                tt2 = run(d, np.roll(reg, k), np.roll(newday, k), start, stop, fund)
                null.append(summ(tt2).get("ev", 0.0))
            null = np.array(null)
            r["circ_null_ev"] = round(float(null.mean()), 3)
            r["circ_p"] = round(float((null >= r["all"]["ev"]).mean()), 4)
        rep[name] = r
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
