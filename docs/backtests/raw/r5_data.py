"""5차 탐색 자료: 선물 15분봉 + 프리미엄·펀딩·metrics(nonprice_data 그대로, 시작만 2020-01-01) + 1시간 외부 자료.

문서: docs/backtests/2026-11-08-btc-round5-hypotheses.md
외부(1시간): 코인베이스 BTC-USD(`~/.minerva/data/btc/coinbase/btcusd_1h.csv`, 끝 이후는 공개 API로 채움),
바이낸스 ETHUSDT 선물 1시간봉(공개 API, `~/.minerva/data/eth/futures_1h.csv`에 저장), Deribit DVOL 1시간
(`~/.minerva/data/btc/deribit/btc_dvol_1h.csv`). 1시간 값은 그 시간이 마감된 15분봉부터 쓴다.
실행: python3 r5_data.py --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import nonprice_data as ND
import numpy as np
from pine_data import FAPI, get

H1 = 3_600_000
M15 = 900_000
HOME = Path("~/.minerva").expanduser()
CB = HOME / "data/btc/coinbase/btcusd_1h.csv"
DVOL = HOME / "data/btc/deribit/btc_dvol_1h.csv"
ETH = HOME / "data/eth/futures_1h.csv"
START = 1_577_836_800_000  # 2020-01-01 UTC


def read_csv(path: Path, tcol: int, vcol: int) -> dict:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            out[int(x[tcol])] = float(x[vcol])
    return out


def fetch_eth() -> dict:
    rows = read_csv(ETH, 0, 4)
    s = max(rows) + H1 if rows else START
    now = int(time.time() * 1000)
    while s < now:
        k = get(f"{FAPI}/klines?symbol=ETHUSDT&interval=1h&limit=1500&startTime={s}")
        k = [x for x in k if int(x[6]) < now]
        if not k:
            break
        for x in k:
            rows[int(x[0])] = float(x[4])
        s = int(k[-1][0]) + H1
        time.sleep(0.3)
    ETH.parent.mkdir(parents=True, exist_ok=True)
    with ETH.open("w") as f:
        f.write("open_time,_,_,_,close\n")
        for t in sorted(rows):
            f.write(f"{t},,,,{rows[t]}\n")
    return rows


def fetch_cb(rows: dict) -> dict:
    """코인베이스 공개 캔들(한 번에 300개). 파일 끝 이후만 채운다(파일에는 쓰지 않는다)."""
    s = max(rows) + H1
    now = int(time.time() * 1000)
    while s < now - H1:
        e = min(s + 300 * H1, now)
        iso = lambda ms: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000))  # noqa: E731
        k = get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
                f"&start={iso(s)}&end={iso(e)}")  # fmt: skip
        for x in k or []:
            t = int(x[0]) * 1000
            if t + H1 <= now:
                rows[t] = float(x[4])
        s = e
        time.sleep(0.4)
    return rows


def hourly_to15(tc15: np.ndarray, rows: dict) -> np.ndarray:
    """15분봉 마감 시각 tc에 마감된 마지막 1시간봉 종가(그 봉 시작 + 1시간 ≤ tc). 3시간 넘게 비면 결측."""
    ts = np.array(sorted(rows), dtype=np.int64)
    v = np.array([rows[int(t)] for t in ts])
    return ND.asof(ts + H1, v, tc15, 3 * H1)


def build(out: Path) -> dict:
    ND.T0 = START
    info = ND.build(out)
    z = dict(np.load(out / "data15.npz"))
    tc = z["t"] + M15
    cb = fetch_cb(read_csv(CB, 0, 4))
    eth = fetch_eth()
    dv = read_csv(DVOL, 0, 4)
    # 바이낸스 선물 1시간 종가 = 그 시간 마지막 15분봉 종가(같은 asof로 맞춘다)
    c = z["c"]
    hr_end = (tc % H1) == 0
    bclose = np.where(hr_end, c, np.nan)
    last_b = np.full(len(c), np.nan)
    k = np.flatnonzero(hr_end)
    pos = np.searchsorted(k, np.arange(len(c)), side="right") - 1
    ok = pos >= 0
    last_b[ok] = bclose[k[pos[ok]]]
    cb15 = hourly_to15(tc, cb)
    eth15 = hourly_to15(tc, eth)
    dv15 = hourly_to15(tc, dv)
    with np.errstate(invalid="ignore", divide="ignore"):
        cbp = cb15 / last_b - 1
        r4b = last_b / ND.lagged(last_b, 16) - 1
        r4e = eth15 / ND.lagged(eth15, 16) - 1
        dvc = dv15 - ND.lagged(dv15, 16)
    np.savez(out / "extra15.npz", cbp=cbp, zcbp=ND.roll_z(cbp), zeth=ND.roll_z(r4e - r4b),
             dvol=dv15, zdvc=ND.roll_z(dvc))  # fmt: skip
    info.update({"cb_last": int(max(cb)), "eth_rows": len(eth), "eth_first": int(min(eth)),
                 "dvol_first": int(min(dv)), "dvol_last": int(max(dv))})  # fmt: skip
    (out / "data.json").write_text(json.dumps(info, indent=1))
    return info


def load(out: Path) -> dict:
    b = ND.load(out)
    e = np.load(out / "extra15.npz")
    b.update({"x_" + k: e[k] for k in e.files})
    return b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    p = Path(ap.parse_args().out).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    i = build(p)
    print(json.dumps({k: v for k, v in i.items() if k not in ("fut_gaps_after", "prem_gaps_after")},
                     indent=1))  # fmt: skip
    print("gaps", i["fut_gaps_after"][:10], i.get("prem_gaps_after", [])[:10])
