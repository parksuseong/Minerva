"""비트코인 주봉 장세 지도(상승·조정·하락·반등·횡보) 데이터를 매일 다시 만든다.

사용자 2026-11-18 "상승/조정/하락/반등/횡보로 나눠라" → 종가 고점·저점 파동 구조로 칠한 지도,
2026-11-20 "탭으로 만들어줘 매일 자정에 업데이트" → `minerva-btc-regime.timer`(매일 00:00 KST).

- 데이터: 바이낸스 USDT-M 선물 BTCUSDT 주봉(공개 GET 한 번, 무게 5). 주문·키 없음.
- 큰 파동: 종가가 고점에서 50% 빠지면 하락장, 저점에서 50% 오르면 상승장(`MAJOR`).
- 작은 파동: 종가 15% 되돌림으로 파동을 끊는다(`MINOR`). 상승장 안 오름=상승, 내림=조정,
  하락장 안 내림=하락, 오름=반등. 직전 고점·저점을 다 못 넘고 안쪽에서 끝난 파동=횡보.
- 고점·저점이 확정된 뒤에야 아는 사후 이름표다. 마지막 구간과 진행 중인 주는 바뀔 수 있다.
- 결과: `~/.minerva/btc_regime/regime.json` → `/api/btc/regime`
  → 화면 🧭 장세 탭(`static/btc_regime.html`).

실행: `python -m minerva.btc_regime [--dry]`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

KLINE_URL = "https://fapi.binance.com/fapi/v1/klines"
START_MS = 1567382400000  # 2019-09-02 주(바이낸스 BTCUSDT 선물 첫 주)
MAJOR = 0.50
MINOR = 0.15


def out_path() -> Path:
    return Path(
        os.environ.get("MINERVA_BTC_REGIME_OUT", "~/.minerva/btc_regime/regime.json")
    ).expanduser()


def fetch_weekly(symbol: str = "BTCUSDT") -> list[list]:
    q = f"{KLINE_URL}?symbol={symbol}&interval=1w&startTime={START_MS}&limit=1000"
    req = urllib.request.Request(q, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def zigzag(c: list[float], th: float) -> list[tuple[int, str]]:
    """종가 지그재그: th 비율 되돌림마다 고점(H)·저점(L) 확정. 마지막 극점도 넣는다."""
    n = len(c)
    lo = hi = 0
    d = 0
    piv: list[tuple[int, str]] = []
    for i in range(1, n):
        if d == 0:
            if c[i] > c[hi]:
                hi = i
            if c[i] < c[lo]:
                lo = i
            if c[i] >= c[lo] * (1 + th):
                piv = [(lo, "L")]
                d = 1
                hi = i
            elif c[i] <= c[hi] * (1 - th):
                piv = [(hi, "H")]
                d = -1
                lo = i
        elif d == 1:
            if c[i] > c[hi]:
                hi = i
            elif c[i] <= c[hi] * (1 - th):
                piv.append((hi, "H"))
                d = -1
                lo = i
        else:
            if c[i] < c[lo]:
                lo = i
            elif c[i] >= c[lo] * (1 + th):
                piv.append((lo, "L"))
                d = 1
                hi = i
    if d == 1:
        piv.append((hi, "H"))
    elif d == -1:
        piv.append((lo, "L"))
    if piv and piv[0][0] > 0:
        piv.insert(0, (0, "H" if piv[0][1] == "L" else "L"))
    return piv


def _sma(x: list[float], k: int) -> list[float | None]:
    return [None if i < k - 1 else sum(x[i - k + 1 : i + 1]) / k for i in range(len(x))]


def labels(c: list[float]) -> tuple[list[str | None], list[tuple[int, str]]]:
    n = len(c)
    major = zigzag(c, MAJOR)
    minor = zigzag(c, MINOR)

    def majdir(i: int) -> int:
        for k in range(len(major) - 1):
            if major[k][0] <= i < major[k + 1][0]:
                return 1 if major[k + 1][1] == "H" else -1
        return 1 if major[-1][1] == "H" else -1

    reg: list[str | None] = [None] * n
    for k in range(len(minor) - 1):
        a, b = minor[k][0], minor[k + 1][0]
        up = minor[k + 1][1] == "H"
        hs = [x[0] for x in minor[: k + 1] if x[1] == "H"]
        ls = [x[0] for x in minor[: k + 1] if x[1] == "L"]
        inside = False
        if up and hs and len(ls) >= 2:
            inside = c[b] < c[hs[-1]] and c[a] > c[ls[-2]]
        if (not up) and ls and len(hs) >= 2:
            inside = c[b] > c[ls[-1]] and c[a] < c[hs[-2]]
        m = majdir((a + b) // 2)
        if inside:
            r = "side"
        elif m == 1:
            r = "up" if up else "corr"
        else:
            r = "bounce" if up else "down"
        for i in range(a + 1, b + 1):
            reg[i] = r
    if n > 1:
        reg[0] = reg[1]
    return reg, major


def build(klines: list[list], now_ms: int) -> dict:
    """주봉 kline → 화면 데이터. 아직 안 끝난 주는 `open: true`로 넣고 이름표 계산에도 쓴다."""
    t = [int(b[0]) // 1000 for b in klines]
    o = [float(b[1]) for b in klines]
    h = [float(b[2]) for b in klines]
    lo = [float(b[3]) for b in klines]
    c = [float(b[4]) for b in klines]
    reg, major = labels(c)
    s20, s50 = _sma(c, 20), _sma(c, 50)
    r1 = lambda v: None if v is None else round(v, 1)  # noqa: E731
    bars = [
        {"t": t[i], "o": o[i], "h": h[i], "l": lo[i], "c": c[i],
         "s20": r1(s20[i]), "s50": r1(s50[i]), "r": reg[i]}
        for i in range(len(c))
    ]
    if klines and int(klines[-1][6]) >= now_ms:
        bars[-1]["open"] = True
    segs: list[dict] = []
    for b in bars:
        if not b["r"]:
            continue
        if segs and segs[-1]["r"] == b["r"]:
            segs[-1].update(to=b["t"], c1=b["c"])
            segs[-1]["n"] += 1
        else:
            segs.append(
                {"r": b["r"], "from": b["t"], "to": b["t"], "n": 1, "c0": b["o"], "c1": b["c"]}
            )
    # 지금 큰 흐름: 마지막 확정 큰 극점부터
    big = None
    if len(major) >= 2:
        i, k = major[-2]
        big = {"dir": "down" if k == "H" else "up", "from": t[i], "price": c[i]}
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now_ms / 1000)),
        "symbol": "BTCUSDT",
        "major_pct": MAJOR, "minor_pct": MINOR,
        "major": big,
        "bars": bars,
        "segments": segs,
    }


def write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true", help="파일을 쓰지 않고 요약만 출력")
    a = ap.parse_args(argv)
    kl = fetch_weekly()
    if len(kl) < 60:
        print(f"주봉이 너무 적음({len(kl)}), 파일을 그대로 둠", file=sys.stderr)
        return 1
    data = build(kl, int(time.time() * 1000))
    last = data["bars"][-1]
    seg = data["segments"][-1]
    print(f"bars={len(data['bars'])} last={last['r']} open={last.get('open', False)} "
          f"seg={seg['r']} n={seg['n']} major={data['major']}")
    if not a.dry:
        write(data, out_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
