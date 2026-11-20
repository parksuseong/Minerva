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
from datetime import date
from pathlib import Path

from minerva.strategy import indicators as ind

KLINE_URL = "https://fapi.binance.com/fapi/v1/klines"
START_MS = 1567382400000  # 2019-09-02 주(바이낸스 BTCUSDT 선물 첫 주)
MAJOR = 0.50
MINOR = 0.15
DAY_MS = 86_400_000
# 봉마다 큰 흐름·작은 파동 비율(사용자 2026-11-20 "월봉이나 일봉 4시간봉까지").
# 일봉·4시간은 같은 15%면 파동이 너무 적어 화면에서 비슷한 개수가 되게 줄였다(2019~ 일봉 111번,
# 4시간 2년 78번 전환). 숫자는 보기용으로 고른 값이고 검증한 값이 아니다.
TF = {
    "1M": {"name": "월봉", "unit": "달", "major": 0.50, "minor": 0.15, "days": None},
    "1w": {"name": "주봉", "unit": "주", "major": 0.50, "minor": 0.15, "days": None},
    "1d": {"name": "일봉", "unit": "일", "major": 0.30, "minor": 0.10, "days": None},
    "4h": {"name": "4시간봉", "unit": "봉", "major": 0.15, "minor": 0.05, "days": 730},
}
# 아래 SCR 칸: 원문 설정(Sto Fast %K 5, CCI 20, RSI 14)과 과열·침체 선
SCR = {"stoch": 5, "cci": 20, "rsi": 14, "stoch_lo": 20, "stoch_hi": 80,
       "cci_lo": -100, "cci_hi": 100, "rsi_lo": 30, "rsi_hi": 70}


def out_path(tf: str = "1w") -> Path:
    """주봉은 예전 이름(regime.json) 그대로, 나머지는 regime_<봉>.json."""
    week = Path(
        os.environ.get("MINERVA_BTC_REGIME_OUT", "~/.minerva/btc_regime/regime.json")
    ).expanduser()
    return week if tf == "1w" else week.with_name(f"regime_{tf}.json")


def fetch(interval: str, start_ms: int, symbol: str = "BTCUSDT") -> list[list]:
    """공개 봉을 1500개씩 이어 받는다(한 번 무게 10, 엔진과 같은 IP라 0.5초 쉼)."""
    out: list[list] = []
    while True:
        q = f"{KLINE_URL}?symbol={symbol}&interval={interval}&startTime={start_ms}&limit=1500"
        req = urllib.request.Request(q, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            got = json.load(r)
        out += got
        if len(got) < 1500:
            return out
        start_ms = int(got[-1][0]) + 1
        time.sleep(0.5)


def fetch_weekly(symbol: str = "BTCUSDT") -> list[list]:
    return fetch("1w", START_MS, symbol)


def scan(c: list[float], th: float) -> dict:
    """종가 지그재그를 한 주씩 앞으로만 훑는다(그 주까지의 종가만 씀).

    - `piv`: 확정된 고점(H)·저점(L) (막대 번호, 종류, 확정된 주 번호)
    - `level[i]`: i주 마감 때 알 수 있는 전환 가격. 다음 종가가 이 값을 넘으면 파동이 바뀐다
      (오름 파동이면 지금까지 최고 종가 × (1−th) 아래, 내림 파동이면 최저 종가 × (1+th) 위)
    - `dir[i]`: i주 마감 때 진행 중인 파동 방향(1 오름, −1 내림, 0 아직 모름)
    """
    n = len(c)
    lo = hi = 0
    d = 0
    piv: list[tuple[int, str, int]] = []
    level: list[float | None] = [None] * n
    dirs: list[int] = [0] * n
    for i in range(1, n):
        if d == 0:
            if c[i] > c[hi]:
                hi = i
            if c[i] < c[lo]:
                lo = i
            if c[i] >= c[lo] * (1 + th):
                piv = [(lo, "L", i)]
                d = 1
                hi = i
            elif c[i] <= c[hi] * (1 - th):
                piv = [(hi, "H", i)]
                d = -1
                lo = i
        elif d == 1:
            if c[i] > c[hi]:
                hi = i
            elif c[i] <= c[hi] * (1 - th):
                piv.append((hi, "H", i))
                d = -1
                lo = i
        else:
            if c[i] < c[lo]:
                lo = i
            elif c[i] >= c[lo] * (1 + th):
                piv.append((lo, "L", i))
                d = 1
                hi = i
        dirs[i] = d
        if d == 1:
            level[i] = c[hi] * (1 - th)
        elif d == -1:
            level[i] = c[lo] * (1 + th)
    return {"piv": piv, "level": level, "dir": dirs, "hi": hi, "lo": lo, "d": d}


def zigzag(c: list[float], th: float) -> list[tuple[int, str]]:
    """종가 지그재그: th 비율 되돌림마다 고점(H)·저점(L) 확정. 마지막 극점도 넣는다."""
    s = scan(c, th)
    piv = [(i, k) for i, k, _ in s["piv"]]
    if s["d"] == 1:
        piv.append((s["hi"], "H"))
    elif s["d"] == -1:
        piv.append((s["lo"], "L"))
    if piv and piv[0][0] > 0:
        piv.insert(0, (0, "H" if piv[0][1] == "L" else "L"))
    return piv


def _sma(x: list[float], k: int) -> list[float | None]:
    return [None if i < k - 1 else sum(x[i - k + 1 : i + 1]) / k for i in range(len(x))]


def labels(
    c: list[float], major_pct: float = MAJOR, minor_pct: float = MINOR
) -> tuple[list[str | None], list[tuple[int, str]]]:
    n = len(c)
    major = zigzag(c, major_pct)
    minor = zigzag(c, minor_pct)

    def majdir(i: int) -> int:
        for k in range(len(major) - 1):
            if major[k][0] <= i < major[k + 1][0]:
                return 1 if major[k + 1][1] == "H" else -1
        return (1 if major[-1][1] == "H" else -1) if major else 1

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


def build(
    klines: list[list], now_ms: int, tf: str = "1w"
) -> dict:
    """주봉 kline → 화면 데이터. 아직 안 끝난 주는 `open: true`로 넣고 이름표 계산에도 쓴다."""
    t = [int(b[0]) // 1000 for b in klines]
    o = [float(b[1]) for b in klines]
    h = [float(b[2]) for b in klines]
    lo = [float(b[3]) for b in klines]
    c = [float(b[4]) for b in klines]
    cfg = TF[tf]
    reg, major = labels(c, cfg["major"], cfg["minor"])
    s20, s50 = _sma(c, 20), _sma(c, 50)
    r1 = lambda v: None if v is None else round(v, 1)  # noqa: E731
    bars = [
        {"t": t[i], "o": o[i], "h": h[i], "l": lo[i], "c": c[i],
         "s20": r1(s20[i]), "s50": r1(s50[i]), "r": reg[i]}
        for i in range(len(c))
    ]
    if klines and int(klines[-1][6]) >= now_ms:
        bars[-1]["open"] = True
    ib = [ind.Bar(date(2000, 1, 1), o[i], h[i], lo[i], c[i]) for i in range(len(c))]
    st, cc, rs = ind.stoch_fast_k(ib, SCR["stoch"]), ind.cci(ib, SCR["cci"]), ind.rsi(c, SCR["rsi"])
    for i, b in enumerate(bars):
        b["st"], b["cci"], b["rsi"] = r1(st[i]), r1(cc[i]), r1(rs[i])
    # 그때그때 알 수 있던 전환 가격(사후 아님): 그 주 마감까지의 종가만 쓴다
    mi, mj = scan(c, cfg["minor"]), scan(c, cfg["major"])
    for i, b in enumerate(bars):
        b["sw"], b["swd"] = r1(mi["level"][i]), mi["dir"][i]
        b["mj"], b["mjd"] = r1(mj["level"][i]), mj["dir"][i]
    turns = [
        {"kind": k, "t": t[p], "c": c[p], "known_t": t[q], "known_c": c[q], "lag": q - p}
        for p, k, q in mi["piv"]
    ]
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
        "tf": tf, "tf_name": cfg["name"], "unit": cfg["unit"],
        "major_pct": cfg["major"], "minor_pct": cfg["minor"], "scr": SCR,
        "major": big,
        "bars": bars,
        "segments": segs,
        "turns": turns,
        "now": {
            "minor": _now(mi, c, t, cfg["minor"]),
            "major": _now(mj, c, t, cfg["major"]),
        },
    }


def _now(s: dict, c: list[float], t: list[int], th: float) -> dict | None:
    """마지막 주 마감 기준: 진행 중인 파동, 그 극값, 바뀌는 종가."""
    if s["d"] == 0:
        return None
    i = s["hi"] if s["d"] == 1 else s["lo"]
    return {
        "dir": "up" if s["d"] == 1 else "down",
        "ext_t": t[i], "ext_c": c[i],
        "level": round(s["level"][-1], 1),
        "pct": th,
    }


def write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true", help="파일을 쓰지 않고 요약만 출력")
    ap.add_argument("--tf", choices=list(TF), action="append", help="봉(여러 번 가능, 기본 전부)")
    a = ap.parse_args(argv)
    bad = 0
    for tf in a.tf or list(TF):
        now = int(time.time() * 1000)
        days = TF[tf]["days"]
        try:
            kl = fetch(tf, now - days * DAY_MS if days else START_MS)
        except OSError as e:  # 한 봉이 실패해도 나머지는 만든다(지난 파일은 그대로)
            print(f"{tf}: 받기 실패 {e}", file=sys.stderr)
            bad += 1
            continue
        if len(kl) < 60:
            print(f"{tf}: 봉이 너무 적음({len(kl)}), 파일을 그대로 둠", file=sys.stderr)
            bad += 1
            continue
        data = build(kl, now, tf)
        last, seg = data["bars"][-1], data["segments"][-1]
        print(f"{tf}: bars={len(data['bars'])} last={last['r']} open={last.get('open', False)} "
              f"seg={seg['r']} n={seg['n']} turns={len(data['turns'])}")
        if not a.dry:
            write(data, out_path(tf))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
