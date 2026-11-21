"""자동매매 종목 범위: 바이낸스 USDⓈ-M 거래량 상위 50 (사용자 2026-11-12
"선물로 매매할 수 있는 리스트 중에 거래량 많은 상위 50개로 확대해. 우선순위는 너가 알아서").

`python -m minerva.btc_universe refresh [--dry]` (주 1회, `minerva-btc-universe.timer`):
1. exchangeInfo에서 TRADING · USDT · 무기한(PERPETUAL, TRADIFI_PERPETUAL) 종목만 고르고
2. 24시간 거래대금(`quoteVolume`) 상위 50개를 고른 뒤
3. 종목마다 엔진과 같은 SCR ① 15분봉 신호로 롱↔숏 갈아타기(선정 백테스트 3배, 손절 3%, 수수료
   편도 0.05%, 다음 봉 시가 진입)를 최근 365일(약 1년, 상장이 늦으면 상장 이후) 백테스트한다.
   엔진은 2026-11-17~18 1배였고 11-18부터 다시 3배다. 선정도 3배.
   화면 실험실
   `switchSim`(static/scrlab_core.js)을 그대로 옮긴 `switch_sim`이다.
4. `~/.minerva/btc_trade/universe.json`에 원자적으로 쓴다.

엔진은 이 파일이 14일 안이면 그 종목들을, 아니면 예전 네 종목을 쓴다(`load`).
우선순위는 백테스트 승률(사용자 "승률 높은 걸 우선순위로"). 거래 20회 미만이거나 자료 60일 미만은
맨 뒤로 보낸다(표본이 작아 승률을 믿기 어렵다).

같은 IP를 실전 엔진이 쓰므로 요청 무게를 아낀다(분당 약 1000 이하, 바이낸스 한도 2400):
봉은 1000개씩(무게 5) 0.4초 간격, `X-MBX-USED-WEIGHT-1M`이 1200을 넘으면 다음 분까지 쉬고,
429·418이면 바로 멈춘다(계속 보내면 IP 차단). 엔진 타이머(15분마다 :20초) 앞뒤 1분은 비켜 간다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

from minerva.kis import KST, _write_private
from minerva.strategy import scr_lab

log = logging.getLogger(__name__)

BASE = "https://fapi.binance.com"
TOP_N = 50
# 거래량 상위 50 밖이어도 늘 넣는 종목(사용자 요청). 승률 바닥은 똑같이 적용된다.
EXTRA_SYMBOLS = ("AAPLUSDT",)  # 2026-11-12 "응 넣어봐"(1년 승률 62.7%)
CONTRACTS = ("PERPETUAL", "TRADIFI_PERPETUAL")
FRESH_DAYS = 14
MIN_TRADES = 20
MIN_DAYS = 60
# 2026-11-17 시점 시뮬레이션 뒤 365일로(사용자 위임 "가장 최선을 네가 알아서").
# 그 전 182일(11-13 "최근 6개월만")
WINDOW_DAYS = 365
WARMUP = 300
FEE = 0.0005
# 종목 선정 백테스트 배율. 엔진(btc_trade.LEVERAGE)은 2026-11-17부터 1배지만, 365일 창을 고른
# 시점 시뮬레이션이 3배 switch_sim으로 종목을 골랐으므로 선정은 3배 그대로 둔다
LEVERAGE = 3
STOP = 0.03
BAR_MS = 15 * 60_000
PAGE = 1000  # 봉 1000개 = 무게 5 (1500개는 무게 10)
GAP_S = 0.4
WEIGHT_SOFT = 1200
RULE = (
    "SCR ① 15분봉(Stoch 5·CCI 20·RSI 14, W3, 3/3, 넘어옴) 롱↔숏 갈아타기 · 선정 백테스트 3배 · "
    "최근 365일(약 1년, 상장이 늦으면 상장 이후) · 손절 3% · 수수료 편도 0.05% · "
    "다음 봉 시가 진입 · 엔진 실매매는 3배"
)
DIR = Path(os.environ.get("MINERVA_BTC_TRADE_DIR", "~/.minerva/btc_trade")).expanduser()
# 사용자 2026-11-16 "승률관계없이 최근 일주일동안 하루에 3프로 이상씩 움직인 종목있으면 매매하자"
# → 같은 날 "2프로 이상에 백테스트 수익 +만": 최근 마감 일봉 7개의 하루 변동폭((고가−저가)/시가)
# 평균이 2% 이상인 종목만 새로 들어간다. 매일 09:10 KST `ranges`(minerva-btc-ranges.timer)가 쓴다.
RANGE_DAYS = 7
MIN_DAY_RANGE_PCT = 2.0
RANGE_FRESH_HOURS = 48
# 사용자 2026-11-18 "승률 60프로이상인거로 해. Bz제외하고": 최근 7일 엔진 신호 백테스트(선정과 같은
# switch_sim) 승률 ≥ 60%인 종목만 새로 들어간다(1년 수익 + 조건 대신). 거래 3회 미만은 승률이
# 우연(1/1 = 100%)이라 뺀다. 매일 09:10 `ranges`가 변동폭과 함께 recent.json을 쓴다.
RECENT_DAYS = 7
MIN_RECENT_WINRATE = 60.0
MIN_RECENT_TRADES = 3
DAY_MS = 86_400_000

_sleep = time.sleep


class RateLimited(RuntimeError):
    """429·418: 더 보내면 IP가 막힌다(실전 엔진도 같은 IP)."""


# ---------------------------------------------------------------- 읽기(엔진·화면)


def load(path: Path, now_ms: int, fresh_days: int = FRESH_DAYS) -> dict | None:
    """쓸 수 있는 universe.json이면 그 내용, 없거나·깨졌거나·오래됐으면 None."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    syms = data.get("symbols")
    stats = data.get("stats")
    if not isinstance(syms, list) or not syms or not isinstance(stats, dict):
        return None
    if not all(isinstance(s, str) and s.endswith("USDT") for s in syms):
        return None
    try:
        at = datetime.strptime(str(data.get("updated_at")), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
    except ValueError:
        return None
    age_ms = now_ms - at.timestamp() * 1000
    if age_ms > fresh_days * 86_400_000 or age_ms < -86_400_000:
        return None
    return data


def low_sample(st: dict | None) -> bool:
    """표본이 작아 승률을 믿기 어려운 종목(우선순위 맨 뒤)."""
    st = st or {}
    return int(st.get("trades") or 0) < MIN_TRADES or float(st.get("days") or 0) < MIN_DAYS


def priority(symbols: list[str], stats: dict) -> list[str]:
    """백테스트 승률 높은 순, 같으면 거래량 순위. 표본이 작은 종목은 맨 뒤(그 안에서도 같은 순)."""

    def key(s: str):
        st = stats.get(s) or {}
        rank = st.get("rank") or symbols.index(s) + 1
        return (low_sample(st), -float(st.get("winrate") or 0), rank)

    return sorted(symbols, key=key)


# ---------------------------------------------------------------- 백테스트(화면 switchSim)


def switch_sim(
    o: list[float],
    h: list[float],
    lo: list[float],
    c: list[float],
    sigs: list[tuple[int, int]],
    stop: float = STOP,
    tp: float = 0.0,
    lev: float = LEVERAGE,
    fee: float = FEE,
) -> dict:
    """화면 `switchSim`과 같은 계산(같은 순서). sigs = [(봉 번호, +1 롱 / -1 숏)].

    신호 다음 봉 시가에 들어가고 반대 신호면 다음 봉 시가에 뒤집는다. 진입가 ±stop에 고가·저가가
    닿으면 그 값(시가가 이미 넘었으면 시가)에 나가고 다음 신호까지 쉰다. 같은 봉 롱·숏은 건너뛴다.
    손절이 없거나 청산가(±1/lev)보다 멀면 청산가에서 강제 청산. 마지막 포지션은 종가로 정리.
    """
    tp = tp if tp > 0 else 0.0
    liq_at = 1 / lev
    use_stop = 0 < stop < liq_at
    stop_ = stop if use_stop else liq_at
    at: dict[int, int] = {}
    for i, d in sigs:
        at[i] = 0 if i in at and at[i] != d else d
    st = {"eq": 1.0, "pos": 0, "entry": 0.0, "opened": 0, "trades": 0, "wins": 0}
    peak, mdd = 1.0, 0.0
    stops = tps = liqs = 0
    exits: list[dict] = []

    def close(px: float, j: int, kind: str) -> None:
        pos, entry = st["pos"], st["entry"]
        net = lev * pos * (px / entry - 1) - lev * fee * (1 + px / entry)
        st["eq"] = max(0.0, st["eq"] * (1 + net))
        st["trades"] += 1
        if net > 0:
            st["wins"] += 1
        exits.append({"i": j, "from": st["opened"], "dir": pos, "kind": kind, "net": net * 100})
        st["pos"] = 0

    first = min([i + 1 for i, _ in sigs] + [len(c)])
    for j in range(first, len(c)):
        d = at.get(j - 1)
        if d and d != st["pos"]:
            if st["pos"]:
                close(o[j], j, "flip")
            st.update(pos=d, entry=o[j], opened=j)
        pos = st["pos"]
        if pos:
            entry = st["entry"]
            sp = entry * (1 - pos * stop_)
            tpx = entry * (1 + pos * tp) if tp else 0.0
            if (lo[j] <= sp) if pos > 0 else (h[j] >= sp):
                close(min(o[j], sp) if pos > 0 else max(o[j], sp), j, "stop" if use_stop else "liq")
                if use_stop:
                    stops += 1
                else:
                    liqs += 1
            elif tp and ((h[j] >= tpx) if pos > 0 else (lo[j] <= tpx)):
                close(max(o[j], tpx) if pos > 0 else min(o[j], tpx), j, "tp")
                tps += 1
        pos = st["pos"]
        mark = st["eq"] * (1 + lev * pos * (c[j] / st["entry"] - 1)) if pos else st["eq"]
        if mark > peak:
            peak = mark
        if peak > 0:
            mdd = max(mdd, (1 - mark / peak) * 100)
    holding = st["pos"] != 0
    if holding:
        close(c[-1], len(c) - 1, "end")
    return {
        "trades": st["trades"],
        "wins": st["wins"],
        "stops": stops,
        "tps": tps,
        "liqs": liqs,
        "ret": (st["eq"] - 1) * 100,
        "mdd": mdd,
        "holding": holding,
        "exits": exits,
    }


def backtest(rows: list[list], params: dict, from_ms: int) -> dict:
    """마감 봉 rows(바이낸스 모양)로 엔진 신호를 계산하고, from_ms 이후 신호만으로 시뮬레이션한다
    (화면 실험실이 보이는 기간 신호만 넣는 것과 같다). 지표는 앞쪽 봉으로 자리 잡는다."""
    o = [float(r[1]) for r in rows]
    h = [float(r[2]) for r in rows]
    lo = [float(r[3]) for r in rows]
    c = [float(r[4]) for r in rows]
    t = [int(r[0]) for r in rows]
    sig = scr_lab.signals(h, lo, c, params)
    start = next((i for i, x in enumerate(t) if x >= from_ms), len(t))
    start = max(start, WARMUP)
    sigs = [(i, 1) for i in sig["long"] if i >= start]
    sigs += [(i, -1) for i in sig["short"] if i >= start]
    sigs.sort(key=lambda x: (x[0], -x[1]))
    res = switch_sim(o, h, lo, c, sigs, lev=LEVERAGE)  # 기본값은 불러올 때 묶인다
    days = (t[-1] + BAR_MS - t[start]) / 86_400_000 if start < len(t) else 0.0
    return {
        "trades": res["trades"],
        "wins": res["wins"],
        "winrate": round(res["wins"] / res["trades"] * 100, 1) if res["trades"] else 0.0,
        "ret": round(res["ret"], 1),
        "mdd": round(res["mdd"], 1),
        "stops": res["stops"],
        "days": round(days, 1),
        # 끝에 아직 열린 포지션(종가로 정리해 위 수에 들어감): 최근 7일 승률은 이걸 뺀다
        "holding": bool(res["holding"]),
        "end_win": bool(res["holding"] and res["exits"] and res["exits"][-1]["net"] > 0),
    }


# ---------------------------------------------------------------- 공개 시세(무게 아끼기)


def _engine_window(now: float) -> bool:
    """엔진 타이머(15분마다 :20초) 앞뒤 1분이면 True."""
    sec = now % 900
    return 20 - 60 <= sec <= 20 + 60 or sec >= 900 - 40


def get_json(path: str, params: dict | None = None, timeout: float = 20):
    """공개 GET. 무게 헤더를 보고 쉬고, 429·418이면 RateLimited."""
    while _engine_window(time.time()):
        _sleep(5)
    query = "&".join(f"{k}={v}" for k, v in (params or {}).items())
    url = f"{BASE}{path}" + (f"?{query}" if query else "")
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-universe"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise RateLimited(
                f"HTTP {e.code}, Retry-After {e.headers.get('Retry-After')}"
            ) from None
        raise
    if used > WEIGHT_SOFT:
        wait = 61 - time.time() % 60
        log.info("weight %d > %d → %.0fs 쉼", used, WEIGHT_SOFT, wait)
        _sleep(wait)
    return data


def fetch_15m(symbol: str, start_ms: int, end_ms: int, get=get_json) -> list[list]:
    """start_ms부터 end_ms 전에 마감한 15분봉(1000개씩, 간격을 두고)."""
    out: list[list] = []
    t = start_ms
    while t < end_ms:
        page = get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": "15m", "startTime": t, "limit": PAGE},
        )
        _sleep(GAP_S)
        if not isinstance(page, list) or not page:
            break
        out.extend([r[:7] for r in page if int(r[6]) < end_ms])
        nxt = int(page[-1][0]) + BAR_MS
        if nxt <= t or len(page) < PAGE:
            break
        t = nxt
    seen: set[int] = set()
    rows = []
    for r in out:
        if int(r[0]) not in seen:
            seen.add(int(r[0]))
            rows.append(r)
    return rows


def select_top(
    info: dict, tickers: list[dict], n: int = TOP_N, extra: tuple = ()
) -> list[dict]:
    """거래 가능한 USDT 무기한 중 24시간 거래대금 상위 n. [{symbol, quote_volume, contract,
    underlying, onboard}] 거래대금 순. extra 종목은 상위 n 밖이면 뒤에 붙인다(거래 가능할 때만)."""
    ok = {
        s["symbol"]: s
        for s in info.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and s.get("contractType") in CONTRACTS
    }
    rows = []
    for t in tickers:
        s = ok.get(t.get("symbol"))
        if not s:
            continue
        try:
            qv = float(t.get("quoteVolume") or 0)
        except ValueError:
            continue
        rows.append(
            {
                "symbol": s["symbol"],
                "quote_volume": qv,
                "contract": s["contractType"],
                "underlying": s.get("underlyingType"),
                "onboard": s.get("onboardDate"),
            }
        )
    rows.sort(key=lambda r: -r["quote_volume"])
    top = rows[:n]
    have = {r["symbol"] for r in top}
    top += [r for r in rows[n:] if r["symbol"] in extra and r["symbol"] not in have]
    return top


def refresh(
    out: Path,
    params: dict,
    now_ms: int | None = None,
    get: Callable = get_json,
    progress: Callable[[str], None] = log.info,
) -> dict:
    """상위 50을 고르고 종목마다 백테스트해 out에 쓴다. 한 종목씩(메모리) 처리한다.
    한 종목 실패는 그 종목 통계만 비운다(표본 0 → 우선순위 맨 뒤). 429·418은 전체를 멈춘다."""
    now_ms = now_ms or int(time.time() * 1000)
    info = get("/fapi/v1/exchangeInfo")
    tickers = get("/fapi/v1/ticker/24hr")
    top = select_top(info, tickers, extra=EXTRA_SYMBOLS)
    end = now_ms - now_ms % BAR_MS  # 진행 중인 봉은 뺀다
    from_ms = end - WINDOW_DAYS * 86_400_000
    stats: dict[str, dict] = {}
    for rank, row in enumerate(top, 1):
        sym = row["symbol"]
        st = {
            "rank": rank,
            "quote_volume": round(row["quote_volume"]),
            "contract": row["contract"],
            "underlying": row["underlying"],
        }
        bars: list[list] = []
        try:
            bars = fetch_15m(sym, from_ms - WARMUP * BAR_MS, end, get)
            st.update(backtest(bars, params, from_ms) if len(bars) > WARMUP + 2 else _empty())
        except RateLimited:
            raise
        except (OSError, ValueError, KeyError, IndexError) as e:
            st.update(_empty(), error=f"{type(e).__name__}: {e}"[:200])
        stats[sym] = st
        progress(f"{rank:2d} {sym} 승률 {st['winrate']}% 거래 {st['trades']} 일 {st['days']}")
        del bars  # 다음 종목 전에 놓는다
    data = {
        "updated_at": datetime.fromtimestamp(now_ms / 1000, KST).strftime("%Y-%m-%d %H:%M"),
        "rule": RULE,
        "symbols": [r["symbol"] for r in top],
        "stats": stats,
    }
    _write_private(out, json.dumps(data, ensure_ascii=False, indent=1))
    return data


def day_range(bars: list[list], now_ms: int, days: int = RANGE_DAYS) -> float | None:
    """마감한 일봉 마지막 days개의 (고가−저가)/시가 평균(%). 봉이 모자라면 None."""
    done = [b for b in bars if int(b[0]) + DAY_MS <= now_ms]
    done = done[-days:]
    if len(done) < days:
        return None
    pcts = [(float(b[2]) - float(b[3])) / float(b[1]) * 100 for b in done if float(b[1]) > 0]
    if len(pcts) < days:
        return None
    return round(sum(pcts) / len(pcts), 2)


def refresh_ranges(
    out: Path,
    symbols: list[str],
    now_ms: int | None = None,
    get: Callable = get_json,
) -> dict:
    """종목마다 일봉 몇 개(무게 1)를 받아 7일 평균 하루 변동폭을 out에 쓴다. 한 종목 실패는
    빼고 간다(파일에 없는 종목은 엔진이 새로 들어가지 않는다). 429·418은 전체를 멈춘다."""
    now_ms = now_ms or int(time.time() * 1000)
    ranges: dict[str, float] = {}
    errors: dict[str, str] = {}
    # 한 번 실패한 종목은 48시간 안의 앞 값을 이어 쓴다(하루 내내 막히지 않게)
    before = load_ranges(out, now_ms) or {}
    for sym in symbols:
        try:
            params = {"symbol": sym, "interval": "1d", "limit": RANGE_DAYS + 2}
            try:
                bars = get("/fapi/v1/klines", params)
            except OSError:
                _sleep(2)
                bars = get("/fapi/v1/klines", params)  # 한 번 더
            r = day_range(bars, now_ms)
            if r is None:
                errors[sym] = "일봉 부족"
            else:
                ranges[sym] = r
        except RateLimited:
            raise
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            errors[sym] = f"{type(e).__name__}: {e}"[:200]
            if sym in before:
                ranges[sym] = before[sym]
        _sleep(GAP_S / 2)
    data = {
        "updated_at": datetime.fromtimestamp(now_ms / 1000, KST).strftime("%Y-%m-%d %H:%M"),
        "days": RANGE_DAYS,
        "min_pct": MIN_DAY_RANGE_PCT,
        "ranges": ranges,
        "errors": errors,
    }
    _write_private(out, json.dumps(data, ensure_ascii=False, indent=1))
    return data


def load_ranges(path: Path, now_ms: int, fresh_hours: float = RANGE_FRESH_HOURS) -> dict | None:
    """쓸 수 있는 ranges.json의 {종목: 변동폭%}. 없거나·깨졌거나·오래됐으면 None
    (엔진은 변동폭 조건 없이)."""
    try:
        data = json.loads(path.read_text())
        at = datetime.strptime(str(data["updated_at"]), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
        ranges = data["ranges"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(ranges, dict):
        return None
    age_ms = now_ms - at.timestamp() * 1000
    if age_ms > fresh_hours * 3_600_000 or age_ms < -DAY_MS:
        return None
    out = {}
    for k, v in ranges.items():
        if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def refresh_recent(
    out: Path,
    symbols: list[str],
    params: dict,
    now_ms: int | None = None,
    get: Callable = get_json,
) -> dict:
    """종목마다 최근 RECENT_DAYS일 15분봉(지표 자리 잡을 앞 봉 포함)으로 엔진 신호를 백테스트해
    {종목: {trades, wins, winrate, ret}}을 out에 쓴다. 마감 봉만 쓴다. 한 종목 실패는 48시간 안의
    앞 값을 이어 쓰고, 없으면 빠진다(빠진 종목은 엔진이 새로 들어가지 않는다). 429·418은 멈춘다."""
    now_ms = now_ms or int(time.time() * 1000)
    end = now_ms - now_ms % BAR_MS
    frm = end - RECENT_DAYS * DAY_MS
    stats: dict[str, dict] = {}
    errors: dict[str, str] = {}
    before = load_recent(out, now_ms) or {}
    for sym in symbols:
        try:
            rows = fetch_15m(sym, frm - (WARMUP + 50) * BAR_MS, end, get)
            if len(rows) <= WARMUP + 2:
                errors[sym] = "15분봉 부족"
                continue
            r = backtest(rows, params, frm)
            # 끝난 거래만 센다(끝에 열린 포지션은 종가 정리라 승패가 아직 안 났다)
            n = r["trades"] - (1 if r.get("holding") else 0)
            w = r["wins"] - (1 if r.get("end_win") else 0)
            stats[sym] = {
                "trades": n,
                "wins": w,
                "winrate": round(w / n * 100, 1) if n else 0.0,
                "ret": r["ret"],
                "at": now_ms,
            }
        except RateLimited:
            raise
        except (OSError, ValueError, KeyError, IndexError, TypeError, ArithmeticError) as e:
            errors[sym] = f"{type(e).__name__}: {e}"[:200]
            # 앞 값은 잰 때(at)부터 48시간 안일 때만 이어 쓴다(매일 실패해도 무한히 안 남게)
            old = before.get(sym) or {}
            try:
                fresh = now_ms - int(old.get("at") or 0) <= RANGE_FRESH_HOURS * 3_600_000
            except (TypeError, ValueError):
                fresh = False
            if old and fresh:
                stats[sym] = old
    data = {
        "updated_at": datetime.fromtimestamp(now_ms / 1000, KST).strftime("%Y-%m-%d %H:%M"),
        "days": RECENT_DAYS,
        "min_winrate": MIN_RECENT_WINRATE,
        "min_trades": MIN_RECENT_TRADES,
        "stats": stats,
        "errors": errors,
    }
    _write_private(out, json.dumps(data, ensure_ascii=False, indent=1))
    return data


def load_recent(path: Path, now_ms: int, fresh_hours: float = RANGE_FRESH_HOURS) -> dict | None:
    """쓸 수 있는 recent.json의 {종목: {trades, wins, winrate, ret}}. 없거나·깨졌거나·오래됐으면
    None(엔진은 새 진입을 멈춘다)."""
    try:
        data = json.loads(path.read_text())
        at = datetime.strptime(str(data["updated_at"]), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
        stats = data["stats"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(stats, dict):
        return None
    age_ms = now_ms - at.timestamp() * 1000
    if age_ms > fresh_hours * 3_600_000 or age_ms < -DAY_MS:
        return None
    return {k: v for k, v in stats.items() if isinstance(k, str) and isinstance(v, dict)}


def recent_ok(st: dict | None) -> bool:
    """최근 7일 승률 조건: 거래 MIN_RECENT_TRADES회 이상이고 승률 ≥ MIN_RECENT_WINRATE."""
    if not st:
        return False
    try:
        return (
            int(st.get("trades") or 0) >= MIN_RECENT_TRADES
            and float(st.get("winrate") or 0) >= MIN_RECENT_WINRATE
        )
    except (TypeError, ValueError):
        return False


def _empty() -> dict:
    return {"trades": 0, "wins": 0, "winrate": 0.0, "ret": 0.0, "mdd": 0.0, "stops": 0, "days": 0.0}


def table(data: dict) -> str:
    stats = data["stats"]
    order = priority(data["symbols"], stats)
    lines = ["순위 우선 종목            종류        승률   거래  수익%   MDD%   일수"]
    for p, s in enumerate(order, 1):
        st = stats[s]
        kind = "TradFi" if st.get("contract") == "TRADIFI_PERPETUAL" else "crypto"
        flag = " *" if low_sample(st) else ""
        lines.append(
            f"{st['rank']:3d} {p:3d}  {s:<16} {kind:<7} {st['winrate']:6.1f} {st['trades']:5d} "
            f"{st['ret']:7.1f} {st['mdd']:6.1f} {st['days']:6.1f}{flag}"
        )
    lines.append("* 거래 20회 미만 또는 60일 미만: 우선순위 맨 뒤")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from minerva.btc_trade import SCR_PARAMS  # 엔진과 같은 값(엔진이 이 모듈을 불러 순환 피함)

    ap = argparse.ArgumentParser(description="자동매매 종목 범위(거래량 상위 50) 갱신")
    ap.add_argument("command", choices=["refresh", "show", "ranges"])
    ap.add_argument("--dry", action="store_true", help="~/.minerva 대신 --out(기본 임시)에 쓴다")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = args.out or (
        Path(os.environ.get("TMPDIR", "/tmp")) / "minerva-universe-dry.json"
        if args.dry
        else DIR / "universe.json"
    )
    if args.command == "ranges":
        uni = json.loads((DIR / "universe.json").read_text())
        out = args.out or (
            Path(os.environ.get("TMPDIR", "/tmp")) / "minerva-ranges-dry.json"
            if args.dry
            else DIR / "ranges.json"
        )
        syms = list(dict.fromkeys(uni["symbols"]))
        try:
            data = refresh_ranges(out, syms)
        except RateLimited as e:
            log.error("바이낸스 요청 한도(%s): 멈춤. 파일은 그대로", e)
            return 2
        rec_out = out.with_name("recent.json") if args.out or args.dry else DIR / "recent.json"
        try:
            rec = refresh_recent(rec_out, syms, SCR_PARAMS)
        except RateLimited as e:
            log.error("바이낸스 요청 한도(%s): 최근 7일 승률 멈춤. 파일은 그대로", e)
            return 2
        rows = sorted(rec["stats"].items(), key=lambda kv: -float(kv[1].get("winrate") or 0))
        for sym, st in rows:
            mark = "" if recent_ok(st) else "  (진입 안 함)"
            print(
                f"{sym:16s} 7일 {st.get('wins', 0)}/{st.get('trades', 0)} "
                f"{float(st.get('winrate') or 0):5.1f}%{mark}"
            )
        for sym, r in sorted(data["ranges"].items(), key=lambda kv: -kv[1]):
            print(f"{sym:16s} {r:6.2f}%{'' if r >= MIN_DAY_RANGE_PCT else '  (2% 미만)'}")
        print(f"→ {out}, 실패 {len(data['errors'])}")
        return 0
    if args.command == "show":
        print(table(json.loads(out.read_text())))
        return 0
    t0 = time.monotonic()
    try:
        data = refresh(out, SCR_PARAMS)
    except RateLimited as e:
        log.error("바이낸스 요청 한도(%s): 멈춤. 파일은 그대로", e)
        return 2
    print(table(data))
    print(f"→ {out} ({time.monotonic() - t0:.0f}초)")
    if not args.dry and args.out is None:
        # 새로 들어온 종목도 바로 변동폭을 잰다(월요일 09:10까지 막히지 않게, 레드팀)
        try:
            refresh_ranges(DIR / "ranges.json", list(dict.fromkeys(data["symbols"])))
        except RateLimited as e:
            log.error("변동폭 갱신 중 요청 한도(%s): 앞 파일 그대로", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
