"""+5% 먼저 vs −3% 먼저 장벽 게임 가상 추적(주문 없음).

사용자 2026-11-22 13:14 KST: "가상추적도 너가 하고싶은거 급등주 급락주 이런거 자동으로 추가하는 로직
넣어놔 급등주는 기다렸다가 숏치거나". 11/22 연구(13~53차, 831개 변형, 통과 0)에서 가장 가까웠던
규칙과 사용자가 말한 급등·급락주 자동 감지 규칙을 앞으로 오는 자료로만 가상으로 따라간다.
과거 연구에서 넷 다 실전 기준 탈락이다. 이 기록은 "앞으로도 그런가"를 보는 시험일 뿐이다.

규칙(UTC 일봉 마감 신호 → 다음 날 00:00 UTC 시가 진입, 한 규칙에서 종목당 한 포지션):
- BRK 알트 55일 신저가 이탈 숏: 거래대금 31~120위, 종가 < 직전 55일 최저가(첫날),
  그날 31~120위의 (90일 수익률 − BTC 90일 수익률) 중앙값 < 0. (연구 47차 T2: 44.3%, +0.39%)
  2026-11-22 교체(11/1~21 42건 11.9%, −2.2%/건): 새 진입 없음, 기록만 남김.
- FLW 테이커 흐름 추종(11/22 추가, +6%/−3%): 1~300위,
  I7 = 7일 (2·테이커 매수 거래대금/거래대금 − 1)이 자기 직전 180일 I7의 상위 10% &
  7일 수익률 > 0 → 롱, 하위 10% & < 0 → 숏(첫날).
- ABS 테이커 흡수 역추세(11/22 추가, +3%/−3%, FLW의 반대편 대조군): 하위 10%인데 7일 ≥ 0 → 롱,
  상위 10%인데 ≤ 0 → 숏(첫날). FLW·ABS 모두 사전 등록 검증 미통과(2021~23·2024~26, 같은 코인·
  같은 달 무작위 대비) → 전진 관찰용.
- PUMP 급등주 기다렸다 숏: 1~300위, 7일 +60% 이상 & BTC 7일 < +10%면 감시 목록에 올림(첫날).
  14일 안에 종가가 감시 시작 뒤 최고가(감지 전 7일 포함)의 85% 이하로 꺾이면 숏. (연구 52차 P3
  비슷한 규칙: 36.7%, −0.20%. 감시 목록 방식은 새로 정한 것, 미검증)
- DUMP 급락주 반등 롱: 1~300위, 7일 −40% 이하 & BTC 7일 > −10%(첫날). (연구 52차 P1: 37.5%, −0.05%)
- NF BTC 거래소 순유입 숏: 코인메트릭스 커뮤니티 BTC 거래소 유입−유출 7일 합의
  z(직전 180일) ≥ 2(첫날).
  그 시점에 받은 값을 날마다 그대로 저장(시점 자료). 다음 정시 진입. (연구 36차: 53%, +1.18%,
  과거 자료가 사후 수정돼 검증 불가 → 이 저장분이 첫 깨끗한 시험)
- 공통: 익절·손절은 규칙별(기본 +5%/−3%, 거래에 적어 둠. 1시간봉, 같은 봉에서 둘 다 닿으면 손절,
  시가가 이미 넘으면 그 시가), 336시간 넘으면 종가 청산, 수수료·미끄러짐 왕복 0.14% + 실제 펀딩.
  손익분기 승률 = (손절 + 0.14%) / (익절 + 손절): +5/−3 39.3%, +6/−3 34.9%, +3/−3 52.3%.
- 자동 교체(`auto_retire`): 끝난 거래 30건 이상에서 승률 윌슨 95% 상한 < 손익분기 또는 거래당
  평균 < 0, 또는 건수와 무관하게 최근 14일 끝난 거래 합 < −15%면 새 진입을 멈추고 `retire.jsonl`에
  남긴다. `QUEUE`에 사전 등록 후보가 있으면 다음 날 00시 UTC부터 켠다(지금은 비어 있음).
- 범위: 바이낸스 USDT-M 무기한, 상장 90일 이상, 직전 30일 거래대금 순위(BTC·지수·스테이블 제외).

자료: 바이낸스 공개 GET(exchangeInfo·klines·fundingRate)과 코인메트릭스 커뮤니티 API만.
키·주문·계좌 조회 없음. 엔진 타이머 앞뒤는 비킨다(edge_paper.get_json).

폴더 `~/.minerva/barrier_paper/`(MINERVA_BARRIER_PAPER_DIR): `state.json`, `trades.json`,
`retire.jsonl`, `netflow/<날짜>.json` 시점 저장, `summary.json`(화면 `/api/barrier-paper`).

실행: `python -m minerva.barrier_paper run|summary`
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import statistics
import sys
import time
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from minerva.edge_paper import (
    FAPI,
    INDEXES,
    PANEL_STABLE,
    RateLimited,
    get_json,
    read_json,
    write_json,
)

log = logging.getLogger("minerva.barrier_paper")

TP = 0.05
SL = 0.03
CAP_H = 336
COST_RT = 0.0014
START = date(2026, 11, 1)  # 백필 시작(그 뒤 상폐된 종목은 빠짐 = 3주치 작은 생존편향)
DAY_MS = 86_400_000
HR_MS = 3_600_000
MIN_LIST_DAYS = 90
DAILY_BARS = 320  # 주문흐름 규칙: 180일 백분위 + 7일 합 + 11/1부터 백필
CM_URL = (
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc"
    "&metrics=FlowInExNtv,FlowOutExNtv&frequency=1d&page_size=400&start_time={start}"
)

FLOW_RULES = {
    "FLW": {
        "side": "LS",
        "kind": "daily",
        "tp": 0.06,
        "sl": 0.03,
        "name": "테이커 흐름 추종",
        "desc": "7일 순테이커 비율이 자기 180일 상위 10% & 7일 수익률 > 0 → 롱 · "
        "하위 10% & < 0 → 숏 (첫날) · 익절 +6% 손절 −3%",
        "past": "과거 검증 미통과, 전진 관찰용 · 2021~23 거래당 −0.04%(같은 코인·같은 달 무작위 "
        "−0.04%), 2024~26 +0.10%(무작위 −0.03%) · 주 단위로 묶으면 무작위와 차이 없음(p 0.50)",
    },
    "ABS": {
        "side": "LS",
        "kind": "daily",
        "tp": 0.03,
        "sl": 0.03,
        "name": "테이커 흡수 역추세(대조군)",
        "desc": "7일 순테이커 비율이 자기 180일 하위 10%인데 7일 수익률 ≥ 0 → 롱 · "
        "상위 10%인데 ≤ 0 → 숏 (첫날) · 익절 +3% 손절 −3%",
        "past": "과거 검증 미통과, 전진 관찰용(흐름 추종의 반대편 대조군) · 2021~23 거래당 −0.02%"
        "(무작위 −0.04%), 2024~26 −0.19%(무작위 −0.02%)",
    },
}
# 자동 교체 때 차례로 켜는 사전 등록 후보(맨 앞부터, 켠 날 다음 00시 UTC부터 진입).
QUEUE: list[str] = []
FLOW_P = 0.10
FLOW_WIN = 180
FLOW_MIN = 120
# 자동 교체: 끝난 거래 30건 이상에서 승률 윌슨 95% 상한 < 손익분기 또는 거래당 평균 < 0,
# 또는 (건수 무관) 최근 14일에 끝난 거래 합 < −15%.
RETIRE_MIN_N = 30
RETIRE_RECENT_DAYS = 14
RETIRE_RECENT_SUM = -0.15

RULES = {
    "BRK": {
        "side": "S",
        "kind": "daily",
        "tp": 0.05,
        "sl": 0.03,
        "name": "알트 55일 신저가 이탈 숏",
        "desc": "거래대금 31~120위 · 55일 최저가 이탈 첫날 · 알트가 BTC보다 약한 국면",
        "past": "과거 2021~26: 44.3%, 거래당 +0.39% (2024~26만 수익, 약세장 베팅으로 탈락)",
        "retired": {
            "date": "2026-11-22",
            "reason": "11/1~21 42건 승률 11.9%, 거래당 −2.2% · 주문흐름 규칙으로 교체",
        },
    },
    "PUMP": {
        "side": "S",
        "kind": "daily",
        "tp": 0.05,
        "sl": 0.03,
        "name": "급등주 기다렸다 숏",
        "desc": "7일 +60% 급등 → 감시 → 14일 안에 고점 대비 −15% 꺾이면 숏",
        "past": "비슷한 규칙 과거 36.7%, 거래당 −0.20% (탈락) · 감시 목록 방식은 미검증",
    },
    "DUMP": {
        "side": "L",
        "kind": "daily",
        "tp": 0.05,
        "sl": 0.03,
        "name": "급락주 반등 롱",
        "desc": "7일 −40% 급락(BTC는 −10%보다 덜 빠짐) 첫날 → 롱",
        "past": "과거 37.5%, 거래당 −0.05% (탈락)",
    },
    "NF": {
        "side": "S",
        "kind": "nf",
        "tp": 0.05,
        "sl": 0.03,
        "name": "BTC 거래소 순유입 급증 숏",
        "desc": "BTC 거래소 유입−유출 7일 합 z(180일) ≥ 2 첫날 → BTC 숏",
        "past": "과거 53%, +1.18%지만 과거 자료 사후 수정 → 이 저장분이 첫 깨끗한 시험",
    },
    # ---- 2026-11-22 교체분(사전 등록 후 검증, 아래 past에 창별 수치). 주문흐름 = 일봉의
    # 테이커 매수 거래대금(klines 10번째 값). 1~300위 · 롱·숏 대칭.
    **FLOW_RULES,
}
PUMP_UP = 0.60
PUMP_BTC = 0.10
PUMP_DROP = 0.85
PUMP_WAIT = 14
DUMP_DN = -0.40
DUMP_BTC = -0.10
NF_Z = 2.0


def base_dir() -> Path:
    return Path(
        os.environ.get("MINERVA_BARRIER_PAPER_DIR", "~/.minerva/barrier_paper")
    ).expanduser()


def day_of(ms: int) -> date:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).date()


def day_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------- 순수 계산


def barrier_exit(
    side: str,
    entry: float,
    bars: list[list[float]],
    tp: float = TP,
    sl: float = SL,
    cap: int = CAP_H,
):
    """bars = 진입 시각부터의 1시간봉 [시각ms, 시가, 고가, 저가, 종가](마감된 봉만).

    끝났으면 (결과 'win'|'loss'|'timeout', 청산가, 청산 봉 index), 아직이면 None."""
    if side == "L":
        up, dn = entry * (1 + tp), entry * (1 - sl)
    else:
        up, dn = entry * (1 + sl), entry * (1 - tp)
    for k, (_, o, h, low, _c) in enumerate(bars[:cap]):
        if side == "L":
            if low <= dn:
                return "loss", (min(dn, o) if k > 0 else dn), k
            if h >= up:
                return "win", up, k
        else:
            if h >= up:
                return "loss", (max(up, o) if k > 0 else up), k
            if low <= dn:
                return "win", dn, k
    if len(bars) >= cap:
        return "timeout", bars[cap - 1][4], cap - 1
    return None


def net_return(side: str, entry: float, exit_px: float, funding: float) -> float:
    """비용·펀딩 뒤 수익률. funding = 보유 중 정산된 펀딩비율 합(롱은 내고 숏은 받음)."""
    sign = 1 if side == "L" else -1
    return sign * (exit_px / entry - 1) - COST_RT - sign * funding


def rank_universe(
    daily: dict[str, list], onboard: dict[str, int], d_idx_ms: int, top: int = 300
) -> list[str]:
    """d_idx_ms 날 00:00 기준 직전 30일 거래대금 순위(상장 90일 이상, BTC 제외)."""
    rows = []
    for s, bars in daily.items():
        if s == "BTCUSDT" or onboard.get(s, 0) + MIN_LIST_DAYS * DAY_MS > d_idx_ms:
            continue
        q = [b[5] for b in bars if d_idx_ms - 30 * DAY_MS <= b[0] < d_idx_ms]
        if len(q) >= 25:
            rows.append((sum(q), s))
    rows.sort(reverse=True)
    return [s for _, s in rows[:top]]


def _series(bars: list, upto_ms: int):
    """upto_ms 날까지(포함) 마감 일봉의 시각·고가·저가·종가."""
    b = [x for x in bars if x[0] <= upto_ms]
    return [x[0] for x in b], [x[2] for x in b], [x[3] for x in b], [x[4] for x in b]


def _ret(c: list[float], k: int, i: int) -> float | None:
    if i - k < 0 or c[i - k] <= 0:
        return None
    return c[i] / c[i - k] - 1


def brk_hit(h, low, c, i: int) -> bool:
    def br(j):
        return j >= 55 and c[j] < min(low[j - 55 : j])

    return br(i) and not br(i - 1)


def dump_hit(c, i: int, btc7: float | None) -> bool:
    def cond(j):
        r = _ret(c, 7, j)
        return r is not None and r <= DUMP_DN

    return btc7 is not None and btc7 > DUMP_BTC and cond(i) and not cond(i - 1)


def pump_trigger(h, c, i: int, btc7_at) -> int | None:
    """i일 종가에 숏 신호가 나면 감시 시작일 index, 아니면 None.

    감시 시작 = 7일 +60% & BTC 7일 < +10%가 처음 된 날 s(i−14 ≤ s ≤ i). 기준 고가 = s−6~i의 최고가.
    s~i−1에 이미 신호가 났으면 None(감시당 한 번)."""

    def up(j):
        r = _ret(c, 7, j)
        b = btc7_at(j)
        return r is not None and r >= PUMP_UP and b is not None and b < PUMP_BTC

    for s in range(max(8, i - PUMP_WAIT), i + 1):
        if not (up(s) and not up(s - 1)):
            continue
        fired = None
        for j in range(s, i + 1):
            if c[j] <= PUMP_DROP * max(h[max(0, s - 6) : j + 1]):
                fired = j
                break
        if fired == i:
            return s
    return None


def netflow_z(values: list[float], window: int = 180) -> list[float | None]:
    """values = 일별 (유입−유출). 7일 합의 z(직전 window일의 7일 합 기준)."""
    s7 = [sum(values[i - 6 : i + 1]) if i >= 6 else None for i in range(len(values))]
    out: list[float | None] = []
    for i, x in enumerate(s7):
        prev = [v for v in s7[max(0, i - window) : i] if v is not None]
        if x is None or len(prev) < window // 2:
            out.append(None)
            continue
        sd = statistics.pstdev(prev)
        out.append((x - statistics.fmean(prev)) / sd if sd > 0 else None)
    return out


def breakeven(tp: float, sl: float, cost: float = COST_RT) -> float:
    """익절 tp·손절 sl 게임의 손익분기 승률(펀딩·기간 만료 무시)."""
    return (sl + cost) / (tp + sl)


def wilson_upper(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = wins / n
    den = 1 + z * z / n
    mid = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (mid + half) / den


def _signed(v: float, digits: int) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


def retire_reason(closed: list[dict], now_ms: int, be: float) -> str | None:
    """자동 교체 사유(없으면 None). closed = 결과가 난 거래(net·exit_ms·result)."""
    recent = [t["net"] for t in closed if t["exit_ms"] > now_ms - RETIRE_RECENT_DAYS * DAY_MS]
    if recent and sum(recent) < RETIRE_RECENT_SUM:
        return (
            f"최근 {RETIRE_RECENT_DAYS}일 끝난 {len(recent)}건 합 {_signed(100 * sum(recent), 1)}%"
        )
    n = len(closed)
    if n < RETIRE_MIN_N:
        return None
    wins = sum(t["result"] == "win" for t in closed)
    up = wilson_upper(wins, n)
    if up < be:
        return (
            f"{n}건 승률 {100 * wins / n:.1f}% "
            f"(95% 상한 {100 * up:.1f}% < 손익분기 {100 * be:.1f}%)"
        )
    avg = statistics.fmean(t["net"] for t in closed)
    if avg < 0:
        return f"{n}건 거래당 평균 {_signed(100 * avg, 2)}%"
    return None


def _prefix(x: list[float]) -> list[float]:
    out = [0.0]
    for v in x:
        out.append(out[-1] + v)
    return out


def flow_series(q: list[float], tb: list[float]) -> list[float | None]:
    """일봉 거래대금·테이커 매수 거래대금 → I7 목록(앞 6일은 None).

    I7 = 2·Σtb/Σq − 1 (7일 순테이커 비율, −1 = 전부 테이커 매도, +1 = 전부 테이커 매수)."""
    pq, pt = _prefix(q), _prefix(tb)
    i7: list[float | None] = []
    for j in range(len(q)):
        if j < 6:
            i7.append(None)
            continue
        sq = pq[j + 1] - pq[j - 6]
        i7.append(2 * (pt[j + 1] - pt[j - 6]) / sq - 1 if sq > 0 else None)
    return i7


def flow_pct(i7: list, i: int) -> float | None:
    """i일 I7의 자기 직전 180일(i−180..i−1) 안 백분위(그보다 작은 값 비율). 120개 미만이면 None."""
    cur = i7[i] if 0 <= i < len(i7) else None
    if cur is None:
        return None
    prior = [v for v in i7[max(0, i - FLOW_WIN) : i] if v is not None]
    if len(prior) < FLOW_MIN:
        return None
    return sum(v < cur for v in prior) / len(prior)


def flow_sides(rule: str, c: list[float], i7: list, i: int) -> list[str]:
    """i일 마감에 rule이 내는 방향(첫날만): ['L'] / ['S'] / []."""

    def cond(j):
        if j < 7:
            return set()
        r7 = _ret(c, 7, j)
        if r7 is None:
            return set()
        p = flow_pct(i7, j)
        if p is None:
            return set()
        lo, hi = p <= FLOW_P, p >= 1 - FLOW_P
        out = set()
        if rule == "ABS":
            if lo and r7 >= 0:
                out.add("L")
            if hi and r7 <= 0:
                out.add("S")
        elif rule == "FLW":
            if hi and r7 > 0:
                out.add("L")
            if lo and r7 < 0:
                out.add("S")
        return out

    now, prev = cond(i), cond(i - 1)
    return sorted(now - prev)


# ---------------------------------------------------------------- 자료


def perps(get=get_json) -> dict[str, int]:
    info = get(f"{FAPI}/fapi/v1/exchangeInfo")
    out = {}
    for s in info.get("symbols", []):
        sym = s.get("symbol", "")
        if (
            s.get("contractType") != "PERPETUAL"
            or s.get("quoteAsset") != "USDT"
            or s.get("status") != "TRADING"
            or s.get("underlyingType") != "COIN"
            or sym in INDEXES
            or s.get("baseAsset") in PANEL_STABLE
        ):
            continue
        out[sym] = int(s.get("onboardDate") or 0)
    return out


def daily_klines(sym: str, get=get_json) -> list[list[float]]:
    """일봉 [시각, 시가, 고가, 저가, 종가, 거래대금, 테이커 매수 거래대금].

    마지막은 오늘 진행 중 봉일 수 있다."""
    rows = get(
        f"{FAPI}/fapi/v1/klines?symbol={urllib.parse.quote(sym)}&interval=1d&limit={DAILY_BARS}"
    )
    return [
        [
            int(r[0]),
            float(r[1]),
            float(r[2]),
            float(r[3]),
            float(r[4]),
            float(r[7]),
            float(r[10]) if len(r) > 10 else None,
        ]
        for r in rows
    ]


def hourly_from(sym: str, start_ms: int, now_ms: int, get=get_json) -> list[list[float]]:
    """start_ms부터 마감된 1시간봉."""
    out = []
    t = start_ms
    while t < now_ms:
        rows = get(
            f"{FAPI}/fapi/v1/klines?symbol={urllib.parse.quote(sym)}&interval=1h&startTime={t}&limit=500"
        )
        if not rows:
            break
        for r in rows:
            if int(r[6]) < now_ms:
                out.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])])
        nxt = int(rows[-1][0]) + HR_MS
        if nxt <= t or len(rows) < 500:
            break
        t = nxt
    return out


def funding_sum(sym: str, entry_ms: int, exit_ms: int, get=get_json) -> float:
    rows = get(
        f"{FAPI}/fapi/v1/fundingRate?symbol={urllib.parse.quote(sym)}&startTime={entry_ms + 1}"
        f"&endTime={exit_ms}&limit=1000"
    )
    return sum(float(r["fundingRate"]) for r in rows if entry_ms < int(r["fundingTime"]) <= exit_ms)


def fetch_netflow(today: date, get=get_json) -> list[dict]:
    start = (today - timedelta(days=400)).isoformat()
    rows, url = [], CM_URL.format(start=start)
    for _ in range(5):
        d = get(url)
        rows += d.get("data", [])
        url = d.get("next_page_url")
        if not url:
            break
    return rows


# ---------------------------------------------------------------- 실행


def load_trades() -> list[dict]:
    p = base_dir() / "trades.json"
    return read_json(p, []) or []


def save_trades(trades: list[dict]) -> None:
    write_json(base_dir() / "trades.json", trades)


def _open_syms(trades, rule):
    return {t["sym"] for t in trades if t["rule"] == rule and t.get("result") is None}


def rule_start_ms(key: str, st: dict) -> int:
    """그 규칙이 진입을 시작하는 시각(백필 시작 또는 대기 후보를 켠 다음 날 00시 UTC)."""
    act = (st.get("activated") or {}).get(key)
    if act:
        return act["from_ms"]
    if key in QUEUE:
        return 10**15
    return day_ms(START)


def retired_info(key: str, st: dict) -> dict | None:
    fixed = RULES[key].get("retired")
    if fixed:
        return {**fixed, "auto": False, "at_ms": day_ms(date.fromisoformat(fixed["date"]))}
    return (st.get("retired") or {}).get(key)


def rule_open(key: str, st: dict, entry_ms: int) -> bool:
    """entry_ms에 이 규칙이 새로 들어갈 수 있나(시작 뒤, 교체 전)."""
    if entry_ms < rule_start_ms(key, st):
        return False
    r = retired_info(key, st)
    return r is None or entry_ms < r["at_ms"]


def _flow_cols(bars: list, upto_ms: int):
    b = [x for x in bars if x[0] <= upto_ms]
    if not all(len(x) >= 7 and x[6] is not None for x in b):
        return None
    return [x[5] for x in b], [x[6] for x in b]


def scan_day(d: date, daily: dict, onboard: dict, trades: list[dict], rules=None) -> list[dict]:
    """d일 일봉 마감 신호 → d+1 00:00 UTC 진입 거래 목록(아직 진입가 없음).

    rules = 이번에 훑을 규칙(None이면 일봉 규칙 전부).
    진입 가능 여부(시작·교체)는 부르는 쪽이 정한다."""
    if rules is None:
        rules = {k for k, r in RULES.items() if r["kind"] == "daily"}
    dm = day_ms(d)
    entry_ms = dm + DAY_MS
    btc = daily.get("BTCUSDT")
    if not btc:
        return []
    bt, _, _, bc = _series(btc, dm)
    if not bt or bt[-1] != dm:
        return []
    bi = len(bc) - 1

    def btc7_at_offset(off):
        return _ret(bc, 7, bi - off)

    uni = rank_universe(daily, onboard, entry_ms)
    mid = uni[30:120]
    new = []
    b90 = _ret(bc, 90, bi)
    rel = []
    ser = {}
    for s in uni:
        ser[s] = _series(daily[s], dm)
        t_, h_, l_, c_ = ser[s]
        if "BRK" in rules and s in mid and t_ and t_[-1] == dm and b90 is not None:
            r = _ret(c_, 90, len(c_) - 1)
            if r is not None:
                rel.append(r - b90)
    bleed = len(rel) >= 10 and statistics.median(rel) < 0
    held = {r: _open_syms(trades, r) for r in RULES}
    flow = [k for k in FLOW_RULES if k in rules]

    def add(rule, s, side, **kw):
        r = RULES[rule]
        new.append(
            {
                "rule": rule,
                "sym": s,
                "side": side,
                "signal_day": d.isoformat(),
                "entry_ms": entry_ms,
                "tp": r["tp"],
                "sl": r["sl"],
                **kw,
            }
        )

    for s in uni:
        t_, h_, l_, c_ = ser[s]
        if not t_ or t_[-1] != dm or len(c_) < 20:
            continue
        i = len(c_) - 1
        if (
            "BRK" in rules
            and bleed
            and s in mid
            and s not in held["BRK"]
            and brk_hit(h_, l_, c_, i)
        ):
            add("BRK", s, "S")
        if "DUMP" in rules and s not in held["DUMP"] and dump_hit(c_, i, btc7_at_offset(0)):
            add("DUMP", s, "L", r7_pct=round(100 * _ret(c_, 7, i), 1))
        if "PUMP" in rules and s not in held["PUMP"]:

            def b7(j, _i=i):
                return btc7_at_offset(_i - j)

            st = pump_trigger(h_, c_, i, b7)
            if st is not None:
                add(
                    "PUMP",
                    s,
                    "S",
                    watch_since=day_of(t_[st]).isoformat(),
                    peak=max(h_[max(0, st - 6) : i + 1]),
                )
        if flow:
            cols = _flow_cols(daily[s], dm)
            if cols is None:
                continue
            i7 = flow_series(*cols)
            for k in flow:
                if s in held[k]:
                    continue
                for side in flow_sides(k, c_, i7, i):
                    add(k, s, side)
    return new


def pump_watch(d: date, daily: dict, onboard: dict) -> list[dict]:
    """d일 마감 기준 감시 중인 급등주(아직 꺾이지 않음)."""
    dm = day_ms(d)
    btc = daily.get("BTCUSDT")
    if not btc:
        return []
    _, _, _, bc = _series(btc, dm)
    bi = len(bc) - 1
    out = []
    for s in rank_universe(daily, onboard, dm + DAY_MS):
        t_, h_, _, c_ = _series(daily[s], dm)
        if not t_ or t_[-1] != dm or len(c_) < 20:
            continue
        i = len(c_) - 1

        def up(j, c_=c_, i=i):
            r = _ret(c_, 7, j)
            b = _ret(bc, 7, bi - (i - j))
            return r is not None and r >= PUMP_UP and b is not None and b < PUMP_BTC

        for st in range(max(8, i - PUMP_WAIT + 1), i + 1):
            if up(st) and not up(st - 1):
                peak = max(h_[max(0, st - 6) : i + 1])
                if all(
                    c_[j] > PUMP_DROP * max(h_[max(0, st - 6) : j + 1]) for j in range(st, i + 1)
                ):
                    out.append(
                        {
                            "sym": s,
                            "since": day_of(t_[st]).isoformat(),
                            "peak": peak,
                            "last": c_[i],
                            "trigger": round(PUMP_DROP * peak, 8),
                            "until": day_of(t_[st] + PUMP_WAIT * DAY_MS).isoformat(),
                        }
                    )
                break
    return out


def run_netflow(now_ms: int, trades: list[dict], st: dict, get=get_json) -> dict:
    """BTC 순유입 신호. 어제 값이 없으면(보통 UTC 01~04시 공개) 3시간마다 다시 받는다.

    받은 그대로 `netflow/<마지막 날>.json`에 저장(시점 자료)."""
    today = day_of(now_ms)
    nd = base_dir() / "netflow"
    want = (today - timedelta(days=1)).isoformat()
    snap = nd / f"{want}.json"
    if not snap.exists() and now_ms - st.get("nf_fetch_ms", 0) >= 3 * HR_MS:
        st["nf_fetch_ms"] = now_ms
        try:
            rows = fetch_netflow(today, get=get)
        except Exception as e:  # noqa: BLE001 - 코인메트릭스 장애는 다음 회에 다시
            log.warning("순유입 받기 실패: %s", e)
            rows = []
        if rows:
            nd.mkdir(parents=True, exist_ok=True)
            write_json(
                nd / f"{rows[-1]['time'][:10]}.json",
                {"fetched_at": datetime.now(timezone.utc).isoformat(), "data": rows},
            )
    files = sorted(nd.glob("*.json")) if nd.exists() else []
    if not files:
        return st.get("netflow", {})
    snap = files[-1]
    rows = (read_json(snap, {}) or {}).get("data", [])
    days, vals = [], []
    for r in rows:
        try:
            vals.append(float(r["FlowInExNtv"]) - float(r["FlowOutExNtv"]))
            days.append(r["time"][:10])
        except (KeyError, TypeError, ValueError):
            continue
    z = netflow_z(vals)
    info = {
        "last_day": days[-1] if days else None,
        "z": round(z[-1], 2) if z and z[-1] is not None else None,
        "snapshot": snap.stem,
        "snapshots": len(files),
    }
    seen = set(st.get("nf_signaled", []))
    if (
        len(z) >= 2
        and z[-1] is not None
        and z[-1] >= NF_Z
        and not (z[-2] is not None and z[-2] >= NF_Z)
    ):
        key = days[-1]
        entry_ms = (now_ms // HR_MS + 1) * HR_MS
        if (
            key not in seen
            and "BTCUSDT" not in _open_syms(trades, "NF")
            and rule_open("NF", st, entry_ms)
        ):
            trades.append(
                {
                    "rule": "NF",
                    "sym": "BTCUSDT",
                    "side": "S",
                    "signal_day": key,
                    "entry_ms": entry_ms,
                    "z": round(z[-1], 2),
                    "tp": RULES["NF"]["tp"],
                    "sl": RULES["NF"]["sl"],
                }
            )
            seen.add(key)
    st["nf_signaled"] = sorted(seen)[-50:]
    st["netflow"] = info
    return info


def trade_geom(t: dict) -> tuple[float, float]:
    """거래에 적힌 익절·손절(옛 거래는 +5%/−3%)."""
    return t.get("tp", TP), t.get("sl", SL)


def auto_retire(trades: list[dict], st: dict, now_ms: int) -> list[dict]:
    """진입 중인 규칙을 점검해 기준에 걸리면 교체(새 진입 멈춤)하고 대기 후보를 켠다.

    `retire.jsonl`에 남기고 새로 교체된 목록을 돌려준다."""
    out = []
    retired = st.setdefault("retired", {})
    activated = st.setdefault("activated", {})
    for key, r in RULES.items():
        if retired_info(key, st) is not None or not rule_open(key, st, now_ms):
            continue
        closed = [t for t in trades if t["rule"] == key and t.get("result") is not None]
        why = retire_reason(closed, now_ms, breakeven(r["tp"], r["sl"]))
        if why is None:
            continue
        today = day_of(now_ms)
        info = {"date": today.isoformat(), "reason": why, "auto": True, "at_ms": now_ms}
        nxt = next((q for q in QUEUE if q not in activated and q not in retired), None)
        if nxt:
            activated[nxt] = {
                "date": today.isoformat(),
                "from_ms": day_ms(today) + DAY_MS,
                "replaces": key,
            }
            info["replaced_by"] = nxt
        retired[key] = info
        row = {"rule": key, "name": r["name"], **info}
        out.append(row)
        log_path = base_dir() / "retire.jsonl"
        with log_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("자동 교체 %s: %s", key, why)
    return out


def resolve(trades: list[dict], now_ms: int, get=get_json) -> None:
    for t in trades:
        if t.get("result") is not None or t["entry_ms"] >= now_ms:
            continue
        bars = hourly_from(t["sym"], t["entry_ms"], now_ms, get=get)
        if not bars or bars[0][0] != t["entry_ms"]:
            continue
        t["entry"] = bars[0][1]
        t["last"] = bars[-1][4]
        t["last_ms"] = bars[-1][0] + HR_MS
        tp, sl = trade_geom(t)
        r = barrier_exit(t["side"], t["entry"], bars, tp=tp, sl=sl)
        if r is None:
            continue
        res, px, k = r
        exit_ms = bars[k][0] + HR_MS
        f = funding_sum(t["sym"], t["entry_ms"], exit_ms, get=get)
        t.update(
            result=res,
            exit=px,
            exit_ms=exit_ms,
            funding=f,
            net=net_return(t["side"], t["entry"], px, f),
        )


def run(get=get_json, now_ms: int | None = None) -> dict:
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    root = base_dir()
    root.mkdir(parents=True, exist_ok=True)
    st = read_json(root / "state.json", {}) or {}
    trades = load_trades()
    last_closed = day_of(now_ms) - timedelta(days=1)
    scanned = st.setdefault("scanned", {})
    if st.get("last_scan") and not scanned:  # 옛 상태: 기존 일봉 규칙은 거기까지 훑었다
        for k in ("BRK", "PUMP", "DUMP"):
            scanned[k] = st["last_scan"]
    daily_rules = [k for k, r in RULES.items() if r["kind"] == "daily"]
    for k in daily_rules:
        if k not in scanned:
            first = day_of(min(rule_start_ms(k, st), day_ms(last_closed) + DAY_MS) - DAY_MS)
            scanned[k] = (first - timedelta(days=1)).isoformat()
    live = [k for k in daily_rules if rule_open(k, st, day_ms(last_closed) + DAY_MS)]
    for k in daily_rules:  # 교체·대기 규칙은 훑지 않고 날짜만 따라간다
        if k not in live:
            scanned[k] = max(scanned[k], last_closed.isoformat())
    done = min((date.fromisoformat(scanned[k]) for k in live), default=last_closed)
    watch = st.get("watch", [])
    if done < last_closed:
        onboard = perps(get=get)
        onboard.setdefault("BTCUSDT", 0)
        daily = {}
        for s in sorted(onboard):
            try:
                daily[s] = daily_klines(s, get=get)
            except RateLimited:
                raise
            except Exception as e:  # noqa: BLE001 - 한 종목 실패는 건너뜀
                log.warning("%s 일봉 실패: %s", s, e)
        d = done + timedelta(days=1)
        while d <= last_closed:
            entry_ms = day_ms(d) + DAY_MS
            todo = {
                k for k in live if date.fromisoformat(scanned[k]) < d and rule_open(k, st, entry_ms)
            }
            if todo:
                trades.extend(scan_day(d, daily, onboard, trades, rules=todo))
            d += timedelta(days=1)
        for k in live:
            scanned[k] = last_closed.isoformat()
        watch = pump_watch(last_closed, daily, onboard)
        st["last_scan"] = last_closed.isoformat()
        st["universe_n"] = len(daily)
        save_trades(trades)
    st["watch"] = watch
    run_netflow(now_ms, trades, st, get=get)
    resolve(trades, now_ms, get=get)
    save_trades(trades)
    auto_retire(trades, st, now_ms)
    st["ran_at"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "state.json", st)
    summ = build_summary(trades, st, now_ms)
    write_json(root / "summary.json", summ)
    return {
        "trades": len(trades),
        "open": sum(t.get("result") is None for t in trades),
        "watch": len(watch),
    }


def _iso(ms):
    return None if ms is None else datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def build_summary(trades: list[dict], st: dict, now_ms: int) -> dict:
    rules = []
    retired_all = []
    for key, r in RULES.items():
        ts = [t for t in trades if t["rule"] == key]
        ret = retired_info(key, st)
        if ret is not None:
            retired_all.append({"key": key, "name": r["name"], **ret})
        if not ts and rule_start_ms(key, st) > now_ms:
            continue  # 아직 안 켠 대기 후보
        closed = [t for t in ts if t.get("result") is not None]
        wins = sum(t["result"] == "win" for t in closed)
        open_ = []
        for t in ts:
            if t.get("result") is not None:
                continue
            e = t.get("entry")
            tp, sl = trade_geom(t)
            mv = (
                None
                if e is None or t.get("last") is None
                else ((t["last"] / e - 1) * (1 if t["side"] == "L" else -1) * 100)
            )
            open_.append(
                {
                    "sym": t["sym"],
                    "dir": "롱" if t["side"] == "L" else "숏",
                    "at": _iso(t["entry_ms"]),
                    "entry": e,
                    "last": t.get("last"),
                    "move_pct": mv,
                    "tp": None if e is None else e * (1 + tp if t["side"] == "L" else 1 - tp),
                    "stop": None if e is None else e * (1 - sl if t["side"] == "L" else 1 + sl),
                }
            )
        last = sorted(closed, key=lambda t: t["exit_ms"], reverse=True)[:20]
        rules.append(
            {
                "key": key,
                "name": r["name"],
                "desc": r["desc"],
                "past": r["past"],
                "tp_pct": 100 * r["tp"],
                "sl_pct": 100 * r["sl"],
                "breakeven_pct": round(100 * breakeven(r["tp"], r["sl"]) + 1e-9, 1),
                "status": "retired" if ret is not None else "active",
                "retired": None
                if ret is None
                else {k: ret.get(k) for k in ("date", "reason", "auto", "replaced_by")},
                "start": datetime.fromtimestamp(rule_start_ms(key, st) / 1000, timezone.utc)
                .date()
                .isoformat(),
                "trades": len(ts),
                "closed": len(closed),
                "open": len(open_),
                "wins": wins,
                "winrate": 100 * wins / len(closed) if closed else None,
                "avg_net_pct": 100 * statistics.fmean(t["net"] for t in closed) if closed else None,
                "sum_net_pct": 100 * sum(t["net"] for t in closed) if closed else None,
                "open_positions": open_,
                "last_trades": [
                    {
                        "sym": t["sym"],
                        "dir": "롱" if t["side"] == "L" else "숏",
                        "at": _iso(t["entry_ms"]),
                        "exit_at": _iso(t["exit_ms"]),
                        "entry": t["entry"],
                        "exit": t["exit"],
                        "result": t["result"],
                        "tp_pct": 100 * trade_geom(t)[0],
                        "sl_pct": 100 * trade_geom(t)[1],
                        "net_pct": 100 * t["net"],
                    }
                    for t in last
                ],
            }
        )
    return {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start": START.isoformat(),
        "last_scan": st.get("last_scan"),
        "universe_n": st.get("universe_n"),
        "breakeven_pct": 39.3,
        "rules": rules,
        "retired": [
            {k: v for k, v in x.items() if k != "at_ms"}
            for x in sorted(retired_all, key=lambda x: x["date"])
        ],
        "queue": [q for q in QUEUE if q not in (st.get("activated") or {})],
        "watch": st.get("watch", []),
        "netflow": st.get("netflow", {}),
    }


def _lock():
    root = base_dir()
    root.mkdir(parents=True, exist_ok=True)
    f = (root / ".lock").open("w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None
    return f


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="minerva.barrier_paper")
    ap.add_argument("cmd", choices=["run", "summary"])
    a = ap.parse_args(argv)
    lk = _lock()
    if lk is None:
        log.info("다른 실행이 돌고 있어 건너뜀")
        return 0
    if a.cmd == "run":
        print(json.dumps(run()))
    else:
        st = read_json(base_dir() / "state.json", {}) or {}
        write_json(
            base_dir() / "summary.json", build_summary(load_trades(), st, int(time.time() * 1000))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
