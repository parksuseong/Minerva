"""USDⓈ-M 선물 자동매매 엔진 (바이낸스, 사용자 결정 2026-11-08 22:16 KST "설정 ①").

종목(사용자 2026-11-12): 거래량 상위 50(`btc_universe`, `universe.json`, 주 1회 갱신, 14일
넘게 오래되거나 없으면 예전 네 종목 BTC·ETH·SOXL·NVDA)에 같은 신호·규칙,
포지션은 한 번에 하나(계좌 전체).
우선순위는 종목별 백테스트 승률(표본 작은 종목은 맨 뒤). 들고 있는 종목은 범위에서 빠져도 끝까지
관리한다(새로 들어가지는 않는다). 들고 있는 종목의 반대 신호만 나면 지금처럼 갈아탄다.
빈손에 신호가 둘 이상이면 우선순위 위 5개 중 Claude가 고른다(`btc_ai.choose`, 답이 없으면 1순위).
들고 있는데 다른 종목 신호가 나면 그 종목 승률이 보유 종목보다 5%p 이상 높고 들어갈 수 있을 때만
(위 3개) 바꾸기 선택지가 되고, 없으면 AI를 부르지 않고 유지(자기 반대 신호면 갈아타기).
TradFi 무기한(주식·ETF·원자재 등)은 약관에 동의하지 않은 계좌는 -4411로 거부된다: 알리고 24시간
TradFi 전체를 쉰다. 그 시장이 쉬는 시간(`tradingSchedule` NO_TRADING)에는 새로 들어가지 않는다.

신호: SCR 실험실(`strategy/scr_lab.py`)과 같은 계산, 15분봉, 마감된 봉만.
Stoch 9(20/80)·CCI 20(±100)·RSI 21(30/70), W=3, 3개 중 3개, 넘어옴(cross) 모드.
매매: 롱↔숏 갈아타기. 반대 신호면 닫고 반대로 연다. 같은 방향 신호면 그대로 둔다.
같은 봉에 롱·숏 신호가 둘 다 나오면 아무것도 하지 않는다(화면 `switchSim`과 같음).
손절: 진입 체결가에서 3% 떨어진 거래소 STOP_MARKET(closePosition, 마크 가격 기준).
손절 뒤에는 다음 신호까지 쉰다. 레버리지 3배, 교차 마진(바꾸지 않는다).
크기: 가용 잔고 × margin_frac × (1 − FEE_BUFFER) × 3 / 가격, 종목 수량 단위로 내림.
추격 한도: 실행 시 가격이 신호 봉 종가에서 진입 방향으로 1%(15분·4시간·일봉 방향이 모두 같으면 3%)
이상 벌어졌으면 들어가지 않는다 (`strategy/btc.entry_gate`).

정지 규칙(신규 진입만 막고 보유·손절·반대 신호 청산은 계속): `NO_ENTRY` 파일을 만들고 알린다.
- 평가액 < 시드 × 0.80
- 최근 30일 청산 승률 < 50% (8회 이상)
- 수동 `HALT` 파일
- 엔진이 열지 않은 포지션·주문이 보이면(수동 개입) 건드리지 않고 알린다 (실전만)

안전 설계(국내 `trade.py`와 같은 방식):
- 이 프로세스만 바이낸스에 POST/DELETE를 보낸다. 웹 서버는 주문하지 않는다.
- `~/.minerva/btc_trade/mode`가 `live`일 때만 실주문. 없거나 다른 값이면 드라이런:
  계산은 다 하고 낼 주문을 `dry.jsonl`에 적고 텔레그램 "[드라이런] …"을 보낸다. POST 없음.
- 주문 ID는 신호 봉과 동작으로 정한다(`mnv-<봉 시작 분>-<동작>`). 보내기 전에 같은 ID를 조회해
  중복을 막는다. 통신 오류만 10초 간격 최대 3회 다시 보낸다. 거래소 거부는 다시 보내지 않는다.
- 매 실행마다 바이낸스와 대사한다(포지션·미체결·체결): 손절 체결 확인, 수동 개입 감지,
  손절 주문이 없으면 다시 걸고 실패하면 시장가로 닫는다.

조건부 주문(STOP_MARKET)은 2025-12-09부터 `/fapi/v1/algoOrder`(Algo Service)로만 받는다.
`/fapi/v1/order`로 보내면 -4120으로 거부된다.

실행: `python -m minerva.btc_trade run` (minerva-btc-trade.timer, 15분 봉 마감 20초 뒤)
점검(GET만): `python -m minerva.btc_trade preflight`
실전 시작(사용자만): `python -m minerva.btc_trade start-live --confirm`
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any, Callable

from minerva import binance, btc_ai, btc_universe, notify
from minerva.kis import KST, _write_private
from minerva.strategy import btc, scr_lab

log = logging.getLogger(__name__)

BASE = binance.BASE
SPOT_BASE = "https://api.binance.com"
# 옛 상태 파일(포지션에 symbol이 없음)의 종목이자 심볼을 모르는 클라이언트의 기본값
SYMBOL = "BTCUSDT"
# 종목 범위 파일이 없거나 오래됐을 때 쓰는 네 종목(사용자 2026-11-11)
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOXLUSDT", "NVDAUSDT")
# 그 네 종목의 백테스트 승률(%): SCR ① 15분·3배·손절 3%, 최근 1년(상장이 늦으면 상장 이후),
# 2026-11-11 측정(사용자 "승률 높은 걸 우선순위로")
DEFAULT_WINRATE = {"BTCUSDT": 54, "NVDAUSDT": 49, "ETHUSDT": 44, "SOXLUSDT": 17}
# 이번 실행의 종목·승률·우선순위. run()·preflight()가 `use_universe`로 universe.json에서 채운다
SYMBOLS: tuple[str, ...] = DEFAULT_SYMBOLS
SYMBOL_WINRATE: dict[str, float] = dict(DEFAULT_WINRATE)
PRIORITY: tuple[str, ...] = tuple(sorted(SYMBOLS, key=lambda s: -SYMBOL_WINRATE.get(s, 0)))
UNIVERSE: dict = {"source": "default", "stats": {}}
# 주문 ID 종목 표시. 네 종목은 실전 상태 파일의 ID와 같게 그대로(BTC는 비움), 나머지는 `cid_tag`
CID_TAG = {"BTCUSDT": "", "ETHUSDT": "E", "SOXLUSDT": "S", "NVDAUSDT": "N"}
# 보유 중 다른 종목으로 바꾸려면 그 종목 승률이 이만큼(%p) 높아야 한다(AI 호출을 드물게)
SWITCH_EDGE_PP = 5.0
# 사용자 2026-11-12 "승률 50프로 미만은 들어가지마" → 같은 날 "50이 아니라 45로"
MIN_ENTRY_WINRATE = 45.0
CHOOSE_FLAT_MAX = 5  # 빈손 동시 신호: AI에 우선순위 위 이만큼만
CHOOSE_SWITCH_MAX = 3  # 보유 중 바꾸기 선택지 최대
# 한 실행에서 진입 점검(시세·거래소 조회)을 하는 다른 종목 수 한도(신호가 몰려도 실행이 늘지 않게)
PREPARE_MAX = 10
# 15분봉 동시 받기(종목 50개). 499봉은 무게 2(1000봉은 5): 엔진 실행 한 번에 약 100
FETCH_WORKERS = 10
FETCH_BUDGET_S = 45
ENGINE_BARS = 499
# TradFi-Perps 약관 미동의. 서명 API는 부르지 않는다: 알리고 이 시간만큼 쉰다
TRADFI_CODE = -4411
UNAVAILABLE_MS = 24 * 3_600_000
TRADFI = "TRADIFI_PERPETUAL"
# exchangeInfo underlyingType → tradingSchedule 시장 키. 모르는 종류는 막지 않는다
SESSION_KEYS = ("EQUITY", "KR_EQUITY", "HK_EQUITY", "CN_EQUITY", "COMMODITY", "FX")
INTERVAL = "15m"
BAR_MS = 15 * 60_000
LEVERAGE = 3
STOP_PCT = 0.03
# AI가 손절·익절 폭을 고르나. 2026-11-12 사용자 승인으로 끔:
# 손절 3% 고정·익절 없음(백테스트와 같은 규칙).
# 근거: 10종목 1년 백테스트에서 AI가 고른 손절 1%·익절 1%는 10종목 모두 손실, 3% 고정은 승률 53%.
AI_EXITS = False
# 고정 익절(가격 %). 사용자 2026-11-13 "디폴트로 30프로 이상되면 강제익절, 1배짜리가 10프로
# 오르면 30프로": 3배라 가격 +10% = 증거금 +30%. 백테스트(10종목 1년)는 익절 없음과 거의 같았다.
# 이 값이 생기기 전에 연 포지션에도 다음 실행에서 한 번 걸어 준다(_backfill_tp).
FIXED_TP_PCT = 10.0
# 수동 포지션 이어받기(사용자 2026-11-13 "수동으로 포지션 정리하거나 사면 주기적으로
# 확인해. 수동으로 샀어도 알아서 손절과 익절해야 해"). 실행마다(15분) 계좌 전체를 본다.
# - 엔진이 빈손인데 포지션이 보이면 이어받아 손절 3%·익절 +10%를 걸고 엔진 포지션처럼 관리한다
#   (거래소 평균 진입가 기준. 이미 손절가를 지났으면 손절을 걸 수 없어 시장가로 닫는다).
# - 엔진 포지션 수량·방향을 손으로 바꿨으면 새 수량·평균가로 다시 이어받는다.
# - 엔진 포지션을 손으로 닫았으면 기록만 하고 매매를 멈추지 않는다.
# - 엔진이 이미 한 종목을 들고 있을 때 다른 종목 수동 포지션은 손절·익절 주문만 걸어 지킨다(guards).
MANUAL_ADOPT = True
# 마크 가격: 체결가(CONTRACT_PRICE)는 한 번의 꼬리·허수 체결로도 손절이 걸린다. 마크 가격은 여러
# 거래소 지수로 만들어 순간 튐에 덜 걸리고 청산 기준과도 같다.
# 대신 급락 때 체결가보다 늦게 걸릴 수 있다.
WORKING_TYPE = "MARK_PRICE"
TP_KIND = "TAKE_PROFIT_MARKET"
# 시험에서 Claude 호출을 바꿔 끼우는 자리. None이면 실제 `claude -p`
AI_CALLER: Callable[[str], str] | None = None
# 사용자 지정 2026-11-08. 실험실 기본값과 같지만, 실험실 기본값이 바뀌어도
# 엔진은 바뀌지 않게 따로 고정한다
SCR_PARAMS: dict = {
    "stoch_n": 9,
    "stoch_lo": 20.0,
    "stoch_hi": 80.0,
    "cci_n": 20,
    "cci_lo": -100.0,
    "cci_hi": 100.0,
    "rsi_n": 21,
    "rsi_lo": 30.0,
    "rsi_hi": 70.0,
    "w": 3,
    "need": 3,
    "mode": "cross",
}
# 가용 잔고를 다 쓰면 수수료(3배 × 0.05%)·체결 미끄러짐 때문에 증거금 부족으로 거부될 수 있다.
# 2026-11-12 23:45 NVDA 진입이 1%로 -2019(Margin is insufficient) 거부: 지정가(매수 호가)가 마크
# 가격보다 위면 그 차이(미실현 손실)도 증거금에 잡힌다. TradFi는 장 시작 무렵 차이가 커서 3%로 늘림.
FEE_BUFFER = 0.03
# 진입 주문 실패 뒤 다시 시도(사용자 2026-11-12 "거래실패시 백오프 기능넣어"). 포지션이 0이고 남은
# 진입 주문을 취소한 뒤에만, 새 주문 ID로 이 간격(초)만큼 쉬고 다시 낸다. 증거금 부족(-2019)이면
# 수량을 ENTRY_SHRINK씩 줄인다. 실행 시간 한도를 넘기면 멈춘다.
ENTRY_BACKOFF_S = (2.0, 5.0, 10.0)
ENTRY_SHRINK = 0.95
MARGIN_CODE = -2019
# 다시 해도 되는 거부: 거래소가 주문을 받지 않았다고 분명히 말한 것만(증거금 부족, 서버 과부하).
# 응답을 잃은 통신 오류·-1001·-1007(처리 여부 모름)은 다시 내지 않는다: 이미 체결됐을 수 있어
# 두 번 들어가게 된다(대사가 이어 받는다).
ENTRY_RETRY_CODES = frozenset({MARGIN_CODE, -1008})
TAKER_FEE = 0.0005  # 드라이런 손익 추정용 편도 테이커 수수료
EQUITY_FLOOR = 0.80
WINRATE_MIN = 0.50
WINRATE_MIN_TRADES = 8
WINRATE_DAYS = 30
RETRIES = 3
RETRY_GAP = 10
# 타이머가 한 번 빠져도 따라잡되, 30분 넘은 신호로는 진입하지 않는다
MAX_SIGNAL_BARS = 2
# 복구 모드(사용자 2026-11-09 "EC2 껐다 켰을 때 문제 안 되도록, 다시 올라오면 리커버리 모드"):
# 지난 실행에서 이만큼 넘게 지났으면 서버가 꺼졌거나 타이머가 빠진 것으로 본다
# (한 번 건너뛴 실행은 평소 2봉 창이 따라잡으므로 그보다 길게)
RECOVER_GAP_MS = 35 * 60_000
# 복구 때 놓친 신호를 찾아보는 최대 봉 수(24시간)
RECOVER_MAX_BARS = 96
# 복구 때 새 진입은 평소와 같은 2봉(30분) 안 신호만. 더 오래된 신호는 반대 포지션 정리만 한다
# (레드팀 2026-11-09: 늦은 진입은 백테스트에 없는 규칙)
RECOVER_ENTRY_BARS = MAX_SIGNAL_BARS
# 부팅 직후 네트워크·시계가 덜 잡혔을 수 있어 복구 실행은 이 시간 안에서 다시 시도한다
# (systemd TimeoutStartSec 900초보다 넉넉히 짧게: 강제 종료되면 알림도 못 보낸다)
RECOVER_DEADLINE_S = 480
RECOVER_TRY_GAP = 30
# 마지막 마감 봉이 이보다 오래됐으면 시세가 멈춘 것으로 보고 새 진입을 하지 않는다
STALE_MS = 2 * BAR_MS
# 거래소 지연으로 실행이 길어지면 새 진입은 하지 않는다(systemd 제한 600초 전에 손절까지 끝나게)
MAX_RUN_SECS = 180
# 진입 방식(사용자 2026-11-08 22:48 "급등/급락이 아니면 지정가로, 안 되면 조금 더 올려서라도"):
# 급변동이면 시장가, 아니면 post-only(GTX) 지정가를 최우선 호가에 걸고 10초마다 한 틱씩 시장 쪽으로
# 옮긴다. 시간이 다 되면 남은 수량은 시장가. `config.json`의 "entry"로 바꿀 수 있다.
ENTRY_DEFAULTS: dict = {
    "fast_1m_pct": 0.3,  # 최근 1분봉 시가 대비 지금 가격 변화(%) 이상이면 급변동
    "fast_range_atr": 1.0,  # 지금 15분봉 고저 폭이 ATR(14, 15분) × 이 값 이상이면 급변동
    "fast_from_signal_pct": 0.5,  # 신호 봉 종가에서 진입 방향으로 이만큼(%) 이미 갔으면 급변동
    "ladder_rounds": 6,  # 지정가 최대 회차
    "ladder_wait_s": 10,  # 회차마다 기다리는 시간
    "ladder_poll_s": 2,  # 체결 확인 간격
    "ladder_max_s": 70,  # 사다리 전체 시간 한도(넘으면 남은 수량 시장가)
}
MAKER_FEE = 0.0002
# 한 실행의 전체 시간 한도(systemd 600초 전에 손절까지 끝나게). 넘으면 사다리는 시장가 없이 멈춘다
RUN_DEADLINE_S = 420
GTX_REJECT = (-5022,)  # post-only가 바로 체결될 가격이라 거부됨 → 새 호가로 다시
# AI가 연달아 기본값으로 돌아가면(인증 만료·한도) 알린다
AI_FALLBACK_ALERT = 3
PREFIX = "mnv-"
NOT_FOUND = (-2013, -2011)

DIR = Path(os.environ.get("MINERVA_BTC_TRADE_DIR", "~/.minerva/btc_trade")).expanduser()
CAPITAL_FILE = binance.STATE_DIR / "binance_capital.json"

_sleep = time.sleep
_clock = time.monotonic


def _name(sym: str) -> str:
    """화면·알림용 짧은 이름(ETHUSDT → ETH)."""
    return sym[:-4] if sym.endswith("USDT") else sym


def _who(sym: str, side: str) -> str:
    return f"{_name(sym)} {SIDE_KO[side]}"


def _px(x: float) -> str:
    """가격 표시. BTC·ETH는 소수 1자리(예전 그대로), SOXL·NVDA처럼 싼 종목은 2자리,
    1보다 싼 종목(PEPE 등)은 유효 숫자 6자리."""
    if abs(x) >= 1000:
        return f"{x:,.1f}"
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.6g}"


def universe(now_ms: int | None = None) -> dict:
    """이번에 쓸 종목 범위(부작용 없음). universe.json이 쓸 만하면 그 상위 50, 아니면 네 종목.
    {"source", "updated_at", "symbols", "winrate", "priority", "stats"}"""
    now_ms = now_ms if now_ms is not None else _ms_now()
    data = btc_universe.load(DIR / "universe.json", now_ms)
    if data is None:
        return {
            "source": "default",
            "updated_at": None,
            "symbols": DEFAULT_SYMBOLS,
            "winrate": dict(DEFAULT_WINRATE),
            "priority": tuple(sorted(DEFAULT_SYMBOLS, key=lambda s: -DEFAULT_WINRATE[s])),
            "stats": {},
            "tradable": _default_tradable(),
        }
    syms = tuple(dict.fromkeys(data["symbols"]))
    stats = {s: data["stats"].get(s) or {} for s in syms}
    return {
        "source": "universe",
        "updated_at": data.get("updated_at"),
        "symbols": syms,
        "winrate": {s: float(stats[s].get("winrate") or 0) for s in syms},
        "priority": tuple(btc_universe.priority(list(syms), stats)),
        "stats": stats,
        "tradable": tradable_set(syms, stats),
    }


def tradable_set(syms, stats: dict) -> frozenset:
    """새로 들어갈 수 있는 종목. 백테스트 승률 ≥ MIN_ENTRY_WINRATE(사용자 2026-11-12 "승률 50프로
    미만은 들어가지마" → 같은 날 45로 낮춤, 네 종목도 예외 없음)이고 표본이 충분한 종목.
    백테스트 수익 > 0 조건은 사용자 "그냥 빼자"(2026-11-12)로 없앴다. 나머지는 신호·화면만."""
    ok = set()
    for s in syms:
        st = stats.get(s) or {}
        if float(st.get("winrate") or 0) >= MIN_ENTRY_WINRATE and not btc_universe.low_sample(st):
            ok.add(s)
    return frozenset(ok)


def _default_tradable() -> frozenset:
    return frozenset(s for s in DEFAULT_SYMBOLS if DEFAULT_WINRATE[s] >= MIN_ENTRY_WINRATE)


def tradable(sym: str) -> bool:
    t = UNIVERSE.get("tradable")
    return sym in (t if t is not None else _default_tradable())


def use_universe(now_ms: int | None = None) -> dict:
    """이번 실행의 종목·승률·우선순위를 모듈 값으로 둔다(한 번 실행하고 끝나는 프로세스)."""
    global SYMBOLS, SYMBOL_WINRATE, PRIORITY, UNIVERSE
    u = universe(now_ms)
    SYMBOLS, SYMBOL_WINRATE, PRIORITY, UNIVERSE = u["symbols"], u["winrate"], u["priority"], u
    _TAGS.clear()
    _TAGS.update(cid_tags(SYMBOLS))
    return u


# 이번 실행의 종목 → 주문 ID 태그(겹침을 막은 값). 없으면 cid_tag
_TAGS: dict[str, str] = {}


def cid_tag(symbol: str) -> str:
    """주문 ID 종목 표시. 네 종목은 예전 그대로, 나머지는 기초자산 글자 앞 4자 + 종목 이름에서 정한
    확인 글자 1자(최대 5자). 종목 이름만으로 정해져 실행마다 같다. 글자가 같은 두 종목
    (예: 1000PEPE·PEPE)도 확인 글자가 다를 가능성이 높고, `cid_tags`가 겹침을 다시 막는다."""
    if symbol in CID_TAG:
        return CID_TAG[symbol]
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    letters = "".join(ch for ch in base.upper() if "A" <= ch <= "Z")[:4] or "X"
    check = chr(ord("A") + zlib.crc32(symbol.encode()) % 26)
    return letters + check


def cid_tags(symbols) -> dict[str, str]:
    """여러 종목의 태그. 겹치면(드묾) 뒤 종목은 이름 해시 5글자로 바꾼다. 네 종목 태그는 그대로."""
    out: dict[str, str] = {}
    used = set(CID_TAG.values())
    for s in sorted(set(symbols), key=lambda x: (x not in CID_TAG, x)):
        t = cid_tag(s)
        if s not in CID_TAG and t in used:
            h = zlib.crc32(("mnv:" + s).encode())
            t = "".join(chr(ord("A") + (h // 26**k) % 26) for k in range(5))
        used.add(t)
        out[s] = t
    return out


def pos_symbol(pos: dict | None) -> str:
    """포지션·pending의 종목. 옛 상태에는 symbol이 없다 → BTCUSDT."""
    return (pos or {}).get("symbol") or SYMBOL


TERMINAL = ("CANCELED", "FILLED", "EXPIRED", "REJECTED")

# ---------------------------------------------------------------- 파일·상태


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _save(path: Path, data) -> None:
    _write_private(path, json.dumps(data, ensure_ascii=False, indent=1))


def _append(name: str, entry: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(DIR / name, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_jsonl(name: str) -> list[dict]:
    path = DIR / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:  # 다른 프로세스가 쓰는 중인 마지막 줄
            log.warning("%s: 읽지 못한 줄을 건너뜀", name)
    return out


def mode() -> str:
    path = DIR / "mode"
    return "live" if path.exists() and path.read_text().strip() == "live" else "dry"


def halted() -> bool:
    return (DIR / "HALT").exists()


def entries_blocked() -> bool:
    return (DIR / "NO_ENTRY").exists()


def entry_cfg() -> dict:
    """진입 방식 설정: ENTRY_DEFAULTS 위에 `config.json`의 "entry"를 덮는다(숫자만)."""
    over = _load(DIR / "config.json", {}).get("entry") or {}
    out = dict(ENTRY_DEFAULTS)
    for k, v in over.items():
        if k in out and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            out[k] = v
    return out


def margin_frac() -> float:
    """가용 잔고 중 증거금으로 쓸 비율. `config.json`의 margin_frac, 기본 1.0, 0~1로 자른다."""
    v = float(_load(DIR / "config.json", {}).get("margin_frac", 1.0))
    return min(max(v, 0.0), 1.0)


def _ms_now() -> int:
    return int(time.time() * 1000)


def _now_iso(now: datetime) -> str:
    return now.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")


def _bar_txt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, KST).strftime("%m-%d %H:%M")


SIDE_KO = {"long": "롱", "short": "숏"}


# ---------------------------------------------------------------- 바이낸스 통신


class NetworkError(RuntimeError):
    """응답을 못 받았다(통신 오류·5xx). 주문이 들어갔는지 모른다."""


class Rejected(RuntimeError):
    """거래소가 거부했다(4xx). 다시 보내도 같다."""

    def __init__(self, code: int, msg: str, status: int = 400):
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg
        self.status = status


class Client:
    """서명 요청. 키는 `~/.minerva/binance.env`에서만 읽고 로그·예외에 넣지 않는다."""

    def __init__(self, cfg: dict | None = None):
        self._cfg = cfg if cfg is not None else binance.load_config()

    def request(self, method: str, path: str, params: dict | None = None, signed: bool = True):
        params = dict(params or {})
        base = SPOT_BASE if path.startswith("/sapi/") else BASE
        headers = {"User-Agent": "minerva-btc-trade"}
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 10000
            query = urllib.parse.urlencode(params)
            sig = hmac.new(self._cfg["secret"].encode(), query.encode(), hashlib.sha256)
            query += "&signature=" + sig.hexdigest()
            headers["X-MBX-APIKEY"] = self._cfg["key"]
        else:
            query = urllib.parse.urlencode(params)
        url = f"{base}{path}?{query}" if query else f"{base}{path}"
        req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            if e.code >= 500:
                raise NetworkError(f"HTTP {e.code}: {body}") from None
            try:
                data = json.loads(body)
                raise Rejected(int(data.get("code", 0)), str(data.get("msg", "")), e.code) from None
            except ValueError:
                raise Rejected(0, body, e.code) from None
        except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
            raise NetworkError(type(e).__name__ + ": " + str(getattr(e, "reason", e))) from None


class SymClient:
    """한 종목에 묶인 클라이언트. 요청은 원래 클라이언트로 보내고 종목만 들고 다닌다
    (모듈 전역 종목을 바꾸지 않아 종목끼리 섞이지 않는다)."""

    def __init__(self, base, symbol: str):
        self.base = getattr(base, "base", base)
        self.symbol = symbol

    def request(self, *a, **kw):
        return self.base.request(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.base, name)


def _sym(client) -> str:
    return getattr(client, "symbol", None) or SYMBOL


def _get(client, path: str, signed: bool = True, **params):
    """GET은 결과가 바뀌지 않으니 통신 오류면 그냥 다시 묻는다."""
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        if attempt:
            _sleep(RETRY_GAP)
        try:
            return client.request("GET", path, params, signed)
        except NetworkError as e:
            last = e
    raise NetworkError(f"GET {path} 실패: {last}")


def _query(client, path: str, id_key: str, cid: str) -> dict | None:
    try:
        params = {id_key: cid}
        if path == "/fapi/v1/order":
            params["symbol"] = _sym(client)
        return client.request("GET", path, params)
    except Rejected as e:
        if e.code in NOT_FOUND:
            return None
        raise


def _send(
    client, method: str, path: str, params: dict, query_path: str, id_key: str, resend: bool = True
) -> dict:
    """주문 보내기(멱등). 같은 ID가 이미 있으면 그걸 돌려준다.

    통신 오류만 RETRY_GAP 간격 RETRIES회 다시 보낸다. 다시 보내기 전에 조회해 중복을 막는다.
    거래소 거부(Rejected)는 그대로 올린다. resend=False면 한 번만 보내고, 응답을 잃으면 조회만
    한다. 바이낸스 ID 중복 검사는 미체결 주문끼리만이라
    즉시 체결된 시장가 진입은 같은 ID로 두 번 들어갈 수 있다.
    """
    sent = False
    cid = params["clientAlgoId" if id_key == "clientAlgoId" else "newClientOrderId"]
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        if attempt:
            _sleep(RETRY_GAP)
        try:
            found = _query(client, query_path, id_key, cid)
        except NetworkError as e:
            last = e
            continue
        if found:
            return {**found, "_found": True}
        if sent and not resend:
            continue
        try:
            sent = True
            return client.request(method, path, params)
        except NetworkError as e:
            last = e
            log.warning("%s %s attempt %d failed: %s", method, cid, attempt + 1, e)
    raise NetworkError(f"{cid} 응답 없음 ({RETRIES + 1}회): {last}")


# 종목별 거래 규칙 캐시 {종목: (받은 시각, 규칙)}. exchangeInfo 한 번에 모든 종목을 담는다
_FILTERS: dict[str, tuple[float, dict]] = {}
FILTER_TTL_S = 600


def filters(client) -> dict:
    """클라이언트 종목의 호가·수량 단위·최소 주문 금액. 종목마다 다르다(ETH 0.001·0.01·20,
    SOXL·NVDA 0.01·0.01·5)."""
    want = _sym(client)
    hit = _FILTERS.get(want)
    if hit and time.time() - hit[0] < FILTER_TTL_S:
        return dict(hit[1])
    info = _get(client, "/fapi/v1/exchangeInfo", signed=False)
    t = time.time()
    for s in info["symbols"]:
        try:
            _FILTERS[s["symbol"]] = (t, _parse_filters(s))
        except (KeyError, TypeError, ValueError, ArithmeticError):
            continue  # 모양이 다른 종목 하나 때문에 전체를 버리지 않는다
    if want not in _FILTERS:
        return {
            "tick": "0.01",
            "step": "0.001",
            "min_qty": 0.0,
            "max_qty": 0.0,
            "min_notional": 0.0,
            "status": "없음",
        }
    return dict(_FILTERS[want][1])


def _parse_filters(sym: dict) -> dict:
    f = {x["filterType"]: x for x in sym["filters"]}
    lot = f.get("MARKET_LOT_SIZE") or f["LOT_SIZE"]
    # SOXL·NVDA는 호가 단위가 "0.01000"으로 온다. 그대로 쓰면 가격이 "61.22000"처럼 나가
    # 종목 정밀도를 넘을 수 있어 뒤 0을 뗀다("0.10" → "0.1", 지수 표기 없이)
    return {
        "tick": format(Decimal(f["PRICE_FILTER"]["tickSize"]).normalize(), "f"),
        "step": format(Decimal(lot["stepSize"]).normalize(), "f"),
        "min_qty": float(lot["minQty"]),
        "max_qty": float(lot["maxQty"]),
        "min_notional": float(f.get("MIN_NOTIONAL", {}).get("notional", 0)),
        "status": sym.get("status"),
        "contract": sym.get("contractType"),
        "underlying": sym.get("underlyingType"),
    }


def _round(x: float, step: str, how=ROUND_FLOOR) -> Decimal:
    s = Decimal(step)
    return (Decimal(str(x)) / s).to_integral_value(rounding=how) * s


def size_qty(available: float, price: float, frac: float, flt: dict) -> tuple[Decimal, str]:
    """(수량, 못 사는 이유). 수량은 LOT_SIZE 단위로 내림."""
    raw = available * frac * (1 - FEE_BUFFER) * LEVERAGE / price
    qty = _round(raw, flt["step"])
    qty = min(qty, _round(flt["max_qty"], flt["step"]))
    if qty <= 0 or float(qty) < flt["min_qty"]:
        return Decimal(0), f"수량 {raw:.6f} < 최소 {flt['min_qty']}"
    if float(qty) * price < flt["min_notional"]:
        return Decimal(0), f"주문 금액 {float(qty) * price:.2f} < 최소 {flt['min_notional']} USDT"
    return qty, ""


def stop_price(side: str, entry: float, tick: str, pct: float = STOP_PCT) -> Decimal:
    """롱은 아래로, 숏은 위로 내림·올림해 정한 폭보다 좁아지지 않게 한다. pct는 비율(0.03)."""
    if side == "long":
        return _round(round(entry * (1 - pct), 10), tick, ROUND_FLOOR)
    return _round(round(entry * (1 + pct), 10), tick, ROUND_CEILING)


def tp_price(side: str, entry: float, tick: str, pct: float) -> Decimal:
    """익절가. 정한 폭보다 가까워지지 않게 롱은 올림, 숏은 내림. 부동소수 끝자리
    (100000 × 1.1 = 110000.00000000001)로 한 틱 밀리지 않게 먼저 10자리로 반올림한다."""
    if side == "long":
        return _round(round(entry * (1 + pct), 10), tick, ROUND_CEILING)
    return _round(round(entry * (1 - pct), 10), tick, ROUND_FLOOR)


def position_amt(client) -> float:
    sym = _sym(client)
    rows = _get(client, "/fapi/v2/positionRisk", symbol=sym)
    return sum(float(r["positionAmt"]) for r in rows if r["symbol"] == sym)


def all_position_amts(client) -> dict[str, float]:
    """계좌 전체 포지션 수량(한 번의 조회, 종목 범위 밖도 포함). 범위 종목은 0이어도 넣는다."""
    out = {s: 0.0 for s in SYMBOLS}
    for r in _get(client, "/fapi/v2/positionRisk"):
        sym = r.get("symbol")
        if sym:
            out[sym] = out.get(sym, 0.0) + float(r.get("positionAmt") or 0)
    return out


def open_algos(client) -> list[dict]:
    rows = _get(client, "/fapi/v1/openAlgoOrders", symbol=_sym(client))
    if isinstance(rows, dict):  # 응답 모양이 바뀌어도 목록만 꺼낸다
        rows = rows.get("orders") or rows.get("rows") or []
    return [r for r in rows if isinstance(r, dict)]


def foreign_orders(client) -> list[str]:
    """엔진이 내지 않은 그 종목 미체결(일반·조건부). 단방향 모드라 체결되면 엔진 포지션에 섞인다."""
    out = [
        o.get("clientOrderId", "")
        for o in _get(client, "/fapi/v1/openOrders", symbol=_sym(client))
        if not str(o.get("clientOrderId", "")).startswith(PREFIX)
    ]
    out += [
        a.get("clientAlgoId", "")
        for a in open_algos(client)
        if not str(a.get("clientAlgoId", "")).startswith(PREFIX)
    ]
    return out


def cancel_engine_stops(client, keep: str | None = None) -> list[str]:
    """엔진 손절(접두어 mnv-)만 취소한다. 사용자 주문은 건드리지 않는다."""
    done = []
    for a in open_algos(client):
        cid = str(a.get("clientAlgoId", ""))
        if not cid.startswith(PREFIX) or cid == keep:
            continue
        try:
            _delete(client, "/fapi/v1/algoOrder", {"clientAlgoId": cid})
        except Rejected as e:
            if e.code not in NOT_FOUND:  # 이미 걸렸거나 취소된 건 괜찮다
                raise
        done.append(cid)
        _append("orders.jsonl", {"at": _now_iso(datetime.now(KST)), "cancel": cid})
    return done


def _delete(client, path: str, params: dict):
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        if attempt:
            _sleep(RETRY_GAP)
        try:
            return client.request("DELETE", path, params)
        except NetworkError as e:
            last = e
    raise NetworkError(f"DELETE {path} 실패: {last}")


def market(client, side: str, qty: Decimal, cid: str, reduce: bool) -> dict:
    """시장가. side는 BUY/SELL. 체결 결과(평균가·수량)를 돌려준다."""
    params = {
        "symbol": _sym(client),
        "side": side,
        "type": "MARKET",
        "quantity": str(qty),
        "newClientOrderId": cid,
        "newOrderRespType": "RESULT",
    }
    if reduce:
        params["reduceOnly"] = "true"
    # 진입(reduce=False)은 다시 보내지 않는다. reduceOnly는 포지션을 늘릴 수 없어 다시 보내도 된다
    r = _send(
        client, "POST", "/fapi/v1/order", params, "/fapi/v1/order", "origClientOrderId", reduce
    )
    if float(r.get("avgPrice") or 0) == 0 and r.get("status") != "FILLED":
        _sleep(1)
        try:
            r = _query(client, "/fapi/v1/order", "origClientOrderId", cid) or r
        except NetworkError:
            pass  # 체결 결과는 대사(포지션 조회)가 다시 확인한다
    _append("orders.jsonl", {"at": _now_iso(datetime.now(KST)), "req": params, "res": _slim(r)})
    return r


def place_stop(client, side: str, trigger: Decimal, cid: str, kind: str = "STOP_MARKET") -> dict:
    """포지션 전체를 닫는 조건부 주문(Algo Service). side는 포지션 방향(long/short).
    kind = STOP_MARKET(손절) 또는 TAKE_PROFIT_MARKET(익절)."""
    params = {
        "algoType": "CONDITIONAL",
        "symbol": _sym(client),
        "side": "SELL" if side == "long" else "BUY",
        "type": kind,
        "triggerPrice": str(trigger),
        "closePosition": "true",
        "workingType": WORKING_TYPE,
        "clientAlgoId": cid,
    }
    r = _send(client, "POST", "/fapi/v1/algoOrder", params, "/fapi/v1/algoOrder", "clientAlgoId")
    _append("orders.jsonl", {"at": _now_iso(datetime.now(KST)), "req": params, "res": _slim(r)})
    return r


def _slim(r: dict) -> dict:
    keys = (
        "orderId",
        "clientOrderId",
        "status",
        "avgPrice",
        "executedQty",
        "algoId",
        "clientAlgoId",
        "algoStatus",
        "triggerPrice",
        "_found",
    )
    return {k: r[k] for k in keys if k in r}


def stop_is_open(client, cid: str, tries: int = 3) -> bool:
    """걸린 조건부 주문이 목록에 보이나. 방금 건 주문은 목록 반영이 늦을 수 있어 몇 번 본다."""
    for k in range(tries):
        if k:
            _sleep(1)
        if any(a.get("clientAlgoId") == cid for a in open_algos(client)):
            return True
    return False


def stop_placed(client, r: dict, cid: str) -> bool:
    """손절이 걸렸나. 응답이 NEW면 걸린 것으로 보고, 아니면 목록에서 확인한다."""
    if r.get("algoStatus") == "NEW":
        return True
    return r.get("algoStatus") is None and stop_is_open(client, cid)


def _close_result(client, pos: dict, close_side: str) -> tuple[float | None, float | None]:
    """(청산 평균가, 순손익 USDT). 진입 이후 체결로 계산: 실현손익 − 수수료(USDT분).

    방금 닫은 체결이 조회에 늦게 보일 수 있어 청산 수량이 다 보일 때까지 몇 번 묻는다.
    userTrades는 한 번에 7일 창이라 진입부터 지금까지를 7일씩 나눠 묻는다.
    청산 체결을 다 못 찾으면 손익은 None(승률 계산에서 빠진다). BNB로 낸 수수료는 빠진다.
    """
    start = int(pos["entry_ms"]) - 1000
    # 갈아타기: 같은 실행에서 닫은 앞 포지션의 체결(손익·수수료)이 이 창에 들어오지 않게
    after = int(pos.get("after_ms") or 0)
    trades: list[dict] = []
    for k in range(3):
        if k:
            _sleep(1)
        trades, t0, end = [], start, int(time.time() * 1000) + 60_000
        while t0 < end:
            t1 = min(t0 + 7 * 86_400_000 - 1, end)
            trades += _get(
                client,
                "/fapi/v1/userTrades",
                symbol=_sym(client),
                startTime=t0,
                endTime=t1,
                limit=1000,
            )
            t0 = t1 + 1
        trades = [t for t in trades if int(t["time"]) > after]
        exits = [t for t in trades if t["side"] == close_side and int(t["time"]) >= start]
        if sum(float(t["qty"]) for t in exits) >= pos["qty"] - 1e-9:
            break
    exits = [t for t in trades if t["side"] == close_side and int(t["time"]) >= start]
    qty = sum(float(t["qty"]) for t in exits)
    px = sum(float(t["qty"]) * float(t["price"]) for t in exits) / qty if qty else None
    if qty < pos["qty"] - 1e-9:
        return px, None
    pos["closed_ms"] = max(int(t["time"]) for t in exits)
    pnl = sum(float(t["realizedPnl"]) for t in trades) - sum(
        float(t["commission"]) for t in trades if t.get("commissionAsset", "USDT") == "USDT"
    )
    return px, pnl


# ---------------------------------------------------------------- 시세·신호


# 공개 시세에서 429·418을 한 번 받으면 이번 실행의 남은 공개 요청은 보내지 않는다
# (레드팀 2026-11-12: 50종목 동시 받기가 429 뒤에도 계속 보내면 418 차단으로 번지고,
# 같은 IP의 주문도 막힌다)
_RATE_HIT = threading.Event()


def _public(url: str, timeout: float = 15):
    if _RATE_HIT.is_set():
        raise ConnectionError("요청 한도(429/418) 뒤라 이번 실행의 공개 시세 요청을 멈춤")
    req = urllib.request.Request(url, headers={"User-Agent": "minerva"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            _RATE_HIT.set()
            log.error("공개 시세 요청 한도 %s: 이번 실행의 공개 요청을 멈춤", e.code)
        raise


def _close_qty(amt: float) -> Decimal:
    """거래소가 준 포지션 수량을 그대로(뒤 0 없이) 쓴다. 이미 그 종목 수량 단위라 내림이 필요 없고,
    0.001로 맞추면 단위 1·0.01 종목에 '12345.000'처럼 정밀도 넘는 값이 나간다(레드팀 2026-11-12)."""
    d = Decimal(repr(abs(float(amt)))).normalize()
    return Decimal(format(d, "f"))


class Market:
    """한 종목의 공개 시세(GET, 키 없음). 봉은 한 번 실행 안에서 캐시한다.
    `for_symbol`로 다른 종목 시세를 얻는다(같은 실행 안에서 종목마다 하나)."""

    def __init__(self, symbol: str = SYMBOL, pool: dict | None = None):
        self.symbol = symbol
        self._cache: dict[str, list[list]] = {}
        self._pool = pool if pool is not None else {}
        self._pool.setdefault(symbol, self)

    def for_symbol(self, symbol: str) -> Market:
        if symbol not in self._pool:
            Market(symbol, self._pool)
        return self._pool[symbol]

    def klines(self, interval: str) -> list[list]:
        if interval not in self._cache:
            # 신호용 15분봉은 499개(무게 2): 50종목을 매번 받는다. RSI 21 와일더 평균은 400봉이면
            # 1000봉과 값이 사실상 같고(차이 1e-6 미만), 복구 창 96봉도 넉넉하다
            n = ENGINE_BARS if interval == INTERVAL else btc.LIMIT
            self._cache[interval] = _public(
                btc.URL.format(sym=self.symbol, iv=interval, n=n), timeout=10
            )
        return self._cache[interval]

    def price(self) -> float:
        try:
            return float(_public(btc.PRICE_URL.format(sym=self.symbol))["price"])
        except (OSError, ValueError, KeyError) as e:
            raise NetworkError(f"{self.symbol} 시세 조회 실패: {type(e).__name__}") from None

    def book(self) -> tuple[float, float]:
        """최우선 매수·매도 호가(공개, 캐시 안 함)."""
        req = urllib.request.Request(
            f"{BASE}/fapi/v1/ticker/bookTicker?symbol={self.symbol}",
            headers={"User-Agent": "minerva"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.load(r)
            return float(d["bidPrice"]), float(d["askPrice"])
        except (OSError, ValueError, KeyError) as e:
            raise NetworkError(f"호가 조회 실패: {type(e).__name__}") from None

    def minutes(self, n: int = 20) -> list[list]:
        """최근 1분봉(공개, 캐시 안 함). 마지막은 진행 중인 봉."""
        url = f"{BASE}/fapi/v1/klines?symbol={self.symbol}&interval=1m&limit={n}"
        req = urllib.request.Request(url, headers={"User-Agent": "minerva"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.load(r)
        except (OSError, ValueError) as e:
            raise NetworkError(f"1분봉 조회 실패: {type(e).__name__}") from None

    def funding(self) -> float | None:
        req = urllib.request.Request(
            f"{BASE}/fapi/v1/premiumIndex?symbol={self.symbol}", headers={"User-Agent": "minerva"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return float(json.load(r)["lastFundingRate"])


def latest_signal(
    rows: list[list], now_ms: int, last_bar: int | None, lookback: int = MAX_SIGNAL_BARS
) -> dict:
    """마감된 봉에서 아직 처리하지 않은 가장 최근 신호(최근 lookback봉 안, 복구 때만 길다).

    {"dir": long/short/None, "bar": 봉 시작 ms, "close": 종가, "conflict": 같은 봉 양쪽 신호,
     "missed": 그 창 안에서 처리 안 한 신호 수}
    """
    closed = [r for r in rows if int(r[6]) < now_ms]
    h = [float(r[2]) for r in closed]
    lo = [float(r[3]) for r in closed]
    c = [float(r[4]) for r in closed]
    t = [int(r[0]) for r in closed]
    out: dict = {"dir": None, "bar": t[-1] if t else None, "close": c[-1] if c else None}
    if len(c) < 100:
        out["error"] = f"15분봉이 {len(c)}개뿐"
        return out
    out["stale"] = now_ms - (t[-1] + BAR_MS) > STALE_MS
    sig = scr_lab.signals(h, lo, c, SCR_PARAMS)
    at: dict[int, str] = {}
    for d in ("long", "short"):
        for i in sig[d]:
            at[i] = "both" if i in at and at[i] != d else d
    n = len(c)
    found = None
    missed = 0
    for i in range(n - 1, max(n - 1 - lookback, -1), -1):
        if last_bar is None and i != n - 1:
            break
        if last_bar is not None and t[i] <= last_bar:
            break
        if i in at:
            missed += 1
            found = i if found is None else found
    if found is not None:
        i = found
        out.update(bar=t[i], close=c[i], age=n - 1 - i, bar_txt=_bar_txt(t[i]), missed=missed)
        out["at_signal"] = {k: _r(sig[k][i]) for k in ("stoch", "cci", "rsi")}
        if at[i] == "both":
            out["conflict"] = True
        else:
            out["dir"] = at[i]
        return out
    out["indicators"] = {
        "stoch": _r(sig["stoch"][-1]),
        "cci": _r(sig["cci"][-1]),
        "rsi": _r(sig["rsi"][-1]),
    }
    return out


def _r(v, n: int = 1):
    return None if v is None else round(v, n)


def chase_gate(market: Market, rows15: list[list], now_ms: int) -> dict:
    """추격 한도(btc.entry_gate 재사용): 15분·4시간·일봉 TM 방향이 모두 같으면 3%, 아니면 1%."""
    frames = []
    for key, name in (("15m", "15분봉"), ("4h", "4시간봉"), ("1d", "일봉")):
        rows = rows15 if key == "15m" else market.klines(key)
        bars, opened = btc.parse(rows, now_ms)
        frames.append(btc.frame_status(key, name, bars, opened))
    return btc.entry_gate(frames)


def chase_ok(side: str, price: float, ref: float, pct: float) -> bool:
    """신호 봉 종가에서 진입 방향으로 pct% 이상 벌어졌으면 False."""
    if side == "long":
        return price < ref * (1 + pct / 100)
    return price > ref * (1 - pct / 100)


# ---------------------------------------------------------------- 정지 규칙


def winrate(mode_: str, now: datetime) -> tuple[int, int]:
    """(최근 WINRATE_DAYS일 청산 수, 이긴 수). 같은 모드만."""
    since = _now_iso(now - timedelta(days=WINRATE_DAYS))
    rows = [
        r
        for r in _read_jsonl("closed.jsonl")
        if r.get("mode") == mode_ and r.get("at", "") >= since and r.get("pnl") is not None
    ]
    return len(rows), sum(1 for r in rows if (r.get("pnl") or 0) > 0)


def seed(mode_: str, state: dict) -> float | None:
    """시드(USDT, 입출금 반영). 실전은 binance_capital.json, 드라이런은 처음 돌 때 잡은 평가액."""
    if mode_ == "live":
        cap = _load(CAPITAL_FILE, {})
        if not cap.get("live") or not cap.get("base"):
            return None
        return float(cap["base"]) + sum(float(d.get("amount", 0)) for d in cap.get("deposits", []))
    return state.get("dry_seed")


def set_no_entry(reason: str, send: Callable[[str], Any]) -> None:
    if entries_blocked():
        return
    _write_private(DIR / "NO_ENTRY", f"{_now_iso(datetime.now(KST))} {reason}\n")
    _append("events.jsonl", {"at": _now_iso(datetime.now(KST)), "event": "no_entry", "why": reason})
    send(
        f"🚨 BTC 자동매매 신규 진입 정지: {reason}\n"
        "보유 포지션·손절·반대 신호 청산은 계속해요.\n"
        "다시 열려면 rm ~/.minerva/btc_trade/NO_ENTRY"
    )


def check_halts(mode_: str, state: dict, equity: float | None, now: datetime, send) -> list[str]:
    """정지 사유 목록. 새로 걸린 규칙은 NO_ENTRY를 만들고 알린다."""
    why = []
    if halted():
        why.append("HALT 파일")
        if not state.get("halt_seen"):
            send("⏸ BTC 자동매매: HALT 파일이 있어 신규 진입을 멈춰요 (보유·손절은 관리)")
        state["halt_seen"] = True
    else:
        state.pop("halt_seen", None)
    s = seed(mode_, state)
    if mode_ == "live" and s is None:
        why.append("시드 없음(start-live 안 함)")
    if s and equity is not None and equity < s * EQUITY_FLOOR:
        reason = f"평가액 {equity:.2f} < 시드 {s:.2f} × {EQUITY_FLOOR:.0%}"
        why.append(reason)
        set_no_entry(reason, send)
    n, wins = winrate(mode_, now)
    if n >= WINRATE_MIN_TRADES and wins / n < WINRATE_MIN:
        reason = f"최근 {WINRATE_DAYS}일 승률 {wins}/{n} = {wins / n:.0%} < {WINRATE_MIN:.0%}"
        why.append(reason)
        set_no_entry(reason + " (30일 창에서 지는 거래가 빠지기 전에는 지워도 다시 걸려요)", send)
    if entries_blocked():
        why.append("NO_ENTRY 파일")
    return why


# ---------------------------------------------------------------- 실행 단위


def market_for(market, symbol: str):
    """그 종목 시세. `for_symbol`이 없는 시세(옛 시험용 가짜)는 BTC만 낸다(없으면 None)."""
    f = getattr(market, "for_symbol", None)
    if f is not None:
        return f(symbol)
    return market if symbol == SYMBOL else None


class Ctx:
    """한 번 실행의 문맥. `at(종목)`은 같은 실행에서 그 종목에 묶인 문맥(클라이언트·시세·봉)."""

    def __init__(self, client, market: Market, send, now: datetime, live: bool):
        self.symbol = SYMBOL
        self.client = client
        self.market = market
        self.send = send
        self.now = now
        self.live = live
        self.tag = "" if live else "[드라이런] "
        self.notes: list[str] = []
        # 종목 문맥끼리 나눠 쓰는 값(AI 연속 기본값 수, 종목별 15분봉)
        self.shared: dict = {"ai_streak": None, "rows": {}}
        self.rows15: list[list] = []
        self.deadline = _clock() + RUN_DEADLINE_S
        self.now_ms = int(now.timestamp() * 1000)

    @property
    def ai_streak(self) -> int | None:
        return self.shared["ai_streak"]

    @ai_streak.setter
    def ai_streak(self, v: int | None) -> None:
        self.shared["ai_streak"] = v

    def at(self, symbol: str) -> Ctx:
        if symbol == self.symbol:
            return self
        c = copy.copy(self)
        c.symbol = symbol
        c.client = SymClient(self.client, symbol)
        c.market = market_for(self.market, symbol)
        c.rows15 = self.shared["rows"].get(symbol, [])
        return c

    def say(self, text: str) -> None:
        self.notes.append(text)
        self.send(self.tag + text)

    def event(self, **kw) -> None:
        _append(
            "events.jsonl", {"at": _now_iso(self.now), "mode": "live" if self.live else "dry", **kw}
        )


def _cid(bar: int, tag: str, symbol: str = SYMBOL) -> str:
    """주문 ID. BTC는 예전 그대로 mnv-<분>-<tag>, ETH·SOXL·NVDA는 mnv-<E|S|N><분>-<tag>,
    나머지는 mnv-<글자 5자 이하><분>-<tag>(갈아타기 때 같은 봉에서 두 종목 주문이 나와도 겹치지
    않게, 36자 안)."""
    t = _TAGS.get(symbol)
    return f"{PREFIX}{cid_tag(symbol) if t is None else t}{bar // 60000}-{tag}"


def _pos_cid(pos: dict, tag: str, n: int) -> str:
    """다시 거는 손절(s)·익절(t)·비상 청산(x) ID. 첫 시도는 예전 그대로 <tag><n>,
    같은 신호 재시도(try ≥ 2)는 <tag><try>-<n>(앞 시도의 <tag><n>·기준 <tag><try>-와 다르다)."""
    t = pos.get("try", 1)
    return _cid(pos["bar"], f"{tag}{n}" if t == 1 else f"{tag}{t}-{n}", pos_symbol(pos))


def _record_close(ctx: Ctx, pos: dict, exit_px: float | None, pnl: float | None, why: str) -> None:
    sign = 1 if pos["side"] == "long" else -1
    pct = (exit_px / pos["entry"] - 1) * sign * 100 if exit_px else None
    _append(
        "closed.jsonl",
        {
            "at": _now_iso(ctx.now),
            "mode": pos.get("mode", "live" if ctx.live else "dry"),
            "symbol": pos_symbol(pos),
            "side": pos["side"],
            "qty": pos["qty"],
            "entry": pos["entry"],
            "exit": exit_px,
            "pnl": pnl,
            "pct": pct,
            "why": why,
            "opened_at": pos.get("opened_at"),
        },
    )
    label = {
        "signal": "반대 신호 청산",
        "stop": "손절 체결",
        "tp": "익절 체결",
        "manual": "수동·강제 청산",
        "protect": "손절 실패로 청산",
    }.get(why, why)
    sym = pos_symbol(pos)
    px = _px(exit_px) if exit_px else "?"
    money = f", 손익 {pnl:+.2f} USDT" if pnl is not None else ""
    pc = f" ({pct:+.2f}% × {LEVERAGE}배)" if pct is not None else ""
    ctx.say(f"{_who(sym, pos['side'])} {label}: {pos['qty']} {_name(sym)} @ {px}{pc}{money}")


def _triggered(st: dict | None) -> bool:
    return bool(st) and st.get("algoStatus") not in ("NEW", "CANCELED", "EXPIRED", "REJECTED")


def reconcile_dry(ctx: Ctx, state: dict, rows15: list[list], now_ms: int) -> None:
    """드라이런 가상 포지션의 손절·익절: 진입 뒤 마감된 봉의 고가·저가가 닿았나.
    한 봉에 둘 다 닿으면 손절로 본다(보수적)."""
    pos = state.get("position")
    if not pos:
        return
    sign = 1 if pos["side"] == "long" else -1
    tp = pos.get("tp")
    for r in rows15:
        if int(r[0]) <= pos["bar"] or int(r[6]) >= now_ms or int(r[0]) <= pos.get("checked", 0):
            continue
        o, h, lo = float(r[1]), float(r[2]), float(r[3])
        pos["checked"] = int(r[0])
        if (lo <= pos["stop"]) if sign > 0 else (h >= pos["stop"]):
            px, why = (min(o, pos["stop"]) if sign > 0 else max(o, pos["stop"])), "stop"
        elif tp and ((h >= tp) if sign > 0 else (lo <= tp)):
            px, why = (max(o, tp) if sign > 0 else min(o, tp)), "tp"
        else:
            continue
        pnl = sign * pos["qty"] * (px - pos["entry"]) - pos["qty"] * (pos["entry"] + px) * TAKER_FEE
        _append(
            "dry.jsonl",
            {
                "at": _now_iso(ctx.now),
                "symbol": pos_symbol(pos),
                "action": f"{why}_fill",
                "price": px,
                "bar": _bar_txt(int(r[0])),
                "cancel_other": True,
            },
        )
        state["position"] = None
        _record_close(ctx, pos, px, pnl, why)
        return


def ensure_stop(ctx: Ctx, state: dict, with_tp: bool = True) -> bool:
    """실전 엔진 포지션에 손절(·익절)이 걸려 있게 한다. 손절을 못 걸면 시장가로 닫는다.
    익절은 못 걸어도 손절이 지키므로 알리고 익절 없이 간다. 보호되면 True."""
    pos = state["position"]
    open_ids = {a.get("clientAlgoId") for a in open_algos(ctx.client)}
    if pos["stop_id"] not in open_ids:
        n = pos.get("stop_n", 0) + 1
        cid = _pos_cid(pos, "s", n)
        try:
            r = place_stop(ctx.client, pos["side"], Decimal(str(pos["stop"])), cid)
            pos.update(stop_id=cid, stop_n=n)
            _save(DIR / "state.json", state)
            ok = stop_placed(ctx.client, r, cid)
            why = "" if ok else f"상태 {r.get('algoStatus')}"
        except (Rejected, NetworkError) as e:
            ok, why = False, str(e)
        if not ok:
            if not with_tp:  # 수동 개입 포지션은 닫지 않는다: 크게 알리기만
                ctx.say(f"🆘 손절을 다시 걸지 못했어요({why}). 바이낸스 앱에서 직접 확인하세요!")
                return False
            ctx.say(f"❌ 손절 재설정 실패({why}) → 시장가로 닫아요")
            protect_close(ctx, state, "protect")
            return state.get("position") is None
        ctx.say(
            f"🛡 {_name(ctx.symbol)} 손절 주문이 없어 다시 걸었어요: {_px(pos['stop'])} "
            f"({WORKING_TYPE})"
        )
    if with_tp and pos.get("tp") and pos.get("tp_id") not in open_ids:
        n = pos.get("tp_n", 0) + 1
        cid = _pos_cid(pos, "t", n)
        try:
            place_stop(ctx.client, pos["side"], Decimal(str(pos["tp"])), cid, TP_KIND)
            pos.update(tp_id=cid, tp_n=n)
            ctx.say(f"🎯 {_name(ctx.symbol)} 익절 주문이 없어 다시 걸었어요: {_px(pos['tp'])}")
        except (Rejected, NetworkError) as e:
            # 응답만 잃었을 수 있다: 걸려 있으면 그대로 쓴다(지우면 그 익절 체결을 '수동'으로 봄)
            try:
                live_tp = isinstance(e, NetworkError) and stop_is_open(ctx.client, cid)
            except (Rejected, NetworkError):
                live_tp = False
            if live_tp:
                pos.update(tp_id=cid, tp_n=n)
            else:
                ctx.say(f"⚠️ 익절 재설정 실패({e}) → 익절 없이 손절·반대 신호로만 관리해요")
                pos.update(tp=None, tp_id=None)
        _save(DIR / "state.json", state)
    return True


def protect_close(ctx: Ctx, state: dict, why: str) -> None:
    """보호 못 하는 엔진 포지션을 시장가로 닫는다. 실패하면 크게 알린다."""
    pos = state["position"]
    close_side = "SELL" if pos["side"] == "long" else "BUY"
    # ID는 포지션·시도마다(실행 분 단위면 같은 실행의 두 번째 비상 청산이 앞 주문을 찾고 끝난다)
    # 시도마다 새 ID(앞 시도가 만료·부분 체결이면 같은 ID는 옛 주문을 찾아 아무것도 안 낸다)
    n = pos.get("x_n", 0) + 1
    pos["x_n"] = n
    _save(DIR / "state.json", state)
    cid = _pos_cid(pos, "x", n)
    try:
        amt = position_amt(ctx.client)
        if amt:
            market(ctx.client, close_side, _close_qty(amt), cid, reduce=True)
        if position_amt(ctx.client) != 0:
            raise RuntimeError("청산 뒤에도 포지션이 남음")
    except (Rejected, NetworkError, RuntimeError) as e:
        # 응답을 잃고 다시 보낸 reduceOnly는 이미 닫힌 뒤라 '포지션 없음'(-2022)으로 거부될 수 있다.
        # 포지션을 남기면 다음 실행이 '수동 청산'으로 보고 거짓 정지를 건다
        try:
            gone = position_amt(ctx.client) == 0
        except (Rejected, NetworkError):
            gone = False
        if not gone:
            ctx.say(
                f"🆘 손절 없는 포지션을 닫지 못했어요: {e}\n지금 바이낸스 앱에서 직접 확인하세요!"
            )
            ctx.event(event="unprotected", error=str(e))
            return
        ctx.event(event="protect_close_error_but_flat", error=str(e))
    # 닫힌 것은 확인됐다: 뒤 조회가 실패해도 포지션을 남기지 않는다
    # (남기면 다음 실행이 '수동 청산'으로 보고 거짓 정지를 건다)
    state["position"] = None
    _save(DIR / "state.json", state)
    try:
        cancel_engine_stops(ctx.client)
    except (Rejected, NetworkError) as e:
        ctx.event(event="cancel_after_protect_failed", error=str(e))  # 다음 대사가 취소한다
    try:
        px, pnl = _close_result(ctx.client, pos, close_side)
    except (Rejected, NetworkError):
        px = pnl = None
    # 같은 실행에서 새로 들어가면 이 청산 체결이 새 포지션 손익에 섞이지 않게(close_position과 같게)
    state["closed_through_ms"] = pos.get("closed_ms") or int(time.time() * 1000) - 2000
    _save(DIR / "state.json", state)
    _record_close(ctx, pos, px, pnl, why)


def _manual_levels(ctx: Ctx, side: str, entry: float) -> tuple[float, float | None]:
    tick = filters(ctx.client)["tick"]
    stop = float(stop_price(side, entry, tick, STOP_PCT))
    tp = float(tp_price(side, entry, tick, FIXED_TP_PCT / 100)) if FIXED_TP_PCT else None
    return stop, tp


def adopt_manual(ctx: Ctx, state: dict, row: dict, why: str) -> dict | None:
    """손으로 연(또는 바꾼) 포지션을 엔진 포지션으로 이어받는다. 손절·익절은 ensure_stop이 건다.
    틱을 못 읽으면 None(이번엔 못 함)."""
    side = "long" if row["amt"] > 0 else "short"
    entry = float(row["entry"])
    try:
        stop, tp = _manual_levels(ctx, side, entry)
    except (Rejected, NetworkError, OSError, ValueError, KeyError):
        return None
    minute = ctx.now_ms - ctx.now_ms % 60_000
    pos = {
        "symbol": ctx.symbol,
        "side": side,
        "qty": abs(row["amt"]),
        "entry": entry,
        "stop": stop,
        "stop_id": None,
        "stop_pct": STOP_PCT * 100,
        "tp": tp,
        "tp_id": None,
        "tp_pct": FIXED_TP_PCT or None,
        "ai_reason": "수동 포지션 이어받음",
        "bar": minute,
        "try": 9,  # 손절·익절 ID가 신호 진입(시도 1~3)과 겹치지 않게
        "entry_ms": ctx.now_ms,
        "opened_at": _now_iso(ctx.now),
        "mode": "live",
        "manual": True,
        "fill": "manual",
    }
    if state.get("closed_through_ms"):
        pos["after_ms"] = state.pop("closed_through_ms")
    state["position"] = pos
    state["last_symbol"] = ctx.symbol
    state.pop("pending", None)
    # 이미 지난 신호로 바로 뒤집지 않게: 마지막으로 마감한 봉까지 처리한 것으로 본다
    last_closed = ctx.now_ms - ctx.now_ms % BAR_MS - BAR_MS
    if last_closed > (last_bar_for(state, ctx.symbol) or 0):
        mark_bar(state, ctx.symbol, last_closed)
    _save(DIR / "state.json", state)
    ctx.event(
        event="adopt_manual", symbol=ctx.symbol, side=side, qty=pos["qty"], entry=entry, why=why
    )
    tp_txt = f", 익절 {_px(tp)}(+{FIXED_TP_PCT:g}%)" if tp else ""
    ctx.say(
        f"🤝 {why}: {_who(ctx.symbol, side)} {pos['qty']} {_name(ctx.symbol)} @ {_px(entry)}를 "
        f"이어받아 관리해요. 손절 {_px(stop)}(−{STOP_PCT * 100:g}%){tp_txt}"
    )
    return pos


def reconcile_live(ctx: Ctx, state: dict) -> bool:
    """거래소와 맞춘다. 엔진이 매매해도 되면 True(수동 개입이 보이면 False)."""
    c = ctx.client
    pending = state.get("pending")
    pos = state.get("position")
    if pending and pending.get("open") and not pos:
        # 진입 도중 끝난 실행: 남은 진입 지정가를 취소하고,
        # 체결된 만큼은 엔진 포지션으로 이어 받는다
        cancel_engine_entries(c)
        row = position_row(c)
        sign = 1 if pending["side"] == "long" else -1
        if row["amt"] and (row["amt"] > 0) == (sign > 0):
            # 손익 창은 진입을 시작한 실행 시각부터(open_position과 같게). 지금부터 세면 진입
            # 체결(수수료)이 빠진다. 같은 실행의 앞 청산은 closed_through_ms → after_ms가 뺀다
            try:
                at = datetime.strptime(pending["at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
                started = int(at.timestamp() * 1000) - 1000
            except (KeyError, TypeError, ValueError):
                started = int(time.time() * 1000)
            o = {
                "executedQty": str(abs(row["amt"])),
                "avgPrice": str(row["entry"]),
                "updateTime": started,
            }
            pos = _adopt(state, pending, o)
            ctx.event(event="adopt_pending", cid=pending["open"], amt=row["amt"])
            ctx.say(
                f"🔁 끝나지 못한 {_who(ctx.symbol, pos['side'])} 진입 {abs(row['amt'])} "
                f"{_name(ctx.symbol)}를 이어서 관리해요"
            )
    amt = position_amt(c)
    if not pos:
        state.pop("pending", None)
        if amt != 0 and MANUAL_ADOPT:
            cancel_engine_entries(c)
            cancel_engine_stops(c)  # 옛 엔진 손절·지킴 주문이 새 수량과 섞이지 않게
            state.get("guards", {}).pop(ctx.symbol, None)
            if adopt_manual(ctx, state, position_row(c), "수동 포지션 확인"):
                state.pop("foreign_key", None)
                return ensure_stop(ctx, state)
        if amt != 0:
            key = f"{amt}"
            if state.get("foreign_key") != key:
                ctx.say(
                    f"⚠️ 엔진이 열지 않은 {ctx.symbol} 포지션 {amt:+} {_name(ctx.symbol)}가 있어요. "
                    "건드리지 않고 신규 진입을 멈춰요."
                )
                state["foreign_key"] = key
            set_no_entry(f"엔진 밖 {ctx.symbol} 포지션 {amt:+}", ctx.send)
            return False
        state.pop("foreign_key", None)
        cancel_engine_stops(c)  # 포지션이 없는데 남은 엔진 손절은 나중 수동 포지션을 닫을 수 있다
        cancel_engine_entries(c)  # 남은 진입 지정가가 나중에 체결되면 엔진 밖 포지션이 된다
        return True
    sign = 1 if pos["side"] == "long" else -1
    close_side = "SELL" if sign > 0 else "BUY"
    if amt == 0:
        why = "manual"
        if pending and pending.get("close"):
            o = _query(c, "/fapi/v1/order", "origClientOrderId", pending["close"])
            if o and o.get("status") == "FILLED":
                why = "signal"
        for key, label in (("stop_id", "stop"), ("tp_id", "tp")):
            if why != "manual" or not pos.get(key):
                continue
            try:
                if _triggered(_query(c, "/fapi/v1/algoOrder", "clientAlgoId", pos[key])):
                    why = label
            except (Rejected, NetworkError):
                pass
        px, pnl = _close_result(c, pos, close_side)
        if why == "manual" and px:
            # 조건부 주문 기록이 사라졌거나 조회가 안 될 때:
            # 청산가가 손절·익절가 근처면 그쪽으로 본다
            if abs(px / pos["stop"] - 1) < 0.005:
                why = "stop"
            elif pos.get("tp") and abs(px / pos["tp"] - 1) < 0.005:
                why = "tp"
        state["position"] = None
        state.pop("pending", None)
        # 실행 시작 뒤에 체결된 손절·익절이 같은 실행의 새 포지션 손익에 섞이지 않게
        state["closed_through_ms"] = pos.get("closed_ms") or int(time.time() * 1000) - 2000
        cancel_engine_stops(c)  # 손절·익절 중 남은 쪽을 취소 (OCO처럼)
        _save(DIR / "state.json", state)
        _record_close(ctx, pos, px, pnl, why)
        if why == "manual" and MANUAL_ADOPT:
            # 사용자가 손으로 정리할 수 있다(2026-11-13): 기록만 하고 멈추지 않는다
            return True
        if why == "manual":
            set_no_entry("엔진 포지션이 손절·익절·신호 없이 닫힘(수동 또는 강제 청산)", ctx.send)
        return why != "manual"
    state.pop("pending", None)
    cancel_engine_entries(c)  # 남은 진입 지정가가 체결되면 수량이 어긋난다
    if abs(amt - sign * pos["qty"]) > 1e-9 and MANUAL_ADOPT:
        flipped = (amt > 0) != (sign > 0)
        if flipped:
            # 손으로 반대로 뒤집었다: 앞 포지션은 수동 청산으로 남기고 새 포지션을 이어받는다
            _record_close(ctx, pos, None, None, "manual")
        cancel_engine_stops(c)
        old = dict(pos)
        state["position"] = None
        if adopt_manual(
            ctx,
            state,
            position_row(c),
            "수동 반대 포지션" if flipped else "수동으로 수량이 바뀐 포지션",
        ):
            return ensure_stop(ctx, state)
        state["position"] = old  # 틱을 못 읽었다: 다음 실행에서 다시
        state["position"]["stop_id"] = None  # 취소했으니 ensure_stop이 다시 건다
        ensure_stop(ctx, state, with_tp=False)
        return False
    if abs(amt - sign * pos["qty"]) > 1e-9:
        key = f"{amt}/{pos['qty']}"
        if state.get("foreign_key") != key:
            ctx.say(
                f"⚠️ 포지션 수량이 엔진 기록과 달라요(거래소 {amt:+}, 엔진 {sign * pos['qty']:+}). "
                "수동 개입으로 보고 건드리지 않아요."
            )
            state["foreign_key"] = key
        set_no_entry("포지션 수량 불일치(수동 개입)", ctx.send)
        # closePosition 손절은 수량과 상관없이 전체를 지킨다. 없으면 다시 건다(닫지는 않는다)
        if (amt > 0) == (sign > 0):
            ensure_stop(ctx, state, with_tp=False)
        return False
    state.pop("foreign_key", None)
    _backfill_tp(ctx, state)
    return ensure_stop(ctx, state)


def _backfill_tp(ctx: Ctx, state: dict) -> None:
    """고정 익절이 생기기 전에 연 엔진 포지션에 익절가를 한 번 정해 둔다(ensure_stop이 건다).
    실패해 지워져도 다시 하지 않는다(tp_backfilled)."""
    pos = state["position"]
    if AI_EXITS or not FIXED_TP_PCT or pos.get("tp") or pos.get("tp_backfilled"):
        return
    try:
        tick = filters(ctx.client)["tick"]
    except (Rejected, NetworkError, OSError, ValueError, KeyError):
        return  # 다음 실행에서 다시
    tp = float(tp_price(pos["side"], float(pos["entry"]), tick, FIXED_TP_PCT / 100))
    pos.update(tp=tp, tp_id=None, tp_pct=FIXED_TP_PCT, tp_backfilled=True)
    _save(DIR / "state.json", state)
    ctx.say(
        f"🎯 {_name(ctx.symbol)} 보유 포지션에 기본 익절 +{FIXED_TP_PCT:g}%"
        f"(증거금 약 +{FIXED_TP_PCT * LEVERAGE:g}%)을 걸어요: {_px(tp)}"
    )


def _adopt(state: dict, pending: dict, order: dict) -> dict:
    entry = float(order.get("avgPrice") or 0) or pending["ref"]
    tick = pending.get("tick", "0.1")
    side = pending["side"]
    stop_pct = pending.get("stop_pct", STOP_PCT * 100)
    tp_pct = pending.get("tp_pct")
    pos = {
        "symbol": pos_symbol(pending),
        "side": side,
        "qty": float(order["executedQty"]),
        "entry": entry,
        "stop": float(stop_price(side, entry, tick, stop_pct / 100)),
        "stop_id": pending["stop"],
        "stop_pct": stop_pct,
        "tp": float(tp_price(side, entry, tick, tp_pct / 100)) if tp_pct else None,
        "tp_id": pending.get("tp"),
        "tp_pct": tp_pct,
        "ai_reason": pending.get("reason"),
        "bar": pending["bar"],
        "entry_ms": int(order.get("updateTime") or order.get("time") or 0),
        "opened_at": pending.get("at"),
        "mode": "live",
    }
    if pending.get("try", 1) > 1:
        pos["try"] = pending["try"]  # 다시 거는 손절·익절·비상 청산 ID도 앞 시도와 겹치지 않게
    if state.get("closed_through_ms"):
        pos["after_ms"] = state.pop("closed_through_ms")
    state["position"] = pos
    state["last_symbol"] = pos["symbol"]
    return pos


MARGIN_NO_CHANGE = -4046  # "No need to change margin type": 이미 교차


def leverage_problem(ctx: Ctx, state: dict | None = None) -> str:
    """그 종목을 교차·3배로 맞춘다(종목마다 따로다: ETH·SOXL·NVDA는 계좌에서 20배였다).
    문제가 없으면 빈 문자열, 진입하면 안 되면 이유. 조회·변경 실패는 예외로 올리지 않는다.
    약관 미동의(-4411)면 그 종목을 쉬게 하고(알림은 mark_unavailable이 한 번) 이유를 돌려준다."""
    sym = ctx.symbol
    try:
        cfg = _get(ctx.client, "/fapi/v1/symbolConfig", symbol=sym)
    except (Rejected, NetworkError) as e:
        return f"레버리지 설정 조회 실패({e})"
    row = next((x for x in cfg if x.get("symbol") == sym), None)
    if row is None:
        return "레버리지 설정이 안 보임"
    cur = int(row["leverage"])
    margin = str(row.get("marginType") or "CROSSED").upper()

    def tradfi(e: Exception) -> str:
        if isinstance(e, Rejected) and e.code == TRADFI_CODE and state is not None:
            mark_unavailable(ctx, state, e)
            return "TradFi 약관 미동의"
        return ""

    if margin not in ("CROSSED", "CROSS"):
        # 교차가 아니면(격리) 포지션·미체결이 하나도 없을 때만 바꾼다. 있으면 건드리지 않는다
        if not ctx.live:
            ctx.event(event="margin_dry", symbol=sym, current=margin)
        else:
            try:
                busy = position_amt(ctx.client) != 0 or bool(
                    _get(ctx.client, "/fapi/v1/openOrders", symbol=sym) or open_algos(ctx.client)
                )
            except (Rejected, NetworkError) as e:
                return f"마진 방식 확인 실패({e})"
            if busy:
                return f"마진 방식이 {margin}인데 포지션·미체결이 있어 교차로 못 바꿈"
            try:
                ctx.client.request(
                    "POST", "/fapi/v1/marginType", {"symbol": sym, "marginType": "CROSSED"}
                )
            except (Rejected, NetworkError) as e:
                if not (isinstance(e, Rejected) and e.code == MARGIN_NO_CHANGE):
                    return tradfi(e) or f"마진 방식 {margin}→교차 변경 실패({e})"
            _append(
                "orders.jsonl",
                {"at": _now_iso(ctx.now), "margin_type": {"symbol": sym, "from": margin}},
            )
            ctx.say(f"⚙️ {_name(sym)} 마진 방식 {margin} → 교차로 바꿨어요")
    if cur == LEVERAGE:
        return ""
    if not ctx.live:
        ctx.event(event="leverage_dry", symbol=sym, current=cur, want=LEVERAGE)
        return ""
    try:
        r = ctx.client.request("POST", "/fapi/v1/leverage", {"symbol": sym, "leverage": LEVERAGE})
    except (Rejected, NetworkError) as e:
        return tradfi(e) or f"레버리지 {cur}→{LEVERAGE}배 변경 실패({e})"
    _append(
        "orders.jsonl",
        {"at": _now_iso(ctx.now), "leverage": {"symbol": sym, "from": cur, "to": r}},
    )
    got = int((r or {}).get("leverage", 0) or 0)
    if got != LEVERAGE:
        return f"레버리지 변경 응답이 {got}배(원한 값 {LEVERAGE}배)"
    ctx.say(f"⚙️ {_name(sym)} 레버리지 {cur}배 → {LEVERAGE}배로 바꿨어요")
    return ""


def ensure_leverage(ctx: Ctx, state: dict | None = None) -> bool:
    """leverage_problem을 알림과 함께. 진입해도 되면 True."""
    why = leverage_problem(ctx, state)
    if why and why != "TradFi 약관 미동의":
        ctx.say(f"❌ {_name(ctx.symbol)} {why} → 진입하지 않아요")
    return not why


def unavailable(state: dict, symbol: str, now_ms: int) -> bool:
    """약관 미동의(-4411)로 쉬는 중인 종목인가."""
    rec = (state.get("unavailable") or {}).get(symbol)
    return bool(rec) and now_ms < int(rec.get("until_ms") or 0)


def mark_unavailable(ctx: Ctx, state: dict, err: Rejected) -> None:
    """TradFi 약관 미동의로 거부된 종목을 24시간 쉬게 한다. 알림은 종목마다 한 번만
    (24시간 뒤 다시 시도해 또 거부돼도 다시 알리지 않는다. 진입에 성공하면 기록을 지운다)."""
    sym = ctx.symbol
    book = state.setdefault("unavailable", {})
    old = book.get(sym) or {}
    told = old.get("notified") or any((r or {}).get("notified") for r in book.values())
    until = ctx.now_ms + UNAVAILABLE_MS
    book[sym] = {
        "at": _now_iso(ctx.now),
        "until_ms": until,
        "code": err.code,
        "msg": err.msg[:120],
        "notified": True,
    }
    # 약관은 계좌 단위라 다른 TradFi 종목도 같이 거부된다: 종목마다 알리지 않게 전체를 쉰다
    state["tradfi_until_ms"] = until
    _save(DIR / "state.json", state)
    ctx.event(event="symbol_unavailable", symbol=sym, code=err.code, msg=err.msg[:200])
    if not told:
        ctx.say(
            f"🚫 {_name(sym)} 주문이 거부됐어요: TradFi-Perps 약관에 동의하지 않은 계좌예요"
            f"({err.code}). TradFi 종목(주식·ETF·원자재)은 24시간 쉬고 다시 시도해요. "
            "코인 종목은 그대로 매매해요. 쓰려면 바이낸스 앱에서 약관에 동의해 주세요."
        )


def close_position(ctx: Ctx, state: dict, bar: int) -> bool:
    """반대 신호 청산. 1) 엔진 손절·익절 취소 2) 시장가 reduceOnly 3) 청산 확인. 성공하면 True."""
    pos = state["position"]
    close_side = "SELL" if pos["side"] == "long" else "BUY"
    if not ctx.live:
        px = ctx.market.price()
        sign = 1 if pos["side"] == "long" else -1
        pnl = sign * pos["qty"] * (px - pos["entry"]) - pos["qty"] * (pos["entry"] + px) * TAKER_FEE
        _append(
            "dry.jsonl",
            {
                "at": _now_iso(ctx.now),
                "action": "close",
                "symbol": ctx.symbol,
                "side": pos["side"],
                "qty": pos["qty"],
                "price": px,
                "type": "MARKET reduceOnly",
                "cid": _cid(bar, "c", ctx.symbol),
                "cancel": [x for x in (pos.get("stop_id"), pos.get("tp_id")) if x],
            },
        )
        state["position"] = None
        _record_close(ctx, pos, px, pnl, "signal")
        return True
    c = ctx.client
    cid = _cid(bar, "c", ctx.symbol)
    state["pending"] = {"close": cid, "symbol": ctx.symbol}
    _save(DIR / "state.json", state)
    try:
        cancel_engine_stops(c)
        amt = position_amt(c)
        if amt:
            market(c, close_side, _close_qty(amt), cid, reduce=True)
        if position_amt(c) != 0:
            raise RuntimeError("청산 주문 뒤에도 포지션이 남음")
    except (Rejected, NetworkError, RuntimeError) as e:
        ctx.say(
            f"❌ {_who(ctx.symbol, pos['side'])} 청산 실패: {e} → 손절을 다시 확인하고 "
            "진입하지 않아요"
        )
        try:
            reconcile_live(ctx, state)
        except (Rejected, NetworkError) as e2:
            ctx.say(f"🆘 청산 실패 뒤 상태 확인도 실패: {e2}. 바이낸스 앱에서 확인하세요!")
        _save(DIR / "state.json", state)
        return False
    # 청산은 확인됐다: 손익 조회가 실패해도 포지션은 닫힌 것으로 적는다
    state["position"] = None
    state.pop("pending", None)
    # 손익 조회가 실패하면 쓰는 값: 거래소와 시계가 조금 어긋나도 새 진입 체결을 빼지 않게 2초 앞
    state["closed_through_ms"] = int(time.time() * 1000) - 2000
    _save(DIR / "state.json", state)
    try:
        px, pnl = _close_result(c, pos, close_side)
    except (Rejected, NetworkError):
        px = pnl = None
    if pos.get("closed_ms"):
        state["closed_through_ms"] = pos["closed_ms"]
        _save(DIR / "state.json", state)
    _record_close(ctx, pos, px, pnl, "signal")
    return True


def _chase_text(side: str, price: float, sig: dict, gate: dict) -> str:
    gap = (price / sig["close"] - 1) * 100
    big = ", 큰 움직임" if gate.get("big") else ""
    return (
        f"현재가 {_px(price)}가 신호 봉 종가 {_px(sig['close'])}에서 {gap:+.2f}% "
        f"(추격 한도 {gate['pct']:.0f}%{big})"
    )


_KIND = {
    "COIN": "crypto",
    "EQUITY": "TradFi US stock/ETF",
    "KR_EQUITY": "TradFi Korea stock",
    "HK_EQUITY": "TradFi Hong Kong stock",
    "CN_EQUITY": "TradFi China stock",
    "COMMODITY": "TradFi commodity",
    "FX": "TradFi FX",
    "INDEX": "TradFi index",
    "PREMARKET": "pre-market (pre-IPO)",
}


def market_kind(flt: dict) -> str | None:
    """AI 자료용 종목 종류(코인/TradFi 주식·ETF/원자재 …). 모르면 None."""
    u = flt.get("underlying")
    if u in _KIND:
        return _KIND[u]
    if flt.get("contract") == TRADFI:
        return "TradFi"
    return "crypto" if flt.get("contract") == "PERPETUAL" else None


def ai_plan(ctx: Ctx, side: str, sig: dict, price: float, now_ms: int) -> dict:
    """Claude에게 손절·익절 폭을 묻는다. 무엇이 잘못돼도 기본값(3%, 익절 없음)."""
    if not AI_EXITS:
        tp_pct = FIXED_TP_PCT or None
        reason = (
            f"고정 규칙(손절 3%·익절 +{tp_pct:g}%, 증거금 약 +{tp_pct * LEVERAGE:g}%)"
            if tp_pct
            else "고정 규칙(손절 3%·익절 없음)"
        )
        used = {"stop_pct": STOP_PCT * 100, "tp_pct": tp_pct, "reason": reason}
        # 화면의 손절·익절 결정 목록이 옛 AI 답만 보이지 않게 고정 규칙도 같은 이벤트로 남긴다
        ctx.event(
            event="ai_exits",
            symbol=ctx.symbol,
            side=side,
            bar=sig.get("bar_txt"),
            used=used,
            fallback=None,
            fixed=True,
            secs=0.0,
        )
        return {"raw": "", "parsed": None, "used": used, "fallback": None, "secs": 0.0}
    try:
        funding = ctx.market.funding()
    except (OSError, ValueError, KeyError, AttributeError):
        funding = None
    mode_ = "live" if ctx.live else "dry"
    # ref_stats.json은 BTCUSDT 과거 1년 통계라 다른 종목에는 넘기지 않는다
    ref = _load(DIR / "ref_stats.json", None) if ctx.symbol == SYMBOL else None
    try:
        kind = market_kind(filters(ctx.client))
    except (NetworkError, Rejected, OSError, ValueError, KeyError):
        kind = None
    try:
        data = btc_ai.packet(
            side,
            sig,
            price,
            ctx.market.klines,
            funding,
            now_ms,
            _read_jsonl("closed.jsonl"),
            mode_,
            ref,
            symbol=ctx.symbol,
            winrate=SYMBOL_WINRATE.get(ctx.symbol),
            market=kind,
        )
    except (OSError, ValueError, KeyError, IndexError) as e:
        data = {
            "symbol": ctx.symbol,
            "backtest_winrate_pct": SYMBOL_WINRATE.get(ctx.symbol),
            "side": side,
            "signal_close": sig.get("close"),
            "price_now": price,
            "packet_error": str(e),
        }
    caller = AI_CALLER or (lambda p: btc_ai.call_claude(p, cwd=DIR))
    out = btc_ai.decide(data, caller)
    streak = _load(DIR / "state.json", {}).get("ai_fallbacks", 0)
    ctx.ai_streak = streak + 1 if out["fallback"] else 0
    if ctx.ai_streak >= AI_FALLBACK_ALERT and out["fallback"]:
        ctx.say(
            f"⚠️ AI 손절·익절이 {ctx.ai_streak}번 연속 기본값이에요"
            f"({out['fallback']}). 확인해 주세요"
        )
    ctx.event(
        event="ai_exits",
        symbol=ctx.symbol,
        side=side,
        bar=sig.get("bar_txt"),
        packet=data,
        ai_raw=out["raw"],
        parsed=out["parsed"],
        used=out["used"],
        fallback=out["fallback"],
        secs=out["secs"],
    )
    return out


# ---------------------------------------------------------------- 진입: 급변동 판단·지정가 사다리


def _atr15(rows15: list[list], now_ms: int, n: int = 14) -> float | None:
    closed = [r for r in rows15 if int(r[6]) < now_ms]
    if len(closed) < n + 2:
        return None
    h = [float(r[2]) for r in closed]
    lo = [float(r[3]) for r in closed]
    c = [float(r[4]) for r in closed]
    tr = [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))]
    a = sum(tr[:n]) / n
    for x in tr[n:]:
        a += (x - a) / n
    return a


def fast_market(
    market: Market, side: str, sig_close: float, price: float, rows15: list[list], now_ms: int
) -> str:
    """급변동이면 이유, 아니면 빈 문자열.
    1분봉을 못 받으면 신호 봉 기준만 본다."""
    cfg = entry_cfg()
    sign = 1 if side == "long" else -1
    moved = (price / sig_close - 1) * 100 * sign
    if moved >= cfg["fast_from_signal_pct"]:
        return f"신호 봉 종가에서 이미 {moved:+.2f}%"
    try:
        mins = market.minutes(20)
    except (NetworkError, OSError, ValueError, KeyError):
        return ""
    closed = [r for r in mins if int(r[6]) < now_ms]
    if closed:
        m1 = abs(price / float(closed[-1][1]) - 1) * 100
        if m1 >= cfg["fast_1m_pct"]:
            return f"1분 움직임 {m1:.2f}%"
    start = now_ms - now_ms % BAR_MS
    cur = [r for r in mins if int(r[0]) >= start]
    atr = _atr15(rows15, now_ms)
    if cur and atr:
        rng = max([float(r[2]) for r in cur] + [price]) - min([float(r[3]) for r in cur] + [price])
        if rng >= cfg["fast_range_atr"] * atr:
            return f"이번 15분봉 폭 {_px(rng)} ≥ ATR {_px(atr)}"
    return ""


class LiveVenue:
    """실제 바이낸스. 지정가는 post-only(GTX)."""

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.c = ctx.client

    def book(self) -> tuple[float, float]:
        return self.ctx.market.book()

    def place(self, cid: str, side: str, qty: Decimal, px: Decimal) -> dict:
        params = {
            "symbol": _sym(self.c),
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTX",
            "quantity": str(qty),
            "price": str(px),
            "newClientOrderId": cid,
            "newOrderRespType": "RESULT",
        }
        # 미체결 지정가는 같은 ID가 막아 주므로 조회 뒤 다시 보내도 된다
        r = _send(self.c, "POST", "/fapi/v1/order", params, "/fapi/v1/order", "origClientOrderId")
        _append("orders.jsonl", {"at": _now_iso(datetime.now(KST)), "req": params, "res": _slim(r)})
        return r

    def status(self, cid: str) -> dict | None:
        return _query(self.c, "/fapi/v1/order", "origClientOrderId", cid)

    def cancel(self, cid: str) -> dict:
        """취소한 뒤 반드시 다시 조회해 그 사이 체결된 수량까지 센다(취소·체결 경합)."""
        try:
            _delete(self.c, "/fapi/v1/order", {"symbol": _sym(self.c), "origClientOrderId": cid})
        except Rejected as e:
            if e.code not in NOT_FOUND:
                raise
        _append("orders.jsonl", {"at": _now_iso(datetime.now(KST)), "cancel": cid})
        for k in range(3):
            if k:
                _sleep(1)
            r = self.status(cid)
            if r and r.get("status") in TERMINAL:
                return r
        return r or {"status": "UNKNOWN", "executedQty": "0"}

    def market(self, cid: str, side: str, qty: Decimal) -> dict:
        return market(self.c, side, qty, cid, reduce=False)


class DryVenue:
    """드라이런: 호가·체결가로 거칠게 흉내 낸다. 내 매수 지정가는 매도 호가가 내려와 닿거나
    최근 체결가가 그 아래로 지나가면 다 체결된 것으로 본다(대기열 순서는 모른다). 숏은 반대."""

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.orders: dict[str, dict] = {}

    def book(self) -> tuple[float, float]:
        return self.ctx.market.book()

    def place(self, cid: str, side: str, qty: Decimal, px: Decimal) -> dict:
        bid, ask = self.book()
        p = float(px)
        if (side == "BUY" and p >= ask) or (side == "SELL" and p <= bid):
            return {"status": "EXPIRED", "executedQty": "0"}
        self.orders[cid] = {"side": side, "qty": qty, "px": p, "status": "NEW", "executedQty": "0"}
        return dict(self.orders[cid])

    def status(self, cid: str) -> dict | None:
        o = self.orders.get(cid)
        if not o or o["status"] != "NEW":
            return dict(o) if o else None
        bid, ask = self.book()
        last = self.ctx.market.price()
        buy = o["side"] == "BUY"
        if (buy and (ask <= o["px"] or last < o["px"])) or (
            not buy and (bid >= o["px"] or last > o["px"])
        ):
            o.update(status="FILLED", executedQty=str(o["qty"]), avgPrice=str(o["px"]))
        return dict(o)

    def cancel(self, cid: str) -> dict:
        r = self.status(cid) or {"status": "CANCELED", "executedQty": "0"}
        if r["status"] == "NEW":
            self.orders[cid]["status"] = r["status"] = "CANCELED"
        return r

    def market(self, cid: str, side: str, qty: Decimal) -> dict:
        bid, ask = self.book()
        return {
            "status": "FILLED",
            "executedQty": str(qty),
            "avgPrice": str(ask if side == "BUY" else bid),
        }


class LadderAbort(RuntimeError):
    """지정가 상태를 확신할 수 없어 사다리를 멈춘다(포지션 조회로 마무리)."""


def ladder(
    venue,
    side: str,
    qty: Decimal,
    sig: dict,
    gate: dict,
    flt: dict,
    base: str,
    price_now: Callable[[], float],
    deadline: float | None = None,
) -> dict:
    """post-only 지정가 사다리. 1회차는 최우선 호가(롱=매수 호가, 숏=매도 호가),
    회차마다 그때 호가에서 한 틱씩 더 시장 쪽으로. 새 가격이 걸어 둔 가격과 같으면(스프레드 1틱)
    취소하지 않고 그대로 둬 대기열 순서를 지킨다. 시간이 다 되면 남은 수량은 시장가.

    추격 한도를 넘으면 멈춘다(그때까지 체결된 수량은 그대로 둔다).
    남은 수량이 최소 수량·최소 주문 금액보다 작으면 그대로 받아들인다.
    deadline(_clock 기준)을 넘으면 시장가 없이 멈춘다(실행 전체 시간 한도).
    지정가 끝 상태를 확인하지 못하면 LadderAbort(더 내지 않는다).
    """
    cfg = entry_cfg()
    order_side = "BUY" if side == "long" else "SELL"
    tick = Decimal(flt["tick"]).normalize()
    filled, cost = Decimal(0), 0.0
    maker, taker = Decimal(0), Decimal(0)
    t0 = _clock()
    path, rounds, note = "", 0, ""
    steps: list[dict] = []
    cur: dict | None = None  # 걸려 있는 지정가 {"cid", "px"}

    def small(rest: Decimal, px: float) -> bool:
        return float(rest) < flt["min_qty"] or float(rest) * px < flt["min_notional"]

    def settle(cid: str, px: Decimal, r: dict | None) -> None:
        """끝난 지정가의 체결 수량을 센다. 끝 상태가 아니면 멈춘다(이중 체결 방지)."""
        nonlocal filled, cost, maker
        if not r or r.get("status") not in TERMINAL:
            raise LadderAbort(f"{cid} 상태 확인 안 됨({(r or {}).get('status')})")
        q = Decimal(str(r.get("executedQty") or 0))
        if q:
            filled += q
            maker += q
            cost += float(q) * (float(r.get("avgPrice") or 0) or float(px))
        steps.append({"cid": cid, "px": float(px), "filled": float(q), "status": r.get("status")})

    def late() -> bool:
        return deadline is not None and _clock() >= deadline

    for k in range(1, int(cfg["ladder_rounds"]) + 1):
        if late():
            path, note = "abandon", "실행 시간 한도"
            break
        bid, ask = venue.book()
        if side == "long":
            px = Decimal(str(bid)) + (k - 1) * tick
            if float(px) >= ask:  # 반대 호가에 닿으면 post-only가 거부된다: 한 틱 안쪽
                px = Decimal(str(ask)) - tick
        else:
            px = Decimal(str(ask)) - (k - 1) * tick
            if float(px) <= bid:
                px = Decimal(str(bid)) + tick
        rounds = k
        if cur and cur["px"] != px:
            settle(cur["cid"], cur["px"], venue.cancel(cur["cid"]))
            cur = None
        if not cur:
            rest = qty - filled
            if rest <= 0 or small(rest, float(px)):
                break
            if not chase_ok(side, float(px), sig["close"], gate["pct"]):
                path, note = "abandon", f"{k}회차 호가 {_px(float(px))}가 추격 한도 밖"
                break
            cid = f"{base}{k}"
            try:
                r = venue.place(cid, order_side, rest, px)
            except Rejected as e:
                if e.code not in GTX_REJECT:
                    raise
                steps.append({"cid": cid, "px": float(px), "result": "GTX 거부"})
                _sleep(1)
                continue
            if r.get("status") == "EXPIRED" and not float(r.get("executedQty") or 0):
                steps.append({"cid": cid, "px": float(px), "result": "GTX 만료"})
                _sleep(1)
                continue
            cur = {"cid": cid, "px": px, "r": r}
        wait_end = _clock() + cfg["ladder_wait_s"]
        if deadline is not None:
            wait_end = min(wait_end, deadline)
        r = cur["r"]
        while r.get("status") != "FILLED" and _clock() < wait_end:
            _sleep(cfg["ladder_poll_s"])
            r = venue.status(cur["cid"]) or r
        cur["r"] = r
        if r.get("status") == "FILLED":
            settle(cur["cid"], cur["px"], r)
            cur = None
            if filled >= qty:
                path = "limit"
                break
        if _clock() - t0 >= cfg["ladder_max_s"]:
            break
    if cur:
        settle(cur["cid"], cur["px"], venue.cancel(cur["cid"]))
        cur = None
    rest = qty - filled
    if filled >= qty:
        path = "limit"
    elif path != "abandon" and rest > 0:
        now_px = price_now()
        if small(rest, now_px):
            note = note or f"남은 {rest}개는 최소 주문보다 작아 그대로 둠"
            path = "limit" if filled else "none"
        elif not chase_ok(side, now_px, sig["close"], gate["pct"]):
            path, note = "abandon", f"시간이 다 됐을 때 가격 {_px(now_px)}가 추격 한도 밖"
        else:
            r = venue.market(f"{base}m", order_side, rest)
            q = Decimal(str(r.get("executedQty") or 0))
            filled += q
            taker += q
            cost += float(q) * (float(r.get("avgPrice") or 0) or now_px)
            path = "market_fallback"
    avg = cost / float(filled) if filled else None
    fee = (float(maker) * MAKER_FEE + float(taker) * TAKER_FEE) * (avg or 0)
    return {
        "qty": float(filled),
        "avg": avg,
        "path": path or "none",
        "rounds": rounds,
        "maker_qty": float(maker),
        "taker_qty": float(taker),
        "fee": fee,
        "note": note,
        "secs": round(_clock() - t0, 1),
        "steps": steps,
    }


def fill_text(f: dict) -> str:
    if f["path"] == "fast":
        return f"시장가(급변동: {f['note']})"
    if f["path"] == "limit":
        return f"지정가 체결({f['rounds']}회차, 메이커)"
    if f["path"] == "market_fallback":
        return (
            f"지정가 {f['maker_qty']} + 시장가 {f['taker_qty']}개(지정가 미체결)"
            if f["maker_qty"]
            else "시장가(지정가 미체결)"
        )
    if f["path"] == "abandon":
        return f"지정가 일부만 체결({f['note']})"
    return f["path"]


def cancel_engine_entries(client) -> list[str]:
    """남아 있는 엔진 진입 지정가(일반 주문, 접두어 mnv-)를 취소한다."""
    done = []
    sym = _sym(client)
    for o in _get(client, "/fapi/v1/openOrders", symbol=sym):
        cid = str(o.get("clientOrderId", ""))
        if not cid.startswith(PREFIX):
            continue
        try:
            _delete(client, "/fapi/v1/order", {"symbol": sym, "origClientOrderId": cid})
        except Rejected as e:
            if e.code not in NOT_FOUND:
                raise
        done.append(cid)
        _append("orders.jsonl", {"at": _now_iso(datetime.now(KST)), "cancel": cid})
    return done


def position_row(client) -> dict:
    sym = _sym(client)
    rows = _get(client, "/fapi/v2/positionRisk", symbol=sym)
    row = next((x for x in rows if x["symbol"] == sym), None) or {"positionAmt": "0"}
    return {"amt": float(row["positionAmt"]), "entry": float(row.get("entryPrice") or 0)}


def _entry_filled(ctx: Ctx, oid: str, since_ms: int) -> bool:
    """이 신호의 진입 주문(ID가 oid로 시작) 중 조금이라도 체결된 게 있나. 못 읽으면 True(다시 내지
    않는 쪽으로)."""
    try:
        rows = _get(ctx.client, "/fapi/v1/allOrders", symbol=ctx.symbol, startTime=since_ms)
    except (Rejected, NetworkError):
        return True
    return any(
        str(o.get("clientOrderId", "")).startswith(oid) and float(o.get("executedQty") or 0) > 0
        for o in rows
    )


def _margin_diag(ctx: Ctx, qty: Decimal) -> dict:
    """증거금 부족(-2019) 원인을 보려고 호가·마크 가격·가용 잔고를 남긴다(실패해도 그냥 넘어감)."""
    out: dict = {"qty": float(qty), "leverage": LEVERAGE}
    try:
        out["bid"], out["ask"] = LiveVenue(ctx).book()
    except Exception:  # noqa: BLE001 — 진단만
        pass
    try:
        url = f"{BASE}/fapi/v1/premiumIndex?symbol={ctx.symbol}"
        out["mark"] = float(_public(url)["markPrice"])
    except Exception:  # noqa: BLE001
        pass
    try:
        out["available"] = float(_get(ctx.client, "/fapi/v2/account")["availableBalance"])
    except Exception:  # noqa: BLE001
        pass
    return out


def _settle_entry(ctx: Ctx, want: float | None, tries: int = 5) -> dict | None:
    """진입 마무리: 엔진 진입 지정가를 취소하고 포지션을 읽는다. 센 수량(want)만큼 보일 때까지
    0.5초 간격으로 다시 읽는다. 끝내 못 읽으면 None(0과 구별한다)."""
    row = None
    for k in range(tries):
        if k:
            _sleep(0.5)
        try:
            cancel_engine_entries(ctx.client)
            row = position_row(ctx.client)
        except (Rejected, NetworkError):
            continue
        if not want or abs(row["amt"]) >= want - 1e-9:
            return row
    return row


def _enter(
    ctx: Ctx, venue, side: str, qty: Decimal, sig: dict, gate: dict, flt: dict, base: str
) -> dict:
    """급변동이면 시장가 한 번, 아니면 지정가 사다리."""
    price = ctx.market.price()
    fast = fast_market(ctx.market, side, sig["close"], price, ctx.rows15, ctx.now_ms)
    if fast:
        r = venue.market(f"{base}f", "BUY" if side == "long" else "SELL", qty)
        q = float(r.get("executedQty") or 0)
        avg = float(r.get("avgPrice") or 0) or price
        return {
            "qty": q,
            "avg": avg if q else None,
            "path": "fast",
            "rounds": 0,
            "maker_qty": 0.0,
            "taker_qty": q,
            "fee": q * avg * TAKER_FEE,
            "note": fast,
            "secs": 0.0,
            "steps": [],
        }
    return ladder(venue, side, qty, sig, gate, flt, base, ctx.market.price, ctx.deadline)


def open_position(ctx: Ctx, state: dict, sig: dict, gate: dict, flt: dict, plan: dict) -> None:
    """4) 진입(급변동이면 시장가, 아니면 post-only 지정가 사다리) 5) 평균 체결가
    6) STOP_MARKET(+TAKE_PROFIT_MARKET) 7) 걸렸는지 확인. 실패하면 닫거나 알린다.
    AI 답을 기다린 뒤라 추격 한도를 새 가격으로 다시 본다(사다리도 회차마다 본다)."""
    side = sig["dir"]
    sym = ctx.symbol
    who = _who(sym, side)
    used = plan["used"]
    price = ctx.market.price()
    if not chase_ok(side, price, sig["close"], gate["pct"]):
        ctx.say(f"⏭ {who} 진입 건너뜀(AI 응답 뒤 다시 봄): {_chase_text(side, price, sig, gate)}")
        ctx.event(
            event="chase_skip",
            symbol=sym,
            side=side,
            price=price,
            ref=sig["close"],
            pct=gate["pct"],
        )
        return
    acct = _get(ctx.client, "/fapi/v2/account")
    if ctx.live and not acct.get("canTrade", False):
        ctx.say("❌ 계좌가 거래 불가(canTrade=false)라 진입하지 않아요")
        return
    if ctx.live and not state.get("position"):
        # 앞 포지션의 손절·익절(closePosition)이 남아 있으면 새 포지션을 닫아 버린다
        try:
            cancel_engine_stops(ctx.client)
        except (Rejected, NetworkError) as e:
            ctx.say(f"⏭ {who} 진입 건너뜀: 남은 엔진 손절을 취소하지 못함({e})")
            return
    avail = float(acct["availableBalance"])
    qty, why = size_qty(avail, price, margin_frac(), flt)
    if not qty:
        ctx.say(f"⏭ {who} 진입 불가: {why} (가용 {avail:.2f} USDT)")
        return
    oid, sid = _cid(sig["bar"], "o", sym), _cid(sig["bar"], "s", sym)
    tid = _cid(sig["bar"], "t", sym) if used["tp_pct"] else None
    icon = "🟢" if side == "long" else "🔴"
    ai_line = ("🤖 " if plan.get("parsed") else "⚙️ ") + used["reason"]

    def exits_text(stop: float, tp: float | None) -> str:
        t = f"손절 {_px(stop)} (−{used['stop_pct']:g}%)"
        t += f" · 익절 {_px(tp)} (+{used['tp_pct']:g}%)" if tp else " · 익절 없음(반대 신호까지)"
        return t

    def fill_line(f: dict) -> str:
        return f"{fill_text(f)} · 수수료 약 {f['fee']:.2f} USDT"

    if not ctx.live:
        fill = _enter(ctx, DryVenue(ctx), side, qty, sig, gate, flt, oid)
        _append(
            "dry.jsonl",
            {
                "at": _now_iso(ctx.now),
                "action": "open",
                "symbol": sym,
                "side": side,
                "want": float(qty),
                "qty": fill["qty"],
                "price": fill["avg"],
                "path": fill["path"],
                "rounds": fill["rounds"],
                "steps": fill["steps"],
                "note": fill["note"],
                "fee": fill["fee"],
                "cid": oid,
                "stop_pct": used["stop_pct"],
                "tp_pct": used["tp_pct"],
                "ai_reason": used["reason"],
                "ai_fallback": plan["fallback"],
                "available": avail,
                "margin_frac": margin_frac(),
                "chase_pct": gate["pct"],
                "signal_bar": sig.get("bar_txt"),
            },
        )
        if not fill["qty"]:
            ctx.say(f"⏭ {who} 진입 안 됨: {fill_text(fill)}")
            return
        entry = fill["avg"]
        stop = float(stop_price(side, entry, flt["tick"], used["stop_pct"] / 100))
        tp = float(tp_price(side, entry, flt["tick"], used["tp_pct"] / 100)) if tid else None
        state["position"] = {
            "symbol": sym,
            "side": side,
            "qty": fill["qty"],
            "entry": entry,
            "stop": stop,
            "stop_id": sid,
            "stop_pct": used["stop_pct"],
            "tp": tp,
            "tp_id": tid,
            "tp_pct": used["tp_pct"],
            "ai_reason": used["reason"],
            "bar": sig["bar"],
            "fill": fill["path"],
            "entry_ms": int(ctx.now.timestamp() * 1000),
            "opened_at": _now_iso(ctx.now),
            "mode": "dry",
        }
        state["last_symbol"] = sym
        ctx.say(
            f"{icon} {who} 진입 예정: {fill['qty']} {_name(sym)} @ {_px(entry)} "
            f"(증거금 약 {fill['qty'] * entry / LEVERAGE:,.1f} USDT, {LEVERAGE}배)\n"
            f"{fill_line(fill)}\n{exits_text(stop, tp)}\n{ai_line}\n"
            f"신호 봉 {sig.get('bar_txt')} 종가 {_px(sig['close'])}"
        )
        return
    c = ctx.client
    if foreign_orders(c):
        ctx.say(
            f"⏭ {who} 진입 안 함: 엔진이 내지 않은 {sym} 미체결 주문이 있어요"
            "(단방향이라 섞일 수 있음)"
        )
        return
    if position_amt(c) != 0:
        ctx.say(f"⏭ {who} 진입 안 함: 진입 직전에 포지션이 보여요")
        return
    if not ensure_leverage(ctx, state):
        return
    # 같은 신호를 다시 시도하면 진입·손절·익절 ID를 새로 쓴다. 같은 ID면 _send가 앞 시도의 옛
    # 주문(취소·만료·체결)을 찾아 돌려주고 새 주문을 내지 않는다: 옛 손절이 이미 체결됐으면
    # '손절 실패'로 보고 새 포지션을 닫는다. 두 번째부터 o<n>-·s<n>-·t<n>-
    # (첫 시도 ID·_pos_cid의 s1… 과 겹치지 않음)
    tries = state.get("entry_tries") or {}
    same = tries.get("bar") == sig["bar"] and pos_symbol(tries) == sym
    n = tries.get("n", 0) + 1 if same else 1
    state["entry_tries"] = {"bar": sig["bar"], "n": n, "symbol": sym}
    if n > 1:
        oid, sid = _cid(sig["bar"], f"o{n}-", sym), _cid(sig["bar"], f"s{n}-", sym)
        tid = _cid(sig["bar"], f"t{n}-", sym) if tid else None
    state["pending"] = {
        "symbol": sym,
        "open": oid,
        "stop": sid,
        "tp": tid,
        "try": n,
        "side": side,
        "bar": sig["bar"],
        "ref": price,
        "tick": flt["tick"],
        "at": _now_iso(ctx.now),
        "stop_pct": used["stop_pct"],
        "tp_pct": used["tp_pct"],
        "reason": used["reason"],
    }
    _save(DIR / "state.json", state)
    fill, err = None, None
    base = oid
    t_try = _ms_now() - 60_000
    for attempt in range(len(ENTRY_BACKOFF_S) + 1):
        try:
            fill = _enter(ctx, LiveVenue(ctx), side, qty, sig, gate, flt, base)
            err = None
            break
        except Exception as e:  # noqa: BLE001 — 무엇이 깨져도 아래에서 취소·포지션 확인·손절로 마무리
            err = e
            ev = {"event": "enter_error", "error": f"{type(e).__name__}: {e}", "try": attempt + 1}
            if isinstance(e, Rejected) and e.code == MARGIN_CODE:
                ev["diag"] = _margin_diag(ctx, qty)
            ctx.event(**ev)
        retryable = isinstance(err, Rejected) and err.code in ENTRY_RETRY_CODES
        if not retryable or attempt >= len(ENTRY_BACKOFF_S):
            break
        wait = ENTRY_BACKOFF_S[attempt]
        if _clock() + wait > ctx.deadline:
            break
        # 다시 내기 전에: 남은 진입 주문 취소, 포지션이 정말 0인지 확인. 조금이라도 들어갔거나
        # 확인을 못 하면 다시 내지 않고 아래 대사(손절 걸기)로 넘긴다
        row0 = _settle_entry(ctx, None)
        if row0 is None or row0["amt"] or _entry_filled(ctx, oid, t_try):
            break
        _sleep(wait)
        if isinstance(err, Rejected) and err.code == MARGIN_CODE:
            qty = _round(float(qty) * ENTRY_SHRINK, flt["step"])
            if float(qty) < flt["min_qty"] or float(qty) * price < flt["min_notional"]:
                break
        try:
            now_px = ctx.market.price()
        except Exception:  # noqa: BLE001 — 가격을 못 보면 다시 내지 않는다
            break
        if not chase_ok(side, now_px, sig["close"], gate["pct"]):
            why_px = _chase_text(side, now_px, sig, gate)
            ctx.say(f"⏭ {who} 다시 시도 안 함(쉬는 사이 추격 한도 밖): {why_px}")
            break
        base = f"{oid}r{attempt + 1}-"
        ctx.say(f"🔁 {who} 진입 실패({type(err).__name__}: {err}) → {wait:g}초 뒤 다시 시도")
    # 남은 진입 지정가를 취소하고 거래소 포지션을 진실로 삼는다(체결 직후 늦게 보일 수 있어 몇 번)
    row = _settle_entry(ctx, fill["qty"] if fill else None)
    why = f"{type(err).__name__}: {err}" if err else ""
    if row is None:
        ctx.say(
            f"🆘 {who} 진입 뒤 포지션을 확인하지 못했어요({why or '조회 실패'}). "
            "다음 실행에서 대사하고 손절을 걸어요. 바이낸스 앱도 확인하세요!"
        )
        return  # pending을 남긴다
    sign = 1 if side == "long" else -1
    if row["amt"] and (row["amt"] > 0) != (sign > 0):
        ctx.say(
            f"🆘 {_name(sym)} 진입 뒤 포지션 방향이 이상해요({row['amt']:+}). "
            "바이낸스 앱에서 확인하세요!"
        )
        set_no_entry("진입 뒤 포지션 방향 불일치", ctx.send)
        return
    if not row["amt"]:
        if fill and fill["qty"] > 0:
            ctx.say("🆘 체결을 셌는데 포지션이 아직 0으로 보여요 → 다음 실행에서 다시 대사해요")
            return  # pending을 남긴다
        if isinstance(err, Rejected) and err.code == TRADFI_CODE and not fill:
            # 약관 미동의(-4411): 주문이 들어가지 않았다(포지션 0 확인). 이 종목만 쉬고 넘어간다
            state.pop("pending", None)
            mark_unavailable(ctx, state, err)
            return
        if err:
            # 응답을 잃은 시장가 진입은 체결됐어도 포지션이 늦게 보일 수 있다. pending을 버리면
            # 다음 실행이 그 포지션을 '엔진 밖'으로 보고 손절 없이 정지한다. 남겨 두면 다음 대사가
            # 이어 받는다(그때도 0이면 pending을 버리고 같은 신호를 새 ID로 다시 본다)
            ctx.say(f"⚠️ {who} 진입 중 오류({why}), 포지션은 아직 0 → 다음 실행에서 다시 대사해요")
            return  # pending을 남긴다
        state.pop("pending", None)
        ctx.say(f"⏭ {who} 진입 안 됨: {fill_text(fill) if fill else why}")
        return
    if err:
        ctx.say(
            f"⚠️ 진입 중 오류({why}) → 체결된 {abs(row['amt'])} {_name(sym)}를 이어 받아 "
            "손절을 걸어요"
        )
    real = abs(row["amt"])
    entry = fill["avg"] if fill and fill["avg"] and abs(real - fill["qty"]) < 1e-9 else row["entry"]
    if fill and abs(real - fill["qty"]) > 1e-9:
        # 취소·체결 경합 등으로 셈이 달라지면 거래소 포지션을 믿는다
        ctx.event(event="fill_mismatch", counted=fill["qty"], position=real, entry=row["entry"])
    o = {"executedQty": str(real), "avgPrice": str(entry), "updateTime": int(time.time() * 1000)}
    pos = _adopt(state, state["pending"], o)
    pos["entry_ms"] = int(ctx.now.timestamp() * 1000) - 1000
    pos["opened_at"] = _now_iso(ctx.now)
    pos["fill"] = fill["path"] if fill else "recovered"
    state.pop("pending", None)
    (state.get("unavailable") or {}).pop(sym, None)  # 진입됐으니 약관 문제는 풀렸다
    if flt.get("contract") == TRADFI:
        state.pop("tradfi_until_ms", None)
    # 포지션과 함께 신호도 처리한 것으로 저장한다: 이 뒤에 죽으면 다음 실행이 같은 신호로 다시
    # 들어가지 않는다(run() 끝에서도 같은 값을 쓴다)
    mark_bar(state, sym, sig["bar"])
    _save(DIR / "state.json", state)
    done = False
    try:
        try:
            s = place_stop(c, side, Decimal(str(pos["stop"])), sid)
            ok = stop_placed(c, s, sid)
            err = "" if ok else f"상태 {s.get('algoStatus')}"
        except (Rejected, NetworkError) as e:
            ok, err = False, str(e)
        if not ok:
            ctx.say(f"❌ 손절 주문 실패({err}) → 방금 연 {who}을 시장가로 닫아요")
            protect_close(ctx, state, "protect")
            done = True
            return
        tp_note = ""
        if pos.get("tp"):
            try:
                place_stop(c, side, Decimal(str(pos["tp"])), tid, TP_KIND)
            except (Rejected, NetworkError) as e:
                # 응답만 잃었을 수 있다: 걸려 있으면 그대로 둔다
                try:
                    tp_live = isinstance(e, NetworkError) and stop_is_open(c, tid)
                except (Rejected, NetworkError):
                    tp_live = False  # 확인도 못 하면 없는 것으로 보고 다음 대사가 다시 건다
                if not tp_live:
                    tp_note = f"\n⚠️ 익절 주문 실패({e}) → 익절 없이 손절·반대 신호로 관리"
                    pos.update(tp=None, tp_id=None)
                _save(DIR / "state.json", state)
        done = True
    finally:
        if not done and state.get("position"):
            # 예상 못 한 오류로 중간에 멈췄다: 손절만은 확인하고 끝낸다
            try:
                ensure_stop(ctx, state, with_tp=False)
            except Exception as e2:  # noqa: BLE001 — 원래 예외를 덮지 않는다
                ctx.say(f"🆘 진입 뒤 손절 확인 실패: {e2}. 바이낸스 앱에서 확인하세요!")
    how = fill_line(fill) if fill else "응답을 놓쳐 포지션으로 확인"
    ctx.say(
        f"{icon} {who} 진입: {pos['qty']} {_name(sym)} @ {_px(pos['entry'])} ({LEVERAGE}배)\n"
        f"{how}\n"
        f"{exits_text(pos['stop'], pos.get('tp'))} 걸어 둠 ({WORKING_TYPE})\n{ai_line}\n"
        f"신호 봉 {sig.get('bar_txt')}{tp_note}"
    )


# ---------------------------------------------------------------- 여러 종목: 신호·선택


def held_symbol(state: dict) -> str:
    """대사할 종목: 보유 포지션 → 끝나지 않은 주문 → 마지막으로 든 종목 → BTCUSDT."""
    for key in ("position", "pending"):
        if state.get(key):
            return pos_symbol(state[key])
    return state.get("last_symbol") or SYMBOL


def last_bar_for(state: dict, symbol: str) -> int | None:
    """종목별 마지막 처리 봉. 처음 보는 종목은 예전 하나뿐이던 last_bar에서 시작한다."""
    bars = state.setdefault("last_bars", {})
    if symbol not in bars and state.get("last_bar") is not None:
        bars[symbol] = state["last_bar"]
    return bars.get(symbol)


def mark_bar(state: dict, symbol: str, bar: int) -> None:
    """그 종목 신호를 처리한 것으로 적는다. last_bar(화면·옛 상태용)는 가장 늦은 봉."""
    state.setdefault("last_bars", {})[symbol] = bar
    if bar > (state.get("last_bar") or 0):
        state["last_bar"] = bar


def fetch_rows(
    ctx: Ctx, extra: tuple[str, ...] = (), budget_s: float | None = None
) -> tuple[dict[str, list[list]], dict[str, str]]:
    """범위 종목(+ 들고 있는 종목) 15분봉을 동시에(FETCH_WORKERS개씩) 받는다. 실패하거나
    시간 안(budget_s)에 못 받은 종목은 이번 실행에서 신호만 빠진다(보유 관리는 계속)."""
    want = list(dict.fromkeys(list(SYMBOLS) + [s for s in extra if s]))
    markets = {s: market_for(ctx.market, s) for s in want}
    todo = [s for s in want if markets[s] is not None]
    rows: dict[str, list[list]] = {}
    errs: dict[str, str] = {}
    pool = ThreadPoolExecutor(max_workers=max(min(len(todo), FETCH_WORKERS), 1))
    try:
        futs = {s: pool.submit(markets[s].klines, INTERVAL) for s in todo}
        wait_futures(list(futs.values()), timeout=FETCH_BUDGET_S if budget_s is None else budget_s)
        for s, f in futs.items():
            if not f.done():
                f.cancel()
                errs[s] = "시간 안에 못 받음"
                continue
            try:
                rows[s] = f.result()
            except Exception as e:  # noqa: BLE001 — 한 종목 시세 실패가 보유 관리를 막으면 안 된다
                errs[s] = f"{type(e).__name__}: {e}"[:200]
    finally:
        # 늦은 요청을 기다리지 않는다(요청마다 시간 한도가 있어 곧 끝난다)
        pool.shutdown(wait=False, cancel_futures=True)
    return rows, errs


def prepare(ctx: Ctx, state: dict, sym: str, sig: dict, halts: list[str], t_start: float) -> dict:
    """진입 후보 점검(예전 한 종목 흐름과 같은 순서): 정지 → 복구 나이 → 시간 → 거래 규칙 → 추격.
    skip이 비어 있으면 들어갈 수 있다. 시세·규칙 조회 실패는 그 종목만 건너뛴다."""
    c = ctx.at(sym)
    side = sig["dir"]
    out: dict = {"sym": sym, "sig": sig, "side": side, "ctx": c, "skip": ""}
    if halts:
        out["skip"] = f"진입 정지({', '.join(halts)})"
    elif sig.get("age", 0) > RECOVER_ENTRY_BARS:
        hours = sig["age"] * BAR_MS / 3_600_000
        out["skip"] = (
            f"복구: 신호가 {hours:.1f}시간 전이라 새로 들어가지 않음"
            "(반대 포지션 정리만, 다음 신호 대기)"
        )
    elif unavailable(state, sym, ctx.now_ms):
        out["skip"] = "TradFi 약관 미동의로 쉬는 중"
    elif sym not in SYMBOLS:
        out["skip"] = "종목 범위(거래량 상위)에서 빠져 새로 들어가지 않음(보유분 정리만)"
    elif not tradable(sym):
        out["skip"] = "승률 45% 미만이거나 표본이 적어 새로 들어가지 않음(보유분 정리만)"
    elif _clock() - t_start > MAX_RUN_SECS:
        out["skip"] = f"이번 실행이 {MAX_RUN_SECS}초를 넘겨 새 진입은 건너뜀(통신 지연)"
    elif c.market is None:
        out["skip"] = "시세 없음"
    else:
        try:
            flt = filters(c.client)
            # 거래 상태·장 시간은 시세(4시간·일봉)를 받기 전에 본다(무게 아끼기)
            closed = "" if flt["status"] != "TRADING" else market_closed(ctx, state, flt)
            if flt["status"] != "TRADING" or closed:
                gate, price0 = {}, None
            else:
                gate = chase_gate(c.market, c.rows15, ctx.now_ms)
                price0 = c.market.price()
        except (NetworkError, Rejected, OSError, ValueError, KeyError) as e:
            out["skip"] = f"시세·거래 규칙 조회 실패({type(e).__name__})"
            return out
        out.update(flt=flt, gate=gate, price=price0)
        if flt["status"] != "TRADING":
            out["skip"] = f"{sym} 상태 {flt['status']}"
        elif closed:
            out["skip"], out["quiet"] = closed, True  # 기록만(장 쉬는 시간은 자주 겹친다)
            ctx.event(event="session_skip", symbol=sym, side=side, why=closed)
        elif not chase_ok(side, price0, sig["close"], gate["pct"]):
            out["skip"] = f"진입 안 함: {_chase_text(side, price0, sig, gate)}"
            ctx.event(
                event="chase_skip",
                symbol=sym,
                side=side,
                price=price0,
                ref=sig["close"],
                pct=gate["pct"],
            )
        elif ctx.live:
            out["skip"] = live_entry_problem(c, state)
            out["quiet"] = out["skip"] == "TradFi 약관 미동의"  # mark_unavailable이 이미 알림
    return out


def market_closed(ctx: Ctx, state: dict, flt: dict) -> str:
    """TradFi 종목이면 진입을 막을 이유: 약관 미동의로 TradFi 전체가 쉬는 중이거나, 그 시장이
    지금 NO_TRADING 시간. 종목 종류·시간표를 모르면 막지 않는다(빈 문자열)."""
    if flt.get("contract") != TRADFI:
        return ""
    if int(state.get("tradfi_until_ms") or 0) > ctx.now_ms:
        return "TradFi 약관 미동의로 쉬는 중"
    key = flt.get("underlying")
    if key not in SESSION_KEYS:
        return ""
    sched = ctx.shared.get("schedule")
    if sched is None:
        try:
            sched = (
                _public(f"{BASE}/fapi/v1/tradingSchedule", timeout=10).get("marketSchedules") or {}
            )
        except (OSError, ValueError, AttributeError) as e:
            ctx.event(event="schedule_error", error=f"{type(e).__name__}: {e}"[:200])
            sched = {}
        ctx.shared["schedule"] = sched
    for ses in (sched.get(key) or {}).get("sessions") or []:
        try:
            if int(ses["startTime"]) <= ctx.now_ms < int(ses["endTime"]):
                if ses.get("type") == "NO_TRADING":
                    return f"{key} 시장이 쉬는 시간(NO_TRADING)이라 새로 들어가지 않음"
                return ""
        except (KeyError, TypeError, ValueError):
            continue
    return ""


def live_entry_problem(c: Ctx, state: dict) -> str:
    """실전 진입 전 거래소 점검(들고 있는 것을 닫기 전에 한다: 갈아탈 종목에 못 들어가면
    닫지 않게). 계좌 거래 가능, 그 종목의 엔진 밖 미체결·포지션, 교차·3배. 문제면 이유."""
    sym = c.symbol
    try:
        acct = c.shared.get("acct") or _get(c.client, "/fapi/v2/account")
        c.shared["acct"] = acct
        if not acct.get("canTrade", False):
            return "계좌가 거래 불가(canTrade=false)"
        if sym != held_symbol(state) or not state.get("position"):
            if foreign_orders(c.client):
                return f"엔진이 내지 않은 {sym} 미체결 주문이 있음(단방향이라 섞일 수 있음)"
            if position_amt(c.client) != 0:
                return f"{sym}에 이미 포지션이 보임"
    except (Rejected, NetworkError) as e:
        return f"거래소 점검 실패({e})"
    return leverage_problem(c, state)


def _candidate_packet(cand: dict) -> dict:
    c, sig = cand["ctx"], cand["sig"]
    data = {
        "symbol": cand["sym"],
        "side": cand["side"],
        "signal_bar_kst": sig.get("bar_txt"),
        "signal_close": sig.get("close"),
        "price_now": cand.get("price"),
        "chase_limit_pct": (cand.get("gate") or {}).get("pct"),
        "backtest_winrate_pct": SYMBOL_WINRATE.get(cand["sym"]),
        "market_kind": market_kind(cand.get("flt") or {}),
        "scr_at_signal": sig.get("at_signal"),
    }
    if cand.get("price") and sig.get("close"):
        data["gap_from_signal_pct"] = round((cand["price"] / sig["close"] - 1) * 100, 3)
    try:
        data.update(btc_ai.brief_frames(c.market.klines, cand["price"], c.now_ms))
    except (OSError, ValueError, KeyError, IndexError, TypeError, NetworkError) as e:
        data["frames_error"] = type(e).__name__
    return data


def _held_packet(ctx: Ctx, pos: dict, opp: bool) -> dict:
    sym = pos_symbol(pos)
    c = ctx.at(sym)
    try:
        price = c.market.price() if c.market is not None else None
    except (NetworkError, OSError, ValueError, KeyError):
        price = None
    sign = 1 if pos["side"] == "long" else -1
    pct = round((price / pos["entry"] - 1) * 100 * sign, 3) if price else None
    return {
        "symbol": sym,
        "backtest_winrate_pct": SYMBOL_WINRATE.get(sym),
        "side": pos["side"],
        "entry": pos["entry"],
        "price_now": price,
        "pnl_pct": pct,
        "pnl_pct_x_leverage": round(pct * LEVERAGE, 2) if pct is not None else None,
        "bars_held_15m": max(0, (ctx.now_ms - int(pos.get("entry_ms") or ctx.now_ms)) // BAR_MS),
        "stop": pos.get("stop"),
        "tp": pos.get("tp"),
        "opposite_signal_now": opp,
    }


def ai_choose(ctx: Ctx, options: dict[str, str], fallback: str, data: dict) -> tuple[str, str]:
    """신호가 겹치면 Claude가 고른다(한 실행에 한 번). (고른 선택지, 이유).
    답이 없거나 틀리면 fallback(빈손: 승률 순 PRIORITY, 보유: 유지·자기 반대면 갈아타기)."""
    caller = AI_CALLER or (lambda p: btc_ai.call_claude(p, timeout=btc_ai.CHOOSE_TIMEOUT, cwd=DIR))
    out = btc_ai.choose({**data, "options": options}, list(options), caller)
    choice = out["choice"] or fallback
    reason = out["reason"] if out["choice"] else f"기본 규칙({out['fallback']})"
    ctx.event(
        event="ai_choose",
        packet=data,
        options=options,
        ai_raw=out["raw"],
        choice=choice,
        fallback=out["fallback"],
        secs=out["secs"],
    )
    return choice, reason


def _opt_text(opt: str, cands: dict[str, dict], pos: dict | None) -> str:
    kind, _, sym = opt.partition(":")
    if kind == "keep":
        return f"{_who(pos_symbol(pos), pos['side'])} 유지" if pos else "유지"
    if kind == "none":
        return "아무것도 안 함"
    c = cands[sym]
    if kind == "flip":
        return f"{_name(sym)} 갈아타기 → {_who(sym, c['side'])}"
    if kind == "close":
        return f"{_name(sym)} 청산만(다시 들어가기는 막힘: {c['skip']})"
    if kind == "switch":
        return f"{_who(pos_symbol(pos), pos['side'])} 닫고 {_who(sym, c['side'])}"
    return f"{_who(sym, c['side'])} 진입"


def _plan_or_skip(ctx: Ctx, cand: dict, t_start: float) -> dict | None:
    """진입 직전 AI 손절·익절. 선택 AI를 기다린 뒤라 시간 한도를 다시 본다."""
    if _clock() - t_start > MAX_RUN_SECS:
        cand["skip"] = f"이번 실행이 {MAX_RUN_SECS}초를 넘겨 새 진입은 건너뜀(AI 대기)"
        return None
    return ai_plan(cand["ctx"], cand["side"], cand["sig"], cand["price"], ctx.now_ms)


def trade(ctx: Ctx, state: dict, sigs: dict[str, dict], halts: list[str], t_start: float) -> dict:
    """이번 봉 매매 결정과 실행. {"decision", "main": 대표 신호, "target": 들어간(갈) 종목}."""
    pos = state.get("position")
    a_sym = pos_symbol(pos) if pos else None
    notes: list[str] = []
    main: dict | None = None
    dirs: dict[str, dict] = {}
    # 우선순위 순(범위 밖 보유 종목은 맨 뒤)
    for sym in list(PRIORITY) + [s for s in sigs if s not in PRIORITY]:
        g = sigs.get(sym)
        if not g:
            continue
        if g.get("conflict"):
            mark_bar(state, sym, g["bar"])
            notes.append(f"{_name(sym)} 같은 봉 롱·숏 동시 신호 → 무시")
            main = main or g
        elif g.get("dir"):
            dirs[sym] = g
    if a_sym in dirs and dirs[a_sym]["dir"] == pos["side"]:
        g = dirs.pop(a_sym)
        mark_bar(state, a_sym, g["bar"])
        notes.append(f"{_who(a_sym, g['dir'])} 신호, 이미 {_who(a_sym, pos['side'])} 보유 → 유지")
        main = main or g
    for sym in [s for s in dirs if s != a_sym and unavailable(state, s, ctx.now_ms)]:
        # 약관 미동의로 쉬는 종목: 이미 한 번 알렸으니 조용히 넘긴다
        g = dirs.pop(sym)
        mark_bar(state, sym, g["bar"])
        notes.append(f"{_who(sym, g['dir'])} 신호, TradFi 약관 미동의로 쉬는 중")
        ctx.event(event="skip_unavailable", symbol=sym, side=g["dir"])
    opp = dirs.pop(a_sym, None) if a_sym else None
    others = dirs
    if not opp and not others:
        return {"decision": " · ".join(notes) or "대기", "main": main}

    # 들어갈 수 있는지 먼저 본다(정지·추격 한도·약관 등). 반대 신호 청산은 막히지 않는다.
    # 다른 종목은 우선순위 순으로 필요한 만큼만 본다(빈손 5개, 보유 중 3개). 보유 중이면 승률이
    # 보유 종목보다 SWITCH_EDGE_PP 이상 높은 종목만 바꾸기 후보(AI 호출을 드물게, 사용자 2026-11-12)
    cands: dict[str, dict] = {}
    if opp:
        cands[a_sym] = prepare(ctx, state, a_sym, opp, halts, t_start)
    a_rate = float(SYMBOL_WINRATE.get(a_sym) or 0) if pos else 0.0
    cap = CHOOSE_SWITCH_MAX if pos else CHOOSE_FLAT_MAX
    viable: list[str] = []
    passed: dict[str, str] = {}  # 보지 않은 신호 {종목: 이유}
    tried = 0
    for sym, g in others.items():
        rate = float(SYMBOL_WINRATE.get(sym) or 0)
        if not tradable(sym):
            passed[sym] = "승률 45% 미만이거나 표본 적음(신호 표시만, 진입 안 함)"
        elif pos and rate < a_rate + SWITCH_EDGE_PP:
            passed[sym] = (
                f"승률 {rate:g}%가 보유 {_name(a_sym)} {a_rate:g}% + {SWITCH_EDGE_PP:g}%p보다 낮음"
            )
        elif len(viable) >= cap or tried >= PREPARE_MAX:
            passed[sym] = f"우선순위 밖(후보 {cap}개까지)"
        else:
            tried += 1
            cands[sym] = prepare(ctx, state, sym, g, halts, t_start)
            if not cands[sym]["skip"]:
                viable.append(sym)
    for sym, why in passed.items():
        g = others[sym]
        mark_bar(state, sym, g["bar"])
        main = main or g
        ctx.event(event="signal_passed", symbol=sym, side=g["dir"], why=why)
    if passed:
        names = ", ".join(_name(s) for s in passed)
        notes.append(f"다른 신호 {len(passed)}개는 후보 아님({names})")
    # 알림은 1순위 신호와 보유 종목만(종목 50개라 막힌 신호를 다 알리면 시끄럽다). 나머지는 기록만
    loud = {a_sym} | set(list(others)[:1])
    a_cand = cands.get(a_sym) if opp else None
    if not cands:
        return {"decision": " · ".join(notes) or "대기", "main": main}

    # 선택지와 AI 기본값
    options: dict[str, str] = {}
    if pos:
        if not opp:  # 자기 반대 신호면 유지는 없다(사용자 규칙: 반대 신호면 갈아타기)
            options["keep"] = ""
        if opp:
            options[("close:" if a_cand["skip"] else "flip:") + a_sym] = ""
        for s in viable:
            options[f"switch:{s}"] = ""
        fallback = (("close:" if a_cand["skip"] else "flip:") + a_sym) if opp else "keep"
    else:
        for s in viable:
            options[f"enter:{s}"] = ""
        options["none"] = ""
        fallback = f"enter:{viable[0]}" if viable else "none"  # viable은 우선순위 순서
    for k in options:
        options[k] = _opt_text(k, cands, pos)

    if viable and (pos or len(viable) >= 2):
        data: dict = {
            "candidates": [_candidate_packet(cands[s]) for s in viable],
            "held": _held_packet(ctx, pos, bool(opp)) if pos else None,
            "fallback_rule": options[fallback],
        }
        choice, reason = ai_choose(ctx, options, fallback, data)
        listed = ", ".join(_who(s, cands[s]["side"]) for s in viable)
        held_txt = f"{_who(a_sym, pos['side'])} 보유 중 " if pos else ""
        ctx.say(f"🤖 신호 겹침: {held_txt}{listed} 신호 → {options[choice]}\n이유: {reason}")
    else:
        choice = fallback
    kind, _, chosen = choice.partition(":")

    # 고르지 않은 신호는 처리한 것으로 적고, 막힌 신호는 예전처럼 알린다
    told: set[str] = set()
    for sym, cand in cands.items():
        if sym == chosen and kind in ("enter", "switch", "flip", "close"):
            continue
        mark_bar(state, sym, cand["sig"]["bar"])
        who = _who(sym, cand["side"])
        main = main or cand["sig"]
        if cand["skip"]:
            notes.append(f"{who} 신호, {cand['skip']}")
            if not cand.get("quiet") and sym in loud:
                ctx.say(f"⏸ {who} 신호가 났지만 {cand['skip']}")
            else:
                ctx.event(event="signal_skip", symbol=sym, side=cand["side"], why=cand["skip"])
            told.add(sym)
        else:
            notes.append(f"{who} 신호, 고르지 않음({options[choice]})")

    if kind in ("keep", "none"):
        return {"decision": " · ".join(notes) or "대기", "main": main}

    target = cands[chosen]
    main = target["sig"]
    sym, side, sig = chosen, target["side"], target["sig"]
    who = _who(sym, side)
    plan = None
    if kind in ("enter", "switch", "flip"):
        plan = _plan_or_skip(ctx, target, t_start)
        if plan is None and kind == "switch":
            # 갈아탈 종목에 못 들어가면 들고 있는 것을 닫지 않는다
            # (자기 반대 신호가 있으면 예전처럼 갈아타기)
            mark_bar(state, sym, sig["bar"])
            notes.append(f"{who} 신호, {target['skip']}")
            ctx.say(f"⏸ {who} 신호가 났지만 {target['skip']}")
            if not opp:
                return {"decision": " · ".join(notes), "main": main}
            kind, sym, target = "flip", a_sym, a_cand
            side, sig, who = target["side"], target["sig"], _who(a_sym, target["side"])
            plan = None if target["skip"] else _plan_or_skip(ctx, target, t_start)
    closed_ok = True
    if pos and kind in ("switch", "flip", "close"):
        closed_ok = close_position(ctx.at(a_sym), state, sig["bar"])
    skip = target["skip"] or ("진입 준비 실패(AI 손절·익절 없음)" if plan is None else "")
    if not closed_ok:
        decision = f"{who} 신호, 청산 실패"
    elif skip:
        decision = f"{who} 신호, {skip}"
        if sym not in told and not target.get("quiet"):
            ctx.say(f"⏸ {who} 신호가 났지만 {skip}")
    else:
        open_position(target["ctx"], state, sig, target["gate"], target["flt"], plan)
        p = state.get("position")
        entered = bool(p and pos_symbol(p) == sym and p["side"] == side)
        decision = f"{who} 진입" if entered else f"{who} 신호, 진입 안 함"
        if kind == "switch" and not p and "pending" not in state:
            ctx.say(
                f"⚠️ {_who(a_sym, pos['side'])}은 닫았지만 {who} 진입은 못 했어요 → 지금 빈손이에요"
                "(이유는 위 알림)"
            )
    if "pending" not in state:
        mark_bar(state, sym, sig["bar"])
    notes.insert(0, decision)
    return {"decision": " · ".join(notes), "main": main, "target": sym}


def check_strays(ctx: Ctx, state: dict, held: str) -> bool:
    """보유 종목이 아닌 종목(범위 밖 포함, 계좌 전체)에 포지션이 보일 때.
    MANUAL_ADOPT면: 엔진이 빈손이면 가장 큰 것을 이어받고, 나머지는 손절·익절 주문으로 지킨다
    (guards). 끝난 지킴은 정리한다. 아니면 예전처럼 알리고 신규 진입을 멈춘다. 문제없으면 True."""
    amts = all_position_amts(ctx.client)
    pos = state.get("position")
    mine = pos_symbol(pos) if pos else None
    stray = {s: a for s, a in amts.items() if s != held and s != mine and a}
    guards = state.setdefault("guards", {})
    if MANUAL_ADOPT:
        for sym in [s for s in guards if s not in stray]:
            gctx = ctx.at(sym)
            try:
                cancel_engine_stops(gctx.client)
            except (Rejected, NetworkError):
                continue  # 다음 실행에서 다시
            guards.pop(sym, None)
            ctx.say(f"👋 {_name(sym)} 수동 포지션이 정리돼 지킴 주문을 거뒀어요")
        _save(DIR / "state.json", state)
        if not stray:
            state.pop("stray_key", None)
            return True
        ok = True
        order = sorted(stray, key=lambda s: -abs(stray[s]) * _mark_or_zero(ctx, s))
        if not state.get("position"):
            sym = order.pop(0)
            gctx = ctx.at(sym)
            try:
                cancel_engine_entries(gctx.client)
                cancel_engine_stops(gctx.client)
                guards.pop(sym, None)
                if adopt_manual(gctx, state, position_row(gctx.client), "수동 포지션 확인"):
                    ok = ensure_stop(gctx, state) and ok
                else:
                    order.insert(0, sym)
            except (Rejected, NetworkError) as e:
                ctx.say(f"⚠️ {_name(sym)} 수동 포지션을 이어받지 못했어요({e}) → 다음 실행에서 다시")
                order.insert(0, sym)
        for sym in order:
            try:
                guard_stray(ctx.at(sym), state)
            except (Rejected, NetworkError, OSError, ValueError, KeyError) as e:
                ctx.say(
                    f"🆘 {_name(sym)} 수동 포지션 손절을 걸지 못했어요({e}). 앱에서 확인하세요!"
                )
                ok = False
        _save(DIR / "state.json", state)
        return ok
    if not stray:
        state.pop("stray_key", None)
        return True
    txt = ", ".join(f"{s} {a:+}" for s, a in stray.items())
    if state.get("stray_key") != txt:
        ctx.say(
            f"⚠️ 엔진이 들고 있지 않은 종목에 포지션이 있어요({txt}). 건드리지 않고 "
            "신규 진입을 멈춰요."
        )
        state["stray_key"] = txt
    set_no_entry(f"엔진 밖 포지션 {txt}", ctx.send)
    return False


def _mark_or_zero(ctx: Ctx, sym: str) -> float:
    try:
        return float(position_row(ctx.at(sym).client)["entry"]) or 1.0
    except (Rejected, NetworkError, OSError, ValueError, KeyError):
        return 1.0


def guard_stray(ctx: Ctx, state: dict) -> None:
    """엔진이 다른 종목을 들고 있을 때 손으로 연 포지션: 손절 3%·익절 +10% 주문만 걸어 지킨다
    (신호로 관리하지는 않는다). 수량·평균가가 바뀌면 다시 건다. 손절가를 이미 지났으면
    시장가로 닫는다."""
    c = ctx.client
    sym = ctx.symbol
    row = position_row(c)
    if not row["amt"]:
        return
    side = "long" if row["amt"] > 0 else "short"
    guards = state.setdefault("guards", {})
    g = guards.get(sym)
    if (
        not g
        or g["side"] != side
        or abs(g["qty"] - abs(row["amt"])) > 1e-9
        or abs(g["entry"] - row["entry"]) > 1e-9
    ):
        cancel_engine_stops(c)
        stop, tp = _manual_levels(ctx, side, row["entry"])
        g = {
            "side": side,
            "qty": abs(row["amt"]),
            "entry": row["entry"],
            "stop": stop,
            "tp": tp,
            "n": (g or {}).get("n", 0),
            "stop_id": None,
            "tp_id": None,
        }
        guards[sym] = g
        _save(DIR / "state.json", state)
        tp_txt = f", 익절 {_px(tp)}" if tp else ""
        ctx.say(
            f"🛡 {_who(sym, side)} 수동 포지션 {g['qty']} @ {_px(row['entry'])}: 엔진이 다른 종목을 "
            f"들고 있어 손절 {_px(stop)}{tp_txt} 주문만 걸어 지켜요"
        )
    open_ids = {a.get("clientAlgoId") for a in open_algos(c)}
    minute = ctx.now_ms - ctx.now_ms % 60_000
    if g.get("stop_id") not in open_ids:
        g["n"] += 1
        cid = _cid(minute, f"g{g['n']}", sym)
        try:
            r = place_stop(c, side, Decimal(str(g["stop"])), cid)
            ok = stop_placed(c, r, cid)
        except Rejected:
            ok = False
        if not ok:
            close_side = "SELL" if side == "long" else "BUY"
            market(
                c, close_side, _close_qty(row["amt"]), _cid(minute, f"gx{g['n']}", sym), reduce=True
            )
            guards.pop(sym, None)
            _save(DIR / "state.json", state)
            ctx.say(f"✂️ {_who(sym, side)} 수동 포지션이 이미 손절가를 지나 시장가로 닫았어요")
            return
        g["stop_id"] = cid
        _save(DIR / "state.json", state)
    if g.get("tp") and g.get("tp_id") not in open_ids:
        g["n"] += 1
        cid = _cid(minute, f"h{g['n']}", sym)
        try:
            place_stop(c, side, Decimal(str(g["tp"])), cid, TP_KIND)
            g["tp_id"] = cid
        except Rejected as e:
            ctx.say(f"⚠️ {_name(sym)} 수동 포지션 익절 주문 실패({e}) → 손절만 걸어 둠")
            g["tp"] = None
        _save(DIR / "state.json", state)


# ---------------------------------------------------------------- 한 번 실행


def run(
    now: datetime | None = None,
    client=None,
    market_: Market | None = None,
    send: Callable[[str], Any] = notify.send,
    boot: bool = False,
    before: dict | None = None,
) -> dict:
    """한 번 실행. boot=True는 부팅 직후 복구 실행(공백이 짧아도 복구 보고를 보낸다)."""
    now = now or datetime.now(KST)
    now_ms = int(now.timestamp() * 1000)
    DIR.mkdir(parents=True, exist_ok=True)
    live = mode() == "live"
    client = client if client is not None else Client()
    market_ = market_ or Market()
    state_file = DIR / "state.json"
    first = not state_file.exists()
    state = _load(state_file, {})
    uni = use_universe(now_ms)
    gap_ms = now_ms - state["last_run_ms"] if state.get("last_run_ms") else None
    recovering = boot or (gap_ms is not None and gap_ms > RECOVER_GAP_MS)
    lookback = MAX_SIGNAL_BARS
    if recovering and gap_ms is not None:
        lookback = min(max(MAX_SIGNAL_BARS, gap_ms // BAR_MS + 1), RECOVER_MAX_BARS)
    pos_before = before if before is not None else state.get("position")
    pos_before = dict(pos_before) if pos_before else None
    ctx = Ctx(client, market_, send, now, live)
    t_start = _clock()
    pos = state.get("position")
    if live and not pos and state.get("orphan_live"):
        state["position"] = state.pop(
            "orphan_live"
        )  # 실전으로 돌아왔다: 남겨 둔 포지션을 다시 관리
        ctx.say("🔁 드라이런 동안 남아 있던 실전 포지션을 다시 관리해요")
    elif pos and pos.get("mode") == "live" and not live:
        # 실전 포지션이 열린 채 드라이런으로 바뀌었다: 버리지 않고 보관하고 크게 알린다
        state["orphan_live"] = pos
        state["position"] = None
        send(
            f"🆘 실전 {_who(pos_symbol(pos), pos['side'])} 포지션이 열린 채 드라이런으로 "
            "바뀌었어요. 거래소 손절은 "
            "걸려 있지만 반대 신호 청산·손절 확인은 멈춰요. 직접 닫거나 live로 되돌리세요."
        )
    elif pos and pos.get("mode") != ("live" if live else "dry"):
        ctx.event(event="drop_position", position=pos)  # 드라이런 가상 포지션은 실전에서 버린다
        state["position"] = None

    can_trade = True
    held = held_symbol(state)
    if live:
        # 시세가 막혀도 손절 확인·체결 대사는 먼저 한다
        dual = _get(client, "/fapi/v1/positionSide/dual")
        if dual.get("dualSidePosition"):
            ctx.say("❌ 헤지 모드(양방향)라 매매하지 않아요. 단방향(One-way)으로 바꿔 주세요")
            can_trade = False
        else:
            can_trade = reconcile_live(ctx.at(held), state)
            # 엔진은 한 종목만 든다: 다른 종목 포지션이 보이면 짐작하지 않고 신규 진입을 멈춘다
            if not check_strays(ctx, state, held):
                can_trade = False
        _save(state_file, state)
    # 들고 있는(끝나지 않은 주문 포함) 종목은 범위에서 빠졌어도 받는다
    extra = tuple(pos_symbol(state[k]) for k in ("position", "pending") if state.get(k))
    t_fetch = _clock()
    rows_by, feed_errs = fetch_rows(ctx, extra + (held,))
    fetch_secs = round(_clock() - t_fetch, 2)
    ctx.shared["rows"] = rows_by
    ctx.rows15 = rows_by.get(ctx.symbol, [])
    if feed_errs:
        ctx.event(event="feed_error", errors=feed_errs)
    hctx = ctx.at(held)
    if not live:
        reconcile_dry(hctx, state, hctx.rows15, now_ms)
    _save(state_file, state)

    acct = _get(client, "/fapi/v2/account")
    ctx.shared["acct"] = acct
    if live:
        equity = float(acct["totalMarginBalance"])
    else:
        state.setdefault("dry_seed", float(acct["totalMarginBalance"]))
        closed = sum(
            r.get("pnl") or 0 for r in _read_jsonl("closed.jsonl") if r.get("mode") == "dry"
        )
        equity = state["dry_seed"] + closed
        p = state.get("position")
        if p and rows_by.get(pos_symbol(p)):
            sign = 1 if p["side"] == "long" else -1
            last_px = float(rows_by[pos_symbol(p)][-1][4])
            equity += sign * p["qty"] * (last_px - p["entry"])
    halts = check_halts("live" if live else "dry", state, equity, now, ctx.say)

    sigs: dict[str, dict] = {}
    for sym in rows_by:
        g = latest_signal(rows_by[sym], now_ms, last_bar_for(state, sym), lookback=lookback)
        g["symbol"] = sym
        sigs[sym] = g
    pos = state.get("position")
    held = pos_symbol(pos) if pos else held
    bad = {s: g.get("error") or "마지막 봉이 오래됨" for s, g in sigs.items()}
    bad = {s: why for s, why in bad.items() if sigs[s].get("error") or sigs[s].get("stale")}
    bad.update({s: f"시세 조회 실패({e})" for s, e in feed_errs.items()})
    if bad:
        ctx.event(event="feed_bad", symbols=bad)
    for s in bad:
        sigs.pop(s, None)
    notes: list[str] = []
    if pos and held in bad:
        notes.append(f"{_name(held)} 시세 이상: {bad[held]}")
        ctx.say(f"⚠️ {_name(held)} 시세 이상: {bad[held]} → 이번 봉은 {_name(held)} 신호를 못 봐요")
    elif not sigs and bad:
        notes.append(f"시세 이상: {', '.join(f'{_name(s)} {w}' for s, w in bad.items())}")
        ctx.say(f"⚠️ {notes[-1]} → 이번 봉은 매매하지 않아요")
    sigged = [g for g in sigs.values() if g.get("dir")]
    if sigged and not can_trade:
        for g in sigged:
            mark_bar(state, g["symbol"], g["bar"])
        names = ", ".join(_who(g["symbol"], g["dir"]) for g in sigged)
        out = {"decision": f"{names} 신호, 수동 개입·설정 문제로 매매 안 함", "main": sigged[0]}
    else:
        out = trade(ctx, state, sigs, halts, t_start)
    parts = ([out["decision"]] if out["decision"] != "대기" else []) + notes
    decision = " · ".join(parts) or "대기"
    sig = out.get("main") or sigs.get(SYMBOL) or {}
    if ctx.ai_streak is not None:
        state["ai_fallbacks"] = ctx.ai_streak
    state["last_run_ms"] = now_ms
    _save(state_file, state)
    try:
        write_near(rows_by, now_ms, uni, held)
    except Exception as e:  # noqa: BLE001 — 화면용 파일 실패가 실행을 깨면 안 된다
        ctx.event(event="near_error", error=f"{type(e).__name__}: {e}"[:200])
    p = state.get("position")
    btc_sig = sigs.get(SYMBOL) or {}
    summary = {
        "event": "run",
        "symbol": sig.get("symbol"),
        "bar": _bar_txt(sig["bar"]) if sig.get("bar") else None,
        "signal": sig.get("dir"),
        "close": sig.get("close"),
        "signals": {s: g["dir"] for s, g in sigs.items() if g.get("dir")},
        "decision": decision,
        "position": {k: p.get(k) for k in ("symbol", "side", "qty", "entry", "stop", "tp")}
        if p
        else None,
        "halts": halts,
        "equity": round(equity, 2),
        "indicators": btc_sig.get("indicators"),
        "universe": {"source": uni["source"], "n": len(uni["symbols"]), "fetched": len(rows_by)},
        "fetch_secs": fetch_secs,
    }
    if p and "symbol" not in p:
        summary["position"]["symbol"] = SYMBOL
    if bad:
        summary["feed_bad"] = bad
    if recovering and not first:
        missed = sum(g.get("missed", 0) for g in sigs.values())
        summary["recover"] = {
            "gap_min": round(gap_ms / 60_000) if gap_ms else None,
            "lookback": lookback,
            "missed": missed,
        }
        stop_ok = None
        if p and live and p.get("stop_id"):
            try:
                stop_ok = stop_is_open(ctx.at(pos_symbol(p)).client, p["stop_id"])
            except (Rejected, NetworkError):
                stop_ok = False
        elif p and live:
            stop_ok = False
        summary["recover"]["stop_ok"] = stop_ok
        rsig = {**sig, "missed": missed}
        send(recover_text(live, gap_ms, rsig, pos_before, p, decision, halts, state, stop_ok))
    ctx.event(**summary)
    if first:
        mode_ko = "실전" if live else "드라이런(주문 없음)"
        held_txt = f"{_who(pos_symbol(p), p['side'])} {p['qty']}" if p else "없음"
        send(
            f"🤖 BTC 자동매매 엔진 시작 — {mode_ko}\n"
            f"{_universe_txt(uni)} · SCR 15분 · Stoch 9/CCI 20/RSI 21 · 3배 · "
            "한 번에 한 종목 · 손절·익절은 진입 때 AI(기본 손절 3%) · 롱↔숏 갈아타기\n"
            f"마지막 봉 {summary['bar']}: {decision} · 포지션 {held_txt}"
            + (f"\n진입 정지: {', '.join(halts)}" if halts else "")
        )
    return summary


def _universe_txt(uni: dict) -> str:
    if uni["source"] == "universe":
        return f"거래량 상위 {len(uni['symbols'])}종목({uni['updated_at']} 기준, 승률 순 우선)"
    return " · ".join(_name(s) for s in uni["symbols"])


def write_near(rows_by: dict[str, list[list]], now_ms: int, uni: dict, held: str | None) -> dict:
    """화면 '신호 가까운 순'용 `near.json`: 종목·방향마다 넘어온 지표 수, 지금 신호인지, 남은 지표가
    넘어오려면 다음 봉이 어디에 마감해야 하는지(어림값). 엔진이 받은 봉으로 계산한다."""
    out = []
    for sym, rows in rows_by.items():
        closed = [r for r in rows if int(r[6]) < now_ms]
        if len(closed) < 100:
            continue
        h = [float(r[2]) for r in closed]
        lo = [float(r[3]) for r in closed]
        c = [float(r[4]) for r in closed]
        st = uni["stats"].get(sym) or {}
        base = {
            "symbol": sym,
            "bar": int(closed[-1][0]),
            "price": c[-1],
            "winrate": uni["winrate"].get(sym),
            "rank": st.get("rank"),
            "priority": PRIORITY.index(sym) + 1 if sym in PRIORITY else None,
            "low_sample": btc_universe.low_sample(st) if st else False,
            "contract": st.get("contract"),
            "held": sym == held,
        }
        for side, r in scr_lab.near(h, lo, c, SCR_PARAMS).items():
            out.append({**base, "side": side, **r})
    data = {
        "updated_at": _now_iso(datetime.fromtimestamp(now_ms / 1000, KST)),
        "source": uni["source"],
        "universe_at": uni.get("updated_at"),
        "w": SCR_PARAMS["w"],
        "rows": out,
    }
    _save(DIR / "near.json", data)
    return data


def recover_text(
    live: bool,
    gap_ms: int | None,
    sig: dict,
    before: dict | None,
    after: dict | None,
    decision: str,
    halts: list[str],
    state: dict,
    stop_ok: bool | None = None,
) -> str:
    """복구 실행 보고. 꺼져 있던 시간, 놓친 신호, 포지션·손절 상태, 한 일."""
    tag = "" if live else "[드라이런] "
    gap = f"약 {gap_ms // 60_000}분" if gap_ms is not None else "알 수 없음(기록 없음)"
    lines = [f"🛟 {tag}BTC 자동매매 복구 실행 — 지난 실행 뒤 {gap} 공백"]
    missed = sig.get("missed", 0)
    if sig.get("dir") or sig.get("conflict"):
        last = SIDE_KO.get(sig.get("dir"), "롱·숏 동시")
        name = _name(sig.get("symbol") or SYMBOL)
        lines.append(f"놓친 신호 {missed}개 · 가장 최근 {sig.get('bar_txt')} {name} {last}")
    else:
        lines.append("놓친 신호 없음")

    def held(p: dict | None) -> str:
        if not p:
            return "없음"
        sym = pos_symbol(p)
        stop = f"손절 {_px(p['stop'])}" if p.get("stop") else "손절 없음"
        return f"{_who(sym, p['side'])} {p['qty']} {_name(sym)} @ {_px(p['entry'])} · {stop}"

    if before and not after:
        lines.append(f"포지션: {held(before)} → 정리됨")
    elif before and after and before.get("entry") != after.get("entry"):
        lines.append(f"포지션: {held(before)} → {held(after)}")
    else:
        lines.append(f"포지션: {held(after)}")
    if after and not live:
        lines.append("가상 손절(드라이런, 거래소 주문 없음)")
    elif after:
        lines.append(
            "바이낸스 손절 주문 걸려 있음 ✅"
            if stop_ok
            else "🆘 바이낸스에 손절 주문이 안 보여요. 앱에서 바로 확인하세요!"
        )
    if gap_ms is not None and gap_ms > RECOVER_MAX_BARS * BAR_MS:
        lines.append(
            f"⚠️ 공백이 {RECOVER_MAX_BARS * BAR_MS // 3_600_000}시간을 넘어 "
            "그 전 신호는 확인하지 못했어요"
        )
    lines.append(f"처리: {decision}")
    if halts:
        lines.append(f"진입 정지: {', '.join(halts)}")
    if state.get("pending"):
        lines.append("⚠️ 끝나지 않은 주문이 있어 다음 실행이 이어 받아요")
    return "\n".join(lines)


# ---------------------------------------------------------------- 점검·실전 시작


def preflight(client=None) -> list[tuple[bool, str]]:
    """GET만 보내는 점검: 키, 거래 규칙, 레버리지, 포지션 모드, 잔고, 미체결."""
    client = client if client is not None else Client()
    out: list[tuple[bool, str]] = []
    uni = use_universe()
    st0 = _load(DIR / "state.json", {})
    # 자세히 보는 종목: 예전 네 종목 + 들고 있는 종목(50종목을 다 적으면 알림이 너무 길다)
    detail = list(dict.fromkeys(DEFAULT_SYMBOLS + (held_symbol(st0),)))
    m = mode()
    out.append((True, f"모드: {'실전(live)' if m == 'live' else '드라이런(dry)'}"))
    out.append((not halted(), "HALT " + ("있음" if halted() else "없음")))
    out.append((not entries_blocked(), "NO_ENTRY " + ("있음" if entries_blocked() else "없음")))
    try:
        r = _get(client, "/sapi/v1/account/apiRestrictions")
        fut = bool(r.get("enableFutures"))
        out.append(
            (
                fut,
                f"키 권한: 선물 거래 {'허용' if fut else '없음'}, "
                f"IP 제한 {'있음' if r.get('ipRestrict') else '없음'}, "
                f"출금 {'허용(위험)' if r.get('enableWithdrawals') else '없음'}",
            )
        )
    except (Rejected, NetworkError) as e:
        out.append((False, f"키 권한 조회 실패: {e}"))
    bad = [s for s in SYMBOLS if filters(SymClient(client, s))["status"] != "TRADING"]
    out.append(
        (
            not bad,
            f"종목 범위: {_universe_txt(uni)} · {len(SYMBOLS)}종목"
            + (f" · 거래 불가 {', '.join(bad)}" if bad else ""),
        )
    )
    for sym in detail:
        flt = filters(SymClient(client, sym))
        out.append(
            (
                flt["status"] == "TRADING",
                f"{sym} {flt['status']} · 호가 {flt['tick']} · "
                f"수량 단위 {flt['step']} (최소 {flt['min_qty']}) · "
                f"최소 주문 {flt['min_notional']} USDT",
            )
        )
    dual = _get(client, "/fapi/v1/positionSide/dual")["dualSidePosition"]
    out.append(
        (
            not dual,
            f"포지션 모드: {'헤지(양방향) — 단방향으로 바꿔야 함' if dual else '단방향(One-way)'}",
        )
    )
    multi = _get(client, "/fapi/v1/multiAssetsMargin").get("multiAssetsMargin")
    out.append((True, f"멀티에셋 모드: {'켜짐' if multi else '꺼짐'}"))
    for sym in detail:
        sc = next(
            (x for x in _get(client, "/fapi/v1/symbolConfig", symbol=sym) if x["symbol"] == sym),
            None,
        )
        if sc is None:
            out.append((False, f"{sym} 레버리지 설정을 못 읽음"))
            continue
        lev = int(sc["leverage"])
        lev_note = "그대로" if lev == LEVERAGE else f"그 종목 첫 실전 진입 때 {LEVERAGE}배로 바꿈"
        out.append((True, f"{_name(sym)} 레버리지 {lev}배 ({lev_note}), 마진 {sc['marginType']}"))
    acct = _get(client, "/fapi/v2/account")
    out.append(
        (
            bool(acct.get("canTrade")),
            f"canTrade {acct.get('canTrade')} · 평가액 "
            f"{float(acct['totalMarginBalance']):.2f} · "
            f"가용 {float(acct['availableBalance']):.2f} USDT",
        )
    )
    st = _load(DIR / "state.json", {})
    held = pos_symbol(st.get("position")) if st.get("position") else None
    for sym, amt in all_position_amts(client).items():
        if amt or sym in detail:
            out.append((amt == 0 or sym == held, f"{sym} 포지션 {amt:+}"))
    unav = {s: r for s, r in (st.get("unavailable") or {}).items() if unavailable(st, s, _ms_now())}
    for sym, r in unav.items():
        why = f"({r.get('code')}, {r.get('at')}부터 24시간)"
        out.append((True, f"{sym} 약관 미동의로 쉬는 중{why}"))
    for path, key in (
        ("/fapi/v1/order", "origClientOrderId"),
        ("/fapi/v1/algoOrder", "clientAlgoId"),
    ):
        try:
            gone = _query(client, path, key, f"{PREFIX}preflight-none") is None
            note = "-2013(없음)으로 와요" if gone else "없는 주문인데 결과가 와요"
        except (Rejected, NetworkError) as e:
            gone, note = False, f"예상 밖 응답 {e} — 중복 방지 조회가 안 맞을 수 있음"
        out.append((gone, f"없는 주문 조회 {path}: {note}"))
    raw = _get(client, "/fapi/v1/openAlgoOrders", symbol=SYMBOL)
    out.append((isinstance(raw, list), f"조건부 미체결 목록 응답 모양: {type(raw).__name__}"))
    for sym in detail:
        sc_ = SymClient(client, sym)
        orders = _get(sc_, "/fapi/v1/openOrders", symbol=sym)
        algos = open_algos(sc_)
        out.append(
            (
                True,
                f"{_name(sym)} 미체결 일반 {len(orders)} · 조건부 {len(algos)} (엔진 손절 "
                f"{sum(1 for a in algos if str(a.get('clientAlgoId', '')).startswith(PREFIX))})",
            )
        )
    flt = filters(client)
    price = btc.fetch_price()
    qty, why = size_qty(float(acct["availableBalance"]), price, margin_frac(), flt)
    out.append(
        (
            bool(qty),
            f"지금 진입하면 {qty} BTC (가격 {price:,.1f}, margin_frac {margin_frac()})"
            + (f" — {why}" if why else ""),
        )
    )
    return out


def start_live(client=None, send: Callable[[str], Any] = notify.send) -> dict:
    """실전 시작(사용자만 실행): 지금 순평가액을 시드로 적고 mode=live."""
    client = client if client is not None else Client()
    use_universe()
    if _get(client, "/fapi/v1/positionSide/dual")["dualSidePosition"]:
        raise SystemExit("헤지 모드입니다. 단방향으로 바꾼 뒤 다시 실행하세요.")
    for sym, amt in all_position_amts(client).items():
        if amt != 0:
            raise SystemExit(f"{sym} 포지션이 열려 있습니다. 비운 뒤 다시 실행하세요.")
    for sym in SYMBOLS:
        if foreign_orders(SymClient(client, sym)):
            raise SystemExit(f"{sym} 미체결 주문이 있습니다. 취소한 뒤 다시 실행하세요.")
    now = datetime.now(KST)
    snap = binance.snapshot(client._cfg, now, {"live": False})
    net = float(snap["net_equity"])
    old = _load(CAPITAL_FILE, {})
    cap = {
        "live": True,
        "since": now.strftime("%Y-%m-%d"),
        "since_at": now.strftime("%Y-%m-%d %H:%M"),
        "base": net,
        "asset": "USDT",
        "deposits": [],
    }
    if "previous" in old:
        cap["previous"] = old["previous"]
    _write_private(CAPITAL_FILE, json.dumps(cap, ensure_ascii=False, indent=1))
    state = _load(DIR / "state.json", {})
    state["position"] = None  # 드라이런 가상 포지션은 버린다
    state.pop("pending", None)
    _save(DIR / "state.json", state)
    _write_private(DIR / "mode", "live\n")
    _append("events.jsonl", {"at": _now_iso(now), "event": "start_live", "base": net})
    send(
        f"🔴 BTC 자동매매 실전 시작: 시드 {net:,.2f} USDT ({cap['since_at']} KST)\n"
        "다음 15분 봉 마감부터 실주문을 내요. 멈추려면 touch ~/.minerva/btc_trade/HALT "
        "(신규 진입만 정지), 완전히 끄려면 echo dry > ~/.minerva/btc_trade/mode"
    )
    return cap


def status() -> dict:
    events = [e for e in _read_jsonl("events.jsonl") if e.get("event") == "run"]
    return {
        "mode": mode(),
        "halt": halted(),
        "no_entry": (DIR / "NO_ENTRY").read_text().strip() if entries_blocked() else None,
        "state": _load(DIR / "state.json", {}),
        "last_run": events[-1] if events else None,
    }


def _uptime() -> float:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def recover() -> dict:
    """부팅 직후 복구 실행(`minerva-btc-recover.service`). 네트워크·시계가 잡힐 때까지 다시 한다.

    거래소 손절·익절 주문은 서버가 꺼져 있어도 바이낸스에서 작동한다. 여기서는 그동안 닫힌
    포지션을 기록하고, 손절이 빠졌으면 다시 걸고, 놓친 신호를 처리한다(`run(boot=True)`).
    """
    st = _load(DIR / "state.json", {})
    booted_ms = int((time.time() - _uptime()) * 1000)
    if st.get("last_run_ms", 0) > booted_ms:
        return {"event": "recover", "skipped": "부팅 뒤 이미 실행됨(타이머가 먼저 복구)"}
    before = st.get("position")  # 재시도 중 앞 시도가 닫아도 보고는 처음 상태 기준
    last: Exception | None = None
    t0 = _clock()
    k = 0
    while True:
        k += 1
        try:
            return run(boot=True, before=before)
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):  # 요청 한도: 빨리 다시 하면 차단이 길어진다
                raise
            last = e
        except (NetworkError, OSError) as e:  # 부팅 직후 DNS·네트워크가 덜 올라옴
            last = e
        except Rejected as e:
            if e.code != -1021:  # 시계가 덜 맞춰짐(timestamp outside recvWindow)만 다시
                raise
            last = e
        log.warning("recover try %d failed: %s", k, last)
        if _clock() - t0 + RECOVER_TRY_GAP > RECOVER_DEADLINE_S:
            raise RuntimeError(f"복구 실행 {k}번 실패({RECOVER_DEADLINE_S}초): {last}")
        _sleep(RECOVER_TRY_GAP)


def _lock(timeout: float = 60):
    import fcntl

    DIR.mkdir(parents=True, exist_ok=True)
    f = open(DIR / ".lock", "w")  # noqa: SIM115 — 프로세스 끝까지 잡고 있어야 한다
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except BlockingIOError:
            if time.time() > deadline:
                f.close()
                return None
            time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BTCUSDT 선물 자동매매 (mode 파일이 live일 때만 실주문)"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "recover", "preflight", "start-live", "status"],
    )
    parser.add_argument("--confirm", action="store_true", help="start-live: 실전 시작 확인")
    parser.add_argument("--notify", action="store_true", help="preflight: 결과를 텔레그램으로")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=1))
        return 0
    if args.command == "preflight":
        checks = preflight()
        text = "\n".join(f"{'✅' if ok else '❌'} {msg}" for ok, msg in checks)
        print(text)
        if args.notify:
            notify.send("🔎 BTC 자동매매 점검\n" + text)
        return 0 if all(ok for ok, _ in checks) else 1
    if args.command == "start-live" and not args.confirm:
        parser.error("start-live는 실주문을 켭니다. 확인했으면 --confirm 을 붙이세요")
    lock = _lock()
    if lock is None:
        notify.send(f"⚠️ BTC 자동매매({args.command}) 건너뜀: 다른 실행이 1분 넘게 진행 중")
        return 0
    try:
        if args.command == "start-live":
            cap = start_live()
            print(json.dumps(cap, ensure_ascii=False))
            return 0
        if args.command == "recover":
            summary = recover()
        else:
            summary = run()
    except Exception as e:  # noqa: BLE001 — 어떤 오류든 조용히 멈추면 안 된다(15분마다 알림)
        log.exception("btc_trade %s failed", args.command)
        notify.send(f"❌ BTC 자동매매({args.command}) 오류: {type(e).__name__}: {str(e)[:300]}")
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
