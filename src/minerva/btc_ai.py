"""진입할 때 Claude에게 손절·익절 폭을 묻는다 (사용자 결정 2026-11-08 22:25 KST "2단계로 바로 해").

엔진(`btc_trade.py`)이 포지션을 열기 직전에 우리 자료로 만든 작은 JSON을 `claude -p`로 넘기고
`{"stop_pct": x, "tp_pct": y 또는 null, "reason": "..."}`만 받는다. 진입 여부·방향은 묻지 않는다.
답이 없거나(60초), JSON이 아니거나, 범위(손절 1~5%, 익절 1~10% 또는 없음)를 벗어나면
기본값(손절 3%, 익절 없음)을 쓴다.

AI가 정한 청산 폭은 백테스트할 수 없다(같은 답을 다시 만들 수 없다). 사용자가 알고 바로 쓰기로 했다.

`--strict-mcp-config`: 텔레그램 플러그인을 띄우지 않는다(같은 봇을 두 프로세스가 받으면 상시
세션의 텔레그램이 끊긴다, `scripts/routine.sh`와 같은 이유). 도구는 모두 막는다.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

from minerva.strategy import btc
from minerva.strategy.indicators import tm_states

CLAUDE = os.environ.get("MINERVA_CLAUDE", str(Path.home() / ".local/bin/claude"))
MODEL = os.environ.get("MINERVA_BTC_AI_MODEL", "sonnet")
TIMEOUT = 60
# 종목 선택(신호 겹침)은 손절·익절 질문이 뒤에 또 있어 더 짧게
CHOOSE_TIMEOUT = 45
BUDGET_USD = "0.50"
DEFAULT_STOP = 3.0
STOP_RANGE = (1.0, 5.0)
TP_RANGE = (1.0, 10.0)
BLOCKED_TOOLS = (
    "Bash Edit Write NotebookEdit Read Grep Glob WebFetch WebSearch Task Agent TodoWrite"
).split()

PROMPT = """너는 바이낸스 USDⓈ-M 선물 자동매매 엔진의 청산 폭 담당이다.
종목(<SYMBOL>)과 진입 여부·방향은 이미 정해졌다(SCR 15분봉 신호, 레버리지 3배). 너는 이번
포지션의 손절 폭과 익절 폭만 정한다. 아래 자료만 근거로 쓴다.

가장 중요한 것은 돈을 잃지 않는 것이다(사용자 지시 "최대한 손해 보면 안 된다").
- 손절은 15분봉 잡음(recent_15m_range_pct, atr14_pct의 15m) 바깥에 두되, 구조(swing20 고저)가
  허락하는 한 가장 가깝게. 잡음 안이면 흔들려 나가고, 너무 멀면 한 번에 크게 잃는다.
- 이미 많이 움직였으면(신호 종가와 지금 가격 차이, 20봉 고저까지 거리) 익절을 걸어 이익을 챙긴다.
  익절 없이(null) 들고 가는 것은 추세가 뚜렷하고 아직 덜 움직였을 때만.
- backtest_winrate_pct는 이 종목에서 이 신호의 과거 1년 승률이다. 낮으면(50% 아래) 익절을 더
  가깝게, 손절도 구조가 허락하는 만큼 좁게 잡는다.
- market_kind는 종목 종류다(crypto, TradFi 미국 주식·ETF, 한국 주식, 원자재 등). TradFi는
  기초 시장이 열린 시간에 크게 움직이고, 쉬는 시간 뒤 첫 봉에 크게 튈 수 있다.
  종목마다 변동성이 크게 다르니 recent_15m_range_pct·atr14_pct를 꼭 보고 그 종목의 잡음 폭에 맞춘다.
- SOXLUSDT는 반도체 3배 ETF라 15분봉 하나가 2~7%씩 움직인다. 손절을 1~2%로 두면 잡음에 바로
  걸린다: 15분 폭을 꼭 보고 정한다. 다른 레버리지 ETF(TQQQ 등)도 같다.

규칙:
- stop_pct: 진입가에서 손절까지 거리(%), 1.0 이상 5.0 이하. 기본은 3.0.
- tp_pct: 진입가에서 익절까지 거리(%), 1.0 이상 10.0 이하.
  익절 없이 반대 신호까지 들고 가려면 null.
- 3배 레버리지라 가격 1%는 계좌 약 3%다(손절 5%면 계좌 약 15%).
- reason: 왜 그렇게 정했는지 한국어 한 문장(80자 이내).

출력은 JSON 객체 하나만. 설명·코드 블록·다른 글자를 붙이지 않는다.
{"stop_pct": 숫자, "tp_pct": 숫자 또는 null, "reason": "..."}

자료:
"""


def _bar_range(rows: list[list], n: int = 16) -> dict | None:
    """최근 n개 15분봉의 고저 폭(종가 대비 %): 평균·가장 큰 값·마지막 봉."""
    win = rows[-n:]
    if not win:
        return None
    rng = [(float(r[2]) - float(r[3])) / float(r[4]) * 100 for r in win if float(r[4])]
    if not rng:
        return None
    return {
        "bars": len(rng),
        "avg": round(sum(rng) / len(rng), 3),
        "max": round(max(rng), 3),
        "last": round(rng[-1], 3),
    }


def _atr_pct(rows: list[list], n: int = 14) -> float | None:
    """Wilder ATR(n)을 마지막 종가 대비 %로."""
    if len(rows) < n + 2:
        return None
    h = [float(r[2]) for r in rows]
    lo = [float(r[3]) for r in rows]
    c = [float(r[4]) for r in rows]
    tr = [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))]
    a = sum(tr[:n]) / n
    for x in tr[n:]:
        a += (x - a) / n
    return round(a / c[-1] * 100, 3)


def _swing(rows: list[list], price: float, n: int = 20) -> dict:
    """최근 n봉 최고가·최저가까지 거리(%)."""
    win = rows[-n:]
    hi = max(float(r[2]) for r in win)
    lo = min(float(r[3]) for r in win)
    return {
        "high_pct": round((hi / price - 1) * 100, 2),
        "low_pct": round((lo / price - 1) * 100, 2),
    }


def _direction(rows: list[list], now_ms: int) -> str:
    bars, _ = btc.parse(rows, now_ms)
    return btc.direction(tm_states(bars)[-1]) if len(bars) >= 30 else "중립"


def own_summary(closed: list[dict], mode: str, last: int = 20) -> dict:
    rows = [r for r in closed if r.get("mode") == mode][-last:]
    pcts = [r["pct"] for r in rows if r.get("pct") is not None]
    return {
        "mode": mode,
        "trades": len(rows),
        "wins": sum(1 for r in rows if (r.get("pnl") or 0) > 0),
        "avg_pct": round(sum(pcts) / len(pcts), 2) if pcts else None,
        "recent": [
            {
                "side": r["side"],
                "pct": round(r["pct"], 2) if r.get("pct") is not None else None,
                "why": r.get("why"),
            }
            for r in rows[-5:]
        ],
    }


def packet(
    side: str,
    sig: dict,
    price: float,
    klines: Callable[[str], list[list]],
    funding: float | None,
    now_ms: int,
    closed: list[dict],
    mode: str,
    ref_stats: dict | None,
    symbol: str = "BTCUSDT",
    winrate: float | None = None,
    market: str | None = None,
) -> dict:
    """AI에게 넘길 자료(작게). klines(interval)는 마감 여부와 상관없이 받은 봉."""
    frames = {k: [r for r in klines(k) if int(r[6]) < now_ms] for k in ("15m", "1h", "4h", "1d")}
    d1 = _direction(frames["1d"], now_ms)
    ref = None
    if ref_stats:
        ref = {
            "source": ref_stats.get("source"),
            "period": ref_stats.get("period"),
            "this_side_regime": ref_stats.get("by", {}).get(f"{side}|{d1}"),
            "this_side_all": ref_stats.get("by", {}).get(f"{side}|all"),
        }
    return {
        "symbol": symbol,
        "market_kind": market,
        "side": side,
        "signal_bar_kst": sig.get("bar_txt"),
        "signal_close": sig.get("close"),
        "price_now": price,
        "scr_at_signal": sig.get("at_signal"),
        "direction_tm": {
            "1h": _direction(frames["1h"], now_ms),
            "4h": _direction(frames["4h"], now_ms),
            "1d": d1,
        },
        "backtest_winrate_pct": winrate,
        "atr14_pct": {k: _atr_pct(frames[k]) for k in ("15m", "1h", "1d")},
        "recent_15m_range_pct": _bar_range(frames["15m"]),
        "swing20": {k: _swing(frames[k], price) for k in ("4h", "1d") if len(frames[k]) >= 20},
        "funding_rate_pct": round(funding * 100, 4) if funding is not None else None,
        "own_trades": own_summary(closed, mode),
        "backtest_ref": ref,
    }


def _json_obj(text: str) -> tuple[dict | None, str]:
    """JSON 객체 하나만 받는다(앞뒤 공백·코드 울타리 하나만 허용)."""
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
        if t.startswith("json"):
            t = t[4:].strip()
    try:
        obj = json.loads(t)
    except ValueError:
        return None, "JSON 아님"
    if not isinstance(obj, dict):
        return None, "JSON 객체 아님"
    return obj, ""


def parse(text: str) -> tuple[dict | None, str]:
    """(값, 실패 이유). JSON 객체 하나만 받는다(앞뒤 공백·코드 울타리 하나만 허용)."""
    obj, err = _json_obj(text)
    if obj is None:
        return None, err
    stop, tp, reason = obj.get("stop_pct"), obj.get("tp_pct", "missing"), obj.get("reason")
    if isinstance(stop, bool) or not isinstance(stop, (int, float)):
        return None, "stop_pct 숫자 아님"
    if tp == "missing":
        return None, "tp_pct 없음"
    if tp is not None and (isinstance(tp, bool) or not isinstance(tp, (int, float))):
        return None, "tp_pct 숫자·null 아님"
    if not isinstance(reason, str) or not reason.strip():
        return None, "reason 없음"
    if not STOP_RANGE[0] <= stop <= STOP_RANGE[1]:
        return None, f"stop_pct {stop} 범위 밖({STOP_RANGE[0]}~{STOP_RANGE[1]})"
    if tp is not None and not TP_RANGE[0] <= tp <= TP_RANGE[1]:
        return None, f"tp_pct {tp} 범위 밖({TP_RANGE[0]}~{TP_RANGE[1]} 또는 null)"
    return {
        "stop_pct": float(stop),
        "tp_pct": None if tp is None else float(tp),
        "reason": reason.strip()[:120],
    }, ""


def call_claude(prompt: str, timeout: float = TIMEOUT, cwd: Path | None = None) -> str:
    """`claude -p` 한 번. 도구 없음, MCP 없음, 세션 저장 없음. 실패하면 예외."""
    cmd = [
        CLAUDE,
        "-p",
        prompt,
        "--model",
        MODEL,
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        "--disallowedTools",
        *BLOCKED_TOOLS,
        "--max-budget-usd",
        BUDGET_USD,
        "--no-session-persistence",
        "--output-format",
        "text",
    ]
    # 새 세션으로 띄워 시간이 넘으면 손자 프로세스까지 묶어 죽인다(출력 통로를 잡고 멈추지 않게)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.communicate()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"claude 종료 코드 {proc.returncode}: {err[-200:]}")
    return out


def decide(data: dict, caller: Callable[[str], str] = call_claude) -> dict:
    """AI 답과 실제로 쓸 값. 무엇이 잘못돼도 기본값(손절 3%, 익절 없음)으로 돌아온다."""
    t0 = time.monotonic()
    raw, err = "", ""
    try:
        head = PROMPT.replace("<SYMBOL>", str(data.get("symbol") or "자료의 symbol"))
        raw = caller(head + json.dumps(data, ensure_ascii=False))
        parsed, err = parse(raw)
    except subprocess.TimeoutExpired:
        parsed, err = None, f"시간 초과({TIMEOUT}초)"
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        parsed, err = None, f"호출 실패: {e}"
    secs = round(time.monotonic() - t0, 1)
    if parsed:
        used = {
            "stop_pct": parsed["stop_pct"],
            "tp_pct": parsed["tp_pct"],
            "reason": parsed["reason"],
        }
    else:
        used = {"stop_pct": DEFAULT_STOP, "tp_pct": None, "reason": f"기본값 사용: {err}"}
    return {
        "raw": raw[:1000],
        "parsed": parsed,
        "used": used,
        "fallback": err or None,
        "secs": secs,
    }


# ---------------------------------------------------------------- 종목 선택(신호 겹침)

CHOOSE_PROMPT = """너는 바이낸스 USDⓈ-M 선물 자동매매 엔진의 종목 선택 담당이다.
엔진은 거래량 상위 종목(최대 50개, 코인과 TradFi 주식·ETF·원자재)에 같은 신호(SCR 15분봉, 마감 봉)를
쓰고, 포지션은 한 번에 하나만 든다(계좌 전체, 레버리지 3배). 이번 봉에 신호가 겹쳤다. candidates는
우선순위(백테스트 승률 순) 위 몇 개만 들어 있다. options 중 하나만 고른다.
- keep: 들고 있는 포지션 유지. flip:종목: 그 종목을 반대로 갈아타기. close:종목: 그 종목 청산만.
- switch:종목: 들고 있는 포지션을 닫고 그 종목 신호 방향으로 진입. enter:종목: 그 종목 진입.
- none: 아무것도 안 함.
자료(candidates: 신호 난 종목, held: 들고 있는 포지션)만 근거로 쓴다. 갈아타면 수수료가 두 번 든다.
backtest_winrate_pct는 종목별 이 신호의 과거 1년 승률이다. 뚜렷한 이유가 없으면 승률이 높은
종목을 고른다.
reason: 왜 골랐는지 한국어 한 문장(80자 이내).

출력은 JSON 객체 하나만. 설명·코드 블록·다른 글자를 붙이지 않는다.
{"choice": "options의 키 하나", "reason": "..."}

자료:
"""


def brief_frames(klines: Callable[[str], list[list]], price: float, now_ms: int) -> dict:
    """선택용 작은 시세 요약: 4시간·일봉 TM 방향, 15분·일봉 ATR%, 4시간·일봉 20봉 고저 거리."""
    frames = {k: [r for r in klines(k) if int(r[6]) < now_ms] for k in ("15m", "4h", "1d")}
    return {
        "direction_tm": {k: _direction(frames[k], now_ms) for k in ("4h", "1d")},
        "atr14_pct": {k: _atr_pct(frames[k]) for k in ("15m", "1d")},
        "swing20": {k: _swing(frames[k], price) for k in ("4h", "1d") if len(frames[k]) >= 20},
    }


def parse_choice(text: str, options: list[str]) -> tuple[dict | None, str]:
    obj, err = _json_obj(text)
    if obj is None:
        return None, err
    choice, reason = obj.get("choice"), obj.get("reason")
    if not isinstance(choice, str) or choice not in options:
        return None, f"choice {choice!r}가 선택지에 없음"
    if not isinstance(reason, str) or not reason.strip():
        return None, "reason 없음"
    return {"choice": choice, "reason": reason.strip()[:120]}, ""


def choose(data: dict, options: list[str], caller: Callable[[str], str] = call_claude) -> dict:
    """신호가 겹칠 때 고른 선택지. 답이 없거나 틀리면 choice None(엔진이 기본 규칙을 쓴다)."""
    t0 = time.monotonic()
    raw, err = "", ""
    try:
        raw = caller(CHOOSE_PROMPT + json.dumps(data, ensure_ascii=False))
        parsed, err = parse_choice(raw, options)
    except subprocess.TimeoutExpired:
        parsed, err = None, f"시간 초과({CHOOSE_TIMEOUT}초)"
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        parsed, err = None, f"호출 실패: {e}"
    return {
        "raw": raw[:1000],
        "choice": parsed["choice"] if parsed else None,
        "reason": parsed["reason"] if parsed else "",
        "fallback": err or None,
        "secs": round(time.monotonic() - t0, 1),
    }
