"""매일 관심 종목에 검증된 규칙을 적용해 화면용 JSON을 만든다.

사용: python -m minerva.strategy.daily [--out PATH] [--notify]
- 확정된 일봉만 쓴다 (장중·당일 미마감 봉은 버림).
- 규칙: momentum-tm §2 TM 눌림 롱, vol-regime §2-1 공포 초입 매수,
  §2 레짐 비중 한도.
- 대상: 코스피 지수·시총 상위 100, S&P500·나스닥100 선물·나스닥 시총 상위 10, 비트코인,
  KODEX 200. 종목별 검증 여부는 자기 차트 백테스트로 자동 판정한다(`verify`).
- `actions`: 오늘 낼 주문(롱 진입·롱 청산). 자동매매가 읽는 목록이다. 숏은 하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from minerva.strategy.backtest import (
    Context,
    Rule,
    benjamini_hochberg,
    evaluate,
    random_timing_pvalue,
    run,
)
from minerva.strategy.indicators import (
    Bar,
    TMState,
    ma_snapshot,
    resample,
    sma,
    stoch_fast_k,
    tm_states,
)
from minerva.strategy.regime import (
    gated_regime,
    htf_direction,
    vix_regimes,
    vix_weekly,
    weekly_gate_at,
)
from minerva.strategy.report import CACHE_DIR, fetch_yahoo
from minerva.strategy.universe import kospi_top, nasdaq_top

OUT = Path(os.environ.get("MINERVA_DAILY_OUT", "~/.minerva/daily/latest.json")).expanduser()

TM_RULE = Rule(level=1, above_ma=50, hold=20, stop=0.10, vix="calm", align=True)
FEAR_LO, FEAR_HI, FEAR_HOLD = 0.05, 0.15, 60
NOTE = (
    "검증됨 = 그 종목 자기 차트로 백테스트해 거래 30회 이상, 표본 안·밖 우위가 모두 + 인 경우 "
    "(주 1회 재검증). 종목이 많아 우연히 통과하는 것도 섞일 수 있으니 비중은 작게. "
    "투자 판단과 책임은 본인에게."
)


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str
    market: str  # US / KR / CRYPTO
    kind: str = "base"  # base / etf(지수 ETF)
    underlying: str = ""
    mult: int = 1
    held_only: bool = False  # 순위 밖 봇 보유 종목: 청산 신호만 받고 새로 사지 않는다


# 사용자 결정 2026-10-31: 분석·검증·신규 매수 모두 시총 상위 100 (200 → 100).
# 밖으로 밀린 봇 보유 종목은 청산 신호를 받도록 분석에만 남긴다 (`watchlist`)
KOSPI_TOP = 100

CORE = [
    Asset("^KS11", "코스피", "KR"),
    Asset("SPY", "S&P500 (SPY)", "US"),
    Asset("NQ=F", "나스닥100 E-mini 선물", "US"),
    Asset("BTC-USD", "비트코인", "CRYPTO"),
]


def held_outside(top: list[tuple[str, str]], owned: dict) -> list[Asset]:
    """순위 밖으로 밀린 봇 보유 종목. 빠뜨리면 청산 신호가 끊긴다."""
    known = {s for s, _ in top} | {p.symbol for p in PRODUCTS}
    held = [(f"{code}.KS", p.get("name", code)) for code, p in owned.items()]
    return [Asset(s, n, "KR", held_only=True) for s, n in held if s not in known]


def watchlist() -> list[Asset]:
    top = kospi_top(KOSPI_TOP)
    kr = [Asset(s, n, "KR") for s, n in top]
    us = [Asset(s, n, "US") for s, n in nasdaq_top(10)]
    try:
        held = held_outside(top, read_owned())
    except Exception as exc:  # noqa: BLE001 — 보유 파일이 깨져도 상위 종목의 청산 신호는 나가야 한다
        print(f"보유 목록을 못 읽음: {exc}", file=sys.stderr)
        from minerva import notify

        notify.send(f"⚠️ 봇 보유 목록을 못 읽었어요 ({type(exc).__name__}). 순위 밖 청산 확인 필요")
        held = []
    return CORE + kr + held + us


# 사용자 방침: 숏·인버스·레버리지는 매매도 화면도 계산도 하지 않는다 (2026-10-30).
# 코스피 숏 규칙이 무작위 타이밍 검정을 못 넘었다(p≈0.17, 2010년 이후 거래당 −0.03%).
PRODUCTS = [
    # 지수 롱: 코스피 TM 눌림 규칙(거래 96회, 무작위 대비 p≈0.015)을 KODEX 200으로 실행한다
    Asset("069500.KS", "KODEX 200", "KR", "etf", "^KS11", 1),
]

VERIFY_TTL = 7 * 24 * 3600


def verify_ttl(symbol: str) -> int:
    """종목마다 4~7일. 한꺼번에 만료되면 그날 분석이 몇 배로 느려진다."""
    return VERIFY_TTL - zlib.crc32(symbol.encode()) % 4 * 24 * 3600


# 검증 캐시 버전: 규칙·관문·자료 처리를 바꾸면 올린다. 다르면 TTL과 상관없이 다시 계산한다.
# 2: 국내 가짜 채움 봉 제거 (2026-10-30). 숏은 계산하지 않는다
VERIFY_VERSION = 2
VERIFY_MIN_TRADES = 30
# 다중 검정: 무작위 타이밍 대비 p값에 거짓 발견 비율(FDR) 20%를 건다 (2026-10-28 추천안).
# 0.10이면 2026-10 기준 통과 종목이 없다. 바꾸면 통과 종목이 크게 달라진다.
FDR_Q = 0.20
CORE_ALPHA = 0.05
# 왕복 거래비용 (수수료 + 세금 + 체결 미끄러짐). 우위가 이 비용을 넘어야 "검증됨"이다.
# 국내 주식: 수수료 0.03% + 매도 거래세 0.20% + 미끄러짐 0.07%. ETF·지수는 거래세가 없다.
COST_KR_STOCK = 0.003
COST_ETF = 0.001
COST_CRYPTO = 0.002


def round_trip_cost(symbol: str, market: str) -> float:
    if market == "CRYPTO":
        return COST_CRYPTO
    if market == "KR" and not symbol.startswith("^"):
        return COST_KR_STOCK
    return COST_ETF  # 지수·선물·ETF


def with_cost(v: dict, cost: float) -> dict:
    """캐시된 판정(비용 전 우위)에 비용을 빼고 다시 판정한다."""
    net = {
        k: None if v.get(k) is None else round(v[k] - cost, 4)
        for k in ("edge", "edge_in", "edge_out")
    }
    n = v.get("trades", 0)
    if n >= VERIFY_MIN_TRADES and all(
        e is not None and e > 0 for e in (net["edge_in"], net["edge_out"])
    ):
        label = "검증됨"
    elif n >= VERIFY_MIN_TRADES and all(e is not None and e <= 0 for e in net.values()):
        label = "부적합"
    else:
        label = "미검증"
    return {
        **v,
        "label": label,
        "cost": cost,
        "net_edge": net["edge"],
        "net_edge_in": net["edge_in"],
        "net_edge_out": net["edge_out"],
        "gross_label": v.get("label"),
    }


def _verdict(res: dict) -> dict:
    a, i, o = res["all"], res["in_sample"], res["out_of_sample"]
    n = a.get("trades", 0)
    edges = (a.get("edge"), i.get("edge"), o.get("edge"))
    if n >= VERIFY_MIN_TRADES and all(e is not None and e > 0 for e in edges[1:]):
        label = "검증됨"
    elif n >= VERIFY_MIN_TRADES and all(e is not None and e <= 0 for e in edges):
        label = "부적합"
    else:
        label = "미검증"
    return {
        "label": label,
        "trades": n,
        "win_rate": a.get("win_rate"),
        "edge": a.get("edge"),
        "edge_in": i.get("edge"),
        "edge_out": o.get("edge"),
    }


def verify(symbol: str, bars: list[Bar], ctx: Context) -> dict:
    """그 종목 자기 차트로 롱 규칙을 백테스트한다. 결과는 주 1회만 다시 계산한다."""
    import time

    path = CACHE_DIR / "verify" / f"{symbol.replace('^', '_').replace('=', '_')}.json"
    if path.exists() and time.time() - path.stat().st_mtime < verify_ttl(symbol):
        out = json.loads(path.read_text())
        if out.get("version") == VERIFY_VERSION:
            return out
    out = {
        "long": {**_verdict(evaluate(bars, TM_RULE, ctx)), "p_value": timing_pvalue(bars, ctx)},
        "asof": bars[-1].day.isoformat(),
        "version": VERIFY_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # 상품 두 개가 같은 기초자산(^KS11)을 동시에 쓸 수 있어 임시 파일에 쓰고 바꾼다
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    os.replace(tmp, path)
    return out


def timing_pvalue(bars: list[Bar], ctx: Context) -> float | None:
    """롱 규칙 타이밍이 무작위보다 나은지 (p). 후보만 정밀하게 다시 잰다 (시간 절약)."""
    p = random_timing_pvalue(bars, TM_RULE, ctx, n_perm=300)
    if p is not None and p < 0.15:
        p = random_timing_pvalue(bars, TM_RULE, ctx, n_perm=3000)
    return None if p is None else round(p, 4)


def apply_fdr(assets: list[dict], q: float | None = None) -> set[str]:
    """종목 수십 개를 한꺼번에 검정하면 우연 통과가 섞인다. 롱 규칙 p값에 거짓 발견 비율 q를
    걸고, 통과하지 못한 "검증됨"은 "미검증"으로 내린다. 통과한 심볼을 돌려준다."""
    q = FDR_Q if q is None else q
    core = {c.symbol for c in CORE}
    pv = {
        a["symbol"]: a["verify"]["long"].get("p_value")
        for a in assets
        if "error" not in a
        and a["symbol"] not in core
        and a["verify"]["long"].get("trades", 0) >= VERIFY_MIN_TRADES
    }
    passed = benjamini_hochberg(pv, q)
    for a in assets:
        if "error" in a:
            continue
        v = a["verify"]["long"]
        if a["symbol"] in core:
            # 지수는 규칙을 처음 만든 대상이라 탐색 묶음에 넣지 않고 단독 검정(p<0.05)만 본다
            v["fdr_pass"] = (v.get("p_value") or 1) < CORE_ALPHA
        else:
            v["fdr_pass"] = a["symbol"] in passed
        if v["label"] == "검증됨" and not v["fdr_pass"]:
            v["label"] = "미검증"
            v["why"] = f"다중 검정 미통과 (p={v.get('p_value')}, q={q})"
    return passed


def trade_family(assets: list[dict]) -> list[dict]:
    """신규 매수가 가능한 시총 상위 종목. 분석 실패 종목도 순위에 넣어 `make_actions`와 맞춘다."""
    stocks = [a for a in assets if a["symbol"].endswith(".KS") and not a.get("held_only")]
    return stocks[:KOSPI_TOP]


def apply_fdr_by_family(assets: list[dict], q: float | None = None) -> None:
    """검정 묶음을 매수 범위와 맞춘다. 미국 종목 등이 기준선을 올리지 않게 나머지는 따로 건다.

    사용자 재량 결정 2026-10-31: `docs/backtests/2026-10-31-bh-scope-hypotheses.md`의 E1.
    사전 등록은 통과하지 못했고, 수익 차이는 잡음 수준이다.
    """
    family = {a["symbol"] for a in trade_family(assets)}
    apply_fdr([a for a in assets if a["symbol"] in family], q)
    apply_fdr([a for a in assets if a["symbol"] not in family], q)


MARKET_CLOSE = {
    "KR": (ZoneInfo("Asia/Seoul"), 15, 40),
    "US": (ZoneInfo("America/New_York"), 16, 10),
    "CRYPTO": (timezone.utc, 0, 0),
}


def is_filler(b: Bar) -> bool:
    """야후가 거래 없는 날을 전날 종가로 채운 가짜 봉 (거래량 0, 시가=고가=저가=종가)."""
    return b.volume == 0 and b.open == b.high == b.low == b.close


def completed(bars: list[Bar], market: str, now: datetime | None = None) -> list[Bar]:
    """마지막 봉이 아직 마감 전(당일 진행 중)이면 버린다. 국내는 가짜 채움 봉도 버린다.

    채움 봉은 실제로 거래된 적 없는 가격이라 지표 창과 백테스트 체결가를 왜곡한다
    (예: 2022-02-08·09 국내 약 180종목). 미국 VIX처럼 원래 거래량이 0인 자료는 그대로 둔다.
    """
    if market == "KR":
        bars = [b for b in bars if not is_filler(b)]
    if not bars:
        return bars
    tz, hh, mm = MARKET_CLOSE[market]
    local = (now or datetime.now(timezone.utc)).astimezone(tz)
    if market == "CRYPTO":
        return bars[:-1] if bars[-1].day >= local.date() else bars
    closed = (local.hour, local.minute) >= (hh, mm)
    if bars[-1].day > local.date() or (bars[-1].day == local.date() and not closed):
        return bars[:-1]
    return bars


EXIT_REASON = {"take": "익절 조건", "stop": "손절가 도달", "time": "20거래일 만료"}


def _r(v: float | None, n: int = 2) -> float | None:
    return None if v is None else round(v, n)


def _week_end(d: date) -> date:
    """d 다음에 오는 주봉 확정일(금요일). d가 금요일이면 다음 주 금요일."""
    return d + timedelta(days=(4 - d.weekday()) % 7 or 7)


def _month_end(d: date) -> date:
    nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def wait_levels(bars: list[Bar], side: str, gate: dict | None = None) -> dict[str, str]:
    """빠질 수 있는 조건별로 '어느 가격까지·언제까지'를 문장으로 만든다.

    Stoch 가격은 최근 5봉 고저가 그대로라는 가정의 근사치다.
    """
    closes = [b.close for b in bars]
    last = bars[-1]
    win = bars[-5:]
    hh, ll = max(b.high for b in win), min(b.low for b in win)
    ma50 = sma(closes, 50)[-1]
    wk = resample(bars, "W")
    mo = resample(bars, "M")
    w20 = sma([b.close for b in wk], 20)[-1]
    m10 = sma([b.close for b in mo], 10)[-1]
    fri = _week_end(last.day).strftime("%m/%d")
    eom = _month_end(last.day).strftime("%m/%d")
    f = lambda v: "-" if v is None else f"{v:,.2f}"  # noqa: E731
    if side == "long":
        return {
            "L1 신호": f"{f(ll + 0.2 * (hh - ll))} 이하로 눌렸다가 반등하는 날 (Stoch 20 재돌파)",
            "50일선 위": f"종가 {f(ma50)} 위로 회복",
            "VIX 주봉 5<10": (
                f"VIX 주봉 5주선 {f((gate or {}).get('w5'))}이 10주선 "
                f"{f((gate or {}).get('w10'))} 아래로 마감 (다음 확정 {fri})"
            ),
            "주봉 20주선 위": f"주봉 종가 {f(w20)} 위로 마감 (다음 확정 {fri})",
            "월봉 10개월선 위": f"월봉 종가 {f(m10)} 위로 마감 (다음 확정 {eom})",
        }
    return {
        "Stoch 80 하향 이탈": f"{f(ll + 0.8 * (hh - ll))} 이상으로 반등했다가 꺾이는 날",
        "50일선 아래": f"종가 {f(ma50)} 아래로 이탈",
        "주봉 20주선 아래": f"주봉 종가 {f(w20)} 아래로 마감 (다음 확정 {fri})",
        "월봉 10개월선 아래": f"월봉 종가 {f(m10)} 아래로 마감 (다음 확정 {eom})",
    }


def tm_status(bars: list[Bar], ctx: Context) -> dict:
    """TM 눌림 규칙의 오늘 상태: 보유 중 / 진입 신호 / 대기(빠진 조건)."""
    i = len(bars) - 1
    last = bars[i]
    closes = [b.close for b in bars]
    ma50 = sma(closes, 50)[i]
    states = tm_states(bars)
    conds = {
        "L1 신호": states[i].long_level >= 1 and (i == 0 or states[i - 1].long_level < 1),
        "50일선 위": ma50 is not None and last.close > ma50,
        "VIX 주봉 5<10": ctx.regime.get(last.day) in ("R0", "R1"),
        "주봉 20주선 위": ctx.weekly_up.get(last.day, False),
        "월봉 10개월선 위": ctx.monthly_up.get(last.day, False),
    }
    trades = run(bars, TM_RULE, ctx)
    if trades and trades[-1].reason != "open" and trades[-1].exit_day == last.day:
        t = trades[-1]
        return {
            "state": "청산 신호",
            "entry_day": t.entry_day.isoformat(),
            "reason": EXIT_REASON.get(t.reason, t.reason),
            "exit_px": _r(t.exit),
            "pnl": _r(t.ret, 4),
            "exit": "손절은 장중 체결" if t.reason == "stop" else "다음 거래일 시가 청산",
            "conditions": conds,
        }
    if trades and trades[-1].reason == "open":
        t = trades[-1]
        return {
            "state": "보유 중",
            "entry_day": t.entry_day.isoformat(),
            "entry": _r(t.entry),
            "stop": _r(t.entry * (1 - TM_RULE.stop)),
            "pnl": _r(last.close / t.entry - 1, 4),
            "days_left": TM_RULE.hold - t.bars,
            "exit_rule": "Stoch 80 하향 이탈 시 익절 / 손절가 / 기간 만료 중 먼저",
            "conditions": conds,
        }
    if all(conds.values()):
        return {
            "state": "진입 신호",
            "entry": "다음 거래일 시가",
            "ref_price": _r(last.close),
            "stop": _r(last.close * (1 - TM_RULE.stop)),
            "max_days": TM_RULE.hold,
            "exit_rule": "Stoch 80 하향 이탈 시 익절 / 진입가 −10% 손절 / 20거래일",
            "conditions": conds,
        }
    return {
        "state": "대기",
        "missing": [k for k, ok in conds.items() if not ok],
        "conditions": conds,
    }


def fear_status(vix: list[Bar], vix3m: list[Bar]) -> dict:
    """공포 초입: VIX/VIX3M−1이 5% 미만에서 5~15%로 처음 들어선 날 → 60거래일."""
    m3 = {b.day: b.close for b in vix3m}
    prem = [(b.day, b.close / m3[b.day] - 1) for b in vix if b.day in m3 and m3[b.day] > 0]
    signal_days = [
        d for (_, p0), (d, p1) in zip(prem, prem[1:]) if p0 < FEAR_LO and FEAR_LO <= p1 < FEAR_HI
    ]
    days = [d for d, _ in prem]
    today, premium = prem[-1] if prem else (None, None)
    out = {"date": today and today.isoformat(), "premium": _r(premium, 4), "state": "없음"}
    if signal_days:
        last_sig = signal_days[-1]
        elapsed = len(days) - 1 - days.index(last_sig)
        out["last_signal"] = last_sig.isoformat()
        if elapsed == 0:
            out["state"] = "신호 발생 (내일부터 지수 ETF 3분할 매수)"
        elif elapsed < FEAR_HOLD:
            out["state"] = f"보유 기간 중 ({FEAR_HOLD - elapsed}거래일 남음)"
    return out


SIZE_BY_REGIME = {"R0": "계획대로", "R1": "계획대로", "R2": "계획의 1/2", "R3": "계획의 1/3"}


def analyze(asset: Asset, vix: list[Bar], gate: dict | None = None) -> dict:
    daily = completed(fetch_yahoo(asset.symbol, "max"), asset.market)
    ctx = _ctx(vix, daily)
    last = daily[-1]
    snap = ma_snapshot([b.close for b in daily])
    st = tm_states(daily)[-1]
    raw = verify(asset.symbol, daily, ctx)
    cost = round_trip_cost(asset.symbol, asset.market)
    ver = {"long": with_cost(raw["long"], cost)}
    return {
        "symbol": asset.symbol,
        "name": asset.name,
        "market": asset.market,
        "verify": ver,
        "date": last.day.isoformat(),
        "close": _r(last.close),
        "change": _r(last.close / daily[-2].close - 1, 4) if len(daily) > 1 else None,
        "stoch": _r(stoch_fast_k(daily)[-1], 1),
        "cci": _r(st.cci, 1),
        "rsi": _r(st.rsi, 1),
        "ma50": _r(snap["ma"].get(50)),
        "ma200": _r(snap["ma"].get(200)),
        "adv": round(sum(b.close * b.volume for b in daily[-20:]) / len(daily[-20:])),
        "tm": _with_wait(tm_status(daily, ctx), wait_levels(daily, "long", gate)),
    }


def _ctx(vix: list[Bar], daily: list[Bar]) -> Context:
    return Context(
        regime=_regime_by_day(vix, daily),
        weekly_up=htf_direction(daily, "W", 20),
        monthly_up=htf_direction(daily, "M", 10),
    )


def product_verdict(raw: dict) -> dict:
    """상품 판정 = 기초자산 규칙의 비용 후 판정 + 단독 검정(p < CORE_ALPHA).

    상품은 탐색 묶음(BH)에 들지 않으므로 지수 자체와 같은 단독 기준을 건다. p값이 없으면
    검증됨으로 두지 않는다.
    """
    v = with_cost(raw, COST_ETF)
    p = raw.get("p_value")
    v["fdr_pass"] = p is not None and p < CORE_ALPHA
    if v["label"] == "검증됨" and not v["fdr_pass"]:
        v["label"] = "미검증"
        v["why"] = f"단독 검정 미통과 (p={p}, 기준 {CORE_ALPHA})"
    return v


def product_status(p: Asset, vix: list[Bar], names: dict[str, str]) -> dict:
    """기초자산(코스피) 롱 규칙 신호를 지수 ETF로 실행한다."""
    daily = completed(fetch_yahoo(p.symbol, "1y"), p.market)
    last = daily[-1]
    under = completed(fetch_yahoo(p.underlying, "max"), p.market)
    ctx = _ctx(vix, under)
    raw = verify(p.underlying, under, ctx)
    if under[-1].day < last.day:
        # 기초자산 자료가 상품보다 늦으면 어제 신호를 오늘 날짜로 다시 내게 된다
        tm = {"state": "대기", "missing": [f"기초자산 자료 지연 ({under[-1].day})"]}
    else:
        tm = tm_status(under, ctx)
        if tm.get("state") == "진입 신호":
            tm["stop"] = _r(last.close * (1 - TM_RULE.stop))  # 손절은 ETF 매입가 기준
    side = "long"
    # 주문 수량은 상품 가격으로 계산해야 한다 (기초자산 가격이 아니다)
    if "ref_price" in tm:
        tm["ref_price"] = _r(last.close)
    tm["wait"] = [f"[{names.get(p.underlying, p.underlying)}] {w}" for w in tm.get("wait", [])]
    return {
        "symbol": p.symbol,
        "name": p.name,
        "market": p.market,
        "kind": p.kind,
        "mult": p.mult,
        "underlying": names.get(p.underlying, p.underlying),
        "verify": {side: product_verdict(raw[side])},
        "date": last.day.isoformat(),
        "close": _r(last.close),
        "change": _r(last.close / daily[-2].close - 1, 4) if len(daily) > 1 else None,
        "tm": tm,
    }


def _with_wait(tm: dict, levels: dict[str, str]) -> dict:
    if tm.get("state") == "대기":
        tm["wait"] = [f"{k}: {levels[k]}" for k in tm.get("missing", []) if k in levels]
    return tm


def _regime_by_day(vix: list[Bar], daily: list[Bar]) -> dict[date, str]:
    """자산 거래일마다 그날 장 마감 때 알 수 있던 VIX 레짐 (매수 허용 = R0·R1).

    같은 날짜의 미국 마감은 한국 마감 뒤(KST 새벽)에 나오므로 전날까지만 쓴다.
    매수 허용은 VIX 주봉 5주선 < 10주선이 정한다 (사용자 결정 2026-10-28, 블로그 방식).
    비중 계수도 이 값을 따른다 (`gated_regime` 참고).
    """
    reg, days, weekly, keys = _vix_tables(vix)
    out, k = {}, -1
    for b in daily:
        while k + 1 < len(days) and days[k + 1] < b.day:
            k += 1
        if k >= 0:
            out[b.day] = gated_regime(reg[days[k]], weekly_gate_at(weekly, b.day, keys)[2])
    return out


_VIX_CACHE: dict = {}


def _vix_tables(vix: list[Bar]) -> tuple:
    """종목 200여 개가 같은 VIX로 레짐을 구하므로 한 번만 계산한다."""
    key = (len(vix), vix[-1].day if vix else None, vix[-1].close if vix else None)
    hit = _VIX_CACHE.get(key) if vix else None  # 빈 VIX(테스트)는 캐시하지 않는다
    if hit is None:
        reg = vix_regimes(vix)
        weekly = vix_weekly(vix)
        hit = (reg, sorted(reg), weekly, [w[0] for w in weekly])
        # 여러 스레드가 동시에 처음 부르면 다른 스레드의 clear()에 지워질 수 있어 지역값을 돌려준다
        _VIX_CACHE.clear()
        _VIX_CACHE[key] = hit
    return hit


MONTHLY_MIN = 60  # 이보다 짧으면 이전 신호를 놓쳐 제한 상태를 알 수 없다
MONTHLY_GAP = 7  # 달력일. 긴 연휴보다 길게 비면 자료가 끊긴 것으로 본다


def monthly_touches(bars: list[Bar]) -> set[date]:
    """월봉 저가가 20개월선에 닿은 달. 깊게 빠졌다고 보고 제한을 푼다 (사용자 결정 2026-10-31)."""
    months = resample(bars, "M")
    ma = sma([b.close for b in months], 20)
    return {b.day for b, m in zip(months, ma) if m is not None and b.low <= m}


def block_spans(
    months: list[TMState], touches: frozenset[date] | set[date] = frozenset()
) -> list[tuple[date, date | None]]:
    """코스피 월봉 3단 과매수 이탈부터 해제 달까지. None은 진행 중.

    해제: 월봉 저가의 20개월선 터치, 또는 CCI<0 또는 RSI<50을 거쳐 회복.
    """
    spans: list[tuple[date, date | None]] = []
    since, dipped = None, False
    for prev, cur in zip(months, months[1:]):
        if since is None:
            if cur.short_level == 3 and prev.short_level < 3:
                since, dipped = cur.day, False
        elif cur.day in touches and cur.short_level < 3:  # 새 이탈이 나온 달은 풀지 않는다
            spans.append((since, cur.day))
            since = None
        elif (cur.cci or 0) < 0 or (cur.rsi or 100) < 50:
            dipped = True
        elif dipped:
            spans.append((since, cur.day))
            since = None
    return spans + [(since, None)] if since else spans


def closed_block(why: str) -> dict:
    """상태를 모르면 신규 매수를 막는다. 청산은 계속 나간다."""
    return {"blocked": True, "since": None, "error": why}


def monthly_block(bars: list[Bar], today: date) -> dict:
    """사용자 결정 2026-10-31. 근거: `docs/backtests/2026-10-31-monthly-block-hypotheses.md`."""
    if not bars:
        return closed_block("자료 없음")
    last = bars[-1].day
    month_end = (last.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    months_behind = (today.year - last.year) * 12 + today.month - last.month
    if months_behind > 1 or (min(today, month_end) - last).days > MONTHLY_GAP:
        return closed_block(f"자료 끊김 {last}")  # 끊긴 달을 마감된 달로 오해하지 않게
    months = [
        s
        for s in tm_states(resample(bars, "M"))
        if (s.day.year, s.day.month) != (today.year, today.month)
    ]
    if len(months) < MONTHLY_MIN:
        return closed_block(f"자료 부족 {len(months)}개월")
    spans = block_spans(months, monthly_touches(bars))
    since = spans[-1][0] if spans and spans[-1][1] is None else None
    if since is None:
        return {"blocked": False, "since": None}
    return {
        "blocked": True,
        "since": since.isoformat(),
        "dipped": any((s.cci or 0) < 0 or (s.rsi or 100) < 50 for s in months if s.day > since),
        "ma20": _r(sma([b.close for b in resample(bars, "M")], 20)[-1]),
    }


def build() -> dict:
    vix = completed(fetch_yahoo("^VIX", "max"), "US")
    vix3m = completed(fetch_yahoo("^VIX3M", "max"), "US")
    raw = vix_regimes(vix).get(vix[-1].day)
    vix_ma50 = sma([b.close for b in vix], 50)[-1]
    today = datetime.now(timezone(timedelta(hours=9))).date()
    w5, w10, calm = weekly_gate_at(vix_weekly(vix), today)
    regime = gated_regime(raw, calm)
    gate = {"w5": _r(w5), "w10": _r(w10), "open": calm}
    try:
        block = monthly_block(completed(fetch_yahoo("^KS11", "max"), "KR"), today)
    except Exception as exc:  # noqa: BLE001 — 지수 조회 실패가 청산 신호까지 막지 않게
        block = closed_block(type(exc).__name__)
    universe = watchlist()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {a.symbol: pool.submit(analyze, a, vix, gate) for a in universe}
    assets = []
    for a in universe:
        try:
            assets.append(futures[a.symbol].result())
        except Exception as exc:  # noqa: BLE001 — 한 종목 실패가 전체를 막지 않게
            assets.append({"symbol": a.symbol, "name": a.name, "error": type(exc).__name__})
    held = {a.symbol for a in universe if a.held_only}
    for a in assets:
        if a["symbol"] in held:
            a["held_only"] = True
    apply_fdr_by_family(assets)
    names = {a["symbol"]: a["name"] for a in assets} | {"QQQ": "나스닥100", "SOXX": "반도체"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pfut = {p.symbol: pool.submit(product_status, p, vix, names) for p in PRODUCTS}
    products = []
    for p in PRODUCTS:
        try:
            products.append(pfut[p.symbol].result())
        except Exception as exc:  # noqa: BLE001
            products.append({"symbol": p.symbol, "name": p.name, "error": type(exc).__name__})
    kst = datetime.now(timezone(timedelta(hours=9)))
    return {
        "generated_at": kst.isoformat(timespec="minutes"),
        "vix": {
            "date": vix[-1].day.isoformat(),
            "close": _r(vix[-1].close),
            "regime": regime,
            "raw_regime": raw,
            "size_limit": SIZE_BY_REGIME.get(regime, "?"),
            "ma50": _r(vix_ma50),
            "gate": gate,
        },
        "fear": fear_status(vix, vix3m),
        "market_block": block,
        "actions": make_actions(assets, products, block["blocked"]),
        "assets": assets,
        "products": products,
        "note": NOTE,
    }


ACTION_LABEL = {
    "LONG_ENTRY": "롱 진입",
    "LONG_EXIT": "롱 청산",
}


def _action(a: dict, side: str, st: dict, code: str) -> dict:
    v = a.get("verify", {}).get(side, {})
    out = {
        "action": code,
        "label": ACTION_LABEL[code],
        "symbol": a["symbol"],
        "name": a["name"],
        "market": a["market"],
        "verified": v.get("label") == "검증됨",
        "verify": v.get("label", "미검증"),
        # 자금이 모자랄 때 우선순위 (minerva.trade): p값 낮은 순 → 비용 후 우위 큰 순
        "p_value": v.get("p_value"),
        "net_edge": v.get("net_edge"),
        "date": a["date"],
    }
    if code == "LONG_ENTRY":
        out.update(
            when="다음 거래일 시가",
            ref_price=st.get("ref_price"),
            stop=st.get("stop_px", st.get("stop")),
            stop_on=a["symbol"],
            take="Stoch 80 하향 이탈 종가",
            max_days=20,
        )
    else:
        out.update(
            when=st.get("exit"),
            reason=st.get("reason"),
            exit_px=st.get("exit_px"),
            pnl=st.get("pnl"),
        )
    return out


# 롱만 한다. 숏·인버스는 보여주지도 계산하지도 않는다 (사용자 결정 2026-10-30).
ASSET_SIDES = ("long",)


def make_actions(assets: list[dict], products: list[dict], blocked: bool = False) -> list[dict]:
    """오늘 낼 주문 목록. 검증된 것을 앞에 둔다. blocked면 청산만 낸다."""
    out = []
    # 국내 종목 중 마지막 봉이 다른 종목보다 오래된 것(거래정지·자료 지연)은 묵은 신호라 뺀다
    kr_days = [x["date"] for x in assets + products if x.get("market") == "KR" and "date" in x]
    latest = max(kr_days) if kr_days else ""
    # 분석 실패 종목은 market이 없어서 심볼로 센다. 봇 보유분의 청산은 순위와 무관하게 따른다
    ranks = {x["symbol"]: i for i, x in enumerate(trade_family(assets))}
    for a in assets:
        # 지수는 직접 살 수 없다. 코스피 신호는 KODEX 200(상품)으로만 주문한다
        if "error" in a or a["symbol"].startswith("^"):
            continue
        if a.get("market") == "KR" and a.get("date", latest) < latest:
            continue
        if a["tm"]["state"] == "진입 신호":
            if not blocked and (a["symbol"] in ranks or not a["symbol"].endswith(".KS")):
                out.append(_action(a, "long", a["tm"], "LONG_ENTRY"))
        elif a["tm"]["state"] == "청산 신호":
            out.append(_action(a, "long", a["tm"], "LONG_EXIT"))
    for p in products:
        if "error" in p:
            continue
        if p["tm"]["state"] == "진입 신호":
            if not blocked:
                out.append(_action(p, "long", p["tm"], "LONG_ENTRY"))
        elif p["tm"]["state"] == "청산 신호":
            out.append(_action(p, "long", p["tm"], "LONG_EXIT"))
    return sorted(out, key=lambda x: (not x["verified"], x["action"]))


def _holds(data: dict) -> list[tuple[dict, str, dict]]:
    out = []
    for a in data["assets"]:
        for side in ASSET_SIDES:
            st = a.get("tm", {})
            if st.get("state") == "보유 중":
                out.append((a, side, st))
    return out


def _near(data: dict, n: int = 5) -> list[tuple[dict, str, dict]]:
    """검증된 규칙 중 빠진 조건이 1개뿐인 후보."""
    out = []
    for a in data["assets"]:
        for side in ASSET_SIDES:
            st = a.get("tm", {})
            ok = a.get("verify", {}).get(side, {}).get("label") == "검증됨"
            if ok and st.get("state") == "대기" and len(st.get("missing", [])) == 1:
                out.append((a, side, st))
    return out[:n]


SIDE = {"long": "롱"}


def _action_line(x: dict) -> str:
    tag = "✅" if x["verified"] else "⚠️미검증"
    if "stop" in x:
        return (
            f"{x['label']} {x['name']} [{tag}] {x['when']} · 손절 {x['stop']} · "
            f"익절 {x['take']} · 최대 {x['max_days']}일"
        )
    pnl = "" if x.get("pnl") is None else f" ({x['pnl']:+.1%})"
    return f"{x['label']} {x['name']} [{tag}] {x['reason']}{pnl} · {x['when']}"


def write_summary(data: dict) -> list[str]:
    """분석 JSON만 근거로 쓴 요약문 (숫자는 모두 data에서 온다)."""
    v = data["vix"]
    out = []
    g = v.get("gate") or {}
    if v["regime"] in ("R0", "R1"):
        out.append(
            f"VIX 주봉 5주선({g.get('w5')})이 10주선({g.get('w10')}) 아래라 롱 규칙이 열려 있어요."
        )
    else:
        out.append(
            f"VIX 주봉 5주선({g.get('w5')})이 10주선({g.get('w10')}) 위라 롱 신규 진입은 막혀 "
            f"있어요. 주봉이 5<10으로 마감하면(금요일) 다음 주에 열려요."
        )
    f = data["fear"]
    if f.get("state", "없음") != "없음":
        out.append(f"공포 초입 매수: {f['state']}.")
    block = data.get("market_block") or {}
    if block.get("blocked"):
        if block.get("since"):
            out.append(f"코스피 월봉 3단 이탈({block['since'][:7]})로 신규 매수 제한 중이에요.")
        else:
            out.append(f"코스피 자료 문제로 신규 매수를 막았어요 ({block.get('error')}).")
    acts = data.get("actions", [])
    if acts:
        ok = [x for x in acts if x["verified"]]
        out.append(f"오늘 주문 {len(acts)}건 (검증된 규칙 {len(ok)}건).")
    else:
        out.append("오늘 새로 낼 주문은 없어요.")
    holds = _holds(data)
    if holds:
        items = [f"{a['name']} {SIDE[side]}({st['days_left']}일 남음)" for a, side, st in holds]
        out.append("보유 중: " + ", ".join(items))
    for a, side, st in _near(data):
        out.append(f"{a['name']} {SIDE[side]} 임박: " + " / ".join(st.get("wait") or st["missing"]))
    return out


def read_owned() -> dict:
    from minerva import trade

    path = trade.TRADE_DIR / "positions.json"
    return json.loads(path.read_text()) if path.exists() else {}


def trade_memo(data: dict, today: date) -> list[str]:
    """신규 매수가 왜 언제까지 막히는지. 막힌 게 없으면 비어 있다 (사용자 요청 2026-10-31)."""
    out = []
    block = data.get("market_block") or {}
    if block.get("blocked") and block.get("since"):
        # 주말이면 이번 달 마감은 이미 끝났으니 다음 영업일의 달을 본다
        cur = today + timedelta(days=7 - today.weekday() if today.weekday() >= 5 else 0)
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        after = (nxt.replace(day=28) + timedelta(days=4)).replace(day=1)
        ma = f"{block['ma20']:,.0f}"
        head = f"코스피 월봉 3단 이탈({block['since'][:7]})로 신규 매수 제한 중이에요."
        if block.get("dipped"):
            out.append(
                f"{head} {cur.month}월 마감에 월봉 CCI 0·RSI 50 위로 회복하거나 저가가 "
                f"20개월선({ma})에 닿으면 {nxt.month}월 초부터 풀려요."
            )
        else:
            out.append(
                f"{head} {cur.month}월 저가가 20개월선({ma})에 닿으면 {nxt.month}월 초부터 풀려요. "
                f"안 닿으면 월봉 CCI 0 또는 RSI 50 아래로 눌렸다가 회복해야 해서 "
                f"빨라도 {after.month}월 초예요."
            )
    elif block.get("blocked"):
        out.append(f"코스피 자료 문제({block.get('error')})로 신규 매수를 막았어요.")
    v = data["vix"]
    if v["regime"] not in ("R0", "R1"):
        g = v.get("gate") or {}
        friday = today + timedelta(days=(4 - today.weekday()) % 7)
        out.append(
            f"VIX 주봉 5주선({g.get('w5')})이 10주선({g.get('w10')}) 위라 신규 매수가 막혀 있어요. "
            f"{friday.month}월 {friday.day}일(금) 마감에 5<10이 되면 다음 주부터 열려요."
        )
    if len(out) == 2:
        out.append("두 조건이 모두 풀려야 매수해요.")
    if out:
        out.append("청산은 막지 않아요.")
    return out


def buy_candidates(data: dict) -> list[str]:
    """지금 검증을 통과해 신호가 나면 살 수 있는 종목. KODEX 200이 맨 앞이다."""
    ok = lambda x: (x.get("verify", {}).get("long") or {}).get("label") == "검증됨"  # noqa: E731
    stocks = trade_family(data["assets"])
    return [x["name"] for x in data.get("products", []) + stocks if "error" not in x and ok(x)]


def trade_plan(data: dict, owned: dict) -> list[str]:
    """앞으로의 매매 계획. 숫자는 실제 주문에 쓰는 trade 상수에서 가져온다."""
    from minerva import trade

    blocked = (data.get("market_block") or {}).get("blocked")
    closed = data["vix"]["regime"] not in ("R0", "R1")
    out = ["매수 제한이 풀릴 때까지 매수 주문은 없어요."] if blocked or closed else []
    if owned:
        names = ", ".join(p.get("name", code) for code, p in owned.items())
        stop = f"{trade.LONG_STOP:.0%}"
        out.append(f"보유 중인 {names}은(는) 청산 신호나 손절(−{stop})이 나오면 팔아요.")
    else:
        out.append("지금 봇 보유 종목이 없어서 청산 주문도 없어요.")
    names = buy_candidates(data)
    if names:
        out.append(
            f"매수 후보는 {', '.join(names)} {len(names)}종목이에요. "
            "이 중 신호가 나온 종목만 사고, 후보는 주 1회 다시 검증해요."
        )
    else:
        out.append("지금은 검증을 통과한 매수 후보가 없어요. 후보는 주 1회 다시 검증해요.")
    out.append(
        f"신호 다음 거래일 장 시작 때 전일 종가 +{trade.GAP_BUFFER - 1:.0%} 지정가로 사요. "
        "더 높게 뜨면 사지 않아요."
    )
    out.append(
        f"비중은 개별주 {trade.WEIGHT:.0%}, KODEX 200 {trade.INDEX_ETF_WEIGHT:.0%}까지, "
        f"최대 {trade.MAX_POSITIONS}종목이에요. 손절은 매입가 −{trade.LONG_STOP:.0%}이고, "
        "남는 돈은 CD금리 ETF에 둬요."
    )
    return out


def add_memo(data: dict, today: date) -> None:
    """화면 문구 오류로 분석 파일(주문 근거) 저장이 멈추면 안 된다."""
    try:
        data["memo"] = trade_memo(data, today)
        data["plan"] = trade_plan(data, read_owned())
    except Exception as exc:  # noqa: BLE001
        data["memo"] = [f"요약을 만들지 못했어요 ({type(exc).__name__})."]
        data["plan"] = []


def telegram_summary(data: dict) -> str:
    v, f = data["vix"], data["fear"]
    lines = [
        f"📊 미네르바 일일 분석 ({data['generated_at'][:16].replace('T', ' ')})",
        f"🌡 VIX {v['close']} · 주봉 5/10 {(v.get('gate') or {}).get('w5')}/"
        f"{(v.get('gate') or {}).get('w10')} {'열림' if v['regime'] in ('R0', 'R1') else '닫힘'}"
        f" → 신규 비중 {v['size_limit']}",
        f"😨 공포 초입: {f['state']} (VIX/VIX3M {f['premium']:+.1%})"
        if f.get("premium") is not None
        else "😨 공포 초입: 데이터 없음",
        "",
    ]
    acts = data.get("actions", [])
    if acts:
        lines.append(f"🎯 오늘 할 일 {len(acts)}건")
        lines.extend(f"• {_action_line(x)}" for x in acts[:20])
        if len(acts) > 20:
            lines.append(f"• 외 {len(acts) - 20}건은 화면에서")
    else:
        lines.append("🎯 오늘 할 일: 없음 (진입·청산 신호 없음)")
    if data.get("summary"):
        lines.append("")
        lines.append("📝 미네르바 요약")
        lines.extend(f"• {s}" for s in data["summary"])
    lines.append("투자 판단과 책임은 본인에게 있습니다.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--notify", action="store_true", help="텔레그램으로 요약 전송")
    args = p.parse_args(argv)
    data = build()
    data["summary"] = write_summary(data)
    add_memo(data, datetime.now(timezone(timedelta(hours=9))).date())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(out)
    # 신호 기록: 실제 성과 추적과 규칙 고도화용 (실행마다 한 파일)
    hist = out.parent / "history"
    hist.mkdir(exist_ok=True)
    stamp = data["generated_at"][:16].replace(":", "").replace("T", "-")
    (hist / f"{stamp}.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"wrote {out}")
    if args.notify:
        from minerva import notify

        notify.send(telegram_summary(data))


if __name__ == "__main__":
    main()
