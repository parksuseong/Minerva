"""매일 관심 종목에 검증된 규칙을 적용해 화면용 JSON을 만든다.

사용: python -m minerva.strategy.daily [--out PATH] [--notify]
- 확정된 일봉만 쓴다 (장중·당일 미마감 봉은 버림).
- 규칙: momentum-tm §2 TM 눌림 규칙, vol-regime §2-1 공포 초입 매수, §2 레짐 비중 한도.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from minerva.strategy.backtest import Context, Rule, run
from minerva.strategy.indicators import (
    Bar,
    crossed_down,
    ma_snapshot,
    resample,
    sma,
    stoch_fast_k,
    tm_states,
)
from minerva.strategy.regime import htf_direction, vix_regimes
from minerva.strategy.report import fetch_yahoo
from minerva.strategy.short import ShortRule, run_short

OUT = Path(os.environ.get("MINERVA_DAILY_OUT", "~/.minerva/daily/latest.json")).expanduser()

TM_RULE = Rule(level=1, above_ma=50, hold=20, stop=0.10, vix="calm", align=True)
FEAR_LO, FEAR_HI, FEAR_HOLD = 0.05, 0.15, 60
# 종목별 검증 결과 (docs/backtests/2026-10-28-results.md "관심 종목별 적용")
VERIFIED = {"SPY", "QQQ", "SOXX", "^KS11", "^KQ11", "005930.KS", "000660.KS"}
UNSUITED = {"BTC-USD", "AAPL", "NVDA", "TSLA"}  # 롱 규칙이 표본 안팎에서 성립하지 않음
NOTE = (
    "롱 규칙 검증: SPY·QQQ·SOXX·코스피·코스닥·삼성전자·하이닉스. "
    "비트코인·애플·엔비디아·테슬라는 '규칙 부적합'. 투자 판단과 책임은 본인에게."
)


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str
    market: str  # US / KR / CRYPTO
    kind: str = "base"  # base / lev(레버리지) / inv(인버스)
    underlying: str = ""
    mult: int = 1


WATCHLIST = [
    Asset("^KS11", "코스피", "KR"),
    Asset("^KQ11", "코스닥", "KR"),
    Asset("SPY", "S&P500 (SPY)", "US"),
    Asset("QQQ", "나스닥100 (QQQ)", "US"),
    Asset("SOXX", "반도체 (SOXX)", "US"),
    Asset("BTC-USD", "비트코인", "CRYPTO"),
    Asset("005930.KS", "삼성전자", "KR"),
    Asset("000660.KS", "SK하이닉스", "KR"),
    Asset("AAPL", "애플", "US"),
    Asset("NVDA", "엔비디아", "US"),
    Asset("TSLA", "테슬라", "US"),
]

# 사용자 방침: 2배까지만 다룬다 (3배 상품 제외).
# 레버리지는 기초자산 신호를 따르고(KODEX 200 계열은 코스피 지수로 근사),
# 인버스는 상품 자체 차트에 같은 TM 규칙을 적용한다 (미검증: 2026-10-28 백테스트에서 우위 없음).
PRODUCTS = [
    Asset("122630.KS", "KODEX 레버리지", "KR", "lev", "^KS11", 2),
    Asset("233740.KS", "KODEX 코스닥150레버리지", "KR", "lev", "^KQ11", 2),
    Asset("QLD", "QLD (나스닥 2배)", "US", "lev", "QQQ", 2),
    Asset("USD", "USD (반도체 2배)", "US", "lev", "SOXX", 2),
    Asset("114800.KS", "KODEX 인버스", "KR", "inv", "^KS11", -1),
    Asset("252670.KS", "KODEX 200선물인버스2X (곱버스)", "KR", "inv", "^KS11", -2),
    Asset("QID", "QID (나스닥 −2배)", "US", "inv", "QQQ", -2),
    Asset("SSG", "SSG (반도체 −2배)", "US", "inv", "SOXX", -2),
]

SHORT_RULE = ShortRule(hold=20, stop=0.07)
SHORT_VERIFIED = {"QQQ", "SOXX", "^KS11", "^KQ11"}  # 숏 규칙이 표본 안팎 모두 + 인 기초자산


def short_flags(bars: list[Bar]) -> tuple[list[bool], dict]:
    """TM 되돌림 숏(S1): Stoch 80 하향 이탈 + 50일선·20주선·10개월선 아래."""
    closes = [b.close for b in bars]
    ma50 = sma(closes, 50)
    st = stoch_fast_k(bars)
    wu, mu = htf_direction(bars, "W", 20), htf_direction(bars, "M", 10)
    flags = []
    for i, b in enumerate(bars):
        flags.append(
            crossed_down(st, i, 80.0)
            and ma50[i] is not None
            and b.close < ma50[i]
            and wu.get(b.day) is False
            and mu.get(b.day) is False
        )
    i, last = len(bars) - 1, bars[-1]
    conds = {
        "Stoch 80 하향 이탈": crossed_down(st, i, 80.0),
        "50일선 아래": ma50[i] is not None and last.close < ma50[i],
        "주봉 20주선 아래": wu.get(last.day) is False,
        "월봉 10개월선 아래": mu.get(last.day) is False,
    }
    return flags, conds


def short_status(under: list[Bar]) -> dict:
    flags, conds = short_flags(under)
    trades = run_short(under, flags, SHORT_RULE)
    last = under[-1]
    if trades and trades[-1].reason == "open":
        t = trades[-1]
        entry = t.exit  # run_short는 숏 수익 계산을 위해 가격을 뒤집어 저장한다
        return {
            "state": "보유 중",
            "entry_day": t.entry_day.isoformat(),
            "stop": f"기초자산 {_r(entry * (1 + SHORT_RULE.stop))} 돌파 시 손절",
            "pnl": _r(entry / last.close - 1, 4),
            "days_left": SHORT_RULE.hold - t.bars,
            "exit_rule": "기초자산 Stoch 20 상향 돌파 익절 / +7% 손절 / 20거래일",
        }
    if all(conds.values()):
        return {
            "state": "진입 신호",
            "entry": "다음 거래일 시가 (인버스 매수)",
            "stop": f"기초자산 {_r(last.close * (1 + SHORT_RULE.stop))} 돌파 시 손절",
            "max_days": SHORT_RULE.hold,
            "exit_rule": "기초자산 Stoch 20 상향 돌파 익절 / +7% 손절 / 20거래일",
        }
    missing = [k for k, ok in conds.items() if not ok]
    return _with_wait({"state": "대기", "missing": missing}, wait_levels(under, "short"))


DECAY_NOTE = "감쇠: 배수가 클수록, 오래 들수록 커진다 (3배 기준 1년 중앙값 −11.6%p)"
INVERSE_NOTE = (
    "기초자산 TM 되돌림 숏 규칙(검증: 개발 3/3, 독립 5/6, 우위 작음) · 비중은 롱의 1/2 이하"
)

MARKET_CLOSE = {
    "KR": (ZoneInfo("Asia/Seoul"), 15, 40),
    "US": (ZoneInfo("America/New_York"), 16, 10),
    "CRYPTO": (timezone.utc, 0, 0),
}


def completed(bars: list[Bar], market: str, now: datetime | None = None) -> list[Bar]:
    """마지막 봉이 아직 마감 전(당일 진행 중)이면 버린다."""
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


def _r(v: float | None, n: int = 2) -> float | None:
    return None if v is None else round(v, n)


def _week_end(d: date) -> date:
    """d 다음에 오는 주봉 확정일(금요일). d가 금요일이면 다음 주 금요일."""
    return d + timedelta(days=(4 - d.weekday()) % 7 or 7)


def _month_end(d: date) -> date:
    nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def wait_levels(bars: list[Bar], side: str, vix_ma50: float | None = None) -> dict[str, str]:
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
            "VIX 평온(R0·R1)": f"VIX 종가가 50일선 {f(vix_ma50)} 아래로 복귀",
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
        "VIX 평온(R0·R1)": ctx.regime.get(last.day) in ("R0", "R1"),
        "주봉 20주선 위": ctx.weekly_up.get(last.day, False),
        "월봉 10개월선 위": ctx.monthly_up.get(last.day, False),
    }
    trades = run(bars, TM_RULE, ctx)
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


def analyze(asset: Asset, vix: list[Bar], vix_ma50: float | None = None) -> dict:
    daily = completed(fetch_yahoo(asset.symbol, "max"), asset.market)
    ctx = Context(
        regime=_regime_by_day(vix, daily),
        weekly_up=htf_direction(daily, "W", 20),
        monthly_up=htf_direction(daily, "M", 10),
    )
    last = daily[-1]
    snap = ma_snapshot([b.close for b in daily])
    st = tm_states(daily)[-1]
    return {
        "symbol": asset.symbol,
        "name": asset.name,
        "market": asset.market,
        "verified": asset.symbol in VERIFIED,
        "unsuited": asset.symbol in UNSUITED,
        "date": last.day.isoformat(),
        "close": _r(last.close),
        "change": _r(last.close / daily[-2].close - 1, 4) if len(daily) > 1 else None,
        "stoch": _r(stoch_fast_k(daily)[-1], 1),
        "cci": _r(st.cci, 1),
        "rsi": _r(st.rsi, 1),
        "ma50": _r(snap["ma"].get(50)),
        "ma200": _r(snap["ma"].get(200)),
        "tm": _with_wait(tm_status(daily, ctx), wait_levels(daily, "long", vix_ma50)),
    }


def regime_is_fear(vix: list[Bar] | None) -> bool:
    """VIX/VIX3M ≥ 1이면 공포 극점: 숏 신규 진입을 막는다 (S5 기각)."""
    if not vix:
        return False
    m3 = completed(fetch_yahoo("^VIX3M", "max"), "US")
    return bool(m3) and m3[-1].day == vix[-1].day and vix[-1].close >= m3[-1].close


def product_status(p: Asset, base: dict, regime: str | None, vix: list[Bar] | None = None) -> dict:
    """레버리지: 기초자산 결과 + 레짐 제약. 인버스: 자기 차트에 TM 규칙(미검증)."""
    daily = completed(fetch_yahoo(p.symbol, "max" if p.kind == "inv" else "1y"), p.market)
    last = daily[-1]
    out = {
        "symbol": p.symbol,
        "name": p.name,
        "market": p.market,
        "kind": p.kind,
        "mult": p.mult,
        "underlying": base.get("name", p.underlying),
        "verified": False,
        "date": last.day.isoformat(),
        "close": _r(last.close),
        "change": _r(last.close / daily[-2].close - 1, 4) if len(daily) > 1 else None,
    }
    btm = base.get("tm", {})
    if p.kind == "lev":
        allowed = regime in ("R0", "R1")
        if btm.get("state") == "진입 신호" and allowed:
            tm = {
                "state": "진입 신호",
                "entry": "다음 거래일 시가 (기초자산 신호)",
                "stop": f"기초자산 {btm.get('stop')} 이탈 시 정리",
                "max_days": TM_RULE.hold,
                "exit_rule": "기초자산 Stoch 80 하향 이탈 / 기초 −10% / 20거래일",
            }
        elif btm.get("state") == "보유 중":
            tm = {
                "state": "보유 중",
                "entry_day": btm.get("entry_day"),
                "days_left": btm.get("days_left"),
                "stop": f"기초자산 {btm.get('stop')}",
                "exit_rule": btm.get("exit_rule"),
                "pnl": None,
            }
        else:
            missing = list(btm.get("missing", []))
            if not allowed:
                missing.insert(0, f"레짐 {regime}: 레버리지 금지")
            tm = {"state": "대기", "missing": missing}
        tm["note"] = DECAY_NOTE
    else:
        under = completed(fetch_yahoo(p.underlying, "max"), p.market)
        tm = short_status(under)
        if tm["state"] == "진입 신호" and regime_is_fear(vix):
            tm = {"state": "대기", "missing": ["백워데이션 중 신규 숏 금지"]}
        tm["wait"] = [f"[{out['underlying']}] {w}" for w in tm.get("wait", [])]
        tm["note"] = INVERSE_NOTE
        out["verified"] = p.underlying in SHORT_VERIFIED
    out["tm"] = tm
    return out


def _with_wait(tm: dict, levels: dict[str, str]) -> dict:
    if tm.get("state") == "대기":
        tm["wait"] = [f"{k}: {levels[k]}" for k in tm.get("missing", []) if k in levels]
    return tm


def _regime_by_day(vix: list[Bar], daily: list[Bar]) -> dict[date, str]:
    """자산 거래일마다 그날까지 확정된 마지막 VIX 레짐 (한국장은 전날 미국 마감 기준)."""
    reg = vix_regimes(vix)
    days = sorted(reg)
    out, k = {}, -1
    for b in daily:
        while k + 1 < len(days) and days[k + 1] <= b.day:
            k += 1
        if k >= 0:
            out[b.day] = reg[days[k]]
    return out


def build() -> dict:
    vix = completed(fetch_yahoo("^VIX", "max"), "US")
    vix3m = completed(fetch_yahoo("^VIX3M", "max"), "US")
    regime = vix_regimes(vix).get(vix[-1].day)
    vix_ma50 = sma([b.close for b in vix], 50)[-1]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {a.symbol: pool.submit(analyze, a, vix, vix_ma50) for a in WATCHLIST}
    assets = []
    for a in WATCHLIST:
        try:
            assets.append(futures[a.symbol].result())
        except Exception as exc:  # noqa: BLE001 — 한 종목 실패가 전체를 막지 않게
            assets.append({"symbol": a.symbol, "name": a.name, "error": type(exc).__name__})
    by_symbol = {a["symbol"]: a for a in assets}
    with ThreadPoolExecutor(max_workers=6) as pool:
        pfut = {
            p.symbol: pool.submit(product_status, p, by_symbol.get(p.underlying, {}), regime, vix)
            for p in PRODUCTS
        }
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
            "size_limit": SIZE_BY_REGIME.get(regime, "?"),
            "leverage": "허용(단기)" if regime in ("R0", "R1") else "금지",
            "ma50": _r(vix_ma50),
        },
        "fear": fear_status(vix, vix3m),
        "assets": assets,
        "products": products,
        "note": NOTE,
    }


def write_summary(data: dict) -> list[str]:
    """분석 JSON만 근거로 쓴 요약문 (숫자는 모두 data에서 온다)."""
    v, f = data["vix"], data["fear"]
    out = []
    if v["regime"] in ("R0", "R1"):
        out.append(f"VIX {v['close']}로 50일선({v['ma50']}) 아래라 롱 규칙이 열려 있어요.")
    else:
        out.append(
            f"VIX {v['close']}가 50일선({v['ma50']}) 위({v['regime']})라 "
            "롱 신규 진입은 막혀 있어요. "
            f"VIX가 {v['ma50']} 아래로 마감하면 롱 규칙이 다시 열려요."
        )
    if f.get("state", "없음") != "없음":
        out.append(f"공포 초입 매수: {f['state']}.")
    rows = [a for a in data["assets"] + data.get("products", []) if "tm" in a]
    active = [a for a in rows if a["tm"]["state"] in ("진입 신호", "보유 중")]
    for a in active:
        out.append(f"{a['name']}: {a['tm']['state']} — {a['tm'].get('stop', '')}")
    near = sorted(
        (a for a in data["assets"] if a.get("tm", {}).get("state") == "대기"),
        key=lambda a: len(a["tm"].get("missing", [])),
    )[:3]
    for a in near:
        waits = a["tm"].get("wait") or a["tm"].get("missing", [])
        out.append(f"{a['name']}(빠진 조건 {len(a['tm']['missing'])}개): " + " / ".join(waits))
    shorts = sorted(
        (a for a in data.get("products", []) if a.get("kind") == "inv" and "tm" in a),
        key=lambda a: len(a["tm"].get("missing", [])),
    )[:1]
    for a in shorts:
        if a["tm"]["state"] == "대기":
            out.append(
                f"숏 쪽 가장 가까운 후보 {a['name']}: " + " / ".join(a["tm"].get("wait", []))
            )
    return out


def telegram_summary(data: dict) -> str:
    v, f = data["vix"], data["fear"]
    lines = [
        f"📊 미네르바 일일 분석 ({data['generated_at'][:16].replace('T', ' ')})",
        f"🌡 VIX {v['close']} · {v['regime']} → 신규 비중 {v['size_limit']}, "
        f"레버리지 {v['leverage']}",
        f"😨 공포 초입: {f['state']} (VIX/VIX3M {f['premium']:+.1%})"
        if f.get("premium") is not None
        else "😨 공포 초입: 데이터 없음",
    ]
    rows = data["assets"] + data.get("products", [])
    for a in rows:
        if "error" in a:
            lines.append(f"• {a['name']}: 데이터 오류")
            continue
        tm = a["tm"]
        badge = "" if a.get("verified") else " (미검증)"
        if tm["state"] == "진입 신호" and a.get("kind") == "inv":
            lines.append(f"🔻 {a['name']}: 숏 진입 신호(인버스 매수) · {tm['stop']} · 최대 20일")
        elif tm["state"] == "진입 신호":
            lines.append(f"🟢 {a['name']}{badge}: 진입 신호 · 손절 {tm['stop']} · 최대 20일")
        elif tm["state"] == "보유 중":
            pnl = "" if tm.get("pnl") is None else f" {tm['pnl']:+.1%}"
            lines.append(
                f"🔵 {a['name']}{badge}: 보유 중{pnl} · 손절 {tm['stop']} · "
                f"{tm['days_left']}일 남음"
            )
    if not any(a.get("tm", {}).get("state") in ("진입 신호", "보유 중") for a in rows):
        lines.append("• 오늘은 진입 신호·보유 포지션 없음 (전 종목 대기)")
    if data.get("products"):
        lines.append(
            f"• 레버리지(2배까지): {data['vix']['leverage']} · 인버스: TM 되돌림 숏 규칙 적용"
        )
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
