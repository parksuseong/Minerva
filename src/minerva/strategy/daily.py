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
from minerva.strategy.indicators import Bar, ma_snapshot, sma, stoch_fast_k, tm_states
from minerva.strategy.regime import htf_direction, vix_regimes
from minerva.strategy.report import fetch_yahoo

OUT = Path(os.environ.get("MINERVA_DAILY_OUT", "~/.minerva/daily/latest.json")).expanduser()

TM_RULE = Rule(level=1, above_ma=50, hold=20, stop=0.10, vix="calm", align=True)
FEAR_LO, FEAR_HI, FEAR_HOLD = 0.05, 0.15, 60
VERIFIED = {"SPY", "QQQ", "SOXX"}
NOTE = "검증 규칙은 SPY·QQQ·SOXX 기준. 그 밖의 종목은 미검증 적용. 투자 판단과 책임은 본인에게."


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str
    market: str  # US / KR / CRYPTO


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


def analyze(asset: Asset, vix: list[Bar]) -> dict:
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
        "date": last.day.isoformat(),
        "close": _r(last.close),
        "change": _r(last.close / daily[-2].close - 1, 4) if len(daily) > 1 else None,
        "stoch": _r(stoch_fast_k(daily)[-1], 1),
        "cci": _r(st.cci, 1),
        "rsi": _r(st.rsi, 1),
        "ma50": _r(snap["ma"].get(50)),
        "ma200": _r(snap["ma"].get(200)),
        "tm": tm_status(daily, ctx),
    }


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
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {a.symbol: pool.submit(analyze, a, vix) for a in WATCHLIST}
    assets = []
    for a in WATCHLIST:
        try:
            assets.append(futures[a.symbol].result())
        except Exception as exc:  # noqa: BLE001 — 한 종목 실패가 전체를 막지 않게
            assets.append({"symbol": a.symbol, "name": a.name, "error": type(exc).__name__})
    kst = datetime.now(timezone(timedelta(hours=9)))
    return {
        "generated_at": kst.isoformat(timespec="minutes"),
        "vix": {
            "date": vix[-1].day.isoformat(),
            "close": _r(vix[-1].close),
            "regime": regime,
            "size_limit": SIZE_BY_REGIME.get(regime, "?"),
            "leverage": "허용(단기)" if regime in ("R0", "R1") else "금지",
        },
        "fear": fear_status(vix, vix3m),
        "assets": assets,
        "note": NOTE,
    }


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
    for a in data["assets"]:
        if "error" in a:
            lines.append(f"• {a['name']}: 데이터 오류")
            continue
        tm = a["tm"]
        badge = "" if a["verified"] else " (미검증)"
        if tm["state"] == "진입 신호":
            lines.append(f"🟢 {a['name']}{badge}: 진입 신호 · 손절 {tm['stop']} · 최대 20일")
        elif tm["state"] == "보유 중":
            lines.append(
                f"🔵 {a['name']}{badge}: 보유 중 {tm['pnl']:+.1%} · 손절 {tm['stop']} · "
                f"{tm['days_left']}일 남음"
            )
    if not any(a.get("tm", {}).get("state") in ("진입 신호", "보유 중") for a in data["assets"]):
        lines.append("• 오늘은 진입 신호·보유 포지션 없음 (전 종목 대기)")
    lines.append("투자 판단과 책임은 본인에게 있습니다.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--notify", action="store_true", help="텔레그램으로 요약 전송")
    args = p.parse_args(argv)
    data = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(out)
    print(f"wrote {out}")
    if args.notify:
        from minerva import notify

        notify.send(telegram_summary(data))


if __name__ == "__main__":
    main()
