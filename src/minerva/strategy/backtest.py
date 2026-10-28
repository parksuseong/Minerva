"""플레이북 롱 규칙의 이벤트 백테스트.

체결 규칙 (미래 정보 금지):
- 신호는 봉 마감에 판정, 진입은 다음 봉 시가.
- 청산은 먼저 오는 것: 손절가 터치(그 가격) → 익절 조건(그 봉 종가) → 최대 보유 봉수(종가).
  익절 조건: exit="stoch"면 L1 과매수 이탈, exit="trend"면 종가가 trail_ma 이평선 아래로 마감.
- 필터: VIX 레짐(calm=R0·R1, normalizing=스파이크 후 정상화), 상위 시간축 정렬(주·월 이평선 위).
- 성과는 앞 70%(표본 내)와 뒤 30%(표본 밖)로 나눠 보고하고,
  같은 보유기간의 무조건 기대수익과 비교한다.

사용: python -m minerva.strategy.backtest SOXX --frame W --level 3 --hold 26 --stop 0.15
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from minerva.strategy.indicators import (
    Bar,
    crossed_down,
    dmi,
    load_csv,
    resample,
    sma,
    stoch_fast_k,
    tm_states,
)
from minerva.strategy.regime import htf_direction, vix_normalizing, vix_regimes


@dataclass(frozen=True)
class Rule:
    level: int = 3  # TM 롱 레벨 최소값
    cci_n: int = 20
    window: int = 3
    above_ma: int | None = None  # 신호 봉 종가가 이 이평선 위일 때만
    require_pdi: bool = False  # +DI > -DI 일 때만
    hold: int = 20  # 최대 보유 봉수
    stop: float | None = 0.10  # 진입가 대비 손절 비율 (None이면 없음)
    vix: str | None = None  # None / "calm" / "normalizing"
    align: bool = False  # 직전 마감 주봉이 20주선 위, 월봉이 10개월선 위일 때만
    exit: str = "stoch"  # "stoch" / "trend"
    trail_ma: int = 20


@dataclass
class Context:
    """필터에 필요한 외부 데이터(날짜별)."""

    regime: dict[date, str] = field(default_factory=dict)
    normalizing: dict[date, bool] = field(default_factory=dict)
    weekly_up: dict[date, bool] = field(default_factory=dict)
    monthly_up: dict[date, bool] = field(default_factory=dict)

    @classmethod
    def build(cls, daily: Sequence[Bar], vix: Sequence[Bar] | None) -> Context:
        return cls(
            regime=vix_regimes(vix) if vix else {},
            normalizing=vix_normalizing(vix) if vix else {},
            weekly_up=htf_direction(daily, "W", 20),
            monthly_up=htf_direction(daily, "M", 10),
        )


@dataclass(frozen=True)
class Trade:
    entry_day: date
    exit_day: date
    entry: float
    exit: float
    bars: int
    mae: float  # 보유 중 최대 역행 (음수)
    reason: str

    @property
    def ret(self) -> float:
        return self.exit / self.entry - 1


def signals(bars: Sequence[Bar], rule: Rule, ctx: Context | None = None) -> list[bool]:
    if (rule.vix or rule.align) and ctx is None:
        raise ValueError("vix/align 필터에는 Context가 필요하다")
    states = tm_states(bars, cci_n=rule.cci_n, window=rule.window)
    ma = sma([b.close for b in bars], rule.above_ma) if rule.above_ma else None
    pdi, mdi, _ = dmi(bars) if rule.require_pdi else (None, None, None)
    out = []
    for i, s in enumerate(states):
        # 레벨이 처음 기준에 도달한 봉만 신호로 센다 (같은 이벤트 중복 진입 방지)
        ok = s.long_level >= rule.level and (i == 0 or states[i - 1].long_level < rule.level)
        if ok and ma is not None:
            ok = ma[i] is not None and bars[i].close > ma[i]
        if ok and pdi is not None:
            ok = pdi[i] is not None and mdi[i] is not None and pdi[i] > mdi[i]
        day = bars[i].day
        if ok and rule.vix == "calm":
            ok = ctx.regime.get(day) in ("R0", "R1")
        if ok and rule.vix == "normalizing":
            ok = ctx.normalizing.get(day, False)
        if ok and rule.align:
            ok = ctx.weekly_up.get(day, False) and ctx.monthly_up.get(day, False)
        out.append(ok)
    return out


def run(bars: Sequence[Bar], rule: Rule, ctx: Context | None = None) -> list[Trade]:
    sig = signals(bars, rule, ctx)
    stoch = stoch_fast_k(bars)
    trail = sma([b.close for b in bars], rule.trail_ma)
    return simulate(bars, rule, sig, stoch, trail)


def simulate(bars: Sequence[Bar], rule: Rule, sig: Sequence[bool], stoch, trail) -> list[Trade]:
    """신호 목록을 받아 같은 진입·청산 규칙으로 거래를 만든다 (무작위 신호 대조에도 쓴다)."""
    trades: list[Trade] = []
    i = 0
    while i < len(bars) - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        entry = bars[e].open
        stop_px = entry * (1 - rule.stop) if rule.stop else None
        low = entry
        j = e
        exit_px, reason = bars[e].close, "end"
        while True:
            low = min(low, bars[j].low)
            if stop_px is not None and bars[j].low <= stop_px:
                exit_px, reason = min(stop_px, bars[j].open), "stop"
                break
            if j > e and _take(rule, stoch, trail, bars, j):
                exit_px, reason = bars[j].close, "take"
                break
            if j - e + 1 >= rule.hold:
                exit_px, reason = bars[j].close, "time"
                break
            if j == len(bars) - 1:
                exit_px, reason = bars[j].close, "open"
                break
            j += 1
        trades.append(
            Trade(bars[e].day, bars[j].day, entry, exit_px, j - e + 1, low / entry - 1, reason)
        )
        i = j + 1
    return trades


def _take(rule: Rule, stoch, trail, bars: Sequence[Bar], j: int) -> bool:
    if rule.exit == "trend":
        return trail[j] is not None and bars[j].close < trail[j]
    return crossed_down(stoch, j, 80.0)


def random_timing_pvalue(
    bars: Sequence[Bar],
    rule: Rule,
    ctx: Context | None = None,
    n_perm: int = 300,
    seed: int = 7,
) -> float | None:
    """규칙의 진입 타이밍이 무작위 타이밍보다 나은지: 같은 횟수의 무작위 신호로 같은 청산 규칙을
    돌려 평균 수익이 규칙 이상인 비율. 작을수록 우연이 아닐 가능성이 크다 (다중 검정 보정용).
    """
    import random

    sig = signals(bars, rule, ctx)
    stoch = stoch_fast_k(bars)
    trail = sma([b.close for b in bars], rule.trail_ma)
    actual = [t.ret for t in simulate(bars, rule, sig, stoch, trail) if t.reason != "open"]
    k = sum(sig)
    if len(actual) < 2 or k < 2:
        return None
    target = statistics.mean(actual)
    rng = random.Random(seed)
    slots = range(len(bars) - 1)
    hits = 0
    for _ in range(n_perm):
        pick = set(rng.sample(slots, k))
        fake = [i in pick for i in range(len(bars))]
        rets = [t.ret for t in simulate(bars, rule, fake, stoch, trail) if t.reason != "open"]
        if rets and statistics.mean(rets) >= target:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def benjamini_hochberg(pvalues: dict[str, float], q: float = 0.10) -> set[str]:
    """거짓 발견 비율 q로 통과한 키들."""
    ranked = sorted((p, k) for k, p in pvalues.items() if p is not None)
    m = len(ranked)
    cut = 0
    for i, (p, _) in enumerate(ranked, 1):
        if p <= q * i / m:
            cut = i
    return {k for _, k in ranked[:cut]}


def baseline(bars: Sequence[Bar], hold: int) -> float | None:
    """아무 봉에서나 다음 시가에 사서 hold봉 보유했을 때의 평균 수익."""
    rets = [
        bars[i + hold].close / bars[i + 1].open - 1
        for i in range(len(bars) - hold - 1)
        if bars[i + 1].open > 0
    ]
    return statistics.mean(rets) if rets else None


def stats(trades: Sequence[Trade], bars: Sequence[Bar]) -> dict:
    closed = [t for t in trades if t.reason != "open"]
    if not closed:
        return {"trades": 0}
    rets = [t.ret for t in closed]
    hold = max(1, round(statistics.median(t.bars for t in closed)))
    base = baseline(bars, hold)
    worst = min(closed, key=lambda t: t.ret)
    return {
        "trades": len(closed),
        "win_rate": round(sum(r > 0 for r in rets) / len(rets), 3),
        "mean": round(statistics.mean(rets), 4),
        "median": round(statistics.median(rets), 4),
        "worst": {"ret": round(worst.ret, 4), "entry": worst.entry_day.isoformat()},
        "avg_mae": round(statistics.mean(t.mae for t in closed), 4),
        "max_mae": round(min(t.mae for t in closed), 4),
        "median_bars": hold,
        "baseline_mean": None if base is None else round(base, 4),
        "edge": None if base is None else round(statistics.mean(rets) - base, 4),
        "exits": {r: sum(t.reason == r for t in closed) for r in ("take", "stop", "time")},
    }


def evaluate(
    bars: Sequence[Bar], rule: Rule, ctx: Context | None = None, split: float = 0.7
) -> dict:
    trades = run(bars, rule, ctx)
    cut = bars[int(len(bars) * split)].day if len(bars) > 1 else bars[-1].day
    ins = [t for t in trades if t.entry_day < cut]
    oos = [t for t in trades if t.entry_day >= cut]
    in_bars = [b for b in bars if b.day < cut]
    oos_bars = [b for b in bars if b.day >= cut]
    result = {
        "period": f"{bars[0].day} ~ {bars[-1].day}",
        "split_at": cut.isoformat(),
        "all": stats(trades, bars),
        "in_sample": stats(ins, in_bars),
        "out_of_sample": stats(oos, oos_bars),
    }
    n = result["all"]["trades"]
    if n < 30:
        result["warning"] = f"표본 부족 (거래 {n}회 < 30)"
    return result


def main(argv: list[str] | None = None) -> None:
    from minerva.strategy.report import fetch_yahoo

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("symbols", nargs="*")
    p.add_argument("--csv")
    p.add_argument("--range", default="max", help="야후 기간 (예: 10y, max)")
    p.add_argument("--frame", default="D", choices=["D", "W", "M"])
    p.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--cci", type=int, default=20)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--above-ma", type=int)
    p.add_argument("--pdi", action="store_true", help="+DI > -DI 필터")
    p.add_argument("--hold", type=int, default=20)
    p.add_argument("--stop", type=float, default=0.10, help="0이면 손절 없음")
    p.add_argument("--vix", choices=["calm", "normalizing"], help="VIX 레짐 필터")
    p.add_argument("--align", action="store_true", help="주·월봉 이평선 위일 때만")
    p.add_argument("--exit", default="stoch", choices=["stoch", "trend"])
    p.add_argument("--trail-ma", type=int, default=20)
    p.add_argument("--trades", action="store_true", help="개별 거래 목록 출력")
    args = p.parse_args(argv)
    rule = Rule(
        level=args.level,
        cci_n=args.cci,
        window=args.window,
        above_ma=args.above_ma,
        require_pdi=args.pdi,
        hold=args.hold,
        stop=args.stop or None,
        vix=args.vix,
        align=args.align,
        exit=args.exit,
        trail_ma=args.trail_ma,
    )
    vix = fetch_yahoo("^VIX", args.range) if args.vix else None
    targets = [(args.csv, load_csv(args.csv))] if args.csv else []
    targets += [(s, fetch_yahoo(s, args.range)) for s in args.symbols]
    for name, daily in targets:
        bars = resample(daily, args.frame)
        ctx = Context.build(daily, vix) if (rule.vix or rule.align) else None
        out = {
            "symbol": name,
            "frame": args.frame,
            "rule": rule.__dict__,
            **evaluate(bars, rule, ctx),
        }
        if args.trades:
            out["trade_list"] = [
                [t.entry_day.isoformat(), t.exit_day.isoformat(), round(t.ret, 4), t.reason]
                for t in run(bars, rule, ctx)
            ]
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
