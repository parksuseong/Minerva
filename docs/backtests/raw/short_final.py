# 3차 등록 S1~S5 최종 판정: 등록된 청산 규칙(손절 +7%, Stoch 20 상향 익절, 20일)
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-ec2-user-work/f8d55b9a-83eb-45d3-8889-1f47cc871e35/scratchpad")
from short_screen import flags  # noqa: E402  (같은 신호 정의 재사용)
from minerva.strategy.report import fetch_yahoo
from minerva.strategy.short import ShortRule, evaluate_short
from minerva.strategy.indicators import sma, crossed_down, stoch_fast_k
from minerva.strategy.regime import htf_direction
rule = ShortRule(hold=20, stop=0.07)
def s4(d):
    st = stoch_fast_k(d); mu = htf_direction(d, "M", 10); out = []; armed = False; prev = None
    for i, b in enumerate(d):
        m = mu.get(b.day)
        if prev is True and m is False: armed = True
        if m is not False: armed = False
        sig = armed and crossed_down(st, i, 80.0)
        if sig: armed = False
        out.append(sig); prev = m
    return out
for sym in ["SPY", "QQQ", "SOXX", "IWM", "DIA", "EFA", "EWJ", "EEM", "^KS11"]:
    d = fetch_yahoo(sym, "max"); F = flags(d); F["S4"] = s4(d); row = []
    for k in ("S1", "S2", "S3", "S4", "S5"):
        e = evaluate_short(d, F[k], rule)
        a, i, o = e["all"], e["in_sample"], e["out_of_sample"]
        row.append(f"{k} n={a.get('trades')} win={a.get('win_rate')} edge={a.get('edge')} 내={i.get('edge')} 밖={o.get('edge')}")
    print(sym, " | ".join(row))
