# 사전 고정: TM 눌림 규칙(H2)을 인버스 상품 자체 차트에 그대로 적용 / VIX 필터만 뺀 변형
from minerva.strategy.report import fetch_yahoo
from minerva.strategy.backtest import Rule, Context, evaluate
vix = fetch_yahoo("^VIX", "max")
rules = {"H2 그대로": Rule(level=1, above_ma=50, hold=20, stop=0.10, vix="calm", align=True),
         "VIX 필터 제외": Rule(level=1, above_ma=50, hold=20, stop=0.10, align=True)}
for sym in ["SH", "PSQ", "SDS", "QID", "SSG", "114800.KS", "252670.KS"]:
    d = fetch_yahoo(sym, "max"); ctx = Context.build(d, vix)
    for k, r in rules.items():
        e = evaluate(d, r, ctx); a, i, o = e["all"], e["in_sample"], e["out_of_sample"]
        print(f"{sym:10} {k:8} {e['period']} n={a.get('trades')} win={a.get('win_rate')} mean={a.get('mean')} edge={a.get('edge')} | in={i.get('trades')}/{i.get('edge')} out={o.get('trades')}/{o.get('edge')} maxMAE={a.get('max_mae')}")
