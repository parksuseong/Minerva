# ROADMAP 6: TM 눌림 규칙(롱)과 TM 되돌림 숏을 국내·코인·개별주에 그대로 적용 (규칙·기준 변경 없음)
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-ec2-user-work/f8d55b9a-83eb-45d3-8889-1f47cc871e35/scratchpad")
from short_screen import flags
from minerva.strategy.report import fetch_yahoo
from minerva.strategy.backtest import Rule, Context, evaluate
from minerva.strategy.short import ShortRule, evaluate_short
from minerva.strategy.daily import _regime_by_day, completed
from minerva.strategy.regime import htf_direction
vix = fetch_yahoo("^VIX", "max")
L = Rule(level=1, above_ma=50, hold=20, stop=0.10, vix="calm", align=True)
for sym, mk in [("^KS11","KR"),("^KQ11","KR"),("005930.KS","KR"),("000660.KS","KR"),("BTC-USD","CRYPTO"),("AAPL","US"),("NVDA","US"),("TSLA","US")]:
    d = completed(fetch_yahoo(sym, "max"), mk)
    ctx = Context(regime=_regime_by_day(vix, d), weekly_up=htf_direction(d,"W",20), monthly_up=htf_direction(d,"M",10))
    e = evaluate(d, L, ctx); s = evaluate_short(d, flags(d)["S1"], ShortRule())
    g = lambda r: f"n={r['all'].get('trades')} win={r['all'].get('win_rate')} edge={r['all'].get('edge')} 내={r['in_sample'].get('edge')} 밖={r['out_of_sample'].get('edge')}"
    print(f"{sym:10} 롱[{g(e)}]  숏[{g(s)}]")
