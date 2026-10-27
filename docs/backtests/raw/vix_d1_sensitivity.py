import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, D
from minerva.strategy.events import study
from minerva.strategy.regime import vix_normalizing
vix = load("^VIX"); OUT = os.path.join(D, "results_vix.md")
nz = vix_normalizing(vix); vd = [b.day for b in vix]
ds = [vd[i] for i in range(1,len(vd)) if nz.get(vd[i]) and not nz.get(vd[i-1])]
lines = ["\n### D1 민감도(채택 결정용 아님): 이웃 보유기간·gap, 다른 자산"]
for sym in ("SPY","QQQ"):
    bars = load(sym); days = [b.day for b in bars]; f = [False]*len(bars)
    for d in ds:
        k = bisect.bisect_left(days, d)
        if k < len(bars): f[k] = True
    for gap in (5,10,20):
        r = study(bars, f, (10,20,30), gap=gap)
        for h in (10,20,30):
            x = r["by_h"][h]; lines.append(f"- {sym} gap{gap} h{h}: n={x['all']['n']} edge={x['all']['edge']} 内={x['in']['edge']} 外={x['out']['edge']}")
s = "\n".join(lines); print(s); open(OUT,"a").write(s+"\n")
