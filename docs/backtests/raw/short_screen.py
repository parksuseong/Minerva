# 숏 후보 S1~S5 1차 스크리닝: 고정 20일 보유(청산 규칙 미적용). 최종 판정 아님.
from minerva.strategy.report import fetch_yahoo
from minerva.strategy.events import study
from minerva.strategy.indicators import sma, stoch_fast_k, crossed_down
from minerva.strategy.regime import htf_direction, vix_regimes
vix = fetch_yahoo("^VIX", "max"); v3 = {b.day: b.close for b in fetch_yahoo("^VIX3M", "max")}
reg = vix_regimes(vix); vc = {b.day: b.close for b in vix}
def flags(d):
    c = [b.close for b in d]; m50, m200 = sma(c, 50), sma(c, 200); st = stoch_fast_k(d)
    wu, mu = htf_direction(d, "W", 20), htf_direction(d, "M", 10)
    out = {k: [] for k in ("S1", "S2", "S3", "S5")}
    for i, b in enumerate(d):
        x = crossed_down(st, i, 80.0)
        s1 = x and m50[i] is not None and b.close < m50[i] and wu.get(b.day) is False and mu.get(b.day) is False
        s3 = x and m200[i] is not None and m50[i] is not None and b.close < m200[i] and m50[i] < m200[i]
        bw = b.day in v3 and vc.get(b.day, 0) >= v3[b.day]
        out["S1"].append(s1); out["S2"].append(s1 and reg.get(b.day) in ("R2", "R3"))
        out["S3"].append(s3); out["S5"].append(s3 and bw)
    return out
for sym in ["SPY", "QQQ", "SOXX", "IWM", "DIA", "EFA", "EWJ", "EEM", "^KS11"]:
    d = fetch_yahoo(sym, "max"); F = flags(d); row = []
    for k, f in F.items():
        r = study(d, f, horizons=(20,), gap=5)["by_h"][20]
        se = lambda s: None if not s.get("n") else round(-s["edge"], 4)
        row.append(f"{k}: n={r['all'].get('n',0)} 숏edge={se(r['all'])} 내={se(r['in'])} 밖={se(r['out'])}")
    print(sym, " | ".join(row))
