import json, os, pickle, sys
from minerva.strategy.report import fetch_yahoo
from minerva.strategy.events import study
from minerva.strategy.indicators import tm_states, sma, dmi, stoch_fast_k
from minerva.strategy.regime import htf_direction
D = os.path.dirname(os.path.abspath(__file__))
def load(s):
    p = f"{D}/mom_{s}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    b = fetch_yahoo(s, "max"); pickle.dump(b, open(p, "wb")); return b
def fmt(r, h):
    x = r["by_h"][h]
    def c(s): return f"{s.get('n',0)}/{s.get('edge','-')}"
    return f"all {c(x['all'])} | in {c(x['in'])} | out {c(x['out'])}"
out = []
def rep(title, bars, flags, hs):
    r = study(bars, flags, horizons=hs, gap=5)
    line = f"{title}: " + " ;; ".join(f"{h}d [{fmt(r,h)}]" for h in hs)
    print(line); out.append(line); return r
for s in ["SPY", "QQQ", "SOXX"]:
    bars = load(s); n = len(bars)
    st = tm_states(bars); lv = [x.long_level for x in st]
    sig = lambda L: [i > 0 and lv[i] >= L and lv[i-1] < L for i in range(n)]
    S1, S3 = sig(1), sig(3)
    c = [b.close for b in bars]
    m5, m10, m20, m50 = sma(c,5), sma(c,10), sma(c,20), sma(c,50)
    ok = lambda i: None not in (m5[i], m10[i], m20[i], m50[i])
    bull = [ok(i) and m5[i] > m10[i] > m20[i] > m50[i] for i in range(n)]
    bear = [ok(i) and m5[i] < m10[i] < m20[i] < m50[i] for i in range(n)]
    hdr = f"\n### {s} ({bars[0].day}~{bars[-1].day}, bars {n}, L1 sig {sum(S1)}, L3 sig {sum(S3)})"
    print(hdr); out.append(hdr)
    out.append("H:")
    for nm, S in (("L1", S1), ("L3", S3)):
        for an, A in (("bull", bull), ("bear", bear)):
            rep(f"H {nm}&{an}", bars, [S[i] and A[i] for i in range(n)], (5, 20))
    out.append("I:")
    M = htf_direction(bars, "M", 10); W = htf_direction(bars, "W", 20)
    for mu in (True, False):
        for wu in (True, False):
            f = [S1[i] and M.get(bars[i].day) is mu and W.get(bars[i].day) is wu for i in range(n)]
            rep(f"I L1 M={'T' if mu else 'F'} W={'T' if wu else 'F'}", bars, f, (20,))
    out.append("K (fwd from i+2):")
    k = stoch_fast_k(bars)
    wh = [False]*n; nw = [False]*n
    for i in range(n-2):
        if S3[i]:
            w = any(k[j] is not None and k[j] >= 80 for j in (i+1, i+2))
            (wh if w else nw)[i+2] = True
    rep("K whipsaw", bars, wh, (5, 20)); rep("K no-whipsaw", bars, nw, (5, 20))
    out.append("L:")
    pdi, mdi, adx = dmi(bars)
    ct = [S3[i] and m50[i] is not None and c[i] < m50[i] and adx[i] is not None for i in range(n)]
    rep("L L3 ct ADX>=50", bars, [ct[i] and adx[i] >= 50 for i in range(n)], (20,))
    rep("L L3 ct ADX<50", bars, [ct[i] and adx[i] < 50 for i in range(n)], (20,))
    dv = lambda i: pdi[i] is not None and mdi[i] is not None
    rep("L L1 +DI>-DI", bars, [S1[i] and dv(i) and pdi[i] > mdi[i] for i in range(n)], (20,))
    rep("L L1 +DI<-DI", bars, [S1[i] and dv(i) and pdi[i] < mdi[i] for i in range(n)], (20,))
    r = study(bars, [False]*n, horizons=(5,20)); out.append(f"baseline 5d {r['by_h'][5]['baseline']} 20d {r['by_h'][20]['baseline']} split {r['split_at']}")
with open(f"{D}/results_mom.md", "a") as fh:
    fh.write("\n## H/I/K/L (momentum agent) — cell = n/edge\n" + "\n".join(out) + "\n")
