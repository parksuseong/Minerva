import sys, os, bisect, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, D
from minerva.strategy.events import study, max_rise, dedupe
from minerva.strategy.indicators import sma, rsi, resample
from minerva.strategy.regime import vix_regimes, vix_normalizing
OUT = os.path.join(D, "results_vix.md")
def w(s):
    print(s); open(OUT, "a").write(s + "\n")
vix = load("^VIX"); v3 = load("^VIX3M")
A = {s: load(s) for s in ["SPY", "QQQ", "SOXX"]}
spy = A["SPY"]
def flags_on(bars, dates):
    days = [b.day for b in bars]; f = [False]*len(bars)
    for d in dates:
        k = bisect.bisect_left(days, d)
        if k < len(bars): f[k] = True
    return f
def row(name, r, h):
    x = r["by_h"][h]; g = lambda s, k: s.get(k, "-")
    return f"| {name} | {h} | {x['baseline']:.4f} | {g(x['all'],'n')} | {g(x['all'],'mean')} | {g(x['all'],'edge')} | {g(x['in'],'n')} | {g(x['in'],'edge')} | {g(x['out'],'n')} | {g(x['out'],'edge')} | {g(x['all'],'hit')} | {g(x['all'],'worst')} |"
HDR = "| 신호 | h | 무조건 | n | 평균 | edge | n内 | edge内 | n外 | edge外 | 승률 | 최악 |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
which = sys.argv[1]
if which == "A":
    wk = resample(vix, "W"); c = [b.close for b in wk]; s5, s10 = sma(c,5), sma(c,10)
    gold = [wk[k].day for k in range(1,len(wk)) if None not in (s5[k],s10[k],s5[k-1],s10[k-1]) and s5[k]>s10[k] and s5[k-1]<=s10[k-1]]
    dead = [wk[k].day for k in range(1,len(wk)) if None not in (s5[k],s10[k],s5[k-1],s10[k-1]) and s5[k]<s10[k] and s5[k-1]>=s10[k-1]]
    w("\n## A. VIX 주봉 5/10 교차 → 4·8·13주(20·40·65거래일) 수익\n기대: 골든 edge<0, 데드 edge>0 (표본 내·밖 부호 일치). gap=1(주봉 교차라 자연 분리)\n"); w(HDR)
    ok = tot = 0
    for sym, bars in A.items():
        for nm, ds, sign in (("골든", gold, -1), ("데드", dead, 1)):
            r = study(bars, flags_on(bars, ds), (20,40,65), gap=1)
            for h in (20,40,65):
                w(row(f"{sym} {nm}", r, h)); x = r["by_h"][h]
                tot += 1; ok += (x['in']['edge']*sign > 0 and x['out']['edge']*sign > 0)
        w(f"| {sym} 분할일 | {r['split_at']} |")
    w(f"\n부호 일치(내·밖 모두 기대 방향): {ok}/{tot}")
if which == "B":
    m3 = {b.day: b.close for b in v3}
    days = [b.day for b in vix if b.day in m3]; prem = [vc/m3[d]-1 for d, vc in ((b.day,b.close) for b in vix if b.day in m3)]
    def bk(p):
        if p < 0: return None
        return 0 if p < .05 else 1 if p < .15 else 2 if p < .30 else 3
    w("\n## B. VIX/VIX3M 스팟 프리미엄 단계 진입 첫날 → SPY 20·60일 (gap 10, 2006-07~)\n기대: 단계별 edge 단조 증가\n"); w(HDR)
    sub = [b for b in spy if b.day >= days[0]]
    res = {}
    for b_ in range(4):
        ent = [days[i] for i in range(1,len(days)) if bk(prem[i]) == b_ and bk(prem[i-1]) != b_]
        r = study(sub, flags_on(sub, ent), (20,60), gap=10); res[b_] = r
        for h in (20,60): w(row(["0-5%","5-15%","15-30%","30%+"][b_], r, h))
    w(f"\n분할일 {r['split_at']}. 표본: SPY 2006-07 이후(VIX3M 시작)")
    for h in (20,60):
        for part in ("all","in","out"):
            e = [res[b_]["by_h"][h][part].get("edge") for b_ in range(4)]
            mono = all(e[i] is not None and e[i+1] is not None and e[i+1] > e[i] for i in range(3))
            w(f"- h={h} {part}: edge {e} → 단조증가 {mono}")
if which == "C":
    reg = vix_regimes(vix); c = [b.close for b in vix]; m50, m120, m200 = sma(c,50), sma(c,120), sma(c,200)
    r4 = {vix[i].day: (m200[i] is not None and m50[i] > m120[i] > m200[i]) for i in range(len(vix))}
    vd = [b.day for b in vix]
    w("\n## C. VIX 레짐 진입일 → SPY 20·60일 (gap 10)\n기대: R0 > R2·R3, R4 낮음(edge<0, R0보다 낮음), 표본 내·밖 유지\n"); w(HDR)
    res = {}
    for R in ("R0","R1","R2","R3","R4"):
        if R == "R4": ent = [vd[i] for i in range(1,len(vd)) if r4[vd[i]] and not r4[vd[i-1]]]
        else: ent = [vd[i] for i in range(1,len(vd)) if reg.get(vd[i]) == R and reg.get(vd[i-1]) != R]
        r = study(spy, flags_on(spy, ent), (20,60), gap=10); res[R] = r
        for h in (20,60): w(row(R, r, h))
    w(f"\n분할일 {r['split_at']}")
    for h in (20,60):
        for part in ("in","out"):
            e = {R: res[R]["by_h"][h][part].get("edge") for R in res}
            ok = e["R0"] > e["R2"] and e["R0"] > e["R3"] and e["R4"] < 0 and e["R4"] < e["R0"]
            w(f"- h={h} {part}: " + ", ".join(f"{k}={v}" for k,v in e.items()) + f" → 순서 유지 {ok}")
if which == "D":
    vd = [b.day for b in vix]; c = [b.close for b in vix]
    nz = vix_normalizing(vix); d1 = [vd[i] for i in range(1,len(vd)) if nz.get(vd[i]) and not nz.get(vd[i-1])]
    rs = rsi(c, 14); d2 = [vd[i] for i in range(1,len(vd)) if rs[i-1] is not None and rs[i-1] >= 70 and rs[i] < 70]
    m3 = {b.day: b.close for b in v3}; dd = [b for b in vix if b.day in m3]
    d3 = [dd[i].day for i in range(1,len(dd)) if dd[i-1].close/m3[dd[i-1].day] > 1 and dd[i].close/m3[dd[i].day] < 1]
    w("\n## D. VIX 정상화 신호 → SPY 20일 (gap 10)\n기대: 표본 내·밖 edge > 0\n"); w(HDR)
    for nm, ds in (("D1 정상화 첫날", d1), ("D2 RSI70 하향", d2), ("D3 비율 >1→<1", d3)):
        sub = spy if nm[:2] != "D3" else [b for b in spy if b.day >= dd[0].day]
        r = study(sub, flags_on(sub, ds), (20,), gap=10); w(row(nm, r, 20)); w(f"|  분할일 {r['split_at']} |")
if which == "G":
    mo = resample(vix, "M"); ch = [(mo[k].day, mo[k].close/mo[k-1].close-1) for k in range(1,len(mo)) if mo[k].day.month != 10 or mo[k].day.year != 2026]
    yrs = sorted({d.year for d,_ in ch}); half = yrs[len(yrs)//2]
    exp = {2:1,3:1,4:1,8:1,9:1,10:1,5:-1,6:-1,7:-1,11:-1,12:-1,1:-1}
    w(f"\n## G. VIX 월별 변화율(월말 종가 기준, 2026-10 미완 월 제외). 전반 <{half}, 후반 >={half}\n기대 부호: 2~4·8~10월 +, 5~7·11~1월 -\n")
    w("| 월 | 기대 | 전체 평균 | 상승비율 | 전반 평균 | 전반 상승비율 | 후반 평균 | 후반 상승비율 | n전/n후 |\n|---|---|---|---|---|---|---|---|---|")
    cnt = {"all":0,"h1":0,"h2":0}
    for m in range(1,13):
        a = [x for d,x in ch if d.month==m]; h1 = [x for d,x in ch if d.month==m and d.year<half]; h2 = [x for d,x in ch if d.month==m and d.year>=half]
        f = lambda v: (st.mean(v), sum(x>0 for x in v)/len(v))
        (ma,pa),(m1,p1),(m2,p2) = f(a),f(h1),f(h2)
        for k,v in (("all",ma),("h1",m1),("h2",m2)): cnt[k] += (v*exp[m] > 0)
        w(f"| {m} | {'+' if exp[m]>0 else '-'} | {ma:+.3f} | {pa:.2f} | {m1:+.3f} | {p1:.2f} | {m2:+.3f} | {p2:.2f} | {len(h1)}/{len(h2)} |")
    w(f"\n기대 부호 일치(평균 기준): 전체 {cnt['all']}/12, 전반 {cnt['h1']}/12, 후반 {cnt['h2']}/12")
if which == "M":
    sc = {b.day: b.close for b in spy}; sd = [b.day for b in spy]
    sp = {sd[i]: sc[sd[i]]/sc[sd[i-1]]-1 for i in range(1,len(sd))}
    idx0 = next(i for i,b in enumerate(vix) if b.day >= spy[1].day)
    vb = vix[idx0-1:]  # 1993-01 이후
    fl = [i>0 and b.day in sp and sp[b.day] >= .01 and b.close/vb[i-1].close-1 >= .01 for i,b in enumerate(vb)]
    ev = dedupe(fl, 10); cut = vb[int(len(vb)*.7)].day
    w(f"\n## M. SPY·VIX 동반 +1% 상승일 → 이후 10일 VIX 최대 상승폭(max_rise, 신호일 종가 대비 고가) vs 무조건 (1993~, gap 10, 분할일 {cut})\n기대: 차이 > 0, 표본 내·밖\n")
    w("| 구간 | n | 신호 평균 | 신호 중앙값 | 무조건 평균 | 무조건 중앙값 | 차이(평균) | 차이(중앙값) |\n|---|---|---|---|---|---|---|---|")
    for nm, sel in (("전체", lambda d: True), ("표본 내", lambda d: d < cut), ("표본 밖", lambda d: d >= cut)):
        e = [max_rise(vb,i,10) for i in ev if sel(vb[i].day)]; e = [x for x in e if x is not None]
        u = [max_rise(vb,i,10) for i in range(len(vb)) if sel(vb[i].day)]; u = [x for x in u if x is not None]
        w(f"| {nm} | {len(e)} | {st.mean(e):.4f} | {st.median(e):.4f} | {st.mean(u):.4f} | {st.median(u):.4f} | {st.mean(e)-st.mean(u):+.4f} | {st.median(e)-st.median(u):+.4f} |")
    raw = sum(fl); w(f"\n원 신호일 {raw}일, gap 10 후 {len(ev)}건")
    # 데이터 점검: 고가=종가 비율 (초기 VIX 고가 품질)
    early = [b for b in vb if b.day.year < 2000]; w(f"점검: 2000년 이전 VIX 고가==종가 비율 {sum(abs(b.high-b.close)<1e-9 for b in early)/len(early):.2f}")
