import json,sys,statistics as st
from datetime import date
from minerva.strategy.indicators import Bar,resample,tm_states
from minerva.strategy.events import study
from minerva.strategy.report import fetch_yahoo
D=sys.argv[1]
def load(s):
    return [Bar(date.fromisoformat(d),o,h,l,c) for d,o,h,l,c in json.load(open(f"{D}/vol_{s}.json"))]
def q(v,p):
    v=sorted(v); return v[int(p*(len(v)-1))]
out=[]
P=lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))
# ---------- E ----------
v=load("VXX"); n=len(v); C=[b.close for b in v]
cut=int(n*0.7)
def succ(i,lo_d,hi_d):
    m=C[i]
    for k in range(i+1,i+hi_d+1):
        m=min(m,C[k])
        if k-i>=lo_d and v[k].high>=1.10*m:
            return k-i, min(b.low for b in v[i+1:k+1])/C[i]-1
    return None, min(b.low for b in v[i+1:i+hi_d+1])/C[i]-1
is_low=[i>=19 and C[i]==min(C[i-19:i+1]) for i in range(n)]
ev=[]; prev=None
for i in range(n):
    if is_low[i]:
        if prev is not None and max(C[prev+1:i],default=0)>=1.10*C[prev]: ev.append(i)
        prev=i
P("## E VXX 20일 신저가(직전 +10% 반등 후)")
P(f"데이터 {v[0].day}~{v[-1].day} ({n}봉), 분할 {v[cut].day}")
for lo_d,hi_d,tag in ((4,12,"등록 4~12일"),(4,23,"비등록 4~23일")):
    P(f"### {tag}")
    P("| 구간 | 이벤트 n | 성공률 | 무조건 n | 무조건 성공률 | 차이(%p) | 성공까지 중앙 일수 | 성공 전 최악 역행 | 전체 최악 역행(실패 포함) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for name,rng in (("전체",range(19,n)),("표본 내",range(19,cut)),("표본 밖",range(cut,n))):
        ok=[i for i in rng if i+hi_d<n]
        r_ev=[succ(i,lo_d,hi_d) for i in ev if i in set(ok)]
        r_all=[succ(i,lo_d,hi_d) for i in ok]
        se=[x for x in r_ev if x[0]]; sa=sum(1 for x in r_all if x[0])
        re=len(se)/len(r_ev) if r_ev else float('nan'); ra=sa/len(r_all)
        P(f"| {name} | {len(r_ev)} | {re:.1%} | {len(r_all)} | {ra:.1%} | {(re-ra)*100:+.1f} | {st.median([x[0] for x in se]) if se else '-'} | {min([x[1] for x in se]):.1%} | {min([x[1] for x in r_ev]):.1%} |")
P("이벤트 날짜: "+", ".join(v[i].day.isoformat() for i in ev))
# ---------- F ----------
P("\n## F VXX 월중 저점 대비 최대 상승")
months={}
for b in v: months.setdefault((b.day.year,b.day.month),[]).append(b)
rows=[]
for k,bs in months.items():
    m=float('inf'); best=0
    for b in bs:
        m=min(m,b.low); best=max(best,b.high/m-1)
    rows.append((k,best,len(bs)))
rows=[r for r in rows if r[2]>=15]  # 부분월(첫/마지막) 제외
P("| 연도 | 월 수 | ≥10% 비율 | 중앙값 | 최소 |"); P("|---|---|---|---|---|")
for y in sorted({r[0][0] for r in rows}):
    ys=[r[1] for r in rows if r[0][0]==y]
    P(f"| {y} | {len(ys)} | {sum(x>=0.10 for x in ys)/len(ys):.0%} | {st.median(ys):.1%} | {min(ys):.1%} |")
al=[r[1] for r in rows]
P(f"| 전체 | {len(al)} | {sum(x>=0.10 for x in al)/len(al):.0%} | {st.median(al):.1%} | {min(al):.1%} |")
P("10% 미만 월: "+", ".join(f"{r[0][0]}-{r[0][1]:02d}({r[1]:.1%})" for r in rows if r[1]<0.10))
# ---------- J ----------
P("\n## J SOXX 주봉 short_level 3 첫 봉")
def J(sym,bars,window=3,tag=""):
    w=resample(bars,"W"); s=tm_states(w,window=window)
    fl=[s[i].short_level==3 and (i==0 or s[i-1].short_level<3) for i in range(len(w))]
    r=study(w,fl,horizons=(4,8,13),gap=4)
    P(f"### {sym} window={window} {tag} 이벤트 {r['events']}, 분할 {r['split_at']}")
    P("| h(주) | 무조건 평균 | 전체 n/평균/edge/승률/최악 | 표본 내 n/평균/edge | 표본 밖 n/평균/edge |"); P("|---|---|---|---|---|")
    for h,d in r["by_h"].items():
        a,i_,o=d["all"],d["in"],d["out"]
        f=lambda x: f"{x.get('n',0)}/{x.get('mean',0):+.2%}/{x.get('edge',0):+.2%}"
        P(f"| {h} | {d['baseline']:+.2%} | {f(a)}/{a['hit']:.0%}/{a['worst']:+.1%} | {f(i_)} | {f(o)} |")
sx=load("SOXX")
J("SOXX",sx)
for wd in (2,5): J("SOXX",sx,wd,"(민감도)")
try:
    J("QQQ",fetch_yahoo("QQQ","max"),3,"(일반성)")
except Exception as e: P("QQQ fetch 실패",e)
# ---------- N ----------
P("\n## N SOXL − 3×SOXX 롤링")
sl={b.day:b.close for b in load("SOXL")}; sxd={b.day:b.close for b in sx}
days=sorted(set(sl)&set(sxd)); L=[sl[d] for d in days]; X=[sxd[d] for d in days]
# 일일 3배 복리 합성(감쇠 분해용)
syn=[1.0]
for i in range(1,len(days)): syn.append(syn[-1]*(1+3*(X[i]/X[i-1]-1)))
P(f"공통 기간 {days[0]}~{days[-1]} ({len(days)}일)")
P("| 창(일) | 창 수 | 중앙 (SOXL−3×SOXX) | 10분위 | SOXL<3×SOXX 비율 | 중앙 (SOXL−일3배합성) | 10분위 | 중앙 (합성−3×SOXX) |"); P("|---|---|---|---|---|---|---|---|")
for w in (20,60,250):
    d1=[];d2=[];d3=[]
    for t in range(len(days)-w):
        rl=L[t+w]/L[t]-1; rx=X[t+w]/X[t]-1; rs=syn[t+w]/syn[t]-1
        d1.append(rl-3*rx); d2.append(rl-rs); d3.append(rs-3*rx)
    P(f"| {w} | {len(d1)} | {st.median(d1):+.2%} | {q(d1,.1):+.2%} | {sum(x<0 for x in d1)/len(d1):.0%} | {st.median(d2):+.2%} | {q(d2,.1):+.2%} | {st.median(d3):+.2%} |")
open(f"{D}/results_vol.md","a").write("\n".join(out)+"\n")
