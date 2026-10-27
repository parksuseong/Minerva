import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, D
from minerva.strategy.events import study
from minerva.strategy.indicators import tm_states
from minerva.strategy.regime import vix_regimes, htf_direction
OUT = os.path.join(D, "results_round2.md")
def w(s): print(s, flush=True); open(OUT, "a").write(s + "\n")
SY = ["IWM", "DIA", "EFA", "EWJ", "EEM", "^KS11"]
A = {s: load(s) for s in SY}
vix = load("^VIX"); v3 = load("^VIX3M")
def flags_on(bars, dates, strict):
    days = [b.day for b in bars]; f = [False]*len(bars)
    for d in dates:
        k = bisect.bisect_right(days, d) if strict else bisect.bisect_left(days, d)
        if k < len(bars): f[k] = True
    return f
def g(x, k): v = x.get(k); return "-" if v is None else (f"{v:+.4f}" if isinstance(v, float) else str(v))
HDR = "| 자산 | 기간 | 무조건 | n | 평균 | edge | 승률 | 최악 | n内 | edge内 | n外 | edge外 | 분할일 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"
def row(s, bars, r, h):
    x = r["by_h"][h]
    return (f"| {s} | {bars[0].day}~{bars[-1].day} | {x['baseline']:+.4f} | {g(x['all'],'n')} | {g(x['all'],'mean')} | {g(x['all'],'edge')} | {g(x['all'],'hit')} | {g(x['all'],'worst')} | "
            f"{g(x['in'],'n')} | {g(x['in'],'edge')} | {g(x['out'],'n')} | {g(x['out'],'edge')} | {r['split_at']} |")
def judge(name, res, h, sign, sub=""):
    cnt = {p: sum(1 for r in res.values() if r["by_h"][h][p].get("edge") is not None and r["by_h"][h][p]["edge"]*sign > 0) for p in ("all","in","out")}
    need = "edge<0" if sign < 0 else "edge>0"
    w(f"\n{name}{sub} 부호 기준({need}) 충족: 전체 {cnt['all']}/6 (기준 5/6 → {'충족' if cnt['all']>=5 else '미충족'}), 표본 내 {cnt['in']}/6, 표본 밖 {cnt['out']}/6")
which = sys.argv[1]
if which == "P1":
    w("\n## P1. (월 하락 M10, 주 상승 W20) 일봉 L1 첫 신호 → 20일, gap 5\n기준: 6개 중 5개 이상 전체 edge < 0\n"); w(HDR); res = {}
    for s, bars in A.items():
        n = len(bars); lv = [x.long_level for x in tm_states(bars)]
        M = htf_direction(bars, "M", 10); W = htf_direction(bars, "W", 20)
        f = [i > 0 and lv[i] >= 1 and lv[i-1] < 1 and M.get(bars[i].day) is False and W.get(bars[i].day) is True for i in range(n)]
        r = study(bars, f, (20,), gap=5); res[s] = r; w(row(s, bars, r, 20))
    judge("P1", res, 20, -1)
if which == "P2":
    reg = vix_regimes(vix); vd = [b.day for b in vix]
    ent = [vd[i] for i in range(1, len(vd)) if reg.get(vd[i]) == "R3" and reg.get(vd[i-1]) != "R3"]
    w(f"\n## P2. VIX R3 진입일 → 60일, gap 10 (R3 진입 원 신호 {len(ent)}회)\n기준: 6개 중 5개 이상 전체 edge > 0. 매핑: 미국 ETF 같은/다음 거래일 신호봉, ^KS11 은 d 이후 첫 봉. 진입은 study의 다음 봉 시가\n"); w(HDR); res = {}
    for s, bars in A.items():
        r = study(bars, flags_on(bars, ent, s == "^KS11"), (60,), gap=10); res[s] = r; w(row(s, bars, r, 60))
    judge("P2", res, 60, 1)
if which == "P3":
    m3 = {b.day: b.close for b in v3}
    dd = [b for b in vix if b.day in m3]; pr = [b.close/m3[b.day]-1 for b in dd]
    inb = lambda p: .05 <= p < .15
    ent = [dd[i].day for i in range(1, len(dd)) if inb(pr[i]) and pr[i-1] < .05]
    ref = [dd[i].day for i in range(1, len(dd)) if inb(pr[i]) and not inb(pr[i-1])]
    for nm, E, primary in (("P3 (등록 정의: 5% 미만 → 5~15%)", ent, True), ("P3-참고 (1차 B 구현: 다른 단계 → 5~15%, 위에서 내려온 경우 포함)", ref, False)):
        w(f"\n## {nm} → 60일, gap 10 (원 신호 {len(E)}회, VIX3M {dd[0].day}~)\n기준: 6개 중 5개 이상 전체 edge > 0. 자산은 VIX3M 시작일 이후로 자름\n"); w(HDR); res = {}
        for s, bars in A.items():
            sub = [b for b in bars if b.day >= dd[0].day]
            r = study(sub, flags_on(sub, E, s == "^KS11"), (60,), gap=10); res[s] = r; w(row(s, sub, r, 60))
        judge("P3" if primary else "P3-참고", res, 60, 1)
