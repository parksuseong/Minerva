# 가상 추적 규칙 교체 검증 (2026-11-22)

BRK·V06·F 교체 후보 3개를 사전 등록 → 검증 → 레드팀. 결과 0/3 통과. FLW·ABS를 '과거 검증 미통과, 전진 관찰용'으로 장벽 추적에 넣었다. 코드: 연구 스크래치(저장소 밖), 자료: ~/.minerva/research/edge13-2026-11-22/full (tk/ = 1시간봉 테이커 매수 거래대금·체결 수).

## Paper-swap pre-registration (2026-11-22, written BEFORE extracting taker/count data or computing anything)

Purpose: replace retired paper rules BRK (barrier), V06 and F (edge_paper) with up to 3 new forward-observation rules
in the +5%/−3% barrier tracker. Directions must be NOT in edge13 "탈락한 큰 갈래".
Chosen unexplored family: ORDER FLOW from kline fields never used in edge13 — taker-buy quote volume (col 10) and
trade count (col 8). (edge13 used o/h/l/c/quote-volume only; funding/premium/OI not reused.)

## Data
Survivorship-free 918 USDT-M perps, 1h klines from data.binance.vision zips (~/.minerva/research/edge13-2026-11-22/full/k),
2020-10..2026-10. Extract hourly taker_buy_quote (tb) and trade count (n) aligned with the existing h1 arrays.
Daily bars = UTC days (24 complete hours, else NaN). Labels = existing lab/{sym}_5x3.npz (+5/−3, 336h cap,
0.14% RT + real funding, same-bar both → loss, gap → open). Entry = next day 00:00 UTC (hour index (d+1)*24).

## Universe
Each month: ranks 1–300 by prior-30d quote volume, listed ≥ 90 days, excl BTC (identical to edge13 R52 U3).
One open trade per symbol per rule (signals during an open trade skipped).

## Definitions (all causal: day d uses bars up to and including day d close)
- I7(d) = 2·Σtb(d−6..d)/Σq(d−6..d) − 1  (7-day net taker imbalance, −1..+1)
- P(d)  = percentile rank of I7(d) among the coin's own I7(d−180..d−1) (finite values only, need ≥ 120)
- r7(d) = c(d)/c(d−7) − 1
- S7(d) = Σq/Σn over d−6..d (average trade size in USDT); SR(d) = S7(d) / median(S7(d−97..d−7)) (need ≥ 60 finite)
- "first day" = condition true on d and false on d−1.

## Candidates (3, each two-sided = one rule; mirrored side always included)
- C1 ABS "흡수(taker absorption) 역추세": long if P ≤ 0.10 and r7 ≥ 0 (heavy net taker selling but price not down);
  short if P ≥ 0.90 and r7 ≤ 0 (heavy net taker buying but price not up). First day of each side's condition.
- C2 FLW "흐름 추종": long if P ≥ 0.90 and r7 > 0; short if P ≤ 0.10 and r7 < 0. First day.
- C3 WHL "큰손 참여(평균 체결 크기) 추종": long if SR ≥ 2.0 and r7 ≥ +0.10; short if SR ≥ 2.0 and r7 ≤ −0.10. First day.

## Windows
W1 = 2021-01-01..2023-12-31, W2 = 2024-01-01..2026-10-31 (entries; labels already include later exits).
Also report per-year and per-side (descriptive only).

## Baselines
(a) same coin, same side, same calendar month: random hourly entry from all valid label hours of that coin-month,
    500 draws of the full trade list → p = share of draws with mean ≥ rule mean (over W1∪W2).
(b) cross-sectional: same entry hour, same side, random other universe member, 200 draws (descriptive + gate).

## Pass gates (ALL)
1. n ≥ 100 in W1 AND n ≥ 100 in W2.
2. mean net − 0.10% extra slippage > 0 in W1 AND in W2.
3. baseline (a) Holm(3) p < 0.05 over W1∪W2, and rule mean > baseline-(a) mean in each window separately.
4. baseline (b) Holm(3) p < 0.05.
5. no single coin > 20% of total net PnL; ≥ 4 of 6 calendar years (2021–2026) with mean > 0.
Cumulative tested before: 831 (edge13). This round: 832–834.

## Pre-committed decision if 0 pass
Still replace BRK/V06/F with forward experiments labelled "과거 검증 미통과, 전진 관찰용": choose by
(i) not already tracked, (ii) best (rule mean − baseline-(a) mean) over W1∪W2, (iii) both windows' sign reported.
No threshold changes after seeing results. No further variants this round.

## ADDENDUM 2026-11-22 20:35 KST (user changed the game; written BEFORE any backtest was run — only the tb/n extraction had started)
Geometry is no longer fixed. Pre-chosen geometry per candidate (reasoning a priori, not from data):
- C1 ABS (reversal after absorption — expect modest snap-back): symmetric +3/−3 (BE ≈ 52.3% + funding).
- C2 FLW (flow-follow trend — let it run): +6/−3 (BE ≈ 34.9%).
- C3 WHL (large-trade participation follow): +5/−3 (BE ≈ 39.3%).
All: cap 336h, 0.14% RT + real funding, same-bar both → loss, gap → open; labels from lab.labels().
Grid computed for transparency: every candidate × {3x3, 4x3, 5x3, 6x3} = 12 cells. ALL 12 count toward multiple
testing: Holm over 12 cells for gates 3 and 4. A candidate passes only if its PRE-CHOSEN cell passes all gates.
Other cells are descriptive only and cannot be promoted this round (would need a new pre-registration + unseen data).
Baseline (a)/(b) use labels of the SAME geometry. Gate 2 (mean − 0.10% > 0 in W1 and W2) unchanged.
Cumulative count: 831 + 12 cells = 843.

## ADDENDUM 2 (20:55 KST, before any result line was produced): the first run was killed for memory (storing every
coin-month label pool). Baseline p-values switched from 500 Monte-Carlo draws to the exact CLT equivalent
(mean of one random draw per trade: mean = avg of pool means, var = Σ pool var / n²). Same baselines, same gates.

## RESULT (2026-11-22 ~21:45 KST) — 0/3 pass. Cum 843.
Pre-chosen cells (mean net/trade, % ; baseline (a) = same coin·side·month random, same geometry):
- C1 ABS 3x3: n 13,266, win 50.0% (BE 52.3%), −0.13 vs −0.02 (p_a 1.00). W1 n4290 −0.02 vs −0.04; W2 n8976 −0.19 vs −0.02. FAIL g2,g3,g4,g5.
- C2 FLW 6x3: n 23,167, win 35.3% (BE 34.9%), +0.06 vs −0.04 (Holm12 p_a 0.006, p_b 0.042). W1 n6578 −0.04 vs −0.04; W2 n16589 +0.10 vs −0.03. FAIL g2 (W1<0, W2−0.10%≈0), g3 (W1 not above random).
  Post-hoc cluster check (red team, descriptive): week-clustered excess t=0.01 p=0.50; month-clustered p=0.23 → CLT p is an independence artifact.
- C3 WHL 5x3: n 827, win 33.7%, −0.43 vs −0.02 (p_a 1.00). FAIL.
Pre-committed fallback: choose by (rule − baseline a): FLW +0.10, ABS −0.11, WHL −0.41.

## Red team — paper swap round (C1 ABS, C2 FLW, C3 WHL), 2026-11-22

Verdict: 0/3 pass. FLW and ABS go in only as "과거 검증 미통과, 전진 관찰용". WHL not tracked.

1. Lookahead
- Signals use day-d complete UTC daily bars (24 h all present) and enter at (d+1) 00:00 open. I7 percentile uses the
  coin's own I7 over d−180..d−1 (strictly prior). r7 uses c(d)/c(d−7). OK.
- Tracker vs backtest parity on real data (ETH, DOGE; SOL/KNC skipped for data gaps): 13,284 rule-days, 1 mismatch
  (DOGE WHL day 93 — start-of-history edge, rule since removed). The tracker only uses closed daily bars (_series ≤ d).
- Universe: monthly prior-30d volume rank in backtest vs daily rank in tracker (same as PUMP/DUMP). Minor drift.
2. Survivorship
- Survivorship-free 918-perp archive incl. delisted. Tracker backfill from 11-01 misses coins delisted since (≈3 weeks,
  same caveat as existing rules).
3. Costs
- 0.14% RT + real funding in labels; gate adds 0.10% extra slip. FLW's best window (W2 +0.10%) is fully eaten by the
  extra slip. Taker-heavy fills in the first hour after 00:00 UTC on alts can be worse than 0.07%/side.
4. Clustering
- FLW fires in bursts: up to 165 entries on one day, p95 34/day. The CLT p-values (0.006) treat 23k trades as
  independent. Clustering by week: excess t=0.01 (p 0.50); by month: p 0.23. The "edge" is a few heavy weeks.
- Years: FLW 2021 −0.17, 2022 −0.04, 2023 +0.01, 2024 +0.15, 2025 +0.15, 2026 +0.01 (%): positive only in 2024–25.
5. Multiple testing
- 12 cells (3 rules × 4 geometries) Holm-corrected; cumulative 843 variants across edge13 + this round. With this many
  tries, a single pass at p≈0.04 would itself be expected by chance.
6. Geometry
- Pre-chosen per rule before results. FLW looks "better" at 6x3 than 3x3 (+0.06 vs −0.06) — that is the coin-month
  trend drift again (wider target captures drift), not entry timing. Not promoted.
7. Mirror / sign
- ABS is roughly FLW's opposite and loses vs random (p_a 1.00). Tracking both forward is a sign check: if FLW > ABS
  persists forward, flow direction carries information; if both ≈ random, the family is dead.
8. Load
- FLW+ABS backtest rate ≈ 17 trades/day across 300 coins → many open paper positions; resolve() fetches hourly bars
  per open trade every hour. Watch GET volume on the shared IP.

## 배포 전 시험 실행(실제 상태 사본, 공개 GET만, 2026-11-22 21:28 KST)
- 11/1~21 백필: FLW 165건(롱 64·숏 101), ABS 142건(롱 13·숏 129). 9분 11초, 최대 메모리 146 MB.
- 자동 교체 기준을 그대로 적용하면 첫 실행에서 바로 교체되는 규칙: DUMP(최근 14일 끝난 7건 합 −22.2%),
  FLW(163건 거래당 평균 −0.01%). ABS는 130건 승률 63.1%, 거래당 +0.65%(11월은 되돌림 장).

## 자동 교체 판정 범위 수정(같은 날, 머지 전)
- 판정은 규칙별 live_since(장벽 추적 전 규칙 2026-11-22 00:00 UTC) 이후 진입한 거래만 센다. 백필은 화면에만.
  같은 시험 상태로 다시 보면 첫 실행 교체 0건(판정 집계: PUMP 1, FLW 1, ABS 6건).
- 자동 교체가 일어나면 텔레그램 글 한 통(규칙·수치·사유·후속 또는 "대기열 비어 있음 → 자리 비워 둠"),
  교체당 한 번만(state의 notified, 실패하면 다음 실행에 다시).
- edge_paper는 자동 교체 없음. V01 판정 집계는 실시간 시작(2026-11-21 08:00 UTC) 이후만 센다.
