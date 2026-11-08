# quantpaper.com 논문 검토와 BTC 선물 시스템 설계 (2026-11-08)

사용자 요청(텔레그램 03:52): "quantpaper.com 논문도 참조해봐. 시스템/퀀트 트레이딩 모델을 하나 만들고 싶어."
이 문서는 문헌 검토와 설계까지다. 시험은 별도 사전 등록 문서(2026-11-08-btc-layered-model-hypotheses.md)에서 한다.

## 자료원

- quantpaper.com: 퀀트 논문 601편(2026-11-08 기준) 색인. 대부분 arXiv. 공개 API:
  `GET /api/papers?page=N&limit=100&status=published[&search=..][&tags=..]` (목록, 초록 일부),
  `GET /api/papers/<uuid>` (전체 초록 + arxiv_url/pdf_url), `POST /api/search/semantic {"query":..}`.
- 601편 중 암호화폐 관련 40편, 방법론 관련 약 70편을 제목·초록으로 골랐고, 그중 24편의 상세(초록·링크)를 읽었다.
  본문 PDF까지 읽지는 않았다 → 아래 요약은 초록 기준이며, 수치 주장은 저자 주장이다.
- 전체 목록에서 크게 보이는 경향: 2024~26 논문의 상당수가 LLM 에이전트·강화학습이고, 이들은 대개 짧은 실거래 시험이나
  비용을 단순화한 시험만 있다. 15분봉 BTC 방향 예측에 바로 쓸 근거는 적다.

## 쓸모 있는 논문 (요약은 직접 쓴 것)

| # | 논문 | 핵심 | 시장·주기 | 표본 외·비용 | 우리 자료로 가능? | 우리 시험과 대조 |
|---|---|---|---|---|---|---|
| 1 | Borri et al., "Cryptocurrency as an Investable Asset Class: Coming of Age" (2025) [arXiv 2510.14435](https://arxiv.org/abs/2510.14435) | 암호화폐 10가지 정형화된 사실: 위험조정 성과는 전통 자산과 비슷, 소수 요인으로 설명, **점프가 잦고 큼** | 다종 코인, 일·주 | 서술 연구 | 예 | 점프 → 손절이 갭으로 미끄러짐(우리 −7% 비상 손절이 −7.9%로 체결된 사례와 일치). 크기 조절이 필수 |
| 2 | Wood, Roberts, Zohren, "Trading with the Momentum Transformer" (2021) [2112.08534](https://arxiv.org/abs/2112.08534) | 시계열 모멘텀(TSMOM)·되돌림 벤치마크보다 나은 주의(attention) 모델, 비용 반영 후에도 우위, 레짐 적응 | 선물 다종, **일봉** | 표본 외 + 비용 | 부분(단일 자산) | 일봉 추세만 돈을 벌었다는 우리 결과와 같은 방향. 딥러닝 부분은 우리 ML(AUC 0.52~0.56) 실패로 보아 보류 |
| 3 | Saly-Kaufmann et al., "Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance" (2026) [2603.01820](https://arxiv.org/abs/2603.01820) | 일봉 선물(2010~25)에서 포지션 크기를 직접 학습, 손익분기 비용 분석 포함 | 선물 다종, 일봉 | 표본 외, 비용 손익분기 | 부분 | 핵심 교훈: **포지션 크기(변동성 맞춤)** 가 예측보다 중요 |
| 4 | Mason et al., "Enhanced Momentum with Momentum Transformers" (2024) [2412.12516](https://arxiv.org/abs/2412.12516) | 2번을 주식에 확장, 연 4% 수준·샤프 1.1 | 주식 일봉 | 표본 외 | — | 일봉 모멘텀 우위가 작고 꾸준하다는 정도의 근거 |
| 5 | Li et al., "Follow the Leader: Network Momentum" (2025) [2501.07135](https://arxiv.org/abs/2501.07135) | 시장 간 선행·후행으로 추세 신호 보강 | 상품선물 | 부트스트랩 | 아니오(다른 시장 제외 — 사용자 지시) | 해당 없음 |
| 6 | Lis et al., "Overreaction as an indicator for momentum in algorithmic trading" (2026) [2602.18912](https://arxiv.org/abs/2602.18912) | **변동성 대비 극단 수익률(과잉반응)** 을 비용 기준과 함께 정의, 초단기엔 ML이, 그 이후엔 행동적 모멘텀(지속)이 우세 | AAPL 1~15분 | 통계검정 | 예(15분 수익률/변동성) | 새로움. 우리 15분 신호는 지표 교차였지 "변동성 대비 충격 후 지속"은 따로 안 봤다 |
| 7 | Albers et al., "Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets" (2021) [2108.09750](https://arxiv.org/abs/2108.09750) | 초단기(500ms) 수익률은 설명되지만, 시장가(테이커) 전략 손익은 **거래소 수수료 체계가 좌우** | BTC 다거래소, 초 단위 | 비용 반영 | 아니오(호가 자료 없음) | 우리 결론(수수료가 15분 신호를 죽인다)과 같은 메시지. 메이커 체결이 중요 |
| 8 | Zhang et al., "Turnover-Adjusted Information Ratio" (2021) [2105.10306](https://arxiv.org/abs/2105.10306) | 거래 회전을 줄이면 비용 반영 정보비율이 오히려 오를 수 있음 | 이론·모의 | — | 예(원리) | 하루 1~2회 목표와 충돌하는 지점의 이론적 근거 |
| 9 | Gao et al., "Coarse Graining Reveals a Fluctuation-theorem-like Asymmetry in Financial Markets" (2026) [2604.14962](https://arxiv.org/abs/2604.14962) | 대칭 익절/손절에서 롱·숏 보유시간 분포가 꼬리에서 비대칭(방향성 비대칭) | 지수·주식·코인 | 서술 | 예 | 새로움(방향 신호라기보다 청산 시간 설계 단서). 이번엔 보류 |
| 10 | Khanna et al., "History Rhymes: Macro-Contextual Retrieval" (2025) [2511.09754](https://arxiv.org/abs/2511.09754) | 과거 유사 **거시 레짐**을 찾아 예측, 표본 외에서 유일하게 플러스 | S&P 일봉 | 표본 외(1년) | 부분(거시 자료 없음) | 사용자의 "모양 외우기"와 같은 계열이지만, 가격 모양이 아니라 **레짐 맥락**을 검색. 우리 모양 검색(방향 53%) 실패와 대비됨 → 레짐 선(regime-lines) 시험과 연결 |
| 11 | Andreoletti, "Forecast collapse of transformer-based models under squared loss" (2026) [2604.00064](https://arxiv.org/abs/2604.00064) | 조건부 구조가 약하면 최적 예측은 "변화 없음"이고, 복잡한 모델은 잡음만 키움 | FX 고빈도 | 이론+실험 | — | 우리 ML·모양 기억 실패(AUC 0.52, 53%)를 설명. 15분 방향 예측에 복잡 모델을 더 쌓지 말 근거 |
| 12 | Koshiyama & Firoozye, "Avoiding Backtesting Overfitting by Covariance-Penalties" (2019) [1905.05023](https://arxiv.org/abs/1905.05023) | 변수 수·자료 양에 따라 백테스트 성과를 깎아 보정 | 방법론 | — | 예 | 우리 Bonferroni·사전 등록과 같은 목적. 30일 맞춤(fit30) 실패가 전형적 사례 |
| 13 | Jabbar & Jalil, "A Comprehensive Analysis of ML Models for Algorithmic Trading of Bitcoin" (2024) [2407.18334](https://arxiv.org/abs/2407.18334) | 41개 ML 모델, RF·SGD가 낫다고 주장 | BTC | 짧은 전진 시험 | 예 | **이미 반박됨**: 우리 ML(5개 씨드 평균 포함) AUC 0.52~0.56, 수수료 못 넘음 |
| 14 | Pabuccu et al., "Forecasting the movements of Bitcoin prices: ML" (2023) [2303.04642](https://arxiv.org/abs/2303.04642) | 분류 정확도 비교, 비용·매매 성과 없음 | BTC 일봉 | 정확도만 | 예 | 같은 이유로 해당 없음 |
| 15 | Michańków et al., "Mean Absolute Directional Loss" (2023) [2309.10546](https://arxiv.org/abs/2309.10546) | 방향+크기를 함께 보는 손실함수로 하이퍼파라미터 선택 | BTC·원유 | 표본 외 | 예 | ML을 다시 할 때만 의미. 보류 |
| 16 | Tang et al., "Forecasting Volatility with ML and Rough Volatility: Crypto-Winter" (2023) [2311.04727](https://arxiv.org/abs/2311.04727) | 변동성은 단순 모형(파라미터 5개)으로도 잘 예측됨 | 코인 | 표본 외 | 예 | 방향은 못 맞혀도 **변동성(크기)은 맞힌다**는 우리 관찰(모양→크기)과 일치 → 크기 조절에 활용 |
| 17 | Wang, "Exploring Microstructural Dynamics in Crypto LOB" (2025) [2506.05764](https://arxiv.org/abs/2506.05764) | 호가 자료에서는 모델 층보다 입력 전처리가 중요 | BTC 호가 100ms | — | 아니오 | 해당 없음 |
| 18 | LLM 에이전트 계열: QuantAgent (2509.09995), TiMi (2510.04787), TrustTrade (2603.22567), CryptoTrade (2407.09546) | LLM이 지표·패턴을 읽고 매매 결정 | 다양 | 대체로 짧은 시험 | — | 사용자가 이전에 취소한 "매번 Claude에게 묻기"와 같은 계열. 견고한 표본 외 근거가 약해 이번 설계에서 제외 |

## 우리 결과와의 대조 (오늘 밤 5,859개 변형)

- 이미 반박된 것: 15분 지표 교차(SCR 포함), ML 분류(논문 13·14), 가격 모양 검색(10의 가격판), 30일 맞춤 가중치(12의 경고 사례).
- 살아 있는 근거(우리 시험): 일봉 T6D 방향(무작위 대비 보정 후 유의), 뉴욕장 첫 60분 돌파를 일봉 방향으로만(S-NY|GT, 4기간 모두 5배 플러스, 시각 맞춘 무작위 대비 합산 p<0.0003), 프리미엄 이탈 되돌림(H5, 무작위보다 나음).
- 논문이 보태는 새 요소: (6) 변동성 대비 충격 후 지속, (3)(16) 변동성 맞춤 크기 조절, (7)(8) 회전·수수료 관리.

## 설계: 3층 시스템 QL (Quant Layered)

1. **방향층(일봉, 하루 한 번 09:00 KST 확정)** — 일봉 신호 3개: T6D 상태, 20일 돈치안 중간선 위/아래, 30일 수익률 부호(TSMOM).
   방향 = T6D만(기준) / 3개 중 2개 이상 / 3개 모두. 방향이 없으면 쉰다.
2. **진입층(15분봉, 방향과 같은 쪽만, 하루 최대 2회, 한 번에 1포지션)**
   - 뉴욕장 첫 60분 범위 돌파(S-NY, 검증된 유일한 단타 신호)
   - 런던장·아시아장 첫 60분 범위 돌파(같은 규칙, 방향 필터 없이는 실패했던 것 → 필터 효과를 시험)
   - 과잉반응 지속(논문 6): 15분 수익률이 최근 변동성의 3배 이상이고 방향층과 같은 쪽이면 다음 봉 진입
   - 청산: 손절 = 범위 반대편(과잉반응은 충격 봉 반대 끝), 익절 = 범위 높이(과잉반응은 위험의 1.5배), 8시간 시간 청산
3. **크기층** — 레버리지는 5배 고정, 증거금 비율 f = min(1, 2% / (5 × 손절 거리%)) → 손절 한 번이 계좌 −2%. 비교용으로 전액(f=1).
4. **집행층** — (FA) 시장가 진입·청산, (FB) 신호가 지정가 진입(메이커 0.02%, 관통 체결만) + 시장가 청산. 추격 한도 1%/3%.

하루 1회 이상과의 충돌(솔직하게): 뉴욕 돌파만으로는 하루 0.31회. 런던·아시아·과잉반응을 더하면 횟수는 늘지만, 이들은 방향 필터 없이 이미 실패했다.
수수료(시장가 왕복 0.1% × 5배 = 매매당 증거금 −0.5%)가 그대로라 횟수를 늘릴수록 기대값이 줄 가능성이 높다. 시험에서 "횟수 vs 성과"를 그대로 보여준다.

예상 범위(과거 수치 기준, 약속 아님): S-NY|GT 전액 5배는 기간별 +20%~+225%, 최대 낙폭 47~69%. 위험 2% 크기 조절을 하면 수익·낙폭 모두 대략 1/3~1/5로 줄어
낙폭 10~25% 수준이 목표다.

## 검증 계획 (사전 등록 문서에서 고정)

- 구간: 최근 30일, 그 전 30일, 6개월, 1년, 2022-01~2024-06, 2024-07~2025-10, 손대지 않은 2020-03~2021-10. 고정 규칙이라 맞출 변수 없음 → 걷기 시험은 90일 단위 연속 구간의 플러스 비율로 본다.
- 선물+펀딩, 5배 청산 모델, 시장가·지정가 두 방식, 추격 한도, 시각 맞춘 무작위 기준, Bonferroni(이번 변형 수와 오늘 누적 수 함께 표기), 자체 레드팀(1봉 지연, 수수료 1.5배, 연도별).
