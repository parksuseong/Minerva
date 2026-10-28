# Minerva

> TODO: 프로젝트 한 줄 설명 (무엇을, 누구를 위해 만드는가)

## 개요
- 언어: Python
- 상태: 초기 세팅 단계

## 디렉토리 구조
```
Minerva/
├── CLAUDE.md            # 이 파일 — Claude 작업 지시문 (항상 로드됨)
├── .claude/
│   ├── settings.json    # 권한/도구 설정 (팀 공유)
│   ├── rules/           # 주제별 세부 규칙 (자동 로드)
│   ├── skills/          # 반복 작업 절차 (/스킬명 으로 호출)
│   └── agents/          # 전문 서브에이전트
├── src/minerva/         # FastAPI 앱 (main.py, static/)
│   └── strategy/        # 지표(indicators.py), 점검 CLI(report.py), 백테스트(backtest.py)
└── tests/               # pytest 테스트
```

## 자주 쓰는 명령
- 설치: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- 테스트: `.venv/bin/pytest`
- 린트/포맷: `.venv/bin/ruff check . && .venv/bin/ruff format .`
- 서버: systemd 서비스 `minerva`로 HTTPS 443에서 상시 실행 (부팅 시 자동 시작, `deploy/minerva.service`)
  - 코드 반영: `sudo systemctl restart minerva` / 상태·로그: `systemctl status minerva`, `journalctl -u minerva -n 50`
  - 주소: https://minerva-ai.duckdns.org/ (DuckDNS → 3.210.65.41). 인증서는 Let's Encrypt, `certbot-renew.timer`가 갱신하고 `deploy/certbot-deploy-hook.sh`가 저장소 밖 `~/.minerva/tls/`로 복사·재시작한다 (커밋 금지).

## 작업 원칙
- 변경 전에 관련 코드를 먼저 읽고, 기존 스타일을 따른다.
- 작은 단위로 작업하고, 작업 후 테스트/린트를 돌려 확인한다.
- 검증(테스트·린트·리뷰) → 개선 → 커밋 루프를 스스로 반복한다. 코드뿐 아니라 지시문(CLAUDE.md, rules, skills)과 docs도 계속 다듬는다.
- 커밋은 요청 없이 해도 된다 (작은 단위, `/commit` 스킬 절차). 푸시·파괴적 git 작업은 확인 후.
- 전용 서브에이전트가 있으면 좋을 상황이면 사용자에게 먼저 제안한다.
- 모르는 요구사항은 추측하지 말고 질문한다.
- 세부 규칙은 `.claude/rules/` 참고:
  - `code-style.md` — 코딩 스타일
  - `git.md` — 브랜치/커밋 규칙
  - `testing.md` — 테스트 규칙

## 트레이딩 플레이북
투자 질문(종목·지수·코인·변동성)에는 감이 아니라 플레이북으로 답한다.
- `/market-check` — 종합 점검 절차 (진입점)
- `vol-regime` — 변동성 레짐 R0~R3 → 비중 한도 (방향 예측 아님)
- `momentum-tm` — 삼중 모멘텀 L1~L3, 검증된 TM 눌림 규칙(유일한 진입 규칙)
- `risk-rules` — 비중·손절·레버리지 한도 (모든 신호보다 우선)
- `backtest-guard` — 패턴·승률 검증 규칙 (사후 맞춤 금지)
- 검증 현황: `docs/backtests/2026-10-28-results.md` (가설 A~N 판정, 다음 후보 P1~P4)
- 에이전트: `strategist`(분석), `risk-officer`(반박·검증), `backtester`(규칙 검증), `trade-auditor`(실전 주문·보유 점검)
- 지표 값은 `.venv/bin/python -m minerva.strategy.report <심볼...>`로 계산한다.
- 규칙 검증은 `.venv/bin/python -m minerva.strategy.backtest <심볼...> --help` 참고.

## 일일 분석과 화면 (프로세스 분리)
| 프로세스 | 하는 일 | 쓰는 곳 |
|---|---|---|
| `minerva` (웹) | 화면·읽기 전용 API만. 계산·주문·증권사 키 없음 | `src/minerva/main.py`, `routes.py` |
| `minerva-daily.timer` (배치) | 09:10·16:30 KST 종목 분석 → `~/.minerva/daily/latest.json` + 텔레그램 | `strategy/daily.py` |
| `minerva-account.timer` (조회) | 평일 08~19시 30초마다 한국투자증권 잔고·주문·체결 → `~/.minerva/account/latest.json` | `kis.py`. 키 `~/.minerva/kis.env`는 이 프로세스만 읽는다. 조회만 |
| `minerva-trade-{preflight,open,watch,reconcile,preview}.timer` (주문) | 08:50 장전 시장가 주문(휴장일 건너뜀)·09:02 매도 뒤 매수 · 장중 30초 손절 감시 · 08:40 사전 점검 · 15:45 잔고 대사 · 16:45 다음 날 주문 미리보기 → `~/.minerva/trade/` | `trade.py`. `mode` 파일이 `live`일 때만 실주문, `HALT` 파일이 있으면 정지 |

- 대상: 코스피 지수·시총 상위 200(보통주, `KOSPI_TOP`, 사용자 결정 2026-10-28), SPY·나스닥100 E-mini 선물(NQ=F)·나스닥 시총 상위 10, 비트코인, 국내 인버스 1종(KODEX 인버스)·KODEX 200. 곱버스·레버리지 상품은 매매도 화면도 하지 않는다(사용자 결정 2026-10-28). 미국 인버스와 종목 자체 숏은 실행할 수 없어 뺐다(사용자 결정 2026-10-28, `ASSET_SIDES`). 레버리지 상품은 다루지 않는다. 순위는 `strategy/universe.py`가 네이버 증권에서 하루 1번 받는다.
- 종목별 "검증됨/부적합/미검증"은 `verify()`가 그 종목 차트로 롱·숏 규칙을 백테스트해 정한다(거래 30회 이상 + 표본 안·밖 우위 > 0, 주 1회 재계산, `~/.minerva/data/verify/`). 종목이 많아 우연 통과가 섞일 수 있다. 판정은 왕복 비용(국내 주식 0.30%, ETF·지수 0.10%, `round_trip_cost`)을 뺀 우위로 한다. 그다음 무작위 타이밍 대비 p값에 다중 검정(BH, `FDR_Q`=0.20, 사용자 결정 2026-10-28)을 걸어 못 넘으면 "미검증"으로 내린다.
- `actions`(API `/api/signals`): 오늘 낼 주문 — LONG_ENTRY·LONG_EXIT·BUY/SELL(국내 인버스). 자동매매가 읽는 목록이다. 진입은 다음 거래일 시가, 손절은 장중 스톱 주문 가정.
- 진행 중인 봉은 버린다(`completed`).
- 실제 투자는 국내만 한다(사용자 결정). 미국·코인 신호는 참고용이다.
- 롱 매수 관문(조건 ⑤): VIX 주봉 5주선 < 10주선, 지난주 마감 기준(사용자 결정 2026-10-28, 일봉 50일선 레짐 대체). `regime.gated_regime`이 관문을 R0·R1/R2·R3로 바꿔 기존 필터·비중 계산에 그대로 넣는다. 레짐은 그날 한국 마감 때 알 수 있던 전날 미국 VIX까지만 쓴다.
- 주문 대상은 국내·검증됨 롱 + KODEX 200(코스피 신호) + 국내 인버스(1개만). 비중×레짐 계수: 개별주 20%, 지수 ETF 40%, 인버스 합계 20%. 최대 10종목(사용자 결정). 돈이 모자라면 p값 낮은 순 → 비용 후 우위 큰 순.
- 거래량 한도: 한 번 매수 주문은 그 종목 20일 평균 거래대금(`adv`)의 1%까지(`LIQUIDITY_CAP`). 거래가 적은 종목에 시장가로 크게 넣으면 가격이 밀린다.
- 대기 자금 파킹: 09:02에 남은 현금(평가액 2% 제외)을 TIGER CD금리투자KIS(357870)에 넣고(`park.json`), 매수 신호가 오면 08:50에 필요한 만큼 판다(UNPARK). 포트폴리오 백테스트상 현금이 80~95% 놀아서 넣었다.
- 리밸런싱: 봇 종목이 한도의 1.5배를 넘으면 한도까지 판다(TRIM). 교체(ROTATE): 못 산 신호의 p값이 보유 종목의 절반 이하이고 우위 차이가 비용보다 크며 5일 이상 보유한 종목만 판다. 교체·리밸런싱은 백테스트로 검증하지 않은 운용 규칙이다(ROADMAP 13). 사용자가 직접 산 종목은 건드리지 않는다(`positions.json`에 있는 봇 보유분만).
- 매도 먼저: 국내 인버스 매수 신호가 뜨면 봇의 롱을 모두 판다(사용자 결정). 매도가 있는 날은 08:50에 매도만 내고, 매수는 09:02에 매도 대금으로 낸다.
- 자금은 유동적(증액 가능, 사용자 2026-10-28). 비중은 매번 그날 평가액 기준이다.
- 실전 전환은 사용자가 직접 한다: `echo live > ~/.minerva/trade/mode` (Claude가 대신 켜지 않는다). 기본은 모의 실행.
- 봇 손익(청산 `closed.jsonl` + 보유 평가손익)이 고점보다 평가액의 15% 넘게 줄면 `NO_ENTRY` 파일이 생겨 신규 매수만 멈춘다(손절·청산은 계속). 입출금에는 반응하지 않는다.
- 성과: 15:45 대사가 체결된 봇 매도를 `closed.jsonl`에 적고(비용 차감), 금 17:00 `python -m minerva.perf weekly`가 주간 리포트를 보낸다.
- 백업: 평일 16:10 `python -m minerva.backup`이 `~/.minerva/{trade,account/latest.json,daily}`를 tar.gz로 묶어 텔레그램으로 보낸다(키·토큰 파일 제외, 서버엔 7개 보관). 코드는 GitHub.
- 운용 점검: 평일 16:00 `python -m minerva.audit --notify` (`minerva-trade-audit.timer`). ❌·⚠️가 나오면 `trade-auditor` 에이전트로 원인을 찾는다.
- 주문 경로 확인: 장중에 `python -m minerva.trade selftest --yes` (KODEX 인버스 1주 매수→매도, 실주문).
- 텔레그램으로 "매매중지"가 오면 `touch ~/.minerva/trade/HALT`, "매매재개"가 오면 그 파일을 지운다.
- 재주문: 통신 오류·응답 유실만 10초 간격으로 최대 3회 다시 낸다. 다시 내기 전에 주문 내역을 조회해 중복을 막는다. 증권사 거부(rt_cd≠0)는 다시 내지 않는다.
- 💼 계좌 탭과 `/api/account`는 공개다(사용자 요청 2026-10-28). 잠그려면 `~/.minerva/account/view.token`에 토큰을 넣는다(헤더 `X-Minerva-Token`).
- 화면 탭: 🎯 오늘 할 일 / 📈 종목 분석 / 💼 계좌 / 🏢 오피스 (`#today` 같은 해시로 이동).

## 오피스 패널 (🏢 오피스 탭)
- Claude와 서브에이전트가 일하는 모습을 보여준다. 코드: `static/office.{js,css}`, API `src/minerva/office.py`. 훅 `scripts/office_hook.py`가 `~/.minerva/office/events.jsonl`에 이벤트를 쌓는다.
- 기록하는 건 이벤트 종류, 도구 이름, 짧은 라벨(Bash 설명, 파일 이름)뿐이다. 프롬프트·명령 원문·파일 내용은 기록하지 않는다.
- 훅 등록: `python3 scripts/install_office_hooks.py` (사용자 설정 `~/.claude/settings.json`에 추가)
- 기본은 토큰 없이 공개(사용자 요청). 잠그려면 서비스 환경변수 `MINERVA_OFFICE_TOKEN`을 설정한다.
- 서버가 기동하면 `src/minerva/notify.py`가 텔레그램으로 알린다. 설정은 저장소 밖 `~/.minerva/telegram.env` (커밋 금지).

## 환경 메모
- EC2에서 작업 중이며 매일 02:00~08:00(KST) 인스턴스가 꺼진다. 오래 걸리는 작업은 그 전에 끝나도록 한다.
