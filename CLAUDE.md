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
- 에이전트: `strategist`(분석), `risk-officer`(반박·검증), `backtester`(규칙 검증)
- 지표 값은 `.venv/bin/python -m minerva.strategy.report <심볼...>`로 계산한다.
- 규칙 검증은 `.venv/bin/python -m minerva.strategy.backtest <심볼...> --help` 참고.

## 일일 분석과 화면 (프로세스 분리)
| 프로세스 | 하는 일 | 쓰는 곳 |
|---|---|---|
| `minerva` (웹) | 화면·읽기 전용 API만. 계산·주문·증권사 키 없음 | `src/minerva/main.py`, `routes.py` |
| `minerva-daily.timer` (배치) | 09:10·16:30 KST 종목 분석 → `~/.minerva/daily/latest.json` + 텔레그램 | `strategy/daily.py` |
| `minerva-account.timer` (조회) | 평일 08~19시 10분마다 한국투자증권 잔고·주문·체결 → `~/.minerva/account/latest.json` | `kis.py`. 키 `~/.minerva/kis.env`는 이 프로세스만 읽는다. 주문 API 없음 |

- 대상: 코스피 지수·시총 상위 50(보통주), SPY·나스닥100 E-mini 선물(NQ=F)·나스닥 시총 상위 10, 비트코인, 인버스 4종(2배까지). 레버리지 상품은 다루지 않는다. 순위는 `strategy/universe.py`가 네이버 증권에서 하루 1번 받는다.
- 종목별 "검증됨/부적합/미검증"은 `verify()`가 그 종목 차트로 롱·숏 규칙을 백테스트해 정한다(거래 30회 이상 + 표본 안·밖 우위 > 0, 주 1회 재계산, `~/.minerva/data/verify/`). 종목이 많아 우연 통과가 섞일 수 있다.
- `actions`(API `/api/signals`): 오늘 낼 주문 — LONG_ENTRY·LONG_EXIT·SHORT_ENTRY·SHORT_EXIT·BUY/SELL(인버스). 자동매매가 읽는 목록이다. 진입은 다음 거래일 시가, 손절은 장중 스톱 주문 가정.
- 진행 중인 봉은 버린다(`completed`).
- 실제 투자는 국내만 한다(사용자 결정). 미국·코인 신호는 참고용이다.
- 💼 계좌 탭과 `/api/account`는 실계좌라 열람 토큰(`~/.minerva/account/view.token`, 헤더 `X-Minerva-Token`)이 있어야 열린다.
- 화면 탭: 🎯 오늘 할 일 / 📈 종목 분석 / 💼 계좌 / 🏢 오피스 (`#today` 같은 해시로 이동).

## 오피스 패널 (🏢 오피스 탭)
- Claude와 서브에이전트가 일하는 모습을 보여준다. 코드: `static/office.{js,css}`, API `src/minerva/office.py`. 훅 `scripts/office_hook.py`가 `~/.minerva/office/events.jsonl`에 이벤트를 쌓는다.
- 기록하는 건 이벤트 종류, 도구 이름, 짧은 라벨(Bash 설명, 파일 이름)뿐이다. 프롬프트·명령 원문·파일 내용은 기록하지 않는다.
- 훅 등록: `python3 scripts/install_office_hooks.py` (사용자 설정 `~/.claude/settings.json`에 추가)
- 기본은 토큰 없이 공개(사용자 요청). 잠그려면 서비스 환경변수 `MINERVA_OFFICE_TOKEN`을 설정한다.
- 서버가 기동하면 `src/minerva/notify.py`가 텔레그램으로 알린다. 설정은 저장소 밖 `~/.minerva/telegram.env` (커밋 금지).

## 환경 메모
- EC2에서 작업 중이며 매일 02:00~08:00(KST) 인스턴스가 꺼진다. 오래 걸리는 작업은 그 전에 끝나도록 한다.
