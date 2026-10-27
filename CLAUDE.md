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
└── tests/               # pytest 테스트
```

## 자주 쓰는 명령
- 설치: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- 테스트: `.venv/bin/pytest`
- 린트/포맷: `.venv/bin/ruff check . && .venv/bin/ruff format .`
- 서버 실행 (HTTPS 443): `sudo .venv/bin/uvicorn minerva.main:app --host 0.0.0.0 --port 443 --ssl-certfile ~/.minerva/tls/cert.pem --ssl-keyfile ~/.minerva/tls/key.pem`
  - 인증서는 저장소 밖 `~/.minerva/tls/`에 있다 (자체서명, 커밋 금지). 자동 시작용 systemd 유닛: `deploy/minerva.service` (미설치)

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

## 환경 메모
- EC2에서 작업 중이며 매일 01:00~08:00(KST) 인스턴스가 꺼진다. 오래 걸리는 작업은 그 전에 끝나도록 한다.
