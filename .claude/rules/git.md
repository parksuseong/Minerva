# Git 규칙

- `main`에 직접 커밋하지 않는다. 작업은 브랜치에서: `feat/<주제>`, `fix/<주제>`, `chore/<주제>`.
- 커밋 메시지는 Conventional Commits 형식: `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`, `test: ...`, `chore: ...`
  - 제목 50자 내외, 본문에는 "왜" 바꿨는지.
- 한 커밋엔 하나의 논리적 변경만.
- 커밋/푸시/PR은 사용자가 요청했을 때만 한다.
- force push, `reset --hard`, 브랜치 삭제 등 되돌리기 어려운 작업은 반드시 먼저 확인받는다.
