---
name: commit
description: 현재 변경사항을 규칙에 맞게 검사하고 커밋한다. 사용자가 커밋을 요청할 때 사용.
---

# 커밋 절차

1. `git status`, `git diff`로 변경사항 확인. 관련 없는 변경이 섞였으면 사용자에게 알린다.
2. 현재 브랜치가 `main`이면 `.claude/rules/git.md` 규칙에 따라 브랜치를 먼저 만든다.
3. 린트/테스트 실행 (`ruff check .`, `pytest`). 실패하면 커밋하지 말고 보고한다.
4. 비밀값(.env, 키, 토큰)이 포함되지 않았는지 확인.
5. Conventional Commits 형식으로 메시지 작성 후 커밋.
6. 푸시는 사용자가 요청한 경우에만.
