#!/bin/bash
# 정기 요원 작업 (cron, 평일). 읽기 전용 요원을 `claude -p`로 돌리고 결과를 텔레그램으로 보낸다.
#   review   16:15  🦅 운용 감시관: 오늘 주문·체결·손절 점검. 이상 있을 때만 알린다
#   briefing 17:05  🦊 전략가: 내일 나올 수 있는 신호와 대기 가격
# 시험: DRY=1 scripts/routine.sh briefing
# 요원은 파일을 고치거나 주문을 낼 수 없다 (허용 도구: 읽기·검색·점검 명령만).
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
name="$1"
case "$name" in
  review) agent=trade-auditor ;;
  briefing) agent=strategist ;;
  *) echo "usage: $0 review|briefing" >&2; exit 2 ;;
esac
# 요원에게는 명령 실행 권한을 주지 않는다. 필요한 점검 결과는 여기서 미리 만들어 넘긴다.
context="$(
  echo "## 운용 점검 결과 (python -m minerva.audit)"
  .venv/bin/python -m minerva.audit 2>&1 | tail -40
  echo "## 계좌 요약 (~/.minerva/account/latest.json)"
  .venv/bin/python -c 'import json,os; d=json.load(open(os.path.expanduser("~/.minerva/account/latest.json"))); print(json.dumps({k: d.get(k) for k in ("updated_at","equity","cash","stock_value","positions","orders","trades")}, ensure_ascii=False)[:4000])'
)"
prompt="$(sed "s/{{DATE}}/$(TZ=Asia/Seoul date +%F)/g" "scripts/routines/$name.md")

$context"
# --strict-mcp-config: 텔레그램 플러그인을 띄우지 않는다. 같은 봇을 두 프로세스가 받으면
# 상시 세션의 텔레그램 연결이 끊긴다 (2026-10-29 실제로 끊김).
out="$(timeout 900 claude -p "$prompt" --agent "$agent" --permission-mode dontAsk \
  --strict-mcp-config --allowedTools Read Grep Glob \
  --disallowedTools Bash Edit Write NotebookEdit \
  --max-budget-usd 2 --no-session-persistence < /dev/null 2>/dev/null)"
status=$?
out="$(printf '%s' "$out" | sed '/^---$/,$d')"  # 요원이 덧붙인 메모는 보내지 않는다
send() {
  if [ -n "${DRY:-}" ]; then cat; echo; return; fi  # DRY=1: 보내지 않고 출력만 (시험용)
  .venv/bin/python -c 'import sys; from minerva import notify; notify.send(sys.stdin.read()[:3800])'
}
if [ $status -ne 0 ] || [ -z "$out" ]; then
  printf '⚠️ 정기 요원 작업(%s) 실패 (코드 %s)\n%s' "$name" "$status" "${out: -500}" | send
elif [ "$name" = review ] && [ "$(printf '%s' "$out" | head -1 | tr -d '[:space:]')" = "OK" ]; then
  exit 0  # 이상 없으면 조용히 (16:00 운용 점검 알림이 이미 간다)
else
  printf '%s' "$out" | send
fi
