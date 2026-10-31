// 메인 화면 하단 오피스 패널. 전역 이름 충돌을 피하려고 즉시 실행 함수로 감싼다.
(() => {
// 사용자 호칭은 /api/profile에서 받는다 (실명을 코드에 넣지 않으려고)
let BOSS = "사장님";
// [이모지, 이름, 에이전트 타입, 맡은 일]
const CAST = {
  main: ["🦉", "미네르바", "Claude 본체",
    "지시를 받아 일을 나누고, 요원을 투입하고, 결과를 검토해 보고해요. 코드 작성·커밋·푸시도 직접 해요."],
  strategist: ["🦊", "전략가", "strategist",
    "플레이북(변동성 레짐 → 삼중 모멘텀 → 리스크 룰) 순서로 종목을 점검해 롱·숏·관망 결론과 진입·무효화 가격을 내요."],
  "red-team": ["🐻", "레드팀", "red-team",
    "전략가의 결론을 깨뜨릴 근거를 찾아요. 미확정 신호, 시간축 충돌, 비중·손절 위반, 사후 맞춤을 점검해 통과·조건부·기각을 판정해요."],
  backtester: ["🐿️", "백테스터", "backtester",
    "매매 규칙을 과거 데이터로 검증해요. 가설을 먼저 고정하고 표본 밖 성과, 무조건 기대수익 대비 우위(edge), 최악 사례를 보고해요."],
  "trade-auditor": ["🦅", "운용 감시관", "trade-auditor",
    "실계좌에서 봇이 낸 주문·보유·손절이 규칙(국내·검증 종목·비중·손절가·중복 주문)대로였는지 점검하고 이상 원인을 찾아요. 주문은 내지 않아요."],
  reviewer: ["🐧", "리뷰어", "reviewer",
    "바뀐 코드를 독립적으로 리뷰해요. 버그, 보안(비밀값 노출), 빠진 테스트, 규칙 위반을 찾아요."],
  Explore: ["🐕", "탐색가", "Explore",
    "코드와 파일을 넓게 뒤져서 필요한 위치와 사실만 찾아와요. 파일을 고치지 않아요."],
  Plan: ["🦝", "설계자", "Plan",
    "구현 계획을 세워요. 바꿔야 할 파일, 순서, 위험 요소를 정리해요."],
  "general-purpose": ["🐱", "만능 요원", "general-purpose",
    "조사·분석·여러 단계 작업을 맡는 범용 요원이에요. 새 요원 이름이 아직 안 잡힌 세션에서는 전략가·레드팀 역할도 이 책상에서 해요."],
  fork: ["🦉", "분신", "fork", "미네르바가 지금까지의 맥락을 그대로 가지고 나눠진 분신이에요. 큰 일을 병렬로 처리해요."],
};
const TOOL_ICON = { Bash: "⌨️", Read: "📖", Edit: "✏️", Write: "📝", Grep: "🔍", Glob: "🔍",
  Agent: "📣", Task: "📣", WebFetch: "🌐", WebSearch: "🌐", Skill: "📚", reply: "💬",
  Artifact: "🖼️" };
const TOKEN_KEY = "minerva-office-token";
function store(op, value) {
  try {
    if (op === "get") return localStorage.getItem(TOKEN_KEY) || "";
    if (op === "set") localStorage.setItem(TOKEN_KEY, value);
    if (op === "del") localStorage.removeItem(TOKEN_KEY);
  } catch (e) { /* 사파리 사생활 보호 모드 등에서는 저장이 안 돼도 동작은 해야 한다 */ }
  return "";
}
const params = new URLSearchParams(location.search);
let token = params.get("token") || store("get");
if (params.has("token")) {
  store("set", token);
  // 주소창·공유 링크에 토큰이 남지 않게 지운다
  params.delete("token");
  const q = params.toString();
  history.replaceState(null, "", location.pathname + (q ? "?" + q : "") + location.hash);
}
const recent = new Map();  // desk key -> 최근 작업 라벨
const desks = new Map();   // key -> {el, state}
const pending = [];        // Agent 호출 후 아직 agent_id가 안 붙은 책상
let after = 0, stats = { tools: 0, agents: 0, commits: 0 }, failures = 0;

function cast(type) { return CAST[type] || ["🐰", type || "요원", type || "", "임시로 투입된 요원이에요."]; }
function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html; return e; }
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;",
  ">": "&gt;", '"': "&quot;" }[c])); }

function desk(key, type) {
  if (desks.has(key)) return desks.get(key);
  const [emoji, name, sub] = cast(type);
  const node = el("div", "of-desk of-sleeping");
  node.innerHTML = `<div class="of-bubble">출근!</div><div class="of-who">${emoji}</div>
    <div class="of-table"><div class="of-monitor"></div><div class="of-prop"></div></div>
    <div class="of-name">${esc(name)}<small>${esc(sub)}</small></div>`;
  const d = { el: node, type, timer: null, key };
  node.addEventListener("click", () => showInfo(d));
  key === "main" ? document.getElementById("of-desks").prepend(node)
                 : document.getElementById("of-desks").append(node);
  desks.set(key, d);
  return d;
}

function setState(d, state, text, prop) {
  d.el.classList.remove("of-working", "of-sleeping", "of-err");
  d.el.classList.add("of-" + state);
  if (text !== undefined) d.el.querySelector(".of-bubble").textContent = text;
  d.el.querySelector(".of-prop").textContent = prop || "";
  clearTimeout(d.timer);
  if (state === "working") d.timer = setTimeout(() => setState(d, "sleeping", "☕ 잠깐 쉬는 중", "☕"), 45000);
}

function showInfo(d) {
  const [emoji, name, sub, role] = cast(d.type);
  const info = document.getElementById("of-info");
  const items = (recent.get(d.key) || []).map(t => `<li>${esc(t)}</li>`).join("") || "<li>아직 기록 없음</li>";
  info.innerHTML = `<button class="of-close" aria-label="닫기">✕</button>
    <h2>${emoji} ${esc(name)} <small>${esc(sub)}</small></h2><p>${esc(role)}</p>
    <h3>최근 작업</h3><ol>${items}</ol>`;
  info.hidden = false;
  info.querySelector(".of-close").onclick = () => { info.hidden = true; };
}

function remember(key, text) {
  const list = recent.get(key) || [];
  list.unshift(text);
  recent.set(key, list.slice(0, 6));
}

function roster() {
  const box = document.getElementById("of-roster-list");
  box.innerHTML = Object.entries(CAST).filter(([k]) => k !== "fork").map(([, [emoji, name, sub, role]]) =>
    `<div class="of-member"><div class="of-member-who">${emoji}</div><div><b>${esc(name)}</b>
     <small>${esc(sub)}</small><p>${esc(role)}</p></div></div>`).join("");
}

function retire(key) {
  const d = desks.get(key);
  if (!d || key === "main") return;
  setState(d, "sleeping", "✅ 보고 완료! 퇴근", "🎒");
  setTimeout(() => { d.el.classList.add("of-leaving"); setTimeout(() => d.el.remove(), 1000); }, 6000);
  desks.delete(key);
}

function toast(msg) { const t = el("div", "of-toast", esc(msg)); document.getElementById("office").append(t);
  setTimeout(() => t.remove(), 3000); }

function log(ev, who, icon, text, bad) {
  const li = el("li", bad ? "of-bad" : "");
  const t = new Date(ev.ts * 1000).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit",
    timeZone: "Asia/Seoul", hour12: false });
  li.innerHTML = `<time>${t}</time><span>${icon}</span><span>${esc(who)} · ${esc(text)}</span>`;
  const feed = document.getElementById("of-feed");
  feed.prepend(li);
  while (feed.children.length > 80) feed.lastChild.remove();
}

function keyFor(ev) {
  if (!ev.agent_id) return "main";
  if (!desks.has(ev.agent_id)) {
    const i = pending.findIndex(p => p.type === ev.agent_type);
    if (i >= 0) { const p = pending.splice(i, 1)[0]; desks.set(ev.agent_id, desks.get(p.key));
      desks.delete(p.key); }
  }
  return ev.agent_id;
}

function handle(ev, live) {
  const key = keyFor(ev);
  // 화면에 없는 내부 에이전트의 종료 이벤트는 무시한다 (빈 책상이 잠깐 생기는 것 방지)
  if (ev.event === "SubagentStop" && !desks.has(key)) return;
  const d = desk(key, key === "main" ? "main" : ev.agent_type);
  const who = cast(d.type)[1];
  switch (ev.event) {
    case "UserPromptSubmit": {
      const boss = document.getElementById("of-boss");
      if (live) { boss.classList.add("of-walk"); setTimeout(() => boss.classList.remove("of-walk"), 2500); }
      setState(d, "working", "📄 새 지시 접수!", "📄");
      log(ev, BOSS, "📄", "새 지시");
      break;
    }
    case "PreToolUse": {
      const icon = TOOL_ICON[ev.tool] || "🛠️";
      setState(d, "working", `${icon} ${ev.label || ev.tool}`);
      remember(key, `${icon} ${ev.label || ev.tool}`);
      stats.tools++;
      if (ev.tool === "Agent" || ev.tool === "Task") {
        const pk = "pending-" + ev.seq;
        const nd = desk(pk, ev.spawn_type);
        setState(nd, "working", `🚶 ${ev.label || "출근 중"}`);
        pending.push({ key: pk, type: ev.spawn_type });
        stats.agents++;
        if (live) toast(`${cast(ev.spawn_type)[1]} 투입!`);
      }
      if (/커밋|commit|push|푸시/i.test(ev.label || "")) { stats.commits++; if (live) toast("🎉 커밋·푸시!"); }
      log(ev, who, icon, ev.label || ev.tool);
      break;
    }
    case "PostToolUse": case "PostToolUseFailure":
      if (ev.ok === false) { setState(d, "err", `❌ ${ev.label || ev.tool} 실패`);
        log(ev, who, "❌", `${ev.label || ev.tool} 실패`, true); }
      break;
    case "SubagentStart":
      desk(key, ev.agent_type); setState(desks.get(key), "working", "💼 업무 시작");
      break;
    case "SubagentStop":
      log(ev, who, "✅", "보고 완료"); retire(key);
      break;
    case "Stop":
      if (key === "main") { setState(d, "sleeping", "💤 대기 중… 다음 지시 기다리는 중", "💤");
        log(ev, who, "💤", "턴 종료"); }
      else retire(key);
      break;
    case "Notification":
      setState(d, "working", "✋ 확인이 필요해요!", "✋"); log(ev, who, "✋", `${BOSS} 확인 요청`);
      break;
    case "SessionStart":
      setState(d, "sleeping", "☀️ 출근 완료"); log(ev, who, "☀️", "세션 시작");
      break;
  }
  for (const k of Object.keys(stats)) document.getElementById("of-s-" + k).textContent = stats[k];
}

async function poll(first) {
  try {
    const r = await fetch(`/api/office/events?after=${after}&token=${encodeURIComponent(token)}`);
    if (r.status === 403) { store("del"); lock(); return; }
    const data = await r.json();
    if (data.last < after) after = 0;
    for (const ev of data.events) handle(ev, !first);
    after = data.last;
    document.getElementById("of-live").classList.add("of-on");
    failures = 0;
  } catch (e) {
    document.getElementById("of-live").classList.remove("of-on");
    failures++;
  }
  setTimeout(() => poll(false), failures ? 5000 : 1500);
}

function lock() {
  const app = document.getElementById("of-app");
  app.innerHTML = `<form class="of-locked" id="of-unlock">
    <p>🔒 오피스는 ${esc(BOSS)} 전용이에요. 접근 토큰을 한 번만 입력하면 이 기기에서 기억해요.</p>
    <input type="password" id="of-token" placeholder="접근 토큰" autocomplete="off" required>
    <button type="submit">입장</button></form>`;
  document.getElementById("of-unlock").onsubmit = (e) => {
    e.preventDefault();
    store("set", document.getElementById("of-token").value.trim());
    location.reload();
  };
}

function tick() {
  const now = new Date();
  document.getElementById("of-clock").textContent =
    now.toLocaleString("ko-KR", { timeZone: "Asia/Seoul", hour12: false, month: "numeric",
      day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const h = Number(now.toLocaleString("en-US", { timeZone: "Asia/Seoul", hour: "numeric", hour12: false }));
  document.getElementById("of-floor").classList.toggle("of-night", h >= 19 || h < 7);
}

roster();
desk("main", "main");
fetch("/api/profile").then((r) => r.json()).then((p) => {
  BOSS = p.boss || BOSS;
  document.querySelector("#of-boss .of-name").textContent = BOSS;
}).catch(() => { /* 호칭을 못 받아도 기본값으로 동작한다 */ });
setInterval(tick, 1000); tick();
poll(true);
})();
