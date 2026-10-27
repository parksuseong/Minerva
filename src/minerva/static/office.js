// 메인 화면 하단 오피스 패널. 전역 이름 충돌을 피하려고 즉시 실행 함수로 감싼다.
(() => {
const CAST = {
  main: ["🦉", "미네르바", "Claude 본체"],
  strategist: ["🦊", "전략가", "strategist"],
  "risk-officer": ["🐻", "리스크 담당", "risk-officer"],
  backtester: ["🐿️", "백테스터", "backtester"],
  reviewer: ["🐧", "리뷰어", "reviewer"],
  Explore: ["🐕", "탐색가", "Explore"],
  Plan: ["🦝", "설계자", "Plan"],
  "general-purpose": ["🐱", "만능 요원", "general-purpose"],
  fork: ["🦉", "분신", "fork"],
};
const TOOL_ICON = { Bash: "⌨️", Read: "📖", Edit: "✏️", Write: "📝", Grep: "🔍", Glob: "🔍",
  Agent: "📣", Task: "📣", WebFetch: "🌐", WebSearch: "🌐", Skill: "📚", reply: "💬",
  Artifact: "🖼️" };
const params = new URLSearchParams(location.search);
const token = params.get("token") || "";
const desks = new Map();   // key -> {el, state}
const pending = [];        // Agent 호출 후 아직 agent_id가 안 붙은 책상
let after = 0, stats = { tools: 0, agents: 0, commits: 0 }, failures = 0;

function cast(type) { return CAST[type] || ["🐰", type || "요원", type || ""]; }
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
  const d = { el: node, type, timer: null };
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
  const d = desk(key, key === "main" ? "main" : ev.agent_type);
  const who = cast(d.type)[1];
  switch (ev.event) {
    case "UserPromptSubmit": {
      const boss = document.getElementById("of-boss");
      if (live) { boss.classList.add("of-walk"); setTimeout(() => boss.classList.remove("of-walk"), 2500); }
      setState(d, "working", "📄 새 지시 접수!", "📄");
      log(ev, "사장님", "📄", "새 지시");
      break;
    }
    case "PreToolUse": {
      const icon = TOOL_ICON[ev.tool] || "🛠️";
      setState(d, "working", `${icon} ${ev.label || ev.tool}`);
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
      setState(d, "working", "✋ 확인이 필요해요!", "✋"); log(ev, who, "✋", "사장님 확인 요청");
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
    if (r.status === 403) {
      document.getElementById("of-app").innerHTML =
        '<div class="of-locked">🔒 토큰이 필요해요. 주소 끝에 ?token=... 을 붙여 주세요.</div>';
      return;
    }
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

function tick() {
  const now = new Date();
  document.getElementById("of-clock").textContent =
    now.toLocaleString("ko-KR", { timeZone: "Asia/Seoul", hour12: false, month: "numeric",
      day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const h = Number(now.toLocaleString("en-US", { timeZone: "Asia/Seoul", hour: "numeric", hour12: false }));
  document.getElementById("of-floor").classList.toggle("of-night", h >= 19 || h < 7);
}

desk("main", "main");
setInterval(tick, 1000); tick();
poll(true);
})();
