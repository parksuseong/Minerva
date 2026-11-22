// "어드민" 탭. 계좌 탭과 같은 열람 토큰으로 잠근다(같은 저장 키라 계좌를 열었으면 같이 열린다).
// 자동매매 HALT·NO_ENTRY 스위치(/api/admin/status·halt·resume·clear-no-entry)와 접속 기록(/api/admin/access).
// 기록의 UA·경로·리퍼러는 접속자가 마음대로 보낼 수 있는 글자라 innerHTML을 쓰지 않고 textContent로만 넣는다.
(() => {
  const $ = (id) => document.getElementById(id);
  if (!$("tab-admin")) return;
  const KEY = "minerva-account-token";
  let mem = "";
  let login = false; // 방금 입력한 토큰이면 서버가 "직접 입력"으로 기록하게 표시한다
  const store = (op, value) => {
    if (op === "set") mem = value;
    if (op === "del") mem = "";
    try {
      if (op === "get") return localStorage.getItem(KEY) || mem;
      if (op === "set") localStorage.setItem(KEY, value);
      if (op === "del") localStorage.removeItem(KEY);
    } catch (e) { /* 저장이 막힌 브라우저에서도 이번 화면은 동작해야 한다 */ }
    return op === "get" ? mem : "";
  };

  // el("td", {class: "x"}, "글자", 자식…) — 글자는 늘 textContent로
  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (v == null || v === false) return;
      if (k === "class") n.className = v;
      else if (k === "onclick") n.onclick = v;
      else n.setAttribute(k, v === true ? "" : v);
    });
    kids.flat().forEach((c) => { if (c != null) n.append(c instanceof Node ? c : document.createTextNode(String(c))); });
    return n;
  };
  const fill = (id, ...kids) => { const box = $(id); box.replaceChildren(...kids.flat()); return box; };

  const api = (path, opts = {}) => {
    const headers = { "X-Minerva-Token": store("get") };
    if (login) headers["X-Minerva-Login"] = "1";
    return fetch(path, { ...opts, headers }).then((r) => {
      if (r.status === 403 || r.status === 429) {
        const had = store("get");
        if (r.status === 403) store("del");
        login = false;
        lock(r.status === 429 ? "너무 많이 틀렸어요. 15분 뒤에 다시 해 주세요"
          : had ? "토큰이 맞지 않아요" : "어드민은 열람 토큰이 있어야 열려요");
        return null;
      }
      if (!r.ok) throw new Error(String(r.status));
      login = false;
      return r.json();
    });
  };

  function lock(msg) {
    $("admin-body").hidden = true;
    const input = el("input", { type: "password", id: "admin-token", placeholder: "열람 토큰",
      autocomplete: "off", required: true });
    const form = el("form", { class: "filters" }, el("span", null, `🔒 ${msg}`), input,
      el("button", { type: "submit" }, "열기"));
    form.onsubmit = (e) => {
      e.preventDefault();
      store("set", input.value.trim());
      login = true;
      load();
    };
    fill("admin-lock", form).hidden = false;
  }

  const row = (label, value, cls) => el("div", { class: "row" }, el("span", null, label),
    el("span", { class: `px ${cls || ""}` }, value));

  function renderStatus(d) {
    const e = d.engine;
    const p = e.position;
    const pos = p ? `${p.symbol} ${p.side === "long" ? "롱" : "숏"} · 진입 ${p.entry ?? "-"} · 손절 ${p.stop ?? "-"}`
      + `${p.tp ? ` · 익절 ${p.tp}` : ""} · ${p.leverage}배` : "없음";
    const halts = (e.last_halts || []).join(", ");
    fill("admin-engine",
      row("모드", e.mode === "live" ? "🔴 실전" : "🧪 드라이런"),
      row("HALT (수동 정지)", e.halt ? `🛑 켜짐${e.halt_note ? ` · ${e.halt_note}` : ""}` : "꺼짐", e.halt ? "up" : ""),
      row("NO_ENTRY (자동 정지)", e.no_entry ? `⛔ ${e.no_entry}` : "없음", e.no_entry ? "up" : ""),
      row("포지션", pos),
      row("마지막 실행", e.last_run_at ? `${e.last_run_at} · ${e.last_decision || "-"}` : "-"),
      halts ? row("진입 정지 사유(마지막 실행)", halts) : null);

    const btns = [];
    if (e.halt) {
      btns.push(el("button", { type: "button", onclick: () => act("resume",
        "자동매매를 재개할까요?\nHALT 파일을 지워요. 다음 15분 실행부터 새 신호에 다시 들어가요.") }, "▶️ 재개 (HALT 해제)"));
    } else {
      btns.push(el("button", { type: "button", onclick: () => act("halt",
        "자동매매를 정지할까요?\n신규 진입만 멈춰요. 들고 있는 포지션·손절·따라가는 손절·반대 신호 청산은 계속해요.") }, "🛑 정지 (HALT)"));
    }
    if (e.no_entry) {
      btns.push(el("button", { type: "button", onclick: () => act("clear-no-entry",
        `NO_ENTRY를 지울까요?\n사유: ${e.no_entry}\n승률·평가액 규칙이 아직 걸려 있으면 다음 실행에 다시 생겨요.`) }, "🔓 NO_ENTRY 해제"));
    }
    fill("admin-buttons", btns);

    const acts = d.actions || [];
    const names = { halt: "🛑 정지", resume: "▶️ 재개", clear_no_entry: "🔓 NO_ENTRY 해제" };
    fill("admin-actions", acts.length ? table(["시각", "조작", "IP", "기기"], acts.map((a) => [
      a.at, `${names[a.action] || a.action}${a.changed ? "" : " (변화 없음)"}`, a.ip, a.ua ? uaShort(a.ua) : "-"]))
      : el("span", { class: "mini" }, "아직 없음"));
  }
  // 조작 기록 UA는 서버가 짧게 바꾸지 않아 앞부분만
  const uaShort = (ua) => (ua.length > 60 ? `${ua.slice(0, 60)}…` : ua);

  function act(name, question) {
    if (!window.confirm(question)) return;
    api(`/api/admin/${name}`, { method: "POST" })
      .then((d) => {
        if (!d) return;
        $("admin-msg").textContent = d.changed
          ? `완료${d.notified ? " · 텔레그램 알림 보냄" : " · 텔레그램 알림 실패"}`
          : "이미 그 상태예요 (변화 없음)";
        loadStatus();
      })
      .catch(() => { $("admin-msg").textContent = "실패했어요. 다시 해 주세요"; });
  }

  function table(head, rows, rowClass) {
    return el("div", { class: "scroll" }, el("table", { class: "acct" },
      el("thead", null, el("tr", null, head.map((h) => el("th", null, h)))),
      el("tbody", null, rows.map((r, i) => el("tr", { class: rowClass ? rowClass(i) : null },
        r.map((c) => el("td", null, c ?? "-")))))));
  }

  const filt = { auth_only: false, hide_me: false, hours: 24 };
  let rows = [];
  let total = 0;
  let more = false; // "더 보기"로 이어 붙이는 중이면 IP 요약은 그대로 둔다
  const AUTH = { ok: "✅ 성공", fail: "❌ 실패", blocked: "⛔ 차단" };

  function accessUrl(offset) {
    const q = new URLSearchParams({ auth_only: filt.auth_only, hide_me: filt.hide_me,
      hours: filt.hours, limit: 500, offset });
    return `/api/admin/access?${q}`;
  }

  function loadAccess(more) {
    $("admin-access-note").textContent = "불러오는 중…";
    api(accessUrl(more ? rows.length : 0))
      .then((d) => {
        if (!d) return;
        rows = more ? rows.concat(d.rows) : d.rows;
        total = d.total;
        renderAccess(d);
      })
      .catch(() => { $("admin-access-note").textContent = "접속 기록을 불러오지 못했어요"; });
  }

  function renderAccess(d) {
    $("admin-access-note").textContent = `내 IP ${d.me} (${d.me_place}) · ${total}건 중 ${rows.length}건`;
    fill("admin-access", rows.length ? table(["시각", "IP", "위치", "기기", "경로", "인증"], rows.map((r) => [
      r.at, r.mine ? `${r.ip} (나)` : r.ip, r.place, r.device, r.kind === "auth" ? `🔑 ${r.path}` : r.path, AUTH[r.auth] || "-"]))
      : el("span", { class: "mini" }, "기록 없음"));
    $("admin-access-more").hidden = rows.length >= total;
    if (!more) {
      fill("admin-ips", d.ips.length ? table(["IP", "위치", "기기", "횟수", "실패", "처음", "마지막"], d.ips.map((s) => [
        s.mine ? `${s.ip} (나)` : s.ip, s.place, s.device, s.count, s.fails, s.first, s.last]))
        : el("span", { class: "mini" }, "기록 없음"));
    }
  }

  function loadStatus() {
    return api("/api/admin/status").then((d) => {
      if (!d) return false;
      $("admin-lock").hidden = true;
      $("admin-body").hidden = false;
      renderStatus(d);
      return true;
    }).catch(() => { $("admin-msg").textContent = "상태를 불러오지 못했어요"; return false; });
  }

  function load() {
    loadStatus().then((ok) => { if (ok) { more = false; loadAccess(false); } });
  }

  // 필터
  document.querySelectorAll("#admin-filters [data-hours]").forEach((b) => {
    b.onclick = () => {
      filt.hours = Number(b.dataset.hours);
      document.querySelectorAll("#admin-filters [data-hours]").forEach((x) => {
        x.classList.toggle("on", x === b);
        x.setAttribute("aria-pressed", String(x === b));
      });
      more = false; loadAccess(false);
    };
  });
  $("admin-auth-only").onchange = (e) => { filt.auth_only = e.target.checked; more = false; loadAccess(false); };
  $("admin-hide-me").onchange = (e) => { filt.hide_me = e.target.checked; more = false; loadAccess(false); };
  $("admin-access-more").onclick = () => { more = true; loadAccess(true); };
  $("admin-refresh").onclick = () => load();

  // 어드민 탭이 보일 때만 불러온다(다른 탭 사용 중에는 접속 기록·위치 조회를 하지 않음)
  let loaded = false;
  const visible = () => !$("tab-admin").hidden;
  const maybe = () => { if (visible() && !loaded) { loaded = true; load(); } };
  window.addEventListener("hashchange", () => setTimeout(maybe, 0));
  maybe();
  // 열려 있으면 1분마다 상태만 다시(접속 기록은 새로고침 버튼으로)
  setInterval(() => { if (!document.hidden && visible() && !$("admin-body").hidden) loadStatus(); }, 60000);
})();
