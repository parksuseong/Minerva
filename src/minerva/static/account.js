// "계좌" 탭. /api/account (minerva.kis가 쓴 한국투자증권 스냅숏)를 그린다.
// 서버에 열람 토큰이 설정돼 있으면 입력을 받고, 한 번 입력하면 이 기기에 기억한다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: 2 }));
  const won = (v) => (v == null ? "-" : `${num(Math.round(v))}원`);
  const signed = (v, s) => `<span class="${v > 0 ? "up" : v < 0 ? "down" : ""}">${s}</span>`;
  const KEY = "minerva-account-token";
  let mem = "";
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

  function table(cols, items) {
    if (!items || !items.length) return "없음";
    const head = cols.map(([, t]) => `<th>${esc(t)}</th>`).join("");
    const body = items.map((r) => `<tr>${cols.map(([k, , f]) =>
      `<td>${f ? f(r[k], r) : typeof r[k] === "number" ? num(r[k]) : esc(r[k])}</td>`).join("")}</tr>`).join("");
    return `<table class="acct"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function lock(msg) {
    ["acct-balance", "acct-orders", "acct-trades"].forEach((id) => { $(id).textContent = "-"; });
    $("acct-status").innerHTML = `<form id="acct-unlock" class="filters">
      <span>🔒 ${esc(msg)}</span>
      <input type="password" id="acct-token" placeholder="열람 토큰" autocomplete="off" required>
      <button type="submit">열기</button></form>`;
    $("acct-unlock").onsubmit = (e) => {
      e.preventDefault();
      store("set", $("acct-token").value.trim());
      load();
    };
  }

  function render(d) {
    if (d.status === "not_connected") {
      $("acct-status").textContent = d.message;
      ["acct-balance", "acct-orders", "acct-trades"].forEach((id) => { $(id).textContent = "연결 후 표시돼요"; });
      return;
    }
    const err = d.error ? `<div class="state hold">⚠️ 최근 갱신 실패 (${esc(d.error_at)}): ${esc(d.error)}</div>` : "";
    $("acct-status").innerHTML = `
      <div class="mini">${esc(d.broker)} · ${esc(d.market)} · ${esc(d.account)} · 기준 ${esc(d.updated_at)}</div>
      <div class="row"><span>총평가</span><b class="px">${won(d.equity)}</b></div>
      <div class="row"><span>예수금 (D+2)</span><span class="px">${won(d.cash)} (${won(d.cash_d2)})</span></div>
      <div class="row"><span>주식 평가 · 평가손익</span>
        <span class="px">${won(d.stock_value)} · ${signed(d.pnl, won(d.pnl))}</span></div>${err}`;
    $("acct-balance").innerHTML = table([["name", "종목"], ["qty", "수량"], ["avg", "평균가"],
      ["price", "현재가"], ["pnl", "평가손익", (v, r) => signed(v, `${num(v)} (${num(r.pnl_pct)}%)`)]],
      d.positions);
    $("acct-orders").innerHTML = table([["time", "시각", (v) => esc(String(v).slice(11))],
      ["name", "종목"], ["side", "구분"], ["qty", "수량"], ["price", "가격"], ["status", "상태"]],
      d.orders);
    $("acct-trades").innerHTML = `<div class="mini">최근 ${esc(d.trade_days)}일</div>` +
      table([["time", "시각"], ["name", "종목"], ["side", "구분"], ["qty", "수량"],
        ["price", "체결가"]], d.trades);
  }

  function load() {
    fetch("/api/account", { headers: { "X-Minerva-Token": store("get") } })
      .then((r) => {
        if (r.status === 403) {
          const had = store("get");
          store("del");
          lock(had ? "토큰이 맞지 않아요" : "계좌는 사장님 전용이에요");
          return null;
        }
        return r.json();
      })
      .then((d) => { if (d) render(d); })
      .catch(() => { $("acct-status").textContent = "계좌 정보를 불러오지 못했어요"; });
  }

  load();
})();
