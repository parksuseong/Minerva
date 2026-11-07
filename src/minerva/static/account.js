// "계좌" 탭. /api/account (minerva.kis가 쓴 한국투자증권 스냅숏)와
// /api/account/binance (minerva.binance가 쓴 바이낸스 선물 스냅숏, USDT)를 그린다.
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
  let boss = "사장님";
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

  const pct = (v) => `${v > 0 ? "+" : ""}${num(v)}%`;
  const gain = (v, p) => signed(v, `${v > 0 ? "+" : ""}${won(v)} (${pct(p)})`);

  function returns(r) {
    if (!r) return "";
    const today = r.today == null ? "-" : gain(r.today, r.today_pct);
    return `
      <div class="row"><span>누적 수익률 (${esc(r.since)}~, 원금 ${won(r.invested)})</span>
        <b class="px">${gain(r.total, r.total_pct)}</b></div>
      <div class="row"><span>오늘 수익률 (전일 마감 대비)</span><span class="px">${today}</span></div>
      <div class="row"><span>봇 손익 (자동매매 실현+평가)</span>
        <span class="px">${gain(r.bot, r.bot_pct)}</span></div>`;
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
        <span class="px">${won(d.stock_value)} · ${signed(d.pnl, won(d.pnl))}</span></div>${returns(d.returns)}${err}`;
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

  const usdt = (v) => (v == null ? "-" : `${num(v)} USDT`);
  const ugain = (v, p) => signed(v, `${v > 0 ? "+" : ""}${usdt(v)} (${pct(p)})`);
  const BN = ["bn-positions", "bn-orders", "bn-trades"];

  function renderBinance(d) {
    if (d.status !== "ok" && !d.updated_at) {
      $("bn-status").textContent = d.message || d.error || "바이낸스 계좌 정보가 없어요";
      BN.forEach((id) => { $(id).textContent = "-"; });
      return;
    }
    const r = d.returns;
    const err = d.error ? `<div class="state hold">⚠️ 최근 갱신 실패 (${esc(d.error_at)}): ${esc(d.error)}</div>` : "";
    const inc = d.income ? `<div class="row"><span>시드 이후 실현손익 · 펀딩 · 수수료</span>
      <span class="px">${signed(d.income.realized, usdt(d.income.realized))} · ${signed(d.income.funding,
        usdt(d.income.funding))} · ${signed(d.income.commission, usdt(d.income.commission))}</span></div>` : "";
    const ret = r ? `
      <div class="row"><span>누적 수익률 (${esc(r.since)}~, 시드 ${usdt(r.invested)})</span>
        <b class="px">${ugain(r.total, r.total_pct)}</b></div>
      <div class="row"><span>오늘 수익률 (전일 마지막 대비, KST)</span>
        <span class="px">${r.today == null ? "-" : ugain(r.today, r.today_pct)}</span></div>` : "";
    $("bn-status").innerHTML = `
      <div class="mini">${esc(d.broker)} · ${esc(d.market)} · 기준 ${esc(d.updated_at)}</div>
      <div class="row"><span>평가액 (지갑 + 미실현)</span><b class="px">${usdt(d.equity)}</b></div>
      <div class="row"><span>지갑 잔고 · 주문 가능</span>
        <span class="px">${usdt(d.wallet)} · ${usdt(d.available)}</span></div>
      <div class="row"><span>미실현 손익</span><span class="px">${signed(d.unrealized, usdt(d.unrealized))}</span></div>
      ${ret}${inc}${err}`;
    $("bn-positions").innerHTML = table([["symbol", "종목"], ["side", "방향"], ["qty", "수량"],
      ["entry", "진입가"], ["mark", "현재가"], ["liq", "청산가"],
      ["leverage", "레버리지", (v, x) => `${v}배 ${esc(x.margin_type)}`],
      ["pnl", "미실현손익", (v, x) => signed(v, `${num(v)} (${x.pnl_pct == null ? "-" : num(x.pnl_pct)}%)`)]],
    d.positions);
    $("bn-orders").innerHTML = table([["time", "시각"], ["symbol", "종목"], ["side", "구분"],
      ["type", "유형"], ["qty", "수량"], ["price", "가격"], ["status", "상태"]], d.orders);
    $("bn-trades").innerHTML = table([["time", "시각"], ["side", "구분"], ["qty", "수량"],
      ["price", "체결가"], ["realized", "실현손익", (v) => signed(v, num(v))], ["fee", "수수료"]], d.trades);
  }

  function loadBinance() {
    fetch("/api/account/binance", { headers: { "X-Minerva-Token": store("get") } })
      .then((r) => (r.status === 403 ? null : r.json()))
      .then((d) => {
        if (d) renderBinance(d);
        else { $("bn-status").textContent = `계좌는 ${boss} 전용이에요`; }
      })
      .catch(() => { $("bn-status").textContent = "바이낸스 계좌 정보를 불러오지 못했어요"; });
  }

  function load() {
    loadBinance();
    fetch("/api/account", { headers: { "X-Minerva-Token": store("get") } })
      .then((r) => {
        if (r.status === 403) {
          const had = store("get");
          store("del");
          lock(had ? "토큰이 맞지 않아요" : `계좌는 ${boss} 전용이에요`);
          return null;
        }
        return r.json();
      })
      .then((d) => { if (d) render(d); })
      .catch(() => { $("acct-status").textContent = "계좌 정보를 불러오지 못했어요"; });
  }

  fetch("/api/profile").then((r) => r.json()).then((p) => { boss = p.boss || boss; })
    .catch(() => { /* 기본 호칭 유지 */ });
  load();
})();
