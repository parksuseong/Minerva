// "계좌" 탭. /api/account (minerva.kis가 쓴 한국투자증권 스냅숏)와
// /api/account/binance (minerva.binance가 쓴 바이낸스 선물 스냅숏, USDT)를 그린다.
// 서버에 열람 토큰이 설정돼 있으면 입력을 받고, 한 번 입력하면 이 기기에 기억한다.
(() => {
  const $ = (id) => document.getElementById(id);
  const { esc } = window.MinervaUI;
  const num = (v) => window.MinervaUI.num(v, 2);
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
    if (d.live === false) {
      $("bn-status").innerHTML = `
        <div class="mini">${esc(d.broker)} · ${esc(d.market)} · 기준 ${esc(d.updated_at)}</div>
        <div class="state hold">실전 자동매매 시작 전이에요. 지금 수동 매매는 기록하지 않아요.
          시작하면 그때 잔고를 시드로 수익률·자산 기록을 새로 시작해요.</div>
        <div class="row"><span>지갑 잔고 · 주문 가능</span>
          <span class="px">${usdt(d.wallet)} · ${usdt(d.available)}</span></div>`;
      BN.forEach((id) => { $(id).textContent = "실전 시작 전"; });
      return;
    }
    const r = d.returns;
    const err = d.error ? `<div class="state hold">⚠️ 최근 갱신 실패 (${esc(d.error_at)}): ${esc(d.error)}</div>` : "";
    const ret = r ? `
      <div class="row"><span>누적 수익률 (수수료 차감, ${esc(r.since)}~, 시드 ${usdt(r.invested)})</span>
        <b class="px">${ugain(r.total, r.total_pct)}</b></div>
      <div class="row"><span>오늘 수익률 (수수료 차감, 전일 마지막 대비, KST)</span>
        <span class="px">${r.today == null ? "-" : ugain(r.today, r.today_pct)}</span></div>` : "";
    const fr = d.fee_rate;
    const closeFee = d.close_fee_est || 0;
    let breakdown = "";
    if (d.income) {
      const gross = d.income.realized + (d.unrealized || 0);
      const fees = d.income.commission - closeFee;
      breakdown = `
      <div class="row"><span>매매손익 (수수료 전, 실현+미실현)</span>
        <span class="px">${signed(gross, usdt(gross))}</span></div>
      <div class="row"><span>수수료 (낸 것 ${usdt(-d.income.commission)} + 닫을 때 예상 ${usdt(closeFee)})</span>
        <span class="px">${signed(fees, usdt(fees))}</span></div>
      <div class="row"><span>펀딩비</span><span class="px">${signed(d.income.funding, usdt(d.income.funding))}</span></div>`;
    }
    const rate = fr ? `<div class="mini">수수료 요율: 지정가(메이커) ${num(fr.maker * 100)}% · 시장가(테이커)
      ${num(fr.taker * 100)}% · 순평가액 = 지갑 + 미실현 − 지금 닫을 때 수수료(테이커)</div>` : "";
    $("bn-status").innerHTML = `
      <div class="mini">${esc(d.broker)} · ${esc(d.market)} · 기준 ${esc(d.updated_at)}</div>
      <div class="row"><span>순평가액 (수수료 차감)</span><b class="px">${usdt(d.net_equity ?? d.equity)}</b></div>
      <div class="row"><span>지갑 잔고 · 주문 가능</span>
        <span class="px">${usdt(d.wallet)} · ${usdt(d.available)}</span></div>
      ${ret}${breakdown}${rate}${err}`;
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

  // 자산 차트: 시드 위는 상승색, 아래는 하락색. 시각은 서버가 KST로 옮겨 둔다(UTC로 그린다).
  const pad = (n) => String(n).padStart(2, "0");
  const kst = (t, withDate) => {
    const x = new Date(t * 1000);
    const hm = `${pad(x.getUTCHours())}:${pad(x.getUTCMinutes())}`;
    return withDate ? `${pad(x.getUTCMonth() + 1)}/${pad(x.getUTCDate())} ${hm}` : hm;
  };
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  let chart = null;
  let line = null;
  let seedLine = null;

  function drawSeries(d) {
    const LC = window.LightweightCharts;
    const pts = (d.points || []).map((p) => ({ time: p.t, value: p.v }));
    if (d.live === false) {
      $("bn-chart-note").textContent = "실전 자동매매를 시작하면 그때부터 그려요";
      if (line) line.setData([]);
      return;
    }
    $("bn-chart-note").textContent = pts.length < 2
      ? "기록이 쌓이는 중이에요 (5분마다 한 점)"
      : `5분마다 한 점 · 가로선 = 시드 ${usdt(d.seed)} · 수수료를 뺀 순평가액`;
    if (!LC || !pts.length || !$("bn-chart").clientWidth) return;
    const up = css("--up") || "#dc2626", down = css("--down") || "#2563eb";
    const text = css("--muted") || "#6b7280", grid = css("--line") || "#e5e7eb";
    const look = {
      layout: { background: { type: "solid", color: css("--card") || "#fff" }, textColor: text, fontSize: 11 },
      grid: { vertLines: { visible: false }, horzLines: { color: grid } },
      rightPriceScale: { borderColor: grid },
      timeScale: { borderColor: grid },
    };
    if (!chart) {
      chart = LC.createChart($("bn-chart"), { autoSize: true, ...look,
        localization: { locale: "ko-KR", timeFormatter: (t) => kst(t, true),
          priceFormatter: (v) => `${num(v)}` },
        crosshair: { mode: LC.CrosshairMode.Magnet },
        timeScale: { ...look.timeScale, timeVisible: true, secondsVisible: false,
          tickMarkFormatter: (t, type) => (type <= 2 ? kst(t, true).slice(0, 5) : kst(t, false)) } });
      line = chart.addSeries(LC.BaselineSeries, { lineWidth: 2, priceLineVisible: false });
    }
    chart.applyOptions(look);
    const seed = d.seed ?? pts[0].value;
    line.applyOptions({ baseValue: { type: "price", price: seed },
      topLineColor: up, topFillColor1: `${up}33`, topFillColor2: `${up}05`,
      bottomLineColor: down, bottomFillColor1: `${down}05`, bottomFillColor2: `${down}33` });
    if (seedLine) line.removePriceLine(seedLine);
    seedLine = line.createPriceLine({ price: seed, color: text, lineWidth: 1, lineStyle: 2,
      axisLabelVisible: true, title: "시드" });
    line.setData(pts);
  }

  function loadSeries() {
    fetch("/api/account/binance/series", { headers: { "X-Minerva-Token": store("get") } })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d) drawSeries(d); })
      .catch(() => { $("bn-chart-note").textContent = "자산 기록을 불러오지 못했어요"; });
  }

  function loadBinance() {
    loadSeries();
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

  // 주식/바이낸스 중 하나만 보인다. 고른 쪽은 이 기기에 기억한다.
  const PICK = "minerva-account-pick";
  function pick(which) {
    $("acct-kr").hidden = which !== "kr";
    $("acct-bn").hidden = which !== "bn";
    document.querySelectorAll("#acct-pick button").forEach((b) => {
      const on = b.dataset.acct === which;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    });
    try { localStorage.setItem(PICK, which); } catch (e) { /* 저장이 막혀도 전환은 된다 */ }
    // 숨겨진 채로 그린 차트는 폭이 0이라 보일 때 다시 그린다
    if (which === "bn") loadSeries();
  }
  document.querySelectorAll("#acct-pick button").forEach((b) => {
    b.onclick = () => pick(b.dataset.acct);
  });
  let saved = "kr";
  try { saved = localStorage.getItem(PICK) === "bn" ? "bn" : "kr"; } catch (e) { /* 기본 주식 */ }
  pick(saved);

  const tabBtn = document.querySelector('[data-tab="account"]');
  if (tabBtn) tabBtn.addEventListener("click", () => setTimeout(loadSeries, 0));

  // 계좌 탭이 보일 때만 30초마다 다시 불러온다(서버는 1분마다 갱신).
  setInterval(() => {
    if (!document.hidden && !$("tab-account").hidden) load();
  }, 30000);

  fetch("/api/profile").then((r) => r.json()).then((p) => { boss = p.boss || boss; })
    .catch(() => { /* 기본 호칭 유지 */ });
  load();
})();
