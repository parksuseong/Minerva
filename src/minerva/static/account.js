// "계좌" 탭. /api/account (증권사 연동 프로세스가 쓴 스냅숏)를 그린다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: 2 }));

  function table(cols, items) {
    if (!items || !items.length) return "없음";
    const head = cols.map(([, t]) => `<th>${esc(t)}</th>`).join("");
    const body = items.map((r) => `<tr>${cols.map(([k]) =>
      `<td>${typeof r[k] === "number" ? num(r[k]) : esc(r[k])}</td>`).join("")}</tr>`).join("");
    return `<table class="acct"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  fetch("/api/account").then((r) => r.json()).then((d) => {
    if (d.status === "not_connected") {
      $("acct-status").textContent = d.message;
      ["acct-balance", "acct-orders", "acct-trades"].forEach((id) => { $(id).textContent = "연결 후 표시돼요"; });
      return;
    }
    $("acct-status").innerHTML = `기준 ${esc(d.updated_at)} · 예수금 ${num(d.cash)} · 평가 ${num(d.equity)}`;
    $("acct-balance").innerHTML = table([["name", "종목"], ["qty", "수량"], ["avg", "평균가"],
      ["price", "현재가"], ["pnl", "평가손익"]], d.positions);
    $("acct-orders").innerHTML = table([["time", "시각"], ["name", "종목"], ["side", "구분"],
      ["qty", "수량"], ["price", "가격"], ["status", "상태"]], d.orders);
    $("acct-trades").innerHTML = table([["time", "시각"], ["name", "종목"], ["side", "구분"],
      ["qty", "수량"], ["price", "체결가"]], d.trades);
  }).catch(() => { $("acct-status").textContent = "계좌 정보를 불러오지 못했어요"; });
})();
