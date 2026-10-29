// "🤖 자동매매" 카드. /api/trade (minerva.trade가 쓴 상태 파일)를 그린다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: 0 }));
  const ICON = { dry: "🧪", sent: "✅", rejected: "⛔", failed: "❌" };
  const RESULT = { dry: "모의", sent: "접수", rejected: "거부", failed: "실패" };

  function render(d) {
    $("trade-mode").textContent = d.halted ? "⏸ 매매 중지" : d.mode === "live" ? "🔴 실전" : "🧪 모의 실행";
    const pos = Object.entries(d.positions || {});
    $("trade-positions").innerHTML = pos.length ? pos.map(([code, p]) =>
      `<div class="row"><span><b>${esc(p.name)}</b> <span class="ticker">${esc(code)} · 롱</span></span>
        <span class="mini">진입 ${esc(p.opened)} · 손절 ${p.stop ? num(p.stop) + "원" : "체결 대기"}</span></div>`).join("")
      : "봇이 보유 중인 종목 없음 · 평일 08:50 장전 시장가 주문, 장중 30초 손절 감시";
    const orders = (d.orders || []).slice(0, 10);
    $("trade-orders").innerHTML = orders.length ? `<div class="mini" style="margin-top:8px">최근 주문</div>` +
      orders.map((o) => `<div class="act"><span class="tag ${o.side === "buy" ? "BUY" : "SELL"}">${o.side === "buy" ? "매수" : "매도"}</span>
        <div><b>${esc(o.name)}</b> <span class="mini">${o.qty}주 · ${ICON[o.result] || ""} ${RESULT[o.result] || esc(o.result)}</span></div>
        <span></span><div class="detail">${esc(String(o.at || "").replace("T", " ").slice(0, 16))}${o.msg ? " · " + esc(o.msg) : ""}</div></div>`).join("")
      : "";
  }

  const load = () => fetch("/api/trade").then((r) => r.json()).then(render)
    .catch(() => { $("trade-positions").textContent = "자동매매 상태를 불러오지 못했어요"; });
  load();
  setInterval(load, 30000);
})();
