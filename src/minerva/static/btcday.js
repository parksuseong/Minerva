// "오늘 할 일" 탭 맨 위의 비트코인 일일 분석 카드. /api/btc(SCR 봉별 신호)와
// /api/btc/ml(규칙별 모의 추적, 주문 없음)을 한 장으로 요약한다. 표시 전용이다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v, d = 2) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: d }));
  const pct = (v) => (v == null ? "-" : `${v > 0 ? "+" : ""}${Number(v).toFixed(2)}%`);
  const CHIP = { 상방: "good", 하방: "warn", 중립: "" };

  // 15분봉 [t, o, h, l, c, ...]에서 마지막 종가와 24시간 전 종가
  const change24h = (bars) => {
    if (!bars || bars.length < 97) return null;
    const last = bars[bars.length - 1];
    const prev = bars[bars.length - 97];
    return { close: last[4], pct: (last[4] / prev[4] - 1) * 100 };
  };

  const scrHtml = (d) => {
    if (!d || d.status === "pending" || !d.frames) return `<div class="placeholder">SCR 분석 대기 중</div>`;
    const chips = d.frames.map((f) => `${esc(f.name)} <span class="chip ${CHIP[f.direction] || ""}">${esc(f.direction)}</span>`).join(" · ");
    return `<div style="margin-top:6px">${chips}</div>
      <ul style="margin:6px 0 0 16px;padding:0">${(d.summary || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ul>`;
  };

  const ruleRow = (r) => {
    const o = r.open;
    const now = !o ? "대기"
      : o.kind === "보유 중"
        ? `<span class="chip ${o.side === "롱" ? "good" : "warn"}">${esc(o.side)}</span> ${num(o.price)} · 지금 ${pct(o.pnl1 == null ? null : o.pnl1 * 5)} (5배)`
        : `${esc(o.side || "")} 주문 대기`;
    return `<tr><td><b>${esc(r.id)}</b></td><td>${now}</td>
      <td>${r.fills ? `${r.wins}/${r.fills}` : "-"}</td><td>${pct(r.x5)}</td></tr>`;
  };

  const mlHtml = (d) => {
    if (!d || d.status === "pending" || !d.rules) return `<div class="placeholder">모의 추적 대기 중</div>`;
    return `<table class="acct" style="margin-top:6px"><thead><tr><th>규칙</th><th>지금</th><th>승/끝난 매매</th><th>누적 5배</th></tr></thead>
      <tbody>${d.rules.map(ruleRow).join("")}</tbody></table>
      <p class="mini">모의 추적이에요(주문 없음). ${esc(d.start)}부터 기록, 끝난 매매가 아직 적어 운일 수 있어요. 자세한 내용은 ₿ 비트코인 탭.</p>`;
  };

  const render = (scr, ml) => {
    const ch = change24h(ml && ml.chart && ml.chart.bars);
    const when = scr && scr.generated_at ? scr.generated_at.slice(0, 16).replace("T", " ") : "-";
    const price = ch ? `<b class="px">${num(ch.close)}</b> USDT · 24시간 ${pct(ch.pct)}` : "";
    $("btcday-body").innerHTML = `<div class="detail">${price} <span class="mini">· 분석 ${esc(when)} KST · 15분마다 갱신</span></div>
      <h3 style="margin:10px 0 0;font-size:14px">SCR 봉별 방향</h3>${scrHtml(scr)}
      <h3 style="margin:10px 0 0;font-size:14px">ML 규칙 모의 추적</h3>${mlHtml(ml)}`;
  };

  const get = (url) => fetch(url).then((r) => r.json()).catch(() => null);
  const load = () => Promise.all([get("/api/btc"), get("/api/btc/ml")]).then(([scr, ml]) => {
    if (!scr && !ml) $("btcday-body").textContent = "비트코인 분석을 불러오지 못했어요";
    else render(scr, ml);
  });

  load();
  setInterval(load, 60000);
})();
