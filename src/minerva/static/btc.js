// "비트코인" 탭. /api/btc 결과(배치가 15분마다 갱신)를 그린다. 표시 전용이다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: 2 }));
  const level = (n) => (n ? `${n}단` : "-");
  const signal = (s) => (s ? `${s.level}단 · ${esc(s.at)} · ${num(s.close)}` : "-");
  const CHIP = { 상방: "good", 하방: "warn", 중립: "" };

  // 추격 한도(사용자 결정 2026-11-08): 신호 봉 종가에서 1% 넘게 벌어지면 진입 안 함, 큰 움직임 조건이면 3%.
  const gateHtml = (g) => !g ? "" : `<div class="detail" style="margin-top:8px"><b>추격 한도 ±${num(g.pct)}%</b> `
    + (g.big ? `(15분·4시간·일봉 모두 ${esc(g.big_way)} → 큰 움직임 조건, 3%까지 허용)` : "(큰 움직임 조건 아님, 기본 1%)")
    + ` · 지금 15분봉 종가 ${num(g.ref_close)} 기준 롱은 ${num(g.long_max)}까지, 숏은 ${num(g.short_min)}까지만 진입. 신호가 나오면 그 신호 봉 종가 기준으로 다시 잰다.</div>`;

  const render = (d) => {
    $("btc-meta").innerHTML = `<div class="detail">분석 ${esc(d.generated_at.slice(0, 16).replace("T", " "))} KST · 15분마다 갱신 · 마감된 봉 기준</div>` +
      `<ul style="margin:8px 0 0 16px;padding:0">${d.summary.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>` + gateHtml(d.entry_gate);
    $("btc-frames").innerHTML = `<table class="acct"><thead><tr>
        <th>봉</th><th>방향</th><th>종가</th><th>Stoch</th><th>CCI</th><th>RSI</th><th>롱</th><th>숏</th>
      </tr></thead><tbody>${d.frames.map((f) => `<tr>
        <td>${esc(f.name)}</td><td><span class="chip ${CHIP[f.direction]}">${esc(f.direction)}</span></td>
        <td>${num(f.close)}</td><td>${num(f.stoch)}</td><td>${num(f.cci)}</td><td>${num(f.rsi)}</td>
        <td>${level(f.long_level)}</td><td>${level(f.short_level)}</td></tr>`).join("")}</tbody></table>`;
    $("btc-last").innerHTML = `<table class="acct"><thead><tr><th>봉</th><th>마지막 롱 신호</th><th>마지막 숏 신호</th></tr></thead>
      <tbody>${d.frames.map((f) => `<tr><td>${esc(f.name)}</td><td>${signal(f.last_long)}</td><td>${signal(f.last_short)}</td></tr>`).join("")}</tbody></table>`;
  };

  const load = () => fetch("/api/btc").then((r) => r.json()).then((d) => {
    if (d.status === "pending") $("btc-meta").textContent = d.message;
    else render(d);
  }).catch(() => { $("btc-meta").textContent = "분석 데이터를 불러오지 못했어요"; });

  load();
  setInterval(load, 60000);
})();
