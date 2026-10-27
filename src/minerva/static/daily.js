// 메인 화면 "오늘의 전략" 카드. /api/daily 결과(매일 09:10·16:30 갱신)를 그린다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (v) => (v == null ? "-" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`);
  const num = (v) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: 2 }));

  function assetHtml(a) {
    if (a.error) return `<div class="asset"><b>${esc(a.name)}</b> <span class="detail">데이터 오류</span></div>`;
    const tm = a.tm || {};
    const badge = a.verified ? "" : '<span class="badge">미검증</span>';
    let state = `<div class="state">⏳ 대기 · 빠진 조건: ${esc((tm.missing || []).join(", "))}</div>`;
    if (tm.state === "진입 신호") {
      state = `<div class="state go">🟢 진입 신호 · ${esc(tm.entry)}</div>
        <div class="detail">손절 ${num(tm.stop)} (−10%) · 익절: Stoch 80 하향 이탈 · 최대 ${tm.max_days}거래일</div>`;
    } else if (tm.state === "보유 중") {
      state = `<div class="state hold">🔵 보유 중 ${pct(tm.pnl)} · ${tm.days_left}거래일 남음</div>
        <div class="detail">진입 ${esc(tm.entry_day)} @ ${num(tm.entry)} · 손절 ${num(tm.stop)} · ${esc(tm.exit_rule)}</div>`;
    }
    if (tm.state === "규칙 없음") {
      state = `<div class="state">🚫 ${esc((tm.missing || []).join(", "))}</div>`;
    }
    const info = a.kind
      ? `${a.mult > 0 ? "레버리지" : "인버스"} ${Math.abs(a.mult)}배 · 기초자산 ${esc(a.underlying)}${tm.note ? " · " + esc(tm.note) : ""}`
      : `${esc(a.date)} 종가 · Stoch ${a.stoch} · CCI ${a.cci} · RSI ${a.rsi} · 50일선 ${num(a.ma50)}`;
    const ch = a.change == null ? "" : `<span class="${a.change >= 0 ? "up" : "down"}">${pct(a.change)}</span>`;
    return `<div class="asset">
      <div class="asset-head"><div><b>${esc(a.name)}</b>${badge}</div>
        <div class="px">${num(a.close)} ${ch}</div></div>
      ${state}
      <div class="detail">${info}</div>
    </div>`;
  }

  fetch("/api/daily").then((r) => r.json()).then((d) => {
    if (d.status === "pending") { $("daily-meta").textContent = d.message; return; }
    const v = d.vix, f = d.fear;
    const fearCls = f.state === "없음" ? "" : "good";
    const regCls = v.regime === "R0" || v.regime === "R1" ? "good" : "warn";
    $("daily-meta").innerHTML = `<div class="regime">
        <span class="chip ${regCls}">VIX ${v.close} · ${esc(v.regime)}</span>
        <span class="chip">신규 비중 ${esc(v.size_limit)}</span>
        <span class="chip">레버리지 ${esc(v.leverage)}</span>
        <span class="chip ${fearCls}">공포 초입: ${esc(f.state)} (VIX/VIX3M ${pct(f.premium)})</span>
      </div><div class="detail">분석 시각 ${esc(d.generated_at.replace("T", " "))} KST · 매일 09:10·16:30 갱신</div>`;
    const products = (d.products || []).map(assetHtml).join("");
    $("daily-list").innerHTML = d.assets.map(assetHtml).join("") +
      (products ? `<h2 style="margin-top:16px">⚡ 레버리지 · 인버스 · 곱버스</h2>${products}` : "");
    $("daily-note").textContent = d.note;
  }).catch(() => { $("daily-meta").textContent = "분석 데이터를 불러오지 못했어요"; });
})();
