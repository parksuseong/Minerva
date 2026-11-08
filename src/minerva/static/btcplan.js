// 종목 분석 > 비트코인: "오늘의 판"과 "손절·크기 계산". 사용자 원칙 "덜 잃고 크게 먹기"(2026-11-08).
// 손절·크기·추적 청산은 기계가, 진입 여부는 사람이 정한다. 계산만 하고 주문은 없다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v, d = 1) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: d }));
  const CHIP = { 상방: "good", 하방: "warn", 중립: "" };
  const T6D_CHIP = { 1: "good", "-1": "warn", 0: "" };
  const FRAME_NAME = { "1d": "일봉", "4h": "4시간", "15m": "15분" };

  let plan = null;
  let side = "long";
  let risk = 1;
  let wallet = null;
  let available = null;
  let entryTouched = false;
  let stopTouched = false;
  let equityTouched = false;

  const todayHtml = (p, gate) => {
    const frames = ["1d", "4h", "15m"].map((k) =>
      `${FRAME_NAME[k]} <span class="chip ${CHIP[p.big_frames[k]] || ""}">${esc(p.big_frames[k] || "-")}</span>`).join(" · ");
    const big = p.big_move
      ? `<b>⚡ 큰 움직임 가능성 높음 (방향은 반반)</b>`
      : `큰 움직임 표시 없음`;
    const since = p.t6d_since ? ` <span class="mini">${esc(p.t6d_since)} 일봉 신호부터</span>` : "";
    const chase = gate ? `<div class="detail">추격 한도 ±${num(gate.pct, 2)}% (롱 ${num(gate.long_max)}까지, 숏 ${num(gate.short_min)}까지) — 아래 분석 카드와 같은 기준</div>` : "";
    return `<div><b>현재가 ${num(p.price)}</b> <span class="mini">USDT · 바이낸스 선물</span></div>
      <div style="margin-top:8px">${big}</div>
      <div class="detail">${frames}. 셋 다 하방이면 다음 24시간 범위가 평소의 1.03~1.20배(2020~2026 시험). 방향은 맞히지 못해요.</div>
      <div style="margin-top:8px">일봉 방향 <span class="chip ${T6D_CHIP[p.t6d]}">${esc(p.t6d_label)}</span>${since}</div>
      <div class="detail">추세 이탈 규칙(T6D)의 마지막 신호 방향. 마감된 일봉(09:00 KST)만 봐요. 무작위보다 나았지만 단독으로 실전 기준은 통과 못 했어요.</div>
      <div style="margin-top:8px">손절 거리 제안: 일봉 ATR(14) 1배 = ${num(p.atr_d)} (${num(p.atr_d_pct, 2)}%)</div>
      ${chase}`;
  };

  const pick = (group, value) => $(group).querySelectorAll("button").forEach((b) => {
    b.classList.toggle("on", b.dataset.v === String(value));
  });

  const stopButtons = () => {
    if (!plan) return;
    const s = plan.stops[side];
    const opts = [["atr", `일봉 ATR 1배 ${num(s.atr)}`], ["swing", `4시간 ${side === "long" ? "스윙 저점" : "스윙 고점"} ${num(s.swing)}`]];
    $("sz-stops").innerHTML = `<span class="mini">손절 후보</span>` + opts.filter(([k]) => s[k] != null)
      .map(([k, label]) => `<button type="button" data-v="${k}">${esc(label)}</button>`).join("");
  };

  // ATR 손절은 진입가 기준으로 다시 잰다(계산기 진입가가 현재가와 다를 수 있음)
  const stopFor = (kind) => {
    const s = plan.stops[side];
    if (kind === "swing") return s.swing;
    const entry = Number($("sz-entry").value) || plan.price;
    return Math.round((side === "long" ? entry - plan.atr_d : entry + plan.atr_d) * 10) / 10;
  };

  const exitPlan = (r) => {
    const back = side === "long" ? "아래" : "위";
    const prev = side === "long" ? plan.prev_4h_low : plan.prev_4h_high;
    return `<ul class="checklist">
      <li>☐ +1R ${num(r.r_levels["1R"])} 닿으면 → 손절을 본전+수수료 <b>${num(r.breakeven_stop)}</b>로 올리기</li>
      <li>☐ +2R ${num(r.r_levels["2R"])} 닿으면 → 1/3 익절 (${num(r.qty / 3, 3)} BTC 정도)</li>
      <li>☐ 나머지는 직전 마감 4시간봉 ${side === "long" ? "저점" : "고점"} ${back}에 손절을 따라 붙이기
        (지금 ${num(prev)}) 또는 4시간 ATR 2배(${num(plan.atr_4h * 2)}) 거리</li>
      <li>☐ +3R ${num(r.r_levels["3R"])}은 참고선. 추적 손절에 걸릴 때까지 들고 가기</li>
      <li>☐ <b>손절은 절대 넓히지 않기</b>. 줄이는 방향으로만 옮기기</li>
    </ul>`;
  };

  const outHtml = (r) => {
    const warn = r.warnings.map((w) => `<div class="chip warn" style="display:inline-block;margin:4px 0">⚠️ ${esc(w)}</div>`).join("<br>");
    return `${warn}
      <table class="acct"><tbody>
        <tr><td>수량</td><td><b>${num(r.qty, 3)} BTC</b>${r.capped ? ` <span class="mini">(원래 ${num(r.qty_wanted, 3)})</span>` : ""}</td></tr>
        <tr><td>주문 금액</td><td>${num(r.notional, 2)} USDT</td></tr>
        <tr><td>실효 레버리지</td><td>${num(r.leverage, 2)}배 <span class="mini">(계좌 5배 교차 한도)</span></td></tr>
        <tr><td>손절 거리</td><td>${num(r.r)} (${num(r.stop_pct, 2)}%) = 1R</td></tr>
        <tr><td>손절 시 손실</td><td class="up"><b>−${num(r.loss_at_stop, 2)} USDT</b> (계좌의 ${num(r.loss_pct, 2)}%, 왕복 수수료 ${num(r.fees_round_trip, 2)} 포함)</td></tr>
        <tr><td>+1R / +2R / +3R</td><td>${num(r.r_levels["1R"])} / ${num(r.r_levels["2R"])} / ${num(r.r_levels["3R"])}</td></tr>
      </tbody></table>
      <div style="margin-top:8px"><b>크게 먹는 청산 계획</b></div>${exitPlan(r)}
      <p class="mini">수량은 0.001 BTC 단위로 내림, 수수료는 테이커 0.05%/편도로 계산. 진입 여부는 직접 정하세요. 투자 판단과 책임은 본인에게 있어요.</p>`;
  };

  let timer = null;
  const calc = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const entry = $("sz-entry").value;
      const stop = $("sz-stop").value;
      const equity = $("sz-equity").value;
      if (!entry || !stop || !equity) { $("sz-out").textContent = "진입가·손절가·계좌 금액을 넣어 주세요"; return; }
      const q = new URLSearchParams({ side, entry, stop, equity, risk_pct: risk });
      if (available != null && !equityTouched) q.set("available", available);
      fetch(`/api/btc/size?${q}`).then((r) => r.json().then((d) => ({ ok: r.ok, d }))).then(({ ok, d }) => {
        $("sz-out").innerHTML = ok ? outHtml(d) : `<span class="up">${esc(d.detail)}</span>`;
      }).catch(() => { $("sz-out").textContent = "계산하지 못했어요"; });
    }, 250);
  };

  const fill = () => {
    if (plan && !entryTouched) $("sz-entry").value = plan.price;
    if (plan && !stopTouched) $("sz-stop").value = stopFor("atr");
    if (wallet != null && !equityTouched) $("sz-equity").value = Math.round(wallet * 100) / 100;
    stopButtons();
    if (plan && !stopTouched) pick("sz-stops", "atr");
    calc();
  };

  $("sz-side").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    side = b.dataset.v; pick("sz-side", side); stopTouched = false; fill();
  });
  $("sz-risk").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    risk = Number(b.dataset.v); pick("sz-risk", b.dataset.v); calc();
  });
  $("sz-stops").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || !plan) return;
    $("sz-stop").value = stopFor(b.dataset.v); stopTouched = true; pick("sz-stops", b.dataset.v); calc();
  });
  $("sz-entry").addEventListener("input", () => { entryTouched = true; calc(); });
  $("sz-stop").addEventListener("input", () => { stopTouched = true; pick("sz-stops", ""); calc(); });
  $("sz-equity").addEventListener("input", () => { equityTouched = true; calc(); });

  const load = () => {
    fetch("/api/btc").then((r) => r.json()).then((d) => {
      if (!d.plan) { $("btcplan-today").textContent = d.message || "오늘의 판 계산 대기 중 (15분마다 갱신)"; return; }
      plan = d.plan;
      $("btcplan-today").innerHTML = todayHtml(plan, d.entry_gate);
      fill();
    }).catch(() => { $("btcplan-today").textContent = "오늘의 판을 불러오지 못했어요"; });
    let token = "";
    try { token = localStorage.getItem("minerva-account-token") || ""; } catch (_) { /* 저장소 막힘 */ }
    fetch("/api/account/binance", { headers: { "X-Minerva-Token": token } }).then((r) => r.json()).then((d) => {
      if (d.wallet == null) return;
      wallet = d.wallet; available = d.available;
      fill();
    }).catch(() => {});
  };

  load();
  setInterval(load, 60000);
})();
