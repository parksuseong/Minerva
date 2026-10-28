// "오늘 할 일"·"종목 분석" 탭. /api/daily 결과(배치가 매일 09:10·16:30 갱신)를 그린다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (v) => (v == null ? "-" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`);
  const num = (v) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: 2 }));
  const SIDE = { long: "롱", short: "숏" };
  const ICON = { "진입 신호": "🟢", "청산 신호": "🔷", "보유 중": "🔵", "대기": "⏳" };

  const label = (a, side) => (a.verify?.[side]?.label) || "미검증";
  const badge = (a, side) => {
    const l = label(a, side);
    const color = l === "검증됨" ? "#16a34a" : l === "부적합" ? "#b91c1c" : "";
    const v = a.verify?.[side];
    const tip = v ? `거래 ${v.trades}회 · 승률 ${pct(v.win_rate)} · 우위 안 ${pct(v.edge_in)} / 밖 ${pct(v.edge_out)}` : "";
    return `<span class="badge" title="${esc(tip)}"${color ? ` style="color:${color}"` : ""}>${SIDE[side]} ${esc(l)}</span>`;
  };
  // 종목 자체 숏은 실행할 수 없어 숨긴다. 숏은 국내 인버스 ETF로만 (daily.py ASSET_SIDES)
  const sides = (a) => (a.kind ? [["short", a.tm]] : [["long", a.tm]]);

  function sideDetail(side, st) {
    if (!st) return "";
    const name = SIDE[side];
    if (st.state === "진입 신호") {
      const stop = side === "long" ? `손절 ${num(st.stop)} (−10%)` : esc(st.stop);
      return `<div class="state go">🟢 ${name} 진입 신호 · ${esc(st.entry)}</div>
        <div class="detail">${stop} · ${esc(st.exit_rule)}</div>`;
    }
    if (st.state === "청산 신호") {
      return `<div class="state hold">🔷 ${name} 청산 신호 · ${esc(st.reason)} ${pct(st.pnl)}</div>
        <div class="detail">${esc(st.exit)} · 진입 ${esc(st.entry_day)}</div>`;
    }
    if (st.state === "보유 중") {
      const stop = side === "long" ? `손절 ${num(st.stop)}` : esc(st.stop);
      return `<div class="state hold">🔵 ${name} 보유 중 ${st.pnl == null ? "" : pct(st.pnl)} · ${st.days_left}거래일 남음</div>
        <div class="detail">진입 ${esc(st.entry_day)} · ${stop} · ${esc(st.exit_rule)}</div>`;
    }
    const waits = (st.wait || []).map((w) => `<li>${esc(w)}</li>`).join("");
    return `<div class="state">⏳ ${name} 대기 · 빠진 조건 ${(st.missing || []).length}개</div>` +
      (waits ? `<ul class="detail" style="margin:2px 0 0 16px;padding:0">${waits}</ul>`
        : `<div class="detail">${esc((st.missing || []).join(", "))}</div>`);
  }

  function assetHtml(a) {
    if (a.error) return `<div class="asset"><b>${esc(a.name)}</b> <span class="detail">데이터 오류</span></div>`;
    const ch = a.change == null ? "" : `<span class="${a.change >= 0 ? "up" : "down"}">${pct(a.change)}</span>`;
    const brief = sides(a).map(([side, st]) => `${SIDE[side]} ${ICON[st?.state] || ""}${esc(st?.state)}`).join(" · ");
    const info = a.kind
      ? `인버스 ${Math.abs(a.mult)}배 · 기초자산 ${esc(a.underlying)} · ${esc(a.tm?.note)}`
      : `${esc(a.date)} 종가 · Stoch ${a.stoch} · CCI ${a.cci} · RSI ${a.rsi} · 50일선 ${num(a.ma50)}`;
    return `<details class="asset">
      <summary><div class="asset-head"><div><b>${esc(a.name)}</b>${sides(a).map(([s]) => badge(a, s)).join("")}</div>
        <div class="px">${num(a.close)} ${ch}</div></div>
        <div class="mini">${brief}</div></summary>
      ${sides(a).map(([side, st]) => sideDetail(side, st)).join("")}
      <div class="detail">${info}</div>
    </details>`;
  }

  // minerva.trade와 같은 기준: 국내·검증됨·롱/인버스만 실제 주문한다
  const AUTO = ["LONG_ENTRY", "LONG_EXIT", "BUY", "SELL"];
  function autoTag(x) {
    if (x.market !== "KR") return "📎 참고 (해외·코인은 주문 안 함)";
    if (!x.verified) return "📎 참고 (미검증)";
    return "🤖 자동주문 대상";
  }

  function actionHtml(x) {
    const tag = x.verified ? "✅ 검증됨" : `⚠️ ${esc(x.verify)}`;
    const body = x.stop != null
      ? `${esc(x.when)} · 손절 ${num(x.stop)}${x.action === "BUY" ? " (기초자산)" : ""} · 익절 ${esc(x.take)} · 최대 ${x.max_days}거래일`
      : `${esc(x.reason)} ${pct(x.pnl)} · ${esc(x.when)}`;
    return `<div class="act"><span class="tag ${esc(x.action)}">${esc(x.label)}</span>
      <div><b>${esc(x.name)}</b> <span class="mini">${esc(x.symbol)} · ${tag}</span></div>
      <span></span><div class="detail">${body}</div>
      <span></span><div class="mini">${autoTag(x)}</div></div>`;
  }

  const rows = (d) => d.assets.concat(d.products || []).filter((a) => !a.error);

  function renderToday(d) {
    const acts = d.actions || [];
    $("today-actions").innerHTML = acts.length ? acts.map(actionHtml).join("")
      : '<p class="placeholder">오늘은 새로 낼 주문이 없어요 (진입·청산 신호 없음).</p>';
    const holds = [], near = [];
    rows(d).forEach((a) => sides(a).forEach(([side, st]) => {
      if (st?.state === "보유 중") holds.push([a, side, st]);
      if (st?.state === "대기" && label(a, side) === "검증됨" && (st.missing || []).length === 1) near.push([a, side, st]);
    }));
    $("today-holds").innerHTML = holds.length ? holds.map(([a, side, st]) =>
      `<div class="act"><span class="tag">${SIDE[side]}</span><div><b>${esc(a.name)}</b>${badge(a, side)}</div>
        <span></span><div class="detail">${st.pnl == null ? "" : pct(st.pnl) + " · "}${st.days_left}거래일 남음 · 진입 ${esc(st.entry_day)} · ${side === "long" ? "손절 " + num(st.stop) : esc(st.stop)}</div></div>`).join("")
      : "규칙상 보유 중인 포지션 없음";
    $("today-near").innerHTML = near.length ? near.map(([a, side, st]) =>
      `<div class="act"><span class="tag">${SIDE[side]}</span><div><b>${esc(a.name)}</b></div>
        <span></span><div class="detail">${esc((st.wait || st.missing).join(" / "))}</div></div>`).join("")
      : "없음";
  }

  let data = null, market = "all";
  function renderList() {
    const onlyOk = $("an-verified").checked;
    const list = rows(data).filter((a) => {
      if (market === "inv") { if (!a.kind) return false; }
      else if (a.kind || (market !== "all" && a.market !== market)) return false;
      return !onlyOk || sides(a).some(([side]) => label(a, side) === "검증됨");
    });
    $("an-count").textContent = `${list.length}종목`;
    $("daily-list").innerHTML = list.map(assetHtml).join("");
  }

  $("an-filters").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || !data) return;
    market = b.dataset.m;
    $("an-filters").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    renderList();
  });
  $("an-verified").addEventListener("change", () => data && renderList());

  fetch("/api/daily").then((r) => r.json()).then((d) => {
    if (d.status === "pending") { $("daily-meta").textContent = d.message; return; }
    data = d;
    const v = d.vix, f = d.fear;
    const regCls = v.regime === "R0" || v.regime === "R1" ? "good" : "warn";
    $("daily-meta").innerHTML = `<div class="regime">
        <span class="chip ${regCls}">VIX ${v.close} · ${esc(v.regime)}</span>
        <span class="chip">신규 비중 ${esc(v.size_limit)}</span>
        <span class="chip ${f.state === "없음" ? "" : "good"}">공포 초입: ${esc(f.state)} (VIX/VIX3M ${pct(f.premium)})</span>
      </div><div class="detail">분석 ${esc(d.generated_at.replace("T", " "))} KST · 매일 09:10·16:30 갱신 · 숏은 롱 계획의 1/2 이하</div>` +
      (d.summary?.length ? `<ul style="margin:8px 0 0 16px;padding:0">${
        d.summary.map((s) => `<li class="detail" style="color:inherit">${esc(s)}</li>`).join("")}</ul>` : "");
    renderToday(d);
    renderList();
    $("daily-note").textContent = d.note;
  }).catch(() => { $("daily-meta").textContent = "분석 데이터를 불러오지 못했어요"; });
})();
