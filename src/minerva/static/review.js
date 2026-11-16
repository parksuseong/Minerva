// "📝 복기" 탭. /api/btc/reviews (minerva.btc_review가 매일 04:05에 쓴 하루 복기,
// minerva.btc_trade_review가 다음 날 아침 09:20에 쓴 거래별 복기)를 그린다.
// 계좌 손익이 들어 있어 계좌 탭과 같은 열람 토큰(이 기기에 기억된 값)을 보낸다.
(() => {
  const $ = (id) => document.getElementById(id);
  const UI = window.MinervaUI;
  const { esc } = UI;
  const num = (v, d = 2) => UI.num(v, d);
  const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "");
  const sgn = (v, d = 2) => (v == null ? "-" : `<span class="${cls(v)}">${v > 0 ? "+" : ""}${num(v, d)}</span>`);
  const pct = (v, d = 2) => (v == null ? "-" : `${sgn(v, d)}%`);
  const SIDE = { long: "롱", short: "숏" };
  const token = () => {
    try { return localStorage.getItem("minerva-account-token") || ""; } catch (e) { return ""; }
  };
  const list = (title, items) => (items && items.length
    ? `<h3>${esc(title)}</h3><ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "");

  function trades(rows) {
    if (!rows.length) return "<p class=\"mini\">이날 닫힌 거래 없음</p>";
    const body = rows.map((t) => {
      const a = t.after;
      return `<tr><td>${esc(String(t.symbol || "").replace(/USDT$/, ""))} ${esc(SIDE[t.side] || t.side)}</td>
        <td>${esc(String(t.opened_at).slice(5, 16))}<br>${esc(String(t.closed_at).slice(5, 16))}</td>
        <td>${num(t.entry, 4)}<br>${num(t.exit, 4)}</td>
        <td>${esc(t.why_ko || t.why)}${t.result ? `<br><span class="mini">${esc(t.result)}</span>` : ""}</td>
        <td>${sgn(t.pnl)}<br>${pct(t.pct)}</td>
        <td>${pct(t.mfe_pct)}<br>${pct(t.mae_pct)}</td>
        <td>${a ? `고 ${pct(a.high_pct)}<br>저 ${pct(a.low_pct)}<br>끝 ${pct(a.close_pct)}` : "-"}</td></tr>`;
    }).join("");
    return `<div class="scroll"><table class="acct"><thead><tr><th>종목</th><th>진입·청산</th>
      <th>진입가·청산가</th><th>결과</th><th>손익(USDT·%)</th><th>보유 중 최고·최저</th>
      <th>청산 뒤 4시간</th></tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function card(r, i) {
    const acc = r.account;
    const t = r.totals || {};
    const rv = r.review;
    const passedTop = Object.entries(r.passed || {}).slice(0, 6).map(([s, n]) => `${s.replace(/USDT$/, "")} ${n}`).join(", ");
    const choices = (r.choices || []).map((c) =>
      `${String(c.at).slice(11, 16)} ${(c.options || {})[c.choice] || c.choice}${c.reason ? ` (${c.reason})` : ""}`);
    return `<section class="card"${UI.pgAttr("rv", i)}>
      <h2>${esc(r.day)} ${r.partial ? "<span class=\"mini\">(진행 중에 쓴 복기)</span>" : ""}</h2>
      ${rv ? `<p><b>${esc(rv.headline)}</b></p>` : `<p class="mini">AI 복기 없음: ${esc(r.review_error || "-")}</p>`}
      <p>계좌 하루 손익 ${acc ? `${sgn(acc.pnl)} USDT (${pct(acc.pct)}) · 수수료 ${num(acc.commission)} · 펀딩 ${num(acc.funding)}` : "-"}${acc && acc.basis === "realized" ? " <span class=\"mini\">(확정 손익 기준)</span>" : ""}<br>
      닫힌 거래 ${t.n || 0}번 · ${t.losses == null ? `이긴 거래 ${t.wins || 0}` : `${t.wins || 0}승 ${t.losses}패 ${t.draws || 0}무 (승률 ${t.winrate == null ? "-" : `${num(t.winrate * 100, 0)}%`})`} · 합계 ${sgn(t.pnl)} USDT</p>
      ${trades(r.trades || [])}
      ${rv ? `${list("👍 잘한 점", rv.good)}${list("👎 아쉬운 점", rv.bad)}
        ${rv.rule_check ? `<h3>📏 규칙대로 했나</h3><p>${esc(rv.rule_check)}</p>` : ""}
        ${list("👀 다음에 볼 것", rv.tomorrow)}${list("💡 생각해 볼 것 (검증 안 됨, 규칙 아님)", rv.ideas)}` : ""}
      <p class="mini">AI 종목 선택: ${choices.length ? esc(choices.join(" / ")) : "없음"}<br>
      못 들어간 신호(장 쉬는 시간 등): ${r.skipped_total ?? (r.skipped || []).length}개 · 후보 아닌 종목 신호: ${r.passed_total || 0}개${passedTop ? ` (${esc(passedTop)})` : ""}<br>
      ${esc(r.generated_at || "")}에 씀 · 보유 중 최고·최저는 진입가 대비 내 방향 기준</p>
    </section>`;
  }

  const RESULT = { stop: "손절", trail: "따라간 손절", tp: "익절", open: "결론 안 남" };
  const cfLine = (title, c) => (c
    ? `<li>${esc(title)} (${esc(c.rule || "")}): ${esc(RESULT[c.result] || c.result)}${c.at ? ` · ${esc(String(c.at).slice(5, 16))}` : ""}${c.price != null ? ` @ ${num(c.price, 4)}` : ""} ${pct(c.pct)}${c.trail_on_at ? ` · 따라가기 켜짐 ${esc(String(c.trail_on_at).slice(5, 16))}` : ""}${c.note ? ` <span class="mini">(${esc(c.note)})</span>` : ""}</li>` : "");
  const textBlock = (title, v) => (v ? `<h3>${esc(title)}</h3><p>${esc(v)}</p>` : "");
  const items = (title, v) => (Array.isArray(v) ? list(title, v) : textBlock(title, v));

  function tradeCard(t) {
    const rv = t.review;
    const a = t.after;
    const cf = t.counterfactual || {};
    const touched = (t.interventions || []).length > 0;
    const badge = t.manual ? " <span class=\"mini\">[수동 진입]</span>" : (touched ? " <span class=\"mini\">[손댐]</span>" : "");
    const head = `${esc(String(t.symbol || "").replace(/USDT$/, ""))} ${esc(SIDE[t.side] || t.side)} ${pct(t.pct)} · ${esc(t.why_ko || t.why || "-")}${t.result ? ` · ${esc(t.result)}` : ""}${badge}`;
    return `<details class="card">
      <summary><b>${head}</b> <span class="mini">${esc(String(t.closed_at || "").slice(5, 16))} 청산 · ${esc(t.hold_min ?? "-")}분 보유 · ${sgn(t.pnl)} USDT</span>
        ${rv ? `<br>${esc(rv.headline)}` : ""}</summary>
      <p>진입 ${esc(String(t.opened_at || "").slice(5, 16))} @ ${num(t.entry, 4)} → 청산 ${esc(String(t.closed_at || "").slice(5, 16))} @ ${num(t.exit, 4)}<br>
      어떻게 들어갔나: ${esc(t.entry_way || "-")}<br>
      보유 중 최고 ${pct(t.mfe_pct)} · 최저 ${pct(t.mae_pct)}<br>
      청산 뒤${a ? `${a.partial ? ` ${esc(a.bars)}봉(4시간 안 됨)` : " 4시간"}: 고 ${pct(a.high_pct)} · 저 ${pct(a.low_pct)} · 끝 ${pct(a.close_pct)}` : ": -"}</p>
      ${touched ? list("✋ 사용자가 손으로 한 것", t.interventions) : ""}
      ${cf.rule_now || cf.fixed_tp ? `<h3>🔁 같은 진입이었다면 (계산, 봉 안 순서는 손절 먼저)</h3><ul>${cfLine("지금 규칙", cf.rule_now)}${cfLine("고정 익절", cf.fixed_tp)}</ul>` : ""}
      ${t.bars_error ? `<p class="mini">시세를 못 받아 가정 결과 없음: ${esc(t.bars_error)}</p>` : ""}
      ${rv ? `${textBlock("무슨 일이 있었나", rv.what)}${items("👍 잘한 점", rv.good)}${items("👎 아쉬운 점", rv.bad)}
        ${textBlock("✋ 손으로 한 결정 평가", rv.manual_note)}${textBlock("💡 교훈 (검증 안 됨, 규칙 아님)", rv.lesson)}
        ${textBlock("📏 규칙대로였나", rv.rule_check)}` : `<p class="mini">AI 복기 없음: ${esc(t.review_error || "-")}</p>`}
      <p class="mini">${esc(t.generated_at || "")}에 씀 · %는 진입가 대비 가격 %(배율 곱하기 전${t.leverage ? `, 이 거래 ${esc(t.leverage)}배` : ""})</p>
    </details>`;
  }

  function render(d) {
    const rows = d.reviews || [];
    $("rv-status").textContent = rows.length ? "" : "아직 복기가 없어요 (매일 04:05에 전날 것을 써요)";
    $("rv-list").innerHTML = rows.map(card).join("") + UI.moreBtn("rv", rows.length);
    const tr = d.trades || [];
    if ($("rv-trades")) {
      $("rv-trades-status").textContent = tr.length ? "" : "아직 거래별 복기가 없어요 (다음 날 아침 09:20에 써요)";
      $("rv-trades").innerHTML = tr.map(tradeCard).join("");
    }
  }

  function load() {
    fetch("/api/btc/reviews", { headers: { "X-Minerva-Token": token() } })
      .then((r) => (r.status === 403 ? null : r.json()))
      .then((d) => {
        if (d) render(d);
        else {
          $("rv-status").textContent = "계좌 탭에서 열람 토큰을 먼저 넣어 주세요";
          if ($("rv-trades-status")) $("rv-trades-status").textContent = "";
        }
      })
      .catch(() => {
        $("rv-status").textContent = "복기를 불러오지 못했어요";
        if ($("rv-trades-status")) $("rv-trades-status").textContent = "";
      });
  }

  if ($("rv-list")) load();
})();
