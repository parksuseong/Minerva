// "🌱 성장 후보" 탭. /api/growth (minerva.growth가 쓴 ~/.minerva/growth/latest.json)를 그린다.
// 관심 목록일 뿐 매수 추천이 아니다. 주문과 아무 연결이 없다.
(() => {
  const $ = (id) => document.getElementById(id);
  const UI = window.MinervaUI;
  const { esc } = UI;
  const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "");
  const pct = (v) => (v == null ? "-" : `<span class="${cls(v)}">${v > 0 ? "+" : ""}${UI.num(v * 100, 0)}%</span>`);
  const cap = (v) => (v == null ? "-" : v >= 10e9 ? `$${UI.num(v / 1e9, 0)}B` : `$${UI.num(v / 1e9, 1)}B`);
  const price = (v) => (v == null ? "-" : `$${UI.num(v, v >= 100 ? 0 : 2)}`);

  function table(b) {
    if (!b.rows || !b.rows.length) return `<p class="placeholder">조건에 맞는 종목이 없어요</p>`;
    const rows = b.rows.map((r, i) => `<tr>
      <td>${i + 1}</td>
      <td><b>${esc(r.symbol)}</b><br><span class="mini">${esc(r.name)}</span></td>
      <td class="mini">${esc(r.sector || "-")}</td>
      <td>${cap(r.cap)}</td>
      <td>${pct(r.growth)}<br><span class="mini">분기 ${pct(r.q_growth)}</span></td>
      <td>${r.profit === "흑자" ? `<span class="up">흑자</span>` : esc(r.profit || "-")}</td>
      <td>${pct(r.drawdown)}</td>
      <td>${price(r.price)}</td></tr>`).join("");
    return `<div class="scroll"><table class="acct"><thead><tr><th>#</th><th>종목</th><th>업종</th><th>시총</th>
      <th>매출 성장<br><span class="mini">연간</span></th><th>이익</th><th>52주 고점 대비</th><th>가격</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  }

  function draw(d) {
    if (d.status !== "ok") {
      $("growth-status").textContent = d.message || "아직 목록이 없어요";
      return;
    }
    const stale = d.stale
      ? `<p class="chip warn">⚠️ 최신이 아니에요: ${esc(d.stale_reason || "갱신 실패")} (${esc(d.stale_at || d.generated_at)})</p>`
      : "";
    const warns = (d.warnings || []).map((w) => `<li>${esc(w)}</li>`).join("");
    $("growth-status").innerHTML = `${stale}<p class="mini">${esc(d.generated_at)} 갱신 · ${esc(d.method || "")}</p>
      ${warns ? `<ul class="mini">${warns}</ul>` : ""}`;
    $("growth-buckets").innerHTML = (d.buckets || []).map((b) => `<section class="card">
      <h2>${b.key === "x10" ? "🌳" : "🌱"} ${esc(b.label)} <span class="mini">시총 ${cap(b.lo)}~${cap(b.hi)} · 대상 ${UI.num(b.universe, 0)}종목 중 매출 성장 상위 ${(b.rows || []).length}</span></h2>
      ${table(b)}</section>`).join("");
  }

  function load() {
    fetch("/api/growth")
      .then((r) => r.json())
      .then(draw)
      .catch(() => { $("growth-status").textContent = "성장 후보를 불러오지 못했어요"; });
  }

  if ($("tab-growth")) {
    load();
    setInterval(load, 30 * 60 * 1000);
  }
})();
