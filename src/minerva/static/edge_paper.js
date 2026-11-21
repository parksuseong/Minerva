// "🔬 가상 추적" 탭. /api/edge-paper (minerva.edge_paper가 쓴 ~/.minerva/edge_paper/summary.json)를 그린다.
// 연구 규칙의 실시간 가상 추적일 뿐이다. 주문·계좌와 아무 연결이 없다.
(() => {
  const $ = (id) => document.getElementById(id);
  const UI = window.MinervaUI;
  const { esc } = UI;
  const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "");
  const pct = (v, d = 2) => (v == null ? "-" : `<span class="${cls(v)}">${v > 0 ? "+" : ""}${UI.num(v, d)}%</span>`);
  const when = (s) => {
    if (!s) return "-";
    const d = new Date(new Date(s).getTime() + 9 * 3600 * 1000);
    const p2 = (n) => String(n).padStart(2, "0");
    return `${p2(d.getUTCMonth() + 1)}-${p2(d.getUTCDate())} ${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}`;
  };
  const px = (v) => (v == null ? "-" : UI.num(v, v >= 100 ? 2 : v >= 1 ? 4 : 6));

  function openTable(rows) {
    if (!rows.length) return `<p class="mini">열린 가상 포지션 없음</p>`;
    return `<div class="scroll"><table class="acct"><thead><tr><th>종목</th><th>방향</th><th>진입</th>
      <th>진입가</th><th>지금</th><th>움직임</th><th>손절가</th><th>청산 예정</th></tr></thead><tbody>${rows.map((r) => `<tr>
      <td><b>${esc(r.sym)}</b></td><td>${esc(r.dir)}</td><td class="mini">${when(r.at)}</td><td>${px(r.entry)}</td>
      <td>${px(r.last)}</td><td>${pct(r.move_pct)}</td><td>${px(r.stop)}</td><td class="mini">${when(r.exit_due)}</td></tr>`).join("")}
      </tbody></table></div>`;
  }

  function tradeTable(rows) {
    if (!rows.length) return `<p class="mini">끝난 가상 거래 없음</p>`;
    return `<div class="scroll"><table class="acct"><thead><tr><th>종목</th><th>방향</th><th>진입</th><th>청산</th>
      <th>진입가</th><th>청산가</th><th>확신도</th><th>손익(비용·펀딩 후)</th></tr></thead><tbody>${rows.map((r) => `<tr>
      <td><b>${esc(r.sym)}</b></td><td>${esc(r.dir)}</td><td class="mini">${when(r.at)}</td>
      <td class="mini">${when(r.exit_at)}${r.stopped ? " · 손절" : ""}</td><td>${px(r.entry)}</td><td>${px(r.exit)}</td>
      <td>${UI.num(r.conf, 3)}</td><td>${pct(r.net_pct)}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function card(r) {
    const stats = [
      ["거래", `${r.trades}건 (끝남 ${r.closed} · 열림 ${r.open})`],
      ["승률", r.winrate == null ? "-" : `${UI.num(r.winrate, 1)}% (${r.wins}/${r.closed})`],
      ["거래당 평균", pct(r.avg_net_pct, 3)],
      ["누적 수익률(장부)", pct(r.return_pct)],
      ["최대 낙폭", r.maxdd_pct == null ? "-" : `${UI.num(r.maxdd_pct, 2)}%`],
      ["손절", `${r.stops}건`],
    ].map(([k, v]) => `<div><span class="mini">${k}</span><br><b>${v}</b></div>`).join("");
    return `<section class="card">
      <h2>${esc(r.name)} <span class="mini">${esc(r.desc)}</span></h2>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:6px 0">${stats}</div>
      ${r.liquidated ? `<p class="chip warn">장부 청산됨</p>` : ""}
      <h3>열린 가상 포지션</h3>${openTable(r.open_positions || [])}
      <h3>최근 거래 20</h3>${tradeTable(r.last_trades || [])}</section>`;
  }

  function draw(d) {
    if (d.status !== "ok") {
      $("paper-status").textContent = d.message || "아직 기록이 없어요";
      return;
    }
    const stale = d.stale ? `<p class="chip warn">⚠️ 최신이 아니에요 (${when(d.generated_at)})</p>` : "";
    $("paper-status").innerHTML = `${stale}<p class="mini">시작 ${when(d.start)} · 자료 ${when(d.data_until)}까지 · 마지막 결정 ${when(d.last_decision)} · ${when(d.generated_at)} 갱신</p>
      <p class="mini">손익은 수수료·미끄러짐 왕복 0.14%와 실제 펀딩을 뺀 값. 누적 수익률은 위험 0.5%(평가액 0.1배, 바구니는 칸마다 1/5) 장부, 열린 포지션은 마지막 종가로 평가. 과거 연구에서 V01은 한쪽 검증만 통과(다른 쪽 실패), 이 기록이 첫 깨끗한 시험입니다.</p>`;
    $("paper-rules").innerHTML = (d.rules || []).map(card).join("");
  }

  function load() {
    fetch("/api/edge-paper")
      .then((r) => r.json())
      .then(draw)
      .catch(() => { $("paper-status").textContent = "가상 추적을 불러오지 못했어요"; });
  }

  if ($("tab-paper")) {
    load();
    setInterval(load, 30 * 60 * 1000);
  }
})();
