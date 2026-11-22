// "🔬 가상 추적" 탭 아래 칸. /api/barrier-paper (minerva.barrier_paper가 쓴 summary.json)를 그린다.
// +5% 먼저 vs −3% 먼저 장벽 게임 가상 추적. 주문·계좌와 연결 없음.
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
  const RES = { win: "익절 +5%", loss: "손절 −3%", timeout: "14일 청산" };

  function openTable(rows) {
    if (!rows.length) return `<p class="mini">열린 가상 포지션 없음</p>`;
    return `<div class="scroll"><table class="acct"><thead><tr><th>종목</th><th>방향</th><th>진입</th>
      <th>진입가</th><th>지금</th><th>움직임</th><th>익절가</th><th>손절가</th></tr></thead><tbody>${rows.map((r) => `<tr>
      <td><b>${esc(r.sym)}</b></td><td>${esc(r.dir)}</td><td class="mini">${when(r.at)}</td><td>${px(r.entry)}</td>
      <td>${px(r.last)}</td><td>${pct(r.move_pct)}</td><td>${px(r.tp)}</td><td>${px(r.stop)}</td></tr>`).join("")}
      </tbody></table></div>`;
  }

  function tradeTable(rows) {
    if (!rows.length) return `<p class="mini">끝난 가상 거래 없음</p>`;
    return `<div class="scroll"><table class="acct"><thead><tr><th>종목</th><th>방향</th><th>진입</th><th>청산</th>
      <th>진입가</th><th>청산가</th><th>결과</th><th>손익(비용·펀딩 후)</th></tr></thead><tbody>${rows.map((r) => `<tr>
      <td><b>${esc(r.sym)}</b></td><td>${esc(r.dir)}</td><td class="mini">${when(r.at)}</td>
      <td class="mini">${when(r.exit_at)}</td><td>${px(r.entry)}</td><td>${px(r.exit)}</td>
      <td>${esc(RES[r.result] || r.result)}</td><td>${pct(r.net_pct)}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function card(r, be) {
    const wr = r.winrate == null ? "-" : `${UI.num(r.winrate, 1)}% (${r.wins}/${r.closed})`;
    const stats = [
      ["거래", `${r.trades}건 (끝남 ${r.closed} · 열림 ${r.open})`],
      ["승률", r.winrate == null ? wr : `<span class="${r.winrate >= be ? "up" : "down"}">${wr}</span>`],
      ["거래당 평균", pct(r.avg_net_pct, 3)],
      ["합계(1배·건당 같은 크기)", pct(r.sum_net_pct)],
    ].map(([k, v]) => `<div><span class="mini">${k}</span><br><b>${v}</b></div>`).join("");
    return `<section class="card">
      <h2>${esc(r.name)} <span class="mini">${esc(r.desc)}</span></h2>
      <p class="mini">${esc(r.past)}</p>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:6px 0">${stats}</div>
      <h3>열린 가상 포지션</h3>${openTable(r.open_positions || [])}
      <h3>최근 거래 20</h3>${tradeTable(r.last_trades || [])}</section>`;
  }

  function watchCard(rows) {
    const body = rows.length
      ? `<div class="scroll"><table class="acct"><thead><tr><th>종목</th><th>감시 시작</th><th>고점</th><th>지금(종가)</th>
        <th>숏 들어갈 가격(종가 ≤)</th><th>감시 끝</th></tr></thead><tbody>${rows.map((w) => `<tr>
        <td><b>${esc(w.sym)}</b></td><td>${esc(w.since)}</td><td>${px(w.peak)}</td><td>${px(w.last)}</td>
        <td>${px(w.trigger)}</td><td>${esc(w.until)}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="mini">지금 감시 중인 급등주 없음</p>`;
    return `<section class="card"><h2>👀 급등주 감시 목록 <span class="mini">7일 +60% 이상 → 고점 대비 −15% 꺾이는 날 종가에 숏 신호</span></h2>${body}</section>`;
  }

  function draw(d) {
    if (d.status !== "ok") {
      $("barrier-status").textContent = d.message || "아직 기록이 없어요";
      return;
    }
    const stale = d.stale ? `<p class="chip warn">⚠️ 최신이 아니에요 (${when(d.generated_at)})</p>` : "";
    const nf = d.netflow || {};
    const nfLine = nf.last_day
      ? `BTC 거래소 순유입 z ${nf.z == null ? "-" : UI.num(nf.z, 2)} (${esc(nf.last_day)} 값, 저장 ${nf.snapshots || 0}일치)`
      : "BTC 거래소 순유입: 아직 저장 전";
    $("barrier-status").innerHTML = `${stale}<p class="mini">시작 ${esc(d.start)} · 마지막 일봉 ${esc(d.last_scan || "-")} · 종목 ${d.universe_n || "-"}개 · ${when(d.generated_at)} 갱신</p>
      <p class="mini">${nfLine}</p>
      <p class="mini">진입 = 신호 다음 날 00시(UTC) 시가, 1시간봉으로 +5%·−3% 먼저 닿는 쪽(같은 봉이면 손절), 14일 넘으면 종가 청산. 손익은 왕복 0.14%와 실제 펀딩을 뺀 값. 11/1~오늘은 백필이고 그 사이 상폐된 종목은 빠져 있어요.</p>`;
    $("barrier-rules").innerHTML = watchCard(d.watch || []) + (d.rules || []).map((r) => card(r, d.breakeven_pct || 39.3)).join("");
  }

  function load() {
    fetch("/api/barrier-paper")
      .then((r) => r.json())
      .then(draw)
      .catch(() => { $("barrier-status").textContent = "장벽 가상 추적을 불러오지 못했어요"; });
  }

  if ($("barrier-status")) {
    load();
    setInterval(load, 30 * 60 * 1000);
  }
})();
