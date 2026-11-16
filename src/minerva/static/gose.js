// "📜 고서" 탭. /api/gose (minerva.gose가 쓴 ~/.minerva/gose/*와 저장소 규칙책 요약)를 그린다.
// 가상매매만 보여 준다. 실제 주문과는 아무 연결이 없다.
(() => {
  const $ = (id) => document.getElementById(id);
  const UI = window.MinervaUI;
  const { esc } = UI;
  const num = (v, d = 2) => UI.num(v, d);
  const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "");
  const pct = (v, d = 2) => (v == null ? "-" : `<span class="${cls(v)}">${v > 0 ? "+" : ""}${num(v, d)}%</span>`);
  const usd = (v) => (v == null ? "-" : `$${num(v, 2)}`);

  function rulebook(r) {
    if (!r || r.status === "pending") return "<p class=\"placeholder\">규칙책 요약을 아직 못 읽었어요</p>";
    const t = r.totals || {};
    const cols = r.topic_cols || {};
    const topics = (r.topics || []).map((x) => `<tr><td>${esc(x.id)}</td><td>${esc(x.name)}</td>
      <td>${x.auto}</td><td>${x.vague}</td><td>${x.manual}</td><td>${x.auto + x.vague + x.manual}</td></tr>`).join("");
    const conflicts = (r.conflicts || []).map((c) => `<li><b>${esc(c.id)}</b> ${esc(c.text)}</li>`).join("");
    const cands = (r.candidates || []).map((c) => `<tr><td>${esc(c.key)}</td><td>${esc(c.name)}</td>
      <td>${esc(c.text)}</td><td class="mini">${esc(c.rules)}</td><td>${esc(c.tested)}</td></tr>`).join("");
    return `<p>${esc(r.note || "")}</p>
      <p>규칙 <b>${num(t.rules, 0)}</b>개 · 서로 부딪히는 규칙 <b>${num(t.conflicts, 0)}</b>쌍 ·
        작성자 통계 ${num(t.author_stats, 0)}개(미검증) · 값이 정해지지 않은 말 ${num(t.undefined_values, 0)}개</p>
      <details><summary>주제별 개수</summary><div class="scroll"><table class="acct"><thead><tr><th>ID</th><th>주제</th>
        <th>${esc(cols.auto || "가능")}</th><th>${esc(cols.vague || "애매")}</th><th>${esc(cols.manual || "불가")}</th>
        <th>합계</th></tr></thead><tbody>${topics}</tbody></table></div></details>
      <details><summary>시험에 영향이 큰 충돌</summary><ul>${conflicts}</ul>
        <p class="mini">${esc(r.conflict_note || "")}</p></details>
      <h3>후보 매매법 ${(r.candidates || []).length}개</h3>
      <div class="scroll"><table class="acct"><thead><tr><th>#</th><th>이름</th><th>요약(우리 말)</th><th>규칙</th>
        <th>시험한 변형</th></tr></thead><tbody>${cands}</tbody></table></div>
      <p class="mini">${esc(r.candidate_note || "")}</p>`;
  }

  function backtest(b) {
    if (!b || b.status === "pending") return "<p class=\"placeholder\">백테스트 요약이 아직 없어요</p>";
    const m = b.meta || {};
    const main = (m.main || []).join(" ~ ");
    const all = b.rows || [];
    const near = all.filter((r) => r.basic_pass && !r.verified);
    // 2차 줄은 주의점(2021-22 성적, 종목 쏠림)을 판정 칸 아래에 붙인다
    const tr = (r) => `<tr>
      <td><b>${esc(r.variant)}</b><br><span class="mini">${esc(r.name)}</span></td>
      <td>${esc(r.stop)}</td><td>${num(r.n, 0)}</td><td>${r.win == null ? "-" : `${num(r.win, 1)}%`}</td>
      <td>${pct(r.mean)}</td><td>${pct(r.cagr, 1)}</td><td>${pct(r.mdd, 1)}</td>
      <td>${r.random_pct == null ? "-" : num(r.random_pct, 1)}</td>
      <td><b class="${r.verified ? "up" : ""}">${esc(r.verdict)}</b><br><span class="mini">${esc(r.reason)}</span>
        ${r.round === 2 && (r.caveats || []).length ? `<ul class="mini">${r.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` : ""}</td></tr>`;
    const head = `<thead><tr><th>변형</th><th>손절</th><th>거래</th><th>승률</th><th>평균</th>
        <th>연수익</th><th>최대 낙폭</th><th>무작위 백분위</th><th>판정</th></tr></thead>`;
    const r1 = all.filter((r) => r.round !== 2);
    const r2 = all.filter((r) => r.round === 2);
    const rd = b.rounds || {};
    const rows = r1.map(tr).join("");
    const rows2 = r2.length ? `<h3>${esc((rd["2"] || {}).label || "2차")} ${r2.length}개 변형 · 검증 통과 ${(rd["2"] || {}).verified ?? 0}개</h3>
      <p class="mini">1차 결과를 본 뒤 그림 판독으로 뜻을 정해 다시 등록한 규칙. 변동성 신호는 UVXY 일봉·주봉(UVIX 대리)으로 보고 지수 ETF만 삼.
        특히 3v2는 2021-22 하락장에서 손실, 11v2a는 2021-22 신호 0번이고 수익이 SOXL에 쏠려 있어요.</p>
      <div class="scroll"><table class="acct">${head}<tbody>${r2.map(tr).join("")}</tbody></table></div>` : "";
    const nearHtml = near.map((r) => `<div class="asset"><b>${esc(r.variant)}</b> ${esc(r.name)} · 손절 ${esc(r.stop)}
      <ul>${(r.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div>`).join("");
    return `<p style="font-size:18px;margin:4px 0"><b>검증 통과 ${b.verified}개</b> <span class="mini">/ ${r2.length ? "1·2차 합계 " : ""}${b.variants}개 변형</span></p>
      <p class="mini">최근 5년(${esc(main)}, 진입일 기준) · 종목 ${num(m.symbols, 0)}개 · 수수료 편도 ${num((m.fee_per_side || 0) * 100, 2)}% ·
        다음 날 시가 체결 · 1배. 결과를 보기 전에 규칙과 통과 기준을 먼저 적어 두고 돌렸어요.</p>
      <details><summary>통과 기준</summary><ol>${(b.gates || []).map((g) => `<li>${esc(g)}</li>`).join("")}</ol></details>
      ${near.length ? `<h3>근접 ${near.length}개 (검증 통과 아님)</h3>${nearHtml}` : ""}
      ${r2.length ? `<h3>1차 ${r1.length}개 변형</h3>` : ""}
      <div class="scroll"><table class="acct">${head}<tbody>${rows}</tbody></table></div>
      ${rows2}
      <p class="mini">평균 = 거래 한 번 수수료 뒤 수익. 연수익·최대 낙폭 = 그날 열린 거래를 같은 비중으로 묶은 계좌.
        무작위 백분위 = 같은 종목·같은 보유 일수로 아무 날 들어간 1000번 중 몇 %보다 나았나. 과거 결과는 미래를 보장하지 않아요.</p>`;
  }

  // 작은 자산 곡선(인라인 SVG). 시작 1만 달러 점선 위는 up색, 아래는 down색.
  function spark(curve, base) {
    if (!curve || curve.length < 2) return "<div class=\"mini\">곡선은 이틀째부터 그려요</div>";
    const W = 300, H = 60, P = 3;
    const vs = curve.map((p) => p[1]);
    const lo = Math.min(base, ...vs), hi = Math.max(base, ...vs);
    const span = hi - lo || 1;
    const x = (i) => P + (i * (W - 2 * P)) / (curve.length - 1);
    const y = (v) => H - P - ((v - lo) * (H - 2 * P)) / span;
    const pts = vs.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    const last = vs[vs.length - 1];
    const color = last >= base ? "var(--up)" : "var(--down)";
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none" role="img"
      aria-label="자산 곡선 ${esc(curve[0][0])} ~ ${esc(curve[curve.length - 1][0])}">
      <line x1="0" x2="${W}" y1="${y(base).toFixed(1)}" y2="${y(base).toFixed(1)}" stroke="var(--muted)"
        stroke-dasharray="3 3" stroke-width="1" vector-effect="non-scaling-stroke"/>
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
  }

  function account(a, cap) {
    const sig = (a.signals || []).map((s) => `<li>${esc(s.date)} <b>${esc(s.sym)}</b> ${esc(s.action)}
      ${s.slot === false ? "<span class=\"mini\">(자리 없음, 건너뜀 예정)</span>" : ""}
      <span class="mini">종가 ${num(s.close, 4)} · ${esc(s.reason)}</span></li>`).join("");
    const open = (a.open || []).map((p) => `<tr><td>${esc(p.sym)}${p.side === "short" ? " 숏" : ""}</td>
      <td>${esc(p.entry_date)}</td><td>${num(p.entry_px, 4)}</td><td>${num(p.last_px, 4)}<br><span class="mini">${esc(p.last_date)}</span></td>
      <td>${usd(p.invested)}</td><td>${pct(p.ret_pct)}</td></tr>`).join("");
    const closed = (a.closed || []).slice(0, 20).map((c) => `<tr><td>${esc(c.sym)}</td>
      <td>${esc(c.entry_date)}<br>${esc(c.exit_date)}</td><td>${num(c.entry_px, 4)}<br>${num(c.exit_px, 4)}</td>
      <td>${pct(c.ret_pct)}</td><td class="mini">${esc(c.reason_out)}</td></tr>`).join("");
    const stop = a.stop_pct ? `손절 ${num(a.stop_pct * 100, 0)}%` : "원문 청산(손절 없음)";
    const v2 = a.round === 2 ? ` · <b>${esc(a.label || "2차")}</b>` : "";
    const src = a.signal_source ? `<p class="mini">신호: ${esc(a.signal_source)} 일봉(매매 안 함${a.signal_source_through ? `, ${esc(a.signal_source_through)}까지` : ", 자료 없음"}) →
      ${esc((a.symbols || []).join("·"))}만 매수</p>` : "";
    return `<section class="card">
      <h2>${esc(a.key)} · ${esc(a.name)} <span class="mini">${esc(stop)} · ${esc((a.rule_ids || []).join(", "))}${a.side === "short" ? " · 숏(참고)" : ""}${v2}</span></h2>
      ${src}
      <p><b>${pct(a.return_pct)}</b> 시작 뒤 · 평가 ${usd(a.equity)} / ${usd(cap)} · 최대 낙폭 ${pct(a.mdd_pct)}<br>
        닫힌 거래 ${a.n_closed}번 · 승률 ${a.win_rate == null ? "-" : `${num(a.win_rate, 1)}%`} ·
        보유 ${(a.open || []).length}/10 · 자리 없어 건너뜀 ${a.skipped}번</p>
      ${spark(a.curve, cap)}
      ${sig ? `<h3>오늘 신호 <span class="mini">다음 시가 체결 예정</span></h3><ul>${sig}</ul>` : "<p class=\"mini\">오늘 신호 없음</p>"}
      ${open ? `<details open><summary>보유 ${(a.open || []).length}개</summary><div class="scroll"><table class="acct"><thead><tr>
        <th>종목</th><th>진입일</th><th>진입가</th><th>지금</th><th>넣은 돈</th><th>평가</th></tr></thead><tbody>${open}</tbody></table></div></details>` : ""}
      ${closed ? `<details><summary>닫힌 거래 (최근 20)</summary><div class="scroll"><table class="acct"><thead><tr>
        <th>종목</th><th>진입·청산</th><th>가격</th><th>수익</th><th>청산 이유</th></tr></thead><tbody>${closed}</tbody></table></div></details>` : ""}
    </section>`;
  }

  function paper(p) {
    if (!p || p.status === "pending") {
      $("gose-paper-status").textContent = "가상매매가 아직 시작되지 않았어요 (첫 실행 뒤에 보여요)";
      $("gose-paper").innerHTML = "";
      return;
    }
    const r = p.rules || {};
    const accts = p.accounts || [];
    const summary = accts.map((a) => `<tr><td>${esc(a.key)}${a.round === 2 ? ` <span class="mini">${esc(a.label || "2차")}</span>` : ""}</td><td>${pct(a.return_pct)}</td>
      <td>${a.win_rate == null ? "-" : `${num(a.win_rate, 1)}%`} <span class="mini">(${a.n_closed})</span></td>
      <td>${(a.open || []).length}</td><td>${(a.signals || []).length}</td></tr>`).join("");
    $("gose-paper-status").innerHTML = `<b>가상(실제 주문 없음)</b> · 시작 ${esc(p.start)} · 자료 ${esc(p.data_through || "-")}까지 ·
      종목 ${num(p.symbols_used, 0)}개${(p.failures || []).length ? ` · 받기 실패 ${p.failures.length}개(옛 자료 사용)` : ""}<br>
      <span class="mini">전략마다 ${usd(r.capital)} · 동시 ${r.max_positions}종목 · 새 포지션은 전날 평가액의 ${num((r.size_frac || 0) * 100, 0)}% ·
      수수료 편도 ${num((r.fee_per_side || 0) * 100, 2)}% · ${esc(r.fill || "")} 체결 · 시작일 뒤 신호만 · ${esc(p.generated_at || "")} 계산</span>
      <div class="scroll"><table class="acct"><thead><tr><th>계좌</th><th>시작 뒤</th><th>승률(닫힌 수)</th><th>보유</th><th>오늘 신호</th></tr></thead>
      <tbody>${summary}</tbody></table></div>`;
    $("gose-paper").innerHTML = accts.map((a) => account(a, r.capital || 10000)).join("");
  }

  const SIDE = { long: "롱", short: "숏" };
  function feed(b) {
    if (!b || b.status === "pending" || !(b.entries || []).length) {
      return "<p class=\"placeholder\">아직 없음 (매일 23:10에 새 글을 확인해요)</p>";
    }
    const rows = b.entries.map((e) => {
      const rules = (e.rules || []).map((r) => `<li>[${esc(r.topic)}] ${esc(r.rule_paraphrase)}
        <span class="mini">기계화 ${esc(r.mechanizable)}${(r.relates_to || []).length ? ` · 관련 ${esc(r.relates_to.join(", "))}` : ""}${r.conflict ? ` · 충돌: ${esc(r.conflict)}` : ""}</span></li>`).join("");
      const calls = (e.calls || []).map((c) => `<li><b>${esc(c.instrument)}</b>${c.instrument !== c.symbol ? ` (${esc(c.symbol)})` : ""}
        <b class="${c.side === "long" ? "up" : "down"}">${SIDE[c.side] || esc(c.side)}</b> · ${esc(c.timeframe || "")}
        <span class="mini">${esc(c.reason || "")}${c.levels ? ` · 수준 ${esc(c.levels)}` : ""}${c.confidence ? ` · ${esc(c.confidence)}` : ""}</span><br>
        우리 점검: ${esc(c.check || "-")} SCR · 판단: <b class="${c.verdict === "추천" ? (c.side === "long" ? "up" : "down") : ""}">${esc(c.verdict || "-")}</b>
        <span class="mini">(${esc(c.verdict_reason || "")})${c.perp ? ` · 바이낸스 ${esc(c.perp)}` : ""}</span></li>`).join("");
      return `<div class="asset"><b>${esc(e.title)}</b> <span class="mini">${esc(e.date)}${e.category ? ` · ${esc(e.category)}` : ""}</span>
        ${e.error ? `<p class="mini">분석 실패: ${esc(e.error)}</p>` : `<p>${esc(e.summary || "").replace(/\n/g, "<br>")}</p>`}
        ${rules ? `<details><summary>규칙 변경 후보 ${(e.rules || []).length}개</summary><ul>${rules}</ul></details>` : ""}
        ${calls ? `<h3>매매 의견</h3><ul>${calls}</ul>` : ""}</div>`;
    }).join("");
    return `<p class="mini">${esc(b.label || "")} · ${esc(b.generated_at || "")} 갱신 · 추천 = 원전 의견과 우리 일·주봉 SCR 방향이 같고 월봉이 반대가 아닐 때. 주문은 하지 않아요.</p>${rows}`;
  }

  function load() {
    fetch("/api/gose")
      .then((r) => r.json())
      .then((d) => {
        $("gose-rulebook").innerHTML = rulebook(d.rulebook);
        $("gose-backtest").innerHTML = backtest(d.backtest);
        paper(d.paper);
        $("gose-feed").innerHTML = feed(d.feed);
      })
      .catch(() => { $("gose-paper-status").textContent = "고서 자료를 불러오지 못했어요"; });
  }

  if ($("tab-gose")) {
    load();
    setInterval(load, 10 * 60 * 1000);
  }
})();
