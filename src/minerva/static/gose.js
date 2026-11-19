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

  // 규칙서 Markdown(저장소 gose_spec.md)을 작게 그린다: 제목·목록·표·코드 블록·굵게·`코드`.
  function inline(t) {
    return esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  }
  function md(text) {
    const out = [];
    const lines = String(text || "").split("\n");
    let i = 0;
    while (i < lines.length) {
      const l = lines[i];
      if (l.startsWith("```")) {
        const buf = [];
        i += 1;
        while (i < lines.length && !lines[i].startsWith("```")) { buf.push(lines[i]); i += 1; }
        out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
        i += 1;
        continue;
      }
      const h = /^(#{1,3}) (.*)$/.exec(l);
      if (h) { const n = Math.min(h[1].length + 1, 4); out.push(`<h${n}>${inline(h[2])}</h${n}>`); i += 1; continue; }
      if (l.startsWith("|")) {
        const rows = [];
        while (i < lines.length && lines[i].startsWith("|")) { rows.push(lines[i]); i += 1; }
        const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const body = rows.filter((r) => !/^\|[\s|:-]+\|$/.test(r));
        const [head, ...rest] = body;
        out.push(`<div class="scroll"><table class="acct"><thead><tr>${cells(head).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead>
          <tbody>${rest.map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }
      const li = /^(\s*)(- |\d+\. )(.*)$/.exec(l);
      if (li) {
        const ordered = /\d/.test(li[2]);
        const items = [];
        while (i < lines.length) {
          const m = /^(\s*)(- |\d+\. )(.*)$/.exec(lines[i]);
          if (m) { items.push({ sub: m[1].length > 0, t: m[3] }); i += 1; continue; }
          if (/^\s{2,}\S/.test(lines[i]) && items.length) { items[items.length - 1].t += ` ${lines[i].trim()}`; i += 1; continue; }
          break;
        }
        const tag = ordered ? "ol" : "ul";
        out.push(`<${tag}>${items.map((x) => `<li${x.sub ? " style=\"margin-left:16px\"" : ""}>${inline(x.t)}</li>`).join("")}</${tag}>`);
        continue;
      }
      if (l.trim()) out.push(`<p>${inline(l)}</p>`);
      i += 1;
    }
    return out.join("");
  }

  function avgSpec(s) {
    const rows = (s.summary || []).map((r) => `<tr><td><b>${esc(r.sym)}</b></td><td class="mini">${esc(r.period)}</td>
      <td>${pct(r.cagr, 1)}</td><td>${pct(r.mdd, 1)}</td><td>${num(r.cycles, 0)} · ${num(r.win, 0)}%</td>
      <td>${pct(r.mae_worst, 1)}</td><td>${num(r.hold_max_bars, 0)}</td><td>${pct(r.bh_cagr, 1)} · ${pct(r.bh_mdd, 1)}</td>
      <td>${r.random_p == null ? "-" : num(r.random_p, 2)}</td></tr>`).join("");
    return `<div class="spec">
      <p><b>${esc(s.title)}</b> · <span class="mini">${esc(s.label)} · ${esc(s.made)} 판 · 가상 계좌 ${esc((s.accounts || []).join(", "))}</span></p>
      <h3>한눈에</h3>
      <pre><code>${esc([
        "신호  일봉 종가에 스토캐(5)>20 · CCI(20)>−100 · RSI(14, 단순식)>30, 그중 하나가 오늘 위로 돌파,",
        "      그리고 직전 10봉(직전 신호 다음 봉부터) 안에 스토캐≤20 · CCI≤−100 · RSI≤33 이 각각 한 번 이상",
        "매수  무포지션이면 다음 날 시가에 1단위(사이클 시작 자산/3)",
        "추가  보유 중 또 신호면 다음 날 시가에 1단위 더, 최대 2번(최대 투입 = 자산 100%)",
        "익절  평단(수수료 포함)×1.10 지정가 전량, 추가할 때마다 다시 계산",
        "손절  없음 · 보유 기간 제한 없음 · 수수료 편도 0.05% · 레버리지 없음",
      ].join("\n"))}</code></pre>
      <h3>솔직한 주의점</h3>
      <ul>${(s.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("")}</ul>
      <h3>백테스트 수치 <span class="mini">가상 계좌와 같은 코드</span></h3>
      <div class="scroll"><table class="acct"><thead><tr><th>종목</th><th>기간</th><th>연복리</th><th>최대 낙폭</th>
        <th>사이클 · 승률</th><th>평단 대비 최저</th><th>최장 보유(거래일)</th><th>그냥 보유</th><th>무작위 p</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <p class="mini">무작위 p = 같은 횟수로 아무 날 들어가 같은 방식으로 익절했을 때 이 규칙보다 연복리가 같거나 높았던 비율(클수록 신호가 의미 없음).</p>
      <details><summary>규칙서 전문 (다른 에이전트용, <code>/api/gose</code>의 <code>spec.items[].markdown</code>)</summary>${md(s.markdown)}</details>
    </div>`;
  }

  // 공용 계좌 전략 규칙서(3-c0/S10, 4): 한눈에 · 숫자 · 주의점 · 전문
  function planSpec(s) {
    const b = s.backtest || {};
    const r = s.replay || {};
    const spy = r.spy || {};
    const line = (k, v) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`;
    return `<div class="spec">
      <p><b>${esc(s.title)}</b> · <span class="mini">${esc(s.label)} · ${esc(s.made)} 판 · 가상 계좌 ${esc((s.accounts || []).join(", "))}</span></p>
      <h3>한눈에</h3>
      <pre><code>${esc((s.brief || []).join("\n"))}</code></pre>
      <h3>솔직한 주의점</h3>
      <ul>${(s.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("")}</ul>
      <h3>5년 백테스트 <span class="mini">${esc((b.window || []).join(" ~ "))} · ${num(b.symbols, 0)}종목 · 계좌 자리 제한 없이 모든 거래</span></h3>
      <div class="scroll"><table class="acct"><tbody>
        ${line("거래 · 승률", `${num(b.n, 0)}번 · ${num(b.win, 1)}%`)}
        ${line("거래당 평균 · 최악", `${pct(b.mean, 2)} · ${pct(b.worst, 1)}`)}
        ${line("동일비중 연복리 · 최대 낙폭", `${pct(b.cagr, 1)} · ${pct(b.mdd, 1)}`)}
        ${line("같은 종목 그냥 보유", `${pct(b.bh_cagr, 1)} · ${pct(b.bh_mdd, 1)}`)}
        ${line("무작위 대비 백분위", `${num(b.random_pct, 1)} (다중검정 기준 ${num(b.bonferroni_pct, 2)})`)}
        ${line("판정", `<b>${esc(b.verdict || "-")}</b>`)}
      </tbody></table></div>
      <p class="mini">${esc(b.win_def || "")}</p>
      <h3>시작일 돌려 보기 <span class="mini">${num(r.starts, 0)}번 시작 · 12개월 · 6절 계좌 그대로</span></h3>
      <div class="scroll"><table class="acct"><thead><tr><th></th><th>이 전략</th><th>SPY 보유</th></tr></thead><tbody>
        <tr><td>중앙값</td><td>${pct(r.median, 1)}</td><td>${pct(spy.median, 1)}</td></tr>
        <tr><td>하위 10%</td><td>${pct(r.p10, 1)}</td><td>${pct(spy.p10, 1)}</td></tr>
        <tr><td>플러스 비율</td><td>${num(r.pos_pct, 0)}%</td><td>${num(spy.pos_pct, 0)}%</td></tr>
        <tr><td>SPY보다 나음</td><td>${num(r.beat_spy_pct, 0)}%</td><td></td></tr>
        <tr><td>계좌 낙폭 중앙 · 최악</td><td>${pct(r.mdd_median, 1)} · ${pct(r.mdd_worst, 1)}</td><td>${pct(spy.mdd_median, 1)} · ${pct(spy.mdd_worst, 1)}</td></tr>
        <tr><td>종목 하나 최악 미실현</td><td>${pct(r.open_dd_worst, 0)}</td><td></td></tr>
        <tr><td>판정</td><td><b>${esc(r.verdict || "-")}</b></td><td></td></tr>
      </tbody></table></div>
      <details><summary>규칙서 전문 (다른 에이전트용, <code>/api/gose</code>의 <code>spec.items[].markdown</code>)</summary>${md(s.markdown)}</details>
    </div>`;
  }

  function refList(s) {
    const rs = s.reference || [];
    if (!rs.length) return "";
    return `<details><summary>나머지 가상 계좌 ${rs.length}개 (참고, 미검증 · 한 줄 정의)</summary>
      <div class="scroll"><table class="acct"><thead><tr><th>계좌</th><th>정의</th><th>12개월 중앙 · 판정</th><th>5년</th><th>코드</th></tr></thead><tbody>
      ${rs.map((r) => `<tr><td><b>${esc(r.key)}</b></td><td class="mini">${esc(r.def)}</td>
        <td>${pct(r.replay_median, 1)} · ${esc(r.replay_verdict || "-")}</td><td>${esc(r.backtest_verdict || "-")}</td>
        <td class="mini"><code>${esc(r.code)}</code></td></tr>`).join("")}
      </tbody></table></div>
      <p class="mini">${esc(s.replay_def || "")} 손절 없음(원래 청산·시간 청산). 이 계좌들은 전문이 없어요: 그대로 매매하려면 코드를 읽어야 해요.</p></details>`;
  }

  const SPEC_KEY = "minerva-gose-spec";
  let specData = null;
  let specSel = (() => { try { return localStorage.getItem(SPEC_KEY); } catch (e) { return null; } })();

  function spec(s) {
    if (!s || s.status !== "ok") return "<p class=\"placeholder\">규칙서를 아직 못 읽었어요</p>";
    const items = s.items && s.items.length ? s.items : [dict(s)];
    if (!items.some((it) => it.key === specSel)) specSel = items[0].key;
    const cur = items.find((it) => it.key === specSel);
    const btns = items.map((it) => `<button type="button" data-spec="${esc(it.key)}" class="${it.key === specSel ? "on" : ""}"
      aria-pressed="${it.key === specSel}">${esc(it.short || it.title || it.key)}</button>`).join("");
    return `<div class="filters" role="group" aria-label="규칙서 고르기">${btns}</div>
      ${cur.key === "avg" ? avgSpec(cur) : planSpec(cur)}${refList(s)}`;
  }
  function dict(s) { return Object.assign({ key: "avg" }, s); }

  function drawSpec() {
    const box = $("gose-spec");
    if (!box) return;
    box.innerHTML = spec(specData);
    box.querySelectorAll("button[data-spec]").forEach((b) => {
      b.addEventListener("click", () => {
        specSel = b.dataset.spec;
        try { localStorage.setItem(SPEC_KEY, specSel); } catch (e) { /* 저장 막혀도 전환은 된다 */ }
        drawSpec();
      });
    });
  }

  function rounds(s) {
    const rs = (s && s.rounds) || [];
    if (!rs.length) return "";
    return `<h3>4·5차 (원저자 패턴 재현 · 물타기 익절 고르기)</h3>
      <div class="scroll"><table class="acct"><thead><tr><th>차수</th><th>무엇</th><th>결과</th><th>판정</th></tr></thead><tbody>
      ${rs.map((r) => `<tr><td>${num(r.round, 0)}차</td><td>${esc(r.what)}</td><td class="mini">${esc(r.result)}</td><td><b>${esc(r.verdict)}</b></td></tr>`).join("")}
      </tbody></table></div>`;
  }

  function rulebook(r) {
    if (!r || r.status === "pending") return "<p class=\"placeholder\">규칙책 요약을 아직 못 읽었어요</p>";
    const t = r.totals || {};
    const cols = r.topic_cols || {};
    const topics = (r.topics || []).map((x) => `<tr><td>${esc(x.id)}</td><td>${esc(x.name)}</td>
      <td>${x.auto}</td><td>${x.vague}</td><td>${x.manual}</td><td>${x.auto + x.vague + x.manual}</td></tr>`).join("");
    const conflicts = (r.conflicts || []).map((c) => `<li><b>${esc(c.id)}</b> ${esc(c.text)}</li>`).join("");
    const cands = (r.candidates || []).map((c) => `<tr><td>${esc(c.key)}</td><td>${esc(c.name)}</td>
      <td>${esc(c.text)}</td><td class="mini">${esc(c.rules)}</td><td>${esc(c.tested)}</td></tr>`).join("");
    const u = r.update;
    const upd = u && Array.isArray(u.rules) ? `<details><summary>새 장 반영 ${esc(u.made || "")}: 규칙 ${u.rules.length}개 추가
        (${esc(u.range || "")}, 새 글 ${num(u.posts, 0)}개)</summary><p class="mini">${esc(u.note || "")}</p>
        <div class="scroll"><table class="acct"><thead><tr><th>ID</th><th>기계화</th><th>내용(우리 말)</th><th>관련</th></tr></thead><tbody>${
          u.rules.map((x) => `<tr><td>${esc(x.id)}</td><td>${esc(x.mech)}</td><td>${esc(x.text)}${
            x.conflict ? `<div class="mini">⚠ ${esc(x.conflict)}</div>` : ""}</td><td class="mini">${esc(x.relates)}</td></tr>`).join("")
        }</tbody></table></div></details>` : "";
    return `<p>${esc(r.note || "")}</p>
      ${upd}
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

  // "SCR 3단 + 물타기" 계좌: 한 종목, 사이클 상태(추가 횟수·평단·익절가·평단 대비 낙폭·보유 일수)
  function avgCycle(p, title) {
    if (!p) return "";
    const fills = (p.fills || []).map((f) => `${esc(f[1])} ${num(f[2], 4)} (${usd(f[3])})`).join(" · ");
    return `<div class="asset"><b>${esc(title)}</b> ${esc(p.sym)} · ${esc(p.entry_date)} 시작 · 보유 ${num(p.days_held, 0)}일(${num(p.bars_held, 0)}거래일)<br>
      추가 ${p.adds}/2번 · 넣은 돈 ${usd(p.invested)} · 평단 ${num(p.avg, 4)} · <b>익절가 ${num(p.tp_px, 4)}</b><br>
      지금 ${num(p.last_px, 4)} <span class="mini">${esc(p.last_date)}</span> · 평가 ${pct(p.ret_pct)} ·
      평단 대비 지금 ${pct(p.mae_now_pct)} · 사이클 최저 ${pct(p.mae_worst_pct)}
      <div class="mini">체결: ${fills}</div></div>`;
  }

  function avgAccount(a, cap) {
    const sig = (a.signals || []).map((s) => `<li>${esc(s.date)} 신호 → <b>${esc(s.action)}</b>
      ${s.amount ? `<span class="mini">약 ${usd(s.amount)} · 종가 ${num(s.close, 4)}</span>` : ""}</li>`).join("");
    const closed = (a.closed || []).slice(0, 20).map((c) => `<tr><td>${esc(c.entry_date)}<br>${esc(c.exit_date)}</td>
      <td>${num(c.entry_px, 4)}<br>${num(c.exit_px, 4)}</td><td>${c.adds}</td><td>${pct(c.ret_pct)}</td><td>${pct(c.mae_worst_pct)}</td>
      <td>${num(c.days_held, 0)}일</td></tr>`).join("");
    const ind = a.last_ind || {};
    const ref = a.reference || {};
    return `<section class="card">
      <h2>${esc(a.key)} · ${esc(a.name)} <span class="mini">${esc((a.symbols || []).join(""))}만 · 손절 없음 · 📘 규칙서대로</span></h2>
      <p><b>${pct(a.return_pct)}</b> 시작 뒤 · 평가 ${usd(a.equity)} / ${usd(cap)} · 현금 ${usd(a.cash)} · 최대 낙폭 ${pct(a.mdd_pct)}<br>
        닫힌 사이클 ${a.n_closed}번 · 승률 ${a.win_rate == null ? "-" : `${num(a.win_rate, 1)}%`}
        <span class="mini">(손절이 없어 닫힌 건 다 익절)</span></p>
      ${a.data === false ? "<p class=\"mini\">일봉 자료가 아직 없어요</p>" : ""}
      ${spark(a.curve, cap)}
      ${(a.open || []).length ? avgCycle(a.open[0], "열린 사이클") : "<p>무포지션 · 다음 신호를 기다려요</p>"}
      ${sig ? `<h3>오늘 신호 <span class="mini">다음 시가 체결 예정</span></h3><ul>${sig}</ul>` : "<p class=\"mini\">오늘 신호 없음</p>"}
      <p class="mini">마지막 봉 ${esc(a.last_bar || "-")} · 스토캐 ${num(ind.stoch_k, 1)} · CCI ${num(ind.cci, 1)} · RSI ${num(ind.rsi, 1)} ·
        마지막 신호 ${esc(a.last_sig || "-")}</p>
      ${ref.open ? `<details><summary>참고: ${esc(ref.from)}부터 이 규칙을 돌렸다면 지금</summary>
        <p class="mini">${esc(ref.note || "")}</p>${avgCycle(ref.open, "규칙상 열린 사이클")}</details>` : ""}
      ${closed ? `<details><summary>닫힌 사이클 (최근 20)</summary><div class="scroll"><table class="acct"><thead><tr>
        <th>시작·익절</th><th>평단·익절가</th><th>추가</th><th>수익</th><th>평단 대비 최저</th><th>보유</th></tr></thead><tbody>${closed}</tbody></table></div></details>` : ""}
    </section>`;
  }

  function account(a, cap) {
    if (a.strategy === "avg") return avgAccount(a, cap);
    const sig = (a.signals || []).map((s) => `<li>${esc(s.date)} <b>${esc(s.sym)}</b> ${esc(s.action)}
      ${s.slot === false ? "<span class=\"mini\">(자리 없음, 건너뜀 예정)</span>" : ""}
      <span class="mini">종가 ${num(s.close, 4)} · ${esc(s.reason)}</span></li>`).join("");
    const open = (a.open || []).map((p) => `<tr><td>${esc(p.sym)}${p.side === "short" ? " 숏" : ""}</td>
      <td>${esc(p.entry_date)}</td><td>${num(p.entry_px, 4)}</td><td>${num(p.last_px, 4)}<br><span class="mini">${esc(p.last_date)}</span></td>
      <td>${usd(p.invested)}</td><td>${pct(p.ret_pct)}</td></tr>`).join("");
    const closed = (a.closed || []).slice(0, 20).map((c) => `<tr><td>${esc(c.sym)}</td>
      <td>${esc(c.entry_date)}<br>${esc(c.exit_date)}</td><td>${num(c.entry_px, 4)}<br>${num(c.exit_px, 4)}</td>
      <td>${pct(c.ret_pct)}</td><td class="mini">${esc(c.reason_out)}</td></tr>`).join("");
    const stop = a.group === "quant" ? "기법 청산(손절 없음)" : a.stop_pct ? `손절 ${num(a.stop_pct * 100, 0)}%` : "원문 청산(손절 없음)";
    const v2 = a.round === 2 ? ` · <b>${esc(a.label || "2차")}</b>` : "";
    const src = a.signal_source ? `<p class="mini">신호: ${esc(a.signal_source)} 일봉(매매 안 함${a.signal_source_through ? `, ${esc(a.signal_source_through)}까지` : ", 자료 없음"}) →
      ${esc((a.symbols || []).join("·"))}만 매수</p>` : "";
    return `<section class="card">
      <h2>${esc(a.key)} · ${esc(a.name)} <span class="mini">${esc(stop)} · ${esc((a.rule_ids || []).join(", "))}${a.side === "short" ? " · 숏(참고)" : ""}${v2}${a.group === "quant" ? ` · <b>${esc(a.label || "퀀트")}</b>` : ""}</span></h2>
      ${src}
      ${a.caveat ? `<p class="mini">⚠️ ${esc(a.caveat)}</p>` : ""}
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

  // 퀀트 기법(비교용): 5년 백테스트 표(backtest.json의 quant) + 가상 계좌 요약
  function quantBacktest(q) {
    if (!q || q.status !== "ok") return "<p class=\"placeholder\">5년 백테스트가 아직 없어요</p>";
    const bh = q.buy_hold || {};
    const spy = q.buy_hold_spy;
    const rows = (q.rows || []).map((r) => {
      const rd = r.random || {};
      return `<tr><td><b>${esc(r.key)}</b><br><span class="mini">${esc(r.name)}</span>${r.caveat ? `<br><span class="mini">⚠️ ${esc(r.caveat)}</span>` : ""}</td>
        <td>${pct(r.cagr, 1)}</td><td>${pct(r.mdd, 1)}</td><td>${num(r.trades, 0)}</td>
        <td>${r.win_rate == null ? "-" : `${num(r.win_rate, 1)}%`}</td><td>${r.avg_hold_days == null ? "-" : `${num(r.avg_hold_days, 0)}일`}</td>
        <td>${pct(rd.cagr_median, 1)}<br><span class="mini">${num(rd.cagr_p05, 1)} ~ ${num(rd.cagr_p95, 1)}</span></td>
        <td>${rd.pct == null ? "-" : num(rd.pct, 1)}${r.beats_random ? " <b class=\"up\">우위</b>" : ""}</td>
        <td>${pct(r.vs_bh, 1)}</td></tr>`;
    }).join("");
    return `<p class="mini">최근 ${num(q.years, 0)}년(${esc((q.window || []).join(" ~ "))}) · 종목 ${num(q.symbols, 0)}개 · 수수료 편도 ${num((q.fee_per_side || 0) * 100, 2)}% ·
        신호 다음 날 시가 · 1만 달러, 동시 10종목, 전날 평가액의 10% · 그날까지 마감된 봉만 씀</p>
      <p>비교 기준: 동일 비중 그냥 보유(창 첫날 있던 ${num(bh.symbols, 0)}종목) 연 ${pct(bh.cagr, 1)} · 최대 낙폭 ${pct(bh.mdd, 1)}
        ${spy ? ` · SPY 그냥 보유 연 ${pct(spy.cagr, 1)} · ${pct(spy.mdd, 1)}` : ""}</p>
      <div class="scroll"><table class="acct"><thead><tr><th>기법</th><th>연수익</th><th>최대 낙폭</th><th>거래</th><th>승률</th><th>평균 보유</th>
        <th>무작위 연수익(중간, 5~95%)</th><th>무작위 백분위</th><th>그냥 보유 대비</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="mini">무작위 = 같은 거래 수·같은 보유 봉 수로 아무 종목·아무 날에 들어간 계좌 ${num(((q.rows || [])[0] || {}).random?.runs, 0)}번. 백분위 95 넘으면 "우위".
        과거 결과는 미래를 보장하지 않아요.</p>
      ${(q.caveats || []).length ? `<details><summary>주의점</summary><ul>${q.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul></details>` : ""}`;
  }

  function quantPaper(p, accts) {
    if (!p || p.status === "pending" || !accts.length) return "<p class=\"mini\">가상 계좌는 다음 계산부터 보여요</p>";
    const summary = accts.map((a) => `<tr><td><b>${esc(a.key)}</b> <span class="mini">${esc(a.source || "")}</span></td><td>${pct(a.return_pct)}</td>
      <td>${a.win_rate == null ? "-" : `${num(a.win_rate, 1)}%`} <span class="mini">(${a.n_closed})</span></td>
      <td>${(a.open || []).length}</td><td>${(a.signals || []).length}</td></tr>`).join("");
    return `<h3>가상 계좌 <span class="mini">시작 ${esc(p.start)} 뒤 신호만 · 실제 주문 없음</span></h3>
      <div class="scroll"><table class="acct"><thead><tr><th>계좌</th><th>시작 뒤</th><th>승률(닫힌 수)</th><th>보유</th><th>오늘 신호</th></tr></thead>
      <tbody>${summary}</tbody></table></div>`;
  }

  function paper(p) {
    if (!p || p.status === "pending") {
      $("gose-paper-status").textContent = "가상매매가 아직 시작되지 않았어요 (첫 실행 뒤에 보여요)";
      $("gose-paper").innerHTML = "";
      $("gose-quant-paper").innerHTML = "";
      return;
    }
    const r = p.rules || {};
    const all = p.accounts || [];
    const accts = all.filter((a) => a.group !== "quant");
    const quant = all.filter((a) => a.group === "quant");
    const summary = accts.map((a) => `<tr><td>${esc(a.key)}${a.round === 2 ? ` <span class="mini">${esc(a.label || "2차")}</span>` : ""}</td><td>${pct(a.return_pct)}</td>
      <td>${a.win_rate == null ? "-" : `${num(a.win_rate, 1)}%`} <span class="mini">(${a.n_closed})</span></td>
      <td>${a.strategy === "avg" ? `${(a.open || []).length ? `추가 ${a.open[0].adds}/2` : 0}` : (a.open || []).length}</td><td>${(a.signals || []).length}</td></tr>`).join("");
    $("gose-paper-status").innerHTML = `<b>가상(실제 주문 없음)</b> · 시작 ${esc(p.start)} · 자료 ${esc(p.data_through || "-")}까지 ·
      종목 ${num(p.symbols_used, 0)}개${(p.failures || []).length ? ` · 받기 실패 ${p.failures.length}개(옛 자료 사용)` : ""}<br>
      <span class="mini">전략마다 ${usd(r.capital)} · (avg- 계좌는 📘 규칙서 크기: 한 종목, 자산/3씩 최대 3번) · 동시 ${r.max_positions}종목 · 새 포지션은 전날 평가액의 ${num((r.size_frac || 0) * 100, 0)}% ·
      수수료 편도 ${num((r.fee_per_side || 0) * 100, 2)}% · ${esc(r.fill || "")} 체결 · 시작일 뒤 신호만 · ${esc(p.generated_at || "")} 계산</span>
      <div class="scroll"><table class="acct"><thead><tr><th>계좌</th><th>시작 뒤</th><th>승률(닫힌 수)</th><th>보유</th><th>오늘 신호</th></tr></thead>
      <tbody>${summary}</tbody></table></div>`;
    $("gose-paper").innerHTML = accts.map((a) => account(a, r.capital || 10000)).join("");
    $("gose-quant-paper").innerHTML = quant.map((a) => account(a, r.capital || 10000)).join("");
    return quant;
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
        specData = d.spec;
        drawSpec();
        $("gose-rulebook").innerHTML = rulebook(d.rulebook);
        $("gose-backtest").innerHTML = backtest(d.backtest) + rounds(d.spec);
        const quant = paper(d.paper) || [];
        $("gose-quant").innerHTML = quantBacktest((d.backtest || {}).quant) + quantPaper(d.paper, quant);
        $("gose-feed").innerHTML = feed(d.feed);
      })
      .catch(() => { $("gose-paper-status").textContent = "고서 자료를 불러오지 못했어요"; });
  }

  if ($("tab-gose")) {
    load();
    setInterval(load, 10 * 60 * 1000);
  }
})();
