const decodeUtf8 = (encoded) => {
  const bytes = Uint8Array.from(atob(encoded.trim()), (character) =>
    character.charCodeAt(0)
  );
  return new TextDecoder().decode(bytes);
};

const viewerData = JSON.parse(
  decodeUtf8(document.querySelector("#viewer-data").textContent)
);
const replayHtml = decodeUtf8(
  document.querySelector("#replay-data").textContent
);

const state = {
  index: 0,
  tab: "overview",
  syncEnabled: true,
  replayUnavailable: false,
  bridgeStartedAt: Date.now(),
  lastPlayback: null,
};

const elements = {
  battleId: document.querySelector("#battle-id"),
  decisionCount: document.querySelector("#decision-count"),
  turnNumber: document.querySelector("#turn-number"),
  decisionPosition: document.querySelector("#decision-position"),
  matchup: document.querySelector("#matchup-strip"),
  panel: document.querySelector("#panel"),
  timeline: document.querySelector("#timeline"),
  previous: document.querySelector("#previous"),
  next: document.querySelector("#next"),
  syncToggle: document.querySelector("#replay-sync"),
  syncLabel: document.querySelector("#sync-label"),
  tabs: [...document.querySelectorAll(".tab")],
  replay: document.querySelector("#replay-frame"),
};

const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const titleCase = (value) =>
  String(value ?? "")
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());

const displayPokemon = (pokemon, fallback = "Unknown") =>
  pokemon?.name || titleCase(pokemon?.species) || fallback;

const percent = (fraction) => {
  const parsed = Number(fraction);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.min(100, Math.round(parsed * 100)));
};

const activePokemon = (side, opponent = false) => {
  const team = opponent ? side?.revealed_team : side?.team;
  if (!Array.isArray(team)) return null;
  return (
    team.find((pokemon) => pokemon.active) ||
    team.find((pokemon) => pokemon.species === side?.active) ||
    null
  );
};

const formatValue = (value) => {
  if (value === true) return "YES";
  if (value === false) return "NO";
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  if (typeof value === "object") {
    const entries = Object.entries(value);
    return entries.length
      ? entries.map(([key, item]) => `${titleCase(key)}: ${formatValue(item)}`).join(" · ")
      : "—";
  }
  return String(value);
};

const currentDecision = () => viewerData.decisions[state.index];

const readReplayState = () => {
  try {
    const battle = elements.replay.contentWindow?.Replays?.battle;
    if (
      !battle ||
      !Number.isFinite(battle.turn) ||
      !Number.isFinite(battle.currentStep)
    ) {
      return null;
    }
    return {
      turn: Number(battle.turn),
      currentStep: Number(battle.currentStep),
      paused: Boolean(battle.paused),
      ended: Boolean(battle.ended),
    };
  } catch {
    return null;
  }
};

const playbackDecisionIndex = (playback) => {
  let target = 0;
  let foundAnchor = false;
  viewerData.decisions.forEach((decision, index) => {
    if (
      Number.isInteger(decision.replay_step) &&
      decision.replay_step <= playback.currentStep
    ) {
      target = index;
      foundAnchor = true;
    }
  });
  if (foundAnchor) return target;

  const exactTurn = viewerData.decisions.findIndex(
    (decision) => decision.turn === playback.turn
  );
  if (exactTurn >= 0) return exactTurn;
  viewerData.decisions.forEach((decision, index) => {
    if (decision.turn <= playback.turn) target = index;
  });
  return target;
};

const updateSyncStatus = (playback) => {
  let mode = "is-connecting";
  let label = "连接播放器…";
  elements.syncToggle.setAttribute(
    "aria-pressed",
    state.syncEnabled ? "true" : "false"
  );
  if (!state.syncEnabled) {
    mode = "is-manual";
    label = "跟随回放";
  } else if (state.replayUnavailable) {
    mode = "is-unavailable";
    label = "同步不可用";
  } else if (!playback) {
    mode = "is-connecting";
    label = "连接播放器…";
  } else if (playback.ended) {
    mode = "is-following";
    label = "回放结束 · 跟随";
  } else if (playback.paused) {
    mode = "is-following";
    label = "已暂停 · 跟随";
  } else {
    mode = "is-playing";
    label = "播放中 · 跟随";
  }
  for (const className of [
    "is-connecting",
    "is-following",
    "is-playing",
    "is-manual",
    "is-unavailable",
  ]) {
    elements.syncToggle.classList.toggle(className, className === mode);
  }
  elements.syncLabel.textContent = label;
};

const syncFromReplay = () => {
  const playback = readReplayState();
  if (!playback) {
    if (Date.now() - state.bridgeStartedAt > 12000) {
      state.replayUnavailable = true;
    }
    updateSyncStatus(null);
    return;
  }

  state.replayUnavailable = false;
  state.lastPlayback = playback;
  elements.syncToggle.dataset.replayTurn = String(playback.turn);
  elements.syncToggle.dataset.replayStep = String(playback.currentStep);
  updateSyncStatus(playback);
  if (!state.syncEnabled) return;

  const target = playbackDecisionIndex(playback);
  if (target !== state.index) {
    selectDecision(target, "replay");
  }
};

const setSyncEnabled = (enabled) => {
  state.syncEnabled = enabled;
  updateSyncStatus(state.lastPlayback);
  if (enabled) syncFromReplay();
};

const hpMarkup = (pokemon) => {
  const hp = percent(pokemon?.hp_fraction);
  const lowClass = hp <= 25 ? "is-low" : "";
  return `<div class="hp-track" title="${hp}% HP"><i class="${lowClass}" style="width:${hp}%"></i></div>`;
};

const renderMatchup = (decision) => {
  const own = activePokemon(decision.snapshot.own_side);
  const opponent = activePokemon(decision.snapshot.opponent_side, true);
  elements.matchup.innerHTML = `
    <div class="matchup-side">
      <small>SHOWDOWNMIND / ACTIVE</small>
      <strong>${escapeHtml(displayPokemon(own, decision.snapshot.own_side?.active))}</strong>
      ${hpMarkup(own)}
    </div>
    <div class="versus">vs.</div>
    <div class="matchup-side is-opponent">
      <small>OPPONENT / REVEALED</small>
      <strong>${escapeHtml(displayPokemon(opponent, decision.snapshot.opponent_side?.active))}</strong>
      ${hpMarkup(opponent)}
    </div>
  `;
};

const badgeMarkup = (decision) => {
  const badges = [];
  if (decision.fallback_used) {
    badges.push('<span class="badge is-danger">FALLBACK USED</span>');
  } else {
    badges.push('<span class="badge is-signal">VALIDATED</span>');
  }
  if (decision.attempts > 1) {
    badges.push(`<span class="badge">${decision.attempts} ATTEMPTS</span>`);
  }
  if (decision.decision_normalizations?.length) {
    badges.push(
      `<span class="badge" title="${escapeHtml(
        decision.decision_normalizations.join(", ")
      )}">NORMALIZED</span>`
    );
  }
  if (decision.tool.call_ids.length) {
    badges.push('<span class="badge">NATIVE TOOL CALL</span>');
  }
  for (const reason of decision.reason_codes) {
    badges.push(`<span class="badge">${escapeHtml(reason)}</span>`);
  }
  return badges.join("");
};

const renderOverview = (decision) => {
  const action = decision.chosen_action;
  const label = action?.label || decision.action_id || "No recorded action";
  const confidence =
    typeof decision.confidence === "number"
      ? `${Math.round(decision.confidence * 100)}%`
      : "—";
  const rationale =
    decision.short_rationale ||
    "本次记录没有公开简短理由；这不影响动作与验证结果的可审计性。";
  return `
    <section class="choice-card">
      <div class="label-row">SELECTED ACTION / 最终选择</div>
      <div class="choice-main">
        <h2>${escapeHtml(label)}</h2>
        <div class="confidence">
          ${confidence}
          <small>CONFIDENCE</small>
        </div>
      </div>
      <div class="badges">${badgeMarkup(decision)}</div>
    </section>
    <p class="rationale">${escapeHtml(rationale)}</p>
    <div class="metrics">
      <div class="metric">
        <span class="metric-label">LATENCY</span>
        <strong>${decision.elapsed_seconds.toFixed(2)}s</strong>
      </div>
      <div class="metric">
        <span class="metric-label">TOKENS</span>
        <strong>${decision.usage.total_tokens.toLocaleString()}</strong>
      </div>
      <div class="metric">
        <span class="metric-label">LEGAL OPTIONS</span>
        <strong>${decision.snapshot.legal_actions.length}</strong>
      </div>
      <div class="metric">
        <span class="metric-label">MODEL</span>
        <strong title="${escapeHtml(decision.model_ids.join(", "))}">
          ${escapeHtml(decision.model_ids.at(-1) || "—")}
        </strong>
      </div>
      <div class="metric">
        <span class="metric-label">INPUT FORMAT</span>
        <strong>${escapeHtml(decision.policy_input.format || "—")}</strong>
      </div>
      <div class="metric">
        <span class="metric-label">REQUEST</span>
        <strong>#${decision.request_id}</strong>
      </div>
    </div>
  `;
};

const actionStats = (action) => {
  const details = action.details || {};
  if (action.kind === "switch") {
    return `${percent(details.hp_fraction)}% HP`;
  }
  const power = details.base_power ? `BP ${details.base_power}` : "STATUS";
  const accuracy =
    typeof details.accuracy === "number"
      ? `${Math.round(details.accuracy * 100)}% ACC`
      : "";
  return [power, accuracy].filter(Boolean).join("<br>");
};

const tacticalAction = (decision, actionId) =>
  (decision.tactical_analysis?.actions || []).find(
    (action) => action.action_id === actionId
  );

const tacticalStats = (analysis) => {
  if (!analysis) return "";
  if (analysis.kind === "move") {
    const multiplier =
      typeof analysis.type_multiplier === "number"
        ? `${analysis.type_multiplier}× TYPE`
        : "";
    const relative =
      typeof analysis.relative_damage === "number"
        ? `${Math.round(analysis.relative_damage * 100)}% REL`
        : "DYNAMIC POWER";
    return [multiplier, relative, titleCase(analysis.move_order)]
      .filter(Boolean)
      .join(" · ");
  }
  if (analysis.kind === "switch") {
    return `MATCHUP ${analysis.matchup_score} · ${titleCase(analysis.speed_relation)}`;
  }
  return "";
};

const renderActions = (decision) => {
  if (!decision.snapshot.legal_actions.length) {
    return '<div class="empty-copy">这个请求没有记录合法动作。</div>';
  }
  return `
    <div class="action-list">
      ${decision.snapshot.legal_actions
        .map(
          (action, index) => `
            <div class="action-card ${action.action_id === decision.action_id ? "is-chosen" : ""}">
              <span class="action-number">${String(index + 1).padStart(2, "0")}</span>
              <div class="action-copy">
                <strong>${escapeHtml(action.label)}</strong>
                <small>${escapeHtml(titleCase(action.kind))} · ${escapeHtml(titleCase(action.details?.type || action.details?.species || ""))}</small>
                <small>${escapeHtml(tacticalStats(tacticalAction(decision, action.action_id)))}</small>
              </div>
              <div class="action-stats">${actionStats(action)}</div>
            </div>
          `
        )
        .join("")}
    </div>
  `;
};

const teamMarkup = (team) => {
  if (!Array.isArray(team) || !team.length) {
    return '<div class="empty-copy">尚未观察到队伍信息。</div>';
  }
  return `
    <div class="team-list">
      ${team
        .map(
          (pokemon) => `
            <div class="pokemon-row ${pokemon.active ? "is-active" : ""} ${pokemon.fainted ? "is-fainted" : ""}">
              <strong>${escapeHtml(displayPokemon(pokemon))}</strong>
              <span>${escapeHtml(pokemon.status || (pokemon.fainted ? "FNT" : "OK"))}</span>
              <span>${percent(pokemon.hp_fraction)}% HP</span>
            </div>
          `
        )
        .join("")}
    </div>
  `;
};

const keyValuesMarkup = (value) => {
  const entries = Object.entries(value || {});
  if (!entries.length) return '<div class="empty-copy">没有已记录的场地效果。</div>';
  return `
    <div class="key-values">
      ${entries
        .map(
          ([key, item]) => `
            <div class="key-value">
              <small>${escapeHtml(titleCase(key))}</small>
              <strong>${escapeHtml(formatValue(item))}</strong>
            </div>
          `
        )
        .join("")}
    </div>
  `;
};

const renderState = (decision) => {
  const { own_side: own, opponent_side: opponent, field, resources } =
    decision.snapshot;
  return `
    <section class="state-section">
      <h3 class="subheading">OWN TEAM / 完整己方信息</h3>
      ${teamMarkup(own?.team)}
    </section>
    <section class="state-section">
      <h3 class="subheading">OPPONENT / 仅已公开信息</h3>
      ${teamMarkup(opponent?.revealed_team)}
    </section>
    <section class="state-section">
      <h3 class="subheading">FIELD & RESOURCES</h3>
      ${keyValuesMarkup({
        weather: field?.weather,
        field: field?.fields,
        ...resources,
      })}
    </section>
  `;
};

const eventMarkup = (events) => {
  if (!Array.isArray(events) || !events.length) {
    return '<div class="empty-copy">本回合没有新的结构化事件。</div>';
  }
  return `
    <div class="agent-list">
      ${events
        .map(
          (event) => `
            <div class="agent-list-item">
              <strong>${escapeHtml(titleCase(event.kind || "event"))}</strong>
              <span>${escapeHtml(event.actor || "")}${event.payload ? ` · ${escapeHtml(formatValue(event.payload))}` : ""}</span>
            </div>
          `
        )
        .join("")}
    </div>
  `;
};

const beliefsMarkup = (beliefState) => {
  const hypotheses = beliefState?.hypotheses || [];
  if (!Array.isArray(hypotheses) || !hypotheses.length) {
    return '<div class="empty-copy">没有达到展示条件的对手假设。</div>';
  }
  return `
    <div class="belief-grid">
      ${hypotheses
        .map(
          (belief) => `
            <div class="belief-card">
              <div>
                <strong>${escapeHtml(belief.subject || "Opponent")}</strong>
                <span class="belief-tier is-${escapeHtml(belief.confidence || "possible")}">${escapeHtml((belief.confidence || "possible").toUpperCase())}</span>
              </div>
              <p>${escapeHtml(titleCase(belief.kind || "hypothesis"))}: ${escapeHtml(formatValue(belief.value))}</p>
              <small>${escapeHtml((belief.evidence_ids || []).join(" · ") || "public prior")}</small>
            </div>
          `
        )
        .join("")}
    </div>
  `;
};

const planMarkup = (decision) => {
  const plan = decision.battle_plan || {};
  if (!plan.schema && !plan.win_condition) {
    return '<div class="empty-copy">这个策略模式没有本局作战计划。</div>';
  }
  const updated = Object.keys(decision.plan_update || {}).length > 0;
  const maintained = Object.keys(decision.plan_maintenance || {}).length > 0;
  const stateLabel = updated ? "UPDATED" : maintained ? "MAINTAINED" : "KEPT";
  return `
    <div class="plan-card">
      <div class="plan-heading">
        <strong>${escapeHtml(plan.win_condition || "Maintain a playable position")}</strong>
        <span class="badge ${updated || maintained ? "is-signal" : ""}">${stateLabel}</span>
      </div>
      <p>${escapeHtml(plan.tera_policy || "No Tera policy recorded.")}</p>
      ${keyValuesMarkup({
        trigger: decision.plan_trigger || "none",
        version: plan.version,
        preserve: plan.preserve,
        priority_targets: plan.priority_targets,
        risk_posture: plan.risk_posture,
        replan_triggers: plan.replan_triggers,
        request_replan: decision.request_replan,
        locally_removed_preserve: decision.plan_maintenance?.removed_preserve,
        locally_removed_targets: decision.plan_maintenance?.removed_priority_targets,
      })}
    </div>
  `;
};

const predictionMarkup = (decision) => {
  const current = decision.opponent_prediction || {};
  const resolution = decision.memory?.previous_prediction_resolution || {};
  return `
    <div class="prediction-grid">
      <div class="plan-card">
        <small class="metric-label">CURRENT PREDICTION</small>
        <strong>${escapeHtml(titleCase(current.kind || "none"))}</strong>
        <p>${escapeHtml(current.detail || "No prediction recorded.")}</p>
        <small>Confidence: ${typeof current.confidence === "number" ? `${Math.round(current.confidence * 100)}%` : "—"}</small>
      </div>
      <div class="plan-card">
        <small class="metric-label">PREVIOUS RESULT</small>
        <strong>${resolution.matched === true ? "MATCHED" : resolution.matched === false ? "MISSED" : "PENDING"}</strong>
        <p>${escapeHtml(resolution.actual_kind ? `Actual: ${titleCase(resolution.actual_kind)} · ${resolution.actual_detail || ""}` : "No prior prediction has resolved.")}</p>
      </div>
    </div>
  `;
};

const renderAgent = (decision) => `
  <section class="state-section">
    <h3 class="subheading">BATTLE PLAN / 本局作战计划</h3>
    ${planMarkup(decision)}
  </section>
  <section class="state-section">
    <h3 class="subheading">OPPONENT PREDICTION / 对手预测</h3>
    ${predictionMarkup(decision)}
  </section>
  <section class="state-section">
    <h3 class="subheading">BELIEFS / 有证据的假设</h3>
    ${beliefsMarkup(decision.belief_state)}
  </section>
  <section class="state-section">
    <h3 class="subheading">NEW MEMORY / 本回合新事件</h3>
    ${eventMarkup(decision.new_events)}
  </section>
`;

const renderTrace = (decision) => {
  const toolSequence = decision.tool.call_ids
    .map((callId, index) => {
      const name =
        decision.tool.names[index] ||
        (index === decision.tool.call_ids.length - 1
          ? decision.tool.name
          : "unknown_tool");
      return `<code>${escapeHtml(name)}</code> <code>${escapeHtml(callId)}</code>`;
    })
    .join(" → ");
  const trace = [
    `<div class="trace-item">模型接口执行 <strong>${decision.model_calls}</strong> 次，正常流程需要 ${decision.expected_model_calls} 次；策略尝试数为 ${decision.attempts}。</div>`,
    decision.tool.call_ids.length
      ? `<div class="trace-item">原生工具链：${toolSequence}</div>`
      : '<div class="trace-item">这条旧记录或测试记录没有原生 tool-call ID。</div>',
    `<div class="trace-item">输入为 <code>${escapeHtml(decision.policy_input.format || "unknown")}</code>，${decision.policy_input.characters.toLocaleString()} characters。</div>`,
    `<div class="trace-item">Token：输入 ${decision.usage.input_tokens.toLocaleString()} / 输出 ${decision.usage.output_tokens.toLocaleString()} / 合计 ${decision.usage.total_tokens.toLocaleString()}。</div>`,
  ];
  if (decision.tactical_analysis?.schema) {
    trace.push(
      `<div class="trace-item">战术计算：速度 <strong>${escapeHtml(titleCase(decision.tactical_analysis.speed_relation))}</strong>；最佳伤害 ${escapeHtml((decision.tactical_analysis.best_damage_action_ids || []).join(", ") || "—")}；最佳换人 ${escapeHtml((decision.tactical_analysis.best_switch_action_ids || []).join(", ") || "—")}。</div>`
    );
  }
  if (decision.planner.model_calls) {
    trace.push(
      `<div class="trace-item">Planner 因 <code>${escapeHtml(decision.plan_trigger || "unknown")}</code> 执行 ${decision.planner.model_calls} 次，耗时 ${decision.planner.elapsed_seconds.toFixed(2)}s，使用 ${decision.planner.usage.total_tokens.toLocaleString()} tokens。</div>`
    );
  }
  if (Object.keys(decision.plan_maintenance || {}).length) {
    trace.push(
      `<div class="trace-item">计划控制器在本地移除：保护对象 ${escapeHtml((decision.plan_maintenance.removed_preserve || []).join(", ") || "—")}；已完成目标 ${escapeHtml((decision.plan_maintenance.removed_priority_targets || []).join(", ") || "—")}。${decision.plan_maintenance.requires_replan ? "该变化仍触发战略重规划。" : "本回合无需调用 Planner。"}</div>`
    );
  }
  for (const error of decision.planner.errors) {
    trace.push(
      `<div class="trace-item is-error">Planner: ${escapeHtml(error)}</div>`
    );
  }
  for (const error of decision.enrichment_errors) {
    trace.push(
      `<div class="trace-item is-error">Enrichment: ${escapeHtml(error)}</div>`
    );
  }
  if (decision.fallback_used) {
    trace.push(
      '<div class="trace-item is-error">模型没有及时给出可执行动作，本回合使用了安全备用策略。</div>'
    );
  }
  for (const error of decision.errors) {
    trace.push(
      `<div class="trace-item is-error">${escapeHtml(error)}</div>`
    );
  }
  if (decision.policy_input.hash) {
    trace.push(
      `<div class="trace-item">输入指纹：<code>${escapeHtml(decision.policy_input.hash)}</code></div>`
    );
  }
  if (Number.isInteger(decision.replay_step)) {
    trace.push(
      `<div class="trace-item">回放锚点：protocol step <code>${decision.replay_step}</code></div>`
    );
  }
  return `<div class="trace-list">${trace.join("")}</div>`;
};

const panelRenderers = {
  overview: renderOverview,
  actions: renderActions,
  agent: renderAgent,
  state: renderState,
  trace: renderTrace,
};

const renderTimeline = () => {
  elements.timeline.innerHTML = viewerData.decisions
    .map(
      (decision, index) => `
        <button
          class="timeline-button ${decision.fallback_used ? "is-fallback" : ""}"
          data-index="${index}"
          title="Turn ${decision.turn} · ${escapeHtml(decision.action_id)} · Step ${decision.replay_step ?? "—"}"
        >
          T${String(decision.turn).padStart(2, "0")}
        </button>
      `
    )
    .join("");
  elements.timeline.addEventListener("click", (event) => {
    const button = event.target.closest("[data-index]");
    if (!button) return;
    selectDecision(Number(button.dataset.index));
  });
};

const render = () => {
  const decision = currentDecision();
  elements.turnNumber.textContent = String(decision.turn).padStart(2, "0");
  elements.decisionPosition.textContent = `${String(state.index + 1).padStart(2, "0")} / ${String(viewerData.decisions.length).padStart(2, "0")}`;
  renderMatchup(decision);
  elements.panel.innerHTML = panelRenderers[state.tab](decision);
  elements.previous.disabled = state.index === 0;
  elements.next.disabled = state.index === viewerData.decisions.length - 1;
  elements.tabs.forEach((tab) =>
    tab.classList.toggle("is-active", tab.dataset.tab === state.tab)
  );
  const timelineButtons = [
    ...elements.timeline.querySelectorAll(".timeline-button"),
  ];
  timelineButtons.forEach((button, index) =>
    button.classList.toggle("is-active", index === state.index)
  );
  timelineButtons[state.index]?.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
    inline: "center",
  });
};

const selectDecision = (index, source = "manual") => {
  if (source === "manual") setSyncEnabled(false);
  state.index = Math.max(
    0,
    Math.min(viewerData.decisions.length - 1, index)
  );
  render();
};

elements.previous.addEventListener("click", () =>
  selectDecision(state.index - 1)
);
elements.next.addEventListener("click", () =>
  selectDecision(state.index + 1)
);
elements.syncToggle.addEventListener("click", () =>
  setSyncEnabled(!state.syncEnabled)
);
elements.tabs.forEach((tab) =>
  tab.addEventListener("click", () => {
    state.tab = tab.dataset.tab;
    render();
  })
);
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") selectDecision(state.index - 1);
  if (event.key === "ArrowRight") selectDecision(state.index + 1);
});

elements.battleId.textContent = viewerData.battle_id;
elements.decisionCount.textContent = `${viewerData.decisions.length} DECISIONS`;
elements.replay.addEventListener("load", () => {
  state.bridgeStartedAt = Date.now();
  state.replayUnavailable = false;
  syncFromReplay();
});
elements.replay.srcdoc = replayHtml;
document.title = `${viewerData.battle_id} · ShowdownMind Replay Lab`;
renderTimeline();
render();
const syncTimer = window.setInterval(syncFromReplay, 100);
window.addEventListener("beforeunload", () => window.clearInterval(syncTimer));
syncFromReplay();
