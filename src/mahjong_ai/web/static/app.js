function mahjongDevTable() {
  return {
    rules: [],
    selectedRuleId: "",
    ruleMeta: "",
    seed: 1,
    controllers: {
      0: { kind: "human", provider: "", base_url: "", token: "", model_name: "" },
      1: { kind: "human", provider: "", base_url: "", token: "", model_name: "" },
      2: { kind: "human", provider: "", base_url: "", token: "", model_name: "" },
      3: { kind: "human", provider: "", base_url: "", token: "", model_name: "" },
    },
    sessionId: null,
    snapshot: null,
    stepResult: null,
    activeJsonTab: "step",
    isAutoRunning: false,
    autoRunStatus: "",
    autoRunDelayMs: 120,
    autoRunStepLimit: 200,
    autoRunSteps: 0,
    isAutoPassing: false,
    controllerReasons: {},
    providerPresets: {
      "apple-fm": { base_url: "", token: "", model_name: "system" },
      openai: { base_url: "https://api.openai.com/v1", model_name: "" },
      gemini: { base_url: "https://generativelanguage.googleapis.com/v1beta/openai", model_name: "gemini-2.5-flash" },
      deepseek: { base_url: "https://api.deepseek.com", model_name: "deepseek-v4-flash" },
      "deepseek-v4-pro": { base_url: "https://api.deepseek.com", model_name: "deepseek-v4-pro" },
      openrouter: { base_url: "https://openrouter.ai/api/v1", model_name: "qwen/qwen3.5-flash-02-23" },
      mistral: { base_url: "https://api.mistral.ai/v1", model_name: "mistral-large-latest" },
      groq: { base_url: "https://api.groq.com/openai/v1", model_name: "openai/gpt-oss-120b" },
      together: { base_url: "https://api.together.xyz/v1", model_name: "" },
      xai: { base_url: "https://api.x.ai/v1", model_name: "grok-4-fast-non-reasoning" },
      "local-openai": { base_url: "http://127.0.0.1:8001/v1", token: "local", model_name: "qwen3.5-2b-transformers" },
      debug: { base_url: "", token: "", model_name: "debug-model" },
    },

    async init() {
      await this.loadRules();
    },

    async api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`${response.status} ${response.statusText}: ${text}`);
      }
      return response.json();
    },

    async loadRules() {
      this.rules = await this.api("/api/rules");
      this.selectedRuleId =
        this.rules.find((rule) => rule.rule_id === "northern_tuidaohe.v1")?.rule_id ||
        this.rules[0]?.rule_id ||
        "";
      this.renderRuleMeta();
    },

    renderRuleMeta() {
      const rule = this.rules.find((item) => item.rule_id === this.selectedRuleId);
      this.ruleMeta = rule
        ? `status=${rule.status} · implementation=${rule.implementation_status || "-"} · hash=${rule.config_hash}`
        : "";
    },

    controllerKindChanged(seat) {
      const controller = this.controllers[seat];
      if (controller.kind === "human") {
        controller.provider = "";
        controller.base_url = "";
        controller.token = "";
        controller.model_name = "";
        return;
      }
      if (!controller.provider) controller.provider = "apple-fm";
      this.controllerProviderChanged(seat);
    },

    controllerProviderChanged(seat) {
      const controller = this.controllers[seat];
      const preset = this.providerPresets[controller.provider] || this.providerPresets.openai;
      const priorModel = controller.model_name;
      const modelWasPreset = Object.values(this.providerPresets).some((item) => item.model_name === priorModel);
      controller.base_url = preset.base_url ?? controller.base_url ?? "";
      if (Object.prototype.hasOwnProperty.call(preset, "token")) {
        controller.token = preset.token;
      }
      if (!controller.model_name || modelWasPreset) {
        controller.model_name = preset.model_name || "";
      }
    },

    controllerNeedsHttpConfig(seat) {
      const provider = this.controllers[seat]?.provider;
      return !["apple-fm", "debug", "", null, undefined].includes(provider);
    },

    setAllControllers(kind) {
      for (const seat of [0, 1, 2, 3]) {
        this.controllers[seat].kind = kind;
        this.controllerKindChanged(seat);
      }
    },

    seatControllerPayload() {
      const payload = {};
      for (const seat of [0, 1, 2, 3]) {
        const controller = this.controllers[seat];
        const preset = this.providerPresets[controller.provider] || {};
        const modelName =
          controller.model_name ||
          preset.model_name ||
          (controller.kind === "llm" && controller.provider === "apple-fm" ? "system" : null);
        payload[seat] = {
          kind: controller.kind === "human" ? "human" : "model",
          provider: controller.provider || (controller.kind === "llm" ? "apple-fm" : controller.kind),
          base_url: controller.base_url || null,
          token: controller.token || null,
          model_name: modelName,
          model_id: modelName,
        };
      }
      return payload;
    },

    async createSession() {
      const session = await this.api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          rule_id: this.selectedRuleId,
          seed: Number(this.seed || 1),
          seat_controllers: this.seatControllerPayload(),
        }),
      });
      this.sessionId = session.session_id;
      this.stepResult = null;
      this.autoRunStatus = "";
      this.controllerReasons = {};
      await this.refreshSnapshot();
    },

    async refreshSnapshot() {
      if (!this.sessionId) return;
      this.snapshot = await this.api(`/api/sessions/${this.sessionId}/snapshot`);
      await this.autoSubmitForcedPasses();
    },

    async advance() {
      if (!this.sessionId) return;
      await this.api(`/api/sessions/${this.sessionId}/advance`, { method: "POST" });
      await this.refreshSnapshot();
    },

    async submitStep(action, options = {}) {
      if (!this.sessionId) return;
      const resumeAutoRun = options.resumeAutoRun ?? true;
      const autoPass = options.autoPass ?? true;
      const payload = {
        actor: action.actor,
        operation: action.kind || action.operation,
        tile: action.tile,
        source: action.source,
        metadata: action.metadata || {},
      };
      this.stepResult = await this.api(`/api/sessions/${this.sessionId}/step`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      this.applyStepResult();
      if (autoPass) {
        await this.autoSubmitForcedPasses();
      }
      if (resumeAutoRun && this.isAutoRunning) {
        await this.runAutoRunLoop();
      }
    },

    async controllerStep(actor, options = {}) {
      if (!this.sessionId) return;
      const autoPass = options.autoPass ?? true;
      this.stepResult = await this.api(`/api/sessions/${this.sessionId}/controller-step`, {
        method: "POST",
        body: JSON.stringify({ actor }),
      });
      this.applyStepResult();
      if (autoPass) {
        await this.autoSubmitForcedPasses();
      }
    },

    async autoRunControllers() {
      if (!this.sessionId || this.isAutoRunning) return;
      this.isAutoRunning = true;
      this.autoRunSteps = 0;
      this.autoRunStatus = "Auto-run started.";
      await this.runAutoRunLoop();
    },

    async runAutoRunLoop() {
      if (!this.sessionId || !this.isAutoRunning) return;
      try {
        while (this.isAutoRunning && this.autoRunSteps < this.autoRunStepLimit) {
          if (!this.snapshot) await this.refreshSnapshot();
          if (this.session?.is_terminal) {
            this.autoRunStatus = `Auto-run stopped: terminal after ${this.autoRunSteps} automatic steps.`;
            this.isAutoRunning = false;
            break;
          }
          const passActor = this.nextForcedPassActor();
          if (passActor !== null) {
            const pass = this.onlyPassAction(passActor);
            this.autoRunStatus = `Auto-run step ${this.autoRunSteps + 1}: auto-pass ${this.seatName(passActor)}`;
            await this.submitStep(pass, { resumeAutoRun: false, autoPass: false });
            this.autoRunSteps += 1;
            await this.autoRunDelay();
            continue;
          }
          const actor = this.nextRunnableControllerActor();
          if (actor !== null) {
            this.autoRunStatus = `Auto-run step ${this.autoRunSteps + 1}: ${this.seatName(actor)} (${this.sessionController(actor)?.provider || "controller"})`;
            await this.controllerStep(actor, { autoPass: true });
            this.autoRunSteps += 1;
            await this.autoRunDelay();
            continue;
          }
          const humanActor = this.nextHumanDecisionActor();
          if (humanActor !== null) {
            this.autoRunStatus = `Auto-run waiting for ${this.seatName(humanActor)} human action after ${this.autoRunSteps} automatic steps.`;
            break;
          }
          this.autoRunStatus = `Auto-run waiting for a legal decision after ${this.autoRunSteps} automatic steps.`;
          break;
        }
        if (this.autoRunSteps >= this.autoRunStepLimit) {
          this.autoRunStatus = `Auto-run stopped: reached ${this.autoRunStepLimit} automatic steps.`;
          this.isAutoRunning = false;
        }
      } catch (error) {
        this.autoRunStatus = `Auto-run error: ${error.message || error}`;
        this.isAutoRunning = false;
      }
    },

    async autoRunDelay() {
      if (this.autoRunDelayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, this.autoRunDelayMs));
      }
    },

    async autoSubmitForcedPasses() {
      if (!this.sessionId || this.isAutoPassing) return;
      this.isAutoPassing = true;
      try {
        let forcedCount = 0;
        while (forcedCount < 16 && !this.session?.is_terminal) {
          const actor = this.nextForcedPassActor();
          if (actor === null) break;
          const pass = this.onlyPassAction(actor);
          if (!pass) break;
          await this.submitStep(pass, {
            resumeAutoRun: false,
            autoPass: false,
          });
          forcedCount += 1;
        }
      } finally {
        this.isAutoPassing = false;
      }
    },

    stopAutoRun() {
      this.isAutoRunning = false;
      this.autoRunStatus = "Auto-run stopping after current request.";
    },

    applyStepResult() {
      this.rememberControllerDecision(this.stepResult?.controller_decision);
      this.snapshot = {
        session: this.stepResult.session,
        state: this.stepResult.state,
        decision: {
          decision_actors: this.stepResult.legal.decision_actors,
          response_window: this.stepResult.legal.response_window,
          legal_actions: this.stepResult.legal.actions,
          pending_responses: this.stepResult.pending.responses,
        },
      };
    },

    rememberControllerDecision(decision) {
      if (!decision) return;
      const reason = decision.natural_language_reason || "";
      this.controllerReasons[decision.seat] = reason;
    },

    async resolveNoResponse() {
      for (const actor of this.decisionActors()) {
        if (this.snapshot?.decision?.pending_responses?.[actor]) continue;
        const pass = this.legalActions(actor).find((action) => action.kind === "pass");
        if (pass) await this.submitStep(pass);
      }
    },

    async downloadFullLog() {
      if (!this.sessionId) return;
      const fullLog = await this.api(`/api/sessions/${this.sessionId}/full-log`);
      const blob = new Blob([JSON.stringify(fullLog, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${this.sessionId}-full-log.json`;
      link.click();
      URL.revokeObjectURL(url);
    },

    get session() {
      return this.snapshot?.session || null;
    },

    get state() {
      return this.snapshot?.state || null;
    },

    statusLine() {
      if (!this.session) return "No session";
      return `${this.session.session_id} · ${this.session.rule_id} · events=${this.session.event_count}`;
    },

    seatName(seat) {
      return { 0: "East", 1: "North", 2: "West", 3: "South" }[seat] || `Seat ${seat}`;
    },

    player(seat) {
      return this.state?.players?.[seat] || null;
    },

    handTiles(seat) {
      return this.sortedTiles(this.player(seat)?.concealed_tiles || []);
    },

    hiddenSlots(seat) {
      return [];
    },

    tileAssetUrl(tile) {
      return `/static/vendor/mahjong-tiles-svg/${tile}.svg`;
    },

    discardStack() {
      return this.state?.discard_stack || [];
    },

    decisionActors() {
      return this.snapshot?.decision?.decision_actors || [];
    },

    legalActions(actor) {
      return this.snapshot?.decision?.legal_actions?.[actor] || [];
    },

    pendingResponses() {
      return this.snapshot?.decision?.pending_responses || {};
    },

    pendingResponse(actor) {
      return this.pendingResponses()[actor] || null;
    },

    pendingActors() {
      return Object.keys(this.pendingResponses()).map((actor) => Number(actor));
    },

    sessionController(actor) {
      return this.session?.controllers?.[actor] || null;
    },

    configuredController(actor) {
      return this.controllers[actor] || null;
    },

    isControllerActor(actor) {
      const sessionController = this.sessionController(actor);
      if (sessionController) return sessionController.kind !== "human";
      return this.configuredController(actor)?.kind !== "human";
    },

    visibleActions(actor) {
      if (this.pendingResponse(actor)) return [];
      return this.sortedActions(this.legalActions(actor).filter((action) => action.kind !== "pass"));
    },

    onlyPassAction(actor) {
      if (this.pendingResponse(actor)) return null;
      const actions = this.legalActions(actor);
      return actions.length === 1 && actions[0].kind === "pass" ? actions[0] : null;
    },

    isOnlyPassSeat(actor) {
      return this.onlyPassAction(actor) !== null;
    },

    isControllerDecisionSeat(actor) {
      return (
        this.decisionActors().includes(actor) &&
        this.isControllerActor(actor) &&
        !this.pendingResponse(actor) &&
        !this.isOnlyPassSeat(actor)
      );
    },

    isHumanDecisionSeat(actor) {
      return (
        this.decisionActors().includes(actor) &&
        !this.isControllerActor(actor) &&
        !this.pendingResponse(actor) &&
        !this.isOnlyPassSeat(actor)
      );
    },

    nextForcedPassActor() {
      for (const actor of this.decisionActors()) {
        if (this.isOnlyPassSeat(actor)) return actor;
      }
      return null;
    },

    nextHumanDecisionActor() {
      for (const actor of this.decisionActors()) {
        if (this.isHumanDecisionSeat(actor)) return actor;
      }
      return null;
    },

    nextRunnableControllerActor() {
      for (const actor of this.decisionActors()) {
        if (this.isControllerDecisionSeat(actor)) return actor;
      }
      return null;
    },

    canAutoRun() {
      return Boolean(
        this.sessionId &&
        !this.isAutoRunning &&
        !this.session?.is_terminal &&
        this.decisionActors().length > 0
      );
    },

    controllerProviderLabel(actor) {
      const controller = this.sessionController(actor) || this.configuredController(actor);
      return controller?.provider || controller?.kind || "controller";
    },

    actorsWithVisibleActions() {
      return this.decisionActors().filter((actor) => this.visibleActions(actor).length > 0);
    },

    canResolveNoResponse() {
      return Boolean(this.sessionId && this.snapshot?.decision?.response_window);
    },

    canAdvanceDraw() {
      return Boolean(
        this.sessionId &&
        !this.session?.is_terminal &&
        !this.snapshot?.decision?.response_window
      );
    },

    responseWindowText() {
      const responseWindow = this.snapshot?.decision?.response_window;
      if (!responseWindow) return "No response window.";
      return `Response window #${responseWindow.window_id}: ${responseWindow.kind}, tile=${responseWindow.tile_type}, source=${this.seatName(responseWindow.source)}, waiting for ${responseWindow.eligible_seats.map((seat) => this.seatName(seat)).join(", ")}. The table will not advance until every eligible seat responds.`;
    },

    actionKey(actor, action, index) {
      return [
        actor,
        index,
        action.kind,
        action.tile || "",
        action.source ?? "",
        JSON.stringify(action.metadata || {}),
      ].join("-");
    },

    actionClass(action) {
      return {
        discard: action.kind === "discard",
        chi: action.kind === "chi",
        peng: action.kind === "peng",
        kong: ["exposed_kong", "concealed_kong", "added_kong"].includes(action.kind),
        win: action.kind === "win",
        declare: action.kind === "declare",
      };
    },

    actionLabel(action) {
      const kind = action.kind || action.operation;
      const source = action.source !== null && action.source !== undefined
        ? ` ← ${this.seatName(action.source)}`
        : "";
      if (kind === "chi" && action.metadata?.sequence) {
        return `chi ${action.metadata.sequence.join("-")}${source}`;
      }
      if (kind === "peng") return `peng ${action.tile}${source}`;
      if (kind === "exposed_kong") return `kong ${action.tile}${source}`;
      if (kind === "concealed_kong") return `concealed kong ${action.tile}`;
      if (kind === "added_kong") return `added kong ${action.tile}`;
      if (kind === "win") {
        const winType = action.metadata?.win_type ? ` · ${action.metadata.win_type}` : "";
        return `win ${action.tile || ""}${source}${winType}`;
      }
      if (kind === "declare") return `declare ${action.tile || ""}`;
      if (kind === "discard") return `discard ${action.tile || ""}`;
      const parts = [kind];
      if (action.tile) parts.push(action.tile);
      if (source) parts.push(source.trim());
      return parts.join(" ");
    },

    sortedTiles(tiles) {
      return [...tiles].sort((left, right) => this.tileSortKey(left).localeCompare(this.tileSortKey(right)));
    },

    sortedActions(actions) {
      return [...actions].sort((left, right) => {
        const kindOrder = {
          win: "00",
          declare: "01",
          concealed_kong: "02",
          added_kong: "03",
          exposed_kong: "04",
          peng: "05",
          chi: "06",
          discard: "07",
          pass: "08",
        };
        const leftKey = `${kindOrder[left.kind] || "99"}-${this.tileSortKey(left.tile || "")}-${JSON.stringify(left.metadata || {})}`;
        const rightKey = `${kindOrder[right.kind] || "99"}-${this.tileSortKey(right.tile || "")}-${JSON.stringify(right.metadata || {})}`;
        return leftKey.localeCompare(rightKey);
      });
    },

    tileSortKey(tile) {
      if (!tile) return "99";
      const suitOrder = { W: "0", B: "1", T: "2", F: "3", J: "4" };
      return `${suitOrder[tile[0]] ?? "9"}${tile.slice(1).padStart(2, "0")}`;
    },

    controllerDecision() {
      return this.stepResult?.controller_decision || null;
    },

    controllerReason() {
      return this.controllerDecision()?.natural_language_reason || "";
    },

    controllerReasonForSeat(seat) {
      return this.controllerReasons[seat] || "";
    },

    controllerActionText() {
      const decision = this.controllerDecision();
      if (!decision) return "-";
      return `${this.seatName(decision.seat)} ${this.actionLabel(decision.selected_action)}`;
    },

    controllerSummaryLabel(seat) {
      const controller = this.sessionController(seat) || this.configuredController(seat);
      if (!controller || controller.kind === "human") return "Controller: Human";
      return `Controller: ${controller.provider || controller.kind}${controller.model_name ? ` / ${controller.model_name}` : ""}`;
    },

    shownEvents() {
      const events = this.stepResult?.events || this.snapshot?.state?.events || [];
      return events.slice(-80);
    },

    tabLabel(tab) {
      return {
        step: "Step Result",
        state: "State",
        full: "Full State",
        legal: "Legal",
        controller: "Controller",
      }[tab];
    },

    jsonData() {
      if (!this.snapshot) return {};
      if (this.activeJsonTab === "step") return this.stepResult || this.snapshot;
      if (this.activeJsonTab === "state") return this.snapshot.state;
      if (this.activeJsonTab === "full") return this.stepResult?.full_state || {};
      if (this.activeJsonTab === "legal") return this.snapshot.decision;
      if (this.activeJsonTab === "controller") return this.controllerDecision() || {};
      return {};
    },

    jsonText() {
      return JSON.stringify(this.jsonData(), null, 2);
    },
  };
}
