/**
 * Fraud Guard — dynamic inputs from GET /feature-config, POST /predict with { values: {...} }
 */

const API_BASE = "http://127.0.0.1:8000";

let featureConfig = null;

const els = {
  apiStatus: document.getElementById("apiStatus"),
  manualForm: document.getElementById("manualForm"),
  manualCard: document.getElementById("manualCard"),
  basicFields: document.getElementById("basicFields"),
  advancedFields: document.getElementById("advancedFields"),
  featureConfigHint: document.getElementById("featureConfigHint"),
  predictLoading: document.getElementById("predictLoading"),
  btnCheck: document.getElementById("btnCheck"),
  csvFile: document.getElementById("csvFile"),
  fileName: document.getElementById("fileName"),
  btnUpload: document.getElementById("btnUpload"),
  outputEmpty: document.getElementById("outputEmpty"),
  outputPanel: document.getElementById("outputPanel"),
  manualResult: document.getElementById("manualResult"),
  resultBadge: document.getElementById("resultBadge"),
  riskScore: document.getElementById("riskScore"),
  riskBarFill: document.getElementById("riskBarFill"),
  riskLevel: document.getElementById("riskLevel"),
  riskFactors: document.getElementById("riskFactors"),
  suggestedActions: document.getElementById("suggestedActions"),
  actionsSection: document.getElementById("actionsSection"),
  systemInsight: document.getElementById("systemInsight"),
  explanation: document.getElementById("explanation"),
  fileWarnings: document.getElementById("fileWarnings"),
  tableWrap: document.getElementById("tableWrap"),
  tableBody: document.getElementById("tableBody"),
  errorBanner: document.getElementById("errorBanner"),
};

function setLoading(button, loading) {
  const text = button.querySelector(".btn__text");
  const spin = button.querySelector(".btn__spinner");
  button.disabled = loading;
  if (spin) spin.hidden = !loading;
  if (text) text.style.opacity = loading ? "0.7" : "1";
}

function setPredictOverlay(visible) {
  if (!els.predictLoading) return;
  els.predictLoading.hidden = !visible;
  els.manualCard.classList.toggle("card--loading", visible);
}

function hideError() {
  els.errorBanner.hidden = true;
  els.errorBanner.textContent = "";
}

function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.hidden = false;
}

async function parseErrorResponse(res) {
  const ct = res.headers.get("content-type") || "";
  try {
    if (ct.includes("application/json")) {
      const data = await res.json();
      if (typeof data.error === "string") return data.error;
      if (typeof data.detail === "string") return data.detail;
      if (Array.isArray(data.detail)) {
        return data.detail.map((d) => d.msg || d).join("; ");
      }
      return JSON.stringify(data);
    }
    const text = await res.text();
    return text || res.statusText;
  } catch {
    return res.statusText || "Request failed";
  }
}

function showOutputShell() {
  els.outputEmpty.hidden = true;
  els.outputPanel.hidden = false;
}

function resetRiskBar() {
  if (!els.riskBarFill) return;
  els.riskBarFill.style.transition = "none";
  els.riskBarFill.style.width = "0%";
}

function animateRiskBar(pct) {
  if (!els.riskBarFill) return;
  const p = Math.min(100, Math.max(0, Number(pct) || 0));
  els.riskBarFill.style.transition = "none";
  els.riskBarFill.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      els.riskBarFill.style.transition = "width 0.9s cubic-bezier(0.22, 1, 0.36, 1)";
      els.riskBarFill.style.width = `${p}%`;
    });
  });
}

function clearManualVisuals() {
  els.manualResult.hidden = true;
  els.resultBadge.textContent = "—";
  els.resultBadge.className = "result-block__badge";
  els.riskScore.textContent = "—";
  els.riskLevel.textContent = "—";
  els.riskLevel.className = "metric__value metric__value--level";
  els.riskFactors.innerHTML = "";
  els.suggestedActions.innerHTML = "";
  els.actionsSection.classList.remove("actions-section--high");
  els.systemInsight.textContent = "";
  els.explanation.textContent = "";
  resetRiskBar();
}

function clearTable() {
  els.tableWrap.hidden = true;
  els.tableBody.innerHTML = "";
  els.fileWarnings.hidden = true;
  els.fileWarnings.innerHTML = "";
}

function levelClass(level) {
  const l = (level || "").toLowerCase();
  if (l === "high") return "metric__value--level--high";
  if (l === "medium") return "metric__value--level--medium";
  return "metric__value--level--low";
}

function tableLevelClass(level) {
  const l = (level || "").toLowerCase();
  if (l === "high") return "cell--level-high";
  if (l === "medium") return "cell--level-medium";
  return "cell--level-low";
}

function renderManualResult(data) {
  const predLabel = data.prediction || data.result || "—";
  const isFraud = predLabel === "Fraud";
  els.manualResult.hidden = false;
  els.resultBadge.textContent = predLabel;
  els.resultBadge.className =
    "result-block__badge " +
    (isFraud ? "result-block__badge--fraud" : "result-block__badge--safe");
  els.riskScore.textContent = `${data.risk_score}%`;
  animateRiskBar(data.risk_score);

  const lvl = data.risk_level || "—";
  els.riskLevel.textContent = lvl;
  els.riskLevel.className = "metric__value metric__value--level " + levelClass(lvl);

  const factors = Array.isArray(data.factors) ? data.factors : [];
  els.riskFactors.innerHTML = factors.map((f) => `<li>${escapeHtml(f)}</li>`).join("");

  const actions = Array.isArray(data.suggested_actions) ? data.suggested_actions : [];
  els.suggestedActions.innerHTML = actions.map((a) => `<li>${escapeHtml(a)}</li>`).join("");
  els.actionsSection.classList.toggle("actions-section--high", (data.risk_level || "") === "High");

  els.systemInsight.textContent = data.insight || "";
  els.explanation.textContent = data.explanation || "";
}

function renderFileWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    els.fileWarnings.hidden = true;
    els.fileWarnings.innerHTML = "";
    return;
  }
  els.fileWarnings.hidden = false;
  const list = warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
  els.fileWarnings.innerHTML = `<strong>Note:</strong><ul>${list}</ul>`;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function escapeAttr(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

function keyFactorSummary(p) {
  const factors = Array.isArray(p.factors) ? p.factors : [];
  let text = factors.length ? factors[0] : p.insight || "—";
  if (text.length > 100) text = text.slice(0, 97) + "…";
  return text;
}

function renderTable(predictions) {
  if (!predictions || predictions.length === 0) {
    els.tableWrap.hidden = true;
    els.tableBody.innerHTML = "";
    return;
  }
  els.tableWrap.hidden = false;
  els.tableBody.innerHTML = predictions
    .map((p) => {
      const pred = p.prediction || p.result || "—";
      const fraud = pred === "Fraud";
      const cls = fraud ? "cell--fraud" : "cell--safe";
      const level = p.risk_level || "—";
      const kf = keyFactorSummary(p);
      const lvlCls = tableLevelClass(level);
      const titleHint = p.insight ? escapeAttr(p.insight) : "";
      return `<tr>
        <td>${p.row}</td>
        <td class="${cls}">${escapeHtml(pred)}</td>
        <td>${escapeHtml(String(p.risk_score))}%</td>
        <td class="${lvlCls}">${escapeHtml(level)}</td>
        <td class="cell--insight" title="${titleHint}">${escapeHtml(kf)}</td>
      </tr>`;
    })
    .join("");
}

function rangeFor(name, cfg) {
  const r = (cfg.ranges && cfg.ranges[name]) || {};
  return {
    min: r.min ?? (name === "Time" ? 0 : name === "Amount" ? 0 : -10),
    max: r.max ?? (name === "Time" ? 200000 : name === "Amount" ? 25000 : 10),
    def: r.default ?? (name === "Time" ? 94813 : name === "Amount" ? 88 : 0),
  };
}

function labelFor(name, cfg) {
  if (name === "Time") return "Transaction time (seconds)";
  if (name === "Amount") return "Transaction amount";
  return (cfg.behavior_labels && cfg.behavior_labels[name]) || "Behavior indicator";
}

function buildBasicField(name, cfg) {
  const { min, max, def } = rangeFor(name, cfg);
  const step = name === "Time" ? "1" : "0.01";
  return `
    <label class="field">
      <span class="field__label">${escapeHtml(labelFor(name, cfg))}</span>
      <input type="number" class="field__input" data-feature="${escapeHtml(name)}"
        min="${min}" max="${max}" step="${step}" value="${def}" />
    </label>`;
}

function buildBehaviorField(name, cfg) {
  const { min, max, def } = rangeFor(name, cfg);
  const lab = labelFor(name, cfg);
  const tip = cfg.signal_help || "";
  const step = 0.05;
  return `
    <div class="behavior-field">
      <div class="behavior-field__head">
        <span class="field__label field__label--tip" title="${escapeAttr(tip)}">${escapeHtml(lab)}</span>
        <span class="behavior-field__value" data-readout-for="${escapeHtml(name)}">${def}</span>
      </div>
      <input type="range" class="behavior-field__range" data-feature="${escapeHtml(name)}"
        min="${min}" max="${max}" step="${step}" value="${def}" />
      <input type="number" class="field__input behavior-field__num" data-feature="${escapeHtml(name)}"
        min="${min}" max="${max}" step="0.01" value="${def}" />
    </div>`;
}

function wireBehaviorSync(container) {
  container.querySelectorAll(".behavior-field").forEach((wrap) => {
    const name = wrap.querySelector(".behavior-field__range")?.getAttribute("data-feature");
    if (!name) return;
    const range = wrap.querySelector(`[data-feature="${name}"].behavior-field__range`);
    const num = wrap.querySelector(`[data-feature="${name}"].behavior-field__num`);
    const readout = wrap.querySelector(`[data-readout-for="${name}"]`);
    const sync = (src) => {
      let v = parseFloat(src.value);
      if (Number.isNaN(v)) v = parseFloat(range.min);
      v = Math.min(parseFloat(range.max), Math.max(parseFloat(range.min), v));
      range.value = String(v);
      num.value = String(v);
      if (readout) readout.textContent = Number.isInteger(v) ? String(v) : v.toFixed(2);
    };
    range.addEventListener("input", () => sync(range));
    num.addEventListener("input", () => sync(num));
  });
}

function collectValues(cfg) {
  const out = {};
  const names = cfg.selected_features || [];
  names.forEach((name) => {
    const el = els.manualForm.querySelector(`[data-feature="${name}"]`);
    if (!el) return;
    const v = parseFloat(el.value);
    if (!Number.isNaN(v)) out[name] = v;
  });
  return out;
}

async function loadFeatureConfig() {
  els.featureConfigHint.textContent = "Loading input configuration from the API…";
  els.btnCheck.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/feature-config`, { method: "GET" });
    if (!res.ok) {
      throw new Error(await parseErrorResponse(res));
    }
    featureConfig = await res.json();
    const sel = featureConfig.selected_features || [];
    els.basicFields.innerHTML = "";
    els.advancedFields.innerHTML = "";

    sel.forEach((name) => {
      const t = featureConfig.feature_types && featureConfig.feature_types[name];
      if (t === "behavior") {
        els.advancedFields.innerHTML += buildBehaviorField(name, featureConfig);
      } else {
        els.basicFields.innerHTML += buildBasicField(name, featureConfig);
      }
    });

    wireBehaviorSync(els.advancedFields);
    els.featureConfigHint.classList.remove("card__hint--error");
    els.featureConfigHint.textContent =
      "Fields are chosen from your dataset and model (time, amount, and top behavioral signals). Defaults use typical values — adjust optional sliders if needed.";
    els.btnCheck.disabled = false;
  } catch (e) {
    els.featureConfigHint.textContent =
      "Could not load feature config. Is the API running? (" + String(e.message || e) + ")";
    els.featureConfigHint.classList.add("card__hint--error");
  }
}

async function checkHealth() {
  const statusText = els.apiStatus.querySelector(".header__status-text");
  const dot = els.apiStatus.querySelector(".dot");
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    const data = await res.json().catch(() => ({}));
    if (data.ok) {
      dot.className = "dot";
      const ml = data.model_loaded === false ? " (no model)" : "";
      statusText.textContent = "API online" + ml;
    } else {
      dot.className = "dot dot--bad";
      statusText.textContent = data.detail ? "API up — model issue" : "API unreachable";
    }
  } catch {
    dot.className = "dot dot--bad";
    statusText.textContent = "Cannot reach API";
  }
}

els.manualForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!featureConfig) {
    showError("Feature configuration not loaded.");
    return;
  }
  hideError();
  clearTable();
  showOutputShell();

  const payload = { values: collectValues(featureConfig) };

  setLoading(els.btnCheck, true);
  setPredictOverlay(true);
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      showError(await parseErrorResponse(res));
      clearManualVisuals();
      return;
    }
    const data = await res.json();
    resetRiskBar();
    renderManualResult(data);
  } catch (err) {
    showError(
      err instanceof TypeError && err.message.includes("fetch")
        ? "Network error — is the backend running on port 8000?"
        : String(err.message || err)
    );
    clearManualVisuals();
  } finally {
    setLoading(els.btnCheck, false);
    setPredictOverlay(false);
  }
});

els.csvFile.addEventListener("change", () => {
  const f = els.csvFile.files[0];
  els.fileName.textContent = f ? f.name : "No file selected";
});

els.btnUpload.addEventListener("click", async () => {
  hideError();
  const file = els.csvFile.files[0];
  if (!file) {
    showError("Please choose a CSV file first.");
    return;
  }

  showOutputShell();
  clearManualVisuals();

  const formData = new FormData();
  formData.append("file", file, file.name);

  setLoading(els.btnUpload, true);
  try {
    const res = await fetch(`${API_BASE}/predict-file`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      showError(await parseErrorResponse(res));
      clearTable();
      return;
    }
    const data = await res.json();
    renderFileWarnings(data.warnings);
    renderTable(data.predictions);
    if (!data.predictions || data.predictions.length === 0) {
      showError("No predictions returned.");
    }
  } catch (err) {
    showError(
      err instanceof TypeError && err.message.includes("fetch")
        ? "Network error — is the backend running on port 8000?"
        : String(err.message || err)
    );
    clearTable();
  } finally {
    setLoading(els.btnUpload, false);
  }
});

(async function init() {
  await loadFeatureConfig();
  checkHealth();
})();
