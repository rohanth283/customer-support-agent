const $ = (selector) => document.querySelector(selector);
const messagesEl = $("#messages");
const inputEl = $("#message-input");
const welcomeEl = $("#welcome");
let state = {};
let busy = false;

async function post(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function setBusy(value) {
  busy = value;
  $("#send-button").disabled = value;
  inputEl.disabled = value;
}

function addMessage(text, role, options = {}) {
  welcomeEl.classList.add("hidden");
  messagesEl.classList.add("active");
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "S";
  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap";
  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = role === "user" ? "YOU" : "SUPPORT AGENT";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(label, bubble);
  if (options.state && role === "assistant") {
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const items = [];
    if (options.state.specialist) items.push(options.state.specialist.replaceAll("_", " "));
    items.push(options.state.resolved ? "resolved" : options.state.needs_approval ? "approval needed" : "unresolved");
    for (const item of items) {
      const chip = document.createElement("span");
      chip.className = "meta-chip";
      chip.textContent = item;
      meta.append(chip);
    }
    wrap.append(meta);
  }
  if (options.approval && role === "assistant") {
    const actions = document.createElement("div");
    actions.className = "approval-actions";
    const confirm = document.createElement("button");
    confirm.textContent = "Confirm cancellation";
    confirm.addEventListener("click", approveCancellation);
    const decline = document.createElement("button");
    decline.className = "decline";
    decline.textContent = "Keep booking";
    decline.addEventListener("click", declineCancellation);
    actions.append(confirm, decline);
    wrap.append(actions);
  }
  if (role === "user") row.append(wrap, avatar);
  else row.append(avatar, wrap);
  messagesEl.append(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

function renderTrace(trace = []) {
  const list = $("#trace-list");
  list.replaceChildren();
  if (!trace.length) {
    const empty = document.createElement("li");
    empty.className = "trace-empty";
    empty.textContent = "Send a message to see the graph run.";
    list.append(empty);
    return;
  }
  const displaySteps = [
    "Context & conversation state",
    "Intent & sentiment triage",
    "specialist",
    "tools",
    "approval gate",
    "Policy retrieval",
    "General support specialist",
    "Response generation & policy grounding",
    "Escalation & outcome logging",
    "Approval declined; no cancellation tool called",
  ];
  const represented = new Set();
  for (const label of trace) {
    const index = displaySteps.findIndex((candidate, i) => {
      if (represented.has(i)) return false;
      return label === candidate || (candidate === "specialist" && /specialist/.test(label)) ||
        (candidate === "tools" && /specialist & tools/.test(label)) ||
        (candidate === "approval gate" && /approval gate/.test(label));
    });
    const item = document.createElement("li");
    item.className = "done";
    const node = document.createElement("span");
    node.className = "trace-node";
    node.textContent = String(list.children.length + 1).padStart(2, "0");
    const text = document.createElement("span");
    text.className = "trace-text";
    text.textContent = label;
    item.append(node, text);
    list.append(item);
    if (index >= 0) represented.add(index);
  }
}

function renderState(next) {
  state = next || {};
  const fields = {
    "#state-intent": state.intent || "—",
    "#state-specialist": state.specialist ? state.specialist.replaceAll("_", " ") : "—",
    "#state-sentiment": state.sentiment || "—",
    "#state-urgency": state.urgency || "—",
    "#state-booking": state.booking_id || "—",
    "#state-resolved": state.resolved === true ? "yes" : state.resolved === false ? "no" : "—",
  };
  for (const [selector, value] of Object.entries(fields)) $(selector).textContent = value;
  $("#escalation-banner").classList.toggle("hidden", !state.escalated);
  renderTrace(state.trace || []);
}

async function submitMessage(text) {
  const message = text.trim();
  if (!message || busy) return;
  addMessage(message, "user");
  inputEl.value = "";
  resizeInput();
  setBusy(true);
  const thinkingRow = addMessage("Checking the request and routing it to a specialist…", "assistant");
  thinkingRow.classList.add("thinking");
  try {
    const result = await post("/api/chat", { state, message });
    state = result.state;
    thinkingRow.remove();
    renderState(state);
    addMessage(state.response || "No response was returned.", "assistant", {
      state,
      approval: Boolean(state.needs_approval),
    });
  } catch (error) {
    thinkingRow.remove();
    addMessage(`I couldn't run that request: ${error.message}`, "assistant");
  } finally {
    setBusy(false);
    inputEl.focus();
  }
}

async function approveCancellation() {
  if (busy || !state.needs_approval) return;
  addMessage("Confirm cancellation", "user");
  setBusy(true);
  try {
    const result = await post("/api/approve", { state });
    renderState(result.state);
    addMessage(result.state.response, "assistant", { state: result.state });
  } catch (error) {
    addMessage(`The confirmation could not be processed: ${error.message}`, "assistant");
  } finally {
    setBusy(false);
  }
}

async function declineCancellation() {
  if (busy || !state.needs_approval) return;
  addMessage("Keep the booking", "user");
  setBusy(true);
  try {
    const result = await post("/api/decline", { state });
    renderState(result.state);
    addMessage(result.state.response, "assistant", { state: result.state });
  } catch (error) {
    addMessage(`That choice could not be saved: ${error.message}`, "assistant");
  } finally {
    setBusy(false);
  }
}

function resizeInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 120)}px`;
}

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage(inputEl.value);
});
inputEl.addEventListener("input", resizeInput);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitMessage(inputEl.value);
  }
});
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => submitMessage(button.dataset.prompt));
});
$("#new-chat").addEventListener("click", async () => {
  if (busy) return;
  try { await post("/api/reset"); } catch (_) { /* the local view can still reset */ }
  state = {};
  messagesEl.replaceChildren();
  messagesEl.classList.remove("active");
  welcomeEl.classList.remove("hidden");
  renderState({});
  inputEl.focus();
});

fetch("/api/health").then((response) => response.json()).then((health) => {
  $("#model-name").textContent = health.provider || "Local model";
}).catch(() => {
  $("#model-name").textContent = "Model status unavailable";
});
