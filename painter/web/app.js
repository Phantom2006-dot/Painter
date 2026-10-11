/* Painter frontend: upload -> process -> inspect -> edit -> export.
   Everything here talks to the real API; there are no simulated results. */

const state = {
  documentId: null,
  jobId: null,
  documents: [],
  pages: [],
  elements: [],
  selected: null,
  view: "original",
  page: 0,
  zoom: 1,
  pollTimer: null,
};

const el = (id) => document.getElementById(id);

// When the server sets PAINTER_API_TOKEN, the token travels in the URL fragment
// (never sent to the server) and is kept in localStorage for later visits.
function authHeaders() {
  const fromHash = new URLSearchParams(location.hash.replace(/^#/, "")).get("token");
  if (fromHash) {
    localStorage.setItem("painterToken", fromHash);
    history.replaceState(null, "", location.pathname + location.search);
  }
  const token = fromHash || localStorage.getItem("painterToken") || "";
  return token ? { Authorization: `Bearer ${token}` } : {};
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...authHeaders(),
    },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch (error) {
      detail = await response.text();
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function setStatus(node, message, kind = "") {
  node.textContent = message || "";
  node.className = `status ${kind}`;
}

// ---------------------------------------------------------------- capabilities
async function loadCapabilities() {
  try {
    const caps = await api("/api/capabilities");
    const models = (caps.models || []).filter((m) => m.present).map((m) => m.name);
    const serpapi = caps.serpapi || {};
    el("capabilities").innerHTML = [
      `YOLO: <b class="${caps.yolo_enabled ? "on" : "off"}">${caps.yolo_enabled ? "enabled" : "disabled"}</b>`,
      `models: <b>${models.length ? models.join(", ") : "none found"}</b>`,
      `OCR: <b>${caps.tesseract_available ? "tesseract" : "unavailable"}</b>`,
      `SerpApi: <b class="${serpapi.key_configured ? "on" : "off"}">${serpapi.key_configured ? (serpapi.enabled ? "enabled" : "key set, disabled") : "no key"}</b>`,
      `verification: <b>${caps.verification_enabled ? "on" : "off"}</b>`,
    ].join(" &middot; ");
    el("uploadHint").textContent = `max ${caps.max_upload_mb} MB — ${caps.supported_formats.join(", ")}`;
  } catch (error) {
    el("capabilities").textContent = `capabilities unavailable: ${error.message}`;
  }
}

// ---------------------------------------------------------------- documents
async function refreshDocuments() {
  try {
    const payload = await api("/api/documents");
    state.documents = payload.documents || [];
  } catch (error) {
    state.documents = [];
  }
  const list = el("documentList");
  list.innerHTML = "";
  state.documents.forEach((doc) => {
    const item = document.createElement("li");
    item.className = doc.document_id === state.documentId ? "active" : "";
    item.innerHTML = `<span>${doc.filename}</span><span class="muted">${doc.status}</span>`;
    item.onclick = () => selectDocument(doc.document_id);
    list.appendChild(item);
  });
  if (!state.documents.length) {
    list.innerHTML = '<li class="muted">no documents yet</li>';
  }
}

async function selectDocument(documentId) {
  state.documentId = documentId;
  state.selected = null;
  state.page = 0;
  el("processBtn").disabled = false;
  el("exportPdf").disabled = false;
  el("exportDocx").disabled = false;
  el("exportHtml").disabled = false;
  await refreshDocuments();
  await loadDocument();
}

async function loadDocument() {
  if (!state.documentId) return;
  try {
    const payload = await api(`/api/documents/${state.documentId}`);
    const stats = payload.stats || {};
    el("stats").innerHTML = Object.entries(stats)
      .map(([key, value]) => `<span>${key}: ${typeof value === "object" ? JSON.stringify(value) : value}</span>`)
      .join("");
    const warnings = payload.warnings || [];
    el("warnings").innerHTML = warnings.length
      ? warnings.slice(0, 40).map((w) => `<div>${escapeHtml(w)}</div>`).join("")
      : '<div class="muted">no warnings</div>';
    await loadElements();
    renderView();
  } catch (error) {
    setStatus(el("uploadStatus"), error.message, "error");
  }
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadElements() {
  if (!state.documentId) return;
  const onlyReview = el("onlyReview").checked;
  const payload = await api(`/api/documents/${state.documentId}/elements${onlyReview ? "?needs_review=true" : ""}`);
  state.elements = payload.elements || [];
  state.pages = payload.pages || [];
  const list = el("elementList");
  list.innerHTML = "";
  state.elements.forEach((element) => {
    const item = document.createElement("li");
    item.className = element.id === state.selected ? "selected" : "";
    const flags = (element.flags || []).join(", ");
    item.innerHTML = `
      <div class="row1"><span class="type">${element.type}</span><span class="muted">conf ${(element.confidence || 0).toFixed(2)}</span></div>
      <div class="text">${escapeHtml((element.text || "").slice(0, 120)) || "<i>no text</i>"}</div>
      ${flags ? `<div class="flags">${escapeHtml(flags)}</div>` : ""}`;
    item.onclick = () => selectElement(element.id);
    list.appendChild(item);
  });
  if (!state.elements.length) {
    list.innerHTML = '<li class="muted">no elements (process the document first)</li>';
  }
  renderQuality();
}

function renderQuality() {
  const page = state.pages[state.page];
  if (!page) { el("qualityPanel").innerHTML = ""; return; }
  const q = page.quality || {};
  const parts = [];
  if (q.similarity != null) parts.push(`similarity <b>${Number(q.similarity).toFixed(3)}</b>`);
  if (q.ssim != null) parts.push(`ssim <b>${Number(q.ssim).toFixed(3)}</b>`);
  if (q.edge_iou != null) parts.push(`edge IoU <b>${Number(q.edge_iou).toFixed(3)}</b>`);
  if (q.text_similarity != null) parts.push(`text match <b>${Number(q.text_similarity).toFixed(3)}</b>`);
  parts.push(`page ${page.index + 1}/${state.pages.length}`);
  if (q.iterations) parts.push(`correction iterations <b>${q.iterations}</b>`);
  el("qualityPanel").innerHTML = parts.join(" &middot; ");
}

// ---------------------------------------------------------------- views
function renderView() {
  const area = el("viewerArea");
  area.innerHTML = "";
  if (!state.documentId) {
    area.innerHTML = '<p class="placeholder">Upload and process a document to see it here.</p>';
    return;
  }
  const base = `/api/documents/${state.documentId}/files`;
  const page = state.pages[state.page] || { index: 0 };
  const pageIndex = page.index || 0;
  const width = Math.round(700 * state.zoom);

  const addImage = (src, label) => {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = `<div class="muted" style="font-size:11.5px;margin-bottom:4px">${label}</div>
      <img src="${src}" style="width:${width}px">`;
    area.appendChild(wrapper);
  };

  if (state.view === "original") {
    const pane = document.createElement("div");
    pane.className = "pane";
    area.appendChild(pane);
    pane.innerHTML = `<div><div class="muted" style="font-size:11.5px;margin-bottom:4px">original page ${pageIndex + 1}</div>
      <img src="${base}/pages/page_${String(pageIndex).padStart(2, "0")}_preview.jpg" style="width:${width}px"></div>`;
  } else if (state.view === "detection") {
    const pane = document.createElement("div");
    pane.className = "pane";
    area.appendChild(pane);
    pane.innerHTML = `<div><div class="muted" style="font-size:11.5px;margin-bottom:4px">detected regions</div>
      <img src="${base}/pages/page_${String(pageIndex).padStart(2, "0")}_overlay.jpg" style="width:${width}px"></div>`;
  } else if (state.view === "reconstruction") {
    const frame = document.createElement("iframe");
    frame.src = `/api/documents/${state.documentId}/preview`;
    frame.style.height = `${Math.round(700 * state.zoom)}px`;
    area.appendChild(frame);
  } else if (state.view === "compare") {
    const pane = document.createElement("div");
    pane.className = "pane";
    area.appendChild(pane);
    pane.innerHTML = `
      <div><div class="muted" style="font-size:11.5px;margin-bottom:4px">original</div>
        <img src="${base}/pages/page_${String(pageIndex).padStart(2, "0")}_preview.jpg" style="width:${width}px"></div>
      <div><div class="muted" style="font-size:11.5px;margin-bottom:4px">reconstruction (rendered)</div>
        <img src="/api/documents/${state.documentId}/preview.png?page=${pageIndex}" style="width:${width}px"></div>`;
  }
  el("pageLabel").textContent = `page ${pageIndex + 1}/${Math.max(state.pages.length, 1)}`;
  el("zoomLabel").textContent = `${Math.round(state.zoom * 100)}%`;
}

// ---------------------------------------------------------------- upload + process
async function uploadFile(file) {
  if (!file) return;
  setStatus(el("uploadStatus"), `uploading ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const payload = await api("/api/documents/upload", { method: "POST", body: form });
    setStatus(el("uploadStatus"), `uploaded ${payload.filename} (${payload.document_id})`, "ok");
    await selectDocument(payload.document_id);
  } catch (error) {
    setStatus(el("uploadStatus"), error.message, "error");
  }
}

async function startProcessing() {
  if (!state.documentId) return;
  const body = {
    enable_verify: el("enableVerify").checked,
    run_correction_loop: el("correctionLoop").checked,
    enable_lens: el("enableLens").checked,
    enable_assets_download: el("enableAssets").checked,
  };
  if (el("ocrProvider").value) body.ocr_provider = el("ocrProvider").value;
  try {
    const payload = await api(`/api/documents/${state.documentId}/process`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.jobId = payload.job_id;
    setStatus(el("jobStatus"), `job ${payload.job_id} queued`);
    pollStatus();
  } catch (error) {
    setStatus(el("jobStatus"), error.message, "error");
  }
}

function pollStatus() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  const tick = async () => {
    if (!state.documentId) return;
    try {
      const query = state.jobId ? `?job_id=${state.jobId}` : "";
      const payload = await api(`/api/documents/${state.documentId}/status${query}`);
      const job = payload.job;
      setStatus(el("jobStatus"), `status: ${job.status}${job.stage ? ` (${job.stage})` : ""} — ${Math.round((job.progress || 0) * 100)}%`);
      const list = el("stageList");
      list.innerHTML = "";
      (job.stages || []).forEach((stage) => {
        const item = document.createElement("li");
        item.className = stage.status === "completed" ? "done" : stage.status === "failed" ? "failed" : "";
        item.innerHTML = `<span>${stage.stage}</span><span>${stage.status}${stage.duration_ms ? ` · ${Math.round(stage.duration_ms)} ms` : ""}</span>`;
        list.appendChild(item);
      });
      if (job.error) setStatus(el("jobStatus"), `status: ${job.status} — ${job.error}`, "error");
      if (job.status === "completed" || job.status === "failed") {
        await loadDocument();
        await refreshDocuments();
        return;
      }
    } catch (error) {
      setStatus(el("jobStatus"), error.message, "error");
    }
    state.pollTimer = setTimeout(tick, 1500);
  };
  tick();
}

// ---------------------------------------------------------------- element editing
function selectElement(elementId) {
  const element = state.elements.find((item) => item.id === elementId);
  if (!element) return;
  state.selected = elementId;
  document.querySelectorAll("#elementList li").forEach((node) => node.classList.remove("selected"));
  renderView();
  const template = el("editorTemplate");
  const clone = template.content.cloneNode(true);
  const container = el("editorBody");
  container.innerHTML = "";
  container.appendChild(clone);
  const form = container.querySelector(".editor-form");
  const set = (field, value) => {
    const input = form.querySelector(`[data-field="${field}"]`);
    if (input && value != null) input.value = value;
  };
  set("type", element.type);
  set("text", element.text || "");
  set("alignment", element.alignment || "left");
  const typography = element.typography || {};
  set("font_family", typography.font_family || "");
  set("font_size_pt", typography.font_size_pt || "");
  set("font_weight", typography.font_weight || "");
  set("color", typography.color || "");
  set("line_height_pt", typography.line_height_pt || "");
  set("paragraph_spacing_pt", typography.paragraph_spacing_pt || "");
  const mm = element.bbox_mm || {};
  set("x_mm", mm.x); set("y_mm", mm.y); set("width_mm", mm.width); set("height_mm", mm.height);
  set("asset_path", element.image && element.image.asset_path ? element.image.asset_path : "");
  form.querySelector("#saveElement").onclick = () => saveElement(element.id);
}

async function saveElement(elementId) {
  const form = el("editorBody").querySelector(".editor-form");
  const patch = {};
  form.querySelectorAll("[data-field]").forEach((input) => {
    const field = input.dataset.field;
    const value = input.value;
    if (value === "" || value == null) return;
    if (field === "cells") {
      try { patch.cells = JSON.parse(value); } catch (error) { throw new Error(`cells JSON invalid: ${error.message}`); }
      return;
    }
    if (["font_size_pt", "font_weight", "line_height_pt", "paragraph_spacing_pt", "x_mm", "y_mm", "width_mm", "height_mm"].includes(field)) {
      const numeric = Number(value);
      if (!Number.isNaN(numeric)) patch[field] = numeric;
      return;
    }
    patch[field] = value;
  });
  try {
    const payload = await api(`/api/documents/${state.documentId}/elements/${elementId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    setStatus(el("editorStatus"), `saved (revision ${payload.revision})`, "ok");
    await loadElements();
    renderView();
  } catch (error) {
    setStatus(el("editorStatus"), error.message, "error");
  }
}

// ---------------------------------------------------------------- export
async function runExport(format) {
  if (!state.documentId) return;
  setStatus(el("exportStatus"), `generating ${format}…`);
  try {
    const payload = await api(`/api/documents/${state.documentId}/export`, {
      method: "POST",
      body: JSON.stringify({ formats: [format] }),
    });
    const descriptor = payload.exports[format];
    setStatus(el("exportStatus"), `${format} ready (${Math.round(descriptor.size_bytes / 1024)} KB)`, "ok");
    window.location.href = descriptor.url;
  } catch (error) {
    setStatus(el("exportStatus"), error.message, "error");
  }
}

// ---------------------------------------------------------------- settings
async function loadSettings() {
  try {
    const settings = await api("/api/settings");
    applySettings(settings);
  } catch (error) {
    setStatus(el("serpApiStatus"), `settings unavailable: ${error.message}`, "bad");
  }
}
function applySettings(settings) {
  el("serpApiEnabled").checked = Boolean(settings.serpapi_enabled);
  const where = settings.serpapi_key_source ? ` (from ${settings.serpapi_key_source})` : "";
  setStatus(
    el("serpApiStatus"),
    settings.serpapi_key_set
      ? `key configured ${settings.serpapi_key_hint}${where}`
      : "no key configured — Tesseract is used for all text",
    settings.serpapi_key_set ? "ok" : "",
  );
  el("serpApiKey").value = "";
}
async function saveSerpApi() {
  const key = el("serpApiKey").value.trim();
  const enabled = el("serpApiEnabled").checked;
  if (!key && !enabled) {
    setStatus(el("serpApiStatus"), "paste a key or tick the enable box first", "bad");
    return;
  }
  setStatus(el("serpApiStatus"), "saving…");
  try {
    const body = { enabled };
    if (key) body.api_key = key;
    const result = await api("/api/settings/serpapi", { method: "PUT", body: JSON.stringify(body) });
    applySettings(result);
    setStatus(
      el("serpApiStatus"),
      `saved — key ${result.serpapi_key_hint}${result.serpapi_enabled ? ", Lens enabled" : ", Lens disabled"}`,
      "ok",
    );
    loadCapabilities();
  } catch (error) {
    setStatus(el("serpApiStatus"), `save failed: ${error.message}`, "bad");
  }
}
async function verifySerpApi() {
  setStatus(el("serpApiStatus"), "asking SerpApi…");
  try {
    const result = await api("/api/settings/serpapi/verify", { method: "POST" });
    setStatus(el("serpApiStatus"), result.message, result.verified ? "ok" : "bad");
  } catch (error) {
    setStatus(el("serpApiStatus"), `verify failed: ${error.message}`, "bad");
  }
}
async function clearSerpApi() {
  try {
    const result = await api("/api/settings/serpapi", { method: "DELETE" });
    applySettings(result);
    setStatus(el("serpApiStatus"), "key cleared; Tesseract remains active", "");
    loadCapabilities();
  } catch (error) {
    setStatus(el("serpApiStatus"), `clear failed: ${error.message}`, "bad");
  }
}
// ---------------------------------------------------------------- wiring
function wire() {
  const input = el("file");
  const dropzone = el("dropzone");
  input.onchange = () => uploadFile(input.files[0]);
  dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    uploadFile(event.dataTransfer.files[0]);
  });

  el("processBtn").onclick = startProcessing;
  el("saveSerpApi").onclick = saveSerpApi;
  el("verifySerpApi").onclick = verifySerpApi;
  el("clearSerpApi").onclick = clearSerpApi;
  el("serpApiKey").addEventListener("keydown", (event) => {
    if (event.key === "Enter") saveSerpApi();
  });
  el("onlyReview").onchange = () => loadElements().catch(() => {});
  el("refreshPreview").onclick = () => loadDocument();
  el("prevPage").onclick = () => { state.page = Math.max(0, state.page - 1); renderView(); renderQuality(); };
  el("nextPage").onclick = () => { state.page = Math.min(Math.max(state.pages.length - 1, 0), state.page + 1); renderView(); renderQuality(); };
  el("zoomIn").onclick = () => { state.zoom = Math.min(2.5, state.zoom + 0.15); renderView(); };
  el("zoomOut").onclick = () => { state.zoom = Math.max(0.35, state.zoom - 0.15); renderView(); };

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll(".tab").forEach((other) => other.classList.remove("active"));
      tab.classList.add("active");
      state.view = tab.dataset.view;
      renderView();
    };
  });

  el("exportPdf").onclick = () => runExport("pdf");
  el("exportDocx").onclick = () => runExport("docx");
  el("exportHtml").onclick = () => runExport("html");
}

wire();
loadCapabilities();
loadSettings();
refreshDocuments();
