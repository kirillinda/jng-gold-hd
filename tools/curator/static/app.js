"use strict";
const $ = s => document.querySelector(s);
const api = async (path, body) => {
  const r = await fetch(path, body ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  } : undefined);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

let ASSETS = [];            // [{rel, status}]
let REL = null, DET = null; // current asset + its /api/asset detail
let W = 0, H = 0, ZOOM = 1;
let maskCanvas = null, mctx = null;   // offscreen, W×H
let brush = 48, erasing = false;
let jobId = null, jobTimer = null;
let selectedOpt = null;
let maskSaveTimer = null;

// ---- sidebar ---------------------------------------------------------------
async function refreshAssets() {
  const d = await api("/api/assets");
  ASSETS = d.assets;
  const done = (d.counts.flux || 0) + (d.counts.gsr || 0);
  $("#progress").textContent =
    `${done} / ${ASSETS.length} curated` +
    (d.counts.skip ? ` · ${d.counts.skip} skipped` : "");
  renderList();
}

function renderList() {
  const q = $("#search").value.toLowerCase();
  const st = $("#statusFilter").value;
  const ul = $("#assetList");
  ul.textContent = "";
  for (const a of ASSETS) {
    if (q && !a.rel.toLowerCase().includes(q)) continue;
    if (st && a.status !== st) continue;
    const li = document.createElement("li");
    li.textContent = a.rel;
    li.title = a.rel;
    li.className = `st-${a.status}` + (a.rel === REL ? " sel" : "");
    li.onclick = () => selectAsset(a.rel);
    ul.appendChild(li);
  }
}

// ---- canvases --------------------------------------------------------------
const paintStacks = () => [$("#stackOrig"), $("#stackUp")];
const allStacks = () => [...paintStacks(), $("#stackCand")];
let UP_IMG = null;    // decoded GSR sheet, for the candidate panel
let CURBOX = null;    // crop box of the current generation batch
let candOpt = null;   // option shown in the candidate panel

function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("load failed: " + src));
    im.src = src;
  });
}

function applyZoom() {
  for (const st of allStacks()) {
    st.style.width = (W * ZOOM) + "px";
    st.style.height = (H * ZOOM) + "px";
    for (const c of st.querySelectorAll("canvas")) {
      c.style.width = (W * ZOOM) + "px";
      c.style.height = (H * ZOOM) + "px";
    }
  }
  $("#zoomSlider").value = Math.round(ZOOM * 100);
  $("#zoomVal").textContent = Math.round(ZOOM * 100);
}

function fitZoom() {
  const avail = ($("#panels").clientWidth - 90) / 3;
  const availH = $("#panels").clientHeight - 60;
  ZOOM = Math.min(1, avail / W, availH / H);
  if (ZOOM <= 0 || !isFinite(ZOOM)) ZOOM = 1;
  applyZoom();
}

function redrawMask() {
  // tint the white mask red once, stamp it on both overlays
  const tint = document.createElement("canvas");
  tint.width = W; tint.height = H;
  const t = tint.getContext("2d");
  t.drawImage(maskCanvas, 0, 0);
  t.globalCompositeOperation = "source-in";
  t.fillStyle = "#ff2848";
  t.fillRect(0, 0, W, H);
  for (const st of paintStacks()) {
    const c = st.querySelector(".maskc");
    const x = c.getContext("2d");
    x.clearRect(0, 0, W, H);
    x.globalAlpha = 0.45;
    x.drawImage(tint, 0, 0);
    x.globalAlpha = 1;
  }
}

function canvasPos(e, el) {
  const r = el.getBoundingClientRect();
  return [(e.clientX - r.left) * (W / r.width),
          (e.clientY - r.top) * (H / r.height)];
}

function stroke(x0, y0, x1, y1) {
  mctx.globalCompositeOperation = erasing ? "destination-out" : "source-over";
  mctx.strokeStyle = "#fff";
  mctx.lineWidth = brush;
  mctx.lineCap = mctx.lineJoin = "round";
  mctx.beginPath();
  mctx.moveTo(x0, y0);
  mctx.lineTo(x1 + 0.01, y1 + 0.01);
  mctx.stroke();
  redrawMask();
  scheduleMaskSave();
}

function bindPainting() {
  for (const st of paintStacks()) {
    const c = st.querySelector(".maskc");
    let last = null;
    c.onpointerdown = e => {
      if (e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      last = canvasPos(e, c);
      stroke(last[0], last[1], last[0], last[1]);
    };
    c.onpointermove = e => {
      if (!last) return;
      const p = canvasPos(e, c);
      stroke(last[0], last[1], p[0], p[1]);
      last = p;
    };
    c.onpointerup = c.onpointercancel = () => { last = null; saveMaskNow(); };
  }
}

function maskDataUrl() { return maskCanvas.toDataURL("image/png"); }

function maskIsEmpty() {
  const d = mctx.getImageData(0, 0, W, H).data;
  for (let i = 3; i < d.length; i += 4) if (d[i] > 8) return false;
  return true;
}

function scheduleMaskSave() {
  clearTimeout(maskSaveTimer);
  maskSaveTimer = setTimeout(saveMaskNow, 1500);
}

async function saveMaskNow() {
  clearTimeout(maskSaveTimer);
  if (!REL || !maskCanvas || maskIsEmpty()) return;
  try { await api("/api/mask", {rel: REL, mask: maskDataUrl()}); } catch (e) {}
}

// ---- asset selection -------------------------------------------------------
async function selectAsset(rel) {
  await saveMaskNow();
  stopPolling();
  REL = rel;
  selectedOpt = null;
  DET = await api("/api/asset?rel=" + encodeURIComponent(rel));
  W = DET.w; H = DET.h;
  location.hash = encodeURIComponent(rel);

  $("#assetName").textContent = rel;
  setBadge(DET.status);
  for (const st of allStacks()) {
    for (const c of st.querySelectorAll("canvas")) { c.width = W; c.height = H; }
  }
  UP_IMG = null;
  CURBOX = DET.gen ? DET.gen.box : null;
  candOpt = candObj = candImg = null;
  candManual = false;
  $("#candLabel").textContent = "";
  maskCanvas = document.createElement("canvas");
  maskCanvas.width = W; maskCanvas.height = H;
  mctx = maskCanvas.getContext("2d", {willReadFrequently: true});

  fitZoom();
  const enc = encodeURIComponent(rel);
  const jobs = [
    loadImage("/img/up?rel=" + enc).then(im => {
      UP_IMG = im;
      $("#stackUp .imgc").getContext("2d").drawImage(im, 0, 0);
    }),
    loadImage("/img/orig?rel=" + enc).then(im => {
      const x = $("#stackOrig .imgc").getContext("2d");
      x.imageSmoothingEnabled = false;
      x.drawImage(im, 0, 0, W, H);
    }).catch(() => {}),
  ];
  if (DET.has_mask) {
    jobs.push(loadImage("/img/mask?rel=" + enc + "&t=" + Date.now()).then(im => {
      mctx.drawImage(im, 0, 0);  // white-on-transparent PNG round-trips as-is
    }).catch(() => {}));
  }
  await Promise.all(jobs);
  redrawMask();
  setParams(DET.params);
  renderOptions(DET.gen ? DET.gen.options : [], DET.choice);
  if (DET.gen) {  // put the saved choice (or the first option) up for comparison
    const opts = DET.gen.options;
    showCandidate(opts.find(o => o.file === DET.choice) || opts[0]);
  }
  $("#jobBox").hidden = true;
  renderList();
}

function setBadge(status) {
  const b = $("#assetStatus");
  b.textContent = status;
  b.className = "badge st-" + status;
}

// ---- params ----------------------------------------------------------------
const P = ["strength", "steps", "guidance", "feather"];
const PARAM_DEFAULTS = {  // mirror of the server's tested-batch defaults
  strength: 0.25, steps: 32, guidance: 3.5, feather: 12,
  v2_scale: 1.0, v3_scale: 1.0, use_v2: true, use_v3: true,
  seed: 7, prompt: "",
};
function setParams(p) {
  for (const k of P) {
    $("#p_" + k).value = p[k];
    $("#p_" + k).nextElementSibling.textContent = p[k];
  }
  $("#p_v2").value = p.v2_scale;
  $("#p_v2").nextElementSibling.textContent = p.v2_scale;
  $("#p_v3").value = p.v3_scale;
  $("#p_v3").nextElementSibling.textContent = p.v3_scale;
  $("#c_v2").checked = p.use_v2;
  $("#c_v3").checked = p.use_v3;
  $("#p_seed").value = p.seed;
  $("#p_prompt").value = p.prompt;
}

function getParams() {
  return {
    strength: +$("#p_strength").value, steps: +$("#p_steps").value,
    guidance: +$("#p_guidance").value, feather: +$("#p_feather").value,
    v2_scale: +$("#p_v2").value, v3_scale: +$("#p_v3").value,
    use_v2: $("#c_v2").checked, use_v3: $("#c_v3").checked,
    seed: +$("#p_seed").value, prompt: $("#p_prompt").value.trim(),
  };
}

// ---- generation ------------------------------------------------------------
async function generate() {
  if (!REL) return;
  if (maskIsEmpty()) { setJobMsg("paint a mask first", true); return; }
  const p = getParams();
  if (!p.use_v2 && !p.use_v3) { setJobMsg("enable v2 and/or v3", true); return; }
  $("#generateBtn").disabled = true;
  try {
    const d = await api("/api/generate", {rel: REL, mask: maskDataUrl(), params: p});
    jobId = d.job;
    $("#jobBox").hidden = false;
    setJobMsg("queued", false);
    $("#jobFill").style.width = "0";
    renderOptions([], null);
    CURBOX = null;  // the new batch gets its own crop box
    candOpt = candObj = candImg = null;
    candManual = false;
    $("#candLabel").textContent = "";
    $("#stackCand .imgc").getContext("2d").clearRect(0, 0, W, H);
    jobTimer = setInterval(poll, 1500);
  } catch (e) {
    setJobMsg(e.message, true);
    $("#generateBtn").disabled = false;
  }
}

function setJobMsg(msg, isErr) {
  $("#jobBox").hidden = false;
  $("#jobMsg").textContent = msg;
  $("#jobMsg").className = isErr ? "error" : "";
}

function stopPolling() {
  clearInterval(jobTimer);
  jobTimer = null;
  jobId = null;
  $("#generateBtn").disabled = false;
}

async function poll() {
  if (!jobId) return;
  let j;
  try { j = await api("/api/job?id=" + jobId); } catch (e) { return; }
  if (j.rel !== REL) { stopPolling(); return; }  // user moved on
  setJobMsg(j.message, j.status === "error");
  if (j.total) $("#jobFill").style.width = (100 * j.done / j.total) + "%";
  renderOptions(j.options, null);
  if (!candManual && j.options.length)  // live-preview the newest option
    showCandidate(j.options[j.options.length - 1]);
  if (j.status === "done" || j.status === "error") {
    clearInterval(jobTimer);
    jobTimer = null;
    jobId = null;
    $("#generateBtn").disabled = false;
  }
}

// ---- candidate panel -------------------------------------------------------
let candManual = false;   // user clicked an option; stop auto-previewing
let candImg = null;       // decoded crop of the shown option, for hold-to-flip
let candObj = null;

function optUrl(file) {
  return "/img/gen?rel=" + encodeURIComponent(REL) +
         "&f=" + encodeURIComponent(file);
}

async function ensureBox() {
  if (CURBOX) return;
  try {
    const d = await api("/api/asset?rel=" + encodeURIComponent(REL));
    CURBOX = d.gen ? d.gen.box : null;
  } catch (e) {}
}

function drawCandBase(x) {
  x.clearRect(0, 0, W, H);
  if (UP_IMG) x.drawImage(UP_IMG, 0, 0);
}

function redrawCandidate(plain) {
  const x = $("#stackCand .imgc").getContext("2d");
  drawCandBase(x);
  if (!plain && candImg && CURBOX) {
    x.clearRect(CURBOX[0], CURBOX[1],
                CURBOX[2] - CURBOX[0], CURBOX[3] - CURBOX[1]);
    x.drawImage(candImg, CURBOX[0], CURBOX[1]);
  }
}

async function showCandidate(o) {
  if (!o || o.file === candOpt) return;
  await ensureBox();
  if (!CURBOX || !UP_IMG) return;
  const img = await loadImage(optUrl(o.file));
  candOpt = o.file;
  candObj = o;
  candImg = img;
  $("#candLabel").textContent = "— " + o.label;
  redrawCandidate(false);
}

function bindCandidate() {
  const c = $("#stackCand .imgc");
  c.style.cursor = "pointer";
  c.onpointerdown = e => {
    if (e.button !== 0) return;
    c.setPointerCapture(e.pointerId);
    redrawCandidate(true);   // hold = plain GSR
  };
  c.onpointerup = c.onpointercancel = () => redrawCandidate(false);
}

// ---- options ---------------------------------------------------------------
function renderOptions(options, chosen) {
  const box = $("#options");
  box.textContent = "";
  for (const o of options || []) {
    const div = document.createElement("div");
    div.className = "opt" + (o.file === selectedOpt ? " sel" : "")
                          + (o.file === chosen ? " chosen" : "");
    const img = document.createElement("img");
    img.src = "/img/gen?rel=" + encodeURIComponent(REL) +
              "&f=" + encodeURIComponent(o.file) + "&t=" + Date.now();
    const cap = document.createElement("div");
    cap.className = "cap";
    cap.textContent = o.label + (o.file === chosen ? " ✓ saved" : "");
    div.append(img, cap);
    div.onclick = () => {
      selectedOpt = o.file;
      $("#saveBtn").disabled = false;
      for (const el of box.children) el.classList.remove("sel");
      div.classList.add("sel");
      candManual = true;
      showCandidate(o);
    };
    div.ondblclick = () => openLightbox(o.file);
    box.appendChild(div);
  }
  $("#saveBtn").disabled = !selectedOpt;
}

function openLightbox(file) {
  const enc = encodeURIComponent(REL);
  const opt = "/img/gen?rel=" + enc + "&f=" + encodeURIComponent(file);
  const gsr = "/img/gen?rel=" + enc + "&f=gsr.png";
  const img = $("#lightboxImg");
  img.src = opt;
  img.onmousedown = e => { if (e.button === 0) img.src = gsr; };
  img.onmouseup = img.onmouseleave = () => { img.src = opt; };
  img.ondragstart = () => false;
  $("#lightbox").hidden = false;
}

// ---- actions ---------------------------------------------------------------
async function finish(action) {
  try {
    if (action === "save") {
      if (!selectedOpt) return;
      await api("/api/choose", {rel: REL, option: selectedOpt});
    } else if (action === "gsr") {
      await api("/api/choose", {rel: REL, option: "gsr"});
    } else if (action === "skip") {
      await api("/api/skip", {rel: REL});
    } else if (action === "reset") {
      await api("/api/reset", {rel: REL});
      await refreshAssets();
      await selectAsset(REL);
      return;
    }
    await refreshAssets();
    nextTodo();
  } catch (e) {
    setJobMsg(e.message, true);
  }
}

function nextTodo() {
  const i = ASSETS.findIndex(a => a.rel === REL);
  for (let k = 1; k <= ASSETS.length; k++) {
    const a = ASSETS[(i + k) % ASSETS.length];
    if (a.status === "todo") { selectAsset(a.rel); return; }
  }
  refreshAssets();  // everything curated
}

function step(delta) {
  const i = ASSETS.findIndex(a => a.rel === REL);
  if (i < 0) return;
  const a = ASSETS[(i + delta + ASSETS.length) % ASSETS.length];
  selectAsset(a.rel);
}

// ---- wiring ----------------------------------------------------------------
function bindUI() {
  $("#search").oninput = renderList;
  $("#statusFilter").onchange = renderList;
  $("#brushSize").oninput = e => {
    brush = +e.target.value;
    $("#brushVal").textContent = brush;
  };
  $("#eraserBtn").onclick = () => {
    erasing = !erasing;
    $("#eraserBtn").classList.toggle("active", erasing);
  };
  $("#clearBtn").onclick = () => {
    mctx.clearRect(0, 0, W, H);
    redrawMask();
  };
  $("#zoomSlider").oninput = e => { ZOOM = +e.target.value / 100; applyZoom(); };
  $("#fitBtn").onclick = fitZoom;
  $("#prevBtn").onclick = () => step(-1);
  $("#nextBtn").onclick = () => step(1);
  $("#generateBtn").onclick = generate;
  $("#saveBtn").onclick = () => finish("save");
  $("#gsrBtn").onclick = () => finish("gsr");
  $("#skipBtn").onclick = () => finish("skip");
  $("#resetBtn").onclick = () => finish("reset");
  $("#defaultsBtn").onclick = () => setParams(PARAM_DEFAULTS);
  $("#presetBtn").onclick = () => {
    $("#p_prompt").value = "highly detailed sharp video game art, metal " +
      "surfaces, rivets, grime, crisp texture";
  };
  for (const k of [...P, "v2", "v3"]) {
    $("#p_" + k).oninput = e =>
      e.target.nextElementSibling.textContent = e.target.value;
  }
  $("#lightbox").onclick = e => {
    if (e.target.id === "lightbox" || e.target.id === "lightboxScroll")
      $("#lightbox").hidden = true;
  };
  document.onkeydown = e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key === "Escape") $("#lightbox").hidden = true;
    else if (e.key === "[") step(-1);
    else if (e.key === "]") step(1);
    else if (e.key === "e") $("#eraserBtn").click();
  };
  bindPainting();
  bindCandidate();
}

async function init() {
  bindUI();
  await refreshAssets();
  const fromHash = decodeURIComponent(location.hash.slice(1));
  if (fromHash && ASSETS.some(a => a.rel === fromHash)) {
    await selectAsset(fromHash);
  } else {
    const first = ASSETS.find(a => a.status === "todo") || ASSETS[0];
    if (first) await selectAsset(first.rel);
  }
}

init();
