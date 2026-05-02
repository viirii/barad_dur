const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");
const airfieldSearch = document.getElementById("airfieldSearch");
const watchlist = document.getElementById("watchlist");
const overlayToggle = document.getElementById("overlayToggle");
const opacitySlider = document.getElementById("opacitySlider");
const resetView = document.getElementById("resetView");
const refreshAnalysisBtn = document.getElementById("refreshAnalysis");
const previousCapture = document.getElementById("previousCapture");
const nextCapture = document.getElementById("nextCapture");
const captureSelect = document.getElementById("captureSelect");
const canvasWrap = document.getElementById("canvasWrap");
const viewerTitle = document.getElementById("viewerTitle");
const viewerSubtitle = document.getElementById("viewerSubtitle");
const selectedCode = document.getElementById("selectedCode");
const airfieldType = document.getElementById("airfieldType");
const lastCapture = document.getElementById("lastCapture");
const baselineSummary = document.getElementById("baselineSummary");
const riskBand = document.getElementById("riskBand");
const summaryText = document.getElementById("summaryText");
const compositionGrid = document.getElementById("compositionGrid");
const signalList = document.getElementById("signalList");
const findingsList = document.getElementById("findingsList");
const timeline = document.getElementById("timeline");
const timelineNote = document.getElementById("timelineNote");

const typeColors = {
  commercial: "#6fb6ff",
  cargo: "#a8d66d",
  business: "#d7b4ff",
  military: "#ff8b72",
  helicopter: "#ffd36e",
  unknown: "#c3c8c1",
};

const state = {
  airfields: [],
  report: null,
  reportImage: null,
  reportImageUrl: "",
  captureIndex: -1,
  overlayEnabled: true,
  overlayOpacity: 0.55,
  panX: 0,
  panY: 0,
  isDragging: false,
  lastPointer: null,
};

function resizeCanvas() {
  const rect = canvasWrap.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  render();
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/** Show YYYY-MM-DD for compact or legacy dates (e.g. 20260330 → 2026-03-30). */
function formatCaptureDateForDisplay(value) {
  if (value == null || value === "") return "";
  const s = String(value).trim();
  if (/^capture\s+\d+/i.test(s)) return s;
  const compact = s.match(/^(\d{4})(\d{2})(\d{2})$/);
  if (compact) return `${compact[1]}-${compact[2]}-${compact[3]}`;
  const separated = s.match(/^(\d{4})[-_/](\d{2})[-_/](\d{2})/);
  if (separated) return `${separated[1]}-${separated[2]}-${separated[3]}`;
  return s;
}

function resetViewport() {
  state.panX = 0;
  state.panY = 0;
  render();
}

function getSeverityClass(severity) {
  if (severity === "high" || severity === "needs_review") return "high";
  if (severity === "medium" || severity === "notable") return "medium";
  if (severity === "low" || severity === "normal") return "low";
  return "neutral";
}

function drawSyntheticAirfield(rect, report) {
  ctx.fillStyle = report?.id === "suu" ? "#252b24" : "#202723";
  ctx.fillRect(rect.x, rect.y, rect.width, rect.height);

  ctx.save();
  ctx.translate(rect.x, rect.y);
  ctx.scale(rect.width, rect.height);

  ctx.fillStyle = "rgba(255,255,255,0.04)";
  for (let i = 0; i < 7; i += 1) {
    ctx.fillRect(0.08 + i * 0.12, 0.08, 0.05, 0.82);
  }

  ctx.strokeStyle = "#5e665d";
  ctx.lineWidth = 0.014;
  ctx.beginPath();
  ctx.moveTo(0.08, 0.74);
  ctx.lineTo(0.92, 0.28);
  ctx.stroke();

  ctx.strokeStyle = "#3f4740";
  ctx.lineWidth = 0.03;
  ctx.beginPath();
  ctx.moveTo(0.1, 0.28);
  ctx.lineTo(0.85, 0.58);
  ctx.stroke();

  ctx.fillStyle = "rgba(160, 176, 156, 0.14)";
  if (report?.id === "sfo") {
    ctx.fillRect(0.1, 0.1, 0.38, 0.16);
    ctx.fillRect(0.45, 0.34, 0.28, 0.14);
  } else {
    ctx.fillRect(0.16, 0.16, 0.32, 0.2);
    ctx.fillRect(0.43, 0.42, 0.38, 0.2);
  }

  ctx.restore();
}

function getSelectedCapture(report) {
  if (!report) return null;
  const captures = report.images || [];
  if (!captures.length) return null;
  const index = state.captureIndex >= 0 ? state.captureIndex : captures.length - 1;
  return captures[clamp(index, 0, captures.length - 1)];
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    if (!url) {
      resolve(null);
      return;
    }

    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

/** Pixel coords (legacy OpenAI cache) vs normalized 0–1 fractions (HF, scaffold, current OpenAI). */
function bboxUsesPixelCoords(bbox, analysis) {
  const sys = analysis?.bboxCoordinateSystem;
  if (sys === "pixels") return true;
  if (sys === "normalized") return false;
  const maxB = Math.max(
    Number(bbox.x),
    Number(bbox.y),
    Number(bbox.width),
    Number(bbox.height),
  );
  return maxB > 1.001;
}

function drawAircraftMarker(aircraft, rect, analysis, dimPrevious = false) {
  const bbox = aircraft.bbox;
  if (!bbox || ![bbox.x, bbox.y, bbox.width, bbox.height].every((v) => Number.isFinite(Number(v)))) return;

  const iw = state.reportImage?.naturalWidth || 1;
  const ih = state.reportImage?.naturalHeight || 1;

  const usePixels = bboxUsesPixelCoords(bbox, analysis);

  const bwPx = usePixels
    ? Math.max(4, (bbox.width / iw) * rect.width)
    : Math.max(4, bbox.width * rect.width);
  const bhPx = usePixels
    ? Math.max(4, (bbox.height / ih) * rect.height)
    : Math.max(4, bbox.height * rect.height);
  const bx = usePixels ? rect.x + (bbox.x / iw) * rect.width : rect.x + bbox.x * rect.width;
  const by = usePixels ? rect.y + (bbox.y / ih) * rect.height : rect.y + bbox.y * rect.height;

  const color =
    typeColors[aircraft.type] || typeColors[aircraft.classification] || typeColors.unknown;
  const alpha = dimPrevious ? 0.35 : 0.9;
  const confidence = aircraft.classificationConfidence || aircraft.detectionConfidence || 0.5;

  ctx.save();
  ctx.globalAlpha = alpha;

  // Map raw bbox (pixels or normalized) → canvas using loaded image natural size only.
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.strokeRect(bx, by, bwPx, bhPx);

  // Confidence: highlight portion of the top edge (same role as the old arc segment).
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(bx + bwPx * confidence, by);
  ctx.stroke();

  if (confidence < 0.5) {
    ctx.strokeStyle = "#ffd36e";
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.strokeRect(bx - 3, by - 3, bwPx + 6, bhPx + 6);
    ctx.setLineDash([]);
  }

  ctx.restore();
}

function getSelectedAnalysis(report) {
  if (!report?.imageAnalyses?.length) return null;
  const index = clamp(state.captureIndex, 0, report.imageAnalyses.length - 1);
  return report.imageAnalyses[index];
}

function drawReportOverlay(rect, report) {
  if (!state.overlayEnabled || !report) return;
  const capture = getSelectedCapture(report);
  const analysis = getSelectedAnalysis(report);
  if (!capture?.imageUrl || !analysis) return;

  // Avoid drawing bbox coordinates for capture B on the bitmap for capture A (race during image load or resize).
  if (state.reportImageUrl !== capture.imageUrl) return;

  const aircraft = Array.isArray(analysis.planes) ? analysis.planes : [];
  ctx.save();
  ctx.globalAlpha = state.overlayOpacity;
  aircraft.forEach((plane) => {
    drawAircraftMarker(plane, rect, analysis, false);
  });
  ctx.restore();
}

function getSceneRect() {
  const width = canvasWrap.clientWidth;
  const height = canvasWrap.clientHeight;
  const activeImage = state.reportImage;
  const iw = activeImage ? activeImage.naturalWidth || activeImage.width : 0;
  const ih = activeImage ? activeImage.naturalHeight || activeImage.height : 0;
  const sceneRatio = iw > 0 && ih > 0 ? iw / ih : 16 / 10;
  const viewportRatio = width / height;
  let sceneWidth = width * 0.88;
  let sceneHeight = sceneWidth / sceneRatio;

  if (viewportRatio > sceneRatio) {
    sceneHeight = height * 0.86;
    sceneWidth = sceneHeight * sceneRatio;
  }

  return {
    x: (width - sceneWidth) / 2 + state.panX,
    y: (height - sceneHeight) / 2 + state.panY,
    width: sceneWidth,
    height: sceneHeight,
  };
}

function render() {
  const width = canvasWrap.clientWidth;
  const height = canvasWrap.clientHeight;
  ctx.clearRect(0, 0, width, height);

  if (!state.report) return;

  const rect = getSceneRect();
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,0.28)";
  ctx.shadowBlur = 24;
  ctx.shadowOffsetY = 12;

  if (state.reportImage) {
    ctx.drawImage(state.reportImage, rect.x, rect.y, rect.width, rect.height);
    drawReportOverlay(rect, state.report);
  } else {
    drawSyntheticAirfield(rect, state.report);
    drawReportOverlay(rect, state.report);
  }

  ctx.shadowBlur = 0;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.width - 1, rect.height - 1);
  ctx.restore();
}

function renderWatchlist() {
  const query = airfieldSearch.value.trim().toLowerCase();
  const airfields = state.airfields.filter((airfield) => {
    return airfield.code.toLowerCase().includes(query);
  });

  watchlist.innerHTML = "";
  airfields.forEach((airfield) => {
    const button = document.createElement("button");
    button.className = "watchlist-item";
    if (state.report?.id === airfield.id) button.classList.add("active");
    button.type = "button";
    button.innerHTML = `
      <div>
        <strong>${airfield.code}</strong>
        <span>${airfield.name} · ${airfield.imageCount || 0} images</span>
      </div>
      <span class="status-pill ${getSeverityClass(airfield.status)}">${airfield.status}</span>
    `;
    button.addEventListener("click", () => loadReport(airfield.code));
    watchlist.appendChild(button);
  });
}

/** Deltas vs prior capture for the selected frame (not global latest-vs-prior). */
const ZERO_COMPOSITION_DELTA = Object.freeze({
  commercial: 0,
  cargo: 0,
  business: 0,
  military: 0,
  helicopter: 0,
  unknown: 0,
});

function compositionDeltaBetween(current, previous) {
  const out = {};
  for (const key of Object.keys(ZERO_COMPOSITION_DELTA)) {
    out[key] = (current[key] || 0) - (previous[key] || 0);
  }
  return out;
}

function renderComposition(report) {
  compositionGrid.innerHTML = "";
  Object.entries(report.composition).forEach(([type, count]) => {
    const card = document.createElement("div");
    card.className = "composition-card";
    const delta = report.deltas[type] ?? 0;
    const deltaText = delta > 0 ? `+${delta}` : String(delta);
    card.innerHTML = `
      <span class="type-dot" style="background:${typeColors[type] || typeColors.unknown}"></span>
      <strong>${count}</strong>
      <span>${type}</span>
      <em>${deltaText}</em>
    `;
    compositionGrid.appendChild(card);
  });
}

function renderSignals(report) {
  signalList.innerHTML = "";
  const signals = Array.isArray(report.signals) ? report.signals : [];
  if (!signals.length) {
    signalList.innerHTML = '<div class="empty-copy">No change signals for this view.</div>';
    return;
  }
  signals.forEach((signal) => {
    const row = document.createElement("div");
    row.className = "category-row";
    row.innerHTML = `
      <span>${signal.label}</span>
      <strong class="${getSeverityClass(signal.severity)}-text">${signal.value}</strong>
    `;
    signalList.appendChild(row);
  });
}

function renderFindings(report) {
  findingsList.innerHTML = "";
  report.findings.forEach((finding) => {
    const card = document.createElement("article");
    card.className = "finding-item";
    card.innerHTML = `
      <div class="finding-topline">
        <span class="status-pill ${getSeverityClass(finding.severity)}">${finding.severity}</span>
        <span>${Math.round(finding.confidence * 100)}%</span>
      </div>
      <h3>${finding.title}</h3>
      <p>${finding.explanation}</p>
      <div class="next-step">${finding.nextStep}</div>
    `;
    findingsList.appendChild(card);
  });
}

function renderTimeline(report) {
  timeline.innerHTML = "";
  timelineNote.textContent = `${report.timeline.length} captures`;
  const images = report.images || [];
  const maxTotal = Math.max(
    ...report.timeline.map((point) => Object.keys(report.composition).reduce((sum, key) => sum + (point[key] || 0), 0)),
    1,
  );

  report.timeline.forEach((point, index) => {
    const column = document.createElement("div");
    column.className = "timeline-column";
    const canJump = images.length > 0 && index < images.length;
    if (canJump) {
      column.classList.add("timeline-column--interactive");
      column.setAttribute("role", "button");
      column.tabIndex = 0;
      column.setAttribute(
        "aria-label",
        `Show capture from ${formatCaptureDateForDisplay(point.date)}: ${images[index]?.fileName || `image ${index + 1}`}`,
      );
      if (index === state.captureIndex) {
        column.classList.add("timeline-column--active");
      }
      column.addEventListener("click", () => {
        setCaptureIndex(index).catch(console.error);
      });
      column.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setCaptureIndex(index).catch(console.error);
        }
      });
    }

    const stack = document.createElement("div");
    stack.className = "timeline-stack";

    Object.keys(report.composition).forEach((type) => {
      const value = point[type] || 0;
      if (!value) return;
      const segment = document.createElement("span");
      segment.style.height = `${Math.max(5, (value / maxTotal) * 100)}%`;
      segment.style.background = typeColors[type] || typeColors.unknown;
      stack.appendChild(segment);
    });

    column.appendChild(stack);
    column.insertAdjacentHTML("beforeend", `<span>${formatCaptureDateForDisplay(point.date)}</span>`);
    timeline.appendChild(column);
  });
}

function renderCaptureControls(report) {
  const captures = report.images || [];
  captureSelect.innerHTML = "";

  if (!captures.length) {
    const option = document.createElement("option");
    option.textContent = "Synthetic";
    option.value = "-1";
    captureSelect.appendChild(option);
    captureSelect.disabled = true;
    previousCapture.disabled = true;
    nextCapture.disabled = true;
    refreshAnalysisBtn.disabled = true;
    return;
  }

  captures.forEach((capture, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = formatCaptureDateForDisplay(capture.capturedAt);
    captureSelect.appendChild(option);
  });

  const index = clamp(state.captureIndex, 0, captures.length - 1);
  captureSelect.value = String(index);
  captureSelect.disabled = false;
  previousCapture.disabled = index <= 0;
  nextCapture.disabled = index >= captures.length - 1;
  refreshAnalysisBtn.disabled = false;
}

function renderReportDetails() {
  if (!state.report) return;
  const report = state.report;
  const selectedCapture = getSelectedCapture(report);
  const selectedAnalysis = getSelectedAnalysis(report);
  const selectedComposition = selectedAnalysis ? countComposition(selectedAnalysis.planes) : report.composition;

  const analyses = report.imageAnalyses || [];
  const idx = analyses.length ? clamp(state.captureIndex, 0, analyses.length - 1) : 0;
  let compositionDeltas = report.deltas || { ...ZERO_COMPOSITION_DELTA };
  if (analyses.length && selectedAnalysis) {
    compositionDeltas =
      idx > 0
        ? compositionDeltaBetween(
            selectedComposition,
            countComposition(analyses[idx - 1].planes || []),
          )
        : { ...ZERO_COMPOSITION_DELTA };
  }

  selectedCode.textContent = report.code;
  selectedCode.className = `status-pill ${getSeverityClass(report.riskBand)}`;
  airfieldType.textContent = report.type;
  lastCapture.textContent = selectedCapture?.capturedAt
    ? formatCaptureDateForDisplay(selectedCapture.capturedAt)
    : report.lastCapture;
  baselineSummary.textContent = report.baseline;
  viewerTitle.textContent = `${report.code} - ${report.name}`;
  viewerSubtitle.textContent = selectedCapture
    ? `Local capture ${state.captureIndex + 1} of ${(report.images || []).length} with aircraft overlay`
    : "Synthetic placeholder imagery until local captures are supplied";
  riskBand.textContent = report.status;
  riskBand.className = `status-pill ${getSeverityClass(report.riskBand)}`;
  summaryText.textContent = report.summary;

  renderWatchlist();
  renderComposition({ ...report, composition: selectedComposition, deltas: compositionDeltas });
  renderSignals(report);
  renderFindings(report);
  renderTimeline(report);
  renderCaptureControls(report);
}

function countComposition(planes) {
  const counts = {
    commercial: 0,
    cargo: 0,
    business: 0,
    military: 0,
    helicopter: 0,
    unknown: 0,
  };
  planes.forEach((plane) => {
    const type = plane.classification || plane.type || "unknown";
    counts[type] = (counts[type] || 0) + 1;
  });
  return counts;
}

async function loadSelectedReportImage() {
  const selectedCapture = getSelectedCapture(state.report);
  const nextUrl = selectedCapture?.imageUrl || "";

  if (!nextUrl) {
    state.reportImage = null;
    state.reportImageUrl = "";
    return;
  }

  if (state.reportImageUrl === nextUrl && state.reportImage) return;

  state.reportImage = await loadImageElement(nextUrl);
  state.reportImageUrl = nextUrl;
}

function normalizeAirportCode(airportCode) {
  return String(airportCode || "").trim().toLowerCase();
}

async function loadReport(airportCode, refreshAnalysis = false) {
  const normalizedAirportCode = normalizeAirportCode(airportCode);
  const query = refreshAnalysis ? "?refresh=1" : "";
  const response = await fetch(`/api/airfields/${encodeURIComponent(normalizedAirportCode)}${query}`);
  if (!response.ok) throw new Error(`Unable to load ${normalizedAirportCode}`);
  state.report = await response.json();
  state.reportImage = null;
  state.reportImageUrl = "";
  state.captureIndex = (state.report.images || []).length - 1;
  await loadSelectedReportImage();
  resetViewport();
  renderReportDetails();
  render();
}

async function loadAirfields() {
  const response = await fetch("/api/airfields");
  const payload = await response.json();
  state.airfields = payload.airfields || [];
  renderWatchlist();
  await loadReport("suu");
}

airfieldSearch.addEventListener("input", renderWatchlist);
airfieldSearch.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  const airportCode = normalizeAirportCode(airfieldSearch.value);
  if (airportCode) loadReport(airportCode).catch(console.error);
});
overlayToggle.addEventListener("click", () => {
  state.overlayEnabled = !state.overlayEnabled;
  overlayToggle.classList.toggle("active", state.overlayEnabled);
  render();
});

opacitySlider.addEventListener("input", () => {
  state.overlayOpacity = Number(opacitySlider.value);
  render();
});

resetView.addEventListener("click", resetViewport);

async function setCaptureIndex(index) {
  if (!state.report?.images?.length) return;
  state.captureIndex = clamp(index, 0, state.report.images.length - 1);
  await loadSelectedReportImage();
  renderReportDetails();
  render();
}

previousCapture.addEventListener("click", () => {
  setCaptureIndex(state.captureIndex - 1).catch(console.error);
});

nextCapture.addEventListener("click", () => {
  setCaptureIndex(state.captureIndex + 1).catch(console.error);
});

captureSelect.addEventListener("change", () => {
  setCaptureIndex(Number(captureSelect.value)).catch(console.error);
});

refreshAnalysisBtn.addEventListener("click", () => {
  const code = state.report?.code || normalizeAirportCode(airfieldSearch.value);
  if (!code) return;
  const prevLabel = refreshAnalysisBtn.textContent;
  refreshAnalysisBtn.disabled = true;
  previousCapture.disabled = true;
  nextCapture.disabled = true;
  captureSelect.disabled = true;
  refreshAnalysisBtn.textContent = "Refreshing…";
  loadReport(code, true)
    .catch(console.error)
    .finally(() => {
      refreshAnalysisBtn.textContent = prevLabel;
      renderReportDetails();
    });
});

canvasWrap.addEventListener("pointerdown", (event) => {
  if (!state.report) return;
  state.isDragging = true;
  state.lastPointer = { x: event.clientX, y: event.clientY };
  canvasWrap.classList.add("dragging");
  canvasWrap.setPointerCapture(event.pointerId);
});

canvasWrap.addEventListener("pointermove", (event) => {
  if (!state.isDragging || !state.lastPointer) return;
  state.panX += event.clientX - state.lastPointer.x;
  state.panY += event.clientY - state.lastPointer.y;
  state.lastPointer = { x: event.clientX, y: event.clientY };
  render();
});

canvasWrap.addEventListener("pointerup", (event) => {
  state.isDragging = false;
  state.lastPointer = null;
  canvasWrap.classList.remove("dragging");
  canvasWrap.releasePointerCapture(event.pointerId);
});

canvasWrap.addEventListener("pointercancel", () => {
  state.isDragging = false;
  state.lastPointer = null;
  canvasWrap.classList.remove("dragging");
});

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
loadAirfields().catch((error) => {
  summaryText.textContent = "Unable to load airfield reports.";
  console.error(error);
});
