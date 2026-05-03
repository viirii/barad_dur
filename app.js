const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");
const airfieldSelect = document.getElementById("airfieldSelect");
const boxesToggle = document.getElementById("boxesToggle");
const resetView = document.getElementById("resetView");
const refreshAnalysisBtn = document.getElementById("refreshAnalysis");
const reclassifyAnalysisBtn = document.getElementById("reclassifyAnalysis");
const previousCapture = document.getElementById("previousCapture");
const nextCapture = document.getElementById("nextCapture");
const captureSelect = document.getElementById("captureSelect");
const canvasWrap = document.getElementById("canvasWrap");
const viewerTitle = document.getElementById("viewerTitle");
const viewerSubtitle = document.getElementById("viewerSubtitle");
const selectedCode = document.getElementById("selectedCode");
const airfieldType = document.getElementById("airfieldType");
const lastCapture = document.getElementById("lastCapture");
const summaryAnomalyBadge = document.getElementById("summaryAnomalyBadge");
const summarySeriesMeta = document.getElementById("summarySeriesMeta");
const summaryHeadline = document.getElementById("summaryHeadline");
const summaryText = document.getElementById("summaryText");
const summaryCoherence = document.getElementById("summaryCoherence");
const refreshSummaryBtn = document.getElementById("refreshSummary");
const compositionGrid = document.getElementById("compositionGrid");
const timeline = document.getElementById("timeline");
const timelineNote = document.getElementById("timelineNote");
const timelinePrev = document.getElementById("timelinePrev");
const timelineNext = document.getElementById("timelineNext");
const analysisOverlay = document.getElementById("analysisOverlay");
const zoomSlider = document.getElementById("zoomSlider");
const zoomReadout = document.getElementById("zoomReadout");

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4;

/** Align with server `normalizeClassificationString`: military | civilian | unknown. */
function canonicalPlaneCategory(raw) {
  const v = String(raw ?? "").trim().toLowerCase();
  if (v === "military") return "military";
  if (v === "unknown") return "unknown";
  return "civilian";
}

const COMPOSITION_ORDER = ["civilian", "military", "unknown"];

const COMPOSITION_LABELS = {
  civilian: "Civilian",
  military: "Military",
  unknown: "Unknown",
};

const typeColors = {
  civilian: "#6fb6ff",
  military: "#ff8b72",
  unknown: "#c4b8ff",
};

/** Roll mixed/legacy timeline rows into civilian / military / unknown buckets. */
function aggregateTimelinePoint(point) {
  const acc = { civilian: 0, military: 0, unknown: 0 };
  for (const [key, val] of Object.entries(point)) {
    if (key === "date") continue;
    const n = Number(val) || 0;
    if (!n) continue;
    acc[canonicalPlaneCategory(key)] += n;
  }
  return acc;
}

const state = {
  airfields: [],
  report: null,
  reportImage: null,
  reportImageUrl: "",
  captureIndex: -1,
  selectedAirportCode: "",
  boxesEnabled: true,
  panX: 0,
  panY: 0,
  zoom: 1,
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

/** True when focus is in a control that uses ←/→ for editing or native UI (skip global date stepping). */
function targetUsesHorizontalArrowsForEditing(target) {
  const node = target?.nodeType === Node.TEXT_NODE ? target.parentElement : target;
  if (!node?.closest) return false;
  if (node.isContentEditable) return true;
  const tag = node.tagName;
  if (tag === "TEXTAREA") return true;
  if (tag === "SELECT") return true;
  if (tag === "INPUT") {
    const t = (node.type || "text").toLowerCase();
    return [
      "text",
      "search",
      "password",
      "email",
      "url",
      "tel",
      "number",
      "range",
      "date",
      "time",
      "datetime-local",
      "month",
      "week",
    ].includes(t);
  }
  return false;
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

function syncZoomUi() {
  if (zoomSlider) zoomSlider.value = String(state.zoom);
  if (zoomReadout) zoomReadout.textContent = `${Math.round(state.zoom * 100)}%`;
}

const analysisPendingBackdropWrap = document.getElementById("analysisPendingBackdropWrap");
const analysisPendingBackdrop = document.getElementById("analysisPendingBackdrop");
const analysisOverlayLabel = document.getElementById("analysisOverlayLabel");

function setAnalysisOverlay(visible) {
  if (!analysisOverlay) return;
  analysisOverlay.classList.toggle("is-hidden", !visible);
  analysisOverlay.setAttribute("aria-hidden", visible ? "false" : "true");
}

/** Latest capture as full-area blurred backdrop while analysis runs (`/api/airfields/:code/images` order is oldest→newest). */
function renderAnalysisPendingBackdrop(images) {
  if (analysisOverlayLabel) {
    analysisOverlayLabel.textContent = "Analysis in progress";
  }
  if (!analysisPendingBackdrop || !analysisPendingBackdropWrap) return;
  if (!images?.length) {
    analysisPendingBackdrop.removeAttribute("src");
    analysisPendingBackdrop.alt = "";
    analysisPendingBackdropWrap.classList.add("is-empty");
    return;
  }
  const latest = images[images.length - 1];
  if (!latest?.imageUrl) {
    analysisPendingBackdrop.removeAttribute("src");
    analysisPendingBackdropWrap.classList.add("is-empty");
    return;
  }
  analysisPendingBackdropWrap.classList.remove("is-empty");
  analysisPendingBackdrop.alt = latest.fileName || formatCaptureDateForDisplay(latest.capturedAt) || "";
  analysisPendingBackdrop.onerror = () => {
    analysisPendingBackdrop.removeAttribute("src");
    analysisPendingBackdropWrap.classList.add("is-empty");
  };
  analysisPendingBackdrop.src = latest.imageUrl;
}

function resetViewport() {
  state.panX = 0;
  state.panY = 0;
  state.zoom = 1;
  syncZoomUi();
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
  if (report?.id === "sfo" || report?.id === "oak") {
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
    typeColors[canonicalPlaneCategory(aircraft.classification || aircraft.type)] || typeColors.civilian;
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
  if (!state.boxesEnabled || !report) return;
  const capture = getSelectedCapture(report);
  const analysis = getSelectedAnalysis(report);
  if (!capture?.imageUrl || !analysis) return;

  // Avoid drawing bbox coordinates for capture B on the bitmap for capture A (race during image load or resize).
  if (state.reportImageUrl !== capture.imageUrl) return;

  const aircraft = Array.isArray(analysis.planes) ? analysis.planes : [];
  aircraft.forEach((plane) => {
    drawAircraftMarker(plane, rect, analysis, false);
  });
}

function getBaseSceneDimensions() {
  const width = canvasWrap.clientWidth;
  const height = canvasWrap.clientHeight;
  const activeImage = state.reportImage;
  const iw = activeImage ? activeImage.naturalWidth || activeImage.width : 0;
  const ih = activeImage ? activeImage.naturalHeight || activeImage.height : 0;
  const sceneRatio = iw > 0 && ih > 0 ? iw / ih : 16 / 10;
  const viewportRatio = width / height;
  let baseW = width * 0.88;
  let baseH = baseW / sceneRatio;

  if (viewportRatio > sceneRatio) {
    baseH = height * 0.86;
    baseW = baseH * sceneRatio;
  }

  return { width, height, baseW, baseH };
}

function getSceneRect() {
  const { width, height, baseW, baseH } = getBaseSceneDimensions();
  const z = clamp(state.zoom, MIN_ZOOM, MAX_ZOOM);
  const sceneWidth = baseW * z;
  const sceneHeight = baseH * z;

  return {
    x: (width - sceneWidth) / 2 + state.panX,
    y: (height - sceneHeight) / 2 + state.panY,
    width: sceneWidth,
    height: sceneHeight,
  };
}

/** Zoom in toward a viewport point (e.g. double-click); keeps that spot on the same image fraction. */
function zoomViewTowardCanvasClientPoint(clientX, clientY, multiplier = 1.35) {
  if (!state.report) return;
  const wrap = canvasWrap.getBoundingClientRect();
  const mx = clientX - wrap.left;
  const my = clientY - wrap.top;

  const rect0 = getSceneRect();
  const z0 = clamp(state.zoom, MIN_ZOOM, MAX_ZOOM);
  const z1 = clamp(z0 * multiplier, MIN_ZOOM, MAX_ZOOM);
  if (Math.abs(z1 - z0) < 1e-6) return;

  const u = (mx - rect0.x) / rect0.width;
  const v = (my - rect0.y) / rect0.height;

  const { width, height, baseW, baseH } = getBaseSceneDimensions();
  const w1 = baseW * z1;
  const h1 = baseH * z1;

  state.zoom = z1;
  state.panX = mx - u * w1 - (width - w1) / 2;
  state.panY = my - v * h1 - (height - h1) / 2;

  syncZoomUi();
  render();
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

function syncAirfieldSelect() {
  if (!airfieldSelect) return;
  const code = state.selectedAirportCode;
  if (!code) return;
  const ok = [...airfieldSelect.options].some((o) => o.value === code);
  if (ok) airfieldSelect.value = code;
}

/** Deltas vs prior capture for the selected frame (not global latest-vs-prior). */
const ZERO_COMPOSITION_DELTA = Object.freeze({
  civilian: 0,
  military: 0,
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
  const total = COMPOSITION_ORDER.reduce((sum, type) => sum + (report.composition[type] ?? 0), 0);
  COMPOSITION_ORDER.forEach((type) => {
    const count = report.composition[type] ?? 0;
    const card = document.createElement("div");
    card.className = "composition-card";
    card.dataset.planeCategory = type;
    const delta = report.deltas[type] ?? 0;
    const prevCount = count - delta;
    const deltaText = delta > 0 ? `+${delta}` : String(delta);
    let pctText = "0%";
    if (prevCount > 0) {
      const pct = Math.round((delta / prevCount) * 100);
      pctText = `${pct > 0 ? "+" : ""}${pct}%`;
    } else if (delta > 0) {
      pctText = "+100%";
    } else if (delta < 0) {
      pctText = "-100%";
    }
    const sharePct = total > 0 ? Math.round((count / total) * 100) : 0;
    const shareLine = total > 0 ? `${sharePct}% of ${total}` : "—";
    const dotColor = typeColors[type] || typeColors.civilian;
    const label = COMPOSITION_LABELS[type] || type;
    card.innerHTML = `
      <div class="composition-card-type">
        <span class="type-dot" style="background:${dotColor}" aria-hidden="true"></span>
        <span class="composition-type-label">${label}</span>
      </div>
      <div class="composition-card-count-col">
        <strong class="composition-stat-count">${count}</strong>
        <span class="composition-stat-sublabel">${shareLine}</span>
      </div>
      <div class="composition-card-delta">
        <span class="composition-delta-abs">${deltaText}</span>
        <span class="composition-delta-pct">${pctText}</span>
        <span class="composition-stat-sublabel">vs prior</span>
      </div>
    `;
    compositionGrid.appendChild(card);
  });
}

function renderTimeline(report) {
  timeline.innerHTML = "";
  timelineNote.textContent = `${report.timeline.length} captures`;
  const images = report.images || [];
  const maxTotal = Math.max(
    ...report.timeline.map((point) => {
      const b = aggregateTimelinePoint(point);
      return COMPOSITION_ORDER.reduce((sum, key) => sum + (b[key] || 0), 0);
    }),
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

    const buckets = aggregateTimelinePoint(point);
    COMPOSITION_ORDER.forEach((type) => {
      const value = buckets[type] || 0;
      if (!value) return;
      const segment = document.createElement("span");
      segment.style.height = `${Math.max(5, (value / maxTotal) * 100)}%`;
      segment.style.background = typeColors[type] || typeColors.civilian;
      stack.appendChild(segment);
    });

    column.appendChild(stack);
    column.insertAdjacentHTML("beforeend", `<span>${formatCaptureDateForDisplay(point.date)}</span>`);
    timeline.appendChild(column);
  });

  requestAnimationFrame(() => {
    const active = timeline.querySelector(".timeline-column--active");
    active?.scrollIntoView({ inline: "nearest", block: "nearest", behavior: "smooth" });
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
    if (reclassifyAnalysisBtn) reclassifyAnalysisBtn.disabled = true;
    if (refreshSummaryBtn) refreshSummaryBtn.disabled = true;
    if (timelinePrev) timelinePrev.disabled = true;
    if (timelineNext) timelineNext.disabled = true;
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
  if (reclassifyAnalysisBtn) reclassifyAnalysisBtn.disabled = false;
  if (refreshSummaryBtn) refreshSummaryBtn.disabled = false;
  if (timelinePrev) timelinePrev.disabled = previousCapture.disabled;
  if (timelineNext) timelineNext.disabled = nextCapture.disabled;
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
  viewerTitle.textContent = `${report.code} - ${report.name}`;
  viewerSubtitle.textContent = selectedCapture
    ? `Local capture ${state.captureIndex + 1} of ${(report.images || []).length} with aircraft boxes`
    : "Synthetic placeholder imagery until local captures are supplied";
  const ss = report.seriesSummary;
  if (summaryAnomalyBadge) {
    const showAnomaly = ss?.anomalous === true;
    summaryAnomalyBadge.classList.toggle("is-hidden", !showAnomaly);
  }

  const unexpected = Array.isArray(ss?.unexpectedObservations) ? ss.unexpectedObservations : [];
  if (ss && (ss.narrative || ss.headline || unexpected.length)) {
    if (summaryHeadline) {
      summaryHeadline.textContent = ss.headline ? String(ss.headline) : "";
      summaryHeadline.classList.toggle("is-hidden", !ss.headline);
    }
    summaryText.textContent =
      ss.narrative != null && String(ss.narrative).trim() !== ""
        ? String(ss.narrative)
        : report.summary;
    summaryText.classList.add("summary-text--primary");
    if (summaryCoherence) {
      summaryCoherence.textContent = "";
      if (unexpected.length) {
        const ul = document.createElement("ul");
        ul.className = "summary-unexpected-list";
        unexpected.forEach((item) => {
          const title = item?.title != null ? String(item.title).trim() : "";
          const explanation = item?.explanation != null ? String(item.explanation).trim() : "";
          if (!title && !explanation) return;
          const li = document.createElement("li");
          const sev = ["high", "medium", "low"].includes(item?.severity) ? item.severity : "low";
          const strong = document.createElement("strong");
          strong.className = `summary-unexpected-title ${getSeverityClass(sev)}-text`;
          strong.textContent = title || "Note";
          li.appendChild(strong);
          li.appendChild(document.createTextNode(` ${explanation}`));
          ul.appendChild(li);
        });
        summaryCoherence.appendChild(ul);
        summaryCoherence.classList.toggle("is-hidden", ul.childElementCount === 0);
      } else {
        summaryCoherence.classList.add("is-hidden");
      }
    }
    if (summarySeriesMeta && (report.images || []).length > 0) {
      summarySeriesMeta.textContent = ss.summaryFromCache ? "Summary from cache" : "";
    }
  } else {
    if (summaryHeadline) summaryHeadline.classList.add("is-hidden");
    summaryText.textContent = report.summary;
    summaryText.classList.remove("summary-text--primary");
    if (summaryCoherence) {
      summaryCoherence.textContent = "";
      summaryCoherence.classList.add("is-hidden");
    }
    if (summarySeriesMeta) summarySeriesMeta.textContent = "";
  }

  syncAirfieldSelect();
  renderComposition({ ...report, composition: selectedComposition, deltas: compositionDeltas });
  renderTimeline(report);
  renderCaptureControls(report);
}

function countComposition(planes) {
  const counts = {
    civilian: 0,
    military: 0,
    unknown: 0,
  };
  planes.forEach((plane) => {
    const cat = canonicalPlaneCategory(plane.classification || plane.type);
    counts[cat] = (counts[cat] || 0) + 1;
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

async function loadReport(airportCode, opts = {}) {
  let refreshAnalysis = false;
  let refreshSummary = false;
  if (opts === true) {
    refreshAnalysis = true;
  } else if (opts && typeof opts === "object") {
    refreshAnalysis = opts.refreshAnalysis === true;
    refreshSummary = opts.refreshSummary === true;
  }

  const normalizedAirportCode = normalizeAirportCode(airportCode);
  state.selectedAirportCode = normalizedAirportCode;
  syncAirfieldSelect();
  renderAnalysisPendingBackdrop(null);
  setAnalysisOverlay(true);
  try {
    const imagesPromise = fetch(
      `/api/airfields/${encodeURIComponent(normalizedAirportCode)}/images`,
    ).then((r) => (r.ok ? r.json() : null));
    const listPayload = await imagesPromise;
    if (listPayload?.images?.length) {
      renderAnalysisPendingBackdrop(listPayload.images);
    }
  } catch {
    /* overlay without backdrop */
  }
  try {
    const params = new URLSearchParams();
    if (refreshAnalysis) params.set("refresh", "1");
    if (refreshSummary) params.set("summaryRefresh", "1");
    const query = params.toString() ? `?${params.toString()}` : "";
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
  } finally {
    renderAnalysisPendingBackdrop(null);
    setAnalysisOverlay(false);
  }
}

async function loadAirfields() {
  const response = await fetch("/api/airfields");
  const payload = await response.json();
  state.airfields = payload.airfields || [];
  if (airfieldSelect) airfieldSelect.value = "oak";
  await loadReport("oak");
}

if (airfieldSelect) {
  airfieldSelect.addEventListener("change", () => {
    const airportCode = normalizeAirportCode(airfieldSelect.value);
    if (airportCode) loadReport(airportCode).catch(console.error);
  });
}
if (boxesToggle) {
  boxesToggle.addEventListener("click", () => {
    state.boxesEnabled = !state.boxesEnabled;
    boxesToggle.classList.toggle("active", state.boxesEnabled);
    render();
  });
}

resetView.addEventListener("click", resetViewport);

if (zoomSlider) {
  zoomSlider.addEventListener("input", () => {
    state.zoom = clamp(Number(zoomSlider.value), MIN_ZOOM, MAX_ZOOM);
    syncZoomUi();
    render();
  });
}

canvasWrap.addEventListener(
  "wheel",
  (event) => {
    if (!state.report) return;
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.06 : 0.06;
    state.zoom = clamp(state.zoom + delta, MIN_ZOOM, MAX_ZOOM);
    syncZoomUi();
    render();
  },
  { passive: false },
);

canvasWrap.addEventListener("dblclick", (event) => {
  if (!state.report) return;
  if (event.button !== 0) return;
  event.preventDefault();
  zoomViewTowardCanvasClientPoint(event.clientX, event.clientY, 1.35);
});

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

if (timelinePrev) {
  timelinePrev.addEventListener("click", () => {
    setCaptureIndex(state.captureIndex - 1).catch(console.error);
  });
}
if (timelineNext) {
  timelineNext.addEventListener("click", () => {
    setCaptureIndex(state.captureIndex + 1).catch(console.error);
  });
}

/** Capture phase so timeline horizontal scroll default loses to date stepping when applicable. */
window.addEventListener(
  "keydown",
  (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    if (!state.report?.images?.length) return;
    if (targetUsesHorizontalArrowsForEditing(event.target)) return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? -1 : 1;
    setCaptureIndex(state.captureIndex + delta).catch(console.error);
  },
  true,
);

refreshAnalysisBtn.addEventListener("click", () => {
  const code = state.report?.code || normalizeAirportCode(airfieldSelect?.value);
  if (!code) return;
  const prevLabel = refreshAnalysisBtn.textContent;
  refreshAnalysisBtn.disabled = true;
  if (reclassifyAnalysisBtn) reclassifyAnalysisBtn.disabled = true;
  if (refreshSummaryBtn) refreshSummaryBtn.disabled = true;
  previousCapture.disabled = true;
  nextCapture.disabled = true;
  captureSelect.disabled = true;
  if (timelinePrev) timelinePrev.disabled = true;
  if (timelineNext) timelineNext.disabled = true;
  refreshAnalysisBtn.textContent = "Refreshing…";
  loadReport(code, { refreshAnalysis: true })
    .catch(console.error)
    .finally(() => {
      refreshAnalysisBtn.textContent = prevLabel;
      renderReportDetails();
    });
});

if (refreshSummaryBtn) {
  refreshSummaryBtn.addEventListener("click", () => {
    const code = state.report?.code || normalizeAirportCode(airfieldSelect?.value);
    if (!code) return;
    const prevLabel = refreshSummaryBtn.textContent;
    refreshSummaryBtn.disabled = true;
    refreshAnalysisBtn.disabled = true;
    if (reclassifyAnalysisBtn) reclassifyAnalysisBtn.disabled = true;
    previousCapture.disabled = true;
    nextCapture.disabled = true;
    captureSelect.disabled = true;
    if (timelinePrev) timelinePrev.disabled = true;
    if (timelineNext) timelineNext.disabled = true;
    refreshSummaryBtn.textContent = "Summarizing…";
    loadReport(code, { refreshSummary: true })
      .catch(console.error)
      .finally(() => {
        refreshSummaryBtn.textContent = prevLabel;
        renderReportDetails();
      });
  });
}

if (reclassifyAnalysisBtn) {
  reclassifyAnalysisBtn.addEventListener("click", () => {
    const code = state.report?.code || normalizeAirportCode(airfieldSelect?.value);
    if (!code) return;
    const prevLabel = reclassifyAnalysisBtn.textContent;
    refreshAnalysisBtn.disabled = true;
    reclassifyAnalysisBtn.disabled = true;
    if (refreshSummaryBtn) refreshSummaryBtn.disabled = true;
    previousCapture.disabled = true;
    nextCapture.disabled = true;
    captureSelect.disabled = true;
    if (timelinePrev) timelinePrev.disabled = true;
    if (timelineNext) timelineNext.disabled = true;
    reclassifyAnalysisBtn.textContent = "Reclassifying…";
    fetch(`/api/airfields/${encodeURIComponent(code)}/reclassify`, { method: "POST" })
      .then(async (r) => {
        const payload = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(payload.error || r.statusText || "Reclassify failed");
        return payload;
      })
      .then((payload) => {
        if (payload.skipped?.length) {
          console.warn("[reclassify] skipped images:", payload.skipped);
        }
        return loadReport(code);
      })
      .catch(console.error)
      .finally(() => {
        reclassifyAnalysisBtn.textContent = prevLabel;
        renderReportDetails();
      });
  });
}

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
syncZoomUi();
loadAirfields().catch((error) => {
  summaryText.textContent = "Unable to load airfield reports.";
  console.error(error);
});
