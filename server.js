const fs = require("fs");
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });

const http = require("http");
const crypto = require("crypto");
const { spawnSync } = require("child_process");

const cliArgs = process.argv.slice(2);
/**
 * When true, skip reading **legacy** per-image cache only (`writeCachedAnalysis` for HF/scaffold);
 * does **not** bypass OpenAI path cache — OpenAI calls only happen when path cache is missing or `?refresh=1`.
 * Set via `--no-analysis-cache` or `ANALYSIS_CACHE=0`.
 */
const analysisCacheReadDisabled =
  cliArgs.includes("--no-analysis-cache") || process.env.ANALYSIS_CACHE === "0";

function isAnalysisCacheReadEnabled() {
  return !analysisCacheReadDisabled;
}

const host = process.env.HOST || "127.0.0.1";
const port = Number(process.env.PORT || 3000);
const root = __dirname;
const airfieldImageRoot = path.join(root, "data", "airfields");
const analysisCacheRoot = path.join(root, "data", "analysis-cache");
const openaiApiKey = process.env.OPENAI_API_KEY || "";
const openaiModel = process.env.OPENAI_MODEL || "gpt-5";

/**
 * Max concurrent `analyze_image` runs per airfield load (order preserved).
 * Does not bypass cache: each call still reads path/legacy cache first; `refreshAnalysis` still forces recompute.
 * Concurrent misses share one OpenAI request per file via `openaiInflightByAbsPath`.
 * Default 4; set `IMAGE_ANALYSIS_CONCURRENCY=1` if HF/YOLO runs spike VRAM.
 */
function getImageAnalysisConcurrency() {
  const raw = process.env.IMAGE_ANALYSIS_CONCURRENCY;
  const n = raw == null || String(raw).trim() === "" ? 4 : Number(raw);
  if (!Number.isFinite(n) || n < 1) return 1;
  return Math.min(32, Math.floor(n));
}

/** Run async tasks in parallel with a fixed pool size; results stay in input order. */
async function mapWithConcurrency(items, concurrency, fn) {
  if (items.length === 0) return [];
  const limit = Math.min(Math.max(1, concurrency), items.length);
  const results = new Array(items.length);
  let next = 0;

  async function worker() {
    for (;;) {
      const i = next;
      next += 1;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }

  await Promise.all(Array.from({ length: limit }, () => worker()));
  return results;
}

/** Aircraft pipeline: `hf` (local YOLO), `openai` (vision API, needs OPENAI_API_KEY), or `scaffold` (deterministic fake). */
function getAircraftAnalysisProvider() {
  const raw = (process.env.AIRCRAFT_ANALYSIS_PROVIDER || "hf").trim().toLowerCase();
  if (raw === "openai" || raw === "hf" || raw === "scaffold") return raw;
  return "hf";
}

/** Weights file shown in API/cache when unset (HF quick mode uses Ultralytics COCO nano). */
function getHfWeightsLabelForAnalysis() {
  return (
    process.env.HF_AIRCRAFT_WEIGHTS ||
    (process.env.HF_AIRCRAFT_QUICK === "1" ? "yolov8n.pt" : "training/experiment-61/best.pt")
  );
}

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
};

const airfieldReports = {
  sfo: {
    id: "sfo",
    code: "SFO",
    name: "San Francisco International Airport",
    type: "Commercial airport",
    status: "Normal",
    riskBand: "normal",
    lastCapture: "2026-05-01 18:20 UTC",
    baseline: "commercial-heavy, cargo activity expected",
    summary:
      "SFO shows high aircraft density, but the latest visible composition is consistent with a commercial airport baseline. No military-style concentration is flagged in this placeholder report.",
    composition: {
      commercial: 48,
      cargo: 7,
      business: 4,
      military: 0,
      helicopter: 0,
      unknown: 2,
    },
    deltas: {
      commercial: 3,
      cargo: 1,
      business: -1,
      military: 0,
      helicopter: 0,
      unknown: 1,
    },
    timeline: [
      { date: "Apr 18", commercial: 43, cargo: 6, business: 5, military: 0, unknown: 1 },
      { date: "Apr 22", commercial: 46, cargo: 5, business: 4, military: 0, unknown: 1 },
      { date: "Apr 26", commercial: 45, cargo: 6, business: 5, military: 0, unknown: 1 },
      { date: "May 01", commercial: 48, cargo: 7, business: 4, military: 0, unknown: 2 },
    ],
    signals: [
      { label: "Aircraft count", value: "Within baseline", severity: "low" },
      { label: "Type mix", value: "Commercial dominant", severity: "low" },
      { label: "Military-style aircraft", value: "None detected", severity: "low" },
      { label: "Uncertain objects", value: "2 low-confidence regions", severity: "medium" },
    ],
    findings: [
      {
        title: "Commercial concentration expected",
        severity: "low",
        confidence: 0.86,
        explanation:
          "Most aircraft-like objects are clustered near terminal gates and cargo aprons, matching expected SFO activity.",
        nextStep: "Use flight data or adjacent timestamps only if a specific aircraft claim needs corroboration.",
      },
      {
        title: "Two low-confidence aircraft-like shapes",
        severity: "medium",
        confidence: 0.46,
        explanation:
          "Two small objects near service areas are ambiguous at this resolution and should not drive conclusions.",
        nextStep: "Compare against a sharper capture or alternate collection angle.",
      },
    ],
    aircraft: [
      { x: 0.18, y: 0.24, type: "commercial", confidence: 0.88, angle: -8 },
      { x: 0.25, y: 0.28, type: "commercial", confidence: 0.91, angle: -8 },
      { x: 0.32, y: 0.32, type: "commercial", confidence: 0.9, angle: -8 },
      { x: 0.48, y: 0.41, type: "cargo", confidence: 0.78, angle: 6 },
      { x: 0.57, y: 0.44, type: "cargo", confidence: 0.75, angle: 6 },
      { x: 0.68, y: 0.55, type: "business", confidence: 0.66, angle: 12 },
      { x: 0.76, y: 0.64, type: "unknown", confidence: 0.42, angle: 12 },
    ],
  },
  suu: {
    id: "suu",
    code: "SUU",
    name: "Travis Air Force Base",
    type: "Military airfield",
    status: "Needs review",
    riskBand: "needs_review",
    lastCapture: "2026-05-01 19:05 UTC",
    baseline: "large military transport/tanker aircraft expected",
    summary:
      "Travis AFB contains expected military aircraft, but the latest placeholder report shows a notable concentration increase near the eastern apron compared with prior captures.",
    composition: {
      commercial: 0,
      cargo: 2,
      business: 0,
      military: 18,
      helicopter: 1,
      unknown: 3,
    },
    deltas: {
      commercial: 0,
      cargo: 0,
      business: 0,
      military: 9,
      helicopter: 1,
      unknown: 2,
    },
    timeline: [
      { date: "Apr 18", commercial: 0, cargo: 2, business: 0, military: 8, unknown: 1 },
      { date: "Apr 22", commercial: 0, cargo: 2, business: 0, military: 10, unknown: 1 },
      { date: "Apr 26", commercial: 0, cargo: 2, business: 0, military: 9, unknown: 1 },
      { date: "May 01", commercial: 0, cargo: 2, business: 0, military: 18, unknown: 3 },
    ],
    signals: [
      { label: "Aircraft count", value: "+12 vs prior capture", severity: "high" },
      { label: "Type mix", value: "Military dominant", severity: "low" },
      { label: "Concentration", value: "Eastern apron buildup", severity: "high" },
      { label: "Decoy / painted risk", value: "1 ambiguous flat-looking region", severity: "medium" },
    ],
    findings: [
      {
        title: "Military aircraft concentration increased",
        severity: "high",
        confidence: 0.74,
        explanation:
          "The latest capture shows a sharp increase of military-style aircraft-like objects on the eastern apron relative to the rolling baseline.",
        nextStep: "Compare with adjacent imagery and known exercise, maintenance, or weather displacement context.",
      },
      {
        title: "Possible decoy or flat aircraft-like marking",
        severity: "medium",
        confidence: 0.53,
        explanation:
          "One aircraft-like shape has weaker shadow contact and flatter texture than nearby aircraft-like objects.",
        nextStep: "Review a different sun angle or higher-resolution capture before treating it as a physical aircraft.",
      },
      {
        title: "Military aircraft expected at this airfield",
        severity: "low",
        confidence: 0.91,
        explanation:
          "The presence of military transport/tanker-style aircraft is consistent with Travis AFB baseline; the anomaly is the concentration change, not the type alone.",
        nextStep: "Prioritize count, parking-zone shift, and temporal behavior over one-off object presence.",
      },
    ],
    aircraft: [
      { x: 0.21, y: 0.27, type: "military", confidence: 0.82, angle: -12 },
      { x: 0.28, y: 0.31, type: "military", confidence: 0.84, angle: -12 },
      { x: 0.35, y: 0.35, type: "military", confidence: 0.8, angle: -12 },
      { x: 0.47, y: 0.42, type: "military", confidence: 0.77, angle: 8 },
      { x: 0.55, y: 0.46, type: "military", confidence: 0.79, angle: 8 },
      { x: 0.63, y: 0.51, type: "military", confidence: 0.76, angle: 8 },
      { x: 0.71, y: 0.58, type: "unknown", confidence: 0.44, angle: 12 },
      { x: 0.77, y: 0.63, type: "unknown", confidence: 0.39, angle: 12 },
    ],
  },
};

function isImageFile(fileName) {
  return [".png", ".jpg", ".jpeg", ".webp"].includes(path.extname(fileName).toLowerCase());
}

/** Normalize filename/manifest dates to YYYY-MM-DD (e.g. suu/20260330.png → 2026-03-30). */
function normalizeCaptureDateInput(raw) {
  if (raw == null || raw === "") return "";
  const s = String(raw).trim();
  if (/^capture\s+\d+/i.test(s)) return s;
  const compact = s.match(/^(\d{4})(\d{2})(\d{2})$/);
  if (compact) return `${compact[1]}-${compact[2]}-${compact[3]}`;
  const separated = s.match(/^(\d{4})[-_/](\d{2})[-_/](\d{2})$/);
  if (separated) return `${separated[1]}-${separated[2]}-${separated[3]}`;
  return s;
}

function toCaptureDateFromFileName(fileName) {
  const match = fileName.match(/(\d{4})[-_]?(\d{2})[-_]?(\d{2})/);
  if (!match) return "";
  return `${match[1]}-${match[2]}-${match[3]}`;
}

function normalizeAirportCode(airportCode) {
  return String(airportCode || "").trim().toLowerCase();
}

function toPublicImagePath(airportCode, fileName) {
  return `/imagery/${encodeURIComponent(normalizeAirportCode(airportCode))}/${encodeURIComponent(fileName)}`;
}

function normalizeCapture(airportCode, capture, index) {
  const normalizedAirportCode = normalizeAirportCode(airportCode);
  const fileName = capture.file || capture.fileName || capture.path;
  if (!fileName || path.basename(fileName) !== fileName || !isImageFile(fileName)) {
    return null;
  }

  const fromManifest = normalizeCaptureDateInput(capture.capturedAt || capture.date || "");
  const fromFile = toCaptureDateFromFileName(fileName);

  return {
    id: capture.id || `${normalizedAirportCode}-${index + 1}`,
    capturedAt: fromManifest || fromFile || `Capture ${index + 1}`,
    source: capture.source || "local",
    cloudCover: capture.cloudCover ?? null,
    notes: capture.notes || "",
    fileName,
    imageUrl: toPublicImagePath(normalizedAirportCode, fileName),
  };
}

function readManifestCaptures(airportCode, airportDir) {
  const manifestPath = path.join(airportDir, "manifest.json");
  if (!fs.existsSync(manifestPath)) return null;

  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const captures = Array.isArray(manifest.captures) ? manifest.captures : [];
  return captures
    .map((capture, index) => normalizeCapture(airportCode, capture, index))
    .filter(Boolean)
    .filter((capture) => fs.existsSync(path.join(airportDir, capture.fileName)));
}

function readDirectoryCaptures(airportCode, airportDir) {
  if (!fs.existsSync(airportDir)) return [];

  return fs
    .readdirSync(airportDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && isImageFile(entry.name))
    .map((entry, index) =>
      normalizeCapture(
        airportCode,
        {
          file: entry.name,
          capturedAt: toCaptureDateFromFileName(entry.name),
        },
        index,
      ),
    )
    .filter(Boolean);
}

function getCaptureFilePath(airportCode, image) {
  const normalizedAirportCode = normalizeAirportCode(airportCode);
  const imagePath = path.join(airfieldImageRoot, normalizedAirportCode, image.fileName);
  if (!imagePath.startsWith(airfieldImageRoot)) return "";
  return imagePath;
}

/** Canonical path for HF JSON keys (matches Python os.path.realpath). */
function resolveRealImagePath(absPath) {
  if (!absPath) return "";
  try {
    return fs.realpathSync(absPath);
  } catch {
    return path.resolve(absPath);
  }
}

function sanitizeCacheTag(value, fallback) {
  const raw = value ?? fallback;
  return String(raw)
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/_+/g, "_")
    .slice(0, 96);
}

/** Namespaces disk cache; must change when inference inputs change (otherwise env tweaks are ignored). */
function getAnalysisProviderKey() {
  const mode = getAircraftAnalysisProvider();
  if (mode === "openai") {
    const modelTag = sanitizeCacheTag(openaiModel, "gpt");
    /** Bump when bbox JSON contract changes (e.g. pixels → normalized) so disk cache is not reused across incompatible formats. */
    return `openai-m-${modelTag}-bbox01`;
  }
  if (mode === "scaffold") {
    return "scaffold-v1";
  }
  const quickMode = process.env.HF_AIRCRAFT_QUICK === "1";
  const weightsRaw =
    process.env.HF_AIRCRAFT_WEIGHTS ||
    (quickMode ? "yolov8n.pt" : "training/experiment-61/best.pt");
  const confRaw =
    process.env.HF_AIRCRAFT_CONF ?? (quickMode ? "0.15" : "0.001");
  const imgszRaw =
    process.env.HF_AIRCRAFT_IMGSZ ?? (quickMode ? "1280" : "native");
  const imgszMaxRaw =
    process.env.HF_AIRCRAFT_IMGSZ_MAX ?? (quickMode ? "4096" : "16384");
  const tileRaw = process.env.HF_AIRCRAFT_TILE ?? "1";
  const overlapRaw =
    process.env.HF_AIRCRAFT_TILE_OVERLAP ?? (quickMode ? "0.25" : "0.40");
  const augmentRaw = process.env.HF_AIRCRAFT_AUGMENT ?? "0";
  const iouRaw =
    process.env.HF_AIRCRAFT_IOU ?? (quickMode ? "0.5" : "0.45");
  const maxDetRaw =
    process.env.HF_AIRCRAFT_MAX_DET ?? (quickMode ? "300" : "500");
  const quickTag = sanitizeCacheTag(process.env.HF_AIRCRAFT_QUICK ?? "0", "0");
  const weightsTag = sanitizeCacheTag(
    weightsRaw,
    quickMode ? "yolov8n.pt" : "training/experiment/61/best.pt",
  );
  const confTag = sanitizeCacheTag(confRaw, quickMode ? "0.15" : "0.001");
  const imgszTag = sanitizeCacheTag(imgszRaw, quickMode ? "1280" : "native");
  const imgszMaxTag = sanitizeCacheTag(imgszMaxRaw, quickMode ? "4096" : "16384");
  const tileTag = sanitizeCacheTag(tileRaw, "1");
  const overlapTag = sanitizeCacheTag(overlapRaw, quickMode ? "0.25" : "0.40");
  const augmentTag = sanitizeCacheTag(augmentRaw, "0");
  const iouTag = sanitizeCacheTag(iouRaw, quickMode ? "0.5" : "0.45");
  const maxDetTag = sanitizeCacheTag(maxDetRaw, quickMode ? "300" : "500");
  /** Quick mode = COCO yolov8n; bump when quick defaults change. */
  return (
    `hf-yolo-w-${weightsTag}-c-${confTag}-sz-${imgszTag}-mx-${imgszMaxTag}-t-${tileTag}-o-${overlapTag}-aug-${augmentTag}-iou-${iouTag}-md-${maxDetTag}-q-${quickTag}-quick-v1-stub-commercial`
  );
}

function getAnalysisCachePath(airportCode, image) {
  const normalizedAirportCode = normalizeAirportCode(airportCode);
  const cacheDir = path.join(analysisCacheRoot, getAnalysisProviderKey(), normalizedAirportCode);
  if (!cacheDir.startsWith(analysisCacheRoot)) return "";
  return path.join(cacheDir, `${image.id}.json`);
}

/** Stable OpenAI cache filename: SHA-256(hex) of the resolved absolute path string (path-addressed, not file-content hash). */
function hashResolvedImagePath(absResolvedPath) {
  return crypto.createHash("sha256").update(absResolvedPath, "utf8").digest("hex");
}

function getOpenaiPathCachePath(absResolvedPath) {
  if (!absResolvedPath) return "";
  const canonical = resolveRealImagePath(absResolvedPath);
  const h = hashResolvedImagePath(canonical);
  const dir = path.join(analysisCacheRoot, getAnalysisProviderKey(), "by-path");
  if (!dir.startsWith(analysisCacheRoot)) return "";
  return path.join(dir, `${h}.json`);
}

function mergeCachedAnalysisWithImage(cached, image) {
  return {
    ...cached,
    imageId: image.id,
    capturedAt: image.capturedAt,
  };
}

function readOpenaiPathCache(absResolvedPath) {
  const cachePath = getOpenaiPathCachePath(absResolvedPath);
  if (!cachePath || !fs.existsSync(cachePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(cachePath, "utf8"));
  } catch {
    return null;
  }
}

function writeOpenaiPathCache(absResolvedPath, analysis) {
  const cachePath = getOpenaiPathCachePath(absResolvedPath);
  if (!cachePath) return;
  const payload = {
    ...analysis,
    openaiCacheSourcePath: resolveRealImagePath(absResolvedPath),
    openaiCachedAt: new Date().toISOString(),
  };
  fs.mkdirSync(path.dirname(cachePath), { recursive: true });
  fs.writeFileSync(cachePath, JSON.stringify(payload, null, 2));
}

function readCachedAnalysis(airportCode, image) {
  const cachePath = getAnalysisCachePath(airportCode, image);
  if (!cachePath || !fs.existsSync(cachePath)) return null;

  try {
    return JSON.parse(fs.readFileSync(cachePath, "utf8"));
  } catch {
    return null;
  }
}

function writeCachedAnalysis(airportCode, image, analysis) {
  const cachePath = getAnalysisCachePath(airportCode, image);
  if (!cachePath) return;

  fs.mkdirSync(path.dirname(cachePath), { recursive: true });
  fs.writeFileSync(cachePath, JSON.stringify(analysis, null, 2));
}

function get_images_over_time(airportCode) {
  const normalizedAirportCode = normalizeAirportCode(airportCode);
  const airportDir = path.join(airfieldImageRoot, normalizedAirportCode);

  if (!airportDir.startsWith(airfieldImageRoot)) return [];

  const manifestCaptures = fs.existsSync(airportDir) ? readManifestCaptures(normalizedAirportCode, airportDir) : null;
  const captures = manifestCaptures ?? readDirectoryCaptures(normalizedAirportCode, airportDir);

  return captures.sort((a, b) => String(a.capturedAt).localeCompare(String(b.capturedAt)));
}

function hashString(value) {
  return String(value)
    .split("")
    .reduce((hash, char) => (hash * 31 + char.charCodeAt(0)) >>> 0, 7);
}

function identify_planes_scaffold(image, airportCode) {
  const seed = hashString(`${airportCode}:${image.id}:${image.capturedAt}`);
  const count = airportCode === "suu" ? 4 + (seed % 7) : 8 + (seed % 10);

  return Array.from({ length: count }, (_, index) => {
    const row = Math.floor(index / 4);
    const column = index % 4;
    const jitterX = ((seed + index * 17) % 13) / 100;
    const jitterY = ((seed + index * 29) % 11) / 100;
    const x = Math.min(0.9, 0.16 + column * 0.18 + jitterX);
    const y = Math.min(0.86, 0.22 + row * 0.16 + jitterY);
    const width = airportCode === "suu" ? 0.065 : 0.08;
    const height = airportCode === "suu" ? 0.038 : 0.046;

    return {
      id: `${image.id}-plane-${index + 1}`,
      bbox: { x, y, width, height },
      angle: ((seed + index * 23) % 50) - 25,
      detectionConfidence: 0.52 + (((seed + index * 19) % 38) / 100),
    };
  });
}

function classify_planes_scaffold(planeDetections, image, airportCode) {
  return planeDetections.map((plane, index) => {
    let classification = "unknown";
    if (airportCode === "suu") {
      classification = index % 6 === 5 ? "unknown" : "military";
    } else if (index % 8 === 6) {
      classification = "cargo";
    } else if (index % 8 === 7) {
      classification = "business";
    } else {
      classification = "commercial";
    }

    return {
      ...plane,
      classification,
      type: classification,
      classificationConfidence: classification === "unknown" ? 0.44 : 0.64 + ((index % 4) * 0.07),
      source: "scaffold",
    };
  });
}

/** Temporary pass-2: all detections labeled commercial until a real classifier is wired. */
function classify_planes_stub_commercial(planeDetections) {
  return planeDetections.map((plane, index) => ({
    ...plane,
    classification: "commercial",
    type: "commercial",
    classificationConfidence: Math.max(
      0.35,
      Math.min(1, plane.detectionConfidence != null ? plane.detectionConfidence : 0.75),
    ),
    realness: "uncertain",
    realnessConfidence: 0.5,
    rationale: "Stub classification: HF detection only; types not inferred yet.",
    source: "stub_commercial",
  }));
}

function getHfDetectScriptPath() {
  const scriptPath = path.join(root, "scripts", "hf_aircraft_detect.py");
  return fs.existsSync(scriptPath) ? scriptPath : "";
}

function isHfDetectionEnabled() {
  return process.env.HF_AIRCRAFT_DISABLE !== "1";
}

function resolvePythonCommand() {
  const venvPy =
    process.platform === "win32"
      ? path.join(root, ".venv", "Scripts", "python.exe")
      : path.join(root, ".venv", "bin", "python3");
  if (fs.existsSync(venvPy)) return venvPy;
  return process.env.PYTHON || "python3";
}

function runHfDetector(absPaths) {
  const scriptPath = getHfDetectScriptPath();
  if (!scriptPath || !absPaths.length) return null;

  const t0 = Date.now();
  const label =
    absPaths.length <= 2
      ? absPaths.map((p) => path.basename(p)).join(", ")
      : `${absPaths.length} images (e.g. ${path.basename(absPaths[0])} …)`;
  console.log(`[hf:inference] start ${label}`);

  const pythonBin = resolvePythonCommand();
  const timeoutMs = Number(process.env.HF_DETECT_TIMEOUT_MS || 600000);
  const detectorEnv = { ...process.env };
  // PyTorch 2.6+: TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 overrides even weights_only=False inside torch.load()
  // and breaks Ultralytics YOLO .pt checkpoints. Detector subprocess should not inherit that.
  delete detectorEnv.TORCH_FORCE_WEIGHTS_ONLY_LOAD;
  delete detectorEnv.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD;

  const mplDir = path.join(root, ".matplotlib-cache");
  fs.mkdirSync(mplDir, { recursive: true });
  detectorEnv.MPLCONFIGDIR = mplDir;

  const result = spawnSync(pythonBin, [scriptPath, ...absPaths], {
    encoding: "utf8",
    maxBuffer: 50 * 1024 * 1024,
    env: detectorEnv,
    timeout: timeoutMs,
  });

  if (result.error) {
    console.warn(
      `[hf:inference] done ms=${Date.now() - t0} failed spawn: ${result.error.message}`,
    );
    console.warn("HF aircraft detector spawn error:", result.error.message);
    return null;
  }
  if (result.status !== 0) {
    console.warn(
      `[hf:inference] done ms=${Date.now() - t0} exit=${result.status}`,
    );
    console.warn("HF aircraft detector exit:", result.status, (result.stderr || "").slice(0, 2400));
    return null;
  }

  let parsed;
  try {
    parsed = JSON.parse((result.stdout || "").trim());
  } catch (e) {
    const head = (result.stdout || "").slice(0, 320).replace(/\s+/g, " ");
    console.warn(`[hf:inference] done ms=${Date.now() - t0} invalid JSON`);
    console.warn("HF aircraft detector invalid JSON:", e.message, "stdout.head=", head);
    return null;
  }

  const stderrTail = (result.stderr || "").trim();
  if (stderrTail) {
    const cap = 12000;
    const slice = stderrTail.length > cap ? `${stderrTail.slice(0, cap)}…` : stderrTail;
    console.warn(`[hf:inference] python stderr (${stderrTail.length} chars):\n${slice}`);
  }

  const map = new Map();
  let detectionCount = 0;
  for (const [key, value] of Object.entries(parsed)) {
    const resolved = resolveRealImagePath(key);
    const list = Array.isArray(value?.detections) ? value.detections : [];
    detectionCount += list.length;
    map.set(resolved, list);
  }
  console.log(
    `[hf:inference] done ms=${Date.now() - t0} images=${absPaths.length} detections=${detectionCount}`,
  );
  return map;
}

function getMimeType(fileName) {
  return mimeTypes[path.extname(fileName).toLowerCase()]?.split(";")[0] || "image/png";
}

/** Read PNG/JPEG header only — align bbox interpretation with intrinsic pixel grid. */
function readPngDimensions(buf) {
  if (buf.length < 24) return null;
  if (buf[0] !== 0x89 || buf[1] !== 0x50 || buf[2] !== 0x4e || buf[3] !== 0x47) return null;
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  if (width > 0 && width < 1e6 && height > 0 && height < 1e6) return { width, height };
  return null;
}

function readJpegDimensions(buf) {
  if (buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8) return null;
  let o = 2;
  while (o + 9 < buf.length) {
    if (buf[o] !== 0xff) {
      o++;
      continue;
    }
    const marker = buf[o + 1];
    const len = buf.readUInt16BE(o + 2);
    if (len < 2 || o + 2 + len > buf.length) break;
    if (marker >= 0xc0 && marker <= 0xc3) {
      const height = buf.readUInt16BE(o + 5);
      const width = buf.readUInt16BE(o + 7);
      if (width > 0 && height > 0) return { width, height };
    }
    o += 2 + len;
  }
  return null;
}

function readImageDimensionsFromFile(imagePath) {
  try {
    const stat = fs.statSync(imagePath);
    const toRead = Math.min(stat.size, 65536);
    const fd = fs.openSync(imagePath, "r");
    try {
      const buf = Buffer.alloc(toRead);
      const n = fs.readSync(fd, buf, 0, toRead, 0);
      const slice = buf.subarray(0, n);
      return readPngDimensions(slice) || readJpegDimensions(slice);
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return null;
  }
}

function normalizeModelDetection(plane, image, index, source = "openai_vision") {
  const bbox = plane.bbox || {};
  const x = Number(bbox.x);
  const y = Number(bbox.y);
  const width = Number(bbox.width);
  const height = Number(bbox.height);

  if (![x, y, width, height].every(Number.isFinite)) return null;

  return {
    id: plane.id || `${image.id}-plane-${index + 1}`,
    bbox: {
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y)),
      width: Math.max(0.005, Math.min(1, width)),
      height: Math.max(0.005, Math.min(1, height)),
    },
    angle: Number.isFinite(Number(plane.angle)) ? Number(plane.angle) : 0,
    detectionConfidence: Math.max(0, Math.min(1, Number(plane.detectionConfidence) || 0.5)),
    source,
  };
}

function normalizeModelClassification(classification) {
  if (!["commercial", "private", "military", "unknown"].includes(classification.classification)) {
    return "unknown";
  }
  return classification.classification === "private" ? "business" : classification.classification;
}

/** One OpenAI call: each plane includes bbox + type + plausibility in a single JSON object. */
function getAircraftCombinedSchema() {
  const planeItem = {
    type: "object",
    additionalProperties: false,
    required: [
      "id",
      "bbox",
      "angle",
      "detectionConfidence",
      "classification",
      "classificationConfidence",
      "realness",
      "realnessConfidence",
      "rationale",
    ],
    properties: {
      id: { type: "string" },
      bbox: {
        type: "object",
        additionalProperties: false,
        description:
          "Axis-aligned box as fractions of full-image width and height (not pixels). Origin (0,0) is top-left; +x right; +y down; (x,y) is the top-left corner of the box. x and width are fractions of image width; y and height are fractions of image height — all in [0, 1].",
        required: ["x", "y", "width", "height"],
        properties: {
          x: {
            type: "number",
            minimum: 0,
            maximum: 1,
            description: "Left edge of the box: distance from image left, as a fraction of image width (0 = left edge, 1 = right edge).",
          },
          y: {
            type: "number",
            minimum: 0,
            maximum: 1,
            description: "Top edge of the box: distance from image top, as a fraction of image height.",
          },
          width: {
            type: "number",
            minimum: 0,
            maximum: 1,
            description: "Box width as a fraction of the full image width (positive).",
          },
          height: {
            type: "number",
            minimum: 0,
            maximum: 1,
            description: "Box height as a fraction of the full image height (positive).",
          },
        },
      },
      angle: { type: "number" },
      detectionConfidence: { type: "number" },
      classification: {
        type: "string",
        enum: ["commercial", "private", "military", "unknown"],
      },
      classificationConfidence: { type: "number" },
      realness: {
        type: "string",
        enum: ["real", "painted_or_decoy", "digitally_modified", "uncertain"],
      },
      realnessConfidence: { type: "number" },
      rationale: { type: "string" },
    },
  };

  return {
    type: "object",
    additionalProperties: false,
    required: ["planes", "imageQuality", "notes"],
    properties: {
      planes: {
        type: "array",
        items: planeItem,
      },
      imageQuality: {
        type: "string",
        enum: ["good", "usable", "poor"],
      },
      notes: { type: "string" },
    },
  };
}

/**
 * Normalized bbox contract (prompt). Fractions align with how vision models “see” the image; pixel counts often mismatch the UI.
 */
function buildOpenAiAircraftPrompt(imageDims) {
  const sizeLine =
    imageDims?.width > 0 && imageDims?.height > 0
      ? `The decoded image is ${imageDims.width} pixels wide by ${imageDims.height} pixels tall. Express every bbox as fractions of that width and height (see below), not as raw pixel counts.`
      : "Express every bbox as fractions of the image width and height (see below).";

  return [
    "Analyze this satellite or aerial image of an airport or airfield.",
    "",
    sizeLine,
    "",
    "Coordinate system for bbox.x, bbox.y, bbox.width, bbox.height (all numbers in the closed interval [0, 1]):",
    "  • Origin (0, 0) is the top-left corner of the image.",
    "  • +x runs to the right; +y runs downward.",
    "  • Do not use a bottom-left origin.",
    "  • x and width are fractions of the image width (not pixels).",
    "  • y and height are fractions of the image height (not pixels).",
    "  • Example: x=0, y=0 is the top-left; x=1 spans to the right edge; y=1 spans to the bottom edge.",
    "",
    "In one step: (1) Find distinct aircraft-like objects (fixed-wing or helicopters on ramps, aprons, or taxiways).",
    "(2) For each object, output axis-aligned bbox fields:",
    "    x = left edge of the box, as a fraction of full image width (0–1).",
    "    y = top edge of the box, as a fraction of full image height (0–1).",
    "    width = box width as a fraction of full image width (0–1).",
    "    height = box height as a fraction of full image height (0–1).",
    "    The point (x, y) is the top-left corner of the rectangle in this normalized space.",
    "(3) Classify each as commercial, private, military, or unknown based on visible shape, size, context (e.g. terminal vs military apron), and livery if readable.",
    "(4) Judge whether each detection looks real, painted_or_decoy, digitally_modified, or uncertain.",
    "",
    "Exclude: buildings, trucks, cars, runway markings, text labels, shadows without clear aircraft shape, and ambiguous blobs.",
    "Be conservative: omit doubtful objects rather than inventing aircraft.",
    "Give each detection a stable string id (e.g. plane-1, plane-2). Brief rationale per object.",
  ].join("\n");
}

function getImageDataUrl(image, airportCode) {
  const imagePath = getCaptureFilePath(airportCode, image);
  if (!openaiApiKey || !imagePath || !fs.existsSync(imagePath)) return null;
  const imageBytes = fs.readFileSync(imagePath);
  return `data:${getMimeType(image.fileName)};base64,${imageBytes.toString("base64")}`;
}

function extractOutputText(payload) {
  return payload.output_text || payload.output?.flatMap((item) => item.content || []).find((item) => item.text)?.text || "";
}

async function createOpenAIJsonResponse({ imageUrl, text, schema, name, logContext = "" }) {
  if (!openaiApiKey || !imageUrl) return null;

  const ctx = logContext ? ` ${logContext}` : "";

  const requestBody = JSON.stringify({
    model: openaiModel,
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text,
          },
          {
            type: "input_image",
            image_url: imageUrl,
            detail: "high",
          },
        ],
      },
    ],
    text: {
      format: {
        type: "json_schema",
        name,
        strict: true,
        schema,
      },
    },
  });

  console.log(
    `[openai:http] sending POST https://api.openai.com/v1/responses${ctx} schema=${name} model=${openaiModel} body_bytes=${Buffer.byteLength(requestBody, "utf8")}`,
  );

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      authorization: `Bearer ${openaiApiKey}`,
      "content-type": "application/json",
    },
    body: requestBody,
  });

  if (!response.ok) {
    const errorText = await response.text();
    const errSlice = errorText.length > 800 ? `${errorText.slice(0, 800)}…` : errorText;
    console.warn(
      `[openai:http] error POST https://api.openai.com/v1/responses${ctx} schema=${name} model=${openaiModel} status=${response.status} body=${errSlice}`,
    );
    throw new Error(`OpenAI ${name} failed: ${response.status} ${errorText}`);
  }

  const payload = await response.json();
  const respId = payload.id || "";
  console.log(
    `[openai:http] received POST https://api.openai.com/v1/responses${ctx} schema=${name} model=${openaiModel} status=${response.status}` +
      (respId ? ` response_id=${respId}` : ""),
  );

  const outputText = extractOutputText(payload);
  if (!outputText) {
    console.warn(`[openai:http] schema=${name}${ctx} status=${response.status} empty structured output`);
    return null;
  }

  return JSON.parse(outputText);
}

/** Persists model bbox numbers as returned (normalized 0–1 per schema). */
function mapOpenAICombinedPlane(raw, image, index) {
  const b = raw.bbox;
  if (!b) return null;
  const x = Number(b.x);
  const y = Number(b.y);
  const w = Number(b.width);
  const h = Number(b.height);
  if (![x, y, w, h].every(Number.isFinite)) return null;
  const cls = { classification: raw.classification };
  const normalizedClassification = normalizeModelClassification(cls);
  return {
    id: String(raw.id || `${image.id}-plane-${index + 1}`),
    bbox: { x, y, width: w, height: h },
    angle: Number.isFinite(Number(raw.angle)) ? Number(raw.angle) : 0,
    detectionConfidence: Math.max(0, Math.min(1, Number(raw.detectionConfidence) || 0.5)),
    classification: normalizedClassification,
    type: normalizedClassification,
    classificationConfidence: Math.max(0, Math.min(1, Number(raw.classificationConfidence) || 0.5)),
    realness: ["real", "painted_or_decoy", "digitally_modified", "uncertain"].includes(raw.realness)
      ? raw.realness
      : "uncertain",
    realnessConfidence: Math.max(0, Math.min(1, Number(raw.realnessConfidence) || 0.5)),
    rationale: String(raw.rationale || ""),
    source: "openai_vision",
  };
}

async function analyze_image_with_openai(image, airportCode) {
  if (!openaiApiKey) return null;
  const normalizedCode = normalizeAirportCode(airportCode);
  const capturePath = getCaptureFilePath(normalizedCode, image);
  const imageDims =
    capturePath && fs.existsSync(capturePath) ? readImageDimensionsFromFile(capturePath) : null;

  const imageUrl = getImageDataUrl(image, airportCode);
  const airportLabel = String(airportCode || "").trim() || "unknown";
  const promptText = `${buildOpenAiAircraftPrompt(imageDims)}\n\nAirfield context (ICAO/IATA or slug): ${airportLabel}.`;

  const parsed = await createOpenAIJsonResponse({
    imageUrl,
    name: "aircraft_detect_and_classify",
    schema: getAircraftCombinedSchema(),
    text: promptText,
    logContext: `${airportCode}/${image.fileName}`,
  });

  if (!parsed) return null;

  const planes = (parsed.planes || []).map((row, index) => mapOpenAICombinedPlane(row, image, index)).filter(Boolean);

  return {
    imageId: image.id,
    capturedAt: image.capturedAt,
    status: "analyzed",
    provider: "openai",
    model: openaiModel,
    bboxCoordinateSystem: "normalized",
    imageQuality: parsed.imageQuality,
    notes: parsed.notes,
    planes,
  };
}

/** Concurrent GETs for the same image used to each call OpenAI before any cache write — duplicate spend + varying counts. Share one in-flight request per resolved path. */
const openaiInflightByAbsPath = new Map();

function getOrRunOpenaiAnalysis(absPath, image, airportCode) {
  let pending = openaiInflightByAbsPath.get(absPath);
  if (pending) return pending;

  pending = (async () => {
    const nc = normalizeAirportCode(airportCode);
    const t0 = Date.now();
    console.log(`[openai:inference] start ${nc}/${image.fileName}`);
    try {
      const openaiAnalysis = await analyze_image_with_openai(image, airportCode);
      const ms = Date.now() - t0;
      const n = Array.isArray(openaiAnalysis?.planes) ? openaiAnalysis.planes.length : 0;
      console.log(`[openai:inference] done ms=${ms} ${nc}/${image.fileName} detections=${n}`);
      if (openaiAnalysis && Array.isArray(openaiAnalysis.planes)) {
        writeOpenaiPathCache(absPath, openaiAnalysis);
        console.log(
          `[analysis:ok] ${nc}/${image.fileName} provider=openai detections=${openaiAnalysis.planes.length}`,
        );
      }
      return openaiAnalysis;
    } catch (err) {
      console.warn(
        `[openai:inference] error ms=${Date.now() - t0} ${nc}/${image.fileName} ${err?.message || err}`,
      );
      throw err;
    } finally {
      openaiInflightByAbsPath.delete(absPath);
    }
  })();

  openaiInflightByAbsPath.set(absPath, pending);
  return pending;
}

function analyze_image_scaffold(image, airportCode) {
  const normalizedAirportCode = normalizeAirportCode(airportCode);
  const planeDetections = identify_planes_scaffold(image, normalizedAirportCode);
  const classifiedPlanes = classify_planes_scaffold(planeDetections, image, normalizedAirportCode);

  return {
    imageId: image.id,
    capturedAt: image.capturedAt,
    status: "scaffold",
    provider: "scaffold",
    bboxCoordinateSystem: "normalized",
    planes: classifiedPlanes,
  };
}

async function analyze_image(image, airportCode, options = {}) {
  const normalizedCode = normalizeAirportCode(airportCode);
  const pipeline = getAircraftAnalysisProvider();
  /** OpenAI: API only on cache miss or explicit `refreshAnalysis`; ignore ANALYSIS_CACHE for path cache. */
  const useOpenaiPathCache = !options.refreshAnalysis;
  const allowLegacyCacheRead = isAnalysisCacheReadEnabled() && !options.refreshAnalysis;
  const imagePath = getCaptureFilePath(normalizedCode, image);
  const absPath =
    imagePath && fs.existsSync(imagePath) ? resolveRealImagePath(path.resolve(imagePath)) : "";

  if (pipeline === "openai" && useOpenaiPathCache && absPath) {
    const pathCached = readOpenaiPathCache(absPath);
    if (pathCached) {
      const merged = mergeCachedAnalysisWithImage(pathCached, image);
      if (options.analysisStats) options.analysisStats.cacheHits += 1;
      return merged;
    }
  }

  if (pipeline !== "openai") {
    const cached = allowLegacyCacheRead ? readCachedAnalysis(airportCode, image) : null;
    if (cached) {
      if (options.analysisStats) options.analysisStats.cacheHits += 1;
      return cached;
    }
  }

  if (pipeline === "openai") {
    if (!openaiApiKey) {
      console.warn(
        `[analysis:openai-skip] ${normalizedCode}/${image.fileName} AIRCRAFT_ANALYSIS_PROVIDER=openai but OPENAI_API_KEY is unset; using scaffold`,
      );
      const analysis = analyze_image_scaffold(image, normalizedCode);
      console.log(
        `[analysis:ok] ${normalizedCode}/${image.fileName} provider=${analysis.provider} detections=${analysis.planes.length}`,
      );
      return analysis;
    }
    if (!absPath || !fs.existsSync(absPath)) {
      console.warn(`[analysis:openai-skip] ${normalizedCode}/${image.fileName} missing image file path=${absPath || imagePath}`);
      const analysis = analyze_image_scaffold(image, normalizedCode);
      return analysis;
    }
    try {
      const openaiAnalysis = await getOrRunOpenaiAnalysis(absPath, image, normalizedCode);
      if (openaiAnalysis && Array.isArray(openaiAnalysis.planes)) {
        return {
          ...openaiAnalysis,
          imageId: image.id,
          capturedAt: image.capturedAt,
        };
      }
    } catch (error) {
      console.warn(`[analysis:openai-failed] ${normalizedCode}/${image.fileName} ${error.message}`);
    }
    const fallback = analyze_image_scaffold(image, normalizedCode);
    const degraded = {
      ...fallback,
      status: "degraded",
      provider: "openai_failed",
      model: openaiModel,
      notes: "OpenAI vision analysis failed; showing scaffold detections instead.",
    };
    writeOpenaiPathCache(absPath, degraded);
    console.log(
      `[analysis:ok] ${normalizedCode}/${image.fileName} provider=openai_failed detections=${degraded.planes.length}`,
    );
    return degraded;
  }

  if (pipeline === "scaffold") {
    const analysis = analyze_image_scaffold(image, normalizedCode);
    writeCachedAnalysis(airportCode, image, analysis);
    console.log(
      `[analysis:ok] ${normalizedCode}/${image.fileName} provider=scaffold detections=${analysis.planes.length}`,
    );
    return analysis;
  }

  let rawDetections;
  const { hfBatchMap } = options;
  const hfRequested = isHfDetectionEnabled() && absPath && fs.existsSync(absPath) && getHfDetectScriptPath();

  if (hfRequested) {
    if (hfBatchMap instanceof Map && hfBatchMap.has(absPath)) {
      rawDetections = hfBatchMap.get(absPath);
    } else {
      const singleMap = runHfDetector([absPath]);
      if (singleMap?.has(absPath)) rawDetections = singleMap.get(absPath);
    }
  }

  const usedHf = hfRequested && Array.isArray(rawDetections);
  const planeDetections = usedHf
    ? rawDetections.map((p, index) => normalizeModelDetection(p, image, index, "hf_yolo")).filter(Boolean)
    : hfRequested
      ? []
      : identify_planes_scaffold(image, normalizedCode);

  const classifiedPlanes = classify_planes_stub_commercial(planeDetections);

  const analysis = {
    imageId: image.id,
    capturedAt: image.capturedAt,
    status: usedHf ? "analyzed" : hfRequested ? "degraded" : "scaffold",
    provider: usedHf ? "hf_yolo" : hfRequested ? "hf_yolo_failed" : "scaffold",
    model: usedHf || hfRequested ? getHfWeightsLabelForAnalysis() : "scaffold",
    bboxCoordinateSystem: "normalized",
    planes: classifiedPlanes,
  };

  if (hfRequested && !usedHf) {
    console.warn(`[analysis:hf-failed] ${normalizedCode}/${image.fileName} path=${absPath}`);
  }
  if (hfRequested && usedHf && classifiedPlanes.length === 0) {
    console.warn(
      `[analysis:hf-empty] ${normalizedCode}/${image.fileName} 0 planes — on clear ramps this is usually a pipeline/weights issue, not "model too dumb". Try HF_DETECT_DEBUG=1, another HF_AIRCRAFT_WEIGHTS, or confirm imagery loads as 8-bit RGB/BGR.`,
    );
  }

  writeCachedAnalysis(airportCode, image, analysis);
  console.log(
    `[analysis:ok] ${normalizedCode}/${image.fileName} provider=${analysis.provider} detections=${analysis.planes.length}`,
  );
  return analysis;
}

async function analyze_images_over_time(images, airportCode, options = {}) {
  const normalizedCode = normalizeAirportCode(airportCode);
  const refresh = options.refreshAnalysis === true;
  /** OpenAI path cache: always honor unless refresh (same contract as `analyze_image`). */
  const useOpenaiPathCache = !refresh;
  const allowLegacyCacheRead = isAnalysisCacheReadEnabled() && !refresh;
  const analysisStats = options.analysisStats || null;

  const providerIsOpenai = getAircraftAnalysisProvider() === "openai";
  const uncachedPaths = [];
  for (const image of images) {
    const p = getCaptureFilePath(normalizedCode, image);
    const abs =
      p && fs.existsSync(p) ? resolveRealImagePath(path.resolve(p)) : "";
    if (providerIsOpenai && useOpenaiPathCache && abs && readOpenaiPathCache(abs)) {
      continue;
    }
    if (!providerIsOpenai && allowLegacyCacheRead && readCachedAnalysis(airportCode, image)) continue;
    if (abs) uncachedPaths.push(abs);
  }

  if (images.length) {
    console.log(
      `[analysis] ${normalizedCode} images=${images.length} uncached=${uncachedPaths.length} concurrency=${getImageAnalysisConcurrency()}`,
    );
  }

  let hfBatchMap = null;
  if (
    getAircraftAnalysisProvider() === "hf" &&
    isHfDetectionEnabled() &&
    uncachedPaths.length &&
    getHfDetectScriptPath()
  ) {
    hfBatchMap = runHfDetector(uncachedPaths);
  }

  const mergedOpts = { ...options, hfBatchMap };
  const analyses = await mapWithConcurrency(
    images,
    getImageAnalysisConcurrency(),
    (image) => analyze_image(image, airportCode, mergedOpts),
  );

  if (images.length && analysisStats && analysisStats.cacheHits > 0) {
    console.log(
      `[analysis:cache] ${normalizedCode} disk hits=${analysisStats.cacheHits}/${images.length} (no re-inference for hits)`,
    );
  }
  if (refresh && images.length) {
    console.log(
      `[analysis:refresh] ${normalizedCode} recomputed=${images.length} images; cache updated on disk`,
    );
  }
  return analyses;
}

function countPlaneTypes(planes) {
  const counts = {
    commercial: 0,
    cargo: 0,
    business: 0,
    military: 0,
    helicopter: 0,
    unknown: 0,
  };

  planes.forEach((plane) => {
    counts[plane.classification] = (counts[plane.classification] || 0) + 1;
  });

  return counts;
}

function buildAnalysisTimeline(imageAnalyses) {
  return imageAnalyses.map((analysis) => ({
    date: analysis.capturedAt,
    ...countPlaneTypes(analysis.planes),
  }));
}

function subtractCounts(current, previous = {}) {
  return Object.fromEntries(
    Object.keys(current).map((key) => [key, current[key] - (previous[key] || 0)]),
  );
}

const ZERO_PLANE_DELTA = Object.freeze({
  commercial: 0,
  cargo: 0,
  business: 0,
  military: 0,
  helicopter: 0,
  unknown: 0,
});

const PLANE_CATEGORIES = ["commercial", "cargo", "business", "military", "helicopter", "unknown"];

function sumFleetComposition(comp) {
  return PLANE_CATEGORIES.reduce((total, key) => total + (comp[key] || 0), 0);
}

function buildChangeSignals({
  latestAnalysis,
  previousAnalysis,
  latestComposition,
  deltas,
  analysisSource,
  imagesLength,
}) {
  const signals = [];
  const latestTotal = sumFleetComposition(latestComposition);
  const prevTotal = previousAnalysis ? sumFleetComposition(countPlaneTypes(previousAnalysis.planes)) : 0;

  if (!imagesLength) {
    signals.push({
      label: "Imagery",
      value: "No local captures — showing baseline counts only",
      severity: "medium",
    });
    signals.push({
      label: "Series comparison",
      value: "Add dated images under data/airfields/<code>/",
      severity: "low",
    });
    signals.push({
      label: "Analysis source",
      value:
        analysisSource === "placeholder"
          ? "Placeholder report"
          : String(analysisSource || "—"),
      severity: "low",
    });
    signals.push({
      label: "Fleet (baseline)",
      value: `${latestTotal} aircraft (not from live imagery)`,
      severity: "low",
    });
    return signals;
  }

  if (!previousAnalysis) {
    signals.push({
      label: "Fleet count (latest)",
      value: `${latestTotal} aircraft in this capture`,
      severity: "low",
    });
  } else {
    const deltaTotal = latestTotal - prevTotal;
    const severity =
      Math.abs(deltaTotal) >= 8 ? "high" : Math.abs(deltaTotal) >= 3 ? "medium" : "low";
    signals.push({
      label: "Count vs prior capture",
      value:
        deltaTotal === 0
          ? `Stable at ${latestTotal}`
          : `${deltaTotal > 0 ? "+" : ""}${deltaTotal} (${latestTotal} vs ${prevTotal})`,
      severity,
    });
  }

  let maxAbs = 0;
  let maxKey = "unknown";
  for (const key of PLANE_CATEGORIES) {
    const dv = deltas[key] || 0;
    if (Math.abs(dv) > maxAbs) {
      maxAbs = Math.abs(dv);
      maxKey = key;
    }
  }

  if (previousAnalysis && maxAbs > 0) {
    const dv = deltas[maxKey] || 0;
    signals.push({
      label: "Largest type swing",
      value: `${maxKey} ${dv > 0 ? "+" : ""}${dv}`,
      severity: maxAbs >= 5 ? "high" : maxAbs >= 2 ? "medium" : "low",
    });
  } else {
    let bestKey = "unknown";
    let bestN = -1;
    for (const key of PLANE_CATEGORIES) {
      const n = latestComposition[key] || 0;
      if (n > bestN) {
        bestN = n;
        bestKey = key;
      }
    }
    const pct = latestTotal ? Math.round(((latestComposition[bestKey] || 0) / latestTotal) * 100) : 0;
    signals.push({
      label: "Dominant type (latest)",
      value: `${bestKey} · ${pct}% of fleet`,
      severity: "low",
    });
  }

  const provider = latestAnalysis?.provider || "unknown";
  const pipelineParts = [];
  if (provider === "openai") pipelineParts.push("OpenAI vision");
  else if (provider === "hf_yolo") pipelineParts.push("HF YOLO");
  else if (provider === "hf_yolo_failed") pipelineParts.push("HF detector failed (degraded)");
  else if (provider === "openai_failed") pipelineParts.push("OpenAI failed (scaffold fallback)");
  else if (provider === "scaffold") pipelineParts.push("Scaffold / heuristic");
  else pipelineParts.push(String(provider));

  if (latestAnalysis?.imageQuality && provider === "openai") {
    pipelineParts.push(`imagery ${latestAnalysis.imageQuality}`);
  }

  signals.push({
    label: "Detection pipeline",
    value: pipelineParts.join(" · "),
    severity: provider.includes("failed") || provider === "scaffold" ? "medium" : "low",
  });

  const planes = latestAnalysis?.planes || [];
  const lowDet = planes.filter((p) => (p.detectionConfidence ?? 0.55) < 0.45).length;
  const ambReal = planes.filter((p) =>
    ["uncertain", "painted_or_decoy"].includes(p.realness),
  ).length;
  const unknownN = latestComposition.unknown || 0;

  const ambParts = [];
  if (unknownN) ambParts.push(`${unknownN} unknown-type`);
  if (lowDet) ambParts.push(`${lowDet} low detection confidence`);
  if (ambReal) ambParts.push(`${ambReal} ambiguous realness (vision)`);

  signals.push({
    label: "Ambiguity",
    value: ambParts.length ? ambParts.join("; ") : "None flagged",
    severity: unknownN || lowDet >= 3 || ambReal ? "medium" : "low",
  });

  return signals;
}

async function attachImageSeries(report, options = {}) {
  const images = get_images_over_time(report.code);
  const analysisStats = { cacheHits: 0 };
  const imageAnalyses = await analyze_images_over_time(images, report.code, { ...options, analysisStats });
  const latestAnalysis = imageAnalyses.at(-1) || null;
  const previousAnalysis = imageAnalyses.length > 1 ? imageAnalyses.at(-2) : null;
  const latestComposition = latestAnalysis ? countPlaneTypes(latestAnalysis.planes) : report.composition;
  const previousComposition = previousAnalysis ? countPlaneTypes(previousAnalysis.planes) : {};
  const deltas = previousAnalysis
    ? subtractCounts(latestComposition, previousComposition)
    : { ...ZERO_PLANE_DELTA };

  const analysisSource = imageAnalyses.find((analysis) => analysis.provider === "hf_yolo")
    ? "hf_yolo"
    : imageAnalyses.find((analysis) => analysis.provider === "openai")
      ? "openai"
      : images.length
        ? "scaffold"
        : "placeholder";

  const signals = buildChangeSignals({
    latestAnalysis,
    previousAnalysis,
    latestComposition,
    deltas,
    analysisSource,
    imagesLength: images.length,
  });

  return {
    ...report,
    images,
    imageAnalyses,
    analysisSource,
    composition: latestComposition,
    deltas,
    signals,
    timeline: imageAnalyses.length ? buildAnalysisTimeline(imageAnalyses) : report.timeline,
    aircraft: latestAnalysis?.planes || report.aircraft,
    imageSource: images.length ? "local" : "synthetic",
    latestImage: images.at(-1) || null,
    previousImage: images.length > 1 ? images.at(-2) : null,
    latestAnalysis,
    previousAnalysis,
  };
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error("Request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

function buildPlaceholderTiles(width, height) {
  const columns = 12;
  const rows = 8;
  const tileWidth = width / columns;
  const tileHeight = height / rows;
  const tiles = [];

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const x = column * tileWidth;
      const y = row * tileHeight;
      const horizontalScore = column / (columns - 1);
      const verticalRipple = Math.sin((row / rows) * Math.PI) * 0.08;
      const score = Math.max(0, Math.min(1, horizontalScore + verticalRipple));
      tiles.push({
        x,
        y,
        width: tileWidth,
        height: tileHeight,
        score,
        label: score > 0.72 ? "higher_concern" : score > 0.42 ? "review" : "lower_concern",
      });
    }
  }

  return tiles;
}

async function handleAnalyze(req, res) {
  try {
    const body = await readBody(req);
    const payload = JSON.parse(body || "{}");
    const width = Number(payload.width);
    const height = Number(payload.height);

    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      sendJson(res, 400, { error: "Image width and height are required." });
      return;
    }

    sendJson(res, 200, {
      mode: "custom_image_placeholder",
      riskBand: "needs_review",
      tiles: buildPlaceholderTiles(width, height),
      findings: [
        {
          title: "Custom image requires aircraft/drone classification",
          severity: "medium",
          confidence: 0.5,
          explanation:
            "Uploaded imagery uses the placeholder tile overlay until the aircraft detector is connected.",
          nextStep: "Run object detection, classify aircraft type, and compare against airfield baseline.",
        },
      ],
      summary:
        "Custom image placeholder analysis generated. The main MVP flow is airfield search and time-series monitoring.",
    });
  } catch (error) {
    sendJson(res, 500, { error: error.message || "Analysis failed." });
  }
}

function listAirfields() {
  return Object.values(airfieldReports).map((report) => ({
    id: normalizeAirportCode(report.code),
    code: report.code,
    name: report.name,
    type: report.type,
    status: report.status,
    lastCapture: report.lastCapture,
    imageCount: get_images_over_time(report.code).length,
  }));
}

function serveLocalImagery(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const [, , airportId, fileName] = url.pathname.split("/");
  const airportCode = normalizeAirportCode(airportId);

  if (!airportCode || !fileName || path.basename(fileName) !== fileName || !isImageFile(fileName)) {
    res.writeHead(404);
    res.end("Not found");
    return;
  }

  const imagePath = path.join(airfieldImageRoot, airportCode, fileName);
  if (!imagePath.startsWith(airfieldImageRoot)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(imagePath, (error, content) => {
    if (error) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }

    const extension = path.extname(imagePath).toLowerCase();
    res.writeHead(200, {
      "content-type": mimeTypes[extension] || "application/octet-stream",
      "cache-control": "no-store",
    });
    res.end(content);
  });
}

function serveStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const requestedPath = url.pathname === "/" ? "/index.html" : decodeURIComponent(url.pathname);
  const filePath = path.normalize(path.join(root, requestedPath));

  if (!filePath.startsWith(root)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (error, content) => {
    if (error) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }

    const extension = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "content-type": mimeTypes[extension] || "application/octet-stream",
      "cache-control": "no-store",
    });
    res.end(content);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === "GET" && url.pathname === "/api/airfields") {
    sendJson(res, 200, { airfields: listAirfields() });
    return;
  }

  if (req.method === "GET" && url.pathname.startsWith("/api/airfields/")) {
    const parts = url.pathname.split("/");
    const airportCode = normalizeAirportCode(parts[3]);
    const report = airfieldReports[airportCode];
    if (!report) {
      sendJson(res, 404, { error: "Airfield not found." });
      return;
    }

    if (parts[4] === "images") {
      sendJson(res, 200, { airportCode, images: get_images_over_time(airportCode) });
      return;
    }

    const refresh =
      url.searchParams.get("refresh") === "1" ||
      url.searchParams.get("refresh") === "true";
    sendJson(res, 200, await attachImageSeries(report, { refreshAnalysis: refresh }));
    return;
  }

  if (req.method === "GET" && url.pathname.startsWith("/imagery/")) {
    serveLocalImagery(req, res);
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/analyze") {
    handleAnalyze(req, res);
    return;
  }

  if (req.method === "GET" || req.method === "HEAD") {
    serveStatic(req, res);
    return;
  }

  res.writeHead(405, { allow: "GET, HEAD, POST" });
  res.end("Method not allowed");
});

server.listen(port, host, () => {
  const provider = getAircraftAnalysisProvider();
  console.log(`Barad-dur running at http://${host}:${port}`);
  console.log(
    `[analysis] AIRCRAFT_ANALYSIS_PROVIDER=${provider}` +
      (provider === "openai"
        ? openaiApiKey
          ? ` model=${openaiModel} OpenAI cache=keyed-by-image-path`
          : " (set OPENAI_API_KEY)"
        : ""),
  );
  console.log(
    `[analysis] OpenAI: calls only if path cache missing or GET /api/airfields/:code?refresh=1. Legacy HF/scaffold disk cache reads: ${
      isAnalysisCacheReadEnabled() ? "on" : "off (--no-analysis-cache or ANALYSIS_CACHE=0)"
    }.`,
  );
});
