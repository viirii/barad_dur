const fs = require("fs");
const path = require("path");
const dotenv = require("dotenv");

const root = __dirname;
const envExamplePath = path.join(root, ".env.example");
const envLocalPath = path.join(root, ".env");
/** Tracked defaults (model tuning, safe to commit). Omit keys you override locally. */
if (fs.existsSync(envExamplePath)) {
  dotenv.config({ path: envExamplePath });
}
/** Secrets and overrides — same variable names replace values from `.env.example`. */
if (fs.existsSync(envLocalPath)) {
  dotenv.config({ path: envLocalPath, override: true });
}

/** Required HF/aircraft keys live only in `.env.example` / `.env` (no duplicated literals in code paths below). */
function requireEnv(name) {
  const v = process.env[name];
  if (v === undefined || v === "") {
    throw new Error(
      `Missing required environment variable ${name}. Define it in .env.example and load via dotenv.`,
    );
  }
  return v;
}

const http = require("http");
const crypto = require("crypto");
const { spawnSync } = require("child_process");

const cliArgs = process.argv.slice(2);
/**
 * When true, skip reading per-image analysis JSON (`writeCachedAnalysis` for HF/scaffold);
 * does **not** bypass OpenAI path cache — OpenAI calls when path cache is missing, `?refresh=1`, or `OPENAI_PATH_CACHE_VERSION` changes.
 * Set via `--no-analysis-cache` or `ANALYSIS_CACHE=0`.
 */
const analysisCacheReadDisabled =
  cliArgs.includes("--no-analysis-cache") || process.env.ANALYSIS_CACHE === "0";

function isAnalysisCacheReadEnabled() {
  return !analysisCacheReadDisabled;
}

const host = process.env.HOST || "127.0.0.1";
const port = Number(process.env.PORT || 3000);
const airfieldImageRoot = path.join(root, "data", "airfields");
const analysisCacheRoot = path.join(root, "data", "analysis-cache");
/** Mirrors `analysis-cache/**` layout: `${providerKey}/${airport}/<date-key>.json` — OpenAI classification audit trail. */
const classificationCacheRoot = path.join(root, "data", "classification-cache");
const summaryCacheRoot = path.join(root, "data", "summary-cache");
/** Bump when the series-summary prompt or schema meaning changes (cache fingerprint salt). */
const SERIES_SUMMARY_PROMPT_VERSION = "v5-unknown-buckets-and-summary-attention";
const openaiApiKey = process.env.OPENAI_API_KEY || "";
const openaiModel = process.env.OPENAI_MODEL || "gpt-5.5";

/**
 * Mixed into `hashResolvedImagePath` so OpenAI `by-path` entries under `analysis-cache` miss completions from older prompts.
 * Bump when classify/vision instructions or JSON schema semantics change materially.
 */
const OPENAI_PATH_CACHE_VERSION = "classify-v11-painted-vs-real-explicit-question";

/**
 * Max concurrent `analyze_image` runs per airfield load (order preserved).
 * Does not bypass cache: each call still reads path cache + per-image JSON first; `refreshAnalysis` forces recompute.
 * HF detection for a batch runs once in `runHfDetector`; parallel workers mostly merge + OpenAI classify — tune alongside OPENAI_MAX_CONCURRENT.
 * Default 5 in `.env.example`; set `IMAGE_ANALYSIS_CONCURRENCY=1` to serialize if needed.
 */
function getImageAnalysisConcurrency() {
  const raw = process.env.IMAGE_ANALYSIS_CONCURRENCY;
  const n = raw == null || String(raw).trim() === "" ? 5 : Number(raw);
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

/** Caps concurrent OpenAI API requests (HF classify + full vision). Tunable via OPENAI_MAX_CONCURRENT (default 5). */
function createConcurrencyLimiter(maxConcurrent) {
  let active = 0;
  const queue = [];
  const max = Math.max(1, maxConcurrent);

  const pump = () => {
    while (active < max && queue.length > 0) {
      const job = queue.shift();
      active++;
      Promise.resolve(job.fn())
        .then(job.resolve, job.reject)
        .finally(() => {
          active--;
          pump();
        });
    }
  };

  return (fn) =>
    new Promise((resolve, reject) => {
      queue.push({ fn, resolve, reject });
      pump();
    });
}

function getOpenAiMaxConcurrent() {
  const raw = process.env.OPENAI_MAX_CONCURRENT;
  const n = raw == null || String(raw).trim() === "" ? 5 : Number(raw);
  if (!Number.isFinite(n) || n < 1) return 5;
  return Math.min(64, Math.floor(n));
}

const limitOpenAiHttpConcurrency = createConcurrencyLimiter(getOpenAiMaxConcurrent());

/** Aircraft pipeline: `hf` (local YOLO), `openai` (vision API, needs OPENAI_API_KEY), or `scaffold` (deterministic fake). */
function getAircraftAnalysisProvider() {
  const raw = (process.env.AIRCRAFT_ANALYSIS_PROVIDER || "hf").trim().toLowerCase();
  if (raw === "openai" || raw === "hf" || raw === "scaffold") return raw;
  return "hf";
}

/** Weights label for API/cache — QUICK vs full HF paths come from `.env.example`. */
function getHfWeightsLabelForAnalysis() {
  return requireEnv("HF_AIRCRAFT_QUICK") === "1"
    ? requireEnv("HF_AIRCRAFT_QUICK_WEIGHTS")
    : requireEnv("HF_AIRCRAFT_WEIGHTS");
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
      civilian: 61,
      military: 0,
    },
    deltas: {
      civilian: 4,
      military: 0,
    },
    timeline: [
      { date: "Apr 18", civilian: 54, military: 0 },
      { date: "Apr 22", civilian: 55, military: 0 },
      { date: "Apr 26", civilian: 56, military: 0 },
      { date: "May 01", civilian: 61, military: 0 },
    ],
    signals: [
      { label: "Aircraft count", value: "Within baseline", severity: "low" },
      { label: "Type mix", value: "Civilian dominant", severity: "low" },
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
      { x: 0.18, y: 0.24, type: "civilian", confidence: 0.88, angle: -8 },
      { x: 0.25, y: 0.28, type: "civilian", confidence: 0.91, angle: -8 },
      { x: 0.32, y: 0.32, type: "civilian", confidence: 0.9, angle: -8 },
      { x: 0.48, y: 0.41, type: "civilian", confidence: 0.78, angle: 6 },
      { x: 0.57, y: 0.44, type: "civilian", confidence: 0.75, angle: 6 },
      { x: 0.68, y: 0.55, type: "civilian", confidence: 0.66, angle: 12 },
      { x: 0.76, y: 0.64, type: "civilian", confidence: 0.42, angle: 12 },
    ],
  },
  /** Demo airfield: same baseline story as SFO; imagery & caches cloned from SFO under data/airfields/sfx. */
  sfx: {
    id: "sfx",
    code: "SFX",
    name: "SFX — SFO mirror (demo)",
    type: "Commercial airport",
    status: "Normal",
    riskBand: "normal",
    lastCapture: "2026-05-01 18:20 UTC",
    baseline: "commercial-heavy, cargo activity expected",
    summary:
      "Placeholder profile matches SFO. Local imagery under data/airfields/sfx is a copy of SFO captures for side-by-side or modified-image experiments.",
    composition: {
      civilian: 61,
      military: 0,
    },
    deltas: {
      civilian: 4,
      military: 0,
    },
    timeline: [
      { date: "Apr 18", civilian: 54, military: 0 },
      { date: "Apr 22", civilian: 55, military: 0 },
      { date: "Apr 26", civilian: 56, military: 0 },
      { date: "May 01", civilian: 61, military: 0 },
    ],
    signals: [
      { label: "Aircraft count", value: "Within baseline", severity: "low" },
      { label: "Type mix", value: "Civilian dominant", severity: "low" },
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
      { x: 0.18, y: 0.24, type: "civilian", confidence: 0.88, angle: -8 },
      { x: 0.25, y: 0.28, type: "civilian", confidence: 0.91, angle: -8 },
      { x: 0.32, y: 0.32, type: "civilian", confidence: 0.9, angle: -8 },
      { x: 0.48, y: 0.41, type: "civilian", confidence: 0.78, angle: 6 },
      { x: 0.57, y: 0.44, type: "civilian", confidence: 0.75, angle: 6 },
      { x: 0.68, y: 0.55, type: "civilian", confidence: 0.66, angle: 12 },
      { x: 0.76, y: 0.64, type: "civilian", confidence: 0.42, angle: 12 },
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
      military: 18,
      civilian: 6,
    },
    deltas: {
      military: 9,
      civilian: 3,
    },
    timeline: [
      { date: "Apr 18", military: 8, civilian: 3 },
      { date: "Apr 22", military: 10, civilian: 3 },
      { date: "Apr 26", military: 9, civilian: 3 },
      { date: "May 01", military: 18, civilian: 6 },
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
      { x: 0.71, y: 0.58, type: "civilian", confidence: 0.44, angle: 12 },
      { x: 0.77, y: 0.63, type: "civilian", confidence: 0.39, angle: 12 },
    ],
  },
  /** Demo airfield: same baseline story as SUU; imagery & caches cloned from SUU under data/airfields/sux. */
  sux: {
    id: "sux",
    code: "SUX",
    name: "SUX — SUU mirror (demo)",
    type: "Military airfield",
    status: "Needs review",
    riskBand: "needs_review",
    lastCapture: "2026-05-01 19:05 UTC",
    baseline: "large military transport/tanker aircraft expected",
    summary:
      "Placeholder profile matches SUU. Local imagery under data/airfields/sux is a copy of SUU captures for side-by-side or modified-image experiments.",
    composition: {
      military: 18,
      civilian: 6,
    },
    deltas: {
      military: 9,
      civilian: 3,
    },
    timeline: [
      { date: "Apr 18", military: 8, civilian: 3 },
      { date: "Apr 22", military: 10, civilian: 3 },
      { date: "Apr 26", military: 9, civilian: 3 },
      { date: "May 01", military: 18, civilian: 6 },
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
      { x: 0.71, y: 0.58, type: "civilian", confidence: 0.44, angle: 12 },
      { x: 0.77, y: 0.63, type: "civilian", confidence: 0.39, angle: 12 },
    ],
  },
  oak: {
    id: "oak",
    code: "OAK",
    name: "Oakland International Airport",
    type: "Commercial airport",
    status: "Normal",
    riskBand: "normal",
    lastCapture: "2026-05-01 17:35 UTC",
    baseline: "mixed commercial, regional, and cargo traffic typical for Bay Area gateway",
    summary:
      "OAK traffic in this placeholder report aligns with a mid-size commercial airport: dominant narrow-body activity with modest cargo and general aviation presence.",
    composition: {
      civilian: 37,
      military: 0,
    },
    deltas: {
      civilian: 1,
      military: 0,
    },
    timeline: [
      { date: "Apr 18", civilian: 35, military: 0 },
      { date: "Apr 22", civilian: 36, military: 0 },
      { date: "Apr 26", civilian: 36, military: 0 },
      { date: "May 01", civilian: 37, military: 0 },
    ],
    signals: [
      { label: "Aircraft count", value: "Within baseline", severity: "low" },
      { label: "Type mix", value: "Civilian dominant", severity: "low" },
      { label: "Freight apron", value: "Moderate freight activity", severity: "low" },
      { label: "Uncertain regions", value: "2 peripheral ambiguous shapes", severity: "medium" },
    ],
    findings: [
      {
        title: "Regional commercial baseline",
        severity: "low",
        confidence: 0.84,
        explanation:
          "Aircraft-like clustering matches terminal and apron zones typical for Oakland International in placeholder imagery.",
        nextStep: "Compare temporal deltas only if policy thresholds depend on count swings.",
      },
      {
        title: "Two ambiguous apron-edge shapes",
        severity: "medium",
        confidence: 0.48,
        explanation:
          "Small objects near service roads lack crisp fuselage or wing cues at this resolution.",
        nextStep: "Defer conclusions pending sharper captures or corroborating sensors.",
      },
    ],
    aircraft: [
      { x: 0.22, y: 0.26, type: "civilian", confidence: 0.87, angle: -6 },
      { x: 0.34, y: 0.3, type: "civilian", confidence: 0.89, angle: -6 },
      { x: 0.46, y: 0.38, type: "civilian", confidence: 0.76, angle: 4 },
      { x: 0.58, y: 0.44, type: "civilian", confidence: 0.85, angle: 4 },
      { x: 0.72, y: 0.52, type: "civilian", confidence: 0.62, angle: 10 },
      { x: 0.64, y: 0.61, type: "civilian", confidence: 0.41, angle: 14 },
    ],
  },
  sjc: {
    id: "sjc",
    code: "SJC",
    name: "San Jose Mineta International Airport",
    type: "Commercial airport",
    status: "Normal",
    riskBand: "normal",
    lastCapture: "2026-05-01 17:45 UTC",
    baseline: "commercial passenger and regional traffic expected",
    summary:
      "SJC activity in this placeholder profile remains consistent with a commercial airport, with civilian aircraft dominance and no unusual military mix.",
    composition: {
      civilian: 28,
      military: 0,
    },
    deltas: {
      civilian: 2,
      military: 0,
    },
    timeline: [
      { date: "Apr 18", civilian: 24, military: 0 },
      { date: "Apr 22", civilian: 25, military: 0 },
      { date: "Apr 26", civilian: 26, military: 0 },
      { date: "May 01", civilian: 28, military: 0 },
    ],
    signals: [
      { label: "Aircraft count", value: "Within baseline", severity: "low" },
      { label: "Type mix", value: "Civilian dominant", severity: "low" },
      { label: "Military-style aircraft", value: "None detected", severity: "low" },
      { label: "Uncertain regions", value: "1 peripheral ambiguous shape", severity: "low" },
    ],
    findings: [
      {
        title: "Commercial traffic baseline",
        severity: "low",
        confidence: 0.85,
        explanation:
          "Aircraft-like objects are concentrated around terminal and apron areas, consistent with expected SJC operations.",
        nextStep: "Use additional captures only if investigating a specific deviation threshold.",
      },
    ],
    aircraft: [
      { x: 0.24, y: 0.28, type: "civilian", confidence: 0.86, angle: -7 },
      { x: 0.33, y: 0.33, type: "civilian", confidence: 0.88, angle: -7 },
      { x: 0.45, y: 0.39, type: "civilian", confidence: 0.8, angle: 5 },
      { x: 0.56, y: 0.46, type: "civilian", confidence: 0.77, angle: 5 },
      { x: 0.68, y: 0.53, type: "civilian", confidence: 0.64, angle: 11 },
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

/**
 * Demo mirror folders (sfx/sux) reuse another base’s imagery role in **model prompts only**:
 * pass SFO / San Francisco and SUU / Travis identities instead of the mirror codes.
 */
function getAirfieldCatalogReportForPrompts(airportCode) {
  const key = normalizeAirportCode(airportCode);
  if (key === "sfx") return airfieldReports.sfo;
  if (key === "sux") return airfieldReports.suu;
  return airfieldReports[key] || null;
}

function getAirfieldCatalogIdentityForSeriesSummary(report) {
  if (!report) return null;
  const key = normalizeAirportCode(report.code);
  if (key === "sfx") return airfieldReports.sfo;
  if (key === "sux") return airfieldReports.suu;
  return report;
}

/** Code + catalog name + type for vision prompts (canonical identity for demo mirrors). */
function getAirfieldLabelForClassificationPrompt(airportCode) {
  const report = getAirfieldCatalogReportForPrompts(airportCode);
  const codeUpper = String(report?.code || "").trim().toUpperCase();
  if (!report || !codeUpper) {
    const k = (String(airportCode || "").trim() || "unknown").toUpperCase();
    return `${k} — catalog unknown; use silhouette and image only`;
  }
  const name = String(report.name || "").trim();
  const type = String(report.type || "").trim() || "unknown";
  return name ? `${codeUpper} — ${name} — ${type}` : `${codeUpper} — ${type}`;
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
  const defaultId = `${normalizedAirportCode}-${sanitizeCacheTag(fileName)}`;

  return {
    id: capture.id || defaultId,
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

function sanitizeCacheTag(value) {
  if (value === undefined || value === null) {
    throw new Error("sanitizeCacheTag: missing value");
  }
  return String(value)
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/_+/g, "_")
    .slice(0, 96);
}

/** Namespaces disk cache; must change when inference inputs change (otherwise env tweaks are ignored). */
function getAnalysisProviderKey() {
  const mode = getAircraftAnalysisProvider();
  if (mode === "openai") {
    const modelTag = sanitizeCacheTag(openaiModel);
    /** Bump when bbox JSON contract changes (e.g. pixels → normalized) so disk cache is not reused across incompatible formats. */
    return `openai-m-${modelTag}-bbox01`;
  }
  if (mode === "scaffold") {
    return "scaffold-v1";
  }
  const quickMode = requireEnv("HF_AIRCRAFT_QUICK") === "1";
  const weightsRaw = quickMode
    ? requireEnv("HF_AIRCRAFT_QUICK_WEIGHTS")
    : requireEnv("HF_AIRCRAFT_WEIGHTS");
  const confRaw = requireEnv("HF_AIRCRAFT_CONF");
  const imgszRaw = quickMode
    ? requireEnv("HF_AIRCRAFT_QUICK_IMGSZ")
    : requireEnv("HF_AIRCRAFT_IMGSZ");
  const imgszMaxRaw = quickMode
    ? requireEnv("HF_AIRCRAFT_QUICK_IMGSZ_MAX")
    : requireEnv("HF_AIRCRAFT_IMGSZ_MAX");
  const tileRaw = requireEnv("HF_AIRCRAFT_TILE");
  const overlapRaw = quickMode
    ? requireEnv("HF_AIRCRAFT_QUICK_TILE_OVERLAP")
    : requireEnv("HF_AIRCRAFT_TILE_OVERLAP");
  const augmentRaw = requireEnv("HF_AIRCRAFT_AUGMENT");
  const iouRaw = quickMode
    ? requireEnv("HF_AIRCRAFT_QUICK_IOU")
    : requireEnv("HF_AIRCRAFT_IOU");
  const maxDetRaw = quickMode
    ? requireEnv("HF_AIRCRAFT_QUICK_MAX_DET")
    : requireEnv("HF_AIRCRAFT_MAX_DET");
  const tileMergeRaw = quickMode
    ? requireEnv("HF_AIRCRAFT_QUICK_TILE_MERGE_IOU")
    : requireEnv("HF_AIRCRAFT_TILE_MERGE_IOU");
  const maxBoxAreaRaw = requireEnv("HF_AIRCRAFT_MAX_BOX_AREA");
  const quickTag = sanitizeCacheTag(requireEnv("HF_AIRCRAFT_QUICK"));
  const weightsTag = sanitizeCacheTag(weightsRaw);
  const confTag = sanitizeCacheTag(confRaw);
  const imgszTag = sanitizeCacheTag(imgszRaw);
  const imgszMaxTag = sanitizeCacheTag(imgszMaxRaw);
  const tileTag = sanitizeCacheTag(tileRaw);
  const overlapTag = sanitizeCacheTag(overlapRaw);
  const augmentTag = sanitizeCacheTag(augmentRaw);
  const iouTag = sanitizeCacheTag(iouRaw);
  const maxDetTag = sanitizeCacheTag(maxDetRaw);
  const tileMergeTag = sanitizeCacheTag(tileMergeRaw);
  const quickKeepTag = quickMode
    ? sanitizeCacheTag(requireEnv("HF_QUICK_KEEP_CLASSES"))
    : "na";
  const maxBoxAreaTag = sanitizeCacheTag(maxBoxAreaRaw);
  return (
    `hf-yolo-w-${weightsTag}-c-${confTag}-sz-${imgszTag}-mx-${imgszMaxTag}-t-${tileTag}-o-${overlapTag}-aug-${augmentTag}-iou-${iouTag}-md-${maxDetTag}-tm-${tileMergeTag}-ma-${maxBoxAreaTag}-q-${quickTag}-kc-${quickKeepTag}-quick-v19-stub-civilian`
  );
}

/**
 * Per-capture JSON basename: sanitized **original image file name** (e.g. `20260501.png` → `20260501_png`).
 * Not ordinal index; stable when captures are reordered.
 */
function getImageCacheFileBase(image) {
  if (!image) return "unknown";
  const fn = image.fileName != null ? String(image.fileName).trim() : "";
  if (fn && path.basename(fn) === fn && !fn.includes("..") && !fn.includes("/") && !fn.includes("\\")) {
    return sanitizeCacheTag(fn);
  }
  return "unknown";
}

function getAnalysisCachePath(airportCode, image) {
  const normalizedAirportCode = normalizeAirportCode(airportCode);
  const cacheDir = path.join(analysisCacheRoot, getAnalysisProviderKey(), normalizedAirportCode);
  if (!cacheDir.startsWith(analysisCacheRoot)) return "";
  return path.join(cacheDir, `${getImageCacheFileBase(image)}.json`);
}

function getClassificationCachePath(airportCode, image) {
  const normalizedAirportCode = normalizeAirportCode(airportCode);
  const cacheDir = path.join(classificationCacheRoot, getAnalysisProviderKey(), normalizedAirportCode);
  if (!cacheDir.startsWith(classificationCacheRoot)) return "";
  return path.join(cacheDir, `${getImageCacheFileBase(image)}.json`);
}

/** Written only after HF boxes + successful OpenAI classify (`hf_yolo_openai`); not created for `hf_yolo` stub-only or failed classify. */
function writeClassificationCache(airportCode, image, payload) {
  const cachePath = getClassificationCachePath(airportCode, image);
  if (!cachePath) return;
  fs.mkdirSync(path.dirname(cachePath), { recursive: true });
  fs.writeFileSync(cachePath, JSON.stringify(payload, null, 2));
}

/** Stable OpenAI cache filename: SHA-256(hex) of path + OPENAI_PATH_CACHE_VERSION (path-addressed; version bumps invalidate stale prompts). */
function hashResolvedImagePath(absResolvedPath) {
  const canonical = String(absResolvedPath || "");
  return crypto
    .createHash("sha256")
    .update(`${canonical}\0${OPENAI_PATH_CACHE_VERSION}`, "utf8")
    .digest("hex");
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

  const fromManifest = fs.existsSync(airportDir) ? readManifestCaptures(normalizedAirportCode, airportDir) : null;
  const fromDir = readDirectoryCaptures(normalizedAirportCode, airportDir);

  /**
   * Union manifest + folder scan: manifest supplies ids/metadata when present, but **every**
   * image file on disk must appear even if missing from manifest (otherwise new drops are invisible).
   * Manifest-only rows already omit deleted files (`readManifestCaptures` checks existsSync).
   */
  const byFile = new Map();
  if (fromManifest && fromManifest.length > 0) {
    for (const c of fromManifest) {
      byFile.set(c.fileName, c);
    }
  }
  for (const c of fromDir) {
    if (!byFile.has(c.fileName)) {
      byFile.set(c.fileName, c);
    }
  }

  const captures = Array.from(byFile.values());
  return captures.sort((a, b) => String(a.capturedAt).localeCompare(String(b.capturedAt)));
}

function hashString(value) {
  return String(value)
    .split("")
    .reduce((hash, char) => (hash * 31 + char.charCodeAt(0)) >>> 0, 7);
}

function identify_planes_scaffold(image, airportCode) {
  const seed = hashString(`${airportCode}:${image.id}:${image.capturedAt}`);
  const travisLike = airportCode === "suu" || airportCode === "sux";
  const count = travisLike ? 4 + (seed % 7) : 8 + (seed % 10);

  return Array.from({ length: count }, (_, index) => {
    const row = Math.floor(index / 4);
    const column = index % 4;
    const jitterX = ((seed + index * 17) % 13) / 100;
    const jitterY = ((seed + index * 29) % 11) / 100;
    const x = Math.min(0.9, 0.16 + column * 0.18 + jitterX);
    const y = Math.min(0.86, 0.22 + row * 0.16 + jitterY);
    const width = travisLike ? 0.065 : 0.08;
    const height = travisLike ? 0.038 : 0.046;

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
    let classification = "civilian";
    if (airportCode === "suu" || airportCode === "sux") {
      classification = index % 3 === 0 ? "military" : "civilian";
    } else if (index % 11 === 9) {
      classification = "military";
    } else {
      classification = "civilian";
    }

    return {
      ...plane,
      classification,
      type: classification,
      classificationConfidence: 0.64 + ((index % 4) * 0.07),
      source: "scaffold",
    };
  });
}

/** HF/OpenAI fallback when classify step is skipped or fails: default civilian until a real pass runs. */
function classify_planes_stub_civilian(planeDetections) {
  return planeDetections.map((plane, index) => ({
    ...plane,
    classification: "civilian",
    type: "civilian",
    classificationConfidence: Math.max(
      0.35,
      Math.min(1, plane.detectionConfidence != null ? plane.detectionConfidence : 0.75),
    ),
    realness: "uncertain",
    realnessConfidence: 0.5,
    rationale: "Stub classification: HF detection only; military/civilian/unknown not inferred yet.",
    source: "stub_civilian",
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
  const timeoutMs = Number(requireEnv("HF_DETECT_TIMEOUT_MS"));
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
    if (value?.error) {
      console.warn(
        `[hf:inference] ${path.basename(key)} detector reported: ${value.error}`,
      );
    }
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

/** Canonical labels: military | civilian | unknown. Legacy values map into these. */
function normalizeClassificationString(raw) {
  const v = String(raw ?? "").trim().toLowerCase();
  if (v === "military") return "military";
  if (v === "unknown") return "unknown";
  return "civilian";
}

function normalizeModelClassification(classification) {
  const raw =
    classification && typeof classification === "object" ? classification.classification : classification;
  return normalizeClassificationString(raw);
}

/** JSON-schema description + prompt lines: explicit painted vs real, then tri-label; military recall only for plausible real aircraft. */
const AIRCRAFT_CLASSIFICATION_SENSITIVITY_DESCRIPTION =
  "Tri-label **military | civilian | unknown**. **Ask directly for every box:** given the surrounding ramp/tarmac, lighting, and neighbors — does this look **painted or marked on the ground** (flat graphic, tarmac marking, decoy-like silhouette, colors that read as paint on asphalt rather than a 3D fuselage), or does it **actually look like a real aircraft** sitting on the surface (believable volume, materials, and shadow consistent with the scene)? " +
  "If it reads as **painted / flat / not a real plane in context**, set **classification unknown** and **realness** painted_or_decoy or uncertain (or digitally_modified if composite/paste). **Only if** it **actually looks like a real aircraft**, proceed to mil/civ. " +
  "Other **off or weird** cues (wrong albedo vs neighbors, missing/wrong shadows, halos, geometry that does not read as a parked aircraft) → same: **unknown** + non-real **realness**. **Do not** assign military or civilian when the answer to “real plane vs painted?” is not clearly **real**. " +
  "**Only if** the object plausibly looks like a **real** aircraft: assign **military** when tactical cues support it (swept/delta/compact tactical planform, military transport/tanker/helo, etc.) — including small/distant/hazy crops if those cues exist; **civilian** when the silhouette clearly reads non-military. " +
  "**Unknown** also when mil/civ is not supportable (extreme blur, not clearly an aircraft) even if somewhat plausible. " +
  "When it is clearly a **real** aircraft but military vs civilian stays ambiguous after silhouette review, tie-break **military**; a civilian hub must not override a tactical-looking **real** shape. " +
  "Always align **realness** with how natural the crop looks; cite unnatural cues in **rationale**.";

function buildClassificationSensitivityPromptLines() {
  return [
    "**Painted vs real (answer out loud in rationale):** Does this look like **paint on the ramp** / flat marking / decoy silhouette, or like a **real aircraft** in this image context (lighting, shadow, texture vs neighbors)? If not clearly a **real plane** → **classification unknown**; set **realness** accordingly.",
    "SENSITIVITY (screening): **That question before mil/civ.** Wrong/weird colors, flat graphic look, bad shadows, paste/halos → **unknown** + non-real **realness**.",
    "  • Detector boxes can be false positives — do not trust the label “plane” over your eyes.",
    "  • Only when the crop **plausibly** looks like a **real** aircraft: favor **military** recall if ANY tactical silhouette cue exists (small/distant/hazy OK if cues exist).",
    "  • Clear tube-with-wings airliner / obvious civilian freighter (and looks natural) → civilian.",
    "  • **Unknown** for unnatural/decoy-like/artifact crops OR when mil/civ is not honestly callable. Do not default to civilian to dodge unknown.",
    "  • Tie-break **military** only when the object is a plausible **real** aircraft but mil vs civ remains ambiguous. Airport name must not talk you out of a tactical-looking **real** shape.",
    "  • **Realness** must reflect naturalness (real vs painted_or_decoy / digitally_modified / uncertain); cite specifics in rationale.",
  ];
}

/** HF second pass: classify pre-detected planes (ids must match) + optional high-confidence missed aircraft. */
function getAircraftClassificationOnlySchema() {
  const planeItem = {
    type: "object",
    additionalProperties: false,
    required: [
      "id",
      "classification",
      "classificationConfidence",
      "realness",
      "realnessConfidence",
      "rationale",
    ],
    properties: {
      id: { type: "string" },
      classification: {
        type: "string",
        enum: ["civilian", "military", "unknown"],
        description: AIRCRAFT_CLASSIFICATION_SENSITIVITY_DESCRIPTION,
      },
      classificationConfidence: {
        type: "number",
        description:
          "Confidence in military vs civilian vs unknown; use lower values when the crop is poor, looks unnatural, or you chose unknown because mil/civ or real-aircraft plausibility is weak.",
      },
      realness: {
        type: "string",
        enum: ["real", "painted_or_decoy", "digitally_modified", "uncertain"],
        description:
          "Outcome of: does it look **painted/marked on the ramp** vs **actually like a real aircraft** in this scene? real = convincingly a real plane in context; painted_or_decoy = flat/decoy/marking-like; digitally_modified = composite/paste; uncertain = cannot tell.",
      },
      realnessConfidence: { type: "number" },
      rationale: {
        type: "string",
        description:
          "State explicitly whether it looks **painted/decoy-like vs a real plane given ramp and lighting context**; then silhouette, mil/civ/unknown, and realness.",
      },
    },
  };

  const additionalAircraftItem = {
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
      "additionCertainty",
    ],
    properties: {
      id: {
        type: "string",
        description:
          "New stable id for an aircraft missed by the detector, e.g. added-1, added-2. Must not reuse pre-listed ids.",
      },
      bbox: {
        type: "object",
        additionalProperties: false,
        description: "Normalized axis-aligned box for the newly found aircraft (fractions of image width/height).",
        required: ["x", "y", "width", "height"],
        properties: {
          x: { type: "number", minimum: 0, maximum: 1 },
          y: { type: "number", minimum: 0, maximum: 1 },
          width: { type: "number", minimum: 0, maximum: 1 },
          height: { type: "number", minimum: 0, maximum: 1 },
        },
      },
      angle: { type: "number" },
      detectionConfidence: {
        type: "number",
        description: "Confidence that this box is a real aircraft (your judgment).",
      },
      classification: {
        type: "string",
        enum: ["civilian", "military", "unknown"],
        description: AIRCRAFT_CLASSIFICATION_SENSITIVITY_DESCRIPTION,
      },
      classificationConfidence: {
        type: "number",
        description:
          "Confidence in military vs civilian vs unknown; lower when the crop looks unnatural or unknown was chosen.",
      },
      realness: {
        type: "string",
        enum: ["real", "painted_or_decoy", "digitally_modified", "uncertain"],
        description:
          "Painted/marked/decoy vs actually a real aircraft in context; only use real when clearly the latter.",
      },
      realnessConfidence: { type: "number" },
      rationale: {
        type: "string",
        description:
          "Say painted vs real in context; silhouette; realness. Only add if clearly a real aircraft missed by the detector.",
      },
      additionCertainty: {
        type: "number",
        description:
          "How certain you are that this is a genuine natural-looking aircraft missed by the prior detector (not noise). Only output rows with additionCertainty >= 0.85.",
      },
    },
  };

  return {
    type: "object",
    additionalProperties: false,
    required: ["planes", "additionalAircraft", "imageQuality", "notes"],
    properties: {
      planes: {
        type: "array",
        items: planeItem,
      },
      additionalAircraft: {
        type: "array",
        description:
          "Distinct aircraft clearly visible in the image that were NOT in the pre-listed boxes. Return an empty array if none meet the certainty bar.",
        items: additionalAircraftItem,
      },
      imageQuality: {
        type: "string",
        enum: ["good", "usable", "poor"],
      },
      notes: { type: "string" },
    },
  };
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
        enum: ["civilian", "military", "unknown"],
        description: AIRCRAFT_CLASSIFICATION_SENSITIVITY_DESCRIPTION,
      },
      classificationConfidence: {
        type: "number",
        description:
          "Confidence in military vs civilian vs unknown; lower when imagery is poor, crop looks unnatural, or unknown is chosen because mil/civ or plausibility is weak.",
      },
      realness: {
        type: "string",
        enum: ["real", "painted_or_decoy", "digitally_modified", "uncertain"],
        description:
          "How natural/plausible the crop looks as a physical aircraft; wrong colors, bad shadows, or paste tells → not real.",
      },
      realnessConfidence: { type: "number" },
      rationale: {
        type: "string",
        description:
          "Cite naturalness (colors, shadows), silhouette, realness, and unknown vs military tie-break if applicable.",
      },
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
    "(3) Classify each aircraft as exactly one of: **civilian | military | unknown**. **First ask outright:** does this look **painted or marked on the surface** (decoy-like, flat graphic, tarmac marking) or does it **actually look like a real aircraft** in this scene (ramp context, lighting, believable shadow and volume)? If **painted / not a real plane in context** → **unknown** + non-real **realness**; skip mil/civ.",
    "    **Military** (only if plausible real aircraft) includes fighters, attack jets, trainers, tankers, AWACS, military transports, tactical rotorcraft (compact/tactical proportions, delta or strongly swept wings, short fuselage vs wingspan, wing-body blends, tactical parking).",
    "    **Civilian** when the shape clearly reads non-military **and** the object looks like a natural real aircraft.",
    "    **Unknown** when unnatural/decoy-like, not clearly an aircraft, extreme blur, or mil/civ not honestly callable.",
    "    CRITICAL: Do NOT label military-shaped jets as civilian because the image shows a major civilian airport. Fighters can appear on any ramp in satellite views.",
    "    If unsure between military and civilian after checking silhouette but it is clearly a **natural** aircraft, prefer military when wings dominate and the fuselage is short relative to wingspan.",
    ...buildClassificationSensitivityPromptLines(),
    "(4) Judge **realness** to match that same question: **real** only when it **actually** looks like a physical aircraft in context; otherwise **painted_or_decoy**, **digitally_modified**, or **uncertain** — say which in rationale.",
    "",
    "Exclude: buildings, trucks, cars, runway markings, text labels, shadows without clear aircraft shape, and ambiguous blobs.",
    "Be conservative: omit doubtful objects rather than inventing aircraft.",
    "Give each detection a stable string id (e.g. plane-1, plane-2). Brief rationale per object.",
  ].join("\n");
}

function buildOpenAiClassificationOnlyPrompt(detectionPayload, imageDims, airportLabel) {
  const sizeLine =
    imageDims?.width > 0 && imageDims?.height > 0
      ? `The attached image is full-resolution: ${imageDims.width} pixels wide by ${imageDims.height} pixels tall. Use the same normalized bbox convention (fractions of width/height) as reference only — do not change coordinates.`
      : "The attached image is full-resolution. Bboxes in the JSON are normalized [0,1] fractions; do not change them.";

  return [
    "You classify aircraft only. Pre-listed boxes come from an automated detector — **do not assume every box is a real plane.** Your job is to **decide painted/marked/decoy vs actually a real aircraft in context**, then **military vs civilian vs unknown**, and **realness** per listed id.",
    "MODE: For **each** listed id, **explicitly answer:** “Does this look **painted on** (ground marking, flat graphic, fake aircraft paint, colors that sit on the tarmac like a decal) or does it **actually look like a real aircraft** here given ramp, sun direction, and texture vs neighbors?” If **painted / not a real plane in context** → **classification unknown**, **realness** painted_or_decoy or uncertain (digitally_modified if pasted/composite). **Only if** it **actually looks like a real aircraft**, assign military or civilian. When it clearly does and mil vs civ is ambiguous, favor military (screening); when it clearly does and shape is non-military → civilian.",
    "Rules (listed detections):",
    "  • Output exactly one JSON row per input id; copy each id string exactly. Do not change bbox numbers for listed ids.",
    "  • Classification enum per plane: civilian | military | unknown (exactly one).",
    "  • DECISION ORDER: (0) **Painted vs real in context** — state in rationale whether it looks **painted/marked/decoy-like** vs **actually like a real plane** on the ramp. If painted/not real in scene → **classification unknown** + non-real **realness**; do not force mil/civ. (1) If **actually real-looking** and clearly non-military (tube airliner, obvious civilian GA/bizjet/helo, no tactical cues) → civilian. (2) If **actually real-looking** and clearly military (tactical jet, tanker, military transport, military helo) → military. (3) If **actually real-looking** but mil/civ not callable → **unknown**. (4) If **actually real-looking**, ambiguous on mil/civ after silhouette, apply Airfield context: lean military at military installations; at commercial airports never force-civilian on tactical shapes.",
    "  • SILHOUETTE (after plausibility): military = compact/tactical jets, swept/delta wings, short fuselage vs wingspan, tankers, strategic/tactical transports, ISR stacks, military helos. civilian = clearly non-military **natural** aircraft. unknown = unnatural crop, not a credible aircraft, or no honest mil/civ.",
    "  • Military airfields (see Airfield context): transports and tankers often look like 'fat' or high-wing bodies in satellite view — do NOT default those to civilian airliners. When shape could reasonably be C-17/C-5/KC-135-class or similar, prefer military.",
    "  • Demo imagery folders: the Airfield context line already uses the **operational** base identity (e.g. SFO or SUU); trust that line for priors, not the on-disk folder code.",
    "  • Never pick civilian for objects that look like fighters or tactical jets just because the scene looks like a busy civilian airport.",
    "  • If unsure between military and civilian but it is clearly a **natural real** aircraft, prefer military when wings dominate and the fuselage is short relative to wingspan.",
    "  • **Realness**: must match your **painted vs real** answer; cite what you saw (flat paint on tarmac vs 3D aircraft, shadow, texture) in rationale.",
    "  • Use apron/terminal context only as a weak tie-breaker after shape, except the explicit military/commercial **type** in Airfield context may break ties on ambiguous transports.",
    "",
    "Additional aircraft (optional — strict bar):",
    "  • After classifying all listed boxes, scan the full image for DISTINCT aircraft that were clearly missed (detector false negative).",
    "  • Only add rows in additionalAircraft when you are HIGHLY confident (additionCertainty >= 0.85) that the object is a real aircraft and was missed — not shadows, stains, or ambiguous blobs.",
    "  • Each addition MUST include a NEW id (added-1, added-2, …), full normalized bbox, angle, detectionConfidence, classification fields, rationale, and additionCertainty.",
    "  • Do NOT duplicate coverage of a listed box; do NOT add speculative aircraft.",
    ...buildClassificationSensitivityPromptLines(),
    "",
    sizeLine,
    "",
    "Pre-detected aircraft (classify each row — keep bbox coordinates unchanged):",
    JSON.stringify(detectionPayload, null, 2),
    "",
    `Airfield context (code — name — type from catalog): ${airportLabel}`,
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

  return limitOpenAiHttpConcurrency(async () => {
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
  });
}

/** Text-only structured JSON (no image) — series summary, etc. */
async function createOpenAIJsonResponseTextOnly({ text, schema, name, logContext = "" }) {
  if (!openaiApiKey || !text) return null;

  return limitOpenAiHttpConcurrency(async () => {
    const ctx = logContext ? ` ${logContext}` : "";

    const requestBody = JSON.stringify({
      model: openaiModel,
      input: [
        {
          role: "user",
          content: [{ type: "input_text", text }],
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
      `[openai:http:text] POST responses${ctx} schema=${name} model=${openaiModel} body_bytes=${Buffer.byteLength(requestBody, "utf8")}`,
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
        `[openai:http:text] error responses${ctx} schema=${name} model=${openaiModel} status=${response.status} body=${errSlice}`,
      );
      throw new Error(`OpenAI ${name} failed: ${response.status} ${errorText}`);
    }

    const payload = await response.json();
    const outputText = extractOutputText(payload);
    if (!outputText) {
      console.warn(`[openai:http:text] schema=${name}${ctx} empty structured output`);
      return null;
    }

    return JSON.parse(outputText);
  });
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

function mapOpenAIClassificationRow(raw, basePlane) {
  const cls = { classification: raw.classification };
  const normalizedClassification = normalizeModelClassification(cls);
  return {
    ...basePlane,
    classification: normalizedClassification,
    type: normalizedClassification,
    classificationConfidence: Math.max(0, Math.min(1, Number(raw.classificationConfidence) || 0.5)),
    realness: ["real", "painted_or_decoy", "digitally_modified", "uncertain"].includes(raw.realness)
      ? raw.realness
      : "uncertain",
    realnessConfidence: Math.max(0, Math.min(1, Number(raw.realnessConfidence) || 0.5)),
    rationale: String(raw.rationale || ""),
    source: basePlane.source === "openai_classify_added" ? "openai_classify_added" : "openai_classify",
  };
}

function mapAdditionalAircraftFromClassification(raw, image, index) {
  const baseDet = normalizeModelDetection(
    {
      id: raw.id,
      bbox: raw.bbox,
      angle: raw.angle,
      detectionConfidence: raw.detectionConfidence,
    },
    image,
    index,
    "openai_classify_added",
  );
  if (!baseDet) return null;
  return mapOpenAIClassificationRow(raw, baseDet);
}

function mergeOpenAiClassificationResults(planeDetections, parsed, image) {
  const rows = Array.isArray(parsed?.planes) ? parsed.planes : [];
  const byId = new Map(rows.map((r) => [String(r.id), r]));
  const merged = planeDetections.map((base, index) => {
    const row = byId.get(String(base.id)) || rows[index];
    if (!row || row.classification === undefined) {
      const [stub] = classify_planes_stub_civilian([base]);
      return stub;
    }
    return mapOpenAIClassificationRow(row, base);
  });

  const extras = Array.isArray(parsed?.additionalAircraft) ? parsed.additionalAircraft : [];
  let idx = merged.length;
  for (const raw of extras) {
    const cert = Number(raw?.additionCertainty);
    if (!Number.isFinite(cert) || cert < 0.85) continue;
    const plane = mapAdditionalAircraftFromClassification(raw, image, idx);
    if (plane) {
      merged.push(plane);
      idx += 1;
    }
  }
  return merged;
}

/** Concurrent HF analyses for the same path share one OpenAI classification request. */
const openaiClassifyInflightByAbsPath = new Map();

async function runOpenAiClassificationForHfOnce(absPath, image, airportCode, planeDetections) {
  const nc = normalizeAirportCode(airportCode);
  const t0 = Date.now();
  console.log(`[openai:classify] start ${nc}/${image.fileName} planes=${planeDetections.length}`);
  try {
    const capturePath = getCaptureFilePath(nc, image);
    const imageDims =
      capturePath && fs.existsSync(capturePath) ? readImageDimensionsFromFile(capturePath) : null;
    const imageUrl = getImageDataUrl(image, airportCode);
    if (!imageUrl) {
      console.warn(`[openai:classify] skip ${nc}/${image.fileName} no image URL`);
      return null;
    }
    const detectionPayload = planeDetections.map((p) => ({
      id: p.id,
      bbox: p.bbox,
      detectionConfidence: p.detectionConfidence,
      angle: p.angle,
      source: p.source,
    }));
    const airportLabel = getAirfieldLabelForClassificationPrompt(airportCode);
    const promptText = buildOpenAiClassificationOnlyPrompt(detectionPayload, imageDims, airportLabel);
    const parsed = await createOpenAIJsonResponse({
      imageUrl,
      name: "aircraft_classify_only",
      schema: getAircraftClassificationOnlySchema(),
      text: promptText,
      logContext: `${airportCode}/${image.fileName} classify`,
    });
    const ms = Date.now() - t0;
    const n = Array.isArray(parsed?.planes) ? parsed.planes.length : 0;
    console.log(`[openai:classify] done ms=${ms} ${nc}/${image.fileName} rows=${n}`);
    return parsed;
  } catch (err) {
    console.warn(
      `[openai:classify] error ms=${Date.now() - t0} ${nc}/${image.fileName} ${err?.message || err}`,
    );
    return null;
  }
}

/** @param {{ skipInflightDedupe?: boolean }} [options] — set skipInflightDedupe to force a new OpenAI call (e.g. reclassify from cache). */
function getOrRunOpenAiClassificationForHf(absPath, image, airportCode, planeDetections, options = {}) {
  if (!options.skipInflightDedupe) {
    const existing = openaiClassifyInflightByAbsPath.get(absPath);
    if (existing) return existing;
  }

  const pending = (async () => {
    try {
      return await runOpenAiClassificationForHfOnce(absPath, image, airportCode, planeDetections);
    } finally {
      if (!options.skipInflightDedupe) {
        openaiClassifyInflightByAbsPath.delete(absPath);
      }
    }
  })();

  if (!options.skipInflightDedupe) {
    openaiClassifyInflightByAbsPath.set(absPath, pending);
  }
  return pending;
}

async function analyze_image_with_openai(image, airportCode) {
  if (!openaiApiKey) return null;
  const normalizedCode = normalizeAirportCode(airportCode);
  const capturePath = getCaptureFilePath(normalizedCode, image);
  const imageDims =
    capturePath && fs.existsSync(capturePath) ? readImageDimensionsFromFile(capturePath) : null;

  const imageUrl = getImageDataUrl(image, airportCode);
  const airportLabel = getAirfieldLabelForClassificationPrompt(airportCode);
  const promptText = `${buildOpenAiAircraftPrompt(imageDims)}\n\nAirfield context (code — name — type from catalog): ${airportLabel}.`;

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
      /** Always align ids/dates with the current manifest/folder row (same cache file can follow file swaps). */
      return mergeCachedAnalysisWithImage(cached, image);
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

  const detectionSnapshotForCache =
    usedHf && planeDetections.length > 0
      ? {
          imageId: image.id,
          capturedAt: image.capturedAt,
          status: "analyzed",
          provider: "hf_yolo",
          model: getHfWeightsLabelForAnalysis(),
          bboxCoordinateSystem: "normalized",
          planes: planeDetections.map((p) => ({ ...p })),
        }
      : null;

  let classifiedPlanes;
  let usedOpenAiClassify = false;
  let openAiClassificationParsed = null;

  const wantOpenAiClassify =
    openaiApiKey &&
    process.env.HF_OPENAI_CLASSIFY !== "0" &&
    usedHf &&
    planeDetections.length > 0;

  if (wantOpenAiClassify) {
    openAiClassificationParsed = await getOrRunOpenAiClassificationForHf(
      absPath,
      image,
      airportCode,
      planeDetections,
    );
    if (openAiClassificationParsed && Array.isArray(openAiClassificationParsed.planes)) {
      classifiedPlanes = mergeOpenAiClassificationResults(planeDetections, openAiClassificationParsed, image);
      usedOpenAiClassify = true;
    }
  }
  if (!classifiedPlanes) {
    classifiedPlanes = classify_planes_stub_civilian(planeDetections);
  }

  const analysis = {
    imageId: image.id,
    capturedAt: image.capturedAt,
    status: usedHf ? "analyzed" : hfRequested ? "degraded" : "scaffold",
    provider: usedOpenAiClassify
      ? "hf_yolo_openai"
      : usedHf
        ? "hf_yolo"
        : hfRequested
          ? "hf_yolo_failed"
          : "scaffold",
    model: usedHf || hfRequested ? getHfWeightsLabelForAnalysis() : "scaffold",
    bboxCoordinateSystem: "normalized",
    planes: classifiedPlanes,
    ...(usedOpenAiClassify && openAiClassificationParsed
      ? {
          imageQuality: openAiClassificationParsed.imageQuality,
          classificationNotes: openAiClassificationParsed.notes,
          classificationModel: openaiModel,
        }
      : {}),
  };

  if (usedOpenAiClassify && detectionSnapshotForCache) {
    writeClassificationCache(airportCode, image, {
      schemaVersion: 1,
      cachedAt: new Date().toISOString(),
      openaiModel,
      classificationSchema: "aircraft_classify_only_v9-naturalness",
      analysisProviderKey: getAnalysisProviderKey(),
      airportCode: normalizedCode,
      imageId: image.id,
      fileName: image.fileName,
      absImagePath: absPath,
      detectionSnapshot: detectionSnapshotForCache,
      openaiClassificationResponse: openAiClassificationParsed,
      mergedPlanes: classifiedPlanes,
      finalAnalysis: { ...analysis },
    });
  }

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

async function reclassifyFromCachedAnalysis(airportCode, image) {
  const normalizedCode = normalizeAirportCode(airportCode);
  const cached = readCachedAnalysis(normalizedCode, image);
  if (!cached) {
    return { ok: false, error: "no_cached_analysis" };
  }
  const provider = cached.provider;
  if (provider !== "hf_yolo" && provider !== "hf_yolo_openai") {
    return { ok: false, error: "not_hf_cached", provider };
  }
  const planesRaw = cached.planes;
  if (!Array.isArray(planesRaw) || planesRaw.length === 0) {
    return { ok: false, error: "no_planes_in_cache" };
  }

  if (!openaiApiKey) {
    return { ok: false, error: "openai_key_missing" };
  }
  if (process.env.HF_OPENAI_CLASSIFY === "0") {
    return { ok: false, error: "hf_openai_classify_disabled" };
  }

  const imagePath = getCaptureFilePath(normalizedCode, image);
  const absPath =
    imagePath && fs.existsSync(imagePath) ? resolveRealImagePath(path.resolve(imagePath)) : "";
  if (!absPath || !fs.existsSync(absPath)) {
    return { ok: false, error: "image_file_missing", path: imagePath || "" };
  }

  const planeDetections = planesRaw
    .map((p, index) => normalizeModelDetection(p, image, index, "hf_yolo"))
    .filter(Boolean);

  if (!planeDetections.length) {
    return { ok: false, error: "no_valid_plane_boxes" };
  }

  const openAiClassificationParsed = await getOrRunOpenAiClassificationForHf(
    absPath,
    image,
    normalizedCode,
    planeDetections,
    { skipInflightDedupe: true },
  );

  let classifiedPlanes;
  let usedOpenAiClassify = false;
  if (openAiClassificationParsed && Array.isArray(openAiClassificationParsed.planes)) {
    classifiedPlanes = mergeOpenAiClassificationResults(planeDetections, openAiClassificationParsed, image);
    usedOpenAiClassify = true;
  }
  if (!classifiedPlanes) {
    classifiedPlanes = classify_planes_stub_civilian(planeDetections);
  }

  const detectionSnapshotForCache = {
    imageId: image.id,
    capturedAt: image.capturedAt,
    status: "analyzed",
    provider: "hf_yolo",
    model: getHfWeightsLabelForAnalysis(),
    bboxCoordinateSystem: "normalized",
    planes: planeDetections.map((p) => ({ ...p })),
  };

  const analysis = {
    imageId: image.id,
    capturedAt: image.capturedAt,
    status: cached.status === "degraded" ? "degraded" : "analyzed",
    provider: usedOpenAiClassify ? "hf_yolo_openai" : "hf_yolo",
    model: getHfWeightsLabelForAnalysis(),
    bboxCoordinateSystem: "normalized",
    planes: classifiedPlanes,
    ...(usedOpenAiClassify && openAiClassificationParsed
      ? {
          imageQuality: openAiClassificationParsed.imageQuality,
          classificationNotes: openAiClassificationParsed.notes,
          classificationModel: openaiModel,
          reclassifiedAt: new Date().toISOString(),
        }
      : {}),
  };

  if (usedOpenAiClassify && detectionSnapshotForCache) {
    writeClassificationCache(normalizedCode, image, {
      schemaVersion: 1,
      cachedAt: new Date().toISOString(),
      trigger: "reclassify",
      openaiModel,
      classificationSchema: "aircraft_classify_only_v9-naturalness",
      analysisProviderKey: getAnalysisProviderKey(),
      airportCode: normalizedCode,
      imageId: image.id,
      fileName: image.fileName,
      absImagePath: absPath,
      detectionSnapshot: detectionSnapshotForCache,
      openaiClassificationResponse: openAiClassificationParsed,
      mergedPlanes: classifiedPlanes,
      finalAnalysis: { ...analysis },
    });
  }

  writeCachedAnalysis(normalizedCode, image, analysis);
  console.log(
    `[analysis:reclassify] ${normalizedCode}/${image.fileName} provider=${analysis.provider} detections=${analysis.planes.length}`,
  );
  return { ok: true, analysis };
}

async function reclassifyAirfieldFromCache(airportCode) {
  const normalizedCode = normalizeAirportCode(airportCode);
  const images = get_images_over_time(normalizedCode);
  const rows = await mapWithConcurrency(images, getImageAnalysisConcurrency(), (image) =>
    reclassifyFromCachedAnalysis(normalizedCode, image),
  );

  const summary = {
    airportCode: normalizedCode,
    imageTotal: images.length,
    updated: 0,
    skipped: [],
  };

  for (let i = 0; i < images.length; i += 1) {
    const r = rows[i];
    const img = images[i];
    if (r.ok) {
      summary.updated += 1;
    } else {
      summary.skipped.push({
        imageId: img.id,
        fileName: img.fileName,
        reason: r.error,
        ...(typeof r.provider === "string" ? { provider: r.provider } : {}),
      });
    }
  }

  return summary;
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
    civilian: 0,
    military: 0,
    unknown: 0,
  };

  planes.forEach((plane) => {
    const key = normalizeClassificationString(plane.classification ?? plane.type);
    counts[key] = (counts[key] || 0) + 1;
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
  civilian: 0,
  military: 0,
  unknown: 0,
});

const PLANE_CATEGORIES = ["civilian", "military", "unknown"];

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
  let maxKey = null;
  for (const key of PLANE_CATEGORIES) {
    const dv = deltas[key] || 0;
    if (Math.abs(dv) > maxAbs) {
      maxAbs = Math.abs(dv);
      maxKey = key;
    }
  }

  if (previousAnalysis && maxAbs > 0 && maxKey != null) {
    const dv = deltas[maxKey] || 0;
    signals.push({
      label: "Largest type swing",
      value: `${maxKey} ${dv > 0 ? "+" : ""}${dv}`,
      severity: maxAbs >= 5 ? "high" : maxAbs >= 2 ? "medium" : "low",
    });
  } else {
    let bestKey = null;
    let bestN = -1;
    for (const key of PLANE_CATEGORIES) {
      const n = latestComposition[key] || 0;
      if (n > bestN) {
        bestN = n;
        bestKey = key;
      }
    }
    const pct =
      latestTotal && bestKey != null ? Math.round(((latestComposition[bestKey] || 0) / latestTotal) * 100) : 0;
    signals.push({
      label: "Dominant type (latest)",
      value: bestKey != null ? `${bestKey} · ${pct}% of fleet` : "—",
      severity: "low",
    });
  }

  const provider = latestAnalysis?.provider || "unknown";
  const pipelineParts = [];
  if (provider === "openai") pipelineParts.push("OpenAI vision");
  else if (provider === "hf_yolo_openai") pipelineParts.push("HF YOLO + OpenAI classify");
  else if (provider === "hf_yolo") pipelineParts.push("HF YOLO");
  else if (provider === "hf_yolo_failed") pipelineParts.push("HF detector failed (degraded)");
  else if (provider === "openai_failed") pipelineParts.push("OpenAI failed (scaffold fallback)");
  else if (provider === "scaffold") pipelineParts.push("Scaffold / heuristic");
  else pipelineParts.push(String(provider));

  if (
    latestAnalysis?.imageQuality &&
    (provider === "openai" || provider === "hf_yolo_openai")
  ) {
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
  const unkType = latestComposition.unknown || 0;
  const unkShare = latestTotal ? unkType / latestTotal : 0;
  const ambParts = [];
  if (lowDet) ambParts.push(`${lowDet} low detection confidence`);
  if (ambReal) ambParts.push(`${ambReal} ambiguous realness (vision)`);
  if (unkType)
    ambParts.push(
      `${unkType} unknown mil/civ type (${Math.round(unkShare * 100)}% of latest)`,
    );

  signals.push({
    label: "Ambiguity",
    value: ambParts.length ? ambParts.join("; ") : "None flagged",
    severity:
      lowDet >= 3 || ambReal || unkType >= 4 || unkShare >= 0.35 ? "medium" : "low",
  });

  return signals;
}

function getAirfieldSeriesSummarySchema() {
  return {
    type: "object",
    additionalProperties: false,
    required: ["headline", "narrative", "unexpectedObservations", "confidence", "anomalous"],
    properties: {
      headline: {
        type: "string",
        description:
          "One short line for the operator. If nothing is surprising given the airfield type, say so plainly (e.g. patterns look typical).",
      },
      narrative: {
        type: "string",
        description:
          "2–5 sentences. Lead with the LATEST capture (most recent row): how its **civilian / military / unknown** counts and mix compare to the rest of the series and to what this airfield type would normally show. If the latest capture or recent history has a **large count or high fraction of unknown** (type could not be called military or civilian), explicitly call that out as **worth human attention** (imagery quality, possible decoys, or detector noise). Then summarize multi-date context (trends, sustained shifts, or stability). Focus ONLY on what is unexpected or actionable. Do NOT name models, software, or detection settings. Avoid framing as only 'vs the immediately previous row'—use the whole history for context. It is NOT concerning that a military base shows military aircraft, or that a busy civilian airport shows mostly civilian traffic.",
      },
      anomalous: {
        type: "boolean",
        description:
          "True when a human analyst should review this series or the latest snapshot: e.g. latest departs meaningfully from historical pattern for this airfield type, counts are implausible, **or unknown-classification counts are high** (many detections where military vs civilian could not be judged — deserves attention). False when behavior looks ordinary for this site type including normal traffic volume and low unknown share.",
      },
      unexpectedObservations: {
        type: "array",
        description:
          "Only items a user should act on or understand as anomalous. Empty if the series is unremarkable for this airfield type. Do not list 'normal' military at military fields or normal civilian traffic at commercial hubs. Include items here when **unknown** counts spike (type unclear) in a way that merits review.",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["title", "severity", "explanation"],
          properties: {
            title: { type: "string" },
            severity: { type: "string", enum: ["low", "medium", "high"] },
            explanation: { type: "string" },
          },
        },
      },
      confidence: { type: "number", description: "0–1 confidence in this assessment." },
    },
  };
}

function buildAirfieldSeriesSummaryPrompt(report, images, imageAnalyses) {
  const identity = getAirfieldCatalogIdentityForSeriesSummary(report);
  const code = String(identity?.code || report.code || "").trim();
  const rows = images.map((img, i) => {
    const a = imageAnalyses[i];
    const comp = a ? countPlaneTypes(a.planes || []) : { civilian: 0, military: 0, unknown: 0 };
    return {
      fileName: img.fileName,
      capturedAt: img.capturedAt,
      civilian: comp.civilian,
      military: comp.military,
      unknown: comp.unknown,
      total: sumFleetComposition(comp),
    };
  });

  return [
    "You write briefings for people monitoring airfields from satellite-style imagery.",
    "You are given **civilian, military, and unknown** counts per dated capture (`unknown` = type could not be judged military vs civilian), ordered in time. The LAST row is the latest snapshot—compare it against ALL earlier rows as historical context. Your audience cares about SURPRISES relative to what this airfield is for—not about listing normal operations.",
    "",
    "Expectations (do NOT flag these as problems):",
    "  • Military airfields / AFB: military-type aircraft and military-heavy mixes are normal.",
    "  • Major civilian commercial airports: high counts and civilian-dominated mixes are normal.",
    "  • Composition matching the airfield’s role is typically NOT news.",
    "",
    "What IS worth calling out (if supported by the counts):",
    "  • Total aircraft activity that seems wildly out of scale for the site, or a sustained trend that departs from a plausible baseline for that airfield type.",
    "  • Civilian vs military mix that is surprising given the stated airfield profile (e.g. large military-type presence at a purely civilian hub, or patterns that contradict the base mission—use judgment).",
    "  • **Many unknown classifications** on the latest capture or sustained across recent dates — treat as **attention-worthy** (unclear silhouettes, possible decoys/markings, poor imagery, or detector issues). Suggest setting **anomalous** true when unknown count or share is clearly elevated vs the rest of the series or vs a reasonable baseline.",
    "  • Erratic or implausible swings across the series that suggest bad data rather than real movement—but describe as 'worth verifying' without naming tools.",
    "",
    "Do NOT: mention model names, APIs, software, detector settings, or pipeline steps. Do NOT structure the narrative around consecutive capture pairs or 'vs the previous image'. You may describe multi-date trends (e.g. build-up over several captures).",
    "",
    `Airport code: ${code}`,
    `Airfield type (from catalog): ${identity?.type || report.type || "unknown"}`,
    `Name: ${identity?.name || report.name || ""}`,
    `Number of dated captures in this series: ${images.length}`,
    "",
    "Counts per capture (civilian / military / unknown / total detections). The array is chronological; the last object is the latest snapshot to compare against all earlier ones:",
    JSON.stringify(rows, null, 2),
    "",
    "Set anomalous to true only when a human should take a serious look (latest vs history supports that call). If everything looks ordinary for this airfield type, set anomalous to false.",
    "",
    "Respond with structured JSON only. Use an empty unexpectedObservations array when nothing is genuinely surprising for this airfield type.",
  ].join("\n");
}

function computeSeriesSummaryFingerprint(report, images, imageAnalyses) {
  const rows = images.map((img, i) => {
    const a = imageAnalyses[i];
    const comp = a ? countPlaneTypes(a.planes || []) : { civilian: 0, military: 0, unknown: 0 };
    return {
      fileName: img.fileName,
      civilian: comp.civilian,
      military: comp.military,
      unknown: comp.unknown,
      total: sumFleetComposition(comp),
    };
  });
  rows.sort((a, b) => a.fileName.localeCompare(b.fileName));

  const payload = {
    summaryV: SERIES_SUMMARY_PROMPT_VERSION,
    analysisKey: getAnalysisProviderKey(),
    airport: normalizeAirportCode(report.code),
    airfieldType: report.type || "",
    captures: rows,
  };
  return crypto.createHash("sha256").update(JSON.stringify(payload)).digest("hex");
}

function getSeriesSummaryCachePath(airportCode, fingerprint) {
  const ac = normalizeAirportCode(airportCode);
  const dir = path.join(summaryCacheRoot, ac);
  if (!dir.startsWith(summaryCacheRoot)) return "";
  return path.join(dir, `${fingerprint}.json`);
}

function readSeriesSummaryCache(airportCode, fingerprint) {
  const p = getSeriesSummaryCachePath(airportCode, fingerprint);
  if (!p || !fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return null;
  }
}

function writeSeriesSummaryCache(airportCode, fingerprint, envelope) {
  const p = getSeriesSummaryCachePath(airportCode, fingerprint);
  if (!p) return;
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(envelope, null, 2));
}

async function runAirfieldSeriesSummary(report, images, imageAnalyses, options = {}) {
  if (!openaiApiKey || !images.length) return null;

  const fingerprint = computeSeriesSummaryFingerprint(report, images, imageAnalyses);
  const bypassSummaryCache =
    options.refreshAnalysis === true || options.refreshSeriesSummary === true;

  if (!bypassSummaryCache) {
    const cached = readSeriesSummaryCache(report.code, fingerprint);
    if (cached?.openaiResponse) {
      const o = cached.openaiResponse;
      return {
        ...o,
        anomalous: o.anomalous === true,
        summaryFingerprint: fingerprint,
        summaryFromCache: true,
      };
    }
  }

  try {
    const text = buildAirfieldSeriesSummaryPrompt(report, images, imageAnalyses);
    const parsed = await createOpenAIJsonResponseTextOnly({
      text,
      name: "airfield_series_summary",
      schema: getAirfieldSeriesSummarySchema(),
      logContext: `${report.code}/series-summary`,
    });
    if (!parsed) return null;

    writeSeriesSummaryCache(report.code, fingerprint, {
      cachedAt: new Date().toISOString(),
      fingerprint,
      promptVersion: SERIES_SUMMARY_PROMPT_VERSION,
      analysisProviderKey: getAnalysisProviderKey(),
      openaiModel,
      openaiResponse: parsed,
    });

    return {
      ...parsed,
      anomalous: parsed.anomalous === true,
      summaryFingerprint: fingerprint,
      summaryFromCache: false,
    };
  } catch (e) {
    console.warn(`[openai:series-summary] ${report.code} ${e?.message || e}`);
    return null;
  }
}

function describeAnalysisPipelineSentence(provider) {
  if (provider === "openai") return "Pipeline: OpenAI vision on the full image.";
  if (provider === "hf_yolo_openai") return "Pipeline: HF YOLO regions plus OpenAI classification.";
  if (provider === "hf_yolo") return "Pipeline: HF YOLO detection; classification may be stubbed.";
  if (provider === "hf_yolo_failed") return "Pipeline: HF detector failed — labels are degraded.";
  if (provider === "openai_failed") return "Pipeline: OpenAI failed — heuristic fallback.";
  if (provider === "scaffold") return "Pipeline: scaffold analysis (no full vision pass).";
  return `Pipeline: ${provider || "unknown"}.`;
}

function deriveLiveInspectorFields({
  report,
  images,
  latestAnalysis,
  previousAnalysis,
  latestComposition,
  openAiSeriesSummary,
}) {
  const code = String(report.code || "").toLowerCase();
  const lastImg = images.at(-1);

  if (!latestAnalysis) {
    return {
      summary: `${images.length} local image${images.length === 1 ? "" : "s"} under data/airfields/${code}/; latest analysis is not available yet.`,
      status: "",
      riskBand: "needs_review",
      findings: [],
      baseline: `Local files present (${images.length}); awaiting reliable analysis.`,
      lastCapture: lastImg?.capturedAt ? String(lastImg.capturedAt) : report.lastCapture,
    };
  }

  const total = sumFleetComposition(latestComposition);

  const summaryParts = [
    `${images.length} dated local capture${images.length === 1 ? "" : "s"}.`,
    `Latest frame: ${total} detection${total === 1 ? "" : "s"} (civilian ${latestComposition.civilian || 0}, military ${
      latestComposition.military || 0
    }, unknown ${latestComposition.unknown || 0}).`,
  ];

  let summary = summaryParts.join(" ");
  if (openAiSeriesSummary?.narrative) {
    const head = openAiSeriesSummary.headline ? `${openAiSeriesSummary.headline} ` : "";
    summary = `${head}${openAiSeriesSummary.narrative}`.trim();
  }

  const prov = latestAnalysis.provider || "";
  let riskBand = "normal";
  let status = "";
  if (prov.includes("failed") || prov === "scaffold") {
    riskBand = "needs_review";
  }

  const baseline = `Local evidence only (${images.length} file${
    images.length === 1 ? "" : "s"
  }); not a published operational baseline.`;

  return {
    summary,
    status,
    riskBand,
    findings: [],
    baseline,
    lastCapture: lastImg?.capturedAt ? String(lastImg.capturedAt) : report.lastCapture,
  };
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

  const analysisSource = imageAnalyses.find(
    (analysis) => analysis.provider === "hf_yolo" || analysis.provider === "hf_yolo_openai",
  )
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

  const seriesSummary =
    images.length > 0 ? await runAirfieldSeriesSummary(report, images, imageAnalyses, options) : null;

  const liveInspector =
    images.length > 0
      ? deriveLiveInspectorFields({
          report,
          images,
          latestAnalysis,
          previousAnalysis,
          latestComposition,
          openAiSeriesSummary: seriesSummary,
        })
      : null;

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
    seriesSummary,
    seriesSummaryFingerprint: seriesSummary?.summaryFingerprint ?? null,
    ...(liveInspector || {}),
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

  if (req.method === "POST" && url.pathname.startsWith("/api/airfields/")) {
    const segments = url.pathname.split("/").filter(Boolean);
    if (
      segments.length === 4 &&
      segments[0] === "api" &&
      segments[1] === "airfields" &&
      segments[3] === "reclassify"
    ) {
      const airportCode = normalizeAirportCode(segments[2]);
      const report = airfieldReports[airportCode];
      if (!report) {
        sendJson(res, 404, { error: "Airfield not found." });
        return;
      }
      if (getAircraftAnalysisProvider() !== "hf") {
        sendJson(res, 400, {
          error: "Reclassify applies only when AIRCRAFT_ANALYSIS_PROVIDER=hf (HF boxes + optional OpenAI classify).",
        });
        return;
      }
      try {
        const result = await reclassifyAirfieldFromCache(airportCode);
        sendJson(res, 200, result);
      } catch (e) {
        sendJson(res, 500, { error: e.message || "Reclassify failed." });
      }
      return;
    }
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
    const refreshSeriesSummary =
      url.searchParams.get("summaryRefresh") === "1" ||
      url.searchParams.get("summaryRefresh") === "true";
    sendJson(
      res,
      200,
      await attachImageSeries(report, { refreshAnalysis: refresh, refreshSeriesSummary }),
    );
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
  if (provider === "hf") {
    const quick = process.env.HF_AIRCRAFT_QUICK === "1";
    console.log(
      `[analysis] HF detector mode: ${quick ? "QUICK (COCO yolov8n — set HF_AIRCRAFT_QUICK=0 for performance)" : "performance (Efficient-YOLO-RS — needs Hugging Face download + third_party/yolov9)"}`,
    );
  }
  console.log(
    `[analysis] OpenAI: calls only if path cache missing or GET /api/airfields/:code?refresh=1. Legacy HF/scaffold disk cache reads: ${
      isAnalysisCacheReadEnabled() ? "on" : "off (--no-analysis-cache or ANALYSIS_CACHE=0)"
    }.`,
  );
});
