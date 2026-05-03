#!/usr/bin/env node
/**
 * Clone data/airfields/<src>/ and per-provider analysis + classification caches
 * so the duplicate airfield reuses HF + OpenAI classify results without re-inference.
 *
 * Usage: node scripts/clone_airfield_data.js <srcCode> <dstCode>
 * Example: node scripts/clone_airfield_data.js sfo sfx
 */
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");

function transformJson(text, src, dst) {
  const srcLower = String(src).trim().toLowerCase();
  const dstLower = String(dst).trim().toLowerCase();
  let out = text;
  out = out.replaceAll(`/airfields/${srcLower}/`, `/airfields/${dstLower}/`);
  out = out.replaceAll(`"airportCode": "${srcLower}"`, `"airportCode": "${dstLower}"`);
  out = out.replaceAll(`${srcLower}-`, `${dstLower}-`);
  return out;
}

function processCacheJsonDir(dir, src, dst) {
  if (!fs.existsSync(dir)) return;
  const names = fs.readdirSync(dir);
  for (const name of names) {
    if (!name.endsWith(".json")) continue;
    const oldPath = path.join(dir, name);
    const body = fs.readFileSync(oldPath, "utf8");
    const next = transformJson(body, src, dst);
    const newName = name.startsWith(`${src}-`) ? name.replace(`${src}-`, `${dst}-`) : name;
    const newPath = path.join(dir, newName);
    if (newPath !== oldPath) {
      fs.writeFileSync(newPath, next, "utf8");
      fs.unlinkSync(oldPath);
    } else {
      fs.writeFileSync(oldPath, next, "utf8");
    }
  }
}

function main() {
  const src = process.argv[2];
  const dst = process.argv[3];
  if (!src || !dst || src === dst) {
    console.error("Usage: node scripts/clone_airfield_data.js <srcCode> <dstCode>");
    process.exit(1);
  }
  const srcLower = src.trim().toLowerCase();
  const dstLower = dst.trim().toLowerCase();

  const airSrc = path.join(root, "data", "airfields", srcLower);
  const airDst = path.join(root, "data", "airfields", dstLower);
  if (!fs.existsSync(airSrc)) {
    console.error(`Missing ${airSrc}`);
    process.exit(1);
  }
  if (fs.existsSync(airDst)) {
    console.error(`Already exists: ${airDst} — remove or pick another dst code.`);
    process.exit(1);
  }

  fs.cpSync(airSrc, airDst, {
    recursive: true,
    filter: (s) => !path.basename(s).startsWith(".DS_Store"),
  });
  console.log(`Copied imagery ${airSrc} → ${airDst}`);

  for (const kind of ["analysis-cache", "classification-cache"]) {
    const base = path.join(root, "data", kind);
    if (!fs.existsSync(base)) continue;
    const subs = fs.readdirSync(base, { withFileTypes: true });
    for (const ent of subs) {
      if (!ent.isDirectory()) continue;
      const providerKey = ent.name;
      const sfoDir = path.join(base, providerKey, srcLower);
      if (!fs.existsSync(sfoDir)) continue;
      const sfxDir = path.join(base, providerKey, dstLower);
      fs.cpSync(sfoDir, sfxDir, { recursive: true });
      processCacheJsonDir(sfxDir, srcLower, dstLower);
      console.log(`Cache ${kind}/${providerKey}: ${srcLower} → ${dstLower} (${fs.readdirSync(sfxDir).length} files)`);
    }
  }

  console.log("Done. Add `airfieldReports` + UI entry for the new code if not already present.");
}

main();
