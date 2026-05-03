# Barad-dûr

A **service** that ingests **satellite images over time** for an airfield and analyzes **aircraft population**, **makeup** (for example how activity splits across types), and **anomaly detection** so operators can see whether recent patterns fit what is expected for that site.

Local captures are read from `data/airfields/<airport>/` (see `data/airfields/README.md`).

---

## Repository layout (high level)

| Area | Role |
|------|------|
| `server.js` | HTTP server, airfield catalog, image discovery, analysis orchestration, optional OpenAI calls, disk caches |
| `app.js` / `index.html` / `styles.css` | Browser UI: airport picker, capture timeline, canvas overlay, composition and activity summary |
| `scripts/hf_aircraft_detect.py` | Subprocess invoked by Node to propose aircraft bounding boxes on each image |
| `data/airfields/` | Per-airport imagery (and optional `manifest.json`); see `data/airfields/README.md` |
| `data/analysis-cache/` / `data/classification-cache/` / `data/summary-cache/` | Persisted per-image analysis, classification audit trails, and cached series summaries (created at runtime) |

---

## Analysis (high level)

Each dated image is **detected**, **classified**, and the full time series is **summarized** so you get counts, makeup, and a view of what looks normal versus worth a second look.

---

## Running locally

```bash
npm install
npm start
```

Then open the URL printed by the server (by default `http://127.0.0.1:3000`; override with `HOST` / `PORT` if needed).

For Python detector setup (when using the local HF path), see `package.json` script `setup-detect` and `scripts/requirements-detect.txt`.
