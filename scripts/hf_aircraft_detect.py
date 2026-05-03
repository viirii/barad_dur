#!/usr/bin/env python3
"""
Batch airplane detection for satellite/airfield imagery using Ultralytics YOLO weights
from Hugging Face (default: iturslab/Efficient-YOLO-RS-Airplane-Detection).

Reads local image paths from argv, prints JSON to stdout only:
  { "<resolved_path>": { "detections": [ ... ] }, ... }

Env:
  Defaults live only in repo-root `.env.example` (optional `.env` overrides). This script loads those files at startup
  with the same order as server.js: `.env.example` first, then `.env` overwrites. Pass-through env (e.g. from the
  Node-spawned child process) is left unchanged except for missing keys filled from those files.

  HF_AIRCRAFT_QUICK — 1 uses HF_AIRCRAFT_QUICK_* keys (COCO yolov8n); 0 uses HF_AIRCRAFT_REPO / HF_AIRCRAFT_WEIGHTS / …
  HF_QUICK_KEEP_CLASSES — quick mode only: comma-separated COCO names or ids; `all` or `*` disables class filtering.
  HF_DETECT_DEBUG — set to 1 for stderr diagnostics.
  YOLOV9_REPO (optional) — WongKinYiu/yolov9 clone path (models/common.py). Default: <repo>/third_party/yolov9.
  MPLCONFIGDIR — optional; server may set this for matplotlib cache.
"""

from __future__ import annotations

import inspect
import io
import json
import os
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def _merge_dotenv_file(path: Path, *, override_existing: bool) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if override_existing:
            os.environ[key] = val
        else:
            os.environ.setdefault(key, val)


def _hydrate_env_from_dotenv_files() -> None:
    """Match server.js: `.env.example` then `.env` (override)."""
    root = Path(__file__).resolve().parents[1]
    example = root / ".env.example"
    local = root / ".env"
    if example.is_file():
        _merge_dotenv_file(example, override_existing=False)
    if local.is_file():
        _merge_dotenv_file(local, override_existing=True)


def _env(key: str) -> str:
    v = os.environ.get(key)
    if v is None or v == "":
        print(
            json.dumps({"error": f"missing env {key}; copy defaults from repo .env.example"}),
            file=sys.stderr,
        )
        sys.exit(2)
    return v


def _ensure_bgr_uint8(im_bgr):
    """Orthophotos are often 16‑bit or float; Ultralytics expects uint8 BGR. Returns None if unusable."""
    import numpy as np

    if im_bgr is None or im_bgr.size == 0:
        return None
    if im_bgr.ndim == 2:
        import cv2

        im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_GRAY2BGR)
    elif im_bgr.ndim == 3 and im_bgr.shape[2] == 4:
        import cv2

        im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_BGRA2BGR)
    elif im_bgr.ndim != 3 or im_bgr.shape[2] != 3:
        return None

    if im_bgr.dtype == np.uint8:
        return im_bgr

    x = im_bgr.astype(np.float32)
    hi = float(np.max(x))
    lo = float(np.min(x))
    if hi <= lo:
        return None
    if hi <= 1.01:
        x = np.clip(x, 0.0, 1.0) * 255.0
    elif hi <= 255.5:
        x = np.clip(x, 0.0, 255.0)
    else:
        # 16‑bit / HDR: stretch to 8‑bit
        x = (x - lo) / (hi - lo) * 255.0
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _nms_xyxy(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_threshold: float = 0.45,
) -> list[int]:
    """Greedy NMS; boxes/scores same length."""
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou_xyxy(boxes[i], boxes[j]) < iou_threshold]
    return keep


def _tile_starts(dim: int, tile: int, stride: int) -> list[int]:
    """Start indices for sliding windows along one axis."""
    if dim <= tile:
        return [0]
    starts: list[int] = []
    pos = 0
    while pos + tile < dim:
        starts.append(pos)
        pos += stride
    last = dim - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def _stride_up(n: int, stride: int = 32) -> int:
    """YOLO stride alignment (Ultralytics expects multiples of 32)."""
    return max(stride, ((max(n, 1) + stride - 1) // stride) * stride)


def _model_class_id_to_name(model) -> dict[int, str]:
    names = getattr(model, "names", None) or {}
    out: dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            out[i] = str(v)
    return out


def resolve_quick_keep_class_ids(model) -> frozenset[int] | None:
    """Quick + COCO only: restrict outputs to these class indices. None = no filter (all 80 COCO classes)."""
    raw = _env("HF_QUICK_KEEP_CLASSES").strip()
    if raw.lower() in ("*", "all"):
        return None
    id_to_name = _model_class_id_to_name(model)
    keep: set[int] = set()
    for token in [t.strip() for t in raw.split(",") if t.strip()]:
        if token.isdigit():
            keep.add(int(token))
            continue
        tl = token.lower()
        for cid, nm in id_to_name.items():
            if nm.lower() == tl:
                keep.add(cid)
                break
    if not keep:
        for cid, nm in id_to_name.items():
            if nm.lower() == "airplane":
                keep.add(cid)
        if not keep:
            keep.add(4)
            print(
                "[hf:quick] HF_QUICK_KEEP_CLASSES matched nothing and no 'airplane' in model.names — "
                "using COCO airplane index 4",
                file=sys.stderr,
            )
    return frozenset(keep)


def _detections_from_yolo_result(
    r,
    image_w: int,
    image_h: int,
    keep_cls: frozenset[int] | None,
    max_area_frac: float | None,
) -> list[dict]:
    """Turn one Ultralytics Results object into normalized bbox dicts (optional class filter)."""
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    cls_arr = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None
    if keep_cls is not None and cls_arr is None:
        return []
    fw = float(max(image_w, 1))
    fh = float(max(image_h, 1))
    out: list[dict] = []
    for i in range(len(xyxy)):
        if keep_cls is not None and cls_arr is not None:
            if int(cls_arr[i]) not in keep_cls:
                continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        score = float(scores[i])
        out.append(
            {
                "bbox": {
                    "x": x1 / fw,
                    "y": y1 / fh,
                    "width": (x2 - x1) / fw,
                    "height": (y2 - y1) / fh,
                },
                "detectionConfidence": max(0.0, min(1.0, score)),
                "angle": 0,
            }
        )
    return _apply_max_area_filter(out, max_area_frac)


def _resolve_max_box_area_frac() -> float | None:
    """Optional cap on normalized bbox area (width×height); None = disabled."""
    raw = _env("HF_AIRCRAFT_MAX_BOX_AREA").strip().lower()
    if raw in ("none", "off", "disable", "1", "1.0"):
        return None
    try:
        v = float(raw)
    except ValueError as e:
        raise ValueError(f"HF_AIRCRAFT_MAX_BOX_AREA must be a number or 1/off; got {raw!r}") from e
    if v >= 1.0:
        return None
    if v <= 0:
        return None
    return v


def _apply_max_area_filter(detections: list[dict], max_frac: float | None) -> list[dict]:
    if max_frac is None:
        return detections
    kept: list[dict] = []
    for d in detections:
        bbox = d.get("bbox") or {}
        w = float(bbox.get("width") or 0)
        h = float(bbox.get("height") or 0)
        if w * h <= max_frac:
            kept.append(d)
    return kept


def _resolve_inference_plan(
    full_w: int,
    full_h: int,
    imgsz_env: str,
    imgsz_max: int,
    tile_enabled: bool,
) -> tuple[str, int, int | None]:
    """Returns (mode, single_imgsz_or_ignored, tile_sz_or_none). mode is 'single' or 'tile'."""
    long_edge = max(full_w, full_h)
    spec = (imgsz_env or "native").strip().lower()
    cap = _stride_up(max(imgsz_max, 1), 32)

    if spec not in ("native", "0", ""):
        fixed = _stride_up(int(spec), 32)
        if long_edge <= fixed:
            return "single", fixed, None
        if tile_enabled:
            return "tile", fixed, fixed
        return "single", fixed, None

    # native: one forward pass at ~full resolution when it fits under cap
    if long_edge <= imgsz_max:
        single = min(_stride_up(long_edge, 32), cap)
        return "single", single, None

    if tile_enabled:
        return "tile", cap, cap
    return "single", cap, None


def _predict_tiled_bgr(
    model,
    im_bgr,
    full_w: int,
    full_h: int,
    tile_sz: int,
    overlap: float,
    predict_kw: dict,
    keep_cls: frozenset[int] | None,
    merge_iou: float,
    max_area_frac: float | None,
) -> list[dict]:
    """Run YOLO on overlapping crops in original pixel space; merge with NMS."""
    overlap = min(max(overlap, 0.0), 0.9)
    stride = max(1, int(tile_sz * (1.0 - overlap)))
    xs = _tile_starts(full_w, tile_sz, stride)
    ys = _tile_starts(full_h, tile_sz, stride)

    all_xyxy: list[tuple[float, float, float, float]] = []
    all_scores: list[float] = []

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile_sz, full_h)
            x1 = min(x0 + tile_sz, full_w)
            crop = im_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            results = model.predict(crop, imgsz=tile_sz, **predict_kw)
            if not results:
                continue
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None
            if keep_cls is not None and cls_arr is None:
                continue
            for k in range(len(xyxy)):
                if keep_cls is not None and cls_arr is not None:
                    if int(cls_arr[k]) not in keep_cls:
                        continue
                bx1, by1, bx2, by2 = xyxy[k].tolist()
                scv = float(sc[k])
                all_xyxy.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0))
                all_scores.append(scv)

    if not all_xyxy:
        return []

    keep = _nms_xyxy(all_xyxy, all_scores, iou_threshold=merge_iou)
    out: list[dict] = []
    fw, fh = float(full_w), float(full_h)
    for idx in keep:
        x1, y1, x2, y2 = all_xyxy[idx]
        scv = all_scores[idx]
        out.append(
            {
                "bbox": {
                    "x": x1 / fw,
                    "y": y1 / fh,
                    "width": max(0.0, (x2 - x1) / fw),
                    "height": max(0.0, (y2 - y1) / fh),
                },
                "detectionConfidence": max(0.0, min(1.0, scv)),
                "angle": 0,
            }
        )
    return _apply_max_area_filter(out, max_area_frac)


def _configure_matplotlib_cache() -> None:
    """Avoid writing to ~/.matplotlib (permissions / sandbox); match server-spawned MPLCONFIGDIR."""
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache = Path(__file__).resolve().parent.parent / ".matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)


def _ensure_yolov9_pickling_compat() -> None:
    """Efficient-YOLO-RS `/training/experiment-*/best.pt` files pickle YOLOv9 classes as `models.common.*`.

    Those live in https://github.com/WongKinYiu/yolov9 (not in Ultralytics). Put that repo at:
      <barad_dur>/third_party/yolov9
    or set YOLOV9_REPO / YOLOV9_PATH to its root (directory that contains `models/common.py`).
    """
    env_root = os.environ.get("YOLOV9_REPO") or os.environ.get("YOLOV9_PATH")
    barad_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append(barad_root / "third_party" / "yolov9")

    for root in candidates:
        if (root / "models" / "common.py").is_file():
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            return

    print(
        json.dumps(
            {
                "error": (
                    "YOLOv9 compatibility: Efficient-YOLO-RS weights pickle `models.common` / `models.yolo` "
                    "(WongKinYiu YOLOv9). Clone https://github.com/WongKinYiu/yolov9 into third_party/yolov9 "
                    "or set YOLOV9_REPO to that repo root (must contain models/common.py)."
                )
            }
        ),
        file=sys.stderr,
    )
    sys.exit(4)


def _patch_torch_load_for_ultralytics_checkpoints() -> None:
    """Ultralytics .pt checkpoints need full unpickle (weights_only=False).

    PyTorch 2.6+ defaults weights_only=True when unset. Ultralytics usually passes False,
    but TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 in the environment overrides even explicit False
    inside torch.load — always force False here for trusted local checkpoints.
    """
    import torch

    orig = torch.load
    sig = inspect.signature(orig)
    if "weights_only" not in sig.parameters:
        return

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    torch.load = patched  # type: ignore[assignment]


def _patch_yolov9_ultralytics_api(model) -> None:
    """Align YOLOv9 DetectionModel with current Ultralytics Predictor (fuse verbose, forward embed=...)."""
    inner = getattr(model, "model", None)
    if inner is None:
        return

    if hasattr(inner, "fuse"):
        orig_fuse = inner.fuse
        try:
            fuse_has_verbose = "verbose" in inspect.signature(orig_fuse).parameters
        except (TypeError, ValueError):
            fuse_has_verbose = True
        if not fuse_has_verbose:

            def fuse_wrapper(*args, **kwargs):
                kwargs.pop("verbose", None)
                try:
                    return orig_fuse(*args, **kwargs)
                except TypeError:
                    return orig_fuse()

            inner.fuse = fuse_wrapper  # type: ignore[method-assign]

    orig_forward = inner.forward

    def forward_shim(self, x, augment=False, profile=False, visualize=False, **kwargs):
        """Ultralytics expects a single prediction tensor; YOLOv9 Detect returns (pred, feature_list)."""
        kwargs.pop("embed", None)
        kwargs.pop("distilled", None)
        out = orig_forward(x, augment=augment, profile=profile, visualize=visualize)
        if isinstance(out, tuple) and len(out) >= 1:
            return out[0]
        return out

    inner.forward = types.MethodType(forward_shim, inner)


def main() -> None:
    _hydrate_env_from_dotenv_files()
    paths = [Path(p) for p in sys.argv[1:] if p.strip()]
    if not paths:
        print(json.dumps({"error": "no image paths"}), file=sys.stderr)
        sys.exit(2)

    quick = _env("HF_AIRCRAFT_QUICK").strip() == "1"

    conf = float(_env("HF_AIRCRAFT_CONF"))
    tile_enabled = _env("HF_AIRCRAFT_TILE").strip() != "0"
    use_augment = _env("HF_AIRCRAFT_AUGMENT").strip() != "0"

    if quick:
        weights_rel = _env("HF_AIRCRAFT_QUICK_WEIGHTS")
        imgsz_env = _env("HF_AIRCRAFT_QUICK_IMGSZ").strip()
        imgsz_max = int(_env("HF_AIRCRAFT_QUICK_IMGSZ_MAX"))
        tile_overlap = float(_env("HF_AIRCRAFT_QUICK_TILE_OVERLAP"))
        iou = float(_env("HF_AIRCRAFT_QUICK_IOU"))
        max_det = int(_env("HF_AIRCRAFT_QUICK_MAX_DET"))
        merge_iou = float(_env("HF_AIRCRAFT_QUICK_TILE_MERGE_IOU"))
        repo = ""
    else:
        repo = _env("HF_AIRCRAFT_REPO")
        weights_rel = _env("HF_AIRCRAFT_WEIGHTS")
        imgsz_env = _env("HF_AIRCRAFT_IMGSZ").strip()
        imgsz_max = int(_env("HF_AIRCRAFT_IMGSZ_MAX"))
        tile_overlap = float(_env("HF_AIRCRAFT_TILE_OVERLAP"))
        iou = float(_env("HF_AIRCRAFT_IOU"))
        max_det = int(_env("HF_AIRCRAFT_MAX_DET"))
        merge_iou = float(_env("HF_AIRCRAFT_TILE_MERGE_IOU"))
    predict_kw = {
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "augment": use_augment,
        "verbose": False,
    }

    # Same as server spawn env sanitizer — PyTorch can force weights_only=True via env and break YOLO .pt loads.
    for _env_key in ("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        os.environ.pop(_env_key, None)

    _configure_matplotlib_cache()
    if not quick:
        _ensure_yolov9_pickling_compat()

    try:
        if not quick:
            from huggingface_hub import hf_hub_download

        _patch_torch_load_for_ultralytics_checkpoints()
        from ultralytics import YOLO
    except ImportError as e:
        print(json.dumps({"error": f"missing dependency: {e}"}), file=sys.stderr)
        sys.exit(3)

    # Keep stdout clean for JSON-only output consumed by Node.
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        if quick:
            model = YOLO(weights_rel)
        else:
            model_path = hf_hub_download(repo_id=repo, filename=weights_rel)
            model = YOLO(model_path)
            _patch_yolov9_ultralytics_api(model)

    keep_cls: frozenset[int] | None = None
    if quick:
        keep_cls = resolve_quick_keep_class_ids(model)
        id_to_name = _model_class_id_to_name(model)
        if keep_cls is None:
            print(
                "[hf:quick] COCO yolov8n — class filter off (HF_QUICK_KEEP_CLASSES=all); "
                "all COCO categories may appear.",
                file=sys.stderr,
            )
        else:
            labs = [id_to_name.get(i, "?") for i in sorted(keep_cls)]
            print(
                "[hf:quick] COCO yolov8n — detection uses only "
                f"{sorted(keep_cls)} ({', '.join(labs)}). "
                "HF_AIRCRAFT_QUICK=0 uses Efficient-YOLO-RS airplane weights.",
                file=sys.stderr,
            )

    max_box_area_frac = _resolve_max_box_area_frac()

    try:
        import cv2
    except ImportError:
        print(
            json.dumps(
                {"error": "OpenCV (cv2) required to read images / tile large frames; pip install opencv-python-headless"}
            ),
            file=sys.stderr,
        )
        sys.exit(5)

    out: dict[str, dict] = {}
    for raw in paths:
        # Match Node fs.realpathSync so JSON keys align with the server lookup.
        abs_path = os.path.realpath(str(raw))
        detections: list[dict] = []
        try:
            im_raw = cv2.imread(abs_path, cv2.IMREAD_UNCHANGED)
            if im_raw is None:
                out[abs_path] = {"detections": [], "error": "cv2.imread failed"}
                continue
            im_bgr = _ensure_bgr_uint8(im_raw)
            if im_bgr is None:
                out[abs_path] = {
                    "detections": [],
                    "error": "unsupported channels/depth (need RGB/BGR or gray)",
                }
                continue

            full_h, full_w = im_bgr.shape[:2]
            mode, single_imgsz, tile_sz = _resolve_inference_plan(
                full_w, full_h, imgsz_env, imgsz_max, tile_enabled
            )

            with redirect_stdout(sink), redirect_stderr(sink):
                if mode == "tile" and tile_sz is not None:
                    detections = _predict_tiled_bgr(
                        model,
                        im_bgr,
                        full_w,
                        full_h,
                        tile_sz,
                        tile_overlap,
                        predict_kw,
                        keep_cls,
                        merge_iou,
                        max_box_area_frac,
                    )
                else:
                    results = model.predict(im_bgr, imgsz=single_imgsz, **predict_kw)
                    if not results:
                        detections = []
                    else:
                        r = results[0]
                        h, w = r.orig_shape if r.orig_shape is not None else (1, 1)
                        if not h or not w:
                            detections = []
                        else:
                            detections = _detections_from_yolo_result(
                                r,
                                w,
                                h,
                                keep_cls,
                                max_box_area_frac,
                            )
            if os.environ.get("HF_DETECT_DEBUG", "").strip() == "1":
                isz = tile_sz if mode == "tile" and tile_sz is not None else single_imgsz
                print(
                    f"[hf:debug] {os.path.basename(abs_path)} "
                    f"{full_w}x{full_h} mode={mode} imgsz={isz} "
                    f"dets={len(detections)} conf={conf} iou={iou}",
                    file=sys.stderr,
                )
            elif len(detections) == 0 and max(full_w, full_h) >= 512:
                print(
                    "[hf:warn] 0 detections on a large frame — for busy imagery this usually means "
                    "bit-depth/load mismatch, wrong HF_AIRCRAFT_WEIGHTS, or Ultralytics filters; "
                    "set HF_DETECT_DEBUG=1 and compare another training/experiment-*/best.pt",
                    file=sys.stderr,
                )
        except Exception as e:
            out[abs_path] = {"detections": [], "error": str(e)}
            continue

        out[abs_path] = {"detections": detections}

    print(json.dumps(out))


if __name__ == "__main__":
    main()
