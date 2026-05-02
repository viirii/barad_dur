#!/usr/bin/env python3
"""
Batch airplane detection for satellite/airfield imagery using Ultralytics YOLO weights
from Hugging Face (default: iturslab/Efficient-YOLO-RS-Airplane-Detection).

Reads local image paths from argv, prints JSON to stdout only:
  { "<resolved_path>": { "detections": [ ... ] }, ... }

Env:
  HF_AIRCRAFT_QUICK  — set to 1 for fast pipeline debugging: Ultralytics COCO `yolov8n.pt` (no HF Hub / YOLOv9 deps),
                         smaller letterbox (1280), lower imgsz_max (4096). Use to verify boxes appear at all.
  HF_AIRCRAFT_REPO   — Hugging Face repo id (default: iturslab/Efficient-YOLO-RS-Airplane-Detection); ignored when QUICK=1 unless you override WEIGHTS.
  HF_AIRCRAFT_WEIGHTS — HF path inside repo (default: training/experiment-61/best.pt), or when QUICK=1 default `yolov8n.pt`
  HF_AIRCRAFT_IMGSZ     — `native` (default): use each image's long edge as inference size (local full-res up to cap).
                           Or a fixed integer (e.g. 2048) for letterbox / tile size.
  HF_AIRCRAFT_IMGSZ_MAX — max long-edge pixels when IMGSZ=native (default: 16384). Larger images use tiling or downscale.
  HF_AIRCRAFT_CONF      — confidence threshold (default: 0.001). Ultralytics default conf filter is high — keep low for recall.
  HF_AIRCRAFT_IOU       — NMS IoU during predict (default: 0.45). Lower = merge fewer overlapping boxes (dense parking).
  HF_AIRCRAFT_MAX_DET   — max boxes per tile / full image (default: 500).
  HF_AIRCRAFT_TILE      — 1 (default): if image exceeds IMGSZ_MAX, slide overlapping tiles; 0: single pass scaled to cap only.
  HF_AIRCRAFT_TILE_OVERLAP — overlap between tiles (default: 0.40; higher = fewer misses at seams, more compute)
  HF_AIRCRAFT_AUGMENT   — 0 (default): set 1 for TTA (slower; some checkpoints behave badly with augment).
  HF_DETECT_DEBUG       — set to 1 to print dtype/shape/detections to stderr (progress debugging).
  YOLOV9_REPO (optional) — path to a WongKinYiu/yolov9 clone (must contain models/common.py).
        Default: <barad_dur>/third_party/yolov9. Required for Efficient-YOLO-RS weights (YOLOv9 pickles use `models.*`).
  MPLCONFIGDIR — optional; server sets this to <repo>/.matplotlib-cache when spawned from Node.
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
            for k in range(len(xyxy)):
                bx1, by1, bx2, by2 = xyxy[k].tolist()
                scv = float(sc[k])
                all_xyxy.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0))
                all_scores.append(scv)

    if not all_xyxy:
        return []

    keep = _nms_xyxy(all_xyxy, all_scores, iou_threshold=0.45)
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
    return out


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
        kwargs.pop("embed", None)
        return orig_forward(x, augment=augment, profile=profile, visualize=visualize)

    inner.forward = types.MethodType(forward_shim, inner)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:] if p.strip()]
    if not paths:
        print(json.dumps({"error": "no image paths"}), file=sys.stderr)
        sys.exit(2)

    quick = os.environ.get("HF_AIRCRAFT_QUICK", "").strip() == "1"

    if quick:
        weights_rel = os.environ.get("HF_AIRCRAFT_WEIGHTS") or "yolov8n.pt"
        imgsz_env = (os.environ.get("HF_AIRCRAFT_IMGSZ") or "1280").strip()
        imgsz_max = int(os.environ.get("HF_AIRCRAFT_IMGSZ_MAX") or "4096")
        tile_enabled = (os.environ.get("HF_AIRCRAFT_TILE") or "1").strip() != "0"
        tile_overlap = float(os.environ.get("HF_AIRCRAFT_TILE_OVERLAP") or "0.25")
        conf = float(os.environ.get("HF_AIRCRAFT_CONF") or "0.15")
        iou = float(os.environ.get("HF_AIRCRAFT_IOU") or "0.5")
        max_det = int(os.environ.get("HF_AIRCRAFT_MAX_DET") or "300")
        use_augment = (os.environ.get("HF_AIRCRAFT_AUGMENT") or "0").strip() != "0"
        repo = ""
    else:
        repo = os.environ.get(
            "HF_AIRCRAFT_REPO", "iturslab/Efficient-YOLO-RS-Airplane-Detection"
        )
        weights_rel = os.environ.get(
            "HF_AIRCRAFT_WEIGHTS", "training/experiment-61/best.pt"
        )
        imgsz_env = os.environ.get("HF_AIRCRAFT_IMGSZ", "native").strip()
        imgsz_max = int(os.environ.get("HF_AIRCRAFT_IMGSZ_MAX", "16384"))
        tile_enabled = os.environ.get("HF_AIRCRAFT_TILE", "1").strip() != "0"
        tile_overlap = float(os.environ.get("HF_AIRCRAFT_TILE_OVERLAP", "0.40"))
        conf = float(os.environ.get("HF_AIRCRAFT_CONF", "0.001"))
        iou = float(os.environ.get("HF_AIRCRAFT_IOU", "0.45"))
        max_det = int(os.environ.get("HF_AIRCRAFT_MAX_DET", "500"))
        use_augment = os.environ.get("HF_AIRCRAFT_AUGMENT", "0").strip() != "0"
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

    if quick:
        print(
            "[hf:quick] Ultralytics COCO weights (default yolov8n.pt) — fast pipeline check; "
            "HF_AIRCRAFT_QUICK=0 restores Efficient-YOLO-RS.",
            file=sys.stderr,
        )

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
                        elif r.boxes is None or len(r.boxes) == 0:
                            detections = []
                        else:
                            xyxy = r.boxes.xyxy.cpu().numpy()
                            scores = r.boxes.conf.cpu().numpy()
                            for i in range(len(xyxy)):
                                x1, y1, x2, y2 = xyxy[i].tolist()
                                score = float(scores[i])
                                detections.append(
                                    {
                                        "bbox": {
                                            "x": x1 / w,
                                            "y": y1 / h,
                                            "width": (x2 - x1) / w,
                                            "height": (y2 - y1) / h,
                                        },
                                        "detectionConfidence": max(0.0, min(1.0, score)),
                                        "angle": 0,
                                    }
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
