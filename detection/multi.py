#!/usr/bin/env python3
"""
yolo12/multi.py

Real-time text-prompt detection on Intel RealSense D415.

Backends:
  - YOLOE (Ultralytics) with text prompts
  - OWL-ViT / OWLv2 (HF Transformers) with text prompts

Keys: 'q' or ESC to quit.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
from tqdm import tqdm

from utils.logger import Logger
from utils.error_tracker import ErrorTracker

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

LOG = Logger.get_logger("multi")

# ============================== CONSTANTS ====================================

PKG_DIR = Path(__file__).resolve().parent
MODELS_DIR = PKG_DIR / "models"
OWL_DIR = MODELS_DIR / "owlvit"

ASSETS_BASE = "https://github.com/ultralytics/assets/releases/latest/download/"

# Inference params
IMG_SIZE = 1280
CONF_THRES = 0.5
IOU_THRES = 0.5
MAX_DET = 5

# Camera
CAM_W = 1920
CAM_H = 1080
CAM_FPS = 30
WARMUP_FRAMES = 15
COLOR_CONTRAST: int | None = 15
COLOR_SHARPNESS: int | None = 35

# Defaults
DEFAULT_PROMPTS: Tuple[str] = (
    # "keyboard",
    # "book",
    "cup",
    # "iphone",
    "tea bottle",
    "screwdriver with orange handle",
    "scotch tape",
    "keys",
    "orange plastic card",
    "bottle cap",
)

# Default YOLOE candidates
YOLOE_CANDIDATES: Tuple[str] = (
    # "yoloe-11s-seg.pt",
    # "yoloe-11m-seg.pt",
    # "yoloe-11l-seg.pt",
    # "yoloe-v8s-seg.pt",
    # "yoloe-v8m-seg.pt",
    "yoloe-v8l-seg.pt",
)

# Default OWL repos
OWL_REPOS: Tuple[str] = (
    "google/owlvit-base-patch32",
    "google/owlv2-base-patch16",
    "google/owlv2-large-patch14",
)

# UI
WINDOW_NAME = "Text-Prompt Detection • RealSense D415"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Net
HTTP_TIMEOUT = 30
CHUNK_SIZE = 1024 * 1024  # 1 MiB

# ============================== DATA TYPES ===================================


@dataclass
class Detection:
    boxes: np.ndarray
    cls: np.ndarray
    conf: np.ndarray
    names: Dict[int, str]


# ============================== HELPERS ======================================


def _download_with_tqdm(url: str, dst: Path) -> bool:
    """HTTP download with progress and atomic rename."""
    try:
        with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
            if r.status_code != 200:
                LOG.warning(f"HTTP {r.status_code} for {url}")
                return False
            total = int(r.headers.get("content-length", 0))
            tmp = dst.with_suffix(".part")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with (
                open(tmp, "wb") as f,
                tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=dst.name,
                    dynamic_ncols=True,
                    leave=False,
                ) as pbar,
            ):
                for chunk in r.iter_content(CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if total:
                        pbar.update(len(chunk))
            tmp.replace(dst)
        return True
    except Exception as e:
        LOG.warning(f"Download failed {url}: {e}")
        return False


def ensure_yolo_weight_local(cands: Iterable[str]) -> Path:
    """Ensure one YOLOE weight exists under MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    last = "no candidates"
    for name in cands:
        dst = MODELS_DIR / name
        if dst.exists() and dst.stat().st_size > 0:
            LOG.info(f"Using local weight: {dst}")
            return dst
        url = ASSETS_BASE + name
        LOG.info(f"Downloading '{name}' to {dst}")
        if _download_with_tqdm(url, dst) and dst.stat().st_size > 0:
            LOG.info(f"Saved weight to: {dst}")
            return dst
        last = f"fetch failed for {name}"
    raise RuntimeError(f"YOLOE weight resolution failed: {last}")


def _safe_repo_dir(repo_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", repo_id)
    return OWL_DIR / safe


def ensure_owl_repo_local(
    cands: Iterable[str],
    refresh: bool = False,
) -> Path:
    """Ensure one OWL repo is cached under OWL_DIR."""
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub missing. Install: "
            "pip install -U huggingface_hub transformers pillow torch"
        ) from e

    OWL_DIR.mkdir(parents=True, exist_ok=True)
    last = "no candidates"
    for repo in cands:
        out_dir = _safe_repo_dir(repo)
        if refresh and out_dir.exists():
            import shutil

            LOG.warning(f"Refreshing OWL cache: {out_dir}")
            shutil.rmtree(out_dir, ignore_errors=True)
        if out_dir.exists() and any(out_dir.iterdir()):
            LOG.info(f"Using cached OWL repo: {out_dir}")
            return out_dir
        LOG.info(f"Caching OWL repo: {repo}")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(out_dir),
                local_dir_use_symlinks=False,
                tqdm_class=tqdm,
            )
            if any(out_dir.iterdir()):
                LOG.info(f"Saved OWL repo to: {out_dir}")
                return out_dir
        except Exception as e:
            LOG.warning(f"Skip OWL repo {repo}: {e}")
            last = f"failed {repo}"
            continue
    raise RuntimeError(f"OWL repo resolution failed: {last}")


def select_device(requested: str) -> str:
    """Select compute device based on user request or availability."""
    req = requested.strip().lower()
    if req != "auto":
        if req in ("gpu", "cuda"):
            req = "cuda:0"
        LOG.info(f"Device forced by user: {req}")
        return req

    try:
        import torch

        if torch.cuda.is_available():
            dev = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        LOG.info(f"Device auto-selected: {dev}")
        return dev
    except Exception:
        LOG.info("Torch not available, falling back to CPU")
        return "cpu"


# ============================== CAMERA =======================================


class RealSenseColor:
    """D415 color stream with optional RGB tuning."""

    def __init__(
        self,
        w: int,
        h: int,
        fps: int,
        contrast: int | None,
        sharpness: int | None,
    ) -> None:
        self.w, self.h, self.fps = int(w), int(h), int(fps)
        self.contrast = contrast
        self.sharpness = sharpness
        self._pipe: rs.pipeline | None = None
        self._profile: rs.pipeline_profile | None = None

    def start(self) -> None:
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        self._profile = self._pipe.start(cfg)
        self._apply_rgb_tuning()
        self._warmup(WARMUP_FRAMES)
        LOG.info(f"RealSense started: {self.w}x{self.h}@{self.fps}")

    def _apply_rgb_tuning(self) -> None:
        if self._profile is None:
            return
        dev = self._profile.get_device()
        for s in dev.query_sensors():
            try:
                name = s.get_info(rs.camera_info.name).lower()
            except Exception:
                name = ""
            if "rgb" not in name and "color" not in name:
                continue
            self._set_opt(s, rs.option.contrast, self.contrast, "contrast")
            self._set_opt(s, rs.option.sharpness, self.sharpness, "sharpness")

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _set_opt(
        self, sensor: rs.sensor, opt: rs.option, val: int | None, label: str
    ) -> None:
        if val is None:
            return
        try:
            if not sensor.supports(opt):
                LOG.debug(f"Skip {label}: not supported")
                return
            rng = sensor.get_option_range(opt)
            v = float(self._clamp(val, rng.min, rng.max))
            sensor.set_option(opt, v)
            LOG.info(f"Set {label} to {v:.0f}")
        except Exception as e:
            LOG.warning(f"Failed to set {label}: {e}")

    def _warmup(self, n: int) -> None:
        for _ in range(max(0, n)):
            self._pipe.wait_for_frames()

    def read(self) -> np.ndarray:
        frames = self._pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame")
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        if not self._pipe:
            return
        try:
            self._pipe.stop()
        except Exception:
            pass
        self._pipe = None
        LOG.info("RealSense stopped")


# ============================== DETECTORS ====================================


class BaseDetector:
    display_name: str
    device: str

    def predict(self, bgr: np.ndarray) -> Detection:
        raise NotImplementedError


class YoloETextPromptDetector(BaseDetector):
    """YOLOE with text prompts."""

    def __init__(self, candidates: Iterable[str], prompts: Iterable[str], device: str):
        if YOLO is None:
            raise RuntimeError("Ultralytics not importable. Install 'ultralytics'.")
        weight = ensure_yolo_weight_local(candidates)
        self.model = YOLO(str(weight))
        self.model.to(device)
        self.device = device
        self.display_name = f"YOLOE:{Path(weight).name}"
        self._set_text_prompts(list(prompts))
        LOG.info(f"Backend: {self.display_name} on {self.device}")

    def _set_text_prompts(self, labels: List[str]) -> None:
        pe = self.model.get_text_pe(labels)
        self.model.set_classes(labels, pe)
        LOG.info(f"Prompts set: {labels}")

    def predict(self, bgr: np.ndarray) -> Detection:
        res = self.model.predict(
            source=bgr,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_DET,
            device=self.device,
            verbose=False,
        )[0]

        if not getattr(res, "boxes", None) or len(res.boxes) == 0:
            z4 = np.zeros((0, 4), np.float32)
            z1 = np.zeros((0,), np.int32)
            zf = np.zeros((0,), np.float32)
            names = dict(getattr(res, "names", self.model.names))
            return Detection(z4, z1, zf, names)

        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        cls = res.boxes.cls.cpu().numpy().astype(np.int32)
        conf = res.boxes.conf.cpu().numpy().astype(np.float32)

        m = conf >= float(CONF_THRES)
        boxes, cls, conf = boxes[m], cls[m], conf[m]
        names = dict(getattr(res, "names", self.model.names))
        return Detection(boxes, cls, conf, names)


class OwlVitDetector(BaseDetector):
    """OWL-ViT / OWLv2 open-vocabulary detector."""

    def __init__(
        self,
        repos: Iterable[str],
        labels: Iterable[str],
        device: str,
    ):
        repo_dir = ensure_owl_repo_local(repos)
        self.device = self._resolve_device(device)
        self.labels = list(labels)
        mt = None
        try:
            mt = self._load_model(repo_dir)
        except ValueError as e:
            msg = str(e).lower()
            if "state dictionary" in msg or "corrupted" in msg:
                LOG.warning("OWL cache corrupted, refreshing...")
                repo_dir = ensure_owl_repo_local(repos, refresh=True)
                mt = self._load_model(repo_dir)
            else:
                raise
        name = Path(repo_dir).name
        self.display_name = f"OWL:{name}:{mt}"
        LOG.info(f"Backend: {self.display_name} on {self.device}")

    @staticmethod
    def _resolve_device(dev: str) -> str:
        return "cuda" if dev.startswith("cuda") else dev

    def _load_model(self, repo_dir: Path) -> str:
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.owlvit.modeling_owlvit import (
            OwlViTForObjectDetection,
        )

        try:
            from transformers.models.owlv2.modeling_owlv2 import (
                Owlv2ForObjectDetection,
            )
        except Exception:
            Owlv2ForObjectDetection = None

        cfg = AutoConfig.from_pretrained(str(repo_dir), local_files_only=True)
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(repo_dir), local_files_only=True, use_fast=True
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(
                str(repo_dir), local_files_only=True
            )

        mt = (cfg.model_type or "").lower()
        if mt == "owlv2":
            if Owlv2ForObjectDetection is None:
                raise RuntimeError("Install transformers>=4.41 for owlv2 support.")
            model = Owlv2ForObjectDetection.from_pretrained(
                str(repo_dir), local_files_only=True
            )
        elif mt in ("owlvit", "owl_vit"):
            model = OwlViTForObjectDetection.from_pretrained(
                str(repo_dir), local_files_only=True
            )
        else:
            raise RuntimeError(f"Unsupported OWL model_type: {mt}")

        self.model = model.to(self.device).eval()
        return mt

    def predict(self, bgr: np.ndarray) -> Detection:
        import torch
        from PIL import Image

        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inputs = self.processor(text=[self.labels], images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            h, w = bgr.shape[:2]
            target = torch.tensor([[h, w]], device=self.device)
            res = self.processor.post_process_object_detection(
                outputs=out, target_sizes=target
            )[0]

        scores = res["scores"].detach().cpu().numpy().astype(np.float32)
        labels = res["labels"].detach().cpu().numpy().astype(np.int32)
        boxes = res["boxes"].detach().cpu().numpy().astype(np.float32)

        m = scores >= float(CONF_THRES)
        boxes, labels, scores = boxes[m], labels[m], scores[m]
        names = {i: n for i, n in enumerate(self.labels)}

        if boxes.size == 0:
            z4 = np.zeros((0, 4), np.float32)
            z1 = np.zeros((0,), np.int32)
            zf = np.zeros((0,), np.float32)
            return Detection(z4, z1, zf, names)

        return Detection(boxes, labels, scores, names)


# ============================== DRAW / UI ====================================


def draw_boxes(img: np.ndarray, det: Detection) -> None:
    for (x1, y1, x2, y2), c, p in zip(det.boxes, det.cls, det.conf):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = det.names.get(int(c), str(int(c)))
        txt = f"{label} {p:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 220, 40), 2)
        (tw, th), bl = cv2.getTextSize(txt, FONT, 0.8, 2)
        tx, ty = x1, max(0, y1 - th - 6)
        cv2.rectangle(
            img,
            (tx, ty),
            (tx + tw + 10, ty + th + bl + 6),
            (40, 220, 40),
            -1,
        )
        cv2.putText(img, txt, (tx + 5, ty + th), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def put_fps(img: np.ndarray, fps: float, tag: str) -> None:
    txt = f"{tag} {fps:.1f} FPS"
    cv2.putText(img, txt, (16, 40), FONT, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (16, 40), FONT, 1.0, (0, 200, 255), 2, cv2.LINE_AA)


# ============================== ORCHESTRATION ================================


def build_detector(
    backend: str,
    prompts: List[str],
    device: str,
    yolo_override: List[str] | None = None,
    owl_override: List[str] | None = None,
) -> BaseDetector:
    if backend == "yoloe":
        cands = tuple(yolo_override) if yolo_override else YOLOE_CANDIDATES
        return YoloETextPromptDetector(cands, prompts, device)
    if backend == "owl":
        repos = tuple(owl_override) if owl_override else OWL_REPOS
        return OwlVitDetector(repos, prompts, device)
    raise ValueError(f"Unknown backend: {backend}")


def build_model_and_cam(
    backend: str,
    prompts: List[str],
    device: str,
    yolo_override: List[str] | None = None,
    owl_override: List[str] | None = None,
) -> tuple[BaseDetector, RealSenseColor]:
    det = build_detector(backend, prompts, device, yolo_override, owl_override)
    cam = RealSenseColor(CAM_W, CAM_H, CAM_FPS, COLOR_CONTRAST, COLOR_SHARPNESS)
    cam.start()
    return det, cam


def run_live(
    backend: str,
    prompts: List[str],
    device: str,
    yolo_override: List[str] | None = None,
    owl_override: List[str] | None = None,
) -> int:
    det, cam = build_model_and_cam(
        backend, prompts, device, yolo_override, owl_override
    )
    title = f"{WINDOW_NAME} | {det.display_name} | {det.device}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)

    last = cv2.getTickCount() / cv2.getTickFrequency()
    ema = 0.0
    try:
        while True:
            bgr = cam.read()
            d = det.predict(bgr)
            if len(d.boxes) > 0:
                draw_boxes(bgr, d)

            now = cv2.getTickCount() / cv2.getTickFrequency()
            fps = 1.0 / max(1e-6, now - last)
            last = now
            ema = fps if ema == 0.0 else 0.9 * ema + 0.1 * fps
            put_fps(bgr, ema, det.display_name)

            cv2.imshow(title, bgr)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                LOG.info("Exit requested by user")
                break
        return 0
    finally:
        cam.stop()
        try:
            cv2.destroyWindow(title)
        except Exception:
            pass
        LOG.info("Shutdown complete")


# ============================== CLI / MAIN ===================================


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-prompt detection on RealSense D415")
    p.add_argument(
        "--backend",
        choices=["yoloe", "owl"],
        default="yoloe",
        help="Detector backend",
    )
    p.add_argument(
        "--device",
        default="gpu",
        help="Device: auto|cpu|cuda:0|mps",
    )
    p.add_argument(
        "--prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Text prompts",
    )
    p.add_argument(
        "--conf", type=float, default=CONF_THRES, help="Confidence threshold"
    )
    p.add_argument("--iou", type=float, default=IOU_THRES, help="IOU threshold")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Inference image size")
    p.add_argument("--cam-w", type=int, default=CAM_W, help="Camera width")
    p.add_argument("--cam-h", type=int, default=CAM_H, help="Camera height")
    p.add_argument("--cam-fps", type=int, default=CAM_FPS, help="Camera FPS")
    p.add_argument(
        "--contrast",
        type=int,
        default=COLOR_CONTRAST,
        help="RGB contrast 0..100 (None to skip)",
    )
    p.add_argument(
        "--sharpness",
        type=int,
        default=COLOR_SHARPNESS,
        help="RGB sharpness 0..100 (None to skip)",
    )
    p.add_argument("--refresh-owl", action="store_true", help="Force refresh OWL cache")
    p.add_argument(
        "--owl-repo",
        default="",
        help="HF repo id for OWL (e.g. google/owlvit-base-patch32)",
    )
    p.add_argument(
        "--yolo-weight",
        default="",
        help="YOLOE weight filename in assets (e.g. yoloe-11s.pt)",
    )
    return p.parse_args(argv) if argv is not None else p.parse_args()


def main(argv: List[str] | None = None) -> int:
    ErrorTracker.install_excepthook()
    try:
        ErrorTracker.install_signal_handlers()
    except Exception:
        pass

    args = parse_args(argv or [])
    dev = select_device(args.device)

    global CONF_THRES, IOU_THRES, IMG_SIZE
    global CAM_W, CAM_H, CAM_FPS
    global COLOR_CONTRAST, COLOR_SHARPNESS

    CONF_THRES = float(args.conf)
    IOU_THRES = float(args.iou)
    IMG_SIZE = int(args.imgsz)
    CAM_W, CAM_H, CAM_FPS = int(args.cam_w), int(args.cam_h), int(args.cam_fps)
    COLOR_CONTRAST = args.contrast
    COLOR_SHARPNESS = args.sharpness

    if args.backend == "owl" and args.refresh_owl:
        ensure_owl_repo_local(OWL_REPOS, refresh=True)

    yolo_override = [args.yolo_weight] if args.yolo_weight else None
    owl_override = [args.owl_repo] if args.owl_repo else None

    LOG.info(f"Backend={args.backend}, device={dev}")
    LOG.info(f"Thresholds: conf={CONF_THRES:.2f}, iou={IOU_THRES:.2f}")
    LOG.info(f"Image size: imgsz={IMG_SIZE}")
    LOG.info(f"Camera: {CAM_W}x{CAM_H}@{CAM_FPS}")
    LOG.info(f"RGB tuning: contrast={COLOR_CONTRAST}, sharpness={COLOR_SHARPNESS}")

    return run_live(
        args.backend,
        list(args.prompts),
        dev,
        yolo_override=yolo_override,
        owl_override=owl_override,
    )


if __name__ == "__main__":
    raise SystemExit(main())
