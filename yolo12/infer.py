#!/usr/bin/env python3
"""
Real-time open-vocabulary / YOLO detection on Intel RealSense D415.

Modes:
  1) "owlvit" — open-vocabulary via OWL-ViT.
  2) "yolo"   — fixed classes via Ultralytics YOLO.

OVD defaults (CPU-friendly):
  - Primary:  google/owlvit-base-patch32
  - Fallback: google/owlv2-base-patch32
  - Optional resize and CPU dynamic quantization.

Local cache & offline:
  - Uses ./.cache/hf
  - Set CONFIG.HF_LOCAL_ONLY=True for offline use.

Install (for OVD):
  pip install torch torchvision pillow transformers huggingface_hub tqdm \
      ultralytics --extra-index-url https://download.pytorch.org/whl/cu121
"""

from __future__ import annotations

import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm.auto import tqdm

from utils.logger import Logger, CaptureStderrToLogger
from utils.settings import camera as cam_cfg
from utils.error_tracker import ErrorTracker, CameraConnectionError

# YOLO is optional (only if DETECTOR="yolo")
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # noqa: N816

# =========================
# ====== CONFIG AREA ======
# =========================


@dataclass(frozen=True)
class Config:
    # --- Detector selection ---
    DETECTOR: str = "yolo"  # "owlvit" or "yolo"

    # --- Open-vocabulary labels ---
    OPEN_VOCAB_LABELS: Tuple[str, ...] = (
        "pliers",
        "duct tape",
        "cardboard",
        "metal part",
        "bolt",
        "nut",
        "screwdriver",
        "wrench",
        "hammer",
        "scissors",
        "tape dispenser",
        "gloves",
        "wire cutter",
        "cutters",
        "zip tie",
        "bearing",
    )

    # --- OWL-ViT repos ---
    OWLVIT_PRIMARY: str = "google/owlvit-base-patch32"
    OWLVIT_FALLBACK: str = "google/owlv2-base-patch32"

    # OVD speed knobs
    OVD_MAX_LONG_EDGE: int = 1280
    OVD_CPU_QUANTIZE: bool = False

    # --- HuggingFace caching ---
    HF_CACHE_DIR: str = ".cache/hf"
    HF_LOCAL_ONLY: bool = False
    HF_SILENT: bool = True

    # --- YOLO weights (if DETECTOR="yolo") ---
    MODEL_PATH: str = "yolo12/models/yolo12l.pt"

    # --- Inference params ---
    IMG_SIZE: int = 1280  # only for YOLO
    CONF_THRES: float = 0.25
    IOU_THRES: float = 0.45
    MAX_DET: int = 300

    # --- RealSense capture ---
    USE_ALIGN_TO_COLOR: bool = False
    WARMUP_FRAMES: int = 5

    # --- UI ---
    WINDOW_NAME: str = "Real-time Detection • RealSense D415"
    SHOW_FPS: bool = True


CONFIG = Config()
log = Logger.get_logger("infer")

# =========================
# ====== CAMERA WRAP  =====
# =========================


class RealSenseColor:
    """Minimal color-stream wrapper for D415."""

    def __init__(self, width: int, height: int, fps: int, align_to_color: bool) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.align_to_color = bool(align_to_color)
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None

    def start(self) -> None:
        try:
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            if self.align_to_color:
                cfg.enable_stream(
                    rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
                )

            self._pipeline.start(cfg)
            if self.align_to_color:
                self._align = rs.align(rs.stream.color)

            self._warmup(CONFIG.WARMUP_FRAMES)
            log.info(
                "RealSense started: %dx%d@%d (align=%s)",
                self.width,
                self.height,
                self.fps,
                self.align_to_color,
            )
        except Exception as e:
            raise CameraConnectionError(f"Failed to start RealSense: {e}") from e

    def _warmup(self, n: int) -> None:
        if n <= 0:
            return
        for _ in tqdm(
            range(n),
            desc="Camera warmup (auto-exposure)",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            self._fetch()

    def _fetch(self) -> rs.composite_frame:
        assert self._pipeline is not None
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        return frames

    def read_color(self) -> np.ndarray:
        frames = self._fetch()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame")
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        if self._pipeline is None:
            return
        try:
            self._pipeline.stop()
        except Exception:
            pass
        finally:
            self._pipeline = None
            self._align = None
        log.info("RealSense stopped")


# =========================
# ====== DETECTORS   ======
# =========================


def _yolo_display_from_path(weights_path: str) -> str:
    base = os.path.basename(weights_path).lower()
    m_ver = re.search(r"yolo(\d+)", base)
    family = f"YOLO{m_ver.group(1)}" if m_ver else "YOLO"
    m_var = re.search(r"yolo\d*([nslmx])(?:\.pt)?$", base)
    variant = m_var.group(1).upper() if m_var else ""
    return f"{family}-{variant}" if variant else family


def _ovd_display_from_repo(repo_id: str) -> str:
    return f"OVD: {repo_id}"


class BaseDetector:
    """Unified interface for detectors."""

    def predict(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        raise NotImplementedError


class YoloDetector(BaseDetector):
    """Ultralytics YOLO wrapper."""

    def __init__(self, weights_path: str) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available. Install 'ultralytics'.")
        log.info("Loading YOLO weights: %s", weights_path)
        self.model = YOLO(weights_path)
        self.display_name = _yolo_display_from_path(weights_path)

    def predict(self, bgr: np.ndarray):
        res = self.model.predict(
            source=bgr,
            imgsz=CONFIG.IMG_SIZE,
            conf=CONFIG.CONF_THRES,
            iou=CONFIG.IOU_THRES,
            max_det=CONFIG.MAX_DET,
            verbose=False,
        )[0]

        if not getattr(res, "boxes", None) or len(res.boxes) == 0:
            names = dict(getattr(res, "names", self.model.names))
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.float32),
                names,
            )

        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        cls = res.boxes.cls.cpu().numpy().astype(np.int32)
        conf = res.boxes.conf.cpu().numpy().astype(np.float32)
        names = dict(getattr(res, "names", self.model.names))
        return boxes, cls, conf, names


class OwlVitDetector(BaseDetector):
    """OWL-ViT open-vocabulary detector."""

    def __init__(self, labels: List[str], primary: str, fallback: str) -> None:
        self.labels = list(labels)
        self._setup_hf_env()
        self._suppress_hf_logs()

        self.device = self._pick_device()
        model_dir, final_repo = self._resolve_repo(primary, fallback)
        self.model_name = final_repo
        self.display_name = _ovd_display_from_repo(final_repo)

        self.processor, self.model = self._load_local(model_dir)
        self._maybe_quantize()

    # --- init helpers ---

    @staticmethod
    def _setup_hf_env() -> None:
        os.makedirs(CONFIG.HF_CACHE_DIR, exist_ok=True)
        os.environ.setdefault("HF_HOME", CONFIG.HF_CACHE_DIR)
        os.environ.setdefault("HF_HUB_CACHE", CONFIG.HF_CACHE_DIR)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    @staticmethod
    def _suppress_hf_logs() -> None:
        if not CONFIG.HF_SILENT:
            return
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        try:
            from transformers.utils.logging import set_verbosity_error

            set_verbosity_error()
        except Exception:
            pass

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _resolve_repo(self, primary: str, fallback: str) -> Tuple[str, str]:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError, HFValidationError

        class StdoutTqdm(tqdm):
            def __init__(self, *a, **k):
                k.setdefault("file", sys.stdout)
                k.setdefault("dynamic_ncols", True)
                k.setdefault("leave", False)
                super().__init__(*a, **k)

        repo = primary
        tried_fallback = False
        while True:
            try:
                log.info(
                    "Loading OWL-ViT: %s on %s (offline=%s, cache='%s')",
                    repo,
                    self.device,
                    CONFIG.HF_LOCAL_ONLY,
                    CONFIG.HF_CACHE_DIR,
                )
                path = snapshot_download(
                    repo_id=repo,
                    cache_dir=CONFIG.HF_CACHE_DIR,
                    local_files_only=CONFIG.HF_LOCAL_ONLY,
                    tqdm_class=StdoutTqdm,
                )
                return path, repo
            except LocalEntryNotFoundError:
                if CONFIG.HF_LOCAL_ONLY and not tried_fallback:
                    repo = fallback
                    tried_fallback = True
                    continue
                raise
            except HFValidationError:
                if not tried_fallback:
                    repo = fallback
                    tried_fallback = True
                    continue
                raise

    def _load_local(self, model_dir: str):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection

        processor = OwlViTProcessor.from_pretrained(model_dir, local_files_only=True)
        model = OwlViTForObjectDetection.from_pretrained(
            model_dir, local_files_only=True
        )
        return processor, model.to(self.device).eval()

    def _maybe_quantize(self) -> None:
        if self.device != "cpu" or not CONFIG.OVD_CPU_QUANTIZE:
            return
        try:
            import torch

            try:
                from torch.ao.quantization import quantize_dynamic
            except Exception:
                from torch.quantization import quantize_dynamic
            self.model.eval()
            self.model = quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            log.info("Applied dynamic quantization (CPU, Linear->int8).")
        except Exception as e:
            log.warning("Quantization failed (continuing): %s", e)

    # --- inference ---

    @staticmethod
    def _to_pil_rgb(bgr: np.ndarray):
        from PIL import Image

        return Image.fromarray(bgr[:, :, ::-1], mode="RGB")

    @staticmethod
    def _scaled_image(bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = bgr.shape[:2]
        if CONFIG.OVD_MAX_LONG_EDGE and max(h, w) > CONFIG.OVD_MAX_LONG_EDGE:
            scale = CONFIG.OVD_MAX_LONG_EDGE / float(max(h, w))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            return cv2.resize(bgr, (nw, nh), cv2.INTER_AREA), scale
        return bgr, 1.0

    def predict(self, bgr: np.ndarray):
        import torch

        img, scale = self._scaled_image(bgr)

        with torch.no_grad():
            pil = self._to_pil_rgb(img)
            inputs = self.processor(
                text=[self.labels], images=pil, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([img.shape[:2]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes
            )[0]

        return self._collect_results(results, scale)

    def _collect_results(self, results, scale: float):
        thr = float(CONFIG.CONF_THRES)
        scores = results["scores"]
        boxes = results["boxes"]
        labels = results["labels"]

        bxs: List[np.ndarray] = []
        cls: List[int] = []
        conf: List[float] = []

        for i in range(len(scores)):
            s = float(scores[i].item())
            if s < thr:
                continue
            b = boxes[i].detach().cpu().numpy().astype(np.float32)
            if scale != 1.0:
                b = b / float(scale)
            bxs.append(b)
            cls.append(int(labels[i].item()))
            conf.append(s)

        if not bxs:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.float32),
                {i: n for i, n in enumerate(self.labels)},
            )

        return (
            np.stack(bxs, axis=0),
            np.asarray(cls, np.int32),
            np.asarray(conf, np.float32),
            {i: n for i, n in enumerate(self.labels)},
        )


def build_detector() -> BaseDetector:
    det = CONFIG.DETECTOR.lower().strip()
    if det == "owlvit":
        return OwlVitDetector(
            labels=list(CONFIG.OPEN_VOCAB_LABELS),
            primary=CONFIG.OWLVIT_PRIMARY,
            fallback=CONFIG.OWLVIT_FALLBACK,
        )
    if det == "yolo":
        return YoloDetector(weights_path=CONFIG.MODEL_PATH)
    raise ValueError(f"Unknown DETECTOR={CONFIG.DETECTOR!r}. Use 'owlvit' or 'yolo'.")


# =========================
# ====== DRAWING/UI  ======
# =========================


def draw_boxes(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    names: Dict[int, str],
) -> None:
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.8, 3
    pad_x, pad_y = 8, 6

    for (x1, y1, x2, y2), cls, conf in zip(boxes_xyxy, classes, confidences):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = names.get(int(cls), str(int(cls)))
        txt = f"{label} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

        (tw, th), base = cv2.getTextSize(txt, font, scale, thick)
        box_w = tw + 2 * pad_x
        box_h = th + 2 * pad_y + base
        tx, ty = x1, y1 - box_h
        if ty < 0:
            ty = y1 + 2
        if tx + box_w > w:
            tx = max(0, w - box_w)
        if ty + box_h > h:
            ty = max(0, h - box_h)

        cv2.rectangle(
            img, (tx, ty), (tx + box_w, ty + box_h), (0, 255, 0), -1, cv2.LINE_AA
        )
        cv2.putText(
            img,
            txt,
            (tx + pad_x, ty + pad_y + th),
            font,
            scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )


def put_fps(img: np.ndarray, fps_val: float) -> None:
    txt = f"{fps_val:.1f} FPS"
    cv2.putText(
        img, txt, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 9, cv2.LINE_AA
    )
    cv2.putText(
        img, txt, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7, cv2.LINE_AA
    )


def _build_window(detector: BaseDetector) -> str:
    title = (
        f"{CONFIG.WINDOW_NAME} • "
        f"{getattr(detector, 'display_name', CONFIG.DETECTOR.upper())}"
    )
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)
    return title


def _new_status_bar():
    return tqdm(
        total=0,
        position=0,
        leave=False,
        dynamic_ncols=True,
        bar_format="[{elapsed} | {desc}] {postfix}",
        file=sys.stdout,
    )


def _ema(prev: float, x: float, alpha: float = 0.1) -> float:
    return x if prev == 0.0 else (1.0 - alpha) * prev + alpha * x


# =========================
# ====== MAIN LOOP   ======
# =========================


def run() -> None:
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    cam = RealSenseColor(
        width=cam_cfg.rgb_width,
        height=cam_cfg.rgb_height,
        fps=cam_cfg.fps,
        align_to_color=CONFIG.USE_ALIGN_TO_COLOR,
    )

    def _cleanup() -> None:
        cam.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    ErrorTracker.register_cleanup(_cleanup)

    with CaptureStderrToLogger(Logger.get_logger("stderr-capture")):
        cam.start()
        detector = build_detector()

    window_title = _build_window(detector)
    status = _new_status_bar()

    last_t = time.perf_counter()
    ema_fps = 0.0
    mode = CONFIG.DETECTOR
    device_info = getattr(detector, "device", "auto")

    try:
        while True:
            frame = cam.read_color()

            boxes, cls, conf, names = detector.predict(frame)
            if boxes.shape[0] > 0:
                draw_boxes(frame, boxes, cls, conf, names)

            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, now - last_t)
            last_t = now

            if CONFIG.SHOW_FPS:
                ema_fps = _ema(ema_fps, inst_fps)
                put_fps(frame, ema_fps)

            status.set_description_str(f"{mode.upper()} • device={device_info}")
            status.set_postfix_str(f"fps={ema_fps:.1f} dets={boxes.shape[0]}")

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                log.info("Exit requested by user")
                break

    except CameraConnectionError as e:
        ErrorTracker.report(e)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received")
    except Exception as e:
        ErrorTracker.report(e)
    finally:
        try:
            status.close()
        except Exception:
            pass
        _cleanup()
        log.info("Shutdown complete")


if __name__ == "__main__":
    run()
