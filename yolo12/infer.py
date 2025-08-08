#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time YOLO12 detection on Intel RealSense D415 color stream.

- Uses your project logger and error tracker.
- No argparse; tune constants in CONFIG below.
- Shows OpenCV window with drawn boxes, labels, confidences, and FPS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from utils.logger import Logger, CaptureStderrToLogger
from utils.settings import camera as cam_cfg
from utils.error_tracker import ErrorTracker, CameraConnectionError


# =========================
# ====== CONFIG AREA ======
# =========================


@dataclass(frozen=True)
class Config:
    # Path to detector weights (download them beforehand into ./models)
    MODEL_PATH: str = "models/yolo12n.pt"

    # Inference params
    IMG_SIZE: int = 640
    CONF_THRES: float = 0.25
    IOU_THRES: float = 0.45
    MAX_DET: int = 300

    # RealSense capture
    USE_ALIGN_TO_COLOR: bool = (
        False  # detection uses only color; keep False for lower latency
    )
    WARMUP_FRAMES: int = 5

    # UI
    WINDOW_NAME: str = "YOLO12 • RealSense D415"
    SHOW_FPS: bool = True


CONFIG = Config()

log = Logger.get_logger("realsense_yolo12")


# =========================
# ====== CORE LOGIC  ======
# =========================


class RealSenseColor:
    """Minimal color-stream wrapper for D415."""

    def __init__(self, width: int, height: int, fps: int, align_to_color: bool):
        self.width = width
        self.height = height
        self.fps = fps
        self.align_to_color = align_to_color

        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None

    def start(self) -> None:
        try:
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            # Depth stream is not required for detection; omit to reduce latency.
            profile = self._pipeline.start(cfg)
            if self.align_to_color:
                self._align = rs.align(rs.stream.color)

            # Warm-up for auto-exposure
            for _ in range(CONFIG.WARMUP_FRAMES):
                self._fetch()
            log.info(
                f"RealSense started: {self.width}x{self.height}@{self.fps} (align={self.align_to_color})"
            )
        except Exception as e:
            raise CameraConnectionError(f"Failed to start RealSense: {e}") from e

    def _fetch(self) -> rs.composite_frame:
        assert self._pipeline is not None
        frames = self._pipeline.wait_for_frames()
        return self._align.process(frames) if self._align else frames

    def read_color(self) -> np.ndarray:
        frames = self._fetch()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame")
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            finally:
                self._pipeline = None
                self._align = None
            log.info("RealSense stopped")


def load_model(path: str) -> YOLO:
    """Load YOLO12 detector weights."""
    log.info(f"Loading YOLO12 weights: {path}")
    model = YOLO(path)  # device auto-selection (CUDA if available)
    # Optional: you can set conf/iou/imgsz per-call, not globally
    return model


def draw_boxes(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    names: dict[int, str],
) -> None:
    """Draw boxes and labels in-place."""
    h, w = img.shape[:2]
    for (x1, y1, x2, y2), cls, conf in zip(boxes_xyxy, classes, confidences):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = names.get(int(cls), str(int(cls)))
        txt = f"{label} {conf:.2f}"

        # box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # text bg
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_txt = max(0, y1 - th - 6)
        cv2.rectangle(
            img, (x1, y1_txt), (x1 + tw + 6, y1_txt + th + 6), (0, 255, 0), -1
        )
        # text
        cv2.putText(
            img,
            txt,
            (x1 + 3, y1_txt + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )


def put_fps(img: np.ndarray, fps: float) -> None:
    """Overlay FPS counter."""
    txt = f"{fps:.1f} FPS"
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 3)
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)


def run() -> None:
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    cam = RealSenseColor(
        width=cam_cfg.rgb_width,
        height=cam_cfg.rgb_height,
        fps=cam_cfg.fps,
        align_to_color=CONFIG.USE_ALIGN_TO_COLOR,
    )

    # ensure we always stop camera + close window
    def _cleanup():
        cam.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    ErrorTracker.register_cleanup(_cleanup)

    # Some native libs can spam stderr; capture to our logger
    with CaptureStderrToLogger(Logger.get_logger("stderr-capture")):
        cam.start()
        model = load_model(CONFIG.MODEL_PATH)

    cv2.namedWindow(CONFIG.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG.WINDOW_NAME, 1280, 720)

    last_t = time.perf_counter()
    fps = 0.0

    try:
        while True:
            frame = cam.read_color()  # BGR uint8
            # Inference (sync). You can set batch=1 by default.
            results = model.predict(
                source=frame,
                imgsz=CONFIG.IMG_SIZE,
                conf=CONFIG.CONF_THRES,
                iou=CONFIG.IOU_THRES,
                max_det=CONFIG.MAX_DET,
                verbose=False,
                # device auto; stream=False returns a list of Results
            )

            r = results[0]
            # Boxes are on CPU numpy via .cpu().numpy() (Ultralytics already returns np in r.boxes.*.cpu().numpy())
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                draw_boxes(
                    frame,
                    boxes,
                    cls,
                    conf,
                    r.names if hasattr(r, "names") else model.names,
                )

            # FPS
            if CONFIG.SHOW_FPS:
                now = time.perf_counter()
                dt = max(1e-6, now - last_t)
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
                last_t = now
                put_fps(frame, fps)

            cv2.imshow(CONFIG.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # Esc or Q
                log.info("Exit requested by user")
                break

    except CameraConnectionError as e:
        ErrorTracker.report(e)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received")
    except Exception as e:
        ErrorTracker.report(e)
    finally:
        _cleanup()
        log.info("Shutdown complete")


if __name__ == "__main__":
    run()
