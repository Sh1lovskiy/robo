# calibration/helpers/gen_charuco_patterns.py

from __future__ import annotations

import os
from typing import List
import cv2
from PIL import Image, ImageDraw

from utils.logger import Logger, LoggerType

A4_WIDTH_MM = 297
A4_HEIGHT_MM = 210
DPI = 300
PX_PER_MM = DPI / 25.4
WIDTH_PX = int(A4_WIDTH_MM * PX_PER_MM)
HEIGHT_PX = int(A4_HEIGHT_MM * PX_PER_MM)

# Типы шаблонов для генерации
CHARUCO_CONFIGS = [
    # (squares_x, squares_y, aruco_dict_name, aruco_dict_id)
    (8, 5, "5X5_50", cv2.aruco.DICT_5X5_50),
    (8, 6, "6X6_100", cv2.aruco.DICT_6X6_100),
    (8, 6, "6X6_250", cv2.aruco.DICT_6X6_250),
    (8, 5, "5X5_250", cv2.aruco.DICT_5X5_250),
]


class CharucoPatternGenerator:
    """
    Generator for printable Charuco board patterns on A4 sheet.
    """

    def __init__(
        self,
        dpi: int = DPI,
        a4_width_mm: int = A4_WIDTH_MM,
        a4_height_mm: int = A4_HEIGHT_MM,
        out_dir: str = "charuco_patterns",
        logger: LoggerType = None,
    ):
        self.dpi = dpi
        self.width_px = int(a4_width_mm / 25.4 * dpi)
        self.height_px = int(a4_height_mm / 25.4 * dpi)
        self.out_dir = out_dir
        self.logger = logger or Logger()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def generate_all(self, configs: List[tuple]) -> None:
        """
        Generate all charuco boards from configs list.
        """
        for cfg in configs:
            self.generate_and_save(*cfg)

    def generate_and_save(
        self,
        squares_x: int,
        squares_y: int,
        dict_name: str,
        dict_id: int,
        margin_mm: float = 0,
        marker_rel: float = 0.75,
    ) -> None:
        """
        Generate one Charuco board and save it as PNG with a label.
        """
        square_length_px = min(
            (self.width_px - 2 * margin_mm * PX_PER_MM) / squares_x,
            (self.height_px - 2 * margin_mm * PX_PER_MM) / squares_y,
        )
        square_length_mm = square_length_px / self.dpi * 25.4
        marker_length_mm = square_length_mm * marker_rel

        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length_mm / 1000,
            marker_length_mm / 1000,
            dictionary,
        )
        img = board.generateImage(
            (self.width_px, self.height_px),
            marginSize=int(margin_mm * PX_PER_MM),
        )
        basename = (
            f"charuco_a4_{squares_x}x{squares_y}_{dict_name}_"
            f"s{square_length_mm:.1f}_m{marker_length_mm:.1f}.png"
        )
        out_path = os.path.join(self.out_dir, basename)
        cv2.imwrite(out_path, img)
        self._add_label_and_save(
            out_path,
            squares_x,
            squares_y,
            dict_name,
            square_length_mm,
            marker_length_mm,
        )
        self.logger.info(
            f"Charuco {squares_x}x{squares_y} ({dict_name}) " f"saved: {out_path}"
        )

    def _add_label_and_save(
        self,
        img_path: str,
        squares_x: int,
        squares_y: int,
        dict_name: str,
        square_length_mm: float,
        marker_length_mm: float,
    ) -> None:
        """
        Add a text label to the PNG file for clarity.
        """
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        text = (
            f"Charuco {squares_x}x{squares_y} | "
            f"Square: {square_length_mm:.2f}mm | "
            f"Marker: {marker_length_mm:.2f}mm | Dict: {dict_name}"
        )
        draw.text((24, self.height_px - 50), text, fill=(0, 0, 0))
        img.save(img_path.replace(".png", "_label.png"))


def main():
    logger = Logger.get_logger(__name__)
    generator = CharucoPatternGenerator(logger=logger)
    generator.generate_all(CHARUCO_CONFIGS)
    logger.info("All Charuco A4 patterns generated.")


if __name__ == "__main__":
    main()
