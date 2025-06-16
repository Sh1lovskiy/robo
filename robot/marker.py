from typing import List, Tuple
from robot.controller import RobotController


def run_marker_on_box_top(
    robot: RobotController,
    top_points: List[Tuple[float, float, float]],
    marker_length: float,  # mm
    above_offset: float = 10.0,  # mm
):
    """
    Move robot to draw a straight marker line along the center of the box top face.

    Args:
        robot: Instance of RobotController.
        top_points: List of 4 (x, y, z) tuples for the box top face (mm).
        marker_length: Length of marker line (mm).
        above_offset: Z-offset above the surface (mm).
    """
    import numpy as np

    pts = np.array(top_points)
    # 1. Найти среднее z — это "высота" поверхности (гарантируем фиксированную высоту линии)
    z_plane = float(np.mean(pts[:, 2]))

    # 2. Находим две точки с макс/min Y (длинная сторона)
    ys = pts[:, 1]
    top_idxs = np.where(np.isclose(ys, np.max(ys)))[0]
    bot_idxs = np.where(np.isclose(ys, np.min(ys)))[0]
    top_pair = pts[top_idxs][np.argsort(pts[top_idxs][:, 0])]
    bot_pair = pts[bot_idxs][np.argsort(pts[bot_idxs][:, 0])]

    # 3. Центры левой и правой стороны (по длинной стороне)
    left = (top_pair[0] + bot_pair[0]) / 2
    right = (top_pair[1] + bot_pair[1]) / 2

    # 4. Центр линии
    center_xy = (left[:2] + right[:2]) / 2

    # 5. Направление вдоль длинной стороны (в XY!)
    dir_vec_xy = right[:2] - left[:2]
    dir_vec_xy = dir_vec_xy / np.linalg.norm(dir_vec_xy)
    half_len = marker_length / 2

    # 6. Вычисляем start и end, ЗАДАЁМ Z = z_plane
    start_xy = center_xy - dir_vec_xy * half_len
    end_xy = center_xy + dir_vec_xy * half_len
    start = [float(start_xy[0]), float(start_xy[1]), z_plane]
    end = [float(end_xy[0]), float(end_xy[1]), z_plane]

    # --- Plan poses: above start, draw, above end ---
    pose_up_start = [start[0], start[1], start[2] + above_offset, 180.0, 0.0, 0.0]
    pose_start = [start[0], start[1], start[2], 180.0, 0.0, 0.0]
    pose_end = [end[0], end[1], end[2], 180.0, 0.0, 0.0]
    pose_up_end = [end[0], end[1], end[2] + above_offset, 180.0, 0.0, 0.0]

    # --- Execute movement sequence ---
    robot.move_linear(pose_up_start)
    robot.move_linear(pose_start)
    robot.move_linear(pose_end)
    robot.move_linear(pose_up_end)


# Example usage:
if __name__ == "__main__":
    robot = RobotController()
    top_points = [
        (-280.0, 110.0, 70.0),
        (-350.0, 110.0, 70.0),
        (-280.0, -140.0, 70.0),
        (-350.0, -140.0, 70.0),
    ]
    marker_len = 230.0  # mm
    run_marker_on_box_top(robot, top_points, marker_len)
