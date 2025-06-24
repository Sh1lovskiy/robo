import cv2
import PIL.Image, PIL.ImageDraw


a4_width_mm = 297
a4_height_mm = 210
dpi = 300
width_px = int(a4_width_mm / 25.4 * dpi)
height_px = int(a4_height_mm / 25.4 * dpi)

squares_x = 5
squares_y = 7
aruco_dict = cv2.aruco.DICT_4X4_100
dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)

square_length_px = min(width_px / squares_x, height_px / squares_y)
square_length_mm = square_length_px / dpi * 25.4

marker_length_mm = square_length_mm * 0.75

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length_mm / 1000,
    marker_length_mm / 1000,
    dictionary,
)

board_img = board.generateImage((width_px, height_px), marginSize=0)

cv2.imwrite("charuco_a4_5x7.png", board_img)

img_pil = PIL.Image.open("charuco_a4_5x7.png").convert("RGB")
draw = PIL.ImageDraw.Draw(img_pil)
draw.text(
    (20, height_px - 50),
    f"Charuco {squares_x}x{squares_y} | Square: {square_length_mm:.2f} mm | Marker: {marker_length_mm:.2f} mm | Dict: 4X4_100",
    (0, 0, 0),
)
img_pil.save("charuco_a4_5x7_with_label.png")
