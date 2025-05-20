import sqlite3

db_path = "/home/sha/Documents/work/robohand_v2/colmap/data/database.db"
cameras_txt_path = "/home/sha/Documents/work/robohand_v2/colmap/data/cameras.txt"
images_txt = "/home/sha/Documents/work/robohand_v2/colmap/data/images.txt"

CAMERA_MODEL_MAP = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
}

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# === Import cameras ===

with open(cameras_txt_path, "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue

        parts = line.strip().split()
        camera_id = int(parts[0])
        model = parts[1]
        model_id = CAMERA_MODEL_MAP[model]
        width = int(parts[2])
        height = int(parts[3])
        params = " ".join(parts[4:])
        prior_focal_length = 1

        cur.execute(
            """
            INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (camera_id, model_id, width, height, params, prior_focal_length),
        )

# === Import images ===
with open(images_txt) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 10:
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts
            cur.execute("""
            INSERT INTO images (
            image_id, name, camera_id,
            prior_qw, prior_qx, prior_qy, prior_qz,
            prior_tx, prior_ty, prior_tz
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz))


conn.commit()
conn.close()