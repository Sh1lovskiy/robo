import sqlite3
import numpy as np


db_path = "/home/sha/Documents/work/robohand_v2/colmap/database.db"

# fx, fy, cx, cy
params1 = np.array(
    [1239.0578369789218, 1240.3347715356012, 393.0063269021732, 585.7042502624901],
    dtype=np.float32,
)
params2 = np.array(
    [1245.5327967219548, 1246.5776216623958, 479.3863533681695, 561.0102518858826],
    dtype=np.float32,
)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("DELETE FROM cameras;")
cur.execute("DELETE FROM sqlite_sequence WHERE name='cameras';")

cur.execute(
    """
    INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)
    VALUES (?, ?, ?, ?, ?, ?)
""",
    (1, 1, 600, 800, params1.tobytes(), 1),
)

cur.execute(
    """
    INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)
    VALUES (?, ?, ?, ?, ?, ?)
""",
    (2, 1, 600, 800, params2.tobytes(), 1),
)

cur.execute("UPDATE images SET camera_id = 1 WHERE image_id <= 93;")
cur.execute("UPDATE images SET camera_id = 2 WHERE image_id > 93;")

cur.execute("SELECT camera_id, params FROM cameras")
for cam_id, blob in cur.fetchall():
    arr = np.frombuffer(blob, dtype=np.float32)
    print(f"Camera {cam_id}: {arr}; len = {len(arr)}")
    if len(arr) != 4:
        raise ValueError(f"Invalid parameter length in camera {cam_id}")

conn.commit()
conn.close()
print("Cameras and image links updated.")
