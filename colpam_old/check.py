import sqlite3, numpy as np

conn = sqlite3.connect("/home/sha/Documents/work/robohand_v2/colmap/data/database.db")
cur = conn.cursor()
cur.execute("SELECT camera_id, params FROM cameras")
for cam_id, blob in cur.fetchall():
    arr = np.frombuffer(blob, dtype=np.float32)
    print(f"Camera {cam_id}: {arr}; len = {len(arr)}")
