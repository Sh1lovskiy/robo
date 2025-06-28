# storage package

Abstract storage layer with a RocksDB backend. The `IStorage` interface provides
helpers for JSON blobs, NumPy arrays and images. `RocksDBStorage` implements the
interface using `python-rocksdb` and supports atomic batch writes.

```python
from storage import RocksDBStorage
import cv2
import numpy as np

store = RocksDBStorage("data.db")
pose = {"tcp_coords": [0, 0, 0, 0, 0, 0]}
img = cv2.imread("rgb.png")
depth = np.load("depth.npy")

with store.batch() as b:
    b.put_json("pose:001", pose)
    b.put_image("rgb:001", img)
    b.put_array("depth:001", depth)

pose = store.get_json("pose:001")
img = store.get_image("rgb:001")
depth = store.get_array("depth:001")
```
