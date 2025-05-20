# CVCBOT

This project converts the original Robot.py into an English-translated, high-performance, compiled Cython module called `fairino`. It’s built for seamless integration with the official [FAIR INNOVATION fairino-python-sdk](https://github.com/FAIR-INNOVATION/fairino-python-sdk). We use the [uv](https://github.com/astral-sh/uv) package manager to handle virtual environments and dependencies.

In this project, we have implemented:

1. **Camera Calibration:** The system includes tools for calibrating the camera, calculating the *intrinsic matrix* and *distortion parameters*, and saving them to the appropriate folder.
2. **Aruco Marker Coordinate Detection:** The system detects the coordinates of an Aruco marker attached to the end-effector (6th joint) of the cobot. The marker is aligned with the coordinate axes of the 6th joint's coordinate system.
3. **Constants Management:** In the `constants.py` file, essential constants are defined, including *Denavit-Hartenberg parameters*, *joint angle limits* (soft and hard) for the cobot, and real-world dimensions (in mm) such as the *side length* of the Aruco marker and the *square size* of the chessboard used for stereo camera calibration.
4. **RAG-Based Surface Pattern Generation:** The system includes a Retrieval Augmented Generation (*RAG*) architecture for generating optimal grinding patterns for 3D surfaces based on object boundaries, utilizing a local language model from Hugging Face.

---

## Project Structure

Here’s a sleek overview of the project layout:

```
cvcbot/
├── calibration/                  # Camera and coordinate calibration tools
│   ├── camera_calibration.py    
│   ├── aruco_coords.py    
│   └── calibration_data/         # Data: intrinsics, distortion, etc.
├── robot/                        # Core robot control and kinematics
│   ├── kinematics.py    
│   ├── robot_controller.py    
│   └── restart.py    
├── vision/                       # Vision processing
│   └── stereo_depth.py    
├── misc/                         # Handy helper tools
│   ├── logger.py    
│   └── utils.py    
├── config/                       # Constants and settings
│   └── constants.py    
├── rag/                          # RAG architecture for surface pattern generation
│   └── rag_integration.py        # Main RAG module 
├── data/                         # Data for RAG architecture
│   ├── surface_patterns/         # Stored surface patterns
│   ├── embeddings/               # Vector embeddings of boundaries
│   └── pattern_index.json        # Index of patterns
├── models/                       # Pre-trained models
│   └── microsoft/codereviewer    # Model for RAG
├── main.py                       # Entry point
├── requirements.txt              # Dependencies
└── README.md                     # You're here!
```
The Cython magic happens in the `fairino` subdirectory, where `Robot.py` gets compiled into a lean, mean module.

---

## Dependencies

- **Python 3.12** – The foundation.
- **uv** – Lightning-fast env and package management.
- **Cython** – Auto-installed during setup.
- **PyTorch and Transformers** – For the *RAG* architecture.
- **NumPy and SciPy** – For numerical operations and computations.
- **OpenCV** – For computer vision and camera calibration.

All dependencies are listed in `requirements.txt` for easy installation.

---

## Build & Install

Get started in one command:

```bash
chmode +x run.sh
./run.sh
```

What happens?
* Creates a Python 3.12 virtual env with uv.
* Installs setuptools and cython.
* Moves Robot.py into the `fairino` directory.
* Generates a setup.py for Cython compilation.
* Creates the `fairino.Robot` module.
* Downloads required models for **RAG**.
---
## Usage
Activate your env:

```bash
source venv/bin/activate
```
Then, fire it up in Python:
```bash
from fairino import Robot

robot = Robot.RPC("192.168.58.2")  # Instantiate
print("WaitAI start")              # Do cool robot stuff
error = robot.WaitAI(id=0,sign=0,value=50,maxtime=5000,opt=2)
print("WaitAI return", error)                 
```

## RAG Architecture Features
The RAG (Retrieval Augmented Generation) architecture provides:

1. **Intelligent Surface Pattern Generation:** Creates optimized grinding patterns based on 3D object boundaries.
2. **Vector Similarity Search:** Finds similar patterns from previous operations to inform new pattern generation.
3. **Adaptive Pattern Strategy:** Generates patterns suitable for different surface geometries (flat, curved, complex).
4. **Complete Movement Trajectories:** Provides both position and orientation for each point in the grinding path.
5. **Local Database:** Stores patterns and embeddings locally for efficient retrieval without external servers.