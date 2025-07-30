# HANDEYE

This document provides a structured, mathematical description of the full chain of transformations required to convert 2D pixel measurements and depth values into 3D robot base coordinates. Each mathematical step is detailed clearly and formally, with references to OpenCV and robotics literature included.

## 1. Full Coordinate Transformation Chain

The following transformation chain converts pixel and depth data from camera coordinates into robot base coordinates:
$$
\begin{bmatrix}
u \\
v \\
d
\end{bmatrix}
\xrightarrow{\mathrm{K_{depth}}}
\mathrm{p_{depth}}
\xrightarrow{\mathrm{T_{depth2rgb}}}
\mathrm{p_{rgb}}
\xrightarrow{\mathrm{T_{cam2base}}}
\mathrm{p_{base}}
$$

Each arrow represents a transformation:

- **$\mathrm{K_{depth}}$**: Intrinsic matrix of the depth camera.
- **$\mathrm{T_{depth2rgb}}$**: Extrinsic calibration between Depth and RGB sensors.
- **$\mathrm{T_{cam2base}}$**: Robot pose transform from RGB sensors of camera through TCP to base.

---

## 1.1 Intrinsic Camera Matrix (OpenCV Model)

$$
\mathrm{K} =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix} 
$$

- **$f_x, f_y$**: Focal lengths in pixels.
- **$c_x, c_y$**: Principal point coordinates.

**Reference:** [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

---

## 1.2 Backprojection from RGB image to 3D

$$
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
=
\begin{bmatrix}
(u - c_x) \cdot \dfrac{d}{f_x} \\
(v - c_y) \cdot \dfrac{d}{f_y} \\
d
\end{bmatrix}
\qquad
\begin{aligned}
d~&-\text{depth value (m)} \\
(u, v)~&-\text{RGB image coordinates (pixels)} \\
(x, y, z)~&-\text{camera coordinates}
\end{aligned}
$$

This converts depth and pixel coordinates into 3D points in the camera coordinate system.

---

## 2. Camera-to-Base Transform

$$ 
\mathrm{T_{cam2base}} = \mathrm{T_{tcp2base}} \times \mathrm{T_{cam2tcp}} 
$$

---

## 2.1 Camera-to-TCP Transformation

$$ 
\mathrm{T_{cam2tcp}} =
\begin{bmatrix}
R & t\\
0 & 1 
\end{bmatrix} 
$$

Rigid transformation from the camera coordinate system to the robot TCP (tool center point).

**R**: Rotation matrix $(3\times3)$.

**t**: Translation vector $(3\times1)$.

**Reference**: [cv2.calibrateHandEye](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)

$$
\mathrm{TARGET\_POSE} = [
  \underset{mm}{x, y, z},\ 
  \overset{degree}{r_x, r_y, r_z}
]
$$

Target Pose of in robot base coords.

**$[x, y, z]$**: Translation of the object in millimeters.

**$[r_x, r_y, r_z]$**: Euler angles (XYZ order) in degrees.

---

## 2.2 TCP-to-Base Transformation

$$
\mathrm{T_{tcp2base}} =
\begin{bmatrix}
R(r_x, r_y, r_z) & 0.001 \cdot \left[ x, y, z\right]^T \\
0 & 1 
\end{bmatrix} 
$$

**$R(r_x, r_y, r_z)$**: Rotation matrix from Euler angles (converted to radians).

**$0.001 \cdot \left[ x, y, z\right]^T$** : Converts mm to meters for robotics.

---

## 2.3 Euler Angle Rotation Matrix

$$
R(r_x, r_y, r_z) =
\begin{bmatrix}
c_{r_y} c_{r_z} & c_{r_x} s_{r_y} s_{r_z} - s_{r_x} c_{r_z} & c_{r_x} s_{r_y} c_{r_z} + s_{r_x} s_{r_z} \\
c_{r_y} s_{r_z} & s_{r_x} s_{r_y} s_{r_z} + c_{r_x} c_{r_z} & c_{r_z} s_{r_x} s_{r_y} - c_{r_x} s_{r_z} \\
-s_{r_y} & c_{r_y} s_{r_x} & c_{r_y} c_{r_x}
\end{bmatrix},
\quad
\begin{aligned}
c_{(\cdot)} = \cos{\left[ \cdot \right]}
\\
s_{(\cdot)} = \sin{\left[ \cdot \right]}
\end{aligned}
$$

This formulation follows XYZ (roll-pitch-yaw) convention.

---

## 3. Hand-Eye Calibration

OpenCV implements several closed-form methods for solving the hand–eye calibration problem, which is formulated as the matrix equation:

$$
\mathrm{A}_i \mathrm{X} = \mathrm{X} \mathrm{B}_i
$$

- $\mathrm{A}_i = \mathrm{T_{robot}}^{(i)}$: the known robot pose from base to TCP at time $i$.
- $\mathrm{B}_i = \mathrm{T_{target}}^{(i)}$: the known pose of a calibration target (e.g., Chess board, Charuco board) relative to the camera at time $i$.
- $\mathrm{X} = \mathrm{T_{cam2tcp}}$: the unknown camera-to-TCP transformation to solve for

This is a form of the **Sylvester-type equation**, arising in rigid body calibration problems.

OpenCV’s [`cv2.calibrateHandEye`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b) solves this problem using one of several algorithms:

- **Tsai & Lenz (1989)**: Decoupled rotation/translation.
- **Park & Martin (1994)**: Screw theory-based method.
- **Horaud et al. (1995)**: Quaternion optimization.
- **Daniilidis (1999)**: Dual quaternion method.
- **Andreff et al. (2001)**: Algebraic approach minimizing reprojection error.

The calibration minimizes:

$$
\sum_i \\| \mathrm{A}_i \mathrm{X} - \mathrm{X} \mathrm{B}_i \\|^2
$$

## References

- [OpenCV: cv2.calibrateHandEye](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV Pose Estimation (solvePnP)](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Wikipedia: Euler Angles](https://en.wikipedia.org/wiki/Euler_angles)
- [Wikipedia: Transformation Matrix](https://en.wikipedia.org/wiki/Transformation_matrix)
- [Wikipedia: Sylvester Equation](https://en.wikipedia.org/wiki/Sylvester_equation)
