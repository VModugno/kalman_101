# KALMAN_101

This folder contains three Python scripts that demonstrate different ways of performing Kalman/Extended Kalman Filter (EKF)–based localization for a mobile robot:

1. **`kalman_landmark.py`**  
   Demonstrates EKF localization with **known landmarks**.  
   - The robot’s state is \([x, y, \theta]\).  
   - Each measurement step provides (range, bearing) to a set of discrete landmarks whose positions are known in advance.  
   - The EKF updates the robot’s pose based on how the predicted landmark observations differ from the actual (noisy) measurements.

2. **`kalman_map.py`**  
   Illustrates EKF localization using a **known map** represented as line segments.  
   - The robot simulates a laser-like range sensor by casting rays against the map.  
   - The measurement function is more complex (ray-casting), so the Jacobian is computed numerically.  
   - This script shows how to incorporate a static map into the EKF update without needing discrete “landmarks.”

3. **`kalman_odom.py`**  
   Provides a simpler example focusing on **odometry-based** estimation and the Kalman filter.  
   - The robot integrates its motion using noisy controls (forward velocity and angular velocity).  
   - Measurements may be minimal or purely from odometry, demonstrating how the filter handles accumulated error without additional external corrections.  
   - This script can serve as a baseline comparison to see how uncorrected odometry drifts compared to EKF approaches that fuse more informative sensor data.

Each script logs the **true** robot trajectory, the **odometry** trajectory, and the **EKF** (or Kalman) estimate, and then plots them for comparison.




