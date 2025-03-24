import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) Simulation Parameters
# =============================================================================
DT = 0.1             # time step
T_MAX = 30.0         # total simulation time
N_STEPS = int(T_MAX / DT)

# Noise magnitudes
Q_v = 0.02           # process noise on forward velocity
Q_w = 0.02           # process noise on angular velocity

R_SCAN = 0.05        # std dev of each laser beam measurement

N_BEAMS = 10          # number of laser beams

# Define a simple 2D map as a list of line segments: ((x1,y1),(x2,y2))
# For example, a rectangular boundary plus one internal wall.
MAP_LINES = [
    ((0.0, 0.0), (6.0, 0.0)),
    ((6.0, 0.0), (6.0, 4.0)),
    ((6.0, 4.0), (0.0, 4.0)),
    ((0.0, 4.0), (0.0, 0.0)),
    # internal line
    ((4.0, 3.0), (4.0, 2.0))
]

# =============================================================================
# 2) Basic Robot Kinematics (Exact Integration for the True Robot)
# =============================================================================
def wrap_to_pi(angle):
    """Normalize angle into [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def simulate_dd_robot_exact(x, u):
    """
    Exact unicycle integration for one time step DT.
    x = [x, y, theta],  u = [v, w].
    """
    x_, y_, th_ = x
    v, w = u
    th_next = wrap_to_pi(th_ + w * DT)
    eps = 1e-6
    if abs(w) > eps:
        r = v / w
        x_next = x_ + r * (np.sin(th_next) - np.sin(th_))
        y_next = y_ - r * (np.cos(th_next) - np.cos(th_))
    else:
        # fallback: w ~ 0
        x_next = x_ + v * DT * np.cos(th_)
        y_next = y_ + v * DT * np.sin(th_)
    return np.array([x_next, y_next, th_next])

# =============================================================================
# 3) EKF Prediction Step (Euler for Prediction)
# =============================================================================
def f_euler(x, u):
    """
    Euler-based integration for the prediction model in the EKF.
    x = [x, y, theta],  u = [v, w].
    """
    x_, y_, th_ = x
    v, w = u
    x_next = x_ + v * np.cos(th_) * DT
    y_next = y_ + v * np.sin(th_) * DT
    th_next = wrap_to_pi(th_ + w * DT)
    return np.array([x_next, y_next, th_next])

def jacobian_F_euler(x, u):
    """
    Jacobian of f_euler wrt x, evaluated at (x,u).
    """
    _, _, th_ = x
    v, _ = u
    Fx = np.array([
        [1.0, 0.0, -v * DT * np.sin(th_)],
        [0.0, 1.0,  v * DT * np.cos(th_) ],
        [0.0, 0.0,  1.0]
    ])
    return Fx

# =============================================================================
# 4) Laser Scan Measurement Model
# =============================================================================
def measure_scan_from_map(x, map_lines, n_beams=N_BEAMS):
    """
    Simulate a 'laser scan' of n_beams equally spaced directions around the robot.
    x = [xr, yr, thr]
    map_lines: list of ((x1,y1),(x2,y2)) line segments
    returns: array of shape (n_beams,)
    """
    xr, yr, thr = x
    # Let's define beams in angles relative to thr, from 0..360 deg:
    # e.g. beam_angles = [thr + i*(2*pi/n_beams) for i in range(n_beams)]
    beam_angles = thr + np.linspace(0, 2*np.pi, n_beams, endpoint=False)
    ranges = []
    for angle in beam_angles:
        rng = cast_ray(xr, yr, angle, map_lines)
        ranges.append(rng)
    return np.array(ranges)

def cast_ray(xr, yr, angle, map_lines, max_range=10.0):
    """
    Cast a single ray from (xr,yr) at 'angle', return distance to first intersection
    with any map line. If no intersection, return max_range.
    """
    # parametric form:  p(t) = (xr, yr) + t*(cos(angle), sin(angle)), t>=0
    # find intersection with each line, keep the minimum positive t
    dx = np.cos(angle)
    dy = np.sin(angle)
    t_min = max_range
    for (p1, p2) in map_lines:
        x1, y1 = p1
        x2, y2 = p2
        # Solve for intersection p(t)=p1+u*(p2-p1)
        # Ray:  (xr,yr) + t*(dx,dy)
        # Line: (x1,y1) + u*((x2-x1),(y2-y1)), 0<=u<=1
        den = (dx*(y1-y2) + dy*(x2-x1))
        if abs(den) < 1e-12:
            continue  # parallel or nearly so
        u = ( (xr - x1)*(y1-y2) + (yr - y1)*(x2-x1) ) / den
        if u<0 or u>1:
            continue  # intersection not on the segment
        # Solve for t
        # (xr + t*dx) = x1 + u*(x2-x1)
        # => t = ...
        # We'll just plug in the x or y eq.:
        ix = x1 + u*(x2 - x1)
        iy = y1 + u*(y2 - y1)
        # compare to ray param
        # xr + t*dx = ix => t = (ix - xr)/dx if dx!=0
        # or use y eq if dx=0
        if abs(dx)>1e-12:
            t = (ix - xr)/dx
        else:
            t = (iy - yr)/dy
        if t>0 and t<t_min:
            t_min = t
    return t_min

# =============================================================================
# 5) Numeric Jacobian of the measurement model
# =============================================================================
def numeric_jacobian_h(x, map_lines, n_beams=N_BEAMS, eps=1e-5):
    """
    Approximate Jacobian of the measurement h(x) wrt x, using finite differences.
    h(x) -> R^n_beams
    x in R^3
    returns H: shape (n_beams, 3)
    """
    z0 = measure_scan_from_map(x, map_lines, n_beams)
    H = np.zeros((n_beams, 3))
    for i in range(3):
        dx = np.zeros(3)
        dx[i] = eps
        z1 = measure_scan_from_map(x+dx, map_lines, n_beams)
        H[:, i] = (z1 - z0)/eps
    return H

# =============================================================================
# 6) Main Simulation
# =============================================================================
def main():
    np.random.seed(0)
    
    # True state (use exact integration)
    x_true = np.array([1.0, 1.0, 0.0])
    
    # EKF state
    x_est = np.array([1.5, 1.2, 0.5])
    P_est = np.eye(3)*0.2
    
    # Odometry state
    x_odom = np.array([1.0, 1.0, 0.0])
    
    # Logging
    traj_true = []
    traj_est  = []
    traj_odom = []
    
    # Build measurement noise covariance for n_beams
    R_mat = np.diag([R_SCAN**2]*N_BEAMS)
    
    for k in range(N_STEPS):
        t = k*DT
        
        # 1) Choose control input
        # e.g. drive forward, small sinusoidal turn
        v_cmd = 0.4
        w_cmd = 0.6 * np.sin(0.1*t)
        u_cmd = np.array([v_cmd, w_cmd])
        
        # 2) True state update with noise
        v_noisy = v_cmd + Q_v*np.random.randn()
        w_noisy = w_cmd + Q_w*np.random.randn()
        x_true = simulate_dd_robot_exact(x_true, [v_noisy, w_noisy])
        
        # 3) Odometry update (Euler)
        #    (Here we also apply the same noisy controls)
        x_odom = f_euler(x_odom, [v_noisy, w_noisy])
        
        # 4) EKF Predict
        x_pred = f_euler(x_est, u_cmd)   # using commanded input (no noise)
        F_k = jacobian_F_euler(x_est, u_cmd)
        # process noise (simple)
        Q_k = np.diag([0.01, 0.01, 0.01])
        P_pred = F_k @ P_est @ F_k.T + Q_k
        
        # 5) Simulate "laser" measurement from the true state
        z_true = measure_scan_from_map(x_true, MAP_LINES, N_BEAMS)
        # add measurement noise
        z_meas = z_true + R_SCAN*np.random.randn(N_BEAMS)
        
        # 6) EKF Correction
        #   a) predicted measurement from x_pred
        z_pred = measure_scan_from_map(x_pred, MAP_LINES, N_BEAMS)
        
        #   b) numeric Jacobian
        H_k = numeric_jacobian_h(x_pred, MAP_LINES, N_BEAMS)
        
        #   c) innovation
        nu_k = z_meas - z_pred
        
        #   d) innovation covariance
        S = H_k @ P_pred @ H_k.T + R_mat
        
        #   e) Kalman gain
        K = P_pred @ H_k.T @ np.linalg.inv(S)
        
        #   f) state update
        x_est = x_pred + K @ nu_k
        x_est[2] = wrap_to_pi(x_est[2])
        
        #   g) covariance update
        P_est = (np.eye(3) - K @ H_k) @ P_pred
        
        # 7) Log data
        traj_true.append(x_true.copy())
        traj_est.append(x_est.copy())
        traj_odom.append(x_odom.copy())
    
    traj_true = np.array(traj_true)
    traj_est  = np.array(traj_est)
    traj_odom = np.array(traj_odom)
    
    # =============================================================================
    # 7) Plot Results
    # =============================================================================
    plt.figure(figsize=(10,6))
    
    # Plot the map lines
    for (p1, p2) in MAP_LINES:
        x1,y1 = p1
        x2,y2 = p2
        plt.plot([x1,x2],[y1,y2],'k-',linewidth=2)
    
    plt.plot(traj_true[:,0], traj_true[:,1], 'b-', label='True')
    plt.plot(traj_odom[:,0], traj_odom[:,1], 'g--', label='Odometry')
    plt.plot(traj_est[:,0],  traj_est[:,1],  'r-', label='EKF')
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('EKF Localization on a Simple Map')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
