import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) Simulation parameters
# =============================================================================
DT = 0.1            # Time step [s]
T_MAX = 50          # Total simulation time [s]
N_STEPS = int(T_MAX / DT)

# Control / process noise magnitudes
Q_v = 0.1          # Std dev on forward velocity noise
Q_w = 0.1          # Std dev on angular velocity noise

# Measurement noise for ranges & bearings
R_range = 0.1       # Std dev on range
R_bearing = 0.05    # Std dev on bearing (radians)

# Landmarks (x,y)
landmarks = np.array([
    [2.0,  2.0],
    [4.0,  0.0],
    [-1.0, 3.0]
])  # 3 landmarks

# =============================================================================
# 2) True system dynamics: exact integration for a differential-drive robot
#    x = [x, y, theta]^T,   u = [v, w]^T
# =============================================================================
def simulate_dd_robot_exact(x, u):
    """
    Exact (closed-form) integration for unicycle kinematics over time DT.
    x: [x, y, theta],  u: [v, w]
    returns x_next
    """
    x_, y_, th_ = x
    v, w = u
    eps = 1e-6  # threshold for near-zero angular velocity
    th_next = th_ + w * DT
    th_next = wrap_to_pi(th_next)
    
    if abs(w) > eps:
        # exact integration
        r = v / w
        x_next = x_ + r * (np.sin(th_next) - np.sin(th_))
        y_next = y_ - r * (np.cos(th_next) - np.cos(th_))
    else:
        # fallback to Euler if w ~ 0
        x_next = x_ + v * DT * np.cos(th_)
        y_next = y_ + v * DT * np.sin(th_)
        
    return np.array([x_next, y_next, th_next])

def euler_dd_robot(x, u):
    """
    integrate one step of differential-drive kinematics (exact or small Euler).
    x: state = [x, y, theta]
    u: control = [v, w], both scalars
    returns next state
    """
    x_, y_, th_ = x
    v, w = u
    
    # Simple Euler step:
    x_next = x_ + v * np.cos(th_) * DT
    y_next = y_ + v * np.sin(th_) * DT
    th_next = th_ + w * DT
    # Normalize theta to [-pi, pi], optional:
    th_next = (th_next + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([x_next, y_next, th_next])



def wrap_to_pi(angle):
    """Normalize angle into [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

# =============================================================================
# 3) EKF prediction (Euler integration) and Jacobians
# =============================================================================
def f(x, u):
    """
    Euler-based prediction step for the robot:
       x_{k+1} = f(x_k,u_k).
    x: [x, y, theta],  u: [v, w]
    """
    x_, y_, th_ = x
    v, w = u
    x_next = x_ + v * np.cos(th_) * DT
    y_next = y_ + v * np.sin(th_) * DT
    th_next = wrap_to_pi(th_ + w * DT)
    return np.array([x_next, y_next, th_next])

def jacobian_F(x, u):
    """
    Jacobian of f w.r.t. x at (x,u).
    F is 3x3 because state = (x, y, theta).
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
# 4) Landmark measurement model: range & bearing to each of 3 landmarks
# =============================================================================
def measure_landmarks(x, landmarks):
    """
    Returns the stacked measurement vector [r1, b1, r2, b2, r3, b3].
    x = [x, y, theta].
    landmarks is a shape (3,2) array of (x_l, y_l).
    """
    px, py, th = x
    z = []
    for (lx, ly) in landmarks:
        dx = lx - px
        dy = ly - py
        r  = np.sqrt(dx*dx + dy*dy)
        b  = wrap_to_pi(np.arctan2(dy, dx) - th)
        z.extend([r, b])
    return np.array(z)

def jacobian_H_landmarks(x, landmarks):
    """
    Jacobian of the measurement function w.r.t. the state x.
    Output dimension is 6 (2 per landmark) and state dimension is 3.
    So H is a 6x3 matrix.
    """
    px, py, th = x
    H_list = []
    for (lx, ly) in landmarks:
        dx = lx - px
        dy = ly - py
        r2 = dx*dx + dy*dy
        r  = np.sqrt(r2)

        # range partial derivatives
        #   range = sqrt(dx^2 + dy^2)
        #   d(range)/dx = -dx / r
        #   d(range)/dy = -dy / r
        #   d(range)/dtheta = 0
        dr_dx = -dx / r
        dr_dy = -dy / r
        dr_dtheta = 0.0

        # bearing partial derivatives
        #   bearing = atan2(dy, dx) - theta
        #   d(bearing)/dx =  (dy)/(r^2)
        #   d(bearing)/dy = -(dx)/(r^2)
        #   d(bearing)/dtheta = -1
        db_dx =  dy / r2
        db_dy = -dx / r2
        db_dtheta = -1.0

        H_list.append([dr_dx,    dr_dy,    dr_dtheta])
        H_list.append([db_dx,    db_dy,    db_dtheta])

    # Convert to numpy array of shape (6,3)
    H = np.array(H_list)
    return H

# =============================================================================
# 5) Main simulation: true state (exact), odometry (Euler), and EKF
# =============================================================================
def main():
    np.random.seed(0)  # reproducibility
    
    # True state
    x_true = np.array([0.0, 0.0, 0.0])
    
    # EKF initialization
    x_est = np.array([0.5, 0.5, 0.3])   # initial guess
    P_est = np.eye(3) * 0.5            # initial covariance

    # Odometry initialization
    x_odom = np.array([0.0, 0.0, 0.0])
    
    # Measurement noise covariance for 3 landmarks => 6D measurement
    #   [r1, b1, r2, b2, r3, b3]
    # We'll assume each range has std R_range, each bearing has std R_bearing.
    R_diag = []
    for _ in range(3):
        R_diag.append(R_range**2)
        R_diag.append(R_bearing**2)
    R_mat = np.diag(R_diag)

    # Logging arrays
    traj_true = []
    traj_est  = []
    traj_odom = []

    for k in range(N_STEPS):
        t = k * DT
        
        # ---------------------------------------------------------------------
        # 1) Choose a control input (v, w)
        # ---------------------------------------------------------------------
        # Example: constant forward speed with sinusoidal turn
        v_cmd = 0.5
        w_cmd = 1.0 * np.sin(0.1 * t)
        u_cmd = np.array([v_cmd, w_cmd])
        
        # ---------------------------------------------------------------------
        # 2) True robot evolves with noise (exact integration)
        # ---------------------------------------------------------------------
        v_noisy = v_cmd + Q_v * np.random.randn()
        w_noisy = w_cmd + Q_w * np.random.randn()
        x_true  = simulate_dd_robot_exact(x_true, [v_noisy, w_noisy])
        
        # ---------------------------------------------------------------------
        # 3) Pure odometry update (Euler or exact)
        # ---------------------------------------------------------------------
        # Here we just do the same "exact" for simplicity; 
        # to see drift, you could switch to an Euler-based method:
        x_odom = euler_dd_robot(x_odom, [v_cmd, w_cmd])
        # e.g.:
        # x_odom = f(x_odom, [v_noisy, w_noisy])
        
        # ---------------------------------------------------------------------
        # 4) EKF: Predict
        # ---------------------------------------------------------------------
        x_pred = f(x_est, u_cmd)
        F_k = jacobian_F(x_est, u_cmd)
        # Process noise (simple)
        Q_k = np.diag([0.01, 0.01, 0.01])
        P_pred = F_k @ P_est @ F_k.T + Q_k
        
        # ---------------------------------------------------------------------
        # 5) EKF: Landmark Measurement & Update
        # ---------------------------------------------------------------------
        # Simulate a measurement to each landmark from the true state
        z_true = measure_landmarks(x_true, landmarks)
        # Add measurement noise
        z_meas = z_true + np.random.multivariate_normal(mean=np.zeros(6), cov=R_mat)
        
        # Predicted measurement from x_pred
        z_pred = measure_landmarks(x_pred, landmarks)
        
        # Jacobian of measurement at x_pred
        H_k = jacobian_H_landmarks(x_pred, landmarks)
        
        # Innovation
        nu_k = z_meas - z_pred
        
        # Wrap each bearing difference into [-pi, pi]
        # bearings are at indices 1, 3, 5
        for i in [1, 3, 5]:
            nu_k[i] = wrap_to_pi(nu_k[i])
        
        # Innovation covariance
        S = H_k @ P_pred @ H_k.T + R_mat
        
        # Kalman gain
        K = P_pred @ H_k.T @ np.linalg.inv(S)
        
        # Corrected estimate
        x_est = x_pred + K @ nu_k
        
        # Normalize heading
        x_est[2] = wrap_to_pi(x_est[2])
        
        # Covariance update
        P_est = (np.eye(3) - K @ H_k) @ P_pred
        
        # ---------------------------------------------------------------------
        # 6) Log data
        # ---------------------------------------------------------------------
        traj_true.append(x_true.copy())
        traj_est.append(x_est.copy())
        traj_odom.append(x_odom.copy())
    
    # Convert logs to arrays
    traj_true = np.array(traj_true)
    traj_est  = np.array(traj_est)
    traj_odom = np.array(traj_odom)
    
    # =============================================================================
    # 7) Plot the results
    # =============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(traj_true[:,0], traj_true[:,1], 'b-', label='True trajectory (Exact)')
    plt.plot(traj_est[:,0],  traj_est[:,1],  'r--', label='EKF (Landmark Range-Bearing)')
    plt.plot(traj_odom[:,0], traj_odom[:,1], 'g-.', label='Odometry')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Unicycle EKF with 3 Landmark Range-Bearing Measurements")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()

if __name__ == '__main__':
    main()
