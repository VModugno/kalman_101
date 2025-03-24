import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) Simulation parameters
# =============================================================================
DT = 0.1            # Time step [s]
T_MAX = 500          # Total simulation time [s]
N_STEPS = int(T_MAX / DT)

# Noise magnitudes
Q_v = 0.1          # Std dev on forward velocity noise
Q_w = 0.1          # Std dev on angular velocity noise
R_x = 0.1           # Std dev on x-measurement noise
R_y = 0.1           # Std dev on y-measurement noise

# =============================================================================
# 2) True system dynamics: Exact integration for a differential-drive robot
#    x = [x, y, theta]^T, u = [v, w]^T
# =============================================================================
def simulate_dd_robot_exact(x, u):
    """
    Exact integration for the differential-drive (unicycle) model.
    For a time step DT, the closed-form solution is:
    
        theta_next = theta + w*DT
        if |w| > eps:
            x_next = x + (v/w) * [sin(theta_next) - sin(theta)]
            y_next = y - (v/w) * [cos(theta_next) - cos(theta)]
        else: (w ~ 0)
            x_next = x + v*DT*cos(theta)
            y_next = y + v*DT*sin(theta)
    
    x: state = [x, y, theta]
    u: control = [v, w]
    returns next state.
    """
    x_, y_, th_ = x
    v, w = u
    eps = 1e-6  # tolerance for zero angular velocity
    th_next = th_ + w * DT
    th_next = (th_next + np.pi) % (2 * np.pi) - np.pi  # normalize angle
    
    if abs(w) > eps:
        x_next = x_ + (v / w) * (np.sin(th_next) - np.sin(th_))
        y_next = y_ - (v / w) * (np.cos(th_next) - np.cos(th_))
    else:
        # When w is nearly zero, use Euler integration
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

# =============================================================================
# 3) EKF functions (using Euler integration for prediction)
# =============================================================================
def f(x, u):
    """
    Nonlinear state transition for the robot.
    Uses Euler integration (same as the original odometry).
    """
    x_, y_, th_ = x
    v, w = u
    x_next = x_ + v * np.cos(th_) * DT
    y_next = y_ + v * np.sin(th_) * DT
    th_next = th_ + w * DT
    th_next = (th_next + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_next, y_next, th_next])

def jacobian_F(x, u):
    """
    Jacobian of f with respect to x, evaluated at (x,u).
    """
    _, _, th_ = x
    v, _ = u
    F = np.array([
        [1, 0, -v * DT * np.sin(th_)],
        [0, 1,  v * DT * np.cos(th_)],
        [0, 0, 1]
    ])
    return F
# TODO replace this for landmark-based SLAM and map-based localization
def h(x):
    """
    Measurement function.
    Assumes the sensor measures [x, y] directly.
    """
    return np.array([x[0], x[1]])

def jacobian_H(x):
    """
    Jacobian of h with respect to x.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

# =============================================================================
# 4) Main simulation: True (exact integration), EKF (Euler based), and odometry (Euler)
# =============================================================================
def main():
    np.random.seed(0)  # for reproducibility
    
    # True state: use exact integration
    x_true = np.array([0.0, 0.0, 0.0])
    
    # EKF initialization
    x_est = np.array([0.5, 0.5, 0.3])   # initial guess
    P_est = np.eye(3) * 0.5             # initial covariance

    # Pure odometry initialization: use Euler integration (simulate_dd_robot)
    x_odom = np.array([0.0, 0.0, 0.0])
    
    # Logging arrays
    traj_true = []
    traj_est = []
    traj_odom = []
    
    for k in range(N_STEPS):
        t = k * DT
        
        # ---------------------------------------------------------------------
        # 4a) Choose a control input (v, w)
        # For example: constant forward speed with sinusoidal angular velocity
        v_cmd = 0.5
        w_cmd = 1.0 * np.sin(0.1 * t)
        u_cmd = np.array([v_cmd, w_cmd])
        
        # ---------------------------------------------------------------------
        # 4b) True state evolution using exact integration with process noise
        # ---------------------------------------------------------------------
        # Add noise to simulate unmodeled dynamics on the true state.
        v_noisy = v_cmd + Q_v * np.random.randn()
        w_noisy = w_cmd + Q_w * np.random.randn()
        x_true = simulate_dd_robot_exact(x_true, [v_noisy, w_noisy])
        
        # ---------------------------------------------------------------------
        # 4c) Pure odometry update (using Euler integration)
        # ---------------------------------------------------------------------
        # Here, we update odometry using the Euler integration method.
        # It uses the same noisy control as applied to the true system.
        x_odom = euler_dd_robot(x_odom, [v_cmd, w_noisy])
        # To simulate divergence, you might want to use Euler integration here instead:
        # For example, use a simple Euler integration function (simulate_dd_robot) instead of the exact one.
        # x_odom = simulate_dd_robot(x_odom, [v_noisy, w_noisy])
        
        # ---------------------------------------------------------------------
        # 4d) EKF Predict Step (using Euler integration)
        # ---------------------------------------------------------------------
        x_pred = f(x_est, u_cmd)
        F_k = jacobian_F(x_est, u_cmd)
        Q_k = np.diag([0.01, 0.01, 0.01])
        P_pred = F_k @ P_est @ F_k.T + Q_k
        
        # ---------------------------------------------------------------------
        # 4e) Measurement & EKF Update Step
        # ---------------------------------------------------------------------
        z_true = h(x_true)
        z_meas = z_true + np.array([R_x * np.random.randn(), R_y * np.random.randn()])
        z_pred = h(x_pred)
        H_k = jacobian_H(x_pred)
        R_mat = np.diag([R_x**2, R_y**2])
        nu_k = z_meas - z_pred
        S = H_k @ P_pred @ H_k.T + R_mat
        K = P_pred @ H_k.T @ np.linalg.inv(S)
        x_est = x_pred + K @ nu_k
        P_est = (np.eye(3) - K @ H_k) @ P_pred
        
        # ---------------------------------------------------------------------
        # 4f) Log data
        # ---------------------------------------------------------------------
        traj_true.append(x_true.copy())
        traj_est.append(x_est.copy())
        traj_odom.append(x_odom.copy())
    
    traj_true = np.array(traj_true)
    traj_est  = np.array(traj_est)
    traj_odom = np.array(traj_odom)
    
    # =============================================================================
    # 5) Plot the results
    # =============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(traj_true[:,0], traj_true[:,1], 'b-', label='True trajectory (Exact Integration)')
    plt.plot(traj_est[:,0],  traj_est[:,1],  'r--', label='EKF Estimate')
    plt.plot(traj_odom[:,0], traj_odom[:,1], 'g-.', label='Odometry (Euler Integration)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Differential Drive Robot: True, EKF, & Odometry Trajectories")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()

if __name__ == '__main__':
    main()
