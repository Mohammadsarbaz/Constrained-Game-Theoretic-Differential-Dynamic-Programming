import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# =============================================================================
# PARAMETERS
# =============================================================================
T_final = 4.0
N = 200
dt = T_final / N
g = 9.81
nx = 9
nu = 2
nv = 2
m = 1.0
I = 0.02
L = 0.25
c_d = 0.1
tau_m = 0.05
hover_thrust = m * g / 2

x0 = np.zeros(nx)
x0[6] = hover_thrust
x0[7] = hover_thrust
goal = np.array([5., 4., 0., 0., 0., 0., hover_thrust, hover_thrust, 0.])

max_iter_inner = 30
max_iter_outer = 10
alpha = 0.2

obs_center = np.array([3.5, 3])
obs_radius = 1.0
safe_margin = 0.4
R_safe = obs_radius + safe_margin

S_max = 0.0
lam = 0.0
rho = 8000.0

u_min, u_max = np.array([-4., -4.]), np.array([4., 4.])
v_min, v_max = np.array([-1., -1.]), np.array([1., 1.])

Q = np.diag([20., 20., 5., 5., 10., 5., 0., 0., 0.])
R = 0.1 * np.eye(nu)
Rv = 1 * np.eye(nv)
Qf = 300 * np.diag([20., 20., 10., 10., 20., 10., 0., 0., 0.])

# =============================================================================
# DYNAMICS
# =============================================================================
def f(x, u, v):
    px, py, vx, vy, th, w, f1, f2, z = x
    dx = np.zeros(nx)
    f1_cmd = max(0.1, hover_thrust + u[0] + v[0])
    f2_cmd = max(0.1, hover_thrust + u[1] + v[1])
    
    dx[6] = (f1_cmd - f1) / tau_m
    dx[7] = (f2_cmd - f2) / tau_m
    
    T = f1 + f2
    tau = L * (f2 - f1)
    
    dx[0] = vx
    dx[1] = vy
    dx[2] = -(T / m) * np.sin(th) - c_d * vx
    dx[3] = (T / m) * np.cos(th) - g - c_d * vy
    dx[4] = w
    dx[5] = tau / I
    
    dx_obs, dy_obs = px - obs_center[0], py - obs_center[1]
    dist = np.sqrt(dx_obs**2 + dy_obs**2)
    dx[8] = max(0.0, R_safe - dist)**2
    return dx

def compute_total_cost(x_traj, u_traj, v_traj, lam_val, rho_val):
    J = 0.0
    for k in range(N):
        dx_state = x_traj[k] - goal
        dx_state[8] = 0.0
        J += (0.5 * dx_state.T @ Q @ dx_state + 0.5 * u_traj[k].T @ R @ u_traj[k] - 0.5 * v_traj[k].T @ Rv @ v_traj[k]) * dt
    terminal_error = x_traj[-1] - goal
    constraint = x_traj[-1, 8] - S_max
    return J + 0.5 * terminal_error.T @ Qf @ terminal_error + lam_val * constraint + 0.5 * rho_val * constraint**2

# =============================================================================
# PROPOSED CGT-DDP PIPELINE
# =============================================================================
print("Running CGT-DDP Framework...")
start_time_ddp = time.time()

u_ddp = np.zeros((N, nu))
v_ddp = np.zeros((N, nv))
x_ddp = np.zeros((N + 1, nx))
cost_history = np.zeros((max_iter_outer, max_iter_inner))
lam_ddp = lam

for outer in range(max_iter_outer):
    for it in range(max_iter_inner):
        x_ddp[0] = x0
        for k in range(N):
            x_ddp[k+1] = x_ddp[k] + dt * f(x_ddp[k], u_ddp[k], v_ddp[k])
            
        cost_history[outer, it] = compute_total_cost(x_ddp, u_ddp, v_ddp, lam_ddp, rho)
        terminal_error = x_ddp[-1] - goal
        constraint = x_ddp[-1, 8] - S_max
        
        Vx = Qf @ terminal_error
        Vx[8] += lam_ddp + rho * constraint
        Vxx = Qf.copy()
        Vxx[8, 8] += rho
        
        k_u, k_v = np.zeros((N, nu)), np.zeros((N, nv))
        K_u, K_v = np.zeros((N, nu, nx)), np.zeros((N, nv, nx))
        
        for k in reversed(range(N)):
            dx_state = x_ddp[k] - goal
            dx_state[8] = 0.0
            lx, lu, lv = Q @ dx_state, R @ u_ddp[k], -Rv @ v_ddp[k]
            
            A, Bu, Bv = np.eye(nx), np.zeros((nx, nu)), np.zeros((nx, nv))
            eps = 1e-6
            fxuv = f(x_ddp[k], u_ddp[k], v_ddp[k])
            
            for i in range(nx):
                xp = x_ddp[k].copy(); xp[i] += eps
                A[:, i] += dt * (f(xp, u_ddp[k], v_ddp[k]) - fxuv) / eps
            for i in range(nu):
                up = u_ddp[k].copy(); up[i] += eps
                Bu[:, i] = dt * (f(x_ddp[k], up, v_ddp[k]) - fxuv) / eps
            for i in range(nv):
                vp = v_ddp[k].copy(); vp[i] += eps
                Bv[:, i] = dt * (f(x_ddp[k], u_ddp[k], vp) - fxuv) / eps
                
            Qx, Qu, Qv = lx * dt + A.T @ Vx, lu * dt + Bu.T @ Vx, lv * dt + Bv.T @ Vx
            Qxx, Quu, Qvv = Q * dt + A.T @ Vxx @ A, R * dt + Bu.T @ Vxx @ Bu, -Rv * dt + Bv.T @ Vxx @ Bv
            Qux, Qvx = Bu.T @ Vxx @ A, Bv.T @ Vxx @ A
            Quv = Bu.T @ Vxx @ Bv
            
            Quu += 1e-5 * np.eye(nu)
            Qvv -= 1e-5 * np.eye(nv)
            
            inv_Qvv = np.linalg.inv(Qvv)
            S = Quu - Quv @ inv_Qvv @ Quv.T + 1e-5 * np.eye(nu)
            inv_S = np.linalg.inv(S)
            
            k_u[k] = -inv_S @ (Qu - Quv @ inv_Qvv @ Qv)
            k_v[k] = -inv_Qvv @ (Qv + Quv.T @ k_u[k])
            K_u[k] = -inv_S @ (Qux - Quv @ inv_Qvv @ Qvx)
            K_v[k] = -inv_Qvv @ (Qvx + Quv.T @ K_u[k])
            
            Vx = Qx + Qux.T @ k_u[k] + Qvx.T @ k_v[k]
            Vxx = 0.5 * ((Qxx + Qux.T @ K_u[k] + Qvx.T @ K_v[k]) + (Qxx + Qux.T @ K_u[k] + Qvx.T @ K_v[k]).T)

        x_new, u_new, v_new = np.zeros_like(x_ddp), np.zeros_like(u_ddp), np.zeros_like(v_ddp)
        x_new[0] = x0
        for k in range(N):
            du = k_u[k] + K_u[k] @ (x_new[k] - x_ddp[k])
            dv = k_v[k] + K_v[k] @ (x_new[k] - x_ddp[k])
            u_new[k] = np.clip(u_ddp[k] + alpha * du, u_min, u_max)
            v_new[k] = np.clip(v_ddp[k] + alpha * dv, v_min, v_max)
            x_new[k+1] = x_new[k] + dt * f(x_new[k], u_new[k], v_new[k])
        u_ddp, v_ddp, x_ddp = u_new, v_new, x_new
    lam_ddp += rho * (x_ddp[-1, 8] - S_max)

ddp_elapsed = time.time() - start_time_ddp

# =============================================================================
# FIXED NLP BASELINE: SOLVING PURE TRAJECTORY OPTIMIZATION WITH AN NLP METHOD
# =============================================================================
print("Running Centralized NLP Baseline Optimization...")
start_time_nlp = time.time()

# The optimization space now focuses strictly on finding the optimal tracking controls u
initial_guess_u = np.zeros(N * nu)
bounds_u = [(u_min[0], u_max[0]), (u_min[1], u_max[1])] * N

# Disturbance behavior is treated as a baseline profile (v=0) for standard open-loop trajectory tracking comparison
v_fixed = np.zeros((N, nv))

def nlp_objective_fixed(dec_vector_u):
    u_nlp = dec_vector_u.reshape((N, nu))
    x_nlp = np.zeros((N + 1, nx))
    x_nlp[0] = x0
    for k in range(N):
        x_nlp[k+1] = x_nlp[k] + dt * f(x_nlp[k], u_nlp[k], v_fixed[k])
    return compute_total_cost(x_nlp, u_nlp, v_fixed, lam, rho)

def nlp_safety_constraint_fixed(dec_vector_u):
    u_nlp = dec_vector_u.reshape((N, nu))
    x_nlp = np.zeros((N + 1, nx))
    x_nlp[0] = x0
    for k in range(N):
        x_nlp[k+1] = x_nlp[k] + dt * f(x_nlp[k], u_nlp[k], v_fixed[k])
    return -(x_nlp[-1, 8] - S_max)

nlp_constraints = {'type': 'ineq', 'fun': nlp_safety_constraint_fixed}

nlp_res = minimize(nlp_objective_fixed, initial_guess_u, method='SLSQP', bounds=bounds_u, 
                   constraints=nlp_constraints, options={'maxiter': 100, 'disp': False})

u_final_nlp = nlp_res.x.reshape((N, nu))
x_final_nlp = np.zeros((N + 1, nx))
x_final_nlp[0] = x0
for k in range(N):
    x_final_nlp[k+1] = x_final_nlp[k] + dt * f(x_final_nlp[k], u_final_nlp[k], v_fixed[k])

nlp_elapsed = time.time() - start_time_nlp

# =============================================================================
# PLOTTING BOTH RESULT PATHS SUCCESSFULLY
# =============================================================================
plt.figure(figsize=(7,6))
plt.plot(x_ddp[:, 0], x_ddp[:, 1], linewidth=2.5, label='Proposed CGT-DDP Trajectory')
plt.plot(x_final_nlp[:, 0], x_final_nlp[:, 1], '--m', linewidth=2.5, label='Standard NLP Method (SLSQP)')
plt.scatter(x0[0], x0[1], color='g', s=100, label='Start')
plt.scatter(goal[0], goal[1], color='red', s=120, zorder=5, label='Goal')

circle = plt.Circle(obs_center, obs_radius, edgecolor='black', facecolor='dimgray', fill=True, zorder=3)
safe_circle = plt.Circle(obs_center, R_safe, edgecolor='red', facecolor='lightcoral', fill=True, linestyle='--', linewidth=2, alpha=0.25, zorder=2)
plt.gca().add_patch(circle)
plt.gca().add_patch(safe_circle)

plt.xlabel(r'$p_x$', fontsize=14)
plt.ylabel(r'$p_y$', fontsize=14)
plt.axis('equal')
plt.title("Obstacle Avoidance", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

print(f"\nCGT-DDP Time: {ddp_elapsed:.4f}s | NLP Time: {nlp_elapsed:.4f}s")