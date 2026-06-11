import numpy as np
import matplotlib.pyplot as plt

# ======================================
# PARAMETERS
# ======================================

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

hover_thrust = m*g/2

x0 = np.zeros(nx)
x0[6] = hover_thrust
x0[7] = hover_thrust

goal = np.array([5.,4.,0.,0.,0.,0.,hover_thrust,hover_thrust,0.])

max_iter_inner = 30
max_iter_outer = 10
alpha = 0.2

# ======================================
# OBSTACLE
# ======================================

obs_center = np.array([3.5, 3])
obs_radius = 1.0
safe_margin = 0.4
R_safe = obs_radius + safe_margin

# ======================================
# FUNCTIONAL CONSTRAINT
# ======================================

S_max = 0.0
lam = 0.0
rho = 8000.0

# ======================================
# CONTROL LIMITS (delta thrust)
# ======================================

u_min = np.array([-4., -4.])
u_max = np.array([ 4.,  4.])

v_min = np.array([-1., -1.])
v_max = np.array([ 1.,  1.])

# ======================================
# COST MATRICES
# ======================================

Q  = np.diag([20.,20.,5.,5.,10.,5.,0.,0.,0.])
R  = 0.1*np.eye(nu)
Rv = 1*np.eye(nv)
Qf = 300*np.diag([20.,20.,10.,10.,20.,10.,0.,0.,0.])

# ======================================
# DYNAMICS
# ======================================

def f(x,u,v):

    px, py, vx, vy, th, w, f1, f2, z = x
    dx = np.zeros(nx)

    # Commands are hover + delta
    f1_cmd = hover_thrust + u[0] + v[0]
    f2_cmd = hover_thrust + u[1] + v[1]

    # Enforce physical positivity
    f1_cmd = max(0.1, f1_cmd)
    f2_cmd = max(0.1, f2_cmd)

    dx[6] = (f1_cmd - f1) / tau_m
    dx[7] = (f2_cmd - f2) / tau_m

    T = f1 + f2
    tau = L * (f2 - f1)

    dx[0] = vx
    dx[1] = vy
    dx[2] = -(T/m) * np.sin(th) - c_d * vx
    dx[3] =  (T/m) * np.cos(th) - g - c_d * vy

    dx[4] = w
    dx[5] = tau / I

    dx_obs = px - obs_center[0]
    dy_obs = py - obs_center[1]
    dist = np.sqrt(dx_obs**2 + dy_obs**2)
    violation = max(0.0, R_safe - dist)
    dx[8] = violation**2

    return dx

# ======================================
# TOTAL COST FOR PLOTTING
# ======================================

def compute_total_cost(x_traj, u_traj, v_traj, lam_val, rho_val):
    J = 0.0

    for k in range(N):
        dx_state = x_traj[k] - goal
        dx_state[8] = 0.0

        running_cost = (
            0.5 * dx_state.T @ Q @ dx_state
            + 0.5 * u_traj[k].T @ R @ u_traj[k]
            - 0.5 * v_traj[k].T @ Rv @ v_traj[k]
        )
        J += running_cost * dt

    terminal_error = x_traj[-1] - goal
    terminal_cost = 0.5 * terminal_error.T @ Qf @ terminal_error

    constraint = x_traj[-1, 8] - S_max
    aug_term = lam_val * constraint + 0.5 * rho_val * constraint**2

    J += terminal_cost + aug_term
    return J

# ======================================
# INITIAL TRAJECTORY
# ======================================

u = np.zeros((N,nu))
v = np.zeros((N,nv))
x = np.zeros((N+1,nx))

# cost history: rows = outer iterations, cols = inner iterations
cost_history = np.zeros((max_iter_outer, max_iter_inner))

# ======================================
# OUTER LOOP
# ======================================

for outer in range(max_iter_outer):

    print("\n===== OUTER ITER", outer, "=====")

    for it in range(max_iter_inner):

        x[0] = x0
        for k in range(N):
            x[k+1] = x[k] + dt*f(x[k],u[k],v[k])

        cost_history[outer, it] = compute_total_cost(x, u, v, lam, rho)

        terminal_error = x[-1] - goal
        constraint = x[-1,8] - S_max

        Vx = Qf @ terminal_error
        Vx[8] += lam + rho*constraint

        Vxx = Qf.copy()
        Vxx[8,8] += rho

        k_u = np.zeros((N,nu))
        k_v = np.zeros((N,nv))
        K_u = np.zeros((N,nu,nx))
        K_v = np.zeros((N,nv,nx))

        for k in reversed(range(N)):

            dx_state = x[k] - goal
            dx_state[8] = 0.0

            lx = Q @ dx_state
            lu = R @ u[k]
            lv = -Rv @ v[k]

            A = np.eye(nx)
            Bu = np.zeros((nx,nu))
            Bv = np.zeros((nx,nv))

            eps = 1e-6

            fxuv = f(x[k],u[k],v[k])

            for i in range(nx):
                xp = x[k].copy()
                xp[i] += eps
                A[:,i] += dt*(f(xp,u[k],v[k]) - fxuv)/eps

            for i in range(nu):
                up = u[k].copy()
                up[i] += eps
                Bu[:,i] = dt*(f(x[k],up,v[k]) - fxuv)/eps

            for i in range(nv):
                vp = v[k].copy()
                vp[i] += eps
                Bv[:,i] = dt*(f(x[k],u[k],vp) - fxuv)/eps

            Qx  = lx*dt + A.T @ Vx
            Qu  = lu*dt + Bu.T @ Vx
            Qv  = lv*dt + Bv.T @ Vx

            Qxx = Q*dt + A.T @ Vxx @ A
            Quu = R*dt + Bu.T @ Vxx @ Bu
            Qvv = -Rv*dt + Bv.T @ Vxx @ Bv

            Qux = Bu.T @ Vxx @ A
            Qvx = Bv.T @ Vxx @ A
            Quv = Bu.T @ Vxx @ Bv
            Qvu = Quv.T

            Quu += 1e-5*np.eye(nu)
            Qvv -= 1e-5*np.eye(nv)

            inv_Qvv = np.linalg.inv(Qvv)
            S = Quu - Quv @ inv_Qvv @ Qvu
            S += 1e-5*np.eye(nu)
            inv_S = np.linalg.inv(S)

            k_u[k] = -inv_S @ (Qu - Quv @ inv_Qvv @ Qv)
            k_v[k] = -inv_Qvv @ (Qv + Qvu @ k_u[k])

            K_u[k] = -inv_S @ (Qux - Quv @ inv_Qvv @ Qvx)
            K_v[k] = -inv_Qvv @ (Qvx + Qvu @ K_u[k])

            Vx = Qx + Qux.T @ k_u[k] + Qvx.T @ k_v[k]
            Vxx = Qxx + Qux.T @ K_u[k] + Qvx.T @ K_v[k]
            Vxx = 0.5*(Vxx + Vxx.T)

        x_new = np.zeros_like(x)
        u_new = np.zeros_like(u)
        v_new = np.zeros_like(v)
        x_new[0] = x0

        for k in range(N):

            dx_val = x_new[k] - x[k]
            du = k_u[k] + K_u[k] @ dx_val
            dv = k_v[k] + K_v[k] @ dx_val

            u_new[k] = np.clip(u[k] + alpha*du, u_min, u_max)
            v_new[k] = np.clip(v[k] + alpha*dv, v_min, v_max)

            x_new[k+1] = x_new[k] + dt*f(x_new[k],u_new[k],v_new[k])

        u = u_new
        v = v_new
        x = x_new

    constraint = x[-1,8] - S_max
    lam += rho * constraint
    print("Violation integral =", x[-1,8])

# ======================================
# PLOTS
# ======================================

t = np.linspace(0, T_final, N+1)
t_u = np.linspace(0, T_final-dt, N)

# ---- Trajectory plot ----
plt.figure(figsize=(7,6))
plt.plot(x[:,0], x[:,1], linewidth=2.5, label='Trajectory')
plt.scatter(x0[0], x0[1], color='g', s=100, label='Start')
plt.scatter(goal[0], goal[1], color='r', s=100, label='Goal')


circle = plt.Circle(
    obs_center, obs_radius,
    edgecolor='black',
    fill=True, linewidth=2
)
plt.gca().add_patch(circle)

safe_circle = plt.Circle(
    obs_center, R_safe,
    edgecolor='red', facecolor='lightcoral',
    fill=True, linestyle='--', linewidth=2, alpha=0.35
)
plt.gca().add_patch(safe_circle)

plt.xlabel(r'$p_x$', fontsize=14)
plt.ylabel(r'$p_y$', fontsize=14)
plt.axis('equal')
plt.title("Obstacle Avoidance")
plt.grid(True)
plt.legend()

# ---- Integrated obstacle violation ----
plt.figure(figsize=(7,5))
plt.plot(t, x[:,8], linewidth=2)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Integrated violation', fontsize=12)
plt.title("Integrated Obstacle Violation")
plt.grid(True)

# ---- Cost history for each outer iteration ----
# ---- Cost history for each outer iteration ----
plt.figure(figsize=(8,5))
inner_iters = np.arange(1, max_iter_inner + 1)

colors = plt.cm.tab10(np.linspace(0, 1, max_iter_outer))

for outer in range(max_iter_outer):

    if outer == 0:
        plt.plot(
            inner_iters,
            cost_history[outer, :],
            linewidth=3.5,
            color=colors[outer],
            alpha=1.0,
            label='Outer loop 1',
            zorder=3
        )

    elif outer == max_iter_outer - 1:
        plt.plot(
            inner_iters,
            cost_history[outer, :],
            linewidth=3.5,
            color=colors[outer],
            alpha=1.0,
            label=f'Outer loop {max_iter_outer}',
            zorder=3
        )

    else:
        plt.plot(
            inner_iters,
            cost_history[outer, :],
            linewidth=1.2,
            color=colors[outer],
            alpha=0.55,
            label=f'Outer loop {outer+1}',
            zorder=1
        )

plt.xlabel('Inner iteration', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title("Cost Function per Outer Iteration")
plt.grid(True)
plt.legend()

# ---- Control u ----
plt.figure(figsize=(8,5))
plt.plot(t_u, u[:,0], linewidth=2, label=r'$u_1$')
plt.plot(t_u, u[:,1], linewidth=2, label=r'$u_2$')
plt.axhline(u_min[0], color='k', linestyle='-', linewidth=1.5, label=r'$u_{\min}$')
plt.axhline(u_max[0], color='k', linestyle='-', linewidth=1.5, label=r'$u_{\max}$')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Control u', fontsize=12)
plt.grid(True)
plt.legend()

# ---- Disturbance v ----
plt.figure(figsize=(8,5))
plt.plot(t_u, v[:,0], linewidth=2, label=r'$v_1$')
plt.plot(t_u, v[:,1], linewidth=2, label=r'$v_2$')
plt.axhline(v_min[0], color='k', linestyle='-', linewidth=1.5, label=r'$v_{\min}$')
plt.axhline(v_max[0], color='k', linestyle='-', linewidth=1.5, label=r'$v_{\max}$')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Control v', fontsize=12)
plt.grid(True)
plt.legend()

plt.show()