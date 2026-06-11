import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ======================================
# PARAMETERS
# ======================================

T_final = 4.0
N = 100
dt = T_final / N
g = 9.81

nx = 13
nu = 4
nv = 4

x0 = np.zeros(nx)

goal = np.array([
    5.,5.,5.,      # position
    0.,0.,0.,      # velocity
    0.,0.,0.,      # angles
    0.,0.,0.,      # angular rates
    0.
])

max_iter_inner = 20
max_iter_outer = 10
alpha = 1

# ======================================
# FUNCTIONAL CONSTRAINT
# ======================================

S_max = 20.0
lam = 0.0
rho = 200.0

# ======================================
# CONTROL LIMITS
# ======================================

u_min = np.array([-100., -100., -100., -100.])
u_max = np.array([ 100.,  100.,  100.,  100.])

v_min = np.array([-50., -30., -30., -30.])
v_max = np.array([ 50.,  30.,  30.,  30.])


# ======================================
# COST MATRICES
# ======================================

Q  = np.diag([
    20,20,20,      # position
    20,20,20,      # velocity (increased)
    10,10,10,      # angles
    10,10,10,      # angular rates (increased)
    0
])

R  = 0.01*np.eye(nu)
Rv = 1.0*np.eye(nv)

Qf = 20*np.diag([
    40,10,20,
    20,20,20,
    20,20,20,
    10,10,10,
    0
])

# ======================================
# DYNAMICS (OPTION 2)
# ======================================

def f(x,u,v):

    px,py,pz,vx,vy,vz,phi,theta,psi,p,q,r,s = x
    dx = np.zeros(nx)

    m = 1.0
    g = 9.81

    # Hover-centered thrust
    T = m*g + u[0] + v[0]

    # Rotation matrix
    cphi = np.cos(phi); sphi = np.sin(phi)
    cth  = np.cos(theta); sth = np.sin(theta)
    cpsi = np.cos(psi); spsi = np.sin(psi)

    Rmat = np.array([
        [cth*cpsi,  sphi*sth*cpsi - cphi*spsi,  cphi*sth*cpsi + sphi*spsi],
        [cth*spsi,  sphi*sth*spsi + cphi*cpsi,  cphi*sth*spsi - sphi*cpsi],
        [-sth,      sphi*cth,                   cphi*cth]
    ])

    # Position
    dx[0] = vx
    dx[1] = vy
    dx[2] = vz

    # Translational dynamics
    thrust_world = Rmat @ np.array([0,0,T])
    gravity = np.array([0,0,m*g])
    acc = (thrust_world - gravity)/m

    dx[3] = acc[0] - 0.2*vx
    dx[4] = acc[1] - 0.2*vy
    dx[5] = acc[2] - 0.2*vz

    # Attitude kinematics
    dx[6] = p
    dx[7] = q
    dx[8] = r

    # Angular rate dynamics (simplified)
    dx[9]  = -0.6*p + u[1] + v[1]
    dx[10] = -0.6*q + u[2] + v[2]
    dx[11] = -0.6*r + u[3] + v[3]

    # Path length
    dx[12] = np.sqrt(vx**2 + vy**2 + vz**2)+ 0.5 * psi**2
    #dx[12] = np.sqrt(vx**2 + vy**2 + vz**2) + 0.5 * psi**2

    return dx

# ======================================
# INITIAL TRAJECTORY
# ======================================

u = np.zeros((N,nu))
v = np.zeros((N,nv))
x = np.zeros((N+1,nx))

# ======================================
# OUTER LOOP
# ======================================

for outer in range(max_iter_outer):

    print("\n===== OUTER ITER", outer, "=====")

    for it in range(max_iter_inner):

        x[0] = x0
        for k in range(N):
            x[k+1] = x[k] + dt*f(x[k],u[k],v[k])

        terminal_error = x[-1] - goal
        constraint = x[-1,12] - S_max

        Vx = Qf @ terminal_error
        Vx[12] += lam + rho*constraint

        Vxx = Qf.copy()
        Vxx[12,12] += rho

        k_u = np.zeros((N,nu))
        K_u = np.zeros((N,nu,nx))
        k_v = np.zeros((N,nv))
        K_v = np.zeros((N,nv,nx))

        for k in reversed(range(N)):

            dx_state = x[k] - goal
            dx_state[12] = 0

            lx = Q @ dx_state
            lu = R @ u[k]
            lv = -Rv @ v[k]

            A = np.eye(nx)
            Bu = np.zeros((nx,nu))
            Bv = np.zeros((nx,nv))

            eps = 1e-6

            for i in range(nx):
                xp = x[k].copy()
                xp[i] += eps
                A[:,i] += dt*(f(xp,u[k],v[k]) - f(x[k],u[k],v[k]))/eps

            for i in range(nu):
                up = u[k].copy()
                up[i] += eps
                Bu[:,i] = dt*(f(x[k],up,v[k]) - f(x[k],u[k],v[k]))/eps

            for i in range(nv):
                vp = v[k].copy()
                vp[i] += eps
                Bv[:,i] = dt*(f(x[k],u[k],vp) - f(x[k],u[k],v[k]))/eps

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

            Vx = Qx
            Vxx = Qxx

        # Forward rollout
        x_new = np.zeros_like(x)
        u_new = np.zeros_like(u)
        v_new = np.zeros_like(v)
        x_new[0] = x0

        for k in range(N):

            dx_val = x_new[k] - x[k]
            du = k_u[k] + K_u[k] @ dx_val
            dv = k_v[k] + K_v[k] @ dx_val

            def qp_obj(y):
                return np.sum((y - np.concatenate((du,dv)))**2)

            bounds = []
            for i in range(nu):
                bounds.append((u_min[i]-u[k,i], u_max[i]-u[k,i]))
            for i in range(nv):
                bounds.append((v_min[i]-v[k,i], v_max[i]-v[k,i]))

            res = minimize(qp_obj,
                           np.concatenate((du,dv)),
                           bounds=bounds,
                           method='SLSQP')

            y = res.x
            u_new[k] = u[k] + alpha*y[:nu]
            v_new[k] = v[k] + alpha*y[nu:]
            x_new[k+1] = x_new[k] + dt*f(x_new[k],u_new[k],v_new[k])

        u = u_new
        v = v_new
        x = x_new

    constraint = x[-1,12] - S_max
    lam += rho * constraint
    print("Path integral =", x[-1,12])

# ======================================
# 3D PLOT
# ======================================

t = np.linspace(0, T_final, N+1)

# ---- 3D trajectory ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[:,0], x[:,1], x[:,2])
ax.scatter(goal[0], goal[1], goal[2], color='r')
ax.set_title("3D Trajectory")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# ---- Position vs time ----
plt.figure()
plt.plot(t, x[:,0], label="px")
plt.plot(t, x[:,1], label="py")
plt.plot(t, x[:,2], label="pz")
plt.title("Position")
plt.legend()
plt.grid()

# ---- Velocity vs time ----
plt.figure()
plt.plot(t, x[:,3], label="vx")
plt.plot(t, x[:,4], label="vy")
plt.plot(t, x[:,5], label="vz")
plt.title("Velocity")
plt.legend()
plt.grid()

# ---- Angles vs time ----
plt.figure()
plt.plot(t, x[:,6], label="phi")
plt.plot(t, x[:,7], label="theta")
plt.plot(t, x[:,8], label="psi")
plt.title("Angles")
plt.legend()
plt.grid()

# ---- Angular rates ----
plt.figure()
plt.plot(t, x[:,9], label="p")
plt.plot(t, x[:,10], label="q")
plt.plot(t, x[:,11], label="r")
plt.title("Angular Rates")
plt.legend()
plt.grid()

# ---- Integral constraint ----
plt.figure()
plt.plot(t, x[:,12])
plt.axhline(S_max, color='r')
plt.title("Integral Constraint")
plt.grid()

plt.figure()
plt.plot(t[:-1],u[:,0],label="dT_u_1")
plt.axhline(u_max[0],color='r',linestyle='--')
plt.axhline(u_min[0],color='r',linestyle='--')
plt.title("Control dT")
plt.legend()
plt.grid()

plt.figure()
plt.plot(t[:-1],u[:,1],label="dT_u_2")
plt.axhline(u_max[1],color='r',linestyle='--')
plt.axhline(u_min[1],color='r',linestyle='--')
plt.title("Control dT")
plt.legend()
plt.grid()

plt.figure()
plt.plot(t[:-1],u[:,2],label="dT_u_3")
plt.axhline(u_max[2],color='r',linestyle='--')
plt.axhline(u_min[2],color='r',linestyle='--')
plt.title("Control dT")
plt.legend()
plt.grid()

plt.figure()
plt.plot(t[:-1],u[:,3],label="dT_u_4")
#plt.axhline(u_max[3],color='r',linestyle='--')
#plt.axhline(u_min[3],color='r',linestyle='--')
plt.title("Control dT")
plt.legend()
plt.grid()


plt.show()