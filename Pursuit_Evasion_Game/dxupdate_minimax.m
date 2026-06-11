function dxprime = dxupdate_minimax(t, dx, flag, u, v, x_traj, V)

global m1;
global grav;
global l1;
global I1;
global b1;
global Q_f;
global Ru;
global Rv;
global kappa;
global obs_center R_safe;

global vp ve;

xp_t = ppval(x_traj(1), t);
yp_t = ppval(x_traj(2), t);
theta_p = ppval(x_traj(3), t);
xe_t = ppval(x_traj(4), t);
ye_t = ppval(x_traj(5), t);
z_t  = ppval(x_traj(6), t);   % NEW

Vx = [ ...
ppval(V(2),t); 
ppval(V(3),t); 
ppval(V(4),t); 
ppval(V(5),t); 
ppval(V(6),t); 
ppval(V(7),t)];

Vxx = [ ...
ppval(V(8),t)  ppval(V(14),t) ppval(V(20),t) ppval(V(26),t) ppval(V(32),t) ppval(V(38),t); ...
ppval(V(9),t)  ppval(V(15),t) ppval(V(21),t) ppval(V(27),t) ppval(V(33),t) ppval(V(39),t); ...
ppval(V(10),t) ppval(V(16),t) ppval(V(22),t) ppval(V(28),t) ppval(V(34),t) ppval(V(40),t); ...
ppval(V(11),t) ppval(V(17),t) ppval(V(23),t) ppval(V(29),t) ppval(V(35),t) ppval(V(41),t); ...
ppval(V(12),t) ppval(V(18),t) ppval(V(24),t) ppval(V(30),t) ppval(V(36),t) ppval(V(42),t); ...
ppval(V(13),t) ppval(V(19),t) ppval(V(25),t) ppval(V(31),t) ppval(V(37),t) ppval(V(43),t)];
dim_x = size(x_traj, 2);
dim_u = size(u, 2);
dim_v = size(v, 2);

% pde of dynamics F
dx_obs = xp_t - obs_center(1);
dy_obs = yp_t - obs_center(2);
dist = sqrt(dx_obs^2 + dy_obs^2);

if dist < R_safe && dist > 1e-6
    dz_dx = -2*(R_safe - dist)*(dx_obs/dist);
    dz_dy = -2*(R_safe - dist)*(dy_obs/dist);
else
    dz_dx = 0;
    dz_dy = 0;
end

Fx = [ ...
0 0 -vp*sin(theta_p) 0 0 0;
0 0  vp*cos(theta_p) 0 0 0;
0 0  0               0 0 0;
0 0  0               0 0 0;
0 0  0               0 0 0;
dz_dx dz_dy 0 0 0 0];
Fu = [0;0;1;0;0;0];

Fv = [0;0;0;-ve*sin(ppval(v,t));ve*cos(ppval(v,t));0];


% pde of running cost L
Lu = Ru*ppval(u,t);
Lv = (xp_t-xe_t)*ve*sin(ppval(v,t)) - (yp_t-ye_t)*ve*cos(ppval(v,t));
Luu = Ru;
Lvv = (xp_t-xe_t)*ve*cos(ppval(v,t)) + (yp_t-ye_t)*ve*sin(ppval(v,t));
Lux = zeros(dim_u,dim_x);
Lxu = Lux';
Lvx = [ve*sin(ppval(v,t)), -ve*cos(ppval(v,t)), 0, -ve*sin(ppval(v,t)), ve*cos(ppval(v,t)), 0];
Lxv = Lvx';
Luv = zeros(dim_u,dim_v);
Lvu = zeros(dim_v,dim_u);

Qu = Fu'*Vx + Lu;
Qv = Fv'*Vx + Lv;
Quu = Luu;
Qvv = Lvv;
Qux = 1/2*(Lux) + 1/2*(Lxu)' + Fu'*Vxx;
Qvx = 1/2*(Lvx) + 1/2*(Lxv)' + Fv'*Vxx;
Quv = 1/2*(Luv) + 1/2*(Lvu)';
Qvu = Quv';

lu = -(Quu - Quv/Qvv*Qvu)\(Qu - Quv/Qvv*Qv);
lv = -(Qvv - Qvu/Quu*Quv)\(Qv - Qvu/Quu*Qu);
Ku = -(Quu - Quv/Qvv*Qvu)\(Qux - Quv/Qvv*Qvx);
Kv = -(Qvv - Qvu/Quu*Quv)\(Qvx - Qvu/Quu*Qux);

du = lu + Ku * dx;
dv = lv + Kv * dx;

dxprime = [-vp*sin(theta_p)*dx(3); ...
           vp*cos(theta_p)*dx(3); ...
           du; ...
           -ve*sin(ppval(v,t))*dv; ...
           ve*cos(ppval(v,t))*dv;...
           0];
end