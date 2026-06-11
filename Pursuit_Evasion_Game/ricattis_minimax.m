function vprime = ricattis_minimax(t, V, flag, u, v, x_traj)
% v = [V, Vx(1), Vx(2), Vxx(1, 1), Vxx(1, 2), Vxx(2, 1), Vxx(2, 2)]

global kappa Ru;

global vp ve;
global obs_center R_safe;

xp_t = ppval(x_traj(1), t);
yp_t = ppval(x_traj(2), t);
theta_p = ppval(x_traj(3), t);
xe_t = ppval(x_traj(4), t);
ye_t = ppval(x_traj(5), t);
z_t  = ppval(x_traj(6), t);   % NEW

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
L = (xp_t - xe_t)*(vp*cos(theta_p) - ve*cos(ppval(v,t))) + (yp_t - ye_t)*(vp*sin(theta_p) - ve*sin(ppval(v,t))) + 0.5*Ru*(ppval(u,t))^2;
Lx = [vp*cos(theta_p) - ve*cos(ppval(v,t)); vp*sin(theta_p) - ve*sin(ppval(v,t)); -(xp_t-xe_t)*vp*sin(theta_p) + (yp_t-ye_t)*vp*cos(theta_p); -(vp*cos(theta_p) - ve*cos(ppval(v,t))); -(vp*sin(theta_p) - ve*sin(ppval(v,t)));0];
Lu = Ru*ppval(u,t);
Lv = (xp_t-xe_t)*ve*sin(ppval(v,t)) - (yp_t-ye_t)*ve*cos(ppval(v,t));
Lxx = [0 0 -vp*sin(theta_p) 0 0 0; 0 0 vp*cos(theta_p) 0 0 0; vp*sin(theta_p) -vp*cos(theta_p) -(xp_t-xe_t)*vp*cos(theta_p)-(yp_t-ye_t)*vp*sin(theta_p) -vp*sin(theta_p) vp*cos(theta_p) 0; 0 0 vp*sin(theta_p) 0 0 0; 0 0 -vp*cos(theta_p) 0 0 0; 0 0 0 0 0 0];
Luu = Ru;
Lvv = (xp_t-xe_t)*ve*cos(ppval(v,t)) + (yp_t-ye_t)*ve*sin(ppval(v,t));
Lux = zeros(dim_u,dim_x);
Lxu = Lux';
Lvx = [ve*sin(ppval(v,t)), -ve*cos(ppval(v,t)), 0, -ve*sin(ppval(v,t)), ve*cos(ppval(v,t)), 0];
Lxv = Lvx';
Luv = zeros(dim_u,dim_v);
Lvu = zeros(dim_v,dim_u);                                                                   

Vx = [V(2); V(3); V(4); V(5); V(6); V(7)];
Vxx = [ ...
V(8)  V(14) V(20) V(26) V(32) V(38);
V(9)  V(15) V(21) V(27) V(33) V(39);
V(10) V(16) V(22) V(28) V(34) V(40);
V(11) V(17) V(23) V(29) V(35) V(41);
V(12) V(18) V(24) V(30) V(36) V(42);
V(13) V(19) V(25) V(31) V(37) V(43)];
det_Vxx = det(Vxx);
% if det_Vxx < 0
%     fprintf('Time %f,  det_Vxx = %e, Vxx = [%f %f; %f %f] \n',t, det_Vxx, Vxx(1,1), Vxx(1,2), Vxx(2,1), Vxx(2,2));
% end

Qx = Fx'*Vx + Lx;
Qu = Fu'*Vx + Lu;
Qv = Fv'*Vx + Lv;
Qxx = Lxx + 2*Vxx*Fx;
Quu = Luu;
Qvv = Lvv;
Qux = 1/2*(Lux) + 1/2*(Lxu)' + Fu'*Vxx;
Qvx = 1/2*(Lvx) + 1/2*(Lxv)' + Fv'*Vxx;
Quv = 1/2*(Luv) + 1/2*(Lvu)';
Qvu = Quv';

epsilon = 1e-3;
Inv_u = (Quu - Quv/Qvv*Qvu);
Inv_v = (Qvv - Qvu/Quu*Quv);
% Make sure Inv_u > 0 and Inv_v < 0.
if abs(det(Inv_u)) < epsilon;
    Inv_u = Inv_u + 10*epsilon*eye(size(Inv_u));
end
if abs(det(Inv_v)) < epsilon;
    Inv_v = Inv_v - 10*epsilon*eye(size(Inv_u));
end

lu = -Inv_u\(Qu - Quv/Qvv*Qv);
lv = -Inv_v\(Qv - Qvu/Quu*Qu);
Ku = -Inv_u\(Qux - Quv/Qvv*Qvx);
Kv = -Inv_v\(Qvx - Qvu/Quu*Qux);

dVdt = -(L + lu'*Qu + lv'*Qv + 0.5*lu'*Quu*lu + lu'*Quv*lv + 0.5*lv'*Qvv*lv);
dVxdt =-(Qx + Ku'*Qu + Kv'*Qv + Qux'*lu + Qvx'*lv + Ku'*Quu*lu + Kv'*Qvv*lv + Ku'*Quv*lv + Kv'*Qvu*lu);
dVxxdt =-(Qxx + Ku'*Quu*Ku + Kv'*Qvv*Kv + 2*Ku'*Qux + 2*Kv'*Qvx + 2*Ku'*Quv*Kv);

dVxxdt = dVxxdt/2 + dVxxdt'/2; % for symmetry

vprime = [dVdt; ...
    dVxdt(1); ...
    dVxdt(2); ...
    dVxdt(3); ...
    dVxdt(4); ...
    dVxdt(5); ...
    dVxdt(6); ...
    dVxxdt(1,1); ...
    dVxxdt(2,1); ...
    dVxxdt(3,1); ...
    dVxxdt(4,1); ...
    dVxxdt(5,1); ...
    dVxxdt(6,1); ...
    dVxxdt(1,2); ...
    dVxxdt(2,2); ...
    dVxxdt(3,2); ...
    dVxxdt(4,2); ...
    dVxxdt(5,2); ...
    dVxxdt(6,2); ...
    dVxxdt(1,3); ...
    dVxxdt(2,3); ...
    dVxxdt(3,3); ...
    dVxxdt(4,3); ...
    dVxxdt(5,3); ...
    dVxxdt(6,3); ...
    dVxxdt(1,4); ...
    dVxxdt(2,4); ...
    dVxxdt(3,4); ...
    dVxxdt(4,4); ...
    dVxxdt(5,4); ...
    dVxxdt(6,4); ...
    dVxxdt(1,5); ...
    dVxxdt(2,5); ...
    dVxxdt(3,5); ...
    dVxxdt(4,5); ...
    dVxxdt(5,5); ...
    dVxxdt(6,5); ...
    dVxxdt(1,6); ...
    dVxxdt(2,6); ...
    dVxxdt(3,6); ...
    dVxxdt(4,6); ...
    dVxxdt(5,6); ...
    dVxxdt(6,6)];
end


