clc;
clear ;
close all;

global Q_f;
global Ru;
global Rv;
global kappa;
global obs_center R_safe;
global vp ve;

% Parameter
vp = 1;
ve = 0.3;

% Final Time


Tf = 2.1; % whether it can work depends on the initial control.

% Number of steps for integral + 1
Horizon = 101; 
% Horizon = 1001; 

% Time step
 dt = Tf/(Horizon-1);

% Number of Iterations of Control Updates

num_iter =30;
outer = 5;

% % Weight in Final State:
% Q_f = zeros(2,2);
% Q_f(1,1) = 1;
% Q_f(2,2) = 1;
% 
% % Weight in the Control u (Stablizer) :
Ru = 0.001;


% Target: 
p_target(1,1) = 0;
p_target(2,1) = 0;


% Learning Rate:

gamma = .1;

% kappa = 1 for 2nd order dynamics expansion, kappa = 0 for 1st order
% dynamics expansion.
kappa = 0;

% Time for Iteration 
tt = linspace(0, Tf, Horizon);
tt_u = tt(1:(Horizon - 1));

% Initial Control:          

u = 0.9*ones(1,Horizon-1);
v = 0.9*ones(1,Horizon-1);

du = zeros(1,Horizon-1);
dv = zeros(1,Horizon-1);

u_cont = spline(tt_u,u);
v_cont = spline(tt_u,v);
u_min = 0;
u_max =  4;

v_min = 0;    % optional
v_max = 2;    % optional


% Initial Configuration:
xo = [0 0 0 1 1 0];

obs_center = [0.6, 0.5];
obs_radius = 0.2;
safe_margin = 0.05;
R_safe = obs_radius + safe_margin;
% constraint parameters
lambda = 0;
rho = 10000;
S_max = 0;
% Initial trajectory:
options= odeset('OutputFcn','');
% options= odeset('RelTol',1e-3,'AbsTol',1e-4);
 [T, x_traj] = ode45('pe_game', tt, xo, options, u_cont, v_cont);
 

for i = 1:6
    x_traj_cont(i) = spline(tt, x_traj(:, i));
end
for p = 1:outer
for k = 1:num_iter
% 
% [Cost(:,k)] =  fnCostComp_minimax(x_traj,u,v,p_target,dt);
% 
% fprintf('Iteration %d,  Current Cost = %e \n',k,Cost(1,k));
% 
   
   figure(1); clf;
%    figure();
   subplot(2,2,1)
   hold on
   plot(xo(4), xo(5), '.b', 'MarkerSize', 30)
    plot(x_traj(:,4),x_traj(:,5),'b','linewidth',4)
    plot(x_traj(end,4),x_traj(end,5), '.b', 'MarkerSize', 30)
   plot(xo(1), xo(2), '.r', 'MarkerSize', 30)
   plot(x_traj(:,1),x_traj(:,2),'--r','linewidth',4); 
   plot(x_traj(end,1),x_traj(end,2), '.r', 'MarkerSize', 30)
   % obstacle disk
    theta = linspace(0,2*pi,100);
    obs_x = obs_center(1) + obs_radius*cos(theta);
    obs_y = obs_center(2) + obs_radius*sin(theta);
    plot(obs_x, obs_y, 'k','LineWidth',2)

% safe disk
% safe_x = obs_center(1) + R_safe*cos(theta);
% safe_y = obs_center(2) + R_safe*sin(theta);
% plot(safe_x, safe_y, 'r--','LineWidth',2)
   title('Trajectories','fontsize',20); 
   xlabel('x','fontsize',20)
   ylabel('y','fontsize',20)
   hold off;
   axis equal
%    axis([0, 4, 0, 4])
   grid;
   
   
   subplot(2,2,2);hold on
   plot(tt_u, u(1, :), 'linewidth',2);
   xlabel('Time in sec', 'fontsize',20)
   title('Stabilizing Control', 'fontsize',20)
   hold off;
   
   subplot(2,2,3);hold on
   plot(tt_u, v(1, :), 'linewidth',2);
   xlabel('Time in sec', 'fontsize',20)
   title('Destabilizing Control', 'fontsize',20)
   hold off;
% end
%------------------------------------------------> Initial Condition of the
% Value function
Vxx_end = zeros(6);
Vx_end  = zeros(6,1);


phi = x_traj(end,6) - S_max;

Vx_end(6) = lambda + rho*phi;
Vxx_end(6,6) = rho;
V_end = 0; 

Vo = [V_end, Vx_end(:)', Vxx_end(:)'];

%------------------------------------------------> Integrate Backward the Value Function
tt_back = linspace(Tf, 0, Horizon);

[T, Vvalue] = ode45('ricattis_minimax', tt_back, Vo, options, u_cont, v_cont, x_traj_cont);
% [T, Vvalue] = ode23s('ricattis_minimax', tt_back, Vo, options, u_cont, v_cont, x_traj_cont);


V = Vvalue(end:-1:1, 1);
Vx = Vvalue(end:-1:1, 2:7)';
for i = 1:Horizon
    Vxx(1:6,1,i) = Vvalue(end+1-i, 8:13);
    Vxx(1:6,2,i) = Vvalue(end+1-i, 14:19);
    Vxx(1:6,3,i) = Vvalue(end+1-i, 20:25);
    Vxx(1:6,4,i) = Vvalue(end+1-i, 26:31);
    Vxx(1:6,5,i) = Vvalue(end+1-i, 32:37);
    Vxx(1:6,6,i) = Vvalue(end+1-i, 38:43);
end

for i = 1:size(Vvalue, 2)
    V_cont(i) = spline(tt, Vvalue(end:-1:1, i));
end

%% 
%----------------------------------------------> Update the controls
dxo = [0; 0; 0; 0;0;0];
% du_cont = spline(tt_u,du);
[T, dx] = ode45('dxupdate_minimax', tt, dxo, options, u_cont, v_cont, x_traj_cont, V_cont);
% [T, dx] = ode23s('dxupdate_minimax', tt, dxo, options, u_cont, v_cont, x_traj_cont, V_cont);

for i = 1:(Horizon-1)
    
    dim_x = size(x_traj_cont, 2);
    dim_u = size(u_cont, 2);
    dim_v = size(v_cont, 2);
    
    xp_t = x_traj(i,1);
    yp_t = x_traj(i,2);
    theta_p = x_traj(i,3);
    xe_t = x_traj(i,4);
    ye_t = x_traj(i,5);
    z_t  = x_traj(i,6);
    
    % pde of dynamics F
    % Fx = zeros(4);
    Fu = [0;0;1;0;0;0];
    Fv = [0;0;0;-ve*sin(v(i));ve*cos(v(i));0];

    Fxx2 = zeros(dim_x);
    Fuu1 = -vp*cos(u(i));
    Fuu2 = -vp*sin(u(i));
    Fvv3 = -ve*cos(v(i));
    Fvv4 = -ve*sin(v(i));
    Fux = zeros(dim_u,dim_x);
    Fxu = zeros(dim_x,dim_u);
    Fvx = zeros(dim_v,dim_x);
    Fxv = zeros(dim_x,dim_v);
    Fuv = zeros(dim_u,dim_v);
    Fvu = zeros(dim_v,dim_u);
    
    % pde of running cost L
    Lu = Ru*u(i);
    Lv = (xp_t-xe_t)*ve*sin(v(i)) - (yp_t-ye_t)*ve*cos(v(i));
    Luu = Ru;
    Lvv = (xp_t-xe_t)*ve*cos(v(i)) + (yp_t-ye_t)*ve*sin(v(i));
    Lvx = [ve*sin(v(i)), -ve*cos(v(i)), 0, -ve*sin(v(i)), ve*cos(v(i)),0];
    Lxv = Lvx';
    Luv = zeros(dim_u,dim_v);
    Lvu = zeros(dim_v,dim_u);
    
    Qu = Fu'*Vx(:, i) + Lu;
    Qv = Fv'*Vx(:, i) + Lv;
    Quu = Luu;
    Qvv = Lvv;
    Qux = Fu'*Vxx(:, :, i);
    Qvx = 1/2*(Lvx) + 1/2*(Lxv)' + Fv'*Vxx(:, :, i);
    Quv = 1/2*(Luv + kappa*Fuv) + 1/2*(Lvu)';
    Qvu = Quv';
    
    lu(i) = -(Quu)\(Qu);
    lv(i) = -(Qvv)\(Qv);
    Ku(:, i) = -(Quu)\(Qux);
    Kv(:, i) = -(Qvv)\(Qvx);
    
    % Inv_u(i) = (Quu - Quv/Qvv*Qvu);
    % Inv_v(i) = (Qvv - Qvu/Quu*Quv);
    
    du(i) = lu(i) + Ku(:, i)' * dx(i, :)';
    dv(i) = lv(i) + Kv(:, i)' * dx(i, :)';
    % H = [ Quu,   0;
    %     0,  -Qvv ];
    % 
    % f = [ Qu + Qux * dx(i, :)';
    %     -Qv - Qvx * dx(i, :)' ];
    % 
    % lb = [ u_min - u(i);
    %     v_min - v(i) ];
    % 
    % ub = [ u_max - u(i);
    %     v_max - v(i) ];
    % 
    % y = quadprog(H, f, [], [], [], [], lb, ub);
    % 
    % du(i) = y(1);
    % dv(i) = y(2);
    
    u_new(i) = u(i) + gamma*du(i);
    v_new(i) = v(i) + gamma*dv(i);
end

% figure()
% subplot(2,1,1); plot(Inv_u)
% subplot(2,1,2); plot(Inv_v)

u = u_new;
v = v_new;

%%
%---------------------------------------------> Simulation of the Nonlinear System
% [x_traj] = fnsimulateDDP(xo,u_new,Horizon,dt,0);
[Cost(:,k)] =  fnCostComp_minimax(x_traj,u,v,p_target,dt);
% x1(k,:) = x_traj(1,:);

u_cont = spline(tt_u,u);
v_cont = spline(tt_u,v);
[T, x_traj] = ode45('pe_game', tt, xo, options, u_cont, v_cont);
%  [T, x_traj] = ode23s('pe_game', tt, xo, options, u_cont, v_cont);
 
 for i = 1:size(xo, 2)
    x_traj_cont(i) = spline(tt, x_traj(:, i));
 end

 
end
phi = x_traj(end,6) - S_max;
lambda = lambda + rho*phi;
end

   time(1)=0;
   for i= 2:Horizon
    time(i) =time(i-1) + dt;  
   end

      
%% ---------------------------------------------> Plot Section

   figure(1)
hold on

% trajectories
plot(xo(4), xo(5), '.b', 'MarkerSize', 30)
plot(x_traj(:,4), x_traj(:,5), 'b', 'LineWidth', 2)
plot(x_traj(end,4), x_traj(end,5), '.b', 'MarkerSize', 30)

plot(xo(1), xo(2), '.r', 'MarkerSize', 30)
plot(x_traj(:,1), x_traj(:,2), 'r', 'LineWidth', 2)
plot(x_traj(end,1), x_traj(end,2), '.r', 'MarkerSize', 30)

% disks
theta = linspace(0, 2*pi, 100);

% safe disk: light red
safe_x = obs_center(1) + R_safe*cos(theta);
safe_y = obs_center(2) + R_safe*sin(theta);
fill(safe_x, safe_y, [1 0.6 0.6], ...
    'FaceAlpha', 0.3, ...
    'EdgeColor', 'r', ...
    'LineStyle', '--', ...
    'LineWidth', 2);

% obstacle disk: lighter black (gray-like because of transparency)
obs_x = obs_center(1) + obs_radius*cos(theta);
obs_y = obs_center(2) + obs_radius*sin(theta);
fill(obs_x, obs_y, 'b', ...
    'FaceAlpha', 0.6, ...
    'EdgeColor', 'k', ...
    'LineWidth', 2);

title('Trajectories', 'FontSize', 20)
xlabel('$p_x$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$p_y$', 'Interpreter', 'latex', 'FontSize', 20)
axis equal
grid on
hold off

figure (2); 
subplot(2,1,1)
hold on
plot(tt_u, u,'b', 'linewidth',2)
%plot(tt_u, u_min*ones(1,Horizon-1),'k', 'linewidth',2)
%plot(tt_u, u_max*ones(1,Horizon-1),'k', 'linewidth',2)
xlabel('Time in sec', 'fontsize',20)
title('Pursuer Control', 'fontsize',20)
hold off


subplot(2,1,2); hold on
plot(tt_u, v,'r', 'linewidth',2)
%plot(tt_u, v_min*ones(1,Horizon-1),'k', 'linewidth',2)
%plot(tt_u, v_max*ones(1,Horizon-1),'k', 'linewidth',2)
xlabel('Time in sec', 'fontsize',20)
title('Evader Control', 'fontsize',20)
hold off
% 
figure(3)
   hold on
   plot(Cost, '--rs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor', 'y'); 
   xlabel('Iterations','fontsize',20)
   title('Cost','fontsize',20); 

   figure(4)
clf
hold on

% Keep handle of the main axes
axMain = gca;

% -----------------------------
% Main plot
% -----------------------------

% trajectories
hStartBlue = plot(xo(4), xo(5), 's', 'MarkerSize', 30);
hBlueTraj  = plot(x_traj(:,4), x_traj(:,5), 'b', 'LineWidth', 2);
hEndBlue   = plot(x_traj(end,4), x_traj(end,5), '.b', 'MarkerSize', 30);

hStartRed = plot(xo(1), xo(2), 's', 'MarkerSize', 30);
hRedTraj  = plot(x_traj(:,1), x_traj(:,2), 'r', 'LineWidth', 2);
hEndRed   = plot(x_traj(end,1), x_traj(end,2), '.r', 'MarkerSize', 30);

% disks
theta = linspace(0, 2*pi, 100);

% safe disk
safe_x = obs_center(1) + R_safe*cos(theta);
safe_y = obs_center(2) + R_safe*sin(theta);
fill(safe_x, safe_y, [1 0.6 0.6], ...
    'FaceAlpha', 0.3, ...
    'EdgeColor', 'r', ...
    'LineStyle', '--', ...
    'LineWidth', 2, ...
    'HandleVisibility', 'off');

% obstacle disk
obs_x = obs_center(1) + obs_radius*cos(theta);
obs_y = obs_center(2) + obs_radius*sin(theta);
fill(obs_x, obs_y, 'b', ...
    'FaceAlpha', 0.6, ...
    'EdgeColor', 'k', ...
    'LineWidth', 2, ...
    'HandleVisibility', 'off');

% legend
legend([hStartBlue, hEndBlue, hBlueTraj, hStartRed, hEndRed, hRedTraj], ...
    {'Evader Start', 'Evader End', 'Evader Trajectory', ...
     'Pursuer Start', 'Pursuer End', 'Pursuer Trajectory'}, ...
    'Location', 'best', ...
    'FontSize', 12);

title('Trajectories', 'FontSize', 20)
xlabel('$p_x$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$p_y$', 'Interpreter', 'latex', 'FontSize', 20)
axis equal
grid on

% -----------------------------
% Zoom window on the main plot
% -----------------------------
zoomX = [0.6 0.9];
zoomY = [0.20 0.50];

rectangle(axMain, ...
    'Position', [zoomX(1), zoomY(1), diff(zoomX), diff(zoomY)], ...
    'EdgeColor', 'k', ...
    'LineStyle', '--', ...
    'LineWidth', 1.5, ...
    'HandleVisibility', 'off');

hold off

% -----------------------------
% Inset axes
% -----------------------------
axInset = axes('Position', [0.18 0.56 0.28 0.28]);  
% [left bottom width height] in figure coordinates

hold(axInset, 'on')
box(axInset, 'on')
grid(axInset, 'on')

% Draw disks first in inset
fill(axInset, safe_x, safe_y, [1 0.6 0.6], ...
    'FaceAlpha', 0.3, ...
    'EdgeColor', 'r', ...
    'LineStyle', '--', ...
    'LineWidth', 2, ...
    'HandleVisibility', 'off');

fill(axInset, obs_x, obs_y, 'b', ...
    'FaceAlpha', 0.6, ...
    'EdgeColor', 'k', ...
    'LineWidth', 2, ...
    'HandleVisibility', 'off');

% Replot trajectories in inset
plot(axInset, xo(4), xo(5), '.b', 'MarkerSize', 20, 'HandleVisibility', 'off')
plot(axInset, x_traj(:,4), x_traj(:,5), 'b', 'LineWidth', 2, 'HandleVisibility', 'off')
plot(axInset, x_traj(end,4), x_traj(end,5), '.b', 'MarkerSize', 20, 'HandleVisibility', 'off')

plot(axInset, xo(1), xo(2), '.r', 'MarkerSize', 20, 'HandleVisibility', 'off')
plot(axInset, x_traj(:,1), x_traj(:,2), 'r', 'LineWidth', 2, 'HandleVisibility', 'off')
plot(axInset, x_traj(end,1), x_traj(end,2), '.r', 'MarkerSize', 20, 'HandleVisibility', 'off')

axis(axInset, 'normal')
xlim(axInset, zoomX)
ylim(axInset, zoomY)
set(axInset, 'FontSize', 10)

hold(axInset, 'off')