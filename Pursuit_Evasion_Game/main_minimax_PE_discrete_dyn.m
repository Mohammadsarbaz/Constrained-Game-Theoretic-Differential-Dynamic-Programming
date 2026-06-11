%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Minimax DDP Inverted Pendulum                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Course: Advance Topics on Stochastic Optimal Control and Reinforcement Learning %%%%%%%%%%%%%%%%%%  
%%%%%%%%%%%%%%%%%%%%%  AE8803 Spring 2014                             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Author: Wei Sun                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
close all;

global Q_f;
global Ru;
global Rv;
global kappa;

global vp ve;

% Parameter
vp = 1;
ve = 0.8;

num_state = 4;
num_ctrl_u = 1;
num_ctrl_v = 1;

% Number of steps for integral + 1
Horizon = 100; 
Horizon = 500; 
Horizon = 600; 

% Simulations runs at 100Hz
dt = 0.01;

% Final Time
Tf = Horizon*dt;

% Number of Iterations of Control Updates
num_iter = 10;
num_iter = 1;

% Learning Rate:
% gamma = 0.1;
gamma = 0.4;
% gamma = 1;

% kappa = 1 for 2nd order dynamics expansion, kappa = 0 for 1st order
% dynamics expansion.
kappa = 1;
kappa = 0;

% Time for Iteration 
tt = linspace(0, Tf, Horizon);
tt_u = tt(1:(Horizon - 1));

% Initial Control:          
u = zeros(1,Horizon-1);
v = zeros(1,Horizon-1);

u = 0.1*ones(1,Horizon-1);
v = 0.1*ones(1,Horizon-1);

% u = 0.7*ones(1,Horizon-1);
% v = 0.7*ones(1,Horizon-1);

du = zeros(1,Horizon-1);
dv = zeros(1,Horizon-1);

u_cont = spline(tt_u,u);
v_cont = spline(tt_u,v);

% Initial Configuration:
% xo = [0 0 2 2];
xo = [0 0 1 1]';
% xo = [0 0 .5 .5];

for k = 1:num_iter
 
%%
%---------------------------------------------> Simulation of the Nonlinear System
[x_traj] = simulate(xo,u,v,Horizon,dt);

%---------------------------------------------> Cost calculation
[Cost(:,k)] =  fnCostComp_minimax_dis(x_traj,u,v,dt);
 
fprintf('Iteration %d,  Current Cost = %e \n',k,Cost(1,k));
 
%---------------------------------------------> Plot
%  if mod(k, 10) == 1
%    figure()
%    subplot(2,2,1)
%    hold on
%    plot(x_traj(:,3),x_traj(:,4),'b','linewidth',4);  
%    plot(x_traj(:,1),x_traj(:,2),'--r','linewidth',4); 
%    title('Trajectories','fontsize',20); 
%    xlabel('x','fontsize',20)
%    ylabel('y','fontsize',20)
%    hold off;
%    grid;
   
   
   figure(1); clf;
   subplot(2,2,1)
   hold on
   plot(xo(3), xo(4), '.b', 'MarkerSize', 30)
   plot(x_traj(3,:),x_traj(4,:),'b','linewidth',4);  
   plot(x_traj(3,end),x_traj(4,end), '.b', 'MarkerSize', 30)
   plot(xo(1), xo(2), '.r', 'MarkerSize', 30)
   plot(x_traj(1,:),x_traj(2,:),'--r','linewidth',4); 
   plot(x_traj(1,end),x_traj(2,end), '.r', 'MarkerSize', 30)
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
Vxx = zeros(num_state, num_state, Horizon);
Vx = zeros(num_state, Horizon);
V = zeros(1, Horizon);

Vxx(:,:,Horizon)= zeros(4);
Vx(:,Horizon) = zeros(4,1); 
V(Horizon) = 0; 

%------------------------------------------------> Propagate Backward the Value Function
for i = (Horizon-1):-1:1
    
    dim_x = size(x_traj, 1);
    dim_u = size(u, 1);
    dim_v = size(v, 1);
    
    xp_t = x_traj(1,i);
    yp_t = x_traj(2,i);
    xe_t = x_traj(3,i);
    ye_t = x_traj(4,i);
    
    % pde of dynamics F
    Fx = zeros(4);
    Fu = [-vp*sin(u(i)); vp*cos(u(i)); 0; 0];
    Fv = [0; 0; -ve*sin(v(i)); ve*cos(v(i))];
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
    L = (xp_t - xe_t)*(vp*cos(u(i)) - ve*cos(v(i))) + (yp_t - ye_t)*(vp*sin(u(i)) - ve*sin(v(i))) ;
    Lx = [(vp*cos(u(i)) - ve*cos(v(i))); (vp*sin(u(i)) - ve*sin(v(i))); -(vp*cos(u(i)) - ve*cos(v(i))); -(vp*sin(u(i)) - ve*sin(v(i)))];
    Lu = -(xp_t - xe_t)*vp*sin(u(i)) + (yp_t - ye_t)*vp*cos(u(i));
    Lv = (xp_t - xe_t)*ve*sin(v(i)) - (yp_t - ye_t)*ve*cos(v(i));
    Lxx = zeros(dim_x);
    Luu = -(xp_t - xe_t)*vp*cos(u(i)) - (yp_t - ye_t)*vp*sin(u(i));
    Lvv = (xp_t - xe_t)*ve*cos(v(i)) + (yp_t - ye_t)*ve*sin(v(i));
    Lux = [-vp*sin(u(i)), vp*cos(u(i)), vp*sin(u(i)), -vp*cos(u(i))];
    Lxu = Lux';
    Lvx = [ve*sin(v(i)), -ve*cos(v(i)), -ve*sin(v(i)), ve*cos(v(i))];
    Lxv = Lvx';
    Luv = zeros(dim_u,dim_v);
    Lvu = zeros(dim_v,dim_u);
    
    Qx = Fx'*Vx(:, i+1) + Lx;
    Qu = Fu'*Vx(:, i+1) + Lu;
    Qv = Fv'*Vx(:, i+1) + Lv;
    Qxx = Lxx + 2*Vxx(:, :, i+1)*Fx;
    Quu = Luu + kappa*Vx(1, i+1)*Fuu1 + kappa*Vx(2, i+1)*Fuu2;
    Qvv = Lvv + kappa*Vx(3, i+1)*Fvv3 + kappa*Vx(4, i+1)*Fvv4;
    Qux = 1/2*(Lux + kappa*Fux) + 1/2*(Lxu + kappa*Fxu)' + Fu'*Vxx(:, :, i+1);
    Qvx = 1/2*(Lvx + kappa*Fvx) + 1/2*(Lxv + kappa*Fxv)' + Fv'*Vxx(:, :, i+1);
    Quv = 1/2*(Luv + kappa*Fuv) + 1/2*(Lvu + kappa*Fvu)';
    Qvu = Quv';
    
    lu(i) = -(Quu - Quv/Qvv*Qvu)\(Qu - Quv/Qvv*Qv);
    lv(i) = -(Qvv - Qvu/Quu*Quv)\(Qv - Qvu/Quu*Qu);
    Ku(:, i) = -(Quu - Quv/Qvv*Qvu)\(Qux - Quv/Qvv*Qvx);
    Kv(:, i) = -(Qvv - Qvu/Quu*Quv)\(Qvx - Qvu/Quu*Qux);
    
    Inv_u(i) = (Quu - Quv/Qvv*Qvu);
    Inv_v(i) = (Qvv - Qvu/Quu*Quv);
    
    V(:,i) = (L + lu(i)'*Qu + lv(i)'*Qv + 0.5*lu(i)'*Quu*lu(i) + lu(i)'*Quv*lv(i) + 0.5*lv(i)'*Qvv*lv(i))*dt + V(i+1);
    Vx(:,i) = (Qx + Ku(:, i)*Qu + Kv(:, i)*Qv + Qux'*lu(i) + Qvx'*lv(i) + Ku(:, i)*Quu*lu(i) + Kv(:, i)*Qvv*lv(i) + Ku(:, i)*Quv*lv(i) + Kv(:, i)*Qvu*lu(i))*dt + Vx(:,i+1);
    Vxx(:,:,i) = (Qxx + Ku(:, i)*Quu*Ku(:, i)' + Kv(:, i)*Qvv*Kv(:, i)' + 2*Ku(:, i)*Qux + 2*Kv(:, i)*Qvx + 2*Ku(:, i)*Quv*Kv(:, i)')*dt + Vxx(:,:,i+1);
    
    Vxx(:,:,i) = Vxx(:,:,i)/2 + Vxx(:,:,i)'/2; % for symmetry
end

figure()
subplot(2,1,1); plot(Inv_u)
subplot(2,1,2); plot(Inv_v)

%% 
%----------------------------------------------> Update the controls
dx = [0; 0; 0; 0];

for i = 1:(Horizon-1)
    
    Fx = zeros(4);
    Fu = [-vp*sin(u(i)); vp*cos(u(i)); 0; 0];
    Fv = [0; 0; -ve*sin(v(i)); ve*cos(v(i))];
    
    du(i) = lu(i) + Ku(:, i)' * dx;
    dv(i) = lv(i) + Kv(:, i)' * dx;
    dx = Fx*dx + Fu*du(i) + Fv*dv(i);
    
    u_new(i) = u(i) + gamma*du(i);
    v_new(i) = v(i) + gamma*dv(i);
end

u = u_new;
v = v_new;


 
end

   time(1)=0;
   for i= 2:Horizon
    time(i) =time(i-1) + dt;  
   end

      
%% ---------------------------------------------> Plot Section

   figure()
   subplot(2,2,1)
   hold on
   plot(xo(3), xo(4), '.b', 'MarkerSize', 30)
   plot(x_traj(3,:),x_traj(4,:),'b','linewidth',4);  
   plot(x_traj(3,end),x_traj(4,end), '.b', 'MarkerSize', 30)
   plot(xo(1), xo(2), '.r', 'MarkerSize', 30)
   plot(x_traj(1,:),x_traj(2,:),'--r','linewidth',4); 
   plot(x_traj(1,end),x_traj(2,end), '.r', 'MarkerSize', 30)
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
   title('Pursuer Control', 'fontsize',20)
%    axis equal
%    axis([0,2,-1,2.5])
   hold off;
   
   subplot(2,2,3);hold on
   plot(tt_u, v(1, :), 'linewidth',2);
   xlabel('Time in sec', 'fontsize',20)
   title('Evader Control', 'fontsize',20)
%    axis equal
%    axis([0,2,-1,2.5])
   hold off;
   
   
   subplot(2,2,4);hold on
   plot(Cost,'linewidth',2); 
   xlabel('Iterations','fontsize',20)
   title('Cost','fontsize',20)
%    title(['Cost,', 'Ru =', num2str(Ru), ',Rv =', num2str(Rv)],'fontsize',20);
   hold off;
%    plot(Cost, '--rs',...
%     'LineWidth',4,...
%     'MarkerSize',10,...
%     'MarkerEdgeColor','g',...
%     'MarkerFaceColor', 'y'); 
%    xlabel('Iterations','fontsize',20)
%    title('Cost','fontsize',20);
   
   
%     figure()
%    hold on
%    plot(xo(3), xo(4), '.b', 'MarkerSize', 30)
%    plot(x_traj(3,:),x_traj(4,:),'b','linewidth',4);  
%    plot(x_traj(3,end),x_traj(4,end), '.b', 'MarkerSize', 30)
%    plot(xo(1), xo(2), '.r', 'MarkerSize', 30)
%    plot(x_traj(1,:),x_traj(2,:),'--r','linewidth',4); 
%    plot(x_traj(1,end),x_traj(2,end), '.r', 'MarkerSize', 30)
%    title('Trajectories','fontsize',20); 
%    xlabel('x','fontsize',20)
%    ylabel('y','fontsize',20)
%    hold off;
%    axis equal
% %    axis([0, 4, 0, 4])
%    grid;
%    
%    
%    
%     figure()
%     hold on
%    plot(tt_u, u(1, :), 'linewidth',2);
%    xlabel('Time in sec', 'fontsize',20)
%    title('Pursuer Control', 'fontsize',20)
% %    axis equal
% %    axis([0,2,-1,2.5])
%    hold off;
%    
%    
%     figure()
%     hold on
%    plot(tt_u, v(1, :), 'linewidth',2);
%    xlabel('Time in sec', 'fontsize',20)
%    title('Evader Control', 'fontsize',20)
% %    axis equal
% %    axis([0,2,-1,2.5])
%    hold off;
%    
%    
% %    subplot(2,2,4);hold on
%     figure()
%    plot(Cost,'linewidth',2); 
%    xlabel('Iterations','fontsize',20)
%    title('Cost','fontsize',20)
% %    title(['Cost,', 'Ru =', num2str(Ru), ',Rv =', num2str(Rv)],'fontsize',20);
%    hold off;
%    
%       figure()
%    hold on
%    plot(Cost, '--rs',...
%     'LineWidth',4,...
%     'MarkerSize',10,...
%     'MarkerEdgeColor','g',...
%     'MarkerFaceColor', 'y'); 
%    xlabel('Iterations','fontsize',20)
%    title('Cost','fontsize',20);
   
%    suptitle(['MinimaxDDP-Pursuit-Evasion, R_P = ',  num2str(Ru), ', R_E = ', num2str(Rv), ', v_P = ',  num2str(vp), ', v_E = ', num2str(ve), ', T_f = ', num2str(Tf)])
   
% %   Case: kappa = 1
% if kappa == 1
%    figure(50)
%    hold on
%    plot(Cost, 'g', 'linewidth',4); 
%    xlabel('Iterations','fontsize',20)
%    title('Cost','fontsize',20);
%    
%    figure(51)
%    hold on
%    plot(time(1:end-1), 0*time(1:end-1), 'b', 'linewidth', 4)
%    plot(time(1:end-1), u(1, :), 'g', 'linewidth',4);
%    xlabel('Time in sec', 'fontsize',20)
%    title('Control1', 'fontsize',20)
% end
%    
% % Case: kappa = 0
% if kappa == 0
%    figure(50)
%    hold on
%    plot(Cost, '-.r', 'linewidth',3); 
%    xlabel('Iterations','fontsize',20)
%    title('Cost','fontsize',20);
%    
%    figure(51)
%    hold on
%    plot(time(1:end-1), u(1, :), '-.r', 'linewidth',3);
%    xlabel('Time in sec', 'fontsize',20)
%    title('Control1', 'fontsize',20)
% end
   
    %% Apply control on stochastic dynamics
% sigma = 0.5;
% x_star(:, 1)= zeros(2,1);
% x_star(1, 1)= pi;
% iterations = 10;
% 
% for j = 1:iterations
% for k=1:(Horizon-1)      
%     F_x(1,1) = x_star(2,k);
%       F_x(2,1) = m1 * grav * l1 * sin(x_star(1,k))/I1 - b1*x_star(2,k)/I1;
%         
%       G_x(1,1) = 0;
%       G_x(2,1) = 1/I1;
% u_new(:,k) = u(:,k)+Lt(:,k)'*(x_star(:,k) - x_traj(k,:)');
% 
% x_star(:,k+1) = x_star(:,k) + F_x * dt + G_x * u_new(:,k) * dt  + sigma * randn(2, 1) * sqrt(dt) ;
% end
% 
%    figure(num_iter+2)
%    subplot(2,2,1)
%    hold on
%    plot(time,x_star(1,:),'linewidth',1);  
%    plot(time,p_target(1,1)*ones(1,Horizon),'red','linewidth',2)
%    title('Theta','fontsize',20); 
%    xlabel('Time in sec','fontsize',20)
%    hold off;
%    grid;
%    
%    
%    subplot(2,2,2);hold on;
%    plot(time,x_star(2,:),'linewidth',1); 
%    plot(time,p_target(2,1)*ones(1,Horizon),'red','linewidth',2)
%    title('Theta dot','fontsize',20);
%    xlabel('Time in sec','fontsize',20)
%    hold off;
%    grid;
% end