function [Cost] =  fnCostComp_minimax(x_traj,u_new,v_new,p_target,dt)

global Q_f;
global Ru;
global Rv;

global vp ve;

 [Horizon, numOfStates] = size(x_traj);
 Cost = 0;
 
 for j =1:(Horizon-1)
     
    Cost = Cost + (x_traj(j,1)-x_traj(j,4))*(vp*cos(x_traj(j,3)) - ve*cos(v_new(:,j)))*dt + (x_traj(j,2)-x_traj(j,5))*(vp*sin(x_traj(j,3)) - ve*sin(v_new(:,j)))*dt + 0.5*Ru*(u_new(:,j))^2*dt;
     
 end
 % 
 % TerminalCost= 0;
 % 
 % Cost = 0;
TerminalCost = 0.5*((x_traj(end,1)-x_traj(end,4))^2 + (x_traj(end,2)-x_traj(end,5))^2);
 Cost = Cost + TerminalCost;