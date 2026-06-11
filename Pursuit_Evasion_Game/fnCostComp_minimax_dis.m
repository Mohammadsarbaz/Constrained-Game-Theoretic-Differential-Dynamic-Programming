function [Cost] =  fnCostComp_minimax_dis(x_traj,u_new,v_new,dt)

global vp ve;

 [Horizon, numOfStates] = size(x_traj);
 Cost = 0;
 
 for j =1:(Horizon-1)
     
    Cost = Cost + (x_traj(1,j) - x_traj(3,j))*(vp*cos(u_new(:,j)) - ve*cos(v_new(:,j)))*dt + (x_traj(2,j) - x_traj(4,j))*(vp*sin(u_new(:,j)) - ve*sin(v_new(:,j)))*dt;
     
 end
 
 TerminalCost= 0;
 
 Cost = 0;
 TerminalCost = 0.5*((x_traj(1,end) - x_traj(3,end))^2 + (x_traj(2,end) - x_traj(4,end))^2);
 
 Cost = Cost + TerminalCost;