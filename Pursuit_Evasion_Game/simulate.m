function [x] = simulate(x,u,v,Horizon,dt)
global vp ve

 for k = 1:(Horizon-1)
     xprime = [ vp*cos(u(:,k)); ...
         vp*sin(u(:,k)); ...
         ve*cos(v(:,k)); ...
         ve*sin(v(:,k))];
    x(:,k+1) = x(:,k) + xprime * dt;
 end