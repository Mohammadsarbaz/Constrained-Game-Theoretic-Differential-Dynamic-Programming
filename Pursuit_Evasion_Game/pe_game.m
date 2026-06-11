function xprime = pe_game(t, x, flag, u, v)

global vp ve;
global obs_center R_safe;
xp = x(1);
yp = x(2);
theta_p = x(3);

dx_obs = xp - obs_center(1);
dy_obs = yp - obs_center(2);
dist = sqrt(dx_obs^2 + dy_obs^2);

% if dist < R_safe
violation = 100*max(0,R_safe- dist+ abs(R_safe - dist));
% violation = 300*(R_safe - dist + abs(R_safe - dist));

% else
%     violation = 0;
% end

xprime = [ vp*cos(theta_p); ...
           vp*sin(theta_p); ...
           ppval(u,t); ...
           ve*cos(ppval(v,t)); ...
           ve*sin(ppval(v,t));...
           violation^2];