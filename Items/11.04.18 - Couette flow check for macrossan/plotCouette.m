%plot Couette
clear variables

N = 100;  %number of cells across gap

Kn = 0.1;      %knudsen number
Pr = 0.71;      %Prandtl number
R = 287;        % gas constant
gam = 5/3;      % ratio of specific heats
mu = 1.86e-5;   % viscosity

% flow values
rho0 = 1.165;   % density
T0 = 303;       % temperature
p0 = rho0*R*T0; % pressure

U = 2;                  % multiplier of reference velocity, used for wall velocity
u0 = sqrt(gam*R*T0);    % reference velocity
uwall = U*u0/2;         % wall velocity - divided by two as method uses two walls moving in opposite directions

cp = (gam*R)/(gam - 1);
k = mu*cp/Pr;           

lambda = (2*mu)/(rho0*sqrt((8*R*T0)/pi));   % mean free path

L = lambda/Kn;   %wall separation

y = (-0.5:1/(N-1):0.5);

[TC uC] = couetteConstT(T0,mu,uwall,k,Kn,Pr,gam,y);

%% PLOTTING
figure(1)
clf

subplot(1,2,1)
plot(TC/T0,y,'r');
grid on
axis([1 1.005*max(TC/T0) -0.5 0.5])
xlabel('T/T_0')
ylabel('y/L_0')
title('Temperature Profile')

subplot(1,2,2)
plot(-(uC-uwall)/(2*uwall),y,'r');
grid on
axis([0 1 -0.5 0.5])
xlabel('u_x/u_0')
ylabel('y/L_0')
title('Velocity Profile')



