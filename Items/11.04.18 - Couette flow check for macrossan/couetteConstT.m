function [T u] = couetteConstT(T0,mu,U,k,Kn,Pr,gam,y)
% calculate the thermal couette flow profile 



uWR = -U;

uWL = U;

A = (uWR - uWL)/(1+2*Kn);

B = (uWR + uWL)/2;

nu = mu;

C = (-nu/(2*k))*A^2;

thetaWR = T0;

thetaWL = T0;

h = ((2*gam)/(gam+1))*(1/Pr);

D = (thetaWR - thetaWL)/(1 + 2*h*Kn);

E = (thetaWR - thetaWL)/2 + (nu/(8*k))*A^2*(1 + 4*h*Kn);

u = A*y + B;

T = T0 + C*y.^2 + D*y + E;