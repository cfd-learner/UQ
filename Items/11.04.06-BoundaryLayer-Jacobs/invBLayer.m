% investigate boundary layer profile

clear variables

load('input.mat');

for i = 100
    load(['step ',num2str(i),'.mat']);

u_ref = sqrt(gamma*R*T_ref);

x = 0:dx:Lx;
y = 0:dy:Ly;

[X,Y] = meshgrid(x,y);

figure(1)
clf
subplot(2,2,1)
mesh(X,Y,rho/rho_ref);
title('density')
grid on
axis equal
view(2)

subplot(2,2,2)
mesh(X,Y,sqrt(ux.^2 +uy.^2)/u_ref);
title('velocity')
grid on
axis equal
view(2)

subplot(2,2,3)
mesh(X,Y,p/p_ref);
title('pressure')
grid on
axis equal
view(2)

subplot(2,2,4)
mesh(X,Y,T/T_ref);
title('temperature')
grid on
axis equal
view(2)

%% data pick temp & velocity profile

Tpick = [168,584, 167,30, 1234,585, 169,399, 185,365, 202,325, 220,288, 237,258, 260,225, 288,189, 316,163, 350,142, 393,129, 421,128, 458,136, 490,151, 521,168, 547,184, 577,208, 607,231, 631,248, 657,270, 686,292, 725,319, 761,339, 793,355, 829,369, 877,383, 920,393, 975,401, 1023,405, 1064,406, 1112,408, 1148,409, 1190,409];

[Tx, Ty] = manualCurve(Tpick, 200, 270, 0, 0.006);

Upick = [160,567, 160,41, 1226,572, 162,566, 173,555, 184,544, 194,532, 210,514, 227,497, 248,477, 268,458, 285,440, 304,423, 327,400, 356,372, 379,353, 402,333, 424,313, 445,296, 471,276, 498,256, 526,237, 561,215, 599,194, 642,173, 690,156, 724,147, 764,138, 809,131, 855,128, 899,124, 953,122, 985,121, 1030,122, 1076,121, 1121,122, 1168,121, 1181,121];

[Ux Uy] = manualCurve(Upick, 0, 700, 0, 0.006);

%% get corresponding data from simulation

x = 0.9415;

n = ceil(x/dx);

simUx = ux(:,n);
simT = T(:,n);

%% plot on common axis
ySim = (length(simT)-1)*dy;
ySim = 0:dy:ySim;
ind = find(ySim>max(Ux),1,'first');

ySim = ySim(1:ind);

figure(2)
clf
subplot(1,2,1)
plot(Ux,Uy,'r') %from figure
hold on
plot(ySim,simUx(1:ind),'o');
xlabel('x (m)');
ylabel('U_x (m/s)')
title('Velocity of Boundary Layer')
legend('Jacobs','simLB')

subplot(1,2,2)
plot(Tx,Ty,'r') %from figure
hold on
plot(ySim,simT(1:ind),'o');
xlabel('x (m)');
ylabel('T (K)')
title('Temperature of Boundary Layer')
legend('Jacobs','simLB')
end
