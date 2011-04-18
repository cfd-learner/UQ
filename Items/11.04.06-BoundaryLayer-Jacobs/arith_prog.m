%arithmetic progression

Lx = 1;
Ly = 0.68;

dx1 = 0.05;
dy1 = 0.05;

Nx = 50;
Ny = 100;

dx = (2*(Lx-Nx*dx1))/(Nx*(Nx-1));
dy = (2*(Ly-Ny*dy1))/(Ny*(Ny-1));

DX = zeros(Nx,1);
DY = zeros(Ny,1);

for x = 1:Nx
    DX(x) = dx1 + (x - 1)*dx;
end

for y = 1:Ny
    DY(y) = dy1 + (y - 1)*dy;
end

X = zeros(Nx+1,1);
Y = zeros(Ny+1,1);

X(1) = DX(1)/2;
Y(1) = DY(1)/2;

for x = 2:Nx
    X(x) = X(x-1) + DX(x);
end

for y = 2:Ny
    Y(y) = Y(y-1) + DY(y);
end

%check

sum(DX)
sum(DY)

[X, Y, Z] = meshgrid(X,Y,1);

mesh(X,Y,Z)

