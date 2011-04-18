%power law

ax = 1;
kx = 3;

ay = 1;
ky = 1;


fx = @(x) ax*x^kx;
fy = @(y) ay*y^ky;

N = 100;

x = zeros(N,1);
y = zeros(N,1);

for i = 1:N
    x(i) = fx(i);
    y(i) = fy(i);
end

maxX = max(x);
minX = min(x);

maxY = max(y);
minY = min(y);

x = (x-minX)/(maxX-minX);
y = (y-minY)/(maxY-minY);

[X, Y, Z] = meshgrid(x,y,1);

mesh(X,Y,Z)