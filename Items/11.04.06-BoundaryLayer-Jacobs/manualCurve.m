function [dataX dataY] = manualCurve(data, originY, axisY, originX, axisX, plt)
% import set of coordinates relating to a curve, output scaled data
% relative to origin at (0,0)
% format:
% 1:    origin
% 2:    data point on Y axis
% 3:    data point on x axis
% 4->inf:   data points along curve
% scale points:
% originY: value of Y at origin point
% axisY:    value of Y at point picked on Y axis
% same for X
% plt - switch for plotting of graph

if nargin == 5
    plt = 0;
end


N = length(data);
dataX = data(1:2:N);
dataY = data(2:2:N);

%flip Y
dataY = -dataY - max(dataY);

origin = [dataX(1) dataY(1)];

%axis
Y = [dataX(2) dataY(2)] - origin;
X = [dataX(3) dataY(3)] - origin;

dataX = dataX(4:end) - origin(1);
dataY = dataY(4:end) - origin(2);

%scale
Xscale = (axisX - originX)/X(1);
Yscale = (axisY - originY)/Y(2);


dataX = dataX*Xscale + originX;
dataY = dataY*Yscale + originY;

origin = [originX originY];

X = [X(1)*Xscale+originX X(2)*Yscale+originY];
Y = [Y(1)*Xscale+originX Y(2)*Yscale+originY];

if plt == 1
    figure
    plot([origin(1) Y(1)],[origin(2) Y(2)]);
    hold on
    plot([origin(1) X(1)],[origin(2) X(2)]);
    plot(dataX,dataY);
end