// flow through Gaussian channel

// (1D) Refine factor
//ref = 1;

// Half the length of channel
xend = 1.0;
// Approx height of channel at inlet
height = 0.75;

// Number of spline points
nspline = 200;

// mesh sizes
hc = 0.1/ref;
h = 0.2/ref;

// Lower surface function
// Bump height
a = 0.02;
b = 100.0;
Macro gaussian
y = a*Exp(-b*x^2);
Return

// start of Gaussian bump
x = -xend;
Call gaussian;
Point(0) = { x,   y,  0, hc};

splinepoints[0] = 0;

For i In {1:nspline}
	x = -xend + i*2.0*xend/nspline;
	Call gaussian;
	Point(i) = {x,y,0,hc};
	splinepoints[i] = i;
EndFor

// other points
Point(nspline+1) = { xend,   height,  0, h};
Point(nspline+2) = { -xend,   height,  0, h};

Spline(1) = splinepoints[];
Line(2) = {nspline,nspline+1};
Line(3) = {nspline+1,nspline+2};
Line(4) = {nspline+2,0};

Line Loop(1) = {1,2,3,4};

Plane Surface(1) = {1};

//physical markers

// Walls
Physical Line(2) = {1,3};
// Inlet
Physical Line(3) = {4};
// Outlet
Physical Line(4) = {2};

Physical Surface(1) = {1};

Color Black{ Surface{1}; }
