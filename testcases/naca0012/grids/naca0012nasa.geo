// params
refine = 8;
// ---

meshSizeWing = 0.2/refine;
meshSizeFF = 10*meshSizeWing;
nsplinepoints = 20*refine;

radiusFF = 20;

// Shape of airfoil
Macro topsurface
y = 0.594689181*(0.298222773*Sqrt(x) - 0.127125232*x - 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4);
Return
Macro botsurface
y = -0.594689181*(0.298222773*Sqrt(x) - 0.127125232*x - 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4);
Return

// Airfoil points and splines

tsplinePoints[0] = 0;
Point(0) = {1,0,0,meshSizeWing/2.0};

// assume range is inclusive
For i In {1:nsplinepoints-2}
	x = (1.0 - i/(nsplinepoints-1.0))^2;
	Call botsurface;
	tsplinePoints[i] = newp;
	Point(tsplinePoints[i]) = {x,y,0,meshSizeWing}; 
EndFor
tsplinePoints[nsplinepoints-1] = newp;
Point(tsplinePoints[nsplinepoints-1]) = {0,0,0,meshSizeWing/5.0}; 

bsplinePoints[0] = tsplinePoints[nsplinepoints-1];
For i In {1:nsplinepoints-2}
	x = (i/(nsplinepoints-1.0))^2;
	Call topsurface;
	bsplinePoints[i] = newp;
	Point(bsplinePoints[i]) = {x,y,0,meshSizeWing};
EndFor

bsplinePoints[nsplinepoints-1] = 0;

Spline(1) = tsplinePoints[];
Spline(2) = bsplinePoints[];

Point(10000) = {0,0,0,radiusFF};
Point(10001) = {radiusFF,0,0,meshSizeFF};
Point(10002) = {0,radiusFF,0,meshSizeFF};
Point(10003) = {-radiusFF,0,0,meshSizeFF};
Point(10004) = {0,-radiusFF,0,meshSizeFF};

Circle(5) = {10001,10000,10002};
Circle(6) = {10002,10000,10003};
Circle(7) = {10003,10000,10004};
Circle(8) = {10004,10000,10001};

Line Loop(6) = {5,6,7,8};
Line Loop(7) = {1,2};
Plane Surface(10) = {6,7};
Physical Surface(1) = {10};
Physical Line(2) = {1,2};
Physical Line(4) = {5,6,7,8};


