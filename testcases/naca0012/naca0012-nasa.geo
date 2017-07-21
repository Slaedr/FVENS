// params
meshSizeWing = 0.1;
meshSizeFF = 10*meshSizeWing;
radiusFF = 20;
nsplinepoints = 40;
// ---

// Shape of airfoil
Macro topsurface
y = 0.594689181*(0.298222773*Sqrt(x) - 0.127125232*x - 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4);
Return
Macro botsurface
y = -0.594689181*(0.298222773*Sqrt(x) - 0.127125232*x - 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4);
Return

// Airfoil points and splines

splinePoints[0] = 0;
Point(0) = {1,0,0,meshSizeWing};

// assume range is inclusive
For i In {1:nsplinepoints-1}
	x = 1.0 - i/(nsplinepoints-1.0);
	Call botsurface;
	splinePoints[i] = newp;
	Point(splinePoints[i]) = {x,y,0,meshSizeWing}; 
EndFor

For i In {1:nsplinepoints-1}
	x = i/(nsplinepoints-1.0);
	Call topsurface;
	splinePoints[nsplinepoints-1+i] = newp;
	Point(splinePoints[nsplinepoints-1+i]) = {x,y,0,meshSizeWing};
EndFor

//splinePoints[2*nsplinepoints-2] = 0;

Spline(1) = splinePoints[];
//Line(6) = {2*nsplinepoints-2,0};
Coherence;

Point(1000) = {0,0,0,radiusFF};
Point(1001) = {radiusFF,0,0,meshSizeFF};
Point(1002) = {0,radiusFF,0,meshSizeFF};
Point(1003) = {-radiusFF,0,0,meshSizeFF};
Point(1004) = {0,-radiusFF,0,meshSizeFF};

Circle(2) = {1001,1000,1002};
Circle(3) = {1002,1000,1003};
Circle(4) = {1003,1000,1004};
Circle(5) = {1004,1000,1001};

Line Loop(6) = {2,3,4,5};
Line Loop(7) = {1};
Plane Surface(10) = {6,7};
Physical Surface(1) = {10};
Physical Line(2) = {1};
Physical Line(4) = {2,3,4,5};


