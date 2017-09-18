// params
refine = 2;
// ---

radiusFF = 20;

meshSizeWing = 0.2/refine;
meshSizeLead = meshSizeWing/10.0;
meshSizeTrail = meshSizeWing/5.0;
meshSizeFF = radiusFF*meshSizeWing;

nTangPoin = 20*refine;
nNormPoin = 20*refine;
tangProg = 1.2;
normProg = 1.3;

nsplinepoints = 20*refine;

// Shape of airfoil
Macro topsurface
y = 0.594689181*(0.298222773*Sqrt(x) - 0.127125232*x - 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4);
Return
Macro botsurface
y = -0.594689181*(0.298222773*Sqrt(x) - 0.127125232*x - 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4);
Return

// Airfoil points and splines

tsplinePoints[0] = 0;
Point(0) = {1,0,0,meshSizeTrail};

// assume range is inclusive
For i In {1:nsplinepoints-2}
	x = (1.0 - i/(nsplinepoints-1.0))^2;
	Call botsurface;
	tsplinePoints[i] = newp;
	If(nsplinepoints-2-i < 2*refine)
		Point(tsplinePoints[i]) = {x,y,0,meshSizeLead}; 
	ElseIf(i < 2*refine)
		Point(tsplinePoints[i]) = {x,y,0,meshSizeTrail};
	Else
		Point(tsplinePoints[i]) = {x,y,0,meshSizeWing}; 
	EndIf
EndFor
tsplinePoints[nsplinepoints-1] = newp;
Point(tsplinePoints[nsplinepoints-1]) = {0,0,0,meshSizeLead};
origin = tsplinePoints[nsplinepoints-1];

bsplinePoints[0] = tsplinePoints[nsplinepoints-1];
For i In {1:nsplinepoints-2}
	x = (i/(nsplinepoints-1.0))^2;
	Call topsurface;
	bsplinePoints[i] = newp;
	If(i < 2*refine)
		Point(bsplinePoints[i]) = {x,y,0,meshSizeLead};
	ElseIf(nsplinepoints-2-i < 2*refine)
		Point(bsplinePoints[i]) = {x,y,0,meshSizeTrail};
	Else
		Point(bsplinePoints[i]) = {x,y,0,meshSizeWing};
	EndIf
EndFor

bsplinePoints[nsplinepoints-1] = 0;

Spline(1) = tsplinePoints[]; Transfinite Line{1} = nTangPoin Using Bump tangProg;
Spline(2) = bsplinePoints[]; Transfinite Line{2} = nTangPoin Using Bump tangProg;

// Ellipse for structured portion
Point(100) = {0.5, 1.5, 0.0, meshSizeWing};
Point(101) = {0.5, -1.5,0.0, meshSizeWing};
Point(102) = {0.5, 0.0, 0.0, meshSizeWing};
Point(103) = {-0.1,0.0, 0.0, meshSizeWing};
Point(104) = {1.1, 0,0, 0.0, meshSizeWing};

Point(10000) = {0,0,0,radiusFF};
Point(10001) = {radiusFF,0,0,meshSizeFF/1.5};
Point(10002) = {0,radiusFF,0,meshSizeFF};
Point(10003) = {-radiusFF,0,0,meshSizeFF};
Point(10004) = {0,-radiusFF,0,meshSizeFF};

Circle(5) = {10001,10000,10002};
Circle(6) = {10002,10000,10003};
Circle(7) = {10003,10000,10004};
Circle(8) = {10004,10000,10001};
Line Loop(6) = {5,6,7,8};

//Ellipse(10) = {103, 102, 103, 101};
//Ellipse(11) = {101, 102, 104, 104};
//Ellipse(12) = {104, 102, 104, 100};
//Ellipse(13) = {100, 102, 103, 103};
Ellipse(10) = {103, 102, 103, 104}; Transfinite Line{10} = nTangPoin Using Bump tangProg;
Ellipse(11) = {104, 102, 103, 103}; Transfinite Line{11} = nTangPoin Using Bump tangProg;
Line(12) = {0, 104}; Transfinite Line{12} = nNormPoin Using Prog normProg;
Line(13) = {origin, 103}; Transfinite Line{12} = nNormPoin Using Prog normProg;

Line Loop(7) = {10,11};
Line Loop(8) = {10, -12, 2, 13};
Line Loop(9) = {-13, 1, 12, 11};

Plane Surface(10) = {6,7};
Plane Surface(11) = {8}; Transfinite Surface{11};
Plane Surface(12) = {9}; Transfinite Surface{12};
Physical Surface(1) = {10,11,12};

// Adiabatic wall
Physical Line(6) = {1,2};
// Farfield
Physical Line(4) = {5,6,7,8};

//Mesh.SubdivisionAlgorithm=1;	// ensures all quads, I think

//Mesh.Algorithm=6;		// Frontal

Color Black{ Surface{10}; }
