// params
refine = 2;
// ---

radiusFF = 20;

meshSizeWing = 0.2/refine;
meshSizeLead = meshSizeWing/5.0;
meshSizeTrail = meshSizeWing/5.0;
meshSizeFF = radiusFF*meshSizeWing;

nTangPoin = 30*refine;
nNormPoin = 20*refine;
tangProg = 0.1;
normProg = 1.1;

nsplinepoints = 40*refine;

// Shape of airfoil
Macro topsurface
y = 0.594689181*(0.298222773*Sqrt(x) - 0.127125232*x - 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4);
Return
Macro botsurface
y = -0.594689181*(0.298222773*Sqrt(x) - 0.127125232*x - 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4);
Return

// Airfoil points and splines

bsplinePoints[0] = 0;
Point(0) = {1,0,0,meshSizeTrail};

// assume range is inclusive
For i In {1:nsplinepoints-2}
	x = (1.0 - i/(nsplinepoints-1.0))^2;
	Call botsurface;
	bsplinePoints[i] = newp;
	If(nsplinepoints-2-i < 2*refine)
		Point(bsplinePoints[i]) = {x,y,0,meshSizeLead}; 
	ElseIf(i < 2*refine)
		Point(bsplinePoints[i]) = {x,y,0,meshSizeTrail};
	Else
		Point(bsplinePoints[i]) = {x,y,0,meshSizeWing};
	EndIf
EndFor
bsplinePoints[nsplinepoints-1] = newp;
Point(bsplinePoints[nsplinepoints-1]) = {0,0,0,meshSizeLead};
origin = bsplinePoints[nsplinepoints-1];

tsplinePoints[0] = bsplinePoints[nsplinepoints-1];
For i In {1:nsplinepoints-2}
	x = (i/(nsplinepoints-1.0))^2;
	Call topsurface;
	tsplinePoints[i] = newp;
	If(i < 2*refine)
		Point(tsplinePoints[i]) = {x,y,0,meshSizeLead};
	ElseIf(nsplinepoints-2-i < 2*refine)
		Point(tsplinePoints[i]) = {x,y,0,meshSizeTrail};
	Else
		Point(tsplinePoints[i]) = {x,y,0,meshSizeWing};
	EndIf
EndFor

tsplinePoints[nsplinepoints-1] = 0;

Spline(1) = bsplinePoints[]; Transfinite Line{1} = nTangPoin Using Bump tangProg;
Spline(2) = tsplinePoints[]; Transfinite Line{2} = nTangPoin Using Bump tangProg;

// Points for structured portion
strht2 = 0.025;
blthickness = 0.02;
teangle = 3.14/12;
Point(1000) = {0.6, 3*blthickness,  0.0, meshSizeWing};
Point(1010) = {0.6, -3*blthickness, 0.0, meshSizeWing};
Point(1011) = {0.263, 0.07, 0.0, meshSizeWing};
Point(1012) = {0.263, -0.07, 0.0, meshSizeWing};
Point(1013) = {0.095, 0.055, 0.0, meshSizeWing};
Point(1014) = {0.095, -0.055, 0.0, meshSizeWing};

Point(1020) = {0.5, 0.0, 0.0, meshSizeWing};

Point(1030) = {-0.3*blthickness,0.0, 0.0, meshSizeLead};
Point(1040) = {1+Sin(teangle)*blthickness, Cos(teangle)*blthickness, 0.0, meshSizeTrail};
Point(1041) = {1+Sin(teangle)*blthickness, -Cos(teangle)*blthickness, 0.0, meshSizeTrail};
Point(1050) = {0.01, strht2, 0.0, meshSizeWing};
Point(1060) = {0.01,-strht2, 0.0, meshSizeWing};
//Point(1070) = {-0.1,0.06,0,0,meshSizeLead};
//Point(1080) = {-0.1,-0.06,0,0,meshSizeLead};

//Farfield
Point(10000) = {0,0,0,radiusFF};
Point(10001) = {radiusFF,0,0,meshSizeFF/1.5};
Point(10002) = {0,radiusFF,0,meshSizeFF};
Point(10003) = {-radiusFF,0,0,meshSizeFF};
Point(10004) = {0,-radiusFF,0,meshSizeFF};

Circle(5) = {10001,10000,10002};
Circle(6) = {10002,10000,10003};
Circle(7) = {10003,10000,10004};
Circle(8) = {10004,10000,10001};
//Line(9) = {10001, 1040};
//Line(10) = {1030, 10003};

//topbord[0] = 104; topbord[1] = 100; topbord[2] = 105; topbord[3] = 107; topbord[4] = 103;
topbord[] = {1040, 1000, 1011, 1013, 1050, 1030};
botbord[] = {1030, 1060, 1014, 1012, 1010, 1041};
Spline(20) = topbord[]; Transfinite Line{20} = nTangPoin Using Bump tangProg;
Spline(21) = botbord[]; Transfinite Line{21} = nTangPoin Using Bump tangProg;
Line(22) = {0, 1040}; Transfinite Line{22} = nNormPoin Using Progression normProg;
Line(24) = {0,1041}; Transfinite Line{24} = nNormPoin Using Progression normProg;
Line(23) = {origin, 1030}; Transfinite Line{23} = nNormPoin Using Progression normProg;

//Line Loop(6) = {5,6,-10,-20,-9};
//Line Loop(7) = {7,8,9,-21,10};
Line Loop(6) = {5, 6, 7, 8};
Line Loop(7) = {1,2};

Line Loop(8) = {20, -23, 2, 22};
Line Loop(9) = {-23, -1, 24, -21};

//Plane Surface(10) = {6};
//Plane Surface(11) = {7};
Plane Surface(10) = {6,8,9};
Plane Surface(21) = {8}; Transfinite Surface{21}; Recombine Surface{21};
Plane Surface(22) = {9}; Transfinite Surface{22}; Recombine Surface{22};
//Physical Surface(1) = {10,11,21,22};
Physical Surface(1) = {10,21,22};

// Adiabatic wall
Physical Line(2) = {1,2};
// Farfield
Physical Line(4) = {5,6,7,8};

//Mesh.SubdivisionAlgorithm=1;	// ensures all quads, I think

Mesh.Algorithm=6;		// Frontal

Color Black{ Surface{10,11,21,22}; }
