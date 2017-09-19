// params
refine = 2;
// ---

radiusFF = 20;

meshSizeWing = 0.2/refine;
meshSizeLead = meshSizeWing/5.0;
meshSizeTrail = meshSizeWing/5.0;
meshSizeFF = radiusFF*meshSizeWing;

nTangPoin = 20*refine;
nNormPoin = 20*refine;
tangProg = 0.1;
normProg = 1.1;

nsplinepoints = 20*refine;

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
Point(100) = {0.5, 0.0, 0.0, meshSizeWing};
Point(101) = {1.1, 0.0, 0.0, meshSizeTrail};
Point(102) = {0.5, 0.6, 0.0, meshSizeWing};
Point(103) = {-0.1,0.0, 0.0, meshSizeLead};
Point(104) = {0.5,-0.6, 0.0, meshSizeWing};

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
Line(9) = {10001, 101};
Line(10) = {103, 10003};

/*Circle(18) = {101,100,102}; Transfinite Line{18} = nTangPoin Using Progression tangProg;
Circle(19) = {103,100,102}; Transfinite Line{19} = nTangPoin Using Progression tangProg;
Circle(20) = {103,100,104}; Transfinite Line{20} = nTangPoin Using Progression tangProg;
Circle(21) = {101,100,104}; Transfinite Line{21} = nTangPoin Using Progression tangProg;
Line(22) = {0, 101}; Transfinite Line{22} = nNormPoin Using Progression normProg;
Line(23) = {origin, 103}; Transfinite Line{23} = nNormPoin Using Progression normProg;

Line Loop(6) = {5,6,-10,19,-18,-9};
Line Loop(7) = {7,8,9,21,-20,10};

Line Loop(8) = {20, -21, -22, 1, 23};
Line Loop(9) = {-23, 2, 22, 18, -19};*/

Circle(20) = {101,100,103}; Transfinite Line{20} = nTangPoin Using Bump tangProg;
Circle(21) = {103,100,101}; Transfinite Line{21} = nTangPoin Using Bump tangProg;
Line(22) = {0, 101}; Transfinite Line{22} = nNormPoin Using Progression normProg;
Line(23) = {origin, 103}; Transfinite Line{23} = nNormPoin Using Progression normProg;

Line Loop(6) = {5,6,-10,-20,-9};
Line Loop(7) = {7,8,9,-21,10};

Line Loop(8) = {21, -22, 1, 23};
Line Loop(9) = {-23, 2, 22, 20};


Plane Surface(10) = {6};
Plane Surface(11) = {7};
Plane Surface(21) = {8}; Transfinite Surface{21}; Recombine Surface{21};
Plane Surface(22) = {9}; Transfinite Surface{22}; Recombine Surface{22};
Physical Surface(1) = {10,11,21,22};

// Adiabatic wall
Physical Line(6) = {1,2};
// Farfield
Physical Line(4) = {5,6,7,8};

//Mesh.SubdivisionAlgorithm=1;	// ensures all quads, I think

//Mesh.Algorithm=6;		// Frontal

Color Black{ Surface{10,11,21,22}; }
