// Viscous flow through Gaussian channel

// Number of times to refine
nref = 0;

// Half the length of channel
xend = 8.0;
// Approx height of channel at inlet
height = 7;
// End of bump
xbumpend = 0.4;
// End of wall
xwall = 2.0;

// Number of spline points
nspline = 200;

// mesh sizes
hc = 0.2;
h = 0.75;
// Number of points for structured region
ntang = 25;
ntangfar = 18;   // far from bump
ntangwall = 14; // wall not Gaussian
nbl = 6;
// Progressions
rbl = 1.4;
rtang = 5.0;
rtangfar = 1.3;

// Lower surface function
// Bump height
a = 0.015;
b = 20.0;
blt = 0.05;

Macro gaussian
y = a*Exp(-b*x^2);
Return

Macro gaussianBL
y = a*Exp(-b*x^2) + blt;
Return

// start of Gaussian bump
//x = -xbumpend;
//Call gaussian;
//Point(0) = { x,   y,  0, hc};
Point(0) = { -xbumpend,   0,  0, hc};

splinepoints[0] = 0;

For i In {1:nspline-1}
	x = -xbumpend + i*2.0*xbumpend/nspline;
	Call gaussian;
	Point(i) = {x,y,0,hc};
	splinepoints[i] = i;
EndFor

Point(nspline) = {xbumpend, 0, 0, hc};
splinepoints[nspline] = nspline;

// start of Gaussian BL
//x = -xbumpend;
//all gaussianBL;
Point(nspline+1) = { -xbumpend,   blt,  0, hc};
splinepointsBL[0] = nspline+1;

For i In {1:nspline-1}
	x = -xbumpend + i*2.0*xbumpend/nspline;
	ipoin = i+nspline+1;
	Call gaussianBL;
	Point(ipoin) = {x,y,0,hc};
	splinepointsBL[i] = ipoin;
EndFor

Point(2*nspline+1) = {xbumpend, blt, 0, hc};
splinepointsBL[nspline] = 2*nspline+1;

// other points
Point(1000) = { xend,   height,  0, h};
Point(1001) = { -xend,   height,  0, h};
Point(1002) = { -xend, 0, 0, h};
Point(1003) = { xend, 0, 0, h};
Point(1004) = { -xend, blt, 0, h};
Point(1005) = { xend, blt, 0, h};
Point(1006) = { -xwall, 0, 0, h};
Point(1007) = { xwall, 0, 0, h};
Point(1008) = { -xwall, blt, 0, h};
Point(1009) = { xwall, blt, 0, h };

Spline(1) = splinepoints[]; Transfinite Line {1} = ntang;
Spline(2) = splinepointsBL[]; Transfinite Line {2} = ntang;
Line(3) = {0, nspline+1}; Transfinite Line {3} = nbl Using Progression rbl;
Line(4) = {nspline,2*nspline+1};  Transfinite Line {4} = nbl Using Progression rbl;
Line(5) = {1004, 1001};
Line(6) = {1005, 1000};
Line(7) = {1000, 1001};
Line(8) = {1006, 0}; Transfinite Line {8} = ntangwall Using Bump 1.0/rtang;
Line(9) = {nspline, 1007}; Transfinite Line{9} = ntangwall Using Bump 1.0/rtang;
Line(10) = {1002, 1004}; Transfinite Line {10} = nbl Using Progression rbl;
Line(11) = {1003, 1005};  Transfinite Line {11} = nbl Using Progression rbl;
Line(12) = {nspline+1, 1008}; Transfinite Line {12} = ntangwall Using Bump 1.0/rtang;
Line(13) = {2*nspline+1, 1009}; Transfinite Line {13} = ntangwall Using Bump 1.0/rtang;

Line(14) = {1002, 1006}; Transfinite Line {14} = ntangfar Using Progression 1.0/rtangfar;
Line(15) = {1008, 1004}; Transfinite Line {15} = ntangfar Using Progression rtangfar;
Line(16) = {1007, 1003}; Transfinite Line {16} = ntangfar Using Progression rtangfar;
Line(17) = {1009, 1005}; Transfinite Line {17} = ntangfar Using Progression rtangfar;

Line(18) = {1006, 1008}; Transfinite Line {18} = nbl Using Progression rbl;
Line(19) = {1007, 1009}; Transfinite Line {19} = nbl Using Progression rbl;

Line Loop(1) = {1,4,-2,-3};
Plane Surface(1) = {1}; Transfinite Surface {1}; Recombine Surface {1};

Line Loop(2) = {-18, 8, 3, 12};
Plane Surface(2) = {2}; Transfinite Surface {2}; Recombine Surface {2};

Line Loop(3) = {9, 19, -13, -4};
Plane Surface(3) = {3}; Transfinite Surface {3}; Recombine Surface {3};

Line Loop(4) = {14, 18, 15, -10};
Plane Surface(4) = {4}; Transfinite Surface {4}; Recombine Surface {4};

Line Loop(5) = {16, 11, -17, -19};
Plane Surface(5) = {5}; Transfinite Surface {5}; Recombine Surface {5};

Line Loop(6) = {2, 13, 17, 6, 7, -5, -15, -12};
Plane Surface(6) = {6};

//physical markers

// Solid Walls
Physical Line(2) = {8, 1, 9};
// Slip walls / symmetry planes
Physical Line(3) = {14, 16, 7};
// Inlet
Physical Line(4) = {10, 5};
// Outlet
Physical Line(5) = {11, 6};

Physical Surface(1) = {1, 2, 3, 4, 5, 6};

Mesh 2;

For iref In {1:nref}
	RefineMesh;
	OptimizeMesh "Laplace2D";
EndFor

Color Black{ Surface{6}; }
Color Brown{ Surface{1}; Surface{2}; Surface{3}; Surface{4}; Surface{5}; }
