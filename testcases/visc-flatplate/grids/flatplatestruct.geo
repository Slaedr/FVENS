// flow past cylinder

// (1D) Refine factor
ref = 8;

// end of plate
pend = 1.0;
// top point of domain
top = 1.0;
// dummies
hc = 0.1;
hf = 0.2;

// number of x-points before start of plate
nxi = 5*ref;
// number of x-points on the plate
nxp = 10*ref;
// number of y-points
ny = 10*ref;

// progression ratios
progtp = 1.1;		// tangential for plate
progti = 1.2;		// tangential near inlet before plate
progn = 1.2;		// normal

// start of the plate
Point(1) = { 0,   0,  0, hc};
// other points
Point(2) = { -0.5,   0,  0, hc};
Point(3) = { pend,   0,  0, hc};
Point(4) = { pend, top,  0, hc};
Point(5) = {   0,  top,  0, hc};
Point(6) = { -0.5, top,  0, hc};

Line(1) = {1,2}; Transfinite Line(1) = nxi Using Progression progti;
Line(2) = {1,3}; Transfinite Line(2) = nxp Using Progression progtp;
Line(3) = {3,4}; Transfinite Line(3) = ny Using Progression progn;
Line(4) = {5,4}; Transfinite Line(4) = nxp Using Progression progtp;
Line(5) = {5,6}; Transfinite Line(5) = nxi Using Progression progti;
Line(6) = {2,6}; Transfinite Line(6) = ny Using Progression progn;
Line(7) = {1,5}; Transfinite Line(7) = ny Using Progression progn;

Line Loop(1) = {-1,7,5,-6};
Line Loop(2) = {2,3,-4,-7};

Plane Surface(1) = {1}; Transfinite Surface {1}; Recombine Surface {1};
Plane Surface(2) = {2}; Transfinite Surface {2}; Recombine Surface {2};

//physical markers

//plate
Physical Line(2) = {2};
// Inlet bottom portion
Physical Line(3) = {1};
// Farfield
Physical Line(4) = {6,5,4};
// outlet
Physical Line(5) = {3};

Physical Surface(1) = {1,2};
