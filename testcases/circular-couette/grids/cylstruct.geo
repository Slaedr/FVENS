// flow past cylinder

// (1D) Refine factor
ref = 4;

// actual boundary
srad = 0.5;
// far field
ffs = 1.0;
// dummies
hc = 0.1;
hf = 0.2;

// number of points tangential
nt = 5*ref;
// number of points radial
nr = 7*ref;
// progression ratio
r = 1.02;		// original
//r= 1.3;		// for refinement by splitting

//center
Point(1) = { 0,   0,  0, hc};

//Circle-1 inner cylinder
Point(2) = { srad,   0,  0, hc};
Point(3) = {   0, srad,  0, hc};
Point(4) = {-srad,   0,  0, hc};
Point(5) = {   0,-srad,  0, hc};
Circle(1) = {2,1,3}; Transfinite Line(1) = nt;
Circle(2) = {3,1,4}; Transfinite Line(2) = nt;
Circle(3) = {4,1,5}; Transfinite Line(3) = nt;
Circle(4) = {5,1,2}; Transfinite Line(4) = nt;

// outer cylinder wall
Point(31) = {  ffs, 0, 0, hf};
Point(32) = { 0,  ffs, 0, hf};
Point(33) = { -ffs, 0, 0, hf};
Point(34) = { 0, -ffs, 0, hf};

Circle(31)  = {31, 1, 32}; Transfinite Line(31) = nt;
Circle(32)  = {32, 1, 33}; Transfinite Line(32) = nt;
Circle(33)  = {33, 1, 34}; Transfinite Line(33) = nt;
Circle(34)  = {34, 1, 31}; Transfinite Line(34) = nt;

Line(41) = {2,31}; Transfinite Line(41) = nr Using Progression r;
Line(42) = {3,32}; Transfinite Line(42) = nr Using Progression r;
Line(43) = {4,33}; Transfinite Line(43) = nr Using Progression r;
Line(44) = {5,34}; Transfinite Line(44) = nr Using Progression r;

Line Loop(1) = {-1,-42,31,41};
Line Loop(2) = {-2,-43,32,42};
Line Loop(3) = {-3,-44,33,43};
Line Loop(4) = {-4,-41,34,44};

Plane Surface(1) = {1}; Transfinite Surface {1}; Recombine Surface {1};
Plane Surface(2) = {2}; Transfinite Surface {2}; Recombine Surface {2};
Plane Surface(3) = {3}; Transfinite Surface {3}; Recombine Surface {3};
Plane Surface(4) = {4}; Transfinite Surface {4}; Recombine Surface {4};
Recombine Surface{8};

//physical 
// Inner wall
Physical Line(6) = {1,2,3,4};
// Outer wall
Physical Line(7) = {31,32,33,34};
Physical Surface(1) = {1,2,3,4};

Color Black{ Surface{8}; }
