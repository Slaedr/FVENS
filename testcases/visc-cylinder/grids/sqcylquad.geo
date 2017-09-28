// flow past cylinder

refine = 8;

// actual boundary
srad = 0.5;
// far field
ffs = 40*srad;
// mesh size at cylinder
hc = 0.2/refine;
// mesh size at far field
hf = 40*hc;

//center
Point(1) = { 0,   0,  0, hc};

//Circle-1
Point(2) = { srad,   0,  0, hc};
Point(3) = {   0, srad,  0, hc};
Point(4) = {-srad,   0,  0, hc};
Point(5) = {   0,-srad,  0, hc};
Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};
Line Loop(7) = {1,2,3,4};

// far field
Point(31) = { -ffs, -ffs, 0, hf};
Point(32) = {  ffs, -ffs, 0, hf};
Point(33) = {  ffs,  ffs, 0, hf};
Point(34) = { -ffs,  ffs, 0, hf};
Line(31) = {31, 32};
Line(32) = {32, 33};
Line(33) = {33, 34};
Line(34) = {34, 31};
Line Loop(8) = {31,32, 33, 34};
Plane Surface(8) = {8,7};
Recombine Surface{8};

//physical 
Physical Line(2) = {1,2,3,4};
// Farfield
Physical Line(4) = {31,32,33,34};
Physical Surface(1) = {8};

Color Black{ Surface{8}; }
