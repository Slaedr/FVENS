// flow past cylinder

refine = 4;

// inner cyl
srad = 0.5;
// outer cyl
ffs = 1.0;
// mesh size at cylinder
hc = 0.2/refine;
// mesh size at far field
hf = hc;

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
Point(31) = { -ffs, 0, 0, hf};
Point(32) = {  ffs, 0, 0, hf};
Circle(31)  = {31, 1, 32};
Circle(32)  = {32, 1, 31};
Line Loop(8) = {31,32};
Plane Surface(8) = {8,7};
Recombine Surface{8};

//physical 
Physical Line(6) = {1,2,3,4};
Physical Line(7) = {31,32};
Physical Surface(1) = {8};

Color Black{ Surface{8}; }
