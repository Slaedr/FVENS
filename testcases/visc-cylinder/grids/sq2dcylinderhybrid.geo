// flow past cylinder

// refinement factor
refine = 3;

// actual boundary
srad = 1.0;
// for partition
prad = 3/2.0 * srad;
// far field
ffs = 40.0;
// mesh size at far field
hf = 5.0/refine;

// number of points along the inner circle
n1 = 10*refine;
// number points in normal direction from circle
n2 = 10*refine;
// ratio (r) for geometric progression of transfinite points
rc = 1.1;

//center
Point(1) = {   0,   0,  0, 0.1};

//Circle-1
Point(2) = { srad,   0,  0, 0.1};
Point(3) = {   0, srad,  0, 0.1};
Point(4) = {-srad,   0,  0, 0.1};
Point(5) = {   0,-srad,  0, 0.1};
Circle(1) = {2,1,3}; Transfinite Line {1} = n1;
Circle(2) = {3,1,4}; Transfinite Line {2} = n1;
Circle(3) = {4,1,5}; Transfinite Line {3} = n1;
Circle(4) = {5,1,2}; Transfinite Line {4} = n1;

//Circle-2
Point(12) = { prad,   0,  0, 0.1};
Point(13) = {   0, prad,  0, 0.1};
Point(14) = {-prad,   0,  0, 0.1};
Point(15) = {   0,-prad,  0, 0.1};
Circle(11) = {12,1,13}; Transfinite Line {11} = n1;
Circle(12) = {13,1,14}; Transfinite Line {12} = n1;
Circle(13) = {14,1,15}; Transfinite Line {13} = n1;
Circle(14) = {15,1,12}; Transfinite Line {14} = n1;

// lines joining inner circle (1) points to outer circle (2) points
Line(5) = {2,12}; Transfinite Line {5} = n2 Using Progression rc;
Line(6) = {3,13}; Transfinite Line {6} = n2 Using Progression rc;
Line(7) = {4,14}; Transfinite Line {7} = n2 Using Progression rc;
Line(8) = {5,15}; Transfinite Line {8} = n2 Using Progression rc;

//face-1
Line Loop(1)={5,11,-6,-1};
Plane Surface(1) = {1};
Transfinite Surface {1};
Recombine Surface{1};

//face-2
Line Loop(2)={6,12,-7,-2};
Plane Surface(2) = {2};
Transfinite Surface {2};
Recombine Surface{2};

//face-3
Line Loop(3)={7,13,-8,-3};
Plane Surface(3) = {3};
Transfinite Surface {3};
Recombine Surface{3};

//face-4
Line Loop(4)={8,14,-5,-4};
Plane Surface(4) = {4};
Transfinite Surface {4};
Recombine Surface{4};

// far field
Point(31) = { -ffs, ffs, 0, hf};
Point(32) = { ffs, ffs, 0, hf};
Point(33) = {  ffs, -ffs, 0, hf};
Point(34) = { -ffs, -ffs, 0, hf};
Line(31) = {31, 34};
Line(32) = {34, 33};
Line(33) = {33, 32};
Line(34) = {32, 31};
Line Loop(8) = {31,32,33,34};
Line Loop(9) = {11,12,13,14};
Plane Surface(8) = {8,9};

//physical 
Physical Line(2) = {1,2,3,4};
Physical Line(4) = {31,32,33,34};
Physical Surface(1) = {1,2,3,4,8};

Color Black{ Surface{1,2,3,4,8}; }
