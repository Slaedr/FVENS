// Structured mesh for quarter annulus of L. Krivodonova's supersonic vortex case

// coarsest: 0.2 - useless here
h = 0.2;

// controls mesh size - coarsest mesh 0: n1 = 3; step 2; mesh 6: n1 = 300
n1 = 6;
n2 = 2*n1;

ri = 1.0;
ro = 1.384;

Point(1) = {0,0,0, h};
Point(2) = {ri,0,0, h};
Point(3) = {ro,0,0, h};
Point(4) = {0,ro,0, h};
Point(5) = {0,ri,0, h};

Line(1) = {2, 3}; Transfinite Line {1} = n1;
Circle(2) = {3, 1, 4}; Transfinite Line {2} = n2;
Line(3) = {4, 5}; Transfinite Line {3} = n1;
Circle(4) = {5, 1, 2}; Transfinite Line {4} = n2;

Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};
Transfinite Surface {1};
//Recombine Surface {1};

Physical Surface(1) = {1};

Physical Line(2) = {2,4};
Physical Line(4) = {1};
Physical Line(10) = {3};
