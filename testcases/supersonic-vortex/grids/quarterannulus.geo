h = 0.0125;

ri = 1.0;
ro = 1.384;

Point(1) = {0,0,0, h};
Point(2) = {ri,0,0, h};
Point(3) = {ro,0,0, h};
Point(4) = {0,ro,0, h};
Point(5) = {0,ri,0, h};

Line(1) = {2, 3};
Circle(2) = {3, 1, 4};
Line(3) = {4, 5};
Circle(4) = {5, 1, 2};

Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

Physical Surface(1) = {1};
Physical Line(2) = {2,4};
Physical Line(4) = {1};
Physical Line(10) = {3};
