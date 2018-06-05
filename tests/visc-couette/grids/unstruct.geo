Geometry.Tolerance = 1.0e-14;
Mesh.ToleranceInitialDelaunay = 1.0e-14;

h = 0.2;
length = 1.5;
height = 1.0;

Point(0) = {0,0,0,h};
Point(1) = {length,0,0, h};
Point(2) = {length, height, 0, h};
Point(3) = {0, height, 0, h};

Line(0) = {0,1};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,0};

Periodic Line{1} = {3};

Line Loop(0) = {0,1,2,3};
Plane Surface(1) = {0};

Physical Surface(1) = {1};
Physical Line(4) = {1,3};
Physical Line(2) = {0};
Physical Line(3) = {2};

Color Black{ Surface{1}; }
