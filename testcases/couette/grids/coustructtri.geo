refine = 1;

nhp = 5*refine;
nvp = 15*refine;

h = 0.1/refine;
Point(1) = {0, 0, 0, h};
Point(2) = {1.5, 0, 0, h};
Point(3) = {1.5, 1, 0, h};
Point(4) = {0, 1, 0, h};
Line(1) = {1, 2}; Transfinite Line{1} = nhp;
Line(2) = {2, 3}; Transfinite Line{2} = nvp;
Line(3) = {3, 4}; Transfinite Line{3} = nhp;
Line(4) = {4, 1}; Transfinite Line{4} = nvp;
Line Loop(6) = {1, 2, 3, 4};
Plane Surface(6) = {6}; Transfinite Surface{6};

Physical Line(5) = {2, 4};
Physical Line(6) = {1, 3};
Physical Surface(7) = {6};

Color Black{ Surface{6}; }
