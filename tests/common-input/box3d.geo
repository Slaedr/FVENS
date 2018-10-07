
lx = 1.0;
ly = 1.0;
lz = 1.0;
ox = 0.0;
oy = 0.0;
oz = 0.0;

h = 0.33;

Point(0) = {ox,   oy,   oz,   h};
Point(1) = {ox+lx,oy,   oz,   h};
Point(2) = {ox+lx,oy+ly,oz,   h};
Point(3) = {ox,   oy+ly,oz,   h};
Point(4) = {ox,   oy,   oz+lz,h};
Point(5) = {ox+lx,oy,   oz+lz,h};
Point(6) = {ox+lx,oy+ly,oz+lz,h};
Point(7) = {ox,   oy+ly,oz+lz,h};

Line(0) = {0,1};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,0};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,4};
Line(8) = {0,4};
Line(9) = {1,5};
Line(10) = {2,6};
Line(11) = {3,7};

Curve Loop(0) = {0,1,2,3};
Curve Loop(1) = {4,5,6,7};
Curve Loop(2) = {0,9,-4,-8};
Curve Loop(3) = {1,10,-5,-9};
Curve Loop(4) = {10,6,-11,-2};
Curve Loop(5) = {-3,11,7,-8};
