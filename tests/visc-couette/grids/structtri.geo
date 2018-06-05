refine = 2;

//+
h = DefineNumber[ 0.2/refine, Name "Parameters/h" ];
//+
height = DefineNumber[ 1.0, Name "Parameters/height" ];
//+
length = DefineNumber[ 1.5, Name "Parameters/length" ];
//+
nx = DefineNumber[ 4*refine, Name "Parameters/nx" ];
//+
ny = DefineNumber[ 10*refine, Name "Parameters/ny" ];
//+
Point(1) = {0, 0, 0, h};
//+
Point(2) = {length, 0, 0, h};
//+
Point(3) = {length, height, 0, h};
//+
Point(4) = {0, height, 0, h};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Line Loop(1) = {3, 4, 1, 2};
//+
Periodic Line{2} = {4};
//+
Plane Surface(1) = {1};
//+
Physical Line(4) = {4, 2};
//+
Physical Line(2) = {1};
//+
Physical Line(3) = {3};
//+
Physical Surface(5) = {1};
//+
Transfinite Line {1, 3} = nx Using Progression 1;
//+
Transfinite Line {2, 4} = ny Using Progression 1;
//+
Transfinite Surface {1};
