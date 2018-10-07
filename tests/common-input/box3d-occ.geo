//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
//+
Physical Volume(1) = {1};
//+
Physical Surface(2) = {6, 1, 5, 2, 4, 3};
//+
lc = DefineNumber[ 0.33, Name "Parameters/lc" ];
//+
Characteristic Length {4, 3, 8, 7, 6, 2, 1, 5} = lc;
