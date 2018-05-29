// Simple square test case for linear advection with DGFEM

/*
  This file is part of FVENS.
   FVENS is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   FVENS is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
*/

h = 0.2/refine;
L = 5.0;

xstart = -L;
xend = L;
ystart = -L;
yend = L;
Point(1) = {xstart, ystart, 0, h};
Point(2) = {xend, ystart, 0, h};
Point(3) = {xend, yend, 0, h};
Point(4) = {xstart, yend, 0, h};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(6) = {1, 2, 3, 4};
Plane Surface(6) = {6};
Physical Line(2) = {1, 3};
Physical Line(1) = {2, 4};         // Periodic
Physical Surface(7) = {6};

// Comment out for triangular mesh
//Recombine Surface {6};

Color Black{ Surface{6}; }
