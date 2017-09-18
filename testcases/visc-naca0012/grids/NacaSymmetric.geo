/** Unstructured half NACA0012; only suitable for zero angle of attack
*/

// Modifiable Parameter
Refine = 2;

lc = 0.2/2.0^Refine;
llead = 0.075/2.0^Refine;
ltrail = 0.1/2.0^Refine;

// Farfield spec
rrf = 12.5;			// farfield radius
lf = lc*rrf;		// mesh size at farfield

// Geometry Specification
c  = 1.0;
r  = 0.25;
t  = 0.6*c*Sqrt(r)/1.1019;

a0 =  0.2969;
a1 = -0.1260;
a2 = -0.3516;
a3 =  0.2843;
a4 = -0.1036;

N = 1e2;

Point(0) = {0,0,0,llead};
Point(1) = {-rrf,0,0,lf};
Point(2) = {0,rrf,0,lf};
Point(3) = {rrf,0,0,lf};

Point(6) = {c,0,0,ltrail};

x = 0.5;
y = 5*c*t*(a0*Sqrt(x/c)+a1*(x/c)^1+a2*(x/c)^2+a3*(x/c)^3+a4*(x/c)^4);

Point(8) = {x,y,0,lc};

pListBL[0] = 0;
For i In {1:N-1}
	x = c/2*(i/N);
	y = 5*c*t*(a0*Sqrt(x/c)+a1*(x/c)^1+a2*(x/c)^2+a3*(x/c)^3+a4*(x/c)^4);
	
	pListBL[i] = newp;
	Point(pListBL[i]) = {x,y,0,lc};
EndFor
pListBL[N] = 8;

pListBR[0] = 8;
For i In {1:N-1}
	x = c/2*(1+i/N);
	y = 5*c*t*(a0*Sqrt(x/c)+a1*(x/c)^1+a2*(x/c)^2+a3*(x/c)^3+a4*(x/c)^4);
	
	pListBR[i] = newp;
	Point(pListBR[i]) = {x,y,0,lc};
EndFor
pListBR[N] = 6;


Line(1001)   = {1,0};
Spline(1002) = pListBL[];
Spline(1003) = pListBR[];
Line(1004)   = {6,3};
Circle(1005) = {3,0,2};
Circle(1006) = {2,0,1};

Line Loop (4001) = {1001,1002,1003,1004,1005,1006};

Plane Surface(4001) = {4001}; 
Recombine Surface(4001);

// Physical Parameters for '.msh' file

Physical Line(2) = {1002,1003};                 // Slip wall
Physical Line(4) = {1005,1006};                 // Farfield
Physical Line(6) = {1001,1004};                 // Symmetry

Physical Surface(9401) = {4001};



// Visualization in gmsh

Color Black{ Surface{4001:4004}; }
Geometry.Color.Points = Black;
Coherence;
