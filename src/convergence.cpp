#include <iostream>
#include <fstream>
#include <string>
#include <aoutput.hpp>
#include "aeulerfv-muscl.hpp"

const double Pi = 3.14159265358979323846;

using namespace std;
using namespace acfd;

int main()
{
	// Read control file
	ifstream control("euler-convergence.control");

	string dum, meshfile1, meshfile2, meshfile3, meshfile4, outf;
	double cfl, ttime, M_inf, vinf, alpha, rho_inf, tolerance;

	meshfile1 = "feflo.domn.cylinder.coarse";
	meshfile2 = "feflo.domn.sphere.medium";
	meshfile3 = "feflo.domn.cylinder.fine";
	meshfile4 = "feflo.domn.sphere.vfine";
	control >> dum;
	control >> outf;
	control >> dum;
	control >> cfl;
	control >> dum;
	control >> ttime;
	control >> dum;
	control >> tolerance;
	control >> dum;
	control >> M_inf;
	control >> dum;
	control >> vinf;
	control >> dum;
	control >> alpha;
	control >> dum;
	control >> rho_inf;

	// Set up mesh

	ifstream mesh1(meshfile1);
	ifstream mesh1(meshfile1);
	ifstream mesh1(meshfile1);
	ifstream mesh1(meshfile1);

	UTriMesh m(meshs);

	// Now start computation

	EulerFV prob(&m);
	prob.loaddata(M_inf, vinf, alpha*Pi/180, rho_inf);

	//All hell breaks loose
	prob.solve_rk1_steady(tolerance, cfl);

	prob.postprocess_cell();

	string scalarnames[] = {"density", "mach-number", "pressure"};
	//writeScalarsVectorToVtu(outf, m, prob.scalars, scalarnames, prob.velocities, "velocity");
	writeScalarsVectorToVtu_CellData(outf, m, prob.scalars, scalarnames, prob.velocities, "velocity");

	control.close(); meshs.close();
	cout << "\n--------------- End --------------------- \n";
	return 0;
}
