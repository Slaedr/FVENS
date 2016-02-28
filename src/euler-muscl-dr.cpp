#include <iostream>
#include <fstream>
#include <string>
#include <aoutput.hpp>
#include <aeulerfv-muscl.hpp>

using namespace std;
using namespace acfd;

int main()
{
	// Read control file
	ifstream control("euler-muscl.control");

	string dum, meshfile, outf; double cfl, ttime, M_inf, vinf, alpha, rho_inf, tolerance;

	control >> dum;
	control >> meshfile;
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

	ifstream meshs(meshfile);

	UTriMesh m(meshs);
	m->compute_jacobians();
	m->compute_face_data();

	// Now start computation

	EulerFV prob(&m);
	prob.loaddata(M_inf, vinf, alpha*PI/180, rho_inf);

	//All hell breaks loose
	prob.solve_rk1_steady(tolerance, cfl);

	prob.postprocess_cell();

	double err = prob.compute_error_cell();

	string scalarnames[] = {"density", "mach-number", "pressure"};
	//writeScalarsVectorToVtu(outf, m, prob.scalars, scalarnames, prob.velocities, "velocity");
	writeScalarsVectorToVtu_CellData(outf, m, prob.scalars, scalarnames, prob.velocities, "velocity");

	control.close(); meshs.close();
	cout << "\n--------------- End --------------------- \n";
	return 0;
}
