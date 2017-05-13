#include <iostream>
#include <fstream>
#include <string>
#include "aoutput.hpp"
#include "aexplicitsolver.hpp"

using namespace amat;
using namespace std;
using namespace acfd;

int main(int argc, char* argv[])
{
	if(argc < 2)
	{
		cout << "Please give a control file name.\n";
		return -1;
	}

	// Read control file
	ifstream control(argv[1]);

	string dum, meshfile, outf, invflux, reconst, limiter;
	double cfl, M_inf, vinf, alpha, rho_inf, tolerance;
	int order, maxiter;

	control >> dum;
	control >> meshfile;
	control >> dum;
	control >> outf;
	control >> dum;
	control >> cfl;
	control >> dum; control >> tolerance;
	control >> dum; control >> maxiter;
	control >> dum;
	control >> M_inf;
	control >> dum;
	control >> vinf;
	control >> dum;
	control >> alpha;
	control >> dum;
	control >> rho_inf;
	control >> dum; control >> order;
	control >> dum; control >> invflux;
	control >> dum; control >> reconst;
	control >> dum; control >> limiter;
	control.close(); 

	// Set up mesh

	UMesh2dh m;
	m.readGmsh2(meshfile,2);
	m.compute_topological();
	m.compute_areas();
	m.compute_jacobians();
	m.compute_face_data();

	// set up problem
	
	EulerFV prob(&m, order, invflux, reconst, limiter);
	prob.loaddata(M_inf, vinf, alpha*PI/180, rho_inf);
	ForwardEulerTimeSolver time(&m, &prob);

	// Now start computation

	time.solve(tolerance, maxiter, cfl);

	prob.compute_entropy_cell();

	//prob.postprocess_point();
	Matrix<a_real> scalars = prob.getscalars();
	Matrix<a_real> velocities = prob.getvelocities();

	string scalarnames[] = {"density", "mach-number", "pressure"};
	writeScalarsVectorToVtu_CellData(outf, m, scalars, scalarnames, velocities, "velocity");

	cout << "\n--------------- End --------------------- \n\n";
	return 0;
}
