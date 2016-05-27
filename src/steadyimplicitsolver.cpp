#include <iostream>
#include <fstream>
#include <string>
#include <aoutput.hpp>
#include <aimplicitsolver.hpp>

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

	string dum, meshfile, outf, invflux, reconst, limiter, solver;
	double cfl, initcfl, ttime, M_inf, vinf, alpha, rho_inf, tolerance, lintolerance, omega;
	int order, maxiter, linmaxiter, swtchstp;

	control >> dum; control >> meshfile;
	control >> dum; control >> outf;
	control >> dum; control >> cfl;
	control >> dum; control >> initcfl;
	control >> dum; control >> swtchstp;
	control >> dum; control >> tolerance;
	control >> dum; control >> maxiter;
	control >> dum; control >> M_inf;
	control >> dum; control >> vinf;
	control >> dum; control >> alpha;
	control >> dum; control >> rho_inf;
	control >> dum; control >> order;
	control >> dum; control >> invflux;
	control >> dum; control >> reconst;
	control >> dum; control >> limiter;
	control >> dum; control >> solver;
	control >> dum; control >> omega;
	control >> dum; control >> lintolerance;
	control >> dum; control >> linmaxiter;
	control.close(); 

	// Set up mesh

	UMesh2dh m;
	m.readGmsh2(meshfile,2);
	m.compute_topological();
	m.compute_areas();
	m.compute_jacobians();
	m.compute_face_data();

	// Now start computation

	SteadyStateImplicitSolver prob(&m, order, invflux, reconst, limiter, solver, cfl, initcfl, swtchstp, omega, tolerance, maxiter, lintolerance, linmaxiter);
	prob.loaddata(M_inf, vinf, alpha*PI/180, rho_inf);

	prob.solve();

	double err = prob.compute_entropy_cell();

	//prob.postprocess_point();
	Matrix<acfd_real> scalars = prob.getscalars();
	Matrix<acfd_real> velocities = prob.getvelocities();

	string scalarnames[] = {"density", "mach-number", "pressure"};
	writeScalarsVectorToVtu_CellData(outf, m, scalars, scalarnames, velocities, "velocity");

	cout << "\n--------------- End --------------------- \n";
	return 0;
}
