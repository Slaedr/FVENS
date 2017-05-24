#include <iostream>
#include <fstream>
#include <string>
#include "aoutput.hpp"
#include "aodesolver.hpp"

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
	double cfl, M_inf, vinf, alpha, rho_inf, tolerance, firstcfl, firsttolerance;
	int maxiter, firstmaxiter;
	short inittype, usestarter;

	control >> dum;
	control >> meshfile;
	control >> dum;
	control >> outf;
	control >> dum;
	control >> M_inf;
	control >> dum;
	control >> vinf;
	control >> dum;
	control >> alpha;
	control >> dum;
	control >> rho_inf;
	control >> dum; control >> inittype;
	control >> dum; control >> invflux;
	control >> dum; control >> reconst;
	control >> dum; control >> limiter;
	control >> dum; control >> cfl;
	control >> dum; control >> tolerance;
	control >> dum; control >> maxiter;
	control >> dum; control >> usestarter;
	control >> dum; control >> firstcfl;
	control >> dum; control >> firsttolerance;
	control >> dum; control >> firstmaxiter;
	control.close();

	// Set up mesh

	UMesh2dh m;
	m.readGmsh2(meshfile,2);
	m.compute_topological();
	m.compute_areas();
	m.compute_jacobians();
	m.compute_face_data();

	// set up problem
	
	// the actual spatial didcretization to use
	std::cout << "Setting up main spatial scheme.\n";
	EulerFV prob(&m, invflux, "LLF", reconst, limiter);
	// the first-order discretization to provide a good initial guess
	std::cout << "Setting up spatial scheme for the initial guess.\n";
	EulerFV startprob(&m, invflux, "LLF", "NONE", "NONE");

	SteadyForwardEulerSolver time(&m, &prob, &startprob, usestarter, tolerance, maxiter, cfl, firsttolerance, firstmaxiter, firstcfl);
	prob.loaddata(inittype, M_inf, vinf, alpha*PI/180, rho_inf, time.unknowns());
	startprob.loaddata(inittype, M_inf, vinf, alpha*PI/180, rho_inf, time.unknowns());

	// Now start computation

	time.solve();

	//prob.postprocess_point();
	Array2d<a_real> scalars;
	Array2d<a_real> velocities;
	prob.postprocess_point(time.unknowns(), scalars, velocities);

	string scalarnames[] = {"density", "mach-number", "pressure"};
	writeScalarsVectorToVtu_PointData(outf, m, scalars, scalarnames, velocities, "velocity");

	cout << "\n--------------- End --------------------- \n\n";
	return 0;
}
