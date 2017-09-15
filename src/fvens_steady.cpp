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

	string dum, meshfile, outf, logfile, lognresstr, simtype;
	string invflux, invfluxjac, reconst, limiter, linsolver, prec, timesteptype, usemf;
	double initcfl, endcfl, Minf, vinf, alpha, rhoinf, tolerance, lintol, firstcfl, firsttolerance;
	double Reinf, Tinf, Pr, gamma;
	int maxiter, linmaxiterstart, linmaxiterend, rampstart, rampend, firstmaxiter, restart_vecs;
	short inittype, usestarter;
	unsigned short nbuildsweeps, napplysweeps;
	bool use_matrix_free, lognres;
	char mattype;

	control >> dum; control >> meshfile;
	control >> dum; control >> outf;
	control >> dum; control >> logfile;
	control >> dum; control >> lognresstr;
	control >> dum;
	control >> dum; control >> simtype;
	control >> dum; control >> alpha;
	control >> dum; control >> Minf;
	control >> dum; control >> Tinf;
	control >> dum; control >> Reinf;
	control >> dum; control >> Pr;
	control >> dum; control >> gamma;
	control >> dum; control >> inittype;
	if(inittype == 1) {
		control >> dum; control >> vinf;
		control >> dum; control >> rhoinf;
	}
	control >> dum;
	control >> dum; control >> invflux;
	control >> dum; control >> reconst;
	control >> dum; control >> limiter;
	control >> dum;
	control >> dum; control >> timesteptype;
	control >> dum; control >> initcfl;
	control >> dum; control >> endcfl;
	control >> dum; control >> rampstart;
	control >> dum; control >> rampend;
	control >> dum; control >> tolerance;
	control >> dum; control >> maxiter;
	control >> dum;
	control >> dum; control >> usestarter;
	control >> dum; control >> firstcfl;
	control >> dum; control >> firsttolerance;
	control >> dum; control >> firstmaxiter;
	if(timesteptype == "IMPLICIT") {
		control >> dum;
		control >> dum; control >> invfluxjac;
		control >> dum; control >> usemf;
		control >> dum; control >> mattype;
		control >> dum; control >> linsolver;
		control >> dum; control >> lintol;
		control >> dum; control >> linmaxiterstart;
		control >> dum; control >> linmaxiterend;
		control >> dum; control >> restart_vecs;
		control >> dum; control >> prec;
		control >> dum; control >> nbuildsweeps;
		control >> dum; control >> napplysweeps;
	}
	else
		invfluxjac = invflux;
	control.close();

	if(usemf == "YES")
		use_matrix_free = true;
	else
		use_matrix_free = false;
	
	if(lognresstr == "YES")
		lognres = true;
	else
		lognres = false;
	
	if(meshfile == "READFROMCMD")
	{
		if(argc >= 3)
			meshfile = argv[2];
		else
			std::cout << "! Mesh file not given in command line!\n";
	}

	// Set up mesh

	UMesh2dh m;
	m.readGmsh2(meshfile,2);
	m.compute_topological();
	m.compute_areas();
	m.compute_jacobians();
	m.compute_face_data();

	// set up problem
	
	std::cout << "Setting up main spatial scheme.\n";
	FlowFV prob(&m, gamma, Minf, Tinf, Reinf, Pr, invflux, invfluxjac, reconst, limiter);
	std::cout << "Setting up spatial scheme for the initial guess.\n";
	FlowFV startprob(&m, gamma, Minf, Tinf, Reinf, Pr, invflux, invfluxjac, "NONE", "NONE");
	
	SteadySolver<4>* time;
	if(timesteptype == "IMPLICIT") {
		if(use_matrix_free)
			time = new SteadyMFBackwardEulerSolver<4>(&m, &prob, &startprob, usestarter, initcfl, endcfl, rampstart, rampend, tolerance, maxiter, 
				lintol, linmaxiterstart, linmaxiterend, linsolver, prec, nbuildsweeps, napplysweeps, firsttolerance, firstmaxiter, firstcfl, restart_vecs, lognres);
		else
			time = new SteadyBackwardEulerSolver<4>(&m, &prob, &startprob, usestarter, initcfl, endcfl, rampstart, rampend, tolerance, maxiter, 
				mattype, lintol, linmaxiterstart, linmaxiterend, linsolver, prec, nbuildsweeps, napplysweeps, firsttolerance, firstmaxiter, firstcfl, restart_vecs, lognres);
		std::cout << "Setting up backward Euler temporal scheme.\n";
	}
	else {
		time = new SteadyForwardEulerSolver<4>(&m, &prob, &startprob, usestarter, tolerance, maxiter, initcfl, firsttolerance, firstmaxiter, firstcfl, lognres);
		std::cout << "Setting up explicit forward Euler temporal scheme.\n";
	}
	
	if(inittype == 1)
	{
		startprob.loaddata_special(inittype, vinf, alpha*PI/180, rhoinf, time->unknowns());
		prob.loaddata_special(inittype, vinf, alpha*PI/180, rhoinf, time->unknowns());
	}
	else
	{
		startprob.loaddata(alpha*PI/180.0, time->unknowns());
		prob.loaddata(alpha*PI/180.0, time->unknowns());
	}

	// computation
	time->solve(logfile);

	Array2d<a_real> scalars;
	Array2d<a_real> velocities;
	prob.postprocess_point(time->unknowns(), scalars, velocities);

	string scalarnames[] = {"density", "mach-number", "pressure"};
	writeScalarsVectorToVtu_PointData(outf, m, scalars, scalarnames, velocities, "velocity");

	delete time;
	cout << "\n--------------- End --------------------- \n\n";
	return 0;
}
