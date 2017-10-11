#include <blockmatrices.hpp>
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

	string dum, meshfile, outf, logfile, lognresstr, simtype, recprim, initcondfile;
	string invflux, invfluxjac, reconst, limiter, linsolver, prec, timesteptype, usemf;
	string constvisc, surfnamepref, volnamepref, isVolOutReq, residualsmoothing;
	
	double initcfl, endcfl, Minf, alpha, tolerance, lintol, 
		   firstinitcfl, firstendcfl, firsttolerance;
	double Reinf=0, Tinf=0, Pr=0, gamma, twalltemp=0, twallvel = 0;
	double tpwalltemp=0, tpwallpressure=0, tpwallvel=0, adiawallvel;
	
	int maxiter, linmaxiterstart, linmaxiterend, rampstart, rampend, 
		firstmaxiter, firstrampstart, firstrampend,
		restart_vecs, farfield_marker, inout_marker, slipwall_marker, isothermalwall_marker=-1,
		isothermalpressurewall_marker=-1, adiabaticwall_marker=-1, extrap_marker=-1;
	int out_nwalls, out_nothers;
	short inittype, usestarter;
	unsigned short nbuildsweeps, napplysweeps;
	
	bool use_matrix_free, lognres, reconstPrim, useconstvisc=false, viscsim=false,
		 order2 = true;
	
	char mattype, dumc;
	
	std::vector<int> lwalls, lothers;

	std::getline(control,dum); control >> meshfile;
	control >> dum; control >> outf;
	control >> dum; control >> logfile;
	control >> dum; control >> lognresstr;
	control >> dum;
	control >> dum; control >> simtype;
	control >> dum; control >> gamma;
	control >> dum; control >> alpha;
	control >> dum; control >> Minf;
	if(simtype == "NAVIERSTOKES") {
		control >> dum; control >> Tinf;
		control >> dum; control >> Reinf;
		control >> dum; control >> Pr;
		control >> dum; control >> constvisc;
		if(constvisc == "YES")
			useconstvisc = true;
		viscsim = true;
	}
	control >> dum; control >> inittype;
	if(inittype == 1) {
		control >> dum; control >> initcondfile;
	}
	control.get(dumc); std::getline(control,dum); // FIXME
	control >> dum; control >> slipwall_marker;
	control >> dum; control >> farfield_marker;
	control >> dum; control >> inout_marker;
	control >> dum; control >> extrap_marker;
	if(viscsim) {
		std::getline(control,dum); std::getline(control,dum); control >> isothermalwall_marker;
		std::getline(control,dum); std::getline(control,dum); control >> twalltemp >> twallvel;
		std::getline(control,dum); std::getline(control,dum); control >> adiabaticwall_marker;
		std::getline(control,dum); std::getline(control,dum); control >> adiawallvel;
		std::getline(control,dum); std::getline(control,dum); control >> isothermalpressurewall_marker;
		std::getline(control,dum); std::getline(control,dum); control >> tpwalltemp >> tpwallvel 
			>> tpwallpressure;
	}
	control >> dum; control >> out_nwalls;
	lwalls.resize(out_nwalls);
	if(out_nwalls > 0) {
		control >> dum;
		for(int i = 0; i < out_nwalls; i++)
			control >> lwalls[i];
	}
	control >> dum; control >> out_nothers;
	lothers.resize(out_nothers);
	if(out_nothers > 0) {
		control >> dum;
		for(int i = 0; i < out_nothers; i++)
			control >> lothers[i];
	}
	control >> dum; control >> surfnamepref;
	control >> dum; control >> isVolOutReq;
	if(isVolOutReq == "YES") {
		control >> dum; control >> volnamepref;
	}

	control >> dum;
	control >> dum; control >> invflux;
	control >> dum; control >> reconst;
	if(reconst == "NONE")
		order2 = false;
	control >> dum; control >> limiter;
	control >> dum; control >> recprim;
	control >> dum;
	control >> dum; control >> timesteptype;
	control >> dum; control >> initcfl;
	control >> dum; control >> endcfl;
	control >> dum; control >> rampstart >> rampend;
	control >> dum; control >> tolerance;
	control >> dum; control >> maxiter;
	control >> dum;
	control >> dum; control >> usestarter;
	control >> dum; control >> firstinitcfl;
	control >> dum; control >> firstendcfl;
	control >> dum; control >> firstrampstart >> firstrampend;
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
		control >> dum; control >> nbuildsweeps >> napplysweeps;
	}
	else {
		control >> dum;
		std::string residualsmoothingstr;
		control >> dum; control >> residualsmoothingstr;
		if(residualsmoothingstr == "YES")
			residualsmoothing = true;
		invfluxjac = invflux;
	}
	control.close();

	if(usemf == "YES")
		use_matrix_free = true;
	else
		use_matrix_free = false;
	
	if(lognresstr == "YES")
		lognres = true;
	else
		lognres = false;
	if(recprim == "NO")
		reconstPrim = false;
	else
		reconstPrim = true;
	
	if(meshfile == "READFROMCMD")
	{
		if(argc >= 3)
			meshfile = argv[2];
		else
			std::cout << "! Mesh file not given in command line!\n";
	}

	// Set up mesh

	UMesh2dh m;
	m.readMesh(meshfile);
	m.compute_topological();
	m.compute_areas();
	m.compute_jacobians();
	m.compute_face_data();

	// set up problem
	
	std::cout << "Setting up main spatial scheme.\n";
	FlowFV prob(&m, gamma, Minf, Tinf, Reinf, Pr, alpha*PI/180.0, viscsim, useconstvisc,
			isothermalwall_marker, adiabaticwall_marker, isothermalpressurewall_marker,
			slipwall_marker, farfield_marker, inout_marker, extrap_marker,
			twalltemp, twallvel, adiawallvel, tpwalltemp, tpwallvel, tpwallpressure,
			invflux, invfluxjac, reconst, limiter, order2, reconstPrim);
	std::cout << "Setting up spatial scheme for the initial guess.\n";
	FlowFV startprob(&m, gamma, Minf, Tinf, Reinf, Pr, alpha*PI/180.0, viscsim, useconstvisc,
			isothermalwall_marker, adiabaticwall_marker, isothermalpressurewall_marker,
			slipwall_marker, farfield_marker, inout_marker, extrap_marker,
			twalltemp, twallvel, adiawallvel, tpwalltemp, tpwallvel, tpwallpressure,
			invflux, invfluxjac, "NONE", "NONE",false,true);
	
	// solution vector
	MVector u(m.gnelem(),NVARS);

	// Initialize Jacobian for implicit schemes; no storage allocated here
	blasted::LinearOperator<a_real,a_int> * M;
	if(mattype == 'd') {
		M = new blasted::DLUMatrix<NVARS>(&m,nbuildsweeps,napplysweeps);
		std::cout << " Selected DLU matrix for Jacobian storage, if required.\n";
	}
	else if(mattype == 'c') {
		M = new blasted::BSRMatrix<a_real,a_int,1>(nbuildsweeps,napplysweeps);
		std::cout << " Selected CSR matrix for Jacobian storage, if required.\n";
	}
	else {
		M = new blasted::BSRMatrix<a_real,a_int,NVARS>(nbuildsweeps,napplysweeps);
		std::cout << " Selected BSR matrix for Jacobian storage, if required.\n";
	}

	// set up time discrization

	SteadySolver<4> * starttime=nullptr, * time=nullptr;

	if(timesteptype == "IMPLICIT") 
	{
		if(use_matrix_free)
			std::cout << "!! Matrix-free not implemented yet! Using matrix-storage instead.\n";
			
		// Pre-allocate storage for Jacobian matrix	
		setupMatrixStorage<NVARS>(&m, mattype, M);
		
		if(usestarter != 0)
			starttime = new SteadyBackwardEulerSolver<4>(&m, &startprob, u, M,
				firstinitcfl, firstendcfl, firstrampstart, firstrampend, firsttolerance, firstmaxiter, 
				lintol, linmaxiterstart, linmaxiterend, linsolver, prec, 
				restart_vecs, lognres);

		time = new SteadyBackwardEulerSolver<4>(&m, &prob, u, M,
				initcfl, endcfl, rampstart, rampend, tolerance, maxiter, 
				lintol, linmaxiterstart, linmaxiterend, linsolver, prec, 
				restart_vecs, lognres);

		std::cout << "Set up backward Euler temporal scheme.\n";
	}
	else 
	{
		if(residualsmoothing)
			setupLaplacianSmoothingMatrix<NVARS>(&m, M);

		if(usestarter != 0)
			starttime = new SteadyForwardEulerSolver<4>(&m, &startprob, u,
					firsttolerance, firstmaxiter, firstinitcfl, lognres);

		time = new SteadyForwardEulerSolver<4>(&m, &prob, u,
				tolerance, maxiter, initcfl, lognres);

		std::cout << "Set up explicit forward Euler temporal scheme.\n";
	}
	
	// Ask the spatial discretization context to initialize flow variables
	startprob.initializeUnknowns(false, initcondfile, u);

	// computation
	
	if(usestarter != 0)
		// solve the starter problem to get the initial solution
		starttime->solve(logfile);

	/* Solve the main problem using either the initial solution
	 * set by initializeUnknowns or the one computed by the starter problem.
	 */
	time->solve(logfile);

	// export output to VTU

	Array2d<a_real> scalars;
	Array2d<a_real> velocities;
	prob.postprocess_point(u, scalars, velocities);

	string scalarnames[] = {"density", "mach-number", "pressure", "temperature"};
	writeScalarsVectorToVtu_PointData(outf, m, scalars, scalarnames, velocities, "velocity");

	// export surface data like pressure coeff etc and volume data as plain text files
	
	IdealGasPhysics phy(gamma, Minf, Tinf, Reinf, Pr);
	FlowOutput out(&m, &prob, &phy, alpha);
	
	out.exportSurfaceData(u, lwalls, lothers, surfnamepref);
	
	if(isVolOutReq == "YES")
		out.exportVolumeData(u, volnamepref);

	if(usestarter != 0)
		delete starttime;
	delete time;

	cout << "\n--------------- End --------------------- \n\n";
	return 0;
}
