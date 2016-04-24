#include <iostream>
#include <fstream>
#include <string>
#include <aoutput.hpp>
#include <aexplicitsolver.hpp>

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
	double cfl, ttime, M_inf, vinf, alpha, rho_inf, tolerance;
	int order, maxiter;

	control >> dum;
	control >> meshfile;
	control >> dum;
	control >> outf;
	control >> dum;
	control >> cfl;
	control >> dum;
	control >> ttime;
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

	//ifstream meshs(meshfile);

	UMesh2dh m;
	m.readGmsh2(meshfile,2);
	m.compute_topological();
	m.compute_jacobians();
	m.compute_face_data();

	// Now start computation

	ExplicitSolver prob(&m, order, invflux, reconst, limiter);
	prob.loaddata(M_inf, vinf, alpha*PI/180, rho_inf);

	//All hell breaks loose
	prob.solve_rk1_steady(tolerance, maxiter, cfl);

	prob.postprocess_cell();

	double err = prob.compute_entropy_cell();

	Matrix<acfd_real> scalars = prob.getscalars();
	Matrix<acfd_real> velocities = prob.getvelocities();

	string scalarnames[] = {"density", "mach-number", "pressure"};
	//writeScalarsVectorToVtu(outf, m, prob.scalars, scalarnames, prob.velocities, "velocity");
	writeScalarsVectorToVtu_CellData(outf, m, scalars, scalarnames, velocities, "velocity");

	//meshs.close();
	
	cout << "\n--------------- End --------------------- \n";
	return 0;
}
