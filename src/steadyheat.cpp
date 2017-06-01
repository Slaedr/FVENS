#include "aoutput.hpp"
#include "aodesolver.hpp"
#include <function>

using namespace amat;
using namespace std;
using namespace acfd;

void source(const a_real *const r, const a_real t, const a_real *const u, a_real *const sourceterm)
{
	return 8.0*PI*PI*sin(2*PI*r[0])*sin(2*PI*r[1]);
}
void exact(double x, double y) {
	return sin(2*PI*x)*sin(2*PI*y);
}

int main(int argc, char* argv[])
{
	if(argc < 2)
	{
		cout << "Please give a control file name.\n";
		return -1;
	}

	// Read control file
	ifstream control(argv[1]);

	string dum, meshfile, outf, visflux, reconst, linsolver;
	double initcfl, endcfl, tolerance, lintol, lin_relaxfactor, firstcfl, firsttolerance, diffcoeff, bvalue;
	int maxiter, linmaxiterstart, linmaxiterend, rampstart, rampend, firstmaxiter;
	short inittype, usestarter;

	control >> dum; control >> meshfile;
	control >> dum; control >> outf;
	control >> dum; control >> diffcoeff;
	control >> dum; control >> bvalue;
	control >> dum; control >> inittype;
	control >> dum; control >> visflux;
	control >> dum; control >> reconst;
	control >> dum; control >> initcfl;
	control >> dum; control >> endcfl;
	control >> dum; control >> rampstart;
	control >> dum; control >> rampend;
	control >> dum; control >> tolerance;
	control >> dum; control >> maxiter;
	control >> dum; control >> linsolver;
	control >> dum; control >> lintol;
	control >> dum; control >> linmaxiterstart;
	control >> dum; control >> linmaxiterend;
	control >> dum; control >> lin_relaxfactor;
	control >> dum; control >> usestarter;
	control >> dum; control >> firstcfl;
	control >> dum; control >> firsttolerance;
	control >> dum; control >> firstmaxiter;
	control.close();

	// rhs and exact soln
	auto rhs = [diffcoeff](const a_real *const r, const a_real t, const a_real *const u, a_real *const sourceterm)
		{ sourceterm[0] = diffcoeff*8.0*PI*PI*sin(2*PI*r[0])*sin(2*PI*r[1]); };
	auto uexact = [](const a_real *const r)->a_real { return sin(2*PI*r[0])*sin(2*PI*r[1]); };

	// Set up mesh

	UMesh2dh m;
	m.readGmsh2(meshfile,2);
	m.compute_topological();
	m.compute_areas();
	m.compute_jacobians();
	m.compute_face_data();

	// set up problem
	
	std::cout << "Setting up main spatial scheme.\n";
	DiffusionThinLayer<1> prob(&m, diffcoeff, bvalue);
	DiffusionThinLayer<1> startprob(&m, diffcoeff, bvalue, &rhs);
	SteadyBackwardEulerSolver<1> time(&m, &prob, &startprob, usestarter, initcfl, endcfl, rampstart, rampend, tolerance, maxiter, 
			lintol, linmaxiterstart, linmaxiterend, linsolver, firsttolerance, firstmaxiter, firstcfl);

	// computation
	time.solve();

	Matrix<a_real,Dynamic,Dynamic,RowMajor>& u = time.unknowns();

	a_real err = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		a_real rc[2]; rc[0] = 0; rc[1] = 0;
		for(int inode = 0; inode < m.gnnode(iel); inode++) {
			rc[0] += m->gcoords(m->ginpoel(iel,inode),0);
			rc[1] += m->gcoords(m->ginpoel(iel,inode),1);
		}
		rc[0] /= m->gnnode(iel); rc[1] /= m->gnnode(iel);
		a_real trueval = uexact(rc);
		err += (u(iel,0)-trueval)*(u(iel,0)-trueval)*m.garea(iel);
	}
	err = sqrt(err);
	double h = 1.0/sqrt(m.gnelem());
	cout << "Log of Mesh size and error are " << log10(h) << "  " << log10(err) << endl;

	cout << "\n--------------- End --------------------- \n\n";
	return 0;
}
