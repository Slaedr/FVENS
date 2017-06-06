#include "aoutput.hpp"
#include "aodesolver.hpp"

using namespace amat;
using namespace std;
using namespace acfd;

/*a_real source(const a_real *const r, const a_real t, const a_real *const u, a_real *const sourceterm)
{
	return 8.0*PI*PI*sin(2*PI*r[0])*sin(2*PI*r[1]);
}
a_real exact(double x, double y) {
	return sin(2*PI*x)*sin(2*PI*y);
}*/

int main(int argc, char* argv[])
{
	if(argc < 2)
	{
		cout << "Please give a control file name.\n";
		return -1;
	}

	// Read control file
	ifstream control(argv[1]);

	string dum, meshfile, outf, visflux, reconst, linsolver, timesteptype;
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
	control >> dum; control >> timesteptype;
	control >> dum; control >> initcfl;
	control >> dum; control >> endcfl;
	control >> dum; control >> rampstart;
	control >> dum; control >> rampend;
	control >> dum; control >> tolerance;
	control >> dum; control >> maxiter;
	control >> dum; control >> usestarter;
	control >> dum; control >> firstcfl;
	control >> dum; control >> firsttolerance;
	control >> dum; control >> firstmaxiter;
	if(timesteptype == "IMPLICIT") {
		control >> dum; control >> linsolver;
		control >> dum; control >> lintol;
		control >> dum; control >> linmaxiterstart;
		control >> dum; control >> linmaxiterend;
		control >> dum; control >> lin_relaxfactor;
	}
	control.close();

	// rhs and exact soln
	std::function<void(const a_real *const, const a_real, const a_real *const, a_real *const)> rhs 
		= [diffcoeff](const a_real *const r, const a_real t, const a_real *const u, a_real *const sourceterm)
		{ sourceterm[0] = diffcoeff*8.0*PI*PI*sin(2*PI*r[0])*sin(2*PI*r[1]); };
	auto uexact = [](const a_real *const r)->a_real { return sin(2*PI*r[0])*sin(2*PI*r[1]); };

	a_real err = 0;

	// Set up mesh

	UMesh2dh m;
	m.readGmsh2(meshfile,2);
	m.compute_topological();
	m.compute_areas();
	m.compute_jacobians();
	m.compute_face_data();

	// set up problem
	
	std::cout << "Setting up spatial scheme.\n";
	Diffusion<1>* prob;
	Diffusion<1>* startprob;
	if(visflux == "THINLAYER") {
		prob = new DiffusionThinLayer<1>(&m, diffcoeff, bvalue, rhs);
		startprob = new DiffusionThinLayer<1>(&m, diffcoeff, bvalue, rhs);
		std::cout << " Thin-layer gradients\n";
	} else if(visflux == "MODIFIEDAVERAGE") {
		prob = new DiffusionMA<1>(&m, diffcoeff, bvalue, rhs, reconst);
		startprob = new DiffusionMA<1>(&m, diffcoeff, bvalue, rhs, reconst);
		std::cout << " Modified average gradients\n";
	}
	else {
		prob = new DiffusionThinLayer<1>(&m, diffcoeff, bvalue, rhs);
		startprob = new DiffusionThinLayer<1>(&m, diffcoeff, bvalue, rhs);
		std::cout << " Thin-layer gradients\n";
	}

	Array2d<a_real> outputarr, dummy;
	
	if(timesteptype == "IMPLICIT") {
		SteadyBackwardEulerSolver<1> time(&m, prob, startprob, usestarter, initcfl, endcfl, rampstart, rampend, tolerance, maxiter, 
				lintol, linmaxiterstart, linmaxiterend, linsolver, firsttolerance, firstmaxiter, firstcfl);

		// computation
		time.solve();

		Matrix<a_real,Dynamic,Dynamic,RowMajor>& u = time.unknowns();

		for(int iel = 0; iel < m.gnelem(); iel++)
		{
			a_real rc[2]; rc[0] = 0; rc[1] = 0;
			for(int inode = 0; inode < m.gnnode(iel); inode++) {
				rc[0] += m.gcoords(m.ginpoel(iel,inode),0);
				rc[1] += m.gcoords(m.ginpoel(iel,inode),1);
			}
			rc[0] /= m.gnnode(iel); rc[1] /= m.gnnode(iel);
			a_real trueval = uexact(rc);
			err += (u(iel,0)-trueval)*(u(iel,0)-trueval)*m.garea(iel);
		}

		prob->postprocess_point(u, outputarr);
	}
	else {
		SteadyForwardEulerSolver<1> time(&m, prob, startprob, usestarter, tolerance, maxiter, initcfl,
				firsttolerance, firstmaxiter, firstcfl);

		// computation
		time.solve();

		Matrix<a_real,Dynamic,Dynamic,RowMajor>& u = time.unknowns();

		for(int iel = 0; iel < m.gnelem(); iel++)
		{
			a_real rc[2]; rc[0] = 0; rc[1] = 0;
			for(int inode = 0; inode < m.gnnode(iel); inode++) {
				rc[0] += m.gcoords(m.ginpoel(iel,inode),0);
				rc[1] += m.gcoords(m.ginpoel(iel,inode),1);
			}
			rc[0] /= m.gnnode(iel); rc[1] /= m.gnnode(iel);
			a_real trueval = uexact(rc);
			err += (u(iel,0)-trueval)*(u(iel,0)-trueval)*m.garea(iel);
		}

		prob->postprocess_point(u, outputarr);
	}

	delete prob;
	delete startprob;

	err = sqrt(err);
	double h = 1.0/sqrt(m.gnelem());
	cout << "Log of Mesh size and error are " << log10(h) << "  " << log10(err) << endl;
	
	string scaname[1] = {"some-quantity"}; string vecname;
	writeScalarsVectorToVtu_PointData(outf, m, outputarr, scaname, dummy, vecname);

	cout << "\n--------------- End --------------------- \n\n";
	return 0;
}
