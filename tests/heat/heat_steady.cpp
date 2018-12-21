#include <iostream>
#include <fstream>
#include "spatial/diffusion.hpp"
#include "linalg/alinalg.hpp"
#include "spatial/aoutput.hpp"
#include "ode/aodesolver.hpp"
#include "utilities/aoptionparser.hpp"
#include "utilities/controlparser.hpp"
#include "utilities/afactory.hpp"
#include "utilities/casesolvers.hpp"
#include "mesh/ameshutils.hpp"

#undef NDEBUG

using namespace std;
using namespace fvens;

int main(int argc, char* argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Finite volume solver for the heat equation.\n\
		Arguments needed: FVENS control file and PETSc options file with -options_file,\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);

	if(argc < 2)
	{
		cout << "Please give a control file name.\n";
		return -1;
	}

	// Read control file
	ifstream control(argv[1]);

	string dum, meshprefix, outf, logfile, visflux, reconst, timesteptype, lognresstr;
	double initcfl, endcfl, tolerance, firstinitcfl, firstendcfl, firsttolerance, diffcoeff, bvalue;
	int maxiter, rampstart, rampend, firstmaxiter, firstrampstart, firstrampend, nmesh;
	short inittype, usestarter;
	bool lognres;

	control >> dum; control >> meshprefix;
	control >> dum; control >> nmesh;
	control >> dum; control >> outf;
	control >> dum; control >> logfile;
	control >> dum; control >> lognresstr;
	control >> dum;
	control >> dum; control >> diffcoeff;
	control >> dum; control >> bvalue;
	control >> dum; control >> inittype;
	control >> dum;
	control >> dum; control >> visflux;
	control >> dum; control >> reconst;
	control >> dum; control >> timesteptype;
	control >> dum; control >> initcfl >> endcfl;
	control >> dum; control >> rampstart >> rampend;
	control >> dum; control >> tolerance;
	control >> dum; control >> maxiter;
	control >> dum;
	control >> dum; control >> usestarter;
	control >> dum; control >> firstinitcfl >> firstendcfl;
	control >> dum; control >> firstrampstart >> firstrampend;
	control >> dum; control >> firsttolerance;
	control >> dum; control >> firstmaxiter;
	control.close();

	if(lognresstr == "YES")
		lognres = true;
	else
		lognres = false;

	if(nmesh < 2) {
		std::cout << "Need at least 2 grids for grid convergence test!\n";
		std::abort();
	}
	std::cout << "Diffusion coeff = " << diffcoeff << ", boundary value = " << bvalue << std::endl;
	
	// rhs and exact soln
	
	std::function<void(const a_real *const, const a_real, const a_real *const, a_real *const)> rhs 
		= [diffcoeff](const a_real *const r, const a_real t, const a_real *const u, 
				a_real *const sourceterm)
		{ 
			sourceterm[0] = diffcoeff*8.0*PI*PI*sin(2*PI*r[0])*sin(2*PI*r[1]); 
		};
	
	auto uexact = [](const a_real *const r)->a_real { return sin(2*PI*r[0])*sin(2*PI*r[1]); };

	std::vector<double> lh(nmesh), lerrors(nmesh), slopes(nmesh-1);

	for(int imesh = 0; imesh < nmesh; imesh++)
	{
		std::string meshfile = meshprefix + std::to_string(imesh) + ".msh";

		const UMesh2dh<a_real> m = constructMesh(meshfile);

		// set up problem
		
		std::cout << "Setting up spatial scheme.\n";
		
		Diffusion<1>* prob = nullptr;
		Diffusion<1>* startprob = nullptr;
		if(visflux == "MODIFIEDAVERAGE") {
			prob = new DiffusionMA<1>(&m, diffcoeff, bvalue, rhs, reconst);
			if(usestarter != 0)
				startprob = new DiffusionMA<1>(&m, diffcoeff, bvalue, rhs, "NONE");
		}
		else {
			std::cout << " ! Viscous scheme not available!\n";
			std::abort();
		}

		std::cout << "***\n";

		// solution vector
		Vec u;
		ierr = createGhostedSystemVector(&m, 1, &u); CHKERRQ(ierr);
		ierr = VecSet(u, 0); CHKERRQ(ierr);

		// Initialize Jacobian for implicit schemes
		Mat M;
		ierr = setupSystemMatrix<1>(&m, &M); CHKERRQ(ierr);

		// initialize solver
		KSP ksp;
		ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
		ierr = KSPSetOperators(ksp, M, M); CHKERRQ(ierr);
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		const SteadySolverConfig tconf {
			lognres, logfile,
			initcfl, endcfl, rampstart, rampend,
			tolerance, maxiter
		};

		const SteadySolverConfig startconf {
			lognres, logfile,
			firstinitcfl, firstendcfl, firstrampstart, firstrampend,
			firsttolerance, firstmaxiter
		};

		SteadySolver<1> *time = nullptr;
		SteadySolver<1> *starttime = nullptr;

		// This is bad design, but the nonlinear update scheme factory needs a flow options struct
		//  to build an object, because one update scheme is specific to flow cases.
		//  Here we just supply the string option to select the trivial update scheme.
		FlowParserOptions opts;
		opts.nl_update_scheme = "FULL";
		const NonlinearUpdate<1> *const nlu = create_const_nonlinearUpdateScheme<1>(opts);
		
		if(timesteptype == "IMPLICIT") 
		{
			time = new SteadyBackwardEulerSolver<1>(prob, tconf, ksp, nlu);

			if(usestarter != 0)
			{
				std::cout << "Starting initialization solve..\n";
				starttime = new SteadyBackwardEulerSolver<1>(startprob, startconf, ksp, nlu);

				// solve the starter problem to get the initial solution
				ierr = starttime->solve(u); CHKERRQ(ierr);

				delete starttime;
			}

			/* Solve the main problem using either the initial solution
			 * set by initializeUnknowns or the one computed by the starter problem.
			 */
			std::cout << "Starting main solve..\n";
			ierr = time->solve(u); CHKERRQ(ierr);
		}
		else {
			time = new SteadyForwardEulerSolver<1>(prob, u, tconf);

			if(usestarter != 0)
			{
				starttime = new SteadyForwardEulerSolver<1>(startprob, u, startconf);

				// solve the starter problem to get the initial solution
				ierr = starttime->solve(u); CHKERRQ(ierr);

				delete starttime;
			}

			/* Solve the main problem using either the initial solution
			 * set by initializeUnknowns or the one computed by the starter problem.
			 */
			ierr = time->solve(u); CHKERRQ(ierr);
		}

		// postprocess

		const a_real *uarr;
		ierr = VecGetArrayRead(u, &uarr); CHKERRQ(ierr);
		a_real err = 0;
		for(int iel = 0; iel < m.gnelem(); iel++)
		{
			a_real rc[NDIM];
			for(int j = 0; j < NDIM; j++)
				rc[j] = 0;
			for(int inode = 0; inode < m.gnnode(iel); inode++) {
				for(int j = 0; j < NDIM; j++)
					rc[j] += m.gcoords(m.ginpoel(iel,inode),j);
			}
			for(int j = 0; j < NDIM; j++)
				rc[j] /= m.gnnode(iel);

			const a_real trueval = uexact(rc);
			err += (uarr[iel]-trueval)*(uarr[iel]-trueval)*m.garea(iel);
		}
		ierr = VecRestoreArrayRead(u, &uarr); CHKERRQ(ierr);

		MPI_Allreduce(&err, &err, 1, FVENS_MPI_REAL, MPI_SUM, PETSC_COMM_WORLD);

		err = sqrt(err);
		double h = 1.0/sqrt(m.gnelem());
		cout << "Log of Mesh size and error are " << log10(h) << "  " << log10(err) << endl;
		lh[imesh] = log10(h);
		lerrors[imesh] = log10(err);
		if(imesh > 0)
			slopes[imesh-1] = (lerrors[imesh]-lerrors[imesh-1])/(lh[imesh]-lh[imesh-1]);

		delete prob;
		if(usestarter != 0)
			delete startprob;
		delete time;

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
		ierr = VecDestroy(&u); CHKERRQ(ierr);
		ierr = MatDestroy(&M); CHKERRQ(ierr);
		std::cout << "***\n";
	}

	std::cout << ">> Spatial orders = \n" ;
	for(int i = 0; i < nmesh-1; i++)
		std::cout << "   " << slopes[i] << std::endl;
		
	if(reconst == "LEASTSQUARES") 
	{
		if(slopes[nmesh-2] <= 2.1 && slopes[nmesh-2] >= 1.9)
			std::cout << "Passed!\n";
		else
			throw "Order not correct!";
	}
	
	cout << "\n--------------- End --------------------- \n\n";
	ierr = PetscFinalize(); CHKERRQ(ierr);
	return ierr;
}
