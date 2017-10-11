/** @file aodesolver.cpp
 * @brief Implements driver class(es) for solution of ODEs arising from 
 * Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016
 *
 * Observation: Increasing the number of B-SGS sweeps part way into the simulation 
 * does not help convergence.
 */

#include "aodesolver.hpp"
#include <blockmatrices.hpp>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace acfd {

template<short nvars>
SteadyForwardEulerSolver<nvars>::SteadyForwardEulerSolver(const UMesh2dh *const mesh, 
		Spatial<nvars> *const euler, MVector& soln,
		const double toler, const int maxits, const double cfl_n, 
		const bool use_implicitSmoothing, const LinearOperator<a_real,a_int> *const A,
		bool lognlres)

	: SteadySolver<nvars>(mesh, euler, soln, lognlres), 
	tol{toler}, maxiter{maxits}, cfl{cfl_n}, useImplicitSmoothing{use_implicitSmoothing}, M{A},
	linsolv{nullptr}, prec{nullptr}
{
	residual.resize(m->gnelem(),nvars);
	dtm.setup(m->gnelem(), 1);
	
	if(useImplicitSmoothing) {
		prec = new SGS<nvars>(M);
		std::cout << " SteadyForwardEulerSolver: Selected ";
		std::cout << "SGS preconditioner.\n";
		linsolv = new RichardsonSolver<nvars>(mesh, M, prec);
		std::cout << " SteadyForwardEulerSolver: Richardson iteration selected.\n";

		double lintol = 1e-1; int linmaxiter = 1;
		linsolv->setupPreconditioner();
		linsolv->setParams(lintol, linmaxiter);
	}
}

template<short nvars>
SteadyForwardEulerSolver<nvars>::~SteadyForwardEulerSolver()
{
	delete linsolv;
	delete prec;
}

template<short nvars>
void SteadyForwardEulerSolver<nvars>::solve(std::string logfile)
{
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;

	std::ofstream convout;
	if(lognres)
		convout.open(logfile+".conv", std::ofstream::app);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	std::cout << "  SteadyForwardEulerSolver: solve(): Starting main solver.\n";
	while(resi/initres > tol && step < maxiter)
	{
#pragma omp parallel for simd default(shared)
		for(a_int iel = 0; iel < m->gnelem(); iel++) {
			for(short i = 0; i < nvars; i++)
				residual(iel,i) = 0;
		}

		// update residual
		eul->compute_residual(u, residual, true, dtm);

		a_real errmass = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for simd
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				for(short i = 0; i < nvars; i++)
				{
					//uold(iel,i) = u(iel,i);
					u(iel,i) -= cfl*dtm(iel) * 1.0/m->garea(iel)*residual(iel,i);
				}
			}

#pragma omp for simd reduction(+:errmass)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
			}
		} // end parallel region

		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 50 == 0)
			std::cout << "  SteadyForwardEulerSolver: solve(): Step " << step 
				<< ", rel residual " << resi/initres << std::endl;

		step++;
		if(lognres)
			convout << step << " " << std::setw(10) << resi/initres << '\n';
	}

	if(lognres)
		convout.close();
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);

	if(step == maxiter)
		std::cout << "! SteadyForwardEulerSolver: solve(): Exceeded max iterations!\n";
	std::cout << " SteadyForwardEulerSolver: solve(): Done, steps = " << step << "\n\n";
	std::cout << " SteadyForwardEulerSolver: solve(): Time taken by ODE solver:\n";
	std::cout << "                                   CPU time = " << cputime 
		<< ", wall time = " << walltime << std::endl << std::endl;

	// append data to log file
	int numthreads = 0;
#ifdef _OPENMP
	numthreads = omp_get_max_threads();
#endif
	std::ofstream outf; outf.open(logfile, std::ofstream::app);
	outf << "\t" << numthreads << "\t" << walltime << "\t" << cputime << "\n";
	outf.close();
}

/** By default, the Jacobian is stored in a block sparse row format.
 */
template <short nvars>
SteadyBackwardEulerSolver<nvars>::SteadyBackwardEulerSolver(const UMesh2dh*const mesh, 
		Spatial<nvars> *const spatial, MVector& soln, LinearOperator<a_real,a_int> *const pmat,
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, 
		const double lin_tol, const int linmaxiter_start, const int linmaxiter_end, 
		std::string linearsolver, std::string precond,
		const int mrestart, bool lognlres)

	: SteadySolver<nvars>(mesh, spatial, soln, lognlres), M{pmat}, 
	cflinit{cfl_init}, cflfin{cfl_fin}, rampstart{ramp_start}, rampend{ramp_end}, 
	tol{toler}, maxiter{maxits}, 
	lintol{lin_tol}, linmaxiterstart{linmaxiter_start}, linmaxiterend{linmaxiter_end}
{
	// NOTE: the number of columns here MUST match the static number of columns, which is nvars.
	residual.resize(m->gnelem(),nvars);
	dtm.setup(m->gnelem(), 1);

	// select preconditioner
	if(precond == "J") {
		prec = new Jacobi<nvars>(M);
		std::cout << " SteadyBackwardEulerSolver: Selected ";
		std::cout << "Jacobi preconditioner.\n";
	}
	else if(precond == "SGS") {
		prec = new SGS<nvars>(M);
		std::cout << " SteadyBackwardEulerSolver: Selected ";
		std::cout << "SGS preconditioner.\n";
	}
	else if(precond == "ILU0") {
		prec = new ILU0<nvars>(M);
		std::cout << " SteadyBackwardEulerSolver: Selected ";
		std::cout << " ILU0 preconditioner.\n";
	}
	else {
		prec = new NoPrec<nvars>(M);
		std::cout << " SteadyBackwardEulerSolver: No preconditioning will be applied.\n";
	}

	if(linearsolver == "BCGSTB") {
		linsolv = new BiCGSTAB<nvars>(m, M, prec);
		std::cout << " SteadyBackwardEulerSolver: BiCGStab solver selected.\n";
	}
	else if(linearsolver == "GMRES") {
		linsolv = new GMRES<nvars>(m, M, prec, mrestart);
		std::cout << " SteadyBackwardEulerSolver: GMRES solver selected, restart after " 
			<< mrestart << " iterations\n";
	}
	else {
		linsolv = new RichardsonSolver<nvars>(mesh, M, prec);
		std::cout << " SteadyBackwardEulerSolver: Richardson iteration selected, no acceleration.\n";
	}
}

template <short nvars>
SteadyBackwardEulerSolver<nvars>::~SteadyBackwardEulerSolver()
{
	delete linsolv;
	delete prec;
}

template <short nvars>
void SteadyBackwardEulerSolver<nvars>::solve(std::string logfile)
{
	double curCFL; int curlinmaxiter;
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;
	MVector du = MVector::Zero(m->gnelem(), nvars);

	std::ofstream convout;
	if(lognres)
		convout.open(logfile+".conv", std::ofstream::app);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	unsigned int avglinsteps = 0;

	walltime = cputime = 0;
	linsolv->resetRunTimes();

	std::cout << " SteadyBackwardEulerSolver: solve(): Starting solver.\n";
	while(resi/initres > tol && step < maxiter)
	{
#pragma omp parallel for default(shared)
		for(a_int iel = 0; iel < m->gnelem(); iel++) {
#pragma omp simd
			for(short i = 0; i < nvars; i++) {
				residual(iel,i) = 0;
			}
		}

		M->setAllZero();
		
		// update residual and local time steps
		eul->compute_residual(u, residual, true, dtm);

		eul->compute_jacobian(u, M);
		
		// compute ramped quantities
		if(step < rampstart) {
			curCFL = cflinit;
			curlinmaxiter = linmaxiterstart;
		}
		else if(step < rampend) {
			if(rampend-rampstart <= 0) {
				curCFL = cflfin;
				curlinmaxiter = linmaxiterend;
			}
			else {
				double slopec = (cflfin-cflinit)/(rampend-rampstart);
				curCFL = cflinit + slopec*(step-rampstart);
				double slopei = double(linmaxiterend-linmaxiterstart)/(rampend-rampstart);
				curlinmaxiter = int(linmaxiterstart + slopei*(step-rampstart));
			}
		}
		else {
			curCFL = cflfin;
			curlinmaxiter = linmaxiterend;
		}

		// add pseudo-time terms to diagonal blocks
#pragma omp parallel for default(shared)
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			Matrix<a_real,nvars,nvars,RowMajor> db 
				= Matrix<a_real,nvars,nvars,RowMajor>::Zero();

			for(short i = 0; i < nvars; i++)
				db(i,i) = m->garea(iel) / (curCFL*dtm(iel));
			
			M->updateDiagBlock(iel*nvars, db.data(), nvars);
		}

		// setup and solve linear system for the update du
		linsolv->setupPreconditioner();
		linsolv->setParams(lintol, curlinmaxiter);
		int linstepsneeded = linsolv->solve(residual, du);

		avglinsteps += linstepsneeded;
		a_real errmass = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for
			for(a_int iel = 0; iel < m->gnelem(); iel++) {
				u.row(iel) += du.row(iel);
			}
#pragma omp for simd reduction(+:errmass)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
			}
		}

		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 10 == 0) {
			std::cout << "  SteadyBackwardEulerSolver: solve(): Step " << step 
				<< ", rel residual " << resi/initres << std::endl;
			std::cout << "      CFL = " << curCFL << ", Lin max iters = " << curlinmaxiter 
				<< ", iters used = " << linstepsneeded << std::endl;
		}

		step++;
			
		if(lognres)
			convout << step << " " << std::setw(10)  << resi/initres << '\n';
	}

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	avglinsteps /= step;

	if(lognres)
		convout.close();

	if(step == maxiter)
		std::cout << "! SteadyBackwardEulerSolver: solve(): Exceeded max iterations!\n";
	std::cout << " SteadyBackwardEulerSolver: solve(): Done, steps = " << step << ", rel residual " 
		<< resi/initres << std::endl;

	// print timing data
	double linwtime, linctime;
	linsolv->getRunTimes(linwtime, linctime);
	std::cout << "\n SteadyBackwardEulerSolver: solve(): Time taken by linear solver:\n";
	std::cout << " \t\tWall time = " << linwtime << ", CPU time = " << linctime << std::endl;
	std::cout << "\t\tAverage number of linear solver iterations = " << avglinsteps << std::endl;
	std::cout << "\n SteadyBackwardEulerSolver: solve(): Time taken by ODE solver:" << std::endl;
	std::cout << " \t\tWall time = " << walltime << ", CPU time = " << cputime << "\n\n";

	// append data to log file
	int numthreads = 0;
#ifdef _OPENMP
	numthreads = omp_get_max_threads();
#endif
	std::ofstream outf; outf.open(logfile, std::ofstream::app);
	outf << std::setw(10) << m->gnelem() << " "
		<< std::setw(6) << numthreads << " " << std::setw(10) << linwtime << " " 
		<< std::setw(10) << linctime << " " << std::setw(10) << avglinsteps << " "
		<< std::setw(10) << step
		<< "\n";
	outf.close();
}

template class SteadyForwardEulerSolver<NVARS>;
template class SteadyBackwardEulerSolver<NVARS>;
template class SteadyForwardEulerSolver<1>;
template class SteadyBackwardEulerSolver<1>;

}	// end namespace
