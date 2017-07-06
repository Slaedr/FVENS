/** @file aodesolver.cpp
 * @brief Implements driver class(es) for solution of ODEs arising from Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016
 *
 * Observation: Increasing the number of B-SGS sweeps part way into the simulation 
 * does not help convergence.
 */

#include "aodesolver.hpp"

namespace acfd {

template<short nvars>
SteadyForwardEulerSolver<nvars>::SteadyForwardEulerSolver(const UMesh2dh *const mesh, Spatial<nvars> *const euler, Spatial<nvars> *const starterfv,
		const short use_starter, const double toler, const int maxits, const double cfl_n, const double ftoler, const int fmaxits, const double fcfl_n)

	: SteadySolver<nvars>(mesh, euler, starterfv, use_starter), tol(toler), maxiter(maxits), cfl(cfl_n), starttol(ftoler), startmaxiter(fmaxits), startcfl(fcfl_n)
{
	residual.resize(m->gnelem(),nvars);
	u.resize(m->gnelem(), nvars);
	dtm.setup(m->gnelem(), 1);
}

template<short nvars>
SteadyForwardEulerSolver<nvars>::~SteadyForwardEulerSolver()
{
}

template<short nvars>
void SteadyForwardEulerSolver<nvars>::solve()
{
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	if(usestarter == 1) {
		while(resi/initres > starttol && step < startmaxiter)
		{
#pragma omp parallel for simd default(shared)
			for(int iel = 0; iel < m->gnelem(); iel++) {
				for(int i = 0; i < nvars; i++)
					residual(iel,i) = 0;
			}

			// update residual
			starter->compute_residual(u, residual, true, dtm);

			a_real errmass = 0;

#pragma omp parallel default(shared)
			{
#pragma omp for simd
				for(int iel = 0; iel < m->gnelem(); iel++)
				{
					for(int i = 0; i < nvars; i++)
					{
						//uold(iel,i) = u(iel,i);
						u(iel,i) -= startcfl*dtm(iel) * 1.0/m->garea(iel)*residual(iel,i);
					}
				}

#pragma omp for simd reduction(+:errmass)
				for(int iel = 0; iel < m->gnelem(); iel++)
				{
					errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
				}
			} // end parallel region

			resi = sqrt(errmass);

			if(step == 0)
				initres = resi;

			if(step % 50 == 0)
				std::cout << "  SteadyForwardEulerSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;

			step++;
		}
		std::cout << "  SteadyForwardEulerSolver: solve(): Initial approximate solve done.\n";
		step = 0;
		resi = 100.0;
	}

	std::cout << "  SteadyForwardEulerSolver: solve(): Starting main solver.\n";
	while(resi/initres > tol && step < maxiter)
	{
#pragma omp parallel for simd default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++) {
			for(int i = 0; i < nvars; i++)
				residual(iel,i) = 0;
		}

		// update residual
		eul->compute_residual(u, residual, true, dtm);

		a_real errmass = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for simd
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					//uold(iel,i) = u(iel,i);
					u(iel,i) -= cfl*dtm(iel) * 1.0/m->garea(iel)*residual(iel,i);
				}
			}

#pragma omp for simd reduction(+:errmass)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
			}
		} // end parallel region

		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 50 == 0)
			std::cout << "  SteadyForwardEulerSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;

		step++;
	}
	//std::cout << residual << std::endl;
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);

	if(step == maxiter)
		std::cout << "! SteadyForwardEulerSolver: solve(): Exceeded max iterations!" << std::endl;
	std::cout << " SteadyForwardEulerSolver: solve(): Done, steps = " << step << std::endl << std::endl;
	std::cout << " SteadyForwardEulerSolver: solve(): Time taken by ODE solver:\n";
	std::cout << "                                   CPU time = " << cputime << ", wall time = " << walltime << std::endl << std::endl;
}


template <short nvars>
SteadyBackwardEulerSolver<nvars>::SteadyBackwardEulerSolver(const UMesh2dh*const mesh, Spatial<nvars> *const spatial, Spatial<nvars> *const starterfv, const short use_starter, 
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, const double lin_tol, const int linmaxiter_start, const int linmaxiter_end, std::string linearsolver, std::string precond,
		const unsigned short nbuildsweeps, const unsigned short napplysweeps,
		const double ftoler, const int fmaxits, const double fcfl_n)

	: SteadySolver<nvars>(mesh, spatial, starterfv, use_starter), cflinit(cfl_init), cflfin(cfl_fin), rampstart(ramp_start), rampend(ramp_end), tol(toler), maxiter(maxits), 
	lintol(lin_tol), linmaxiterstart(linmaxiter_start), linmaxiterend(linmaxiter_end), starttol(ftoler), startmaxiter(fmaxits), startcfl(fcfl_n)
{
	/* NOTE: the number of columns here MUST match the static number of columns, which is nvars. */
	residual.resize(m->gnelem(),nvars);
	u.resize(m->gnelem(), nvars);
	dtm.setup(m->gnelem(), 1);

	if(precond == "PSGS") {
		prec = new PointSGS<nvars>(mesh, napplysweeps);
		std::cout << " SteadyBackwardEulerSolver: Selected point SGS preconditioner.\n";
	}
	else if(precond == "BJ") {
		prec = new BlockJacobi<nvars>(mesh);
		std::cout << " SteadyBackwardEulerSolver: Selected Block Jacobi preconditioner.\n";
	}
	else if(precond == "BSGS") {
		prec = new BlockSGS<nvars>(mesh, napplysweeps);
		std::cout << " SteadyBackwardEulerSolver: Selected Block SGS preconditioner.\n";
	}
	else if(precond == "BILU0") {
		prec = new BILU0<nvars>(mesh, nbuildsweeps, napplysweeps);
		std::cout << " SteadyBackwardEulerSolver: Selected Block ILU0 preconditioner.\n";
	}
	else {
		prec = new NoPrec<nvars>(mesh);
		std::cout << " SteadyBackwardEulerSolver: No preconditioning will be applied.\n";
	}

	if(linearsolver == "BCGSTB") {
		linsolv = new BiCGSTAB<nvars>(mesh, prec);
		std::cout << " SteadyBackwardEulerSolver: BiCGSTAB solver selected.\n";
	}
	else {
		linsolv = new RichardsonSolver<nvars>(mesh, prec);
		std::cout << " SteadyBackwardEulerSolver: Richardson iteration selected, ie, no acceleration.\n";
	}

	D = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnelem()];
	L = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnaface()-m->gnbface()];
	U = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnaface()-m->gnbface()];
}

template <short nvars>
SteadyBackwardEulerSolver<nvars>::~SteadyBackwardEulerSolver()
{
	delete linsolv;
	delete prec;
	delete [] D;
	delete [] U;
	delete [] L;
}

template <short nvars>
void SteadyBackwardEulerSolver<nvars>::solve()
{
	double curCFL; int curlinmaxiter;
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;
	Matrix<a_real,Dynamic,Dynamic,RowMajor> du = Matrix<a_real,Dynamic,Dynamic,RowMajor>::Zero(m->gnelem(), nvars);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	if(usestarter == 1) {
		while(resi/initres > starttol && step < startmaxiter)
		{
#pragma omp parallel for default(shared)
			for(int iel = 0; iel < m->gnelem(); iel++) {
#pragma omp simd
				for(int i = 0; i < nvars; i++) {
					residual(iel,i) = 0;
					for(int j = 0; j < nvars; j++)
						D[iel](i,j) = 0;
				}
			}
#pragma omp parallel for default(shared)
			for(int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
#pragma omp simd
				for(int i = 0; i < nvars; i++)
					for(int j = 0; j < nvars; j++) {
						L[iface](i,j) = 0;
						U[iface](i,j) = 0;
					}
			}
			
			// update residual and local time steps
			starter->compute_residual(u, residual, true, dtm);

			starter->compute_jacobian(u, D, L, U);

			// add pseudo-time terms to diagonal blocks
#pragma omp parallel for simd default(shared)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
					D[iel](i,i) += m->garea(iel) / (startcfl*dtm(iel));
			}

			// setup and solve linear system for the update du
			linsolv->setLHS(D,L,U);
			linsolv->setParams(lintol, linmaxiterstart);
			int linstepsneeded = linsolv->solve(residual, du);

			a_real errmass = 0;

#pragma omp parallel default(shared)
			{
#pragma omp for
				for(int iel = 0; iel < m->gnelem(); iel++) {
					u.row(iel) += du.row(iel);
				}
#pragma omp for simd reduction(+:errmass)
				for(int iel = 0; iel < m->gnelem(); iel++)
				{
					errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
				}
			}

			resi = sqrt(errmass);

			if(step == 0)
				initres = resi;

			if(step % 10 == 0) {
				std::cout << "  SteadyBackwardEulerSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;
				std::cout << "      CFL = " << startcfl << ", Lin max iters = " << linmaxiterstart << ", iters used = " << linstepsneeded << std::endl;
			}

			step++;
		}

		std::cout << " SteadyBackwardEulerSolver: solve(): Initial solve done, steps = " << step << ", rel residual " << resi/initres << ".\n";
		step = 0;
		resi = 1.0;
		initres = 1.0;
	}

	std::cout << " SteadyBackwardEulerSolver: solve(): Starting main solver.\n";
	while(resi/initres > tol && step < maxiter)
	{
#pragma omp parallel for default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++) {
#pragma omp simd
			for(int i = 0; i < nvars; i++) {
				residual(iel,i) = 0;
				for(int j = 0; j < nvars; j++)
					D[iel](i,j) = 0;
			}
		}

#pragma omp parallel for default(shared)
		for(int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
#pragma omp simd
			for(int i = 0; i < nvars; i++)
				for(int j = 0; j < nvars; j++) {
					L[iface](i,j) = 0;
					U[iface](i,j) = 0;
				}
		}
		
		// update residual and local time steps
		eul->compute_residual(u, residual, true, dtm);

		eul->compute_jacobian(u, D, L, U);
		
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
#pragma omp parallel for simd default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++)
				D[iel](i,i) += m->garea(iel) / (curCFL*dtm(iel));
		}

		// setup and solve linear system for the update du
		linsolv->setLHS(D,L,U);
		linsolv->setParams(lintol, curlinmaxiter);
		int linstepsneeded = linsolv->solve(residual, du);

		a_real errmass = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for
			for(int iel = 0; iel < m->gnelem(); iel++) {
				u.row(iel) += du.row(iel);
			}
#pragma omp for simd reduction(+:errmass)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
			}
		}

		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 10 == 0) {
			std::cout << "  SteadyBackwardEulerSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;
			std::cout << "      CFL = " << curCFL << ", Lin max iters = " << linmaxiterstart << ", iters used = " << linstepsneeded << std::endl;
		}

		step++;
	}

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);

	if(step == maxiter)
		std::cout << "! SteadyBackwardEulerSolver: solve(): Exceeded max iterations!" << std::endl;
	std::cout << " SteadyBackwardEulerSolver: solve(): Done, steps = " << step << ", rel residual " << resi/initres << std::endl;

	double linwtime, linctime;
	linsolv->getRunTimes(linwtime, linctime);
	std::cout << "\n SteadyBackwardEulerSolver: solve(): Time taken by linear solver:\n";
	std::cout << " \t\tWall time = " << linwtime << ", CPU time = " << linctime << std::endl;
	std::cout << "\n SteadyBackwardEulerSolver: solve(): Time taken by ODE solver:" << std::endl;
	std::cout << " \t\tWall time = " << walltime << ", CPU time = " << cputime << std::endl << std::endl;
}

template <short nvars>
SteadyMFBackwardEulerSolver<nvars>::SteadyMFBackwardEulerSolver(const UMesh2dh*const mesh, Spatial<nvars> *const spatial, Spatial<nvars> *const starterfv, const short use_starter, 
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, const double lin_tol, const int linmaxiter_start, const int linmaxiter_end, std::string linearsolver, std::string precond,
		const unsigned short nbuildsweeps, const unsigned short napplysweeps,
		const double ftoler, const int fmaxits, const double fcfl_n)

	: SteadySolver<nvars>(mesh, spatial, starterfv, use_starter), cflinit(cfl_init), cflfin(cfl_fin), rampstart(ramp_start), rampend(ramp_end), tol(toler), maxiter(maxits), 
	lintol(lin_tol), linmaxiterstart(linmaxiter_start), linmaxiterend(linmaxiter_end), starttol(ftoler), startmaxiter(fmaxits), startcfl(fcfl_n)
{
	/* NOTE: the number of columns here MUST match the static number of columns, which is nvars. */
	std::cout << " Using matrix-free implicit solver.\n";
	residual.resize(m->gnelem(),nvars);
	u.resize(m->gnelem(), nvars);
	dtm.setup(m->gnelem(), 1);

	if(precond == "PSGS") {
		prec = new PointSGS<nvars>(mesh, napplysweeps);
		std::cout << " SteadyMFBackwardEulerSolver: Selected point SGS preconditioner.\n";
	}
	else if(precond == "BJ") {
		prec = new BlockJacobi<nvars>(mesh);
		std::cout << " SteadyMFBackwardEulerSolver: Selected Block Jacobi preconditioner.\n";
	}
	else if(precond == "BSGS") {
		prec = new BlockSGS<nvars>(mesh, napplysweeps);
		std::cout << " SteadyMFBackwardEulerSolver: Selected Block SGS preconditioner.\n";
	}
	else if(precond == "BILU0") {
		prec = new BILU0<nvars>(mesh, nbuildsweeps, napplysweeps);
		std::cout << " SteadyMFBackwardEulerSolver: Selected Block ILU0 preconditioner.\n";
	}
	else {
		prec = new NoPrec<nvars>(mesh);
		std::cout << " SteadyMFBackwardEulerSolver: No preconditioning will be applied.\n";
	}

	if(linearsolver == "BCGSTB") {
		//startlinsolv = new BiCGSTAB<nvars>(mesh, prec, starter);
		//linsolv = new BiCGSTAB<nvars>(mesh, prec, eul);
		std::cout << " SteadyMFBackwardEulerSolver: BiCGSTAB solver selected.\n";
	}
	else {
		startlinsolv = new MFRichardsonSolver<nvars>(mesh, prec, starter);
		linsolv = new MFRichardsonSolver<nvars>(mesh, prec, eul);
		std::cout << " SteadyMFBackwardEulerSolver: Richardson iteration selected, ie, no acceleration.\n";
	}

	D = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnelem()];
	L = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnaface()-m->gnbface()];
	U = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnaface()-m->gnbface()];
	aux.resize(m->gnelem(),nvars);
}

template <short nvars>
SteadyMFBackwardEulerSolver<nvars>::~SteadyMFBackwardEulerSolver()
{
	delete linsolv;
	delete startlinsolv;
	delete prec;
	delete [] D;
	delete [] U;
	delete [] L;
}

template <short nvars>
void SteadyMFBackwardEulerSolver<nvars>::solve()
{
	double curCFL; int curlinmaxiter;
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;
	Matrix<a_real,Dynamic,Dynamic,RowMajor> du = Matrix<a_real,Dynamic,Dynamic,RowMajor>::Zero(m->gnelem(), nvars);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	if(usestarter == 1) {
		while(resi/initres > starttol && step < startmaxiter)
		{
#pragma omp parallel for default(shared)
			for(int iel = 0; iel < m->gnelem(); iel++) {
#pragma omp simd
				for(int i = 0; i < nvars; i++) {
					residual(iel,i) = 0;
					for(int j = 0; j < nvars; j++)
						D[iel](i,j) = 0;
				}
			}
#pragma omp parallel for default(shared)
			for(int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
#pragma omp simd
				for(int i = 0; i < nvars; i++)
					for(int j = 0; j < nvars; j++) {
						L[iface](i,j) = 0;
						U[iface](i,j) = 0;
					}
			}
			
			// update residual and local time steps
			starter->compute_residual(u, residual, true, dtm);

			// compute first-order Jacobian for preconditioner
			starter->compute_jacobian(u, D, L, U);

			// add pseudo-time terms to diagonal blocks
#pragma omp parallel for simd default(shared)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
					D[iel](i,i) += m->garea(iel) / (startcfl*dtm(iel));
			}

			// setup and solve linear system for the update du
			startlinsolv->setLHS(D,L,U);
			startlinsolv->setParams(lintol, linmaxiterstart);
			int linstepsneeded = startlinsolv->solve(u, dtm, residual, aux, du);

			a_real errmass = 0;

#pragma omp parallel default(shared)
			{
#pragma omp for
				for(int iel = 0; iel < m->gnelem(); iel++) {
					u.row(iel) += du.row(iel);
				}
#pragma omp for simd reduction(+:errmass)
				for(int iel = 0; iel < m->gnelem(); iel++)
				{
					errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
				}
			}

			resi = sqrt(errmass);

			if(step == 0)
				initres = resi;

			if(step % 10 == 0) {
				std::cout << "  SteadyMFBackwardEulerSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;
				std::cout << "      CFL = " << startcfl << ", Lin max iters = " << linmaxiterstart << ", iters used = " << linstepsneeded << std::endl;
			}

			step++;
		}

		std::cout << " SteadyMFBackwardEulerSolver: solve(): Initial solve done, steps = " << step << ", rel residual " << resi/initres << ".\n";
		step = 0;
		resi = 1.0;
		initres = 1.0;
	}

	std::cout << " SteadyMFBackwardEulerSolver: solve(): Starting main solver.\n";
	while(resi/initres > tol && step < maxiter)
	{
#pragma omp parallel for default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++) {
#pragma omp simd
			for(int i = 0; i < nvars; i++) {
				residual(iel,i) = 0;
				for(int j = 0; j < nvars; j++)
					D[iel](i,j) = 0;
			}
		}

#pragma omp parallel for default(shared)
		for(int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
#pragma omp simd
			for(int i = 0; i < nvars; i++)
				for(int j = 0; j < nvars; j++) {
					L[iface](i,j) = 0;
					U[iface](i,j) = 0;
				}
		}
		
		// update residual and local time steps
		eul->compute_residual(u, residual, true, dtm);

		eul->compute_jacobian(u, D, L, U);
		
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
#pragma omp parallel for simd default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++)
				D[iel](i,i) += m->garea(iel) / (curCFL*dtm(iel));
		}

		// setup and solve linear system for the update du
		linsolv->setLHS(D,L,U);
		linsolv->setParams(lintol, curlinmaxiter);
		int linstepsneeded = linsolv->solve(u, dtm, residual, aux, du);

		a_real errmass = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for
			for(int iel = 0; iel < m->gnelem(); iel++) {
				u.row(iel) += du.row(iel);
			}
#pragma omp for simd reduction(+:errmass)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
			}
		}

		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 10 == 0) {
			std::cout << "  SteadyMFBackwardEulerSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;
			std::cout << "      CFL = " << curCFL << ", Lin max iters = " << linmaxiterstart << ", iters used = " << linstepsneeded << std::endl;
		}

		step++;
	}

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);

	if(step == maxiter)
		std::cout << "! SteadyMFBackwardEulerSolver: solve(): Exceeded max iterations!" << std::endl;
	std::cout << " SteadyMFBackwardEulerSolver: solve(): Done, steps = " << step << ", rel residual " << resi/initres << std::endl;

	double linwtime, linctime;
	linsolv->getRunTimes(linwtime, linctime);
	std::cout << "\n SteadyMFBackwardEulerSolver: solve(): Time taken by linear solver:\n";
	std::cout << " \t\tWall time = " << linwtime << ", CPU time = " << linctime << std::endl;
	std::cout << "\n SteadyMFBackwardEulerSolver: solve(): Time taken by ODE solver:" << std::endl;
	std::cout << " \t\tWall time = " << walltime << ", CPU time = " << cputime << std::endl << std::endl;
}


template class SteadyForwardEulerSolver<NVARS>;
template class SteadyBackwardEulerSolver<NVARS>;
template class SteadyMFBackwardEulerSolver<NVARS>;
template class SteadyForwardEulerSolver<1>;
template class SteadyBackwardEulerSolver<1>;

}	// end namespace
