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

namespace acfd {

template<short nvars>
SteadyForwardEulerSolver<nvars>::SteadyForwardEulerSolver(const UMesh2dh *const mesh, 
		Spatial<nvars> *const euler, Spatial<nvars> *const starterfv,const short use_starter, 
		const double toler, const int maxits, const double cfl_n, 
		const double ftoler, const int fmaxits, const double fcfl_n)

	: SteadySolver<nvars>(mesh, euler, starterfv, use_starter), 
	tol(toler), maxiter(maxits), cfl(cfl_n), 
	starttol(ftoler), startmaxiter(fmaxits), startcfl(fcfl_n)
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
			for(a_int iel = 0; iel < m->gnelem(); iel++) {
				for(short i = 0; i < nvars; i++)
					residual(iel,i) = 0;
			}

			// update residual
			starter->compute_residual(u, residual, true, dtm);

			a_real errmass = 0;

#pragma omp parallel default(shared)
			{
#pragma omp for simd
				for(a_int iel = 0; iel < m->gnelem(); iel++)
				{
					for(short i = 0; i < nvars; i++)
					{
						//uold(iel,i) = u(iel,i);
						u(iel,i) -= startcfl*dtm(iel) * 1.0/m->garea(iel)*residual(iel,i);
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
		}
		std::cout << "  SteadyForwardEulerSolver: solve(): Initial approximate solve done.\n";
		step = 0;
		resi = 100.0;
	}

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
	}
	//std::cout << residual << std::endl;
	
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
}

/// Ordering function for qsort - ascending order
int order_ascending(const void* a, const void* b) {
	if(*(a_int*)a < *(a_int*)b)
		return -1;
	else if(*(a_int*)a == *(a_int*)b)
		return 0;
	else
		return 1;
}

/** By default, the Jacobian is stored in a block sparse row format.
 */
template <short nvars>
SteadyBackwardEulerSolver<nvars>::SteadyBackwardEulerSolver(const UMesh2dh*const mesh, 
		Spatial<nvars> *const spatial, Spatial<nvars> *const starterfv, 
		const short use_starter, 
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, 
		const char mattype, const double lin_tol, 
		const int linmaxiter_start, const int linmaxiter_end, 
		std::string linearsolver, std::string precond,
		const short nbuildsweeps, const short napplysweeps,
		const double ftoler, const int fmaxits, const double fcfl_n)

	: SteadySolver<nvars>(mesh, spatial, starterfv, use_starter), A(nullptr), 
	cflinit(cfl_init), cflfin(cfl_fin), rampstart(ramp_start), rampend(ramp_end), 
	tol(toler), maxiter(maxits), 
	lintol(lin_tol), linmaxiterstart(linmaxiter_start), linmaxiterend(linmaxiter_end), 
	starttol(ftoler), startmaxiter(fmaxits), startcfl(fcfl_n)
{
	// NOTE: the number of columns here MUST match the static number of columns, which is nvars.
	residual.resize(m->gnelem(),nvars);
	u.resize(m->gnelem(), nvars);
	dtm.setup(m->gnelem(), 1);

	// set Jacobian storage
	if(mattype == 'd') {
		// DLU matrix
		A = new blasted::DLUMatrix<nvars>(m, nbuildsweeps, napplysweeps);
	}
	else if(mattype == 'p') {
		// construct non-zero structure for sparse format

		a_int* colinds, * rowptr;
		a_int nnz = (m->gnelem()+2*(m->gnaface()-m->gnbface()))*nvars*nvars;
		a_int nrows = m->gnelem()*nvars;

		rowptr = new a_int[nrows+1];
		colinds = new a_int[nnz];
		for(a_int i = 0; i < nrows+1; i++)
			rowptr[i] = nvars;

		for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
		{
			// each face represents an interaction between the two cells on either side
			// we add the nonzero block to browptr but in a row shifted by 1
			for(int i = 0; i < nvars; i++) {
				rowptr[m->gintfac(iface,0)*nvars+i + 1] += nvars;
				rowptr[m->gintfac(iface,1)*nvars+i + 1] += nvars;
			}
		}
		for(a_int i = 2; i < nrows+1; i++)
			rowptr[i] += rowptr[i-1];
		rowptr[0] = 0;

		if(rowptr[nrows] != nnz)
			std::cout << "! SteadyBackwardEulerSolver:Point: Row pointer computation is wrong!!\n";
#ifdef DEBUG
		std::cout << "  nnz = " << rowptr[nrows] << "; should be "<< nnz << std::endl;
#endif

		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			size_t bcsize = (rowptr[(iel+1)*nvars]-rowptr[iel*nvars])/(nvars*nvars);
			std::vector<a_int> bcinds;
			bcinds.reserve(bcsize);

			bcinds.push_back(iel);
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++) {
				a_int nbdelem = m->gesuel(iel,ifael);
				if(nbdelem < m->gnelem())
					bcinds.push_back(nbdelem);
			}

			if(bcinds.size() != bcsize)
				std::cout << "! SteadyBackwardEulerSolver:Point: Block-column indices are wrong!!\n";

			std::sort(bcinds.begin(),bcinds.end());

			// i th row in element iel's block-row
			for(size_t i = 0; i < nvars; i++)
				// j th block of the block-row: element iel and its neighbors
				for(size_t j = 0; j < bcinds.size(); j++)
					// k th column in the j th block
					for(int k = 0; k < nvars; k++)
						colinds[rowptr[iel*nvars+i] + j*nvars + k] = bcinds[j]*nvars+k;
		}

		// Create the point matrix
		A = new blasted::BSRMatrix<a_real,a_int,1> (
				nrows, colinds,rowptr, nbuildsweeps,napplysweeps );

		delete [] rowptr;
		delete [] colinds;
	}
	else {
		// construct non-zero structure for block sparse format

		a_int* bcolinds, * browptr;
		a_int nnzb = m->gnelem()+2*(m->gnaface()-m->gnbface());

		browptr = new a_int[m->gnelem()+1];
		bcolinds = new a_int[nnzb];
		for(a_int i = 0; i < m->gnelem()+1; i++)
			browptr[i] = 1;

		for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
		{
			// each face represents an interaction between the two cells on either side
			// we add the nonzero block, shifted by 1 position, to browptr
			browptr[m->gintfac(iface,0)+1] += 1;
			browptr[m->gintfac(iface,1)+1] += 1;
		}
		for(a_int i = 2; i < m->gnelem()+1; i++)
			browptr[i] += browptr[i-1];
		browptr[0] = 0;

		if(browptr[m->gnelem()] != nnzb)
			std::cout << "! SteadyBackwardEulerSolver: Row pointer computation is wrong!!\n";
#ifdef DEBUG
		std::cout << "  nnz = " << browptr[m->gnelem()] << "; should be "<< nnzb << std::endl;
#endif

		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			size_t csize = browptr[iel+1]-browptr[iel];
			std::vector<a_int> cinds;
			cinds.reserve(csize);

			cinds.push_back(iel);
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++) {
				a_int nbdelem = m->gesuel(iel,ifael);
				if(nbdelem < m->gnelem())
					cinds.push_back(nbdelem);
			}

			if(cinds.size() != csize)
				std::cout << "! SteadyBackwardEulerSolver: Block-column indices are wrong!!\n";

			std::sort(cinds.begin(),cinds.end());

			for(size_t i = 0; i < cinds.size(); i++)
				bcolinds[browptr[iel]+i] = cinds[i];
		}

		// Create the matrix
		A = new blasted::BSRMatrix<a_real,a_int,nvars> (
				m->gnelem(), bcolinds,browptr, nbuildsweeps,napplysweeps );
		delete [] browptr;
		delete [] bcolinds;
	}

	// select preconditioner
	if(precond == "BJ") {
		prec = new BlockJacobi<nvars>(A);
		std::cout << " SteadyBackwardEulerSolver: Selected Block Jacobi preconditioner.\n";
	}
	else if(precond == "BSGS") {
		prec = new BlockSGS<nvars>(A);
		std::cout << " SteadyBackwardEulerSolver: Selected Block SGS preconditioner.\n";
		A->allocTempVector();
	}
	else if(precond == "BILU0") {
		prec = new BILU0<nvars>(A);
		std::cout << " SteadyBackwardEulerSolver: Selected Block ILU0 preconditioner.\n";
		A->allocTempVector();
	}
	else {
		prec = new NoPrec<nvars>(A);
		std::cout << " SteadyBackwardEulerSolver: No preconditioning will be applied.\n";
	}

	if(linearsolver == "BCGSTB") {
		linsolv = new BiCGSTAB<nvars>(m, A, prec);
		std::cout << " SteadyBackwardEulerSolver: BiCGSTAB solver selected.\n";
	}
	else {
		linsolv = new RichardsonSolver<nvars>(mesh, A, prec);
		std::cout << " SteadyBackwardEulerSolver: Richardson iteration selected, no acceleration.\n";
	}
}

template <short nvars>
SteadyBackwardEulerSolver<nvars>::~SteadyBackwardEulerSolver()
{
	delete linsolv;
	delete prec;
	if(A)
		delete A;
}

template <short nvars>
void SteadyBackwardEulerSolver<nvars>::solve()
{
	double curCFL; int curlinmaxiter;
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;
	MVector du = MVector::Zero(m->gnelem(), nvars);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	if(usestarter == 1) {
		std::cout << " SteadyBackwardEulerSolver: Starting initialization run..\n";
		while(resi/initres > starttol && step < startmaxiter)
		{
#pragma omp parallel for default(shared)
			for(a_int iel = 0; iel < m->gnelem(); iel++) {
#pragma omp simd
				for(short i = 0; i < nvars; i++) {
					residual(iel,i) = 0;
				}
			}

			A->setDiagZero();
			
			// update residual and local time steps
			starter->compute_residual(u, residual, true, dtm);

			starter->compute_jacobian(u, A);

			// add pseudo-time terms to diagonal blocks
#pragma omp parallel for default(shared)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				Matrix<a_real,nvars,nvars,RowMajor> db 
					= Matrix<a_real,nvars,nvars,RowMajor>::Zero();

				for(short i = 0; i < nvars; i++)
					db(i,i) = m->garea(iel) / (startcfl*dtm(iel));
				
				A->updateDiagBlock(iel*nvars, db.data(), nvars);
			}

			// setup and solve linear system for the update du
			linsolv->setupPreconditioner();
			linsolv->setParams(lintol, linmaxiterstart);
			int linstepsneeded = linsolv->solve(residual, du);

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
				std::cout << "      CFL = " << startcfl << ", Lin max iters = " 
					<< linmaxiterstart << ", iters used = " << linstepsneeded << std::endl;
			}

			step++;
		}

		std::cout << " SteadyBackwardEulerSolver: solve(): Initial solve done, steps = " << step 
			<< ", rel residual " << resi/initres << ".\n";
		step = 0;
		resi = 1.0;
		initres = 1.0;
	}

	std::cout << " SteadyBackwardEulerSolver: solve(): Starting main solver.\n";
	unsigned int avglinsteps = 0;
	while(resi/initres > tol && step < maxiter)
	{
#pragma omp parallel for default(shared)
		for(a_int iel = 0; iel < m->gnelem(); iel++) {
#pragma omp simd
			for(short i = 0; i < nvars; i++) {
				residual(iel,i) = 0;
			}
		}

		A->setDiagZero();
		
		// update residual and local time steps
		eul->compute_residual(u, residual, true, dtm);

		eul->compute_jacobian(u, A);
		
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
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			Matrix<a_real,nvars,nvars,RowMajor> db 
				= Matrix<a_real,nvars,nvars,RowMajor>::Zero();

			for(short i = 0; i < nvars; i++)
				db(i,i) = m->garea(iel) / (curCFL*dtm(iel));
			
			A->updateDiagBlock(iel*nvars, db.data(), nvars);
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
			std::cout << "      CFL = " << curCFL << ", Lin max iters = " << linmaxiterstart 
				<< ", iters used = " << linstepsneeded << std::endl;
		}

		step++;
	}

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	avglinsteps /= step;

	if(step == maxiter)
		std::cout << "! SteadyBackwardEulerSolver: solve(): Exceeded max iterations!\n";
	std::cout << " SteadyBackwardEulerSolver: solve(): Done, steps = " << step << ", rel residual " 
		<< resi/initres << std::endl;

	double linwtime, linctime;
	linsolv->getRunTimes(linwtime, linctime);
	std::cout << "\n SteadyBackwardEulerSolver: solve(): Time taken by linear solver:\n";
	std::cout << " \t\tWall time = " << linwtime << ", CPU time = " << linctime << std::endl;
	std::cout << "\t\tAverage number of linear solver iterations = " << avglinsteps << std::endl;
	std::cout << "\n SteadyBackwardEulerSolver: solve(): Time taken by ODE solver:" << std::endl;
	std::cout << " \t\tWall time = " << walltime << ", CPU time = " << cputime << "\n\n";
}

template <short nvars>
SteadyMFBackwardEulerSolver<nvars>::SteadyMFBackwardEulerSolver(const UMesh2dh*const mesh, 
		Spatial<nvars> *const spatial, Spatial<nvars> *const starterfv, const short use_starter, 
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, 
		const double lin_tol, const int linmaxiter_start, const int linmaxiter_end, 
		std::string linearsolver, std::string precond,
		const short nbuildsweeps, const short napplysweeps,
		const double ftoler, const int fmaxits, const double fcfl_n)

	: SteadySolver<nvars>(mesh, spatial, starterfv, use_starter), 
	cflinit(cfl_init), cflfin(cfl_fin), rampstart(ramp_start), rampend(ramp_end), 
	tol(toler), maxiter(maxits), lintol(lin_tol), 
	linmaxiterstart(linmaxiter_start), linmaxiterend(linmaxiter_end), 
	starttol(ftoler), startmaxiter(fmaxits), startcfl(fcfl_n)
{
	// NOTE: the number of columns here MUST match the static number of columns, which is nvars.
	std::cout << " Using matrix-free implicit solver.\n";
	residual.resize(m->gnelem(),nvars);
	u.resize(m->gnelem(), nvars);
	dtm.setup(m->gnelem(), 1);

	if(precond == "BJ") {
		prec = new BlockJacobi<nvars>(M);
		std::cout << " SteadyMFBackwardEulerSolver: Selected Block Jacobi preconditioner.\n";
	}
	else if(precond == "BSGS") {
		prec = new BlockSGS<nvars>(M);
		std::cout << " SteadyMFBackwardEulerSolver: Selected Block SGS preconditioner.\n";
	}
	else if(precond == "BILU0") {
		prec = new BILU0<nvars>(M);
		std::cout << " SteadyMFBackwardEulerSolver: Selected Block ILU0 preconditioner.\n";
	}
	else {
		prec = new NoPrec<nvars>(M);
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
		std::cout << " SteadyMFBackwardEulerSolver: Richardson solver selected, no acceleration.\n";
	}

	aux.resize(m->gnelem(),nvars);
}

template <short nvars>
SteadyMFBackwardEulerSolver<nvars>::~SteadyMFBackwardEulerSolver()
{
	delete linsolv;
	delete startlinsolv;
	delete prec;
}

/// \fixme FIXME: Broken by matrices storage changes
template <short nvars>
void SteadyMFBackwardEulerSolver<nvars>::solve()
{
	double curCFL; int curlinmaxiter;
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;
	MVector du = MVector::Zero(m->gnelem(), nvars);
	du(0,0) = 1e-8;
	
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
				for(short i = 0; i < nvars; i++) {
					residual(iel,i) = 0;
					for(short j = 0; j < nvars; j++)
						D[iel](i,j) = 0;
				}
			}
#pragma omp parallel for default(shared)
			for(int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
#pragma omp simd
				for(short i = 0; i < nvars; i++)
					for(short j = 0; j < nvars; j++) {
						L[iface](i,j) = 0;
						U[iface](i,j) = 0;
					}
			}
			
			// update residual and local time steps
			starter->compute_residual(u, residual, true, dtm);

			// compute first-order Jacobian for preconditioner
			starter->compute_jacobian(u, M);

			// add pseudo-time terms to diagonal blocks
#pragma omp parallel for simd default(shared)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(short i = 0; i < nvars; i++)
					D[iel](i,i) += m->garea(iel) / (startcfl*dtm(iel));
			}

			// setup and solve linear system for the update du
			startlinsolv->setupPreconditioner();
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

			//if(step % 10 == 0) {
				std::cout << "  SteadyMFBackwardEulerSolver: solve(): Step " << step 
					<< ", rel residual " << resi/initres << std::endl;
				std::cout << "      CFL = " << startcfl << ", Lin max iters = " 
					<< linmaxiterstart << ", iters used = " << linstepsneeded << std::endl;
			//}

			step++;
		}

		std::cout << " SteadyMFBackwardEulerSolver: solve(): Initial solve done, steps = " 
			<< step << ", rel residual " << resi/initres << ".\n";
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
			for(short i = 0; i < nvars; i++) {
				residual(iel,i) = 0;
				for(short j = 0; j < nvars; j++)
					D[iel](i,j) = 0;
			}
		}

#pragma omp parallel for default(shared)
		for(int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
#pragma omp simd
			for(short i = 0; i < nvars; i++)
				for(int j = 0; j < nvars; j++) {
					L[iface](i,j) = 0;
					U[iface](i,j) = 0;
				}
		}
		
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
#pragma omp parallel for simd default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(short i = 0; i < nvars; i++)
				D[iel](i,i) += m->garea(iel) / (curCFL*dtm(iel));
		}

		// setup and solve linear system for the update du
		linsolv->setupPreconditioner();
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
			std::cout << "  SteadyMFBackwardEulerSolver: solve(): Step " << step 
				<< ", rel residual " << resi/initres << std::endl;
			std::cout << "      CFL = " << curCFL << ", Lin max iters = " << linmaxiterstart 
				<< ", iters used = " << linstepsneeded << std::endl;
		}

		step++;
	}

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);

	if(step == maxiter)
		std::cout << "! SteadyMFBackwardEulerSolver: solve(): Exceeded max iterations!\n";
	std::cout << " SteadyMFBackwardEulerSolver: solve(): Done, steps = " << step 
		<< ", rel residual " << resi/initres << std::endl;

	double linwtime, linctime;
	linsolv->getRunTimes(linwtime, linctime);
	std::cout << "\n SteadyMFBackwardEulerSolver: solve(): Time taken by linear solver:\n";
	std::cout << " \t\tWall time = " << linwtime << ", CPU time = " << linctime << std::endl;
	std::cout << "\n SteadyMFBackwardEulerSolver: solve(): Time taken by ODE solver:" << std::endl;
	std::cout << " \t\tWall time = " << walltime << ", CPU time = " << cputime << "\n\n";
}


template class SteadyForwardEulerSolver<NVARS>;
template class SteadyBackwardEulerSolver<NVARS>;
template class SteadyMFBackwardEulerSolver<NVARS>;
template class SteadyForwardEulerSolver<1>;
template class SteadyBackwardEulerSolver<1>;

}	// end namespace
