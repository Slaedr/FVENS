/** @file aodesolver.cpp
 * @brief Implements driver class(es) for solution of ODEs arising from PDE discretizations
 * @author Aditya Kashi
 * @date Feb 24, 2016
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "aodesolver.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <petscksp.h>
#include <petsctime.h>

namespace acfd {

/// Returns an array containing TVD Runge-Kutta coefficients for high-order accuracy
static Matrix<a_real,Dynamic,Dynamic> initialize_TVDRK_Coeffs(const int _order) 
{
	Matrix<a_real,Dynamic,Dynamic> tvdrk(_order,3);
	if(_order == 1) {
		tvdrk(0,0) = 1.0;	tvdrk(0,1) = 0.0; tvdrk(0,2) = 1.0;
	}
	else if(_order == 2) {
		tvdrk(0,0) = 1.0;	tvdrk(0,1) = 0.0; tvdrk(0,2) = 1.0;
		tvdrk(1,0) = 0.5;	tvdrk(1,1) = 0.5;	tvdrk(1,2) = 0.5;
	}
	else if(_order == 3)
	{
		tvdrk(0,0) = 1.0;      tvdrk(0,1) = 0.0;  tvdrk(0,2) = 1.0;
		tvdrk(1,0) = 0.75;	    tvdrk(1,1) = 0.25; tvdrk(1,2) = 0.25;
		tvdrk(2,0) = 0.3333333333333333; 
		tvdrk(2,1) = 0.6666666666666667; 
		tvdrk(2,2) = 0.6666666666666667;
	}
	else {
		std::cout << "! Temporal order " << _order  << " not available!\n";
	}
	return tvdrk;
}

template<int nvars>
SteadyForwardEulerSolver<nvars>::SteadyForwardEulerSolver(
		const Spatial<nvars> *const spatial, const Vec uvec,
		const SteadySolverConfig& conf)

	: SteadySolver<nvars>(spatial, conf)
{
	const UMesh2dh *const m = space->mesh();
	dtm.resize(m->gnelem(), 0);

	StatusCode ierr = VecDuplicate(uvec, &rvec);
	if(ierr) {
		std::cout << "! SteadyForwardEulerSolver: Could not create residual vector!\n";
		std::abort();
	}
}

template<int nvars>
SteadyForwardEulerSolver<nvars>::~SteadyForwardEulerSolver()
{
	int ierr = VecDestroy(&rvec);
	if(ierr)
		std::cout << "! SteadyForwardEulerSolver: Could not destroy residual vector!\n";
}

template<int nvars>
StatusCode SteadyForwardEulerSolver<nvars>::solve(Vec uvec)
{
	StatusCode ierr = 0;
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);

	const UMesh2dh *const m = space->mesh();

	if(config.maxiter <= 0) {
		std::cout << " SteadyForwardEulerSolver: solve(): No iterations to be done.\n";
		return ierr;
	}

	PetscInt locnelem; PetscScalar *uarr; PetscScalar *rarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ierr = VecGetArray(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<MVector> u(uarr, locnelem, nvars);
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);
	Eigen::Map<MVector> residual(rarr, locnelem, nvars);

	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;

	std::ofstream convout;
	if(mpirank==0)
		if(config.lognres)
			convout.open(config.logfile+".conv", std::ofstream::app);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	while(resi/initres > config.tol && step < config.maxiter)
	{
#pragma omp parallel for simd default(shared)
		for(a_int iel = 0; iel < m->gnelem(); iel++) {
			for(int i = 0; i < nvars; i++)
				residual(iel,i) = 0;
		}

		// update residual
		space->compute_residual(uvec, rvec, true, dtm);

		a_real errmass = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for simd
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) -= config.cflinit*dtm[iel] * 1.0/m->garea(iel)*residual(iel,i);
				}
			}

#pragma omp for simd reduction(+:errmass)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				errmass += residual(iel,nvars-1)*residual(iel,nvars-1)*m->garea(iel);
			}
		} // end parallel region

		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 50 == 0)
			if(mpirank==0)
				std::cout << "  SteadyForwardEulerSolver: solve(): Step " << step 
					<< ", rel residual " << resi/initres << std::endl;

		step++;
		if(mpirank==0)
			if(config.lognres)
				convout << step << " " << std::setw(10) << resi/initres << '\n';
	}

	if(mpirank==0)
		if(config.lognres)
			convout.close();
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);

	if(step == config.maxiter && mpirank == 0)
		std::cout << "! SteadyForwardEulerSolver: solve(): Exceeded max iterations!\n";
	if(mpirank == 0) {
		std::cout << " SteadyForwardEulerSolver: solve(): Done, steps = " << step << "\n\n";
		std::cout << " SteadyForwardEulerSolver: solve(): Time taken by ODE solver:\n";
		std::cout << "                                   CPU time = " << cputime 
			<< ", wall time = " << walltime << std::endl << std::endl;
	}

	// append data to log file
	if(mpirank==0) {
		int numthreads = 0;
#ifdef _OPENMP
		numthreads = omp_get_max_threads();
#endif
		std::ofstream outf; outf.open(config.logfile, std::ofstream::app);
		outf << "\t" << numthreads << "\t" << walltime << "\t" << cputime << "\n";
		outf.close();
	}
	
	ierr = VecRestoreArray(uvec, &uarr); CHKERRQ(ierr);
	ierr = VecRestoreArray(rvec, &rarr); CHKERRQ(ierr);
	ierr = VecDestroy(&rvec); CHKERRQ(ierr);
	return ierr;
}

/** By default, the Jacobian is stored in a block sparse row format.
 */
template <int nvars>
SteadyBackwardEulerSolver<nvars>::SteadyBackwardEulerSolver(
		const Spatial<nvars> *const spatial, 
		const SteadySolverConfig& conf,	
		KSP ksp)

	: SteadySolver<nvars>(spatial, conf), solver{ksp}
{
	const UMesh2dh *const m = space->mesh();
	dtm.resize(m->gnelem(), 0);
	Mat M; int ierr;
	ierr = KSPGetOperators(solver, &M, NULL);
	ierr = MatCreateVecs(M, &duvec, &rvec);
	if(ierr)
		std::cout << "! SteadyBackwardEulerSolver: Could not create residual or update vector!\n";
}

template <int nvars>
SteadyBackwardEulerSolver<nvars>::~SteadyBackwardEulerSolver()
{
	int ierr = VecDestroy(&rvec);
	if(ierr)
		std::cout << "! SteadyBackwardEulerSolver: Could not destroy residual vector!\n";
	ierr = VecDestroy(&duvec);
	if(ierr)
		std::cout << "! SteadyBackwardEulerSolver: Could not destroy update vector!\n";
}

/** The scaling of linear iterations is currently ignored; the constant set by -ksp_max_it is used
 * for all time steps.
 */
template <int nvars>
StatusCode SteadyBackwardEulerSolver<nvars>::solve(Vec uvec)
{
	if(config.maxiter <= 0) {
		std::cout << " SteadyBackwardEulerSolver: solve(): No iterations to be done.\n";
		return 0;
	}
	
	const UMesh2dh *const m = space->mesh();
	StatusCode ierr = 0;
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);
	Mat M;
	ierr = KSPGetOperators(solver, &M, NULL); CHKERRQ(ierr);
	//bool ismatrixfree = isMatrixFree(M);

	a_real curCFL; int curlinmaxiter;
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;

	/* Our usage of Eigen Maps in the manner below assumes that VecGetArray returns a pointer to
	 * the primary underlying storage in PETSc Vec. This usually happens, but not for
	 * CUDA, CUSP or ViennaCL vecs.
	 * To be safe, we should use VecGetArray and construct the map immediately before
	 * the usage of the map.
	 */
	PetscScalar *duarr, *rarr, *uarr;
	ierr = VecGetArray(duvec, &duarr); CHKERRQ(ierr);
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);
	ierr = VecGetArray(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<MVector> du(duarr, m->gnelem(), nvars);
	Eigen::Map<MVector> residual(rarr, m->gnelem(), nvars);
	Eigen::Map<MVector> u(uarr, m->gnelem(), nvars);

	std::ofstream convout;
	if(config.lognres)
		if(mpirank == 0)
			convout.open(config.logfile+".conv", std::ofstream::app);
	
	/*struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;*/
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	PetscLogDouble initialwtime;
	PetscTime(&initialwtime);
	
	int avglinsteps = 0;

	walltime = cputime = 0;
	double linwtime = 0, linctime = 0;

	while(resi/initres > config.tol && step < config.maxiter)
	{
#pragma omp parallel for default(shared)
		for(a_int iel = 0; iel < m->gnelem(); iel++) {
#pragma omp simd
			for(int i = 0; i < nvars; i++) {
				residual(iel,i) = 0;
			}
		}
		
		// update residual and local time steps
		ierr = space->compute_residual(uvec, rvec, true, dtm); CHKERRQ(ierr);

		ierr = MatZeroEntries(M); CHKERRQ(ierr);
		ierr = space->compute_jacobian(uvec, M); CHKERRQ(ierr);
		
		// compute ramped quantities
		if(step < config.rampstart) {
			curCFL = config.cflinit;
			curlinmaxiter = config.linmaxiterstart;
		}
		else if(step < config.rampend) {
			if(config.rampend-config.rampstart <= 0) {
				curCFL = config.cflfin;
				curlinmaxiter = config.linmaxiterend;
			}
			else {
				const a_real slopec = (config.cflfin-config.cflinit)
					                    /(config.rampend-config.rampstart);
				curCFL = config.cflinit + slopec*(step-config.rampstart);
				
				const a_real slopei = a_real(config.linmaxiterend-config.linmaxiterstart)
					                    /(config.rampend-config.rampstart);
				curlinmaxiter = int(config.linmaxiterstart + slopei*(step-config.rampstart));
			}
		}
		else {
			curCFL = config.cflfin;
			curlinmaxiter = config.linmaxiterend;
		}

		// add pseudo-time terms to diagonal blocks

#pragma omp parallel for default(shared)
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			Matrix<a_real,nvars,nvars,RowMajor> db 
				= Matrix<a_real,nvars,nvars,RowMajor>::Zero();

			for(int i = 0; i < nvars; i++)
				db(i,i) = m->garea(iel) / (curCFL*dtm[iel]);
	
#pragma omp critical
			{
				MatSetValuesBlocked(M, 1, &iel, 1, &iel, db.data(), ADD_VALUES);
			}
		}

		MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
	
		/// Freezes the non-zero structure for efficiency in subsequent time steps.
		ierr = MatSetOption(M, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE); CHKERRQ(ierr);

		// setup and solve linear system for the update du
	
		PetscLogDouble thislinwtime;
		PetscTime(&thislinwtime);
		double thislinctime = (double)clock() / (double)CLOCKS_PER_SEC;

		ierr = KSPSolve(solver, rvec, duvec); CHKERRQ(ierr);

		PetscLogDouble thisfinwtime; PetscTime(&thisfinwtime);
		double thisfinctime = (double)clock() / (double)CLOCKS_PER_SEC;
		linwtime += (thisfinwtime-thislinwtime); 
		linctime += (thisfinctime-thislinctime);

		int linstepsneeded;
		ierr = KSPGetIterationNumber(solver, &linstepsneeded); CHKERRQ(ierr);
		avglinsteps += linstepsneeded;
		
		a_real resnorm2 = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for
			for(a_int iel = 0; iel < m->gnelem(); iel++) {
				u.row(iel) += du.row(iel);
			}
#pragma omp for simd reduction(+:resnorm2)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				resnorm2 += residual(iel,nvars-1)*residual(iel,nvars-1)*m->garea(iel);
			}
		}

		resi = sqrt(resnorm2);

		if(step == 0)
			initres = resi;

		if(step % 10 == 0)
			if(mpirank == 0) {
				std::cout << "  SteadyBackwardEulerSolver: solve(): Step " << step 
					<< ", rel residual " << resi/initres << std::endl;
				std::cout << "      CFL = " << curCFL << ", Lin max iters = " << curlinmaxiter 
					<< ", iters used = " << linstepsneeded << std::endl;
			}

		step++;
			
		if(config.lognres)
			if(mpirank == 0)
				convout << step << " " << std::setw(10)  << resi/initres << '\n';
	}

	/*gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;*/
	PetscLogDouble finalwtime;
	PetscTime(&finalwtime);
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	avglinsteps /= step;

	if(config.lognres)
		if(mpirank == 0)
			convout.close();

	if(mpirank == 0) {
		std::cout << " SteadyBackwardEulerSolver: solve(): Done, steps = " << step 
			<< ", rel residual " << resi/initres << std::endl;
	}
	if(step == config.maxiter)
		if(mpirank == 0) {
			std::cout << "! SteadyBackwardEulerSolver: solve(): Exceeded max iterations!\n";
			std::cout << " SteadyBackwardEulerSolver: solve(): Done, steps = " << step 
				<< ", rel residual " << resi/initres << std::endl;
		}

	// print timing data
	/*double linwtime, linctime;
	linwtime = linctime = -1;
	linsolv->getRunTimes(linwtime, linctime);*/
	if(mpirank == 0) {
		std::cout << "\t\tAverage number of linear solver iterations = " << avglinsteps << std::endl;
		std::cout << "\n SteadyBackwardEulerSolver: solve(): Time taken by ODE solver:" << std::endl;
		std::cout << " \t\tWall time = " << walltime << ", CPU time = " << cputime << "\n";
		std::cout << " SteadyBackwardEulerSolver: solve(): Time taken by linear solver:\n";
		std::cout << " \t\tWall time = " << linwtime << ", CPU time = " << linctime << std::endl;
	}

	// append data to log file
	int numthreads = 1;
#ifdef _OPENMP
	numthreads = omp_get_max_threads();
#endif
	if(mpirank == 0) {
		std::ofstream outf; outf.open(config.logfile, std::ofstream::app);

		// if the file is empty, write header
		outf.seekp(0, std::ios::end);
		if(outf.tellp() == 0)
			outf << std::setw(10) << "# num-cells "
				<< std::setw(6) << "threads " << std::setw(10) << "wall-time " 
				<< std::setw(10) << "cpu-time " << std::setw(10) << "avg-lin-iters "
				<< std::setw(10) << " time-steps\n";

		// write current info
		outf << std::setw(10) << m->gnelem() << " "
			<< std::setw(6) << numthreads << " " << std::setw(10) << linwtime << " " 
			<< std::setw(10) << linctime << " " << std::setw(10) << avglinsteps << " "
			<< std::setw(10) << step
			<< "\n";
		outf.close();
	}
	
	ierr = VecRestoreArray(duvec, &duarr); CHKERRQ(ierr);
	ierr = VecRestoreArray(rvec, &rarr); CHKERRQ(ierr);
	ierr = VecRestoreArray(uvec, &uarr); CHKERRQ(ierr);
	return ierr;
}

template <int nvars>
UnsteadySolver<nvars>::UnsteadySolver(const Spatial<nvars> *const spatial, Vec soln,
		const int temporal_order, const std::string log_file)
	: space(spatial), uvec(soln), order{temporal_order}, cputime{0.0}, walltime{0.0},
	  logfile{log_file}
{ }

template <int nvars>
TVDRKSolver<nvars>::TVDRKSolver(const Spatial<nvars> *const spatial, 
		Vec soln, const int temporal_order, const std::string log_file, const double cfl_num)
	: UnsteadySolver<nvars>(spatial, soln, temporal_order, log_file), cfl{cfl_num},
	tvdcoeffs(initialize_TVDRK_Coeffs(temporal_order))
{
	dtm.resize(space->mesh()->gnelem(), 0);
	int ierr = VecDuplicate(uvec, &rvec);
	if(ierr)
		std::cout << "! TVDRKSolver: Could not create residual vector!\n";
}

template <int nvars>
TVDRKSolver<nvars>::~TVDRKSolver() {
	int ierr = VecDestroy(&rvec);
	if(ierr)
		std::cout << "! TVDRKSolver: Could not destroy residual vector!\n";
}

template<int nvars>
StatusCode TVDRKSolver<nvars>::solve(const a_real finaltime)
{
	const UMesh2dh *const m = space->mesh();
	StatusCode ierr = 0;
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);

	PetscInt locnelem; PetscScalar *uarr; PetscScalar *rarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ierr = VecGetArray(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<MVector> u(uarr, locnelem, nvars);
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);
	Eigen::Map<MVector> residual(rarr, locnelem, nvars);


	int step = 0;
	a_real time = 0;   //< Physical time elapsed
	a_real dtmin=0;      //< Time step

	// Stage solution vector
	MVector ustage(m->gnelem(),nvars);
#pragma omp parallel for simd default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
		for(int ivar = 0; ivar < nvars; ivar++)
			ustage(iel,ivar) = u(iel,ivar);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	while(time <= finaltime - A_SMALL_NUMBER)
	{
		for(int istage = 0; istage < order; istage++)
		{
#pragma omp parallel for simd default(shared)
			for(a_int iel = 0; iel < m->gnelem(); iel++) {
				for(int i = 0; i < nvars; i++)
					residual(iel,i) = 0;
			}

			// update residual
			space->compute_residual(uvec, rvec, true, dtm);

			// update time step for the first stage of each time step
			if(istage == 0)
				dtmin = *std::min_element(dtm.begin(),dtm.end());

#pragma omp parallel for simd default(shared)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					ustage(iel,i) = tvdcoeffs(istage,0)*u(iel,i)
						          + tvdcoeffs(istage,1)*ustage(iel,i)
								  - tvdcoeffs(istage,2) * dtmin*cfl/m->garea(iel)*residual(iel,i);
				}
			}
		}

#pragma omp parallel for simd default(shared)
		for(a_int iel = 0; iel < m->gnelem(); iel++)
			for(int ivar = 0; ivar < nvars; ivar++)
				u(iel,ivar) = ustage(iel,ivar);


		if(step % 50 == 0)
			if(mpirank == 0)
				std::cout << "  TVDRKSolver: solve(): Step " << step 
					<< ", time " << time << std::endl;

		step++;
		time += dtmin;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);

	if(mpirank == 0) {
		std::cout << " TVDRKSolver: solve(): Done, steps = " << step << "\n\n";
		std::cout << " TVDRKSolver: solve(): Time taken by ODE solver:\n";
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

	ierr = VecRestoreArray(uvec, &uarr); CHKERRQ(ierr);
	ierr = VecRestoreArray(rvec, &rarr); CHKERRQ(ierr);
	return ierr;
}

template class SteadyForwardEulerSolver<NVARS>;
template class SteadyBackwardEulerSolver<NVARS>;
template class SteadyForwardEulerSolver<1>;
template class SteadyBackwardEulerSolver<1>;

template class TVDRKSolver<NVARS>;

}	// end namespace
