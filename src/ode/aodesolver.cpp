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

#include "aodesolver.hpp"
#include "spatial/aoutput.hpp"
#include "linalg/alinalg.hpp"
#include "linalg/petsc_assembly.hpp"
#include "utilities/aoptionparser.hpp"
#include "utilities/aerrorhandling.hpp"
#include "utilities/mpiutils.hpp"

namespace fvens {

/// Returns an array containing TVD Runge-Kutta coefficients for high-order accuracy
static Eigen::Matrix<a_real,Eigen::Dynamic,Eigen::Dynamic> initialize_TVDRK_Coeffs(const int _order) 
{
	Eigen::Matrix<a_real,Eigen::Dynamic,Eigen::Dynamic> tvdrk(_order,3);
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

TimingData::TimingData()
	: nelem{0}, num_threads{1}, lin_walltime{0}, lin_cputime{0}, ode_walltime{0}, ode_cputime{0},
	  total_lin_iters{0}, avg_lin_iters{0}, num_timesteps{0}, converged{false}, precsetup_walltime{0},
	  precapply_walltime{0}, prec_cputime{0}
{ }

template <int nvars>
SteadySolver<nvars>::SteadySolver(const Spatial<a_real,nvars> *const spatial, const SteadySolverConfig& conf)
	: space{spatial}, config{conf}
{
	tdata.nelem = spatial->mesh()->gnelem();
}

template <int nvars>
TimingData SteadySolver<nvars>::getTimingData() const {
	return tdata;
}
	
template <int nvars>
a_real SteadySolver<nvars>::linearRamp(const a_real cstart, const a_real cend, 
                                       const int itstart, const int itend, const int itcur) const
{
	a_real curCFL;
	if(itcur < itstart) {
		curCFL = cstart;
	}
	else if(itcur < itend) {
		if(itend-itstart <= 0) {
			curCFL = cend;
		}
		else {
			const a_real slopec = (cend-cstart)/(itend-itstart);
			curCFL = cstart + slopec*(itcur-itstart);
		}
	}
	else {
		curCFL = cend;
	}
	return curCFL;
}

template <int nvars>
a_real SteadySolver<nvars>::expResidualRamp(const a_real cflmin, const a_real cflmax, 
                                            const a_real prevcfl, const a_real resratio, const a_real paramup, const a_real paramdown)
{
	const a_real newcfl = resratio > 1.0 ? prevcfl * std::pow(resratio, paramup)
		: prevcfl * std::pow(resratio, paramdown);

	if(newcfl < cflmin) return cflmin;
	else if(newcfl > cflmax) return cflmax;
	else return newcfl;
}


template<int nvars>
SteadyForwardEulerSolver<nvars>::SteadyForwardEulerSolver(const Spatial<a_real,nvars> *const spatial,
                                                          const Vec uvec,
                                                          const SteadySolverConfig& conf)

	: SteadySolver<nvars>(spatial, conf)
{ }

template<int nvars>
SteadyForwardEulerSolver<nvars>::~SteadyForwardEulerSolver()
{ }

template<int nvars>
StatusCode SteadyForwardEulerSolver<nvars>::solve(Vec uvec)
{
	StatusCode ierr = 0;
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	const UMesh2dh<a_real> *const m = space->mesh();

	if(config.maxiter <= 0) {
		std::cout << " SteadyForwardEulerSolver: solve(): No iterations to be done.\n";
		return ierr;
	}

	Vec rvec, dtmvec;
	ierr = VecDuplicate(uvec, &rvec); CHKERRQ(ierr);
	ierr = createSystemVector(m, 1, &dtmvec); CHKERRQ(ierr);

	PetscInt locnelem;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	int step = 0;
	a_real resi = 1.0;
	a_real resiold = resi;
	a_real initres = 1.0;

	// struct timeval time1, time2;
	// gettimeofday(&time1, NULL);
	// double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	const double initialwtime = MPI_Wtime();
	const double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	std::cout << " Starting CFL = " << config.cflinit << std::endl;
	a_real curCFL = config.cflinit;

	SteadyStepMonitor convstep;
	if(mpirank == 0)
		writeConvergenceHistoryHeader(std::cout);


	while(resi/initres > config.tol && step < config.maxiter)
	{
		{
			MutableGhostedVecHandler<PetscScalar> rh(rvec);
			PetscScalar *const rarr = rh.getArray();

#pragma omp parallel for simd default(shared)
			for(a_int i = 0; i < m->gnelem()*nvars; i++) {
				rarr[i] = 0;
			}
		}

		ierr = space->compute_residual(uvec, rvec, true, dtmvec); CHKERRQ(ierr);

		ierr = VecGhostUpdateBegin(rvec, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		ierr = VecGhostUpdateEnd(rvec, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

		curCFL = expResidualRamp(config.cflinit, config.cflfin, curCFL, resiold/resi, 0.3, 0.25);

		{
			ConstVecHandler<PetscScalar> dth(dtmvec);
			const PetscScalar *const dtm = dth.getArray();
			MutableGhostedVecHandler<PetscScalar> uh(uvec);
			Eigen::Map<MVector<a_real>> u(uh.getArray(), locnelem, nvars);
			ConstGhostedVecHandler<PetscScalar> rh(rvec);
			Eigen::Map<const MVector<a_real>> residual(rh.getArray(), locnelem, nvars);

#pragma omp parallel for default(shared)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
					u(iel,i) += config.cflinit*dtm[iel] * 1.0/m->garea(iel)*residual(iel,i);
			}
		}

		ierr = VecGhostUpdateBegin(uvec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

		a_real locresenergy = 0;
		{
			ConstGhostedVecHandler<PetscScalar> rh(rvec);
			Eigen::Map<const MVector<a_real>> residual(rh.getArray(), locnelem, nvars);
#pragma omp parallel for simd reduction(+:locresenergy) default(shared)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				locresenergy += residual(iel,nvars-1)*residual(iel,nvars-1)*m->garea(iel);
			}
		}

		// Reduce across and communicate to all processes: the residual norm
		a_real glres = 0;
		MPI_Allreduce(&locresenergy, &glres, 1, FVENS_MPI_REAL, MPI_SUM, PETSC_COMM_WORLD);

		resiold = resi;
		resi = sqrt(glres);

		if(step == 0)
			initres = resi;

		step++;

		const double curtime = MPI_Wtime();
		convstep = {step, (float)(resi/initres), (float)resi, (float)(curtime-initialwtime),
		            0, 0, (float)curCFL};
		if(config.lognres)
			tdata.convhis.push_back(convstep);

		if((step-1) % 50 == 0)
			if(mpirank==0)
				writeStepToConvergenceHistory(convstep, std::cout);

		ierr = VecGhostUpdateEnd(uvec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

		// test for nan
		if(!std::isfinite(resi))
			throw Numerical_error("Steady forward Euler diverged - residual is Nan or inf!");
	}
	
	const double finalwtime = MPI_Wtime();
	const double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	tdata.ode_walltime += (finalwtime-initialwtime); tdata.ode_cputime += (finalctime-initialctime);
	tdata.num_timesteps = step;

	if(mpirank==0)
		writeStepToConvergenceHistory(convstep, std::cout);

	tdata.converged = true;
	if(step == config.maxiter) {
		tdata.converged = false;
		if(mpirank == 0)
			std::cout << "! SteadyForwardEulerSolver: solve(): Exceeded max iterations!\n";
		throw Tolerance_error("Steady forward Euler did not converge to specified tolerance!");
	}
	if(mpirank == 0) {
		std::cout << " SteadyForwardEulerSolver: solve(): Done. ";
		std::cout << " CPU time = " << tdata.ode_cputime << std::endl;
	}

#ifdef _OPENMP
	tdata.num_threads = omp_get_max_threads();
#endif

	ierr = VecDestroy(&rvec); CHKERRQ(ierr);
	ierr = VecDestroy(&dtmvec); CHKERRQ(ierr);
	
	return ierr;
}

/** By default, the Jacobian is stored in a block sparse row format.
 */
template <int nvars>
SteadyBackwardEulerSolver<nvars>::
SteadyBackwardEulerSolver(const Spatial<a_real,nvars> *const spatial,
                          const SteadySolverConfig& conf,
                          KSP ksp, const NonlinearUpdate<nvars> *const nlr)

	: SteadySolver<nvars>(spatial, conf), solver{ksp}, update_relax{nlr}
{
}

template <int nvars>
SteadyBackwardEulerSolver<nvars>::~SteadyBackwardEulerSolver()
{ }

template <int nvars>
StatusCode SteadyBackwardEulerSolver<nvars>::addPseudoTimeTerm(const a_real cfl, Vec dtmvec, Mat M)
{
	StatusCode ierr = 0;
	const UMesh2dh<a_real> *const m = space->mesh();

	MutableVecHandler<PetscScalar> dth(dtmvec);
	PetscScalar *const dtm = dth.getArray();

	// Add pseudo-time terms to diagonal blocks
	// NOTE: After the following loop, dtm will contain Vol/(CFL*dt).
	//   This is required in case of matrix-free solvers.

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		dtm[iel] = m->garea(iel) / (cfl*dtm[iel]);
		const a_int ielg = m->gglobalElemIndex(iel);

		const Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor> db
			= dtm[iel] * Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>::Identity();
	
#pragma omp critical
		{
			MatSetValuesBlocked(M, 1, &ielg, 1, &ielg, db.data(), ADD_VALUES);
		}
	}

	return ierr;
}

template <int nvars>
StatusCode SteadyBackwardEulerSolver<nvars>::addPseudoTimeTerm_slow(const a_real cfl, Vec dtmvec, Mat M)
{
	StatusCode ierr = 0;
	const UMesh2dh<a_real> *const m = space->mesh();

	MutableVecHandler<PetscScalar> dth(dtmvec);
	PetscScalar *const dtm = dth.getArray();

	// Add pseudo-time terms to diagonal blocks
	// NOTE: After the following loop, dtm will contain Vol/(CFL*dt).
	//   This is required in case of matrix-free solvers.

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		dtm[iel] = m->garea(iel) / (cfl*dtm[iel]);
		const a_int ielg = m->gglobalElemIndex(iel);

		for(int i = 0; i < nvars; i++)
		{
			const a_int index = ielg*nvars+i;
#pragma omp critical
			{
				MatSetValues(M, 1, &index, 1, &index, &dtm[iel], ADD_VALUES);
			}
		}
	}

	return ierr;
}

template <int nvars>
StatusCode SteadyBackwardEulerSolver<nvars>::solve(Vec uvec)
{
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	if(config.maxiter <= 0) {
		if(mpirank == 0)
			std::cout << " SteadyBackwardEulerSolver: solve(): No iterations to be done.\n";
		return 0;
	}
	
	const UMesh2dh<a_real> *const m = space->mesh();
	StatusCode ierr = 0;

	Vec rvec, duvec, dtmvec;
	ierr = VecDuplicate(uvec, &rvec); CHKERRQ(ierr);
	ierr = createSystemVector(m, nvars, &duvec); CHKERRQ(ierr);
	ierr = createSystemVector(m, 1, &dtmvec); CHKERRQ(ierr);

	// get the system and preconditioning matrices
	Mat M, A;
	ierr = KSPGetOperators(solver, &A, &M); CHKERRQ(ierr);

	const bool ismatrixfree = isMatrixFree(A);
	if(ismatrixfree) {
		MatrixFreeSpatialJacobian<nvars>* mfA = nullptr;
		ierr = MatShellGetContext(A, (void**)&mfA); CHKERRQ(ierr);
		// uvec, rvec and dtm keep getting updated, but pointers to them can be set just once
		if(mpirank == 0)
			std::cout << " Setting matfree state" << std::endl;
		mfA->set_state(uvec,rvec, dtmvec);
	}

	// get list of iterations at which to recompute AMG interpolation operators, if used
	const std::vector<int> amgrecompute
		= parseOptionalPetscCmd_intArray("-amg_recompute_interpolation",3);

	bool tocomputeamginterpolation = false;

	a_real curCFL=0;
	int step = 0;
	a_real resi = 1.0, resiold = 1.0;
	a_real initres = 1.0;

	if(config.lognres)
		tdata.convhis.reserve(100);
	
	const double initialwtime = MPI_Wtime();
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	double linwtime = 0, linctime = 0;

	SteadyStepMonitor convline;
	if(mpirank == 0)
		writeConvergenceHistoryHeader(std::cout);
		
	while(resi/initres > config.tol && step < config.maxiter)
	{
		{
			MutableGhostedVecHandler<PetscScalar> rh(rvec);
			PetscScalar *const rarr = rh.getArray();

#pragma omp parallel for simd default(shared)
			for(a_int i = 0; i < m->gnelem()*nvars; i++) {
					rarr[i] = 0;
			}
		}

		std::vector<int>::const_iterator it = std::find(amgrecompute.begin(), amgrecompute.end(), step+1);
		if(it != amgrecompute.end()) {
			if(mpirank == 0) {
				std::cout << " SteadyBackwardEulerSolver: solve(): Recomputing AMG interpolation if ";
				std::cout << "required.\n";
			}
			tocomputeamginterpolation = true;
		}

		if(tocomputeamginterpolation) {
			PC pc;
			ierr = KSPGetPC(solver, &pc);
			PCGAMGSetReuseInterpolation(pc, PETSC_FALSE);
			tocomputeamginterpolation = false;
		} else {
			PC pc;
			ierr = KSPGetPC(solver, &pc);
			PCGAMGSetReuseInterpolation(pc, PETSC_TRUE);
		}
		
		// update residual and local time steps
		ierr = space->compute_residual(uvec, rvec, true, dtmvec); CHKERRQ(ierr);

		ierr = VecGhostUpdateBegin(rvec, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

		ierr = MatZeroEntries(M); CHKERRQ(ierr);
		ierr = space->assemble_jacobian(uvec, M); CHKERRQ(ierr);

		// curCFL = linearRamp(config.cflinit, config.cflfin,
		//                     /*config.rampstart*/30, /*config.rampend*/100, step);
		// (void)resiold;
		curCFL = expResidualRamp(config.cflinit, config.cflfin, curCFL, resiold/resi, 0.25, 0.3);

		// Add pseudo-time terms to diagonal blocks
		// NOTE: After the following function call, dtm will contain Vol/(CFL*dt),
		//   since this is required in case of matrix-free solvers.
		ierr = addPseudoTimeTerm(curCFL, dtmvec, M); CHKERRQ(ierr);

		ierr = VecGhostUpdateEnd(rvec, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

		ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
		ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	
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
		tdata.total_lin_iters += linstepsneeded;

		// Update solution and compute residual norm
		{
			MutableGhostedVecHandler<PetscScalar> uh(uvec);
			Eigen::Map<MVector<a_real>> u(uh.getArray(), m->gnelem(), nvars);
			ConstGhostedVecHandler<PetscScalar> rh(rvec);
			Eigen::Map<const MVector<a_real>> residual(rh.getArray(), m->gnelem(), nvars);

			ConstVecHandler<PetscScalar> duh(duvec);
			const PetscScalar *const duarr = duh.getArray();
			Eigen::Map<const MVector<a_real>> du(duarr, m->gnelem(), nvars);


#pragma omp parallel for
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				const a_real omega = update_relax->getLocalRelaxationFactor(duarr+iel*nvars, &u(iel,0));
				u.row(iel) += omega*du.row(iel);
			}
		}

		ierr = VecGhostUpdateBegin(uvec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

		a_real resnorm2 = 0;
		{
			ConstGhostedVecHandler<PetscScalar> rh(rvec);
			Eigen::Map<const MVector<a_real>> residual(rh.getArray(), m->gnelem(), nvars);

#pragma omp parallel for simd reduction(+:resnorm2)
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				resnorm2 += residual(iel,nvars-1)*residual(iel,nvars-1)*m->garea(iel);
			}
		}

		MPI_Allreduce(&resnorm2, &resnorm2, 1, FVENS_MPI_REAL, MPI_SUM, PETSC_COMM_WORLD);
		resiold = resi;
		resi = sqrt(resnorm2);

		// test for nan
		if(!std::isfinite(resi))
			throw Numerical_error("Steady backward Euler diverged - residual is Nan or inf!");

		if(step == 0)
			initres = resi;

		step++;

		const double curtime = MPI_Wtime();
		convline = { step, (float)(resi/initres), (float)resi,
		             (float)(curtime-initialwtime), (float)linwtime, tdata.total_lin_iters,
		             (float)curCFL };

		if(config.lognres)
		{
			tdata.convhis.push_back(convline);
		}

		if((step-1) % 10 == 0)
		{
			if(mpirank == 0) {
				writeStepToConvergenceHistory(convline, std::cout);
			}
		}

		ierr = VecGhostUpdateEnd(uvec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	}

	const double finalwtime = MPI_Wtime();
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	tdata.ode_walltime += (finalwtime-initialwtime); 
	tdata.ode_cputime += (finalctime-initialctime);
	tdata.avg_lin_iters = (int) (tdata.total_lin_iters / (double)step);
	tdata.num_timesteps = step;

	if(mpirank == 0) {
		writeStepToConvergenceHistory(convline, std::cout);
	}

	// print timing data
	if(mpirank == 0) {
		std::cout << "  SteadySolver: Average number of linear solver iterations = "
		          << tdata.avg_lin_iters << '\n';
		std::cout << "  SteadySolver: solve(): Total time taken: ";
		std::cout << " CPU time = " << tdata.ode_cputime << "\n";
		std::cout << "  SteadySolver: solve(): Time taken by linear solver:";
		std::cout << " CPU time = " << linctime << std::endl;
	}

#ifdef _OPENMP
	tdata.num_threads = omp_get_max_threads();
#endif
	tdata.lin_walltime = linwtime; 
	tdata.lin_cputime = linctime;

	tdata.converged = false;
	if(step < config.maxiter && (resi/initres <= config.tol))
		tdata.converged = true;
	else if (step >= config.maxiter){
		if(mpirank == 0) {
			std::cout << "! SteadyBackwardEulerSolver: solve(): Exceeded max iterations!\n";
		}
		throw Tolerance_error("Steady backward Euler did not converge to specified tolerance!");
	}
	else {
		if(mpirank == 0) {
			std::cout << "! SteadyBackwardEulerSolver: solve(): Blew up!\n";
		}
		throw Numerical_error("Steady backward Euler blew up!");
	}

	ierr = VecDestroy(&rvec); CHKERRQ(ierr);
	ierr = VecDestroy(&duvec); CHKERRQ(ierr);
	ierr = VecDestroy(&dtmvec); CHKERRQ(ierr);
	return ierr;
}

template <int nvars>
UnsteadySolver<nvars>::UnsteadySolver(const Spatial<a_real,nvars> *const spatial, Vec soln,
		const int temporal_order, const std::string log_file)
	: space(spatial), uvec(soln), order{temporal_order}, cputime{0.0}, walltime{0.0},
	  logfile{log_file}
{ }

template <int nvars>
TVDRKSolver<nvars>::TVDRKSolver(const Spatial<a_real,nvars> *const spatial, 
		Vec soln, const int temporal_order, const std::string log_file, const double cfl_num)
	: UnsteadySolver<nvars>(spatial, soln, temporal_order, log_file), cfl{cfl_num},
	tvdcoeffs(initialize_TVDRK_Coeffs(temporal_order))
{
	int ierr = VecDuplicate(uvec, &rvec);
	petsc_throw(ierr, "! TVDRKSolver: Could not create residual vector!");
	ierr = createSystemVector(space->mesh(), 1, &dtmvec);
	petsc_throw(ierr, "Could not create dt vec");
	std::cout << " TVDRKSolver: Initialized TVD RK solver of order " << order <<
		", CFL = " << cfl << std::endl;
}

template <int nvars>
TVDRKSolver<nvars>::~TVDRKSolver() {
	int ierr = VecDestroy(&rvec);
	if(ierr)
		std::cout << "! TVDRKSolver: Could not destroy residual vector!\n";
	ierr = VecDestroy(&dtmvec);
	if(ierr)
		std::cout << "! TVDRKSolver: Could not destroy time-step vector!\n";
}

template<int nvars>
StatusCode TVDRKSolver<nvars>::solve(const a_real finaltime)
{
	const UMesh2dh<a_real> *const m = space->mesh();
	StatusCode ierr = 0;
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);

	PetscInt locnelem; PetscScalar *uarr; PetscScalar *rarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ierr = VecGetArray(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<MVector<a_real>> u(uarr, locnelem, nvars);
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);
	Eigen::Map<MVector<a_real>> residual(rarr, locnelem, nvars);


	int step = 0;
	a_real time = 0;   //< Physical time elapsed

	// Stage solution vector
	MVector<a_real> ustage(m->gnelem(),nvars);
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
		a_real dtmin=0;      //< Time step

		for(int istage = 0; istage < order; istage++)
		{
#pragma omp parallel for simd default(shared)
			for(a_int iel = 0; iel < m->gnelem(); iel++) {
				for(int i = 0; i < nvars; i++)
					residual(iel,i) = 0;
			}

			// update residual
			ierr = space->compute_residual(uvec, rvec, true, dtmvec); CHKERRQ(ierr);

			// update time step for the first stage of each time step
			if(istage == 0)
			{
				const PetscScalar *dtm;
				ierr = VecGetArrayRead(dtmvec, &dtm); CHKERRQ(ierr);
				dtmin = *std::min_element(dtm, dtm+m->gnelem());
				ierr = VecRestoreArrayRead(dtmvec, &dtm); CHKERRQ(ierr);
			}

			if(!std::isfinite(dtmin))
				throw Numerical_error("TVDRK solver diverged - dtmin is Nan or inf!");

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


		if(step % 10 == 0)
			if(mpirank == 0)
				std::cout << "  TVDRKSolver: solve(): Step " << step 
				          << ", time " << time << ", time-step = " << dtmin*cfl << std::endl;

		step++;
		time += dtmin*cfl;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);

	if(mpirank == 0) {
		std::cout << " TVDRKSolver: solve(): Done, steps = " << step << ", phy time = "
		          << time << "\n\n";
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

template class SteadySolver<NVARS>;
template class SteadySolver<1>;

template class SteadyForwardEulerSolver<NVARS>;
template class SteadyBackwardEulerSolver<NVARS>;
template class SteadyForwardEulerSolver<1>;
template class SteadyBackwardEulerSolver<1>;

template class TVDRKSolver<NVARS>;

}	// end namespace
