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

SteadyForwardEulerSolver::SteadyForwardEulerSolver(const UMesh2dh *const mesh, EulerFV *const euler) : m(mesh), eul(euler)
{
}

SteadyForwardEulerSolver::~SteadyForwardEulerSolver()
{
}

a_real SteadyForwardEulerSolver::l2norm(const amat::Array2d<a_real>* const v)
{
	a_real norm = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		norm += v->get(iel)*v->get(iel)*m->garea(iel);
	}
	norm = sqrt(norm);
	return norm;
}

void SteadyForwardEulerSolver::solve(const a_real tol, const int maxiter, const a_real cfl)
{
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;
	//amat::Array2d<a_real> uold(m->gnelem(), NVARS);

	while(resi/initres > tol && step < maxiter)
	{
		//std::cout << "EulerFV: solve_rk1_steady(): Entered loop. Step " << step << std::endl;

		// update residual
		eul->compute_residual();

		a_real errmass = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for simd
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < NVARS; i++)
				{
					//uold(iel,i) = u(iel,i);
					eul->unknowns()(iel,i) -= cfl*eul->localTimeSteps()(iel) * 1.0/m->garea(iel)*eul->residuals()(iel,i);
				}
			}

#pragma omp for simd reduction(+:errmass)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				errmass += eul->residuals()(iel,0)*eul->residuals()(iel,0)*m->garea(iel);
			}
		} // end parallel region

		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 50 == 0)
			std::cout << "  SteadyForwardEulerSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;

		step++;
		/*a_real totalenergy = 0;
		for(int i = 0; i < m->gnelem(); i++)
			totalenergy += u(i,3)*m->jacobians(i);
			std::cout << "EulerFV: solve(): Total energy = " << totalenergy << std::endl;*/
		//if(step == 10000) break;
	}

	if(step == maxiter)
		std::cout << "! SteadyForwardEulerSolver: solve(): Exceeded max iterations!" << std::endl;
	std::cout << " SteadyForwardEulerSolver: solve(): Done, steps = " << step << std::endl;
}


SteadyBackwardEulerSolver::SteadyBackwardEulerSolver(const UMesh2dh*const mesh, EulerFV *const spatial, 
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, const int lin_tol, const int linmaxiter_start, const int linmaxiter_end, std::string linearsolver)
	: m(mesh), eul(spatial), cflinit(cfl_init), cflfin(cfl_fin), rampstart(ramp_start), rampend(ramp_end), tol(toler), maxiter(maxits), lintol(lin_tol),
	linmaxiterstart(linmaxiter_start), linmaxiterend(linmaxiter_end)
{
	if(linearsolver == "BSGS")
		linsolv = new SGS_Relaxation(m);
	else
		linsolv = new SGS_Relaxation(m);

	D = new Matrix[m->gnelem()];
	L = new Matrix[m->gnaface()-m->gnbface()];
	U = new Matrix[m->gnaface()-m->gnbface()];
	for(int iel = 0; iel < m->gnelem(); iel++)
		D[iel].resize(NVARS,NVARS);
	for(int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
		L[iface].resize(NVARS,NVARS);
		U[iface].resize(NVARS,NVARS);
	}
}

SteadyBackwardEulerSolver::~SteadyBackwardEulerSolver()
{
	delete linsolv;
	delete [] D;
	delete [] U;
	delete [] L;
}

void SteadyBackwardEulerSolver::solve()
{
	int step = 0;
	double curCFL; int curlinmaxiter;
	a_real resi = 1.0;
	a_real initres = 1.0;
	Matrix du = Matrix::Zero(m->gnelem(), NVARS);

	while(resi/initres > tol && step < maxiter)
	{
		// update residual and local time steps
		eul->compute_residual();

		eul->compute_jacobian(D, L, U);
		
		// compute ramped quantities
		if(step < rampstart) {
			curCFL = cflinit;
			curlinmaxiter = linmaxiterstart;
		}
		else if(step < rampend) {
			double slopec = (cflfin-cflinit)/(rampend-rampstart);
			curCFL = cflinit + slopec*(step-rampstart);
			double slopei = double(linmaxiterend-linmaxiterstart)/(rampend-rampstart);
			curlinmaxiter = int(linmaxiterstart + slopei*(step-rampstart));
			//curlinmaxiter = linmaxiterstart;
		}
		else {
			curCFL = cflfin;
			curlinmaxiter = linmaxiterend;
		}

		// add pseudo-time terms to diagonal blocks
#pragma omp parallel for simd default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < NVARS; i++)
				D[iel](i,i) += m->garea(iel) / (curCFL*eul->localTimeSteps()(iel));
		}

		// setup and solve linear system for the update du
		linsolv->setLHS(D,L,U);
		linsolv->setParams(lintol, curlinmaxiter);
		linsolv->solve(eul->residuals(), du);

		a_real errmass = 0;

#pragma omp parallel default(shared)
		{
#pragma omp for
			for(int iel = 0; iel < m->gnelem(); iel++) {
				eul->unknowns().row(iel) += du.row(iel);
				/*for(int i = 0; i < NVARS; i++)
					du(iel,i) = 0;*/
			}
#pragma omp for simd reduction(+:errmass)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				errmass += eul->residuals()(iel,0)*eul->residuals()(iel,0)*m->garea(iel);
			}
		}

		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 10 == 0) {
			std::cout << "  SteadyBackwardEulerSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;
			std::cout << "         CFL = " << curCFL << ", Lin max iters = " << curlinmaxiter << std::endl;
		}

		step++;
	}

	if(step == maxiter)
		std::cout << "! SteadyBackwardEulerSolver: solve(): Exceeded max iterations!" << std::endl;
	std::cout << " SteadyBackwardEulerSolver: solve(): Done, steps = " << step << std::endl;

	double linwtime, linctime;
	linsolv->getRunTimes(linwtime, linctime);
	std::cout << " SteadyBackwardEulerSolver: solve(): Time taken by linear solver:\n    Wall time = " << linwtime << ", CPU time = " << linctime << std::endl;
}

}	// end namespace
