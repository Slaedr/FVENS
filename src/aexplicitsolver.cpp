/** @file aexplicitsolver.cpp
 * @brief Implements driver class(es) for explicit solution of Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016
 */

#include "aexplicitsolver.hpp"

namespace acfd {

ForwardEulerTimeSolver::ForwardEulerTimeSolver(const UMesh2dh *const mesh, EulerFV *const euler) : m(mesh), eul(euler)
{
}

ForwardEulerTimeSolver::~ForwardEulerTimeSolver()
{
}

a_real ForwardEulerTimeSolver::l2norm(const amat::Matrix<a_real>* const v)
{
	a_real norm = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		norm += v->get(iel)*v->get(iel)*m->garea(iel);
	}
	norm = sqrt(norm);
	return norm;
}

void ForwardEulerTimeSolver::solve(const a_real tol, const int maxiter, const a_real cfl)
{
	int step = 0;
	a_real resi = 1.0;
	a_real initres = 1.0;
	//amat::Matrix<a_real> uold(m->gnelem(), NVARS);

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
			std::cout << "  ForwardEulerTimeSolver: solve(): Step " << step << ", rel residual " << resi/initres << std::endl;

		step++;
		/*a_real totalenergy = 0;
		for(int i = 0; i < m->gnelem(); i++)
			totalenergy += u(i,3)*m->jacobians(i);
			std::cout << "EulerFV: solve(): Total energy = " << totalenergy << std::endl;*/
		//if(step == 10000) break;
	}

	if(step == maxiter)
		std::cout << "! ForwardEulerTimeSolver: solve(): Exceeded max iterations!" << std::endl;
	std::cout << " ForwardEulerTimeSolver: solve(): Done, steps = " << step << std::endl;
}

}	// end namespace
