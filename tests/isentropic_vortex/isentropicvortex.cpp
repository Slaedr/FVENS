/** \file isentropicvortex.cpp
 * \brief Implementation of isentropic vortex solution generation
 * \author Aditya Kashi
 */

#include "isentropicvortex.hpp"
#include <cmath>

namespace fvens_tests {

IsentropicVortexProblem::IsentropicVortexProeblem(const IsenVortexConfig config)
	: conf{config}
{
	pinf = 1.0/(conf.gamma*conf.Minf*conf.Minf);
}

inline a_real IsentropicVortexProblem::computeOmega(std::array<a_real,2> r) const
{
	const a_real f = -0.5/(conf.sigma*conf.sigma*conf.clength*conf.clength)*(r[0]*r[0]+r[1]*r[1]);
	return conf.strength * std::exp(f);
}

inline std::array<a_real,2> IsentropicVortexProblem::computeVelocity(std::array<a_real,2> r) const
{
	return {conf.Minf*std::cos(conf.aoa) - r[1]/conf.clength*omega,
			conf.Minf*std::sin(conf.aoa) + r[0]/conf.clength*omega};
}

inline a_real IsentropicVortexProblem::computeFlowState
	(const std::array<a_real,2> r, a_real *const u) const
{
	const a_real omega = computeOmega(r);
	const a_real dT = -(conf.gamma-1)/2.0*omega*omega;
	const std::array<a_real,2> v = computeVelocity(r);
	const a_real p = 1./(conf.gamma*conf.Minf*conf.Minf)*std::pow(1.0+dT, conf.gamma/(conf.gamma-1));

	u[0] = std::pow(1+dT, 1.0/(conf.gamma-1));
	u[1] = u[0]*v[0];
	u[2] = u[0]*v[1];
	u[3] = p/(conf.gamma-1) + 0.5*u[0]*(v[0]*v[0]+v[1]*v[1]);
}

void IsentropicVortexProblem::getInitialConditionAndExactSolution
	(const UMesh2dh& mesh, const a_real time, a_real *const u, a_real *const uexact) const
{
	for(a_int iel = 0; iel < m.gnelem(); iel++) {

		std::array<a_real,2> cellcentre = {0,0};
		for(int inode = 0; inode < m->gnnode(iel); inode++) {
			for(int j = 0; j < 2; j++)
				cellcentre[j] += m->gcoords(m->ginpoel(iel,inode),j);
		}
		for(int j = 0; j < 2; j++)
			cellcentre[j] /= m->gnnode(iel);

		computeFlowState(cellcentre, &u[iel*4]);

		std::array<a_real,2> advect_point = {conf.centre[0] - time*conf.Minf*std::cos(conf.aoa),
		                                     conf.centre[1] - time*conf.Minf*std::cos(conf.aoa)};
		computeFlowState(advect_point, &uexact[iel*4]);
	}
}

}
