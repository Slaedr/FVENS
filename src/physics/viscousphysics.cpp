/** \file
 * \brief Implementation of physical transormations requried for viscous flux computation
 * \author Aditya Kashi
 */

#include "viscousphysics.hpp"

namespace fvens {

template<typename scalar, int ndim, bool secondOrderRequested>
void getPrimitive2StateAndGradients(const IdealGasPhysics<scalar>& physics,
                                    const scalar *const ucl, const scalar *const ucr,
                                    const scalar *const gradl, const scalar *const gradr,
                                    scalar *const uctl, scalar *const uctr,
                                    scalar *const gradtl, scalar *const gradtr)
{
	static_assert(ndim == NDIM, "3D not implemented yet.");
	constexpr int nvars = ndim+2;

	/* Get proper state variables and grads at cell centres
	 * we start with left- and right- conserved variables, primitive ghost cell-center states and
	 * primitive gradients
	 */

	/*for(int i = 0; i < nvars; i++) {
		uctl[i] = 0;
		uctr[i] = 0;
		}*/

	if(secondOrderRequested)
	{
		physics.getPrimitiveFromConserved(ucl, uctl);
		physics.getPrimitiveFromConserved(ucr, uctr);

		assert(gradl);
		assert(gradr);

		for(int i = 0; i < ndim; i++)
			for(int j = 0; j < nvars; j++) {
				gradtl[i*ndim+j] = gradl[i*ndim+j];
				gradtr[i*ndim+j] = gradr[i*ndim+j];
			}
			
		/* get one-sided temperature gradients from one-sided primitive gradients
		 * and discard grad p in favor of grad T.
		 */
		for(int j = 0; j < ndim; j++) {
			gradtl[j*ndim+ nvars-1] = physics.getGradTemperature(uctl[0], gradl[j*ndim],
			                                                     uctl[nvars-1], gradl[j*ndim+nvars-1]);
			gradtr[j*ndim+ nvars-1] = physics.getGradTemperature(uctr[0], gradr[j*ndim],
			                                                     uctr[nvars-1], gradr[j*ndim+nvars-1]);
		}

		// convert cell-centred variables to primitive-2
		uctl[nvars-1] = physics.getTemperature(uctl[0], uctl[nvars-1]);
		uctr[nvars-1] = physics.getTemperature(uctr[0], uctr[nvars-1]);
	}
	else
	{
		// convert cell-centred variables to primitive-2
		physics.getPrimitive2FromConserved(ucl, uctl);
		physics.getPrimitive2FromConserved(ucr, uctr);
	}
}

template<typename scalar, int ndim, int nvars, bool constVisc>
void computeViscousFlux(const IdealGasPhysics<scalar>& physics, const scalar *const n,
                        const scalar grad[ndim][nvars],
                        const scalar *const ul, const scalar *const ur,
                        scalar *const __restrict vflux)
{
	static_assert(ndim == NDIM, "3D not implemented yet.");
	static_assert(nvars == ndim+2, "Only single-phase ideal gas is supported.");
	
	// Non-dimensional dynamic viscosity divided by free-stream Reynolds number
	const scalar muRe = constVisc ? 
			physics.getConstantViscosityCoeff() 
		:
			0.5*( physics.getViscosityCoeffFromConserved(ul)
			+ physics.getViscosityCoeffFromConserved(ur) );
	
	// Non-dimensional thermal conductivity
	const scalar kdiff = physics.getThermalConductivityFromViscosity(muRe); 

	scalar stress[ndim][ndim];
	for(int i = 0; i < ndim; i++)
		for(int j = 0; j < ndim; j++)
			stress[i][j] = 0;
	
	physics.getStressTensor(muRe, grad, stress);

	vflux[0] = 0;
	
	for(int i = 0; i < ndim; i++)
	{
		vflux[i+1] = 0;
		for(int j = 0; j < ndim; j++)
			vflux[i+1] -= stress[i][j] * n[j];
	}

	// for the energy dissipation, compute avg velocities first
	scalar vavg[ndim];
	for(int j = 0; j < ndim; j++)
		vavg[j] = 0.5*( ul[j+1]/ul[0] + ur[j+1]/ur[0] );

	vflux[nvars-1] = 0;
	for(int i = 0; i < ndim; i++)
	{
		scalar comp = 0;
		
		for(int j = 0; j < ndim; j++)
			comp += stress[i][j]*vavg[j];       // dissipation by momentum flux (friction etc)
		
		comp += kdiff*grad[i][nvars-1];         // dissipation by heat flux

		vflux[nvars-1] -= comp * n[i];
	}
}

template void
getPrimitive2StateAndGradients<double,NDIM,true>(const IdealGasPhysics<double>& physics,
                                                 const double *const ucl, const double *const ucr,
                                                 const double *const gradl, const double *const gradr,
                                                 double *const uctl, double *const uctr,
                                                 double *const gradtl, double *const gradtr);
template void
getPrimitive2StateAndGradients<double,NDIM,false>(const IdealGasPhysics<double>& physics,
                                                  const double *const ucl, const double *const ucr,
                                                  const double *const gradl, const double *const gradr,
                                                  double *const uctl, double *const uctr,
                                                  double *const gradtl, double *const gradtr);

template void
computeViscousFlux<double,NDIM,NVARS,true>(const IdealGasPhysics<double>& physics,
                                           const double *const n,
                                           const double grad[NDIM][NVARS],
                                           const double *const ul, const double *const ur,
                                           double *const __restrict vflux);
template void
computeViscousFlux<double,NDIM,NVARS,false>(const IdealGasPhysics<double>& physics,
                                           const double *const n,
                                           const double grad[NDIM][NVARS],
                                           const double *const ul, const double *const ur,
                                           double *const __restrict vflux);
}
