/** \file
 * \brief Implementation of physical transormations requried for viscous flux computation
 * \author Aditya Kashi
 */

#include "viscousphysics.hpp"

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

template<typename scalar, int ndim, bool secondOrderRequested>
void getPrimitive2StatesAndGradients(const IdealGasPhysics<scalar>& physics,
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
				gradtl[i*nvars+j] = gradl[i*nvars+j];
				gradtr[i*nvars+j] = gradr[i*nvars+j];
			}

		/* get one-sided temperature gradients from one-sided primitive gradients
		 * and discard grad p in favor of grad T.
		 */
		for(int j = 0; j < ndim; j++) {
			gradtl[j*nvars+ nvars-1] = physics.getGradTemperature(uctl[0], gradl[j*nvars],
			                                                     uctl[nvars-1], gradl[j*nvars+nvars-1]);
			gradtr[j*nvars+ nvars-1] = physics.getGradTemperature(uctr[0], gradr[j*nvars],
			                                                     uctr[nvars-1], gradr[j*nvars+nvars-1]);
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

template<typename scalar, int ndim, int nvars, bool constVisc>
void computeViscousFluxJacobian(const IdealGasPhysics<scalar>& jphy,
                                const scalar *const n,
                                const scalar *const ul, const scalar *const ur,
                                const scalar grad[ndim][nvars],
                                const scalar dgradl[ndim][nvars][nvars],
                                const scalar dgradr[ndim][nvars][nvars],
                                scalar *const __restrict dvfi, scalar *const __restrict dvfj)
{
	static_assert(ndim == NDIM, "3D not implemented yet.");
	static_assert(nvars == ndim+2, "Only single-phase ideal gas is supported.");

	scalar vflux[nvars];             // output variable to be differentiated

	// Non-dimensional dynamic viscosity divided by free-stream Reynolds number
	const scalar muRe = constVisc ?
			jphy.getConstantViscosityCoeff()
		:
			0.5*( jphy.getViscosityCoeffFromConserved(ul)
			+ jphy.getViscosityCoeffFromConserved(ur) );

	// Non-dimensional thermal conductivity
	const scalar kdiff = jphy.getThermalConductivityFromViscosity(muRe);

	scalar dmul[nvars], dmur[nvars], dkdl[nvars], dkdr[nvars];
	for(int k = 0; k < nvars; k++) {
		dmul[k] = 0; dmur[k] = 0; dkdl[k] = 0; dkdr[k] = 0;
	}

	if(!constVisc) {
		jphy.getJacobianSutherlandViscosityWrtConserved(ul, dmul);
		jphy.getJacobianSutherlandViscosityWrtConserved(ur, dmur);
		for(int k = 0; k < nvars; k++) {
			dmul[k] *= 0.5;
			dmur[k] *= 0.5;
		}
		jphy.getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(dmul, dkdl);
		jphy.getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(dmur, dkdr);
	}

	scalar stress[ndim][ndim], dstressl[ndim][ndim][nvars], dstressr[ndim][ndim][nvars];
	for(int i = 0; i < ndim; i++)
		for(int j = 0; j < ndim; j++)
		{
			stress[i][j] = 0;
			for(int k = 0; k < nvars; k++) {
				dstressl[i][j][k] = 0;
				dstressr[i][j][k] = 0;
			}
		}

	jphy.getJacobianStress(muRe, dmul, grad, dgradl, stress, dstressl);
	jphy.getJacobianStress(muRe, dmur, grad, dgradr, stress, dstressr);

	vflux[0] = 0;

	for(int i = 0; i < ndim; i++)
	{
		vflux[i+1] = 0;
		for(int j = 0; j < ndim; j++)
		{
			vflux[i+1] -= stress[i][j] * n[j];

			for(int k = 0; k < nvars; k++) {
				dvfi[(i+1)*nvars+k] += dstressl[i][j][k] * n[j];
				dvfj[(i+1)*nvars+k] -= dstressr[i][j][k] * n[j];
			}
		}
	}

	// for the energy dissipation, compute avg velocities first
	scalar vavg[ndim], dvavgl[ndim][nvars], dvavgr[ndim][nvars];
	for(int j = 0; j < ndim; j++)
	{
		vavg[j] = 0.5*( ul[j+1]/ul[0] + ur[j+1]/ur[0] );

		for(int k = 0; k < nvars; k++) {
			dvavgl[j][k] = 0;
			dvavgr[j][k] = 0;
		}

		dvavgl[j][0] = -0.5*ul[j+1]/(ul[0]*ul[0]);
		dvavgr[j][0] = -0.5*ur[j+1]/(ur[0]*ur[0]);

		dvavgl[j][j+1] = 0.5/ul[0];
		dvavgr[j][j+1] = 0.5/ur[0];
	}

	vflux[nvars-1] = 0;
	for(int i = 0; i < ndim; i++)
	{
		scalar comp = 0;
		scalar dcompl[nvars], dcompr[nvars];
		for(int k = 0; k < nvars; k++) {
			dcompl[k] = 0;
			dcompr[k] = 0;
		}

		for(int j = 0; j < ndim; j++)
		{
			comp += stress[i][j]*vavg[j];       // dissipation by momentum flux (friction)

			for(int k = 0; k < nvars; k++) {
				dcompl[k] += dstressl[i][j][k]*vavg[j] + stress[i][j]*dvavgl[j][k];
				dcompr[k] += dstressr[i][j][k]*vavg[j] + stress[i][j]*dvavgr[j][k];
			}
		}

		comp += kdiff*grad[i][nvars-1];         // dissipation by heat flux

		for(int k = 0; k < nvars; k++) {
			dcompl[k] += dkdl[k]*grad[i][nvars-1] + kdiff*dgradl[i][nvars-1][k];
			dcompr[k] += dkdr[k]*grad[i][nvars-1] + kdiff*dgradr[i][nvars-1][k];
		}

		vflux[nvars-1] -= comp * n[i];

		for(int k = 0; k < nvars; k++) {
			dvfi[(nvars-1)*nvars+k] += dcompl[k] * n[i];
			dvfj[(nvars-1)*nvars+k] -= dcompr[k] * n[i];
		}
	}
}

template void
getPrimitive2StatesAndGradients<freal,NDIM,true>(const IdealGasPhysics<freal>& physics,
                                                  const freal *const ucl, const freal *const ucr,
                                                  const freal *const gradl, const freal *const gradr,
                                                  freal *const uctl, freal *const uctr,
                                                  freal *const gradtl, freal *const gradtr);
template void
getPrimitive2StatesAndGradients<freal,NDIM,false>(const IdealGasPhysics<freal>& physics,
                                                   const freal *const ucl, const freal *const ucr,
                                                   const freal *const gradl, const freal *const gradr,
                                                   freal *const uctl, freal *const uctr,
                                                   freal *const gradtl, freal *const gradtr);

template void
computeViscousFlux<freal,NDIM,NVARS,true>(const IdealGasPhysics<freal>& physics,
                                           const freal *const n,
                                           const freal grad[NDIM][NVARS],
                                           const freal *const ul, const freal *const ur,
                                           freal *const __restrict vflux);
template void
computeViscousFlux<freal,NDIM,NVARS,false>(const IdealGasPhysics<freal>& physics,
                                           const freal *const n,
                                           const freal grad[NDIM][NVARS],
                                           const freal *const ul, const freal *const ur,
                                           freal *const __restrict vflux);

template void
computeViscousFluxJacobian<freal,NDIM,NVARS,true>(const IdealGasPhysics<freal>& jphy,
                                                   const freal *const n,
                                                   const freal *const ul, const freal *const ur,
                                                   const freal grad[NDIM][NVARS],
                                                   const freal dgradl[NDIM][NVARS][NVARS],
                                                   const freal dgradr[NDIM][NVARS][NVARS],
                                                   freal *const __restrict dvfi,
                                                   freal *const __restrict dvfj);

template void
computeViscousFluxJacobian<freal,NDIM,NVARS,false>(const IdealGasPhysics<freal>& jphy,
                                                    const freal *const n,
                                                    const freal *const ul, const freal *const ur,
                                                    const freal grad[NDIM][NVARS],
                                                    const freal dgradl[NDIM][NVARS][NVARS],
                                                    const freal dgradr[NDIM][NVARS][NVARS],
                                                    freal *const __restrict dvfi,
                                                    freal *const __restrict dvfj);

//CHANGE HERE
#ifdef USE_ADOLC
template void
getPrimitive2StatesAndGradients<adouble,NDIM,true>(const IdealGasPhysics<adouble>& physics,
                                                  const adouble *const ucl, const adouble *const ucr,
                                                  const adouble *const gradl, const adouble *const gradr,
                                                  adouble *const uctl, adouble *const uctr,
                                                  adouble *const gradtl, adouble *const gradtr);

template void
getPrimitive2StatesAndGradients<adouble,NDIM,false>(const IdealGasPhysics<adouble>& physics,
                                                  const adouble *const ucl, const adouble *const ucr,
                                                  const adouble *const gradl, const adouble *const gradr,
                                                  adouble *const uctl, adouble *const uctr,
                                                  adouble *const gradtl, adouble *const gradtr);

template void
computeViscousFlux<adouble,NDIM,NVARS,true>(const IdealGasPhysics<adouble>& physics,
                                           const adouble *const n,
                                           const adouble grad[NDIM][NVARS],
                                           const adouble *const ul, const adouble *const ur,
                                           adouble *const __restrict vflux);

template void
computeViscousFlux<adouble,NDIM,NVARS,false>(const IdealGasPhysics<adouble>& physics,
                                           const adouble *const n,
                                           const adouble grad[NDIM][NVARS],
                                           const adouble *const ul, const adouble *const ur,
                                           adouble *const __restrict vflux);
#endif
}
