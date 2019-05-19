/** \file aphysics.hpp
 * \brief Provides analytical flux computation contexts
 *
 * Some functions have been defined in this file itself for inlining;
 * others have been defined in the [implementation file](\ref aphysics.cpp).
 *
 * \author Aditya Kashi
 * \date 2017 May 12
 */

#ifndef APHYSICS_H
#define APHYSICS_H

#include "aconstants.hpp"
#include "mathutils.hpp"

namespace fvens {

/// Abstract class providing analytical fluxes and their Jacobians etc
template <typename scalar>
class Physics
{
public:
	virtual ~Physics();

	/// Computes the flux vector along some direction
	virtual void getDirectionalFluxFromConserved(const scalar *const u, const scalar* const n,
			scalar *const flux) const = 0;

	/// Computes the Jacobian of the flux along some direction, at the given state
	virtual void getJacobianDirectionalFluxWrtConserved(const scalar *const u,
			const scalar* const n,
			scalar *const dfdu) const = 0;
};

/// Flow-physics-related computation for single-phase ideal gas
/** The non-dimensionalization assumed is from free-stream velocities and temperatures,
 * as given in section 4.14.2 of \cite matatsuka.
 * Note that several computations here depend on the exact non-dimensionalization scheme used.
 * Unless otherwise specified, all methods are assumed to accept non-dimensional quantities and
 * give non-dimensional output variables.
 *
 * "Primitive-2" variables are density, velocities and temperature, as opposed to
 * "primitive" variables which are density, velocities and pressure.
 *
 * There are also come calculation related to the stress-velocity properties of a Newtonian fluid.
 */
template <typename scalar>
class IdealGasPhysics : public Physics<scalar>
{
public:
	/// Number of conserved variables
	static constexpr int nvars = NDIM+2;

	IdealGasPhysics(const freal _g, const freal M_inf,
			const freal T_inf, const freal Re_inf, const freal _Pr);

	/// Returns an array containing the non-dimensional free-stream state
	/** \param aoa The angle of attack in radians
	 */
	std::array<freal,NVARS> compute_freestream_state(const freal aoa) const;

	/// Computes flux in a given direction efficiently using specific data
	/** Note that this function is independent of what kind of gas it is.
	 * \param[in] uc Vector of conserved variables
	 * \param[in] n Vector along the direction in which the flux vector is needed
	 * \param[in] vn Normal velocity (w.r.t. the the normal n above)
	 * \param[in] p Pressure
	 * \param[in|out] flux Output vector for the flux; note that any pre-existing contents
	 *   will be replaced!
	 */
	void getDirectionalFlux(const scalar *const uc, const scalar *const n,
	                        const scalar vn, const scalar p, scalar *const __restrict flux) const;

	/// Computes the analytical convective flux across a face oriented in some direction
	void getDirectionalFluxFromConserved(const scalar *const u, const scalar* const n,
	                                     scalar *const __restrict flux) const;

	/// Computes the Jacobian of the flux along some direction, at the given state
	/** The flux Jacobian matrix dfdu is assumed stored in a row-major 1-dimensional array.
	 */
	void getJacobianDirectionalFluxWrtConserved(const scalar *const u, const scalar* const n,
	                                            scalar *const __restrict dfdu) const;

	/// Computes derivatives of the squared velocity magnitude w.r.t. conserved variables
	/** \warning uc and dvmag2 must not point to the same memory locations.
	 */
	void getJacobianVmag2WrtConserved(const scalar *const uc,
		scalar *const __restrict dvmag2) const;

	/// Outputs various quantities, especially needed by numerical fluxes
	/** \param[in] uc Conserved variables
	 * \param[in] n Normal vector
	 * \param[out] v Velocity vector (of length NDIM)
	 * \param[out] vn Normal velocity
	 * \param[out] vm2 Square of velocity magnitude
	 * \param[out] p Pressure
	 * \param[out] H Specific enthalpy
	 */
	void getVarsFromConserved(const scalar *const uc, const scalar *const n,
	                          scalar *const v,
	                          scalar& vn,
	                          scalar& p, scalar& H ) const;

	/// Computes derivatives of variables computed in getVarsFromConserved
	/** \param[in] uc Vector of conserved variables
	 * \param[in] n Normal vector
	 * \param[in,out] dv is a NDIM x NVARS matrix stored as a row-major 1D array
	 * \warning The result is added to the original content of the  output variables!
	 */
	void getJacobianVarsWrtConserved(const scalar *const uc, const scalar *const n,
		scalar *const dv, scalar dvn[NVARS],
		scalar dp[NVARS], scalar dH[NVARS]) const;

	/// Returns the non-dimensionalized free-stream pressure
	scalar getFreestreamPressure() const;

	/// Computes pressure from internal energy - here it's the ideal gas relation
	scalar getPressure(const scalar internalenergy) const;

	/// Computes pressure from conserved variables
	scalar getPressureFromConserved(const scalar *const uc) const;

	/// Computes the change in pressure caused by a (large) finite change in the conserved variables
	/** \param[in] u The conserve variables from which the change takes place
	 * \param[in] delta_u The change in the conserved variables
	 * \return The resulting exact change in pressure
	 */
	scalar getDeltaPressureFromConserved(const scalar *const u, const scalar *const delta_u) const;

	/// Computes pressure gradient from conserved variables and their gradients
	scalar getGradPressureFromConservedAndGradConserved(const scalar *const uc,
		const scalar *const guc) const;

	/// Derivative of pressure w.r.t. conserved variables
	void getJacobianPressureWrtConserved(const scalar *const uc, scalar *const __restrict dp) const;

	/// Gives the pressure more efficiently using an additional input - the momemtum magnitude
	void getJacobianPressureWrtConserved(const scalar *const uc, const scalar rho2vmag2,
			scalar *const __restrict dp) const;

	/// Computes temperature from density and pressure - depends on non-dimensionalization.
	scalar getTemperature(const scalar rho, const scalar p) const;

	/// Computes derivatives of temperature w.r.t. either conserved, primitive or primitive2
	/** \note Derivatives can be computed w.r.t. any variable-set that
	 * has density as the first variable; the specific variables w.r.t. which derivatives are
	 * computed depends on dp.
	 */
	void getJacobianTemperature(const scalar rho,
			const scalar p, const scalar *const dp,
			scalar *const __restrict dT) const;

	/// Computes speed of sound from density and pressure
	scalar getSoundSpeed(const scalar rho, const scalar p) const;

	/// Derivative of sound speed
	/** The variable-set w.r.t. which the differentiation happens is the same as that w.r.t.
	 * which the pressure was differentiated for getting dp, similar to the case with
	 * \ref getJacobianTemperature.
	 */
	void getJacobianSoundSpeed(const scalar rho,
			const scalar p, const scalar dp[NVARS], const scalar c,
			scalar *const __restrict dc) const;

	/// Computes speed of sound from conserved variables
	scalar getSoundSpeedFromConserved(const scalar *const uc) const;

	/// Derivative of sound speed w.r.t. conserved variables
	void getJacobianSoundSpeedWrtConserved(const scalar *const uc,
			scalar *const __restrict dc) const;

	/// Derivatives of normal Mach number w.r.t. conserved variables
	void getJacobianMachNormalWrtConserved(const scalar *const uc,
		const scalar *const n,
		scalar *const __restrict dmn) const;

	/// Computes an entropy \f$ p/ \rho^\gamma \f$ from conserved variables
	scalar getEntropyFromConserved(const scalar *const uc) const;

	/// Compute energy from pressure, density and square of magnitude of velocity
	scalar getEnergyFromPressure(const scalar p, const scalar d, const scalar vmag2) const;

	/// Compute energy from temperature, density and square of magnitude of velocity
	scalar getEnergyFromTemperature(const scalar T, const scalar d, const scalar vmag2) const;

	/// Computes derivatives of total energy from derivatives of temperature and |v|^2
	/** Can compute the derivatives w.r.t. any variable-set as long as
	 * the first variable is density. That variable set is decided by what variables the
	 * T and |v|^2 were differentiated with respect to.
	 * \param[in,out] drhoE The derivative of the total energy per unit volume is *added to* this
	 */
	void getJacobianEnergyFromJacobiansTemperatureVmag2(
		const scalar T, const scalar d, const scalar vmag2,
		const scalar *const dT, const scalar *const dvmag2,
		scalar *const drhoE) const;

	/// Computes total energy from primitive variabes
	scalar getEnergyFromPrimitive(const scalar *const up) const;

	/// Computes total energy from a vector of density, velocities and temperature
	/** All quantities are non-dimensional.
	 */
	scalar getEnergyFromPrimitive2(const scalar *const upt) const;

	/// Convert conserved variables to primitive variables (density, velocities, pressure)
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage. The output is assigned to, not updated.
	 */
	void getPrimitiveFromConserved(const scalar *const uc, scalar *const up) const;

	/// Convert conserved variables to primitive-2 variables; depends on non-dimensionalization
	/** The output is assigned to, not updated. So both input and output can point to the same
	 * storage.
	 */
	void getPrimitive2FromConserved(const scalar *const uc, scalar *const up) const;

	/// Computes the Jacobian matrix of the conserved-to-primitive-2 transformation
	/** \f$ \partial \mathbf{u}_{prim2} / \partial \mathbf{u}_{cons} \f$.
	 * The output is stored as 1D rowmajor.
	 *
	 * \warning The Jacobian is *added* to the output jac. If it has garbage at the outset,
	 * it will contain garbage at the end.
	 */
	void getJacobianPrimitive2WrtConserved(const scalar *const uc,
			scalar *const __restrict jac) const;

	/// Convert primitive variables to conserved
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage.
	 */
	void getConservedFromPrimitive(const scalar *const up, scalar *const uc) const;

	/// Computes density from pressure and temperature using ideal gas relation;
	/// All quantities are non-dimensional
	scalar getDensityFromPressureTemperature(const scalar pressure, const scalar temperature) const;

	/// Computes derivatives of density from derivatives of pressure and temperature
	void getJacobianDensityFromJacobiansPressureTemperature(
			const scalar pressure, const scalar temperature,
			const scalar *const dp, const scalar *const dT, scalar *const drho) const;

	/// Computes non-dimensional temperature from non-dimensional conserved variables
	/** \sa IdealGasPhysics
	 */
	scalar getTemperatureFromConserved(const scalar *const uc) const;

	/// Computes derivatives of temperature w.r.t. conserved variables
	/** \param[in|out] dT The derivatives are added to dT, which is not zeroed initially.
	 */
	void getJacobianTemperatureWrtConserved(const scalar *const uc,
			scalar *const __restrict dT) const;

	/// Computes non-dimensional temperature from non-dimensional primitive variables
	scalar getTemperatureFromPrimitive(const scalar *const up) const;

	/// Computes non-dim temperature gradient from non-dim density, pressure and their gradients
	scalar getGradTemperature(const scalar rho, const scalar gradrho,
		const scalar p, const scalar gradp) const;

	/// Compute non-dim temperature spatial derivative
	/// from non-dim conserved variables and their spatial derivatives
	scalar getGradTemperatureFromConservedAndGradConserved(const scalar *const uc,
			const scalar *const guc) const;

	/// Compute spatial derivative of non-dim temperature from non-dim conserved variables and
	/// spatial derivatives of non-dim *primitive* variables
	scalar getGradTemperatureFromConservedAndGradPrimitive(const scalar *const uc,
			const scalar *const gup) const;

	/// Get primitive-2 gradients from conserved variables and their gradients
	/** \warning Here, the output array gp is overwritten, rather than added to.
	 */
	void getGradPrimitive2FromConservedAndGradConserved(const scalar *const __restrict uc,
			const scalar *const guc, scalar *const gp) const;

	/// Computes non-dimensional viscosity coeff using Sutherland's law from temperature
	/** By viscosity coefficient, we mean dynamic viscosity divided by
	 * the free-stream Reynolds number.
	 */
	scalar getViscosityCoeffFromTemperature(const scalar T) const;

	/// Computes non-dimensional viscosity coeff using Sutherland's law from conserved variables
	/** This is the dynamic viscosity divided by the Reynolds number
	 * when non-dimensionalized as stated in IdealGasPhysics.
	 * Note that divergence terms must still be multiplied further by -2/3
	 * and diagonal stress terms by 2.
	 */
	scalar getViscosityCoeffFromConserved(const scalar *const uc) const;

	/// Computes derivatives of Sutherland's dynamic viscosity coeff w.r.t. conserved variables
	void getJacobianSutherlandViscosityWrtConserved(const scalar *const uc,
			scalar *const __restrict dmu) const;

	/// Returns non-dimensional free-stream viscosity coefficient
	scalar getConstantViscosityCoeff() const;

	/// Computes non-dimensional conductivity from non-dimensional dynamic viscosity coeff
	scalar getThermalConductivityFromViscosity(const scalar muhat) const;

	/** \brief Computes derivatives of non-dim thermal conductivity w.r.t. conserved variables
	 * given derivatives of the non-dim viscosity coeff w.r.t. conserved variables.
	 */
	void getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(
			const scalar *const dmuhat, scalar *const __restrict dkhat) const;

	/// Computes the stress tensor using gradients of primitive variables
	/** Can also use gradients of primitive-2 variables - it only uses velocity gradients.
	 * The result is assigned to the output array stress, so prior contents are lost.
	 * \param[in] mu Non-dimensional viscosity divided by free-stream Reynolds number
	 * \param[in] grad Gradients of primitve variables
	 * \param[in,out] stress Components of the stress tensor on output
	 */
	void getStressTensor(const scalar mu, const scalar grad[NDIM][NVARS],
			scalar stress[NDIM][NDIM]) const;

	/// Computes Jacobian of stress tensor using Jacobian of gradients of primitive variables
	/** Assigns the computed Jacobian to the output array components so prior contents are lost.
	 */
	void getJacobianStress(const scalar mu, const scalar *const dmu,
		const scalar grad[NDIM][NVARS], const scalar dgrad[NDIM][NVARS][NVARS],
		scalar stress[NDIM][NDIM], scalar dstress[NDIM][NDIM][NVARS]) const;

	/// Adiabatic constant
	const freal g;
	/// Free-stream Mach number
	const freal Minf;
	/// Free-stream static temperature
	const freal Tinf;
	/// Free-stream Reynolds number
	const freal Reinf;
	/// Prandtl number
	const freal Pr;
	/// Sutherland constant
	const freal sC;
};

}
#endif
