/** \file anumericalflux.hpp
 * \brief Numerical flux schemes for the compressible Euler equations.
 * \author Aditya Kashi
 * \date March 2015, September 2017
 */

#ifndef ANUMERICALFLUX_H
#define ANUMERICALFLUX_H 1

#include "aconstants.hpp"
#include "aphysics.hpp"

namespace acfd {

/// Abstract class from which to derive all numerical flux classes
/** The class is such that given the left and right states and a face normal, 
 * the numerical flux and its Jacobian w.r.t. left and right states is computed.
 */
class InviscidFlux
{
public:
	/// Sets the physics context for the inviscid flux scheme
	InviscidFlux(const IdealGasPhysics *const physcs);

	/** Computes flux across a face with
	 * \param[in] uleft is the vector of left states for the face
	 * \param[in] uright is the vector of right states for the face
	 * \param[in] n is the normal vector to the face
	 * \param[in|out] flux contains the computed flux
	 */
	virtual void get_flux(const a_real *const uleft, const a_real *const uright, 
			const a_real* const n, 
			a_real *const flux) = 0;

	/// Computes the Jacobian of inviscid flux across a face w.r.t. both left and right states
	/** dfdl is the `lower' block formed by the coupling between elements adjoining the face,
	 * while dfdr is the `upper' block.
	 * The negative of the lower block is the contribution to diagonal block of left cell, and
	 * the negative of the upper block is the contribution to diagonal block of right cell.
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	virtual void get_jacobian(const a_real *const uleft, const a_real *const uright, 
			const a_real* const n,
			a_real *const dfdl, a_real *const dfdr) = 0;

	virtual ~InviscidFlux();

protected:
	const IdealGasPhysics *const physics;        ///< Functionality replated to gas constitutive law
	const a_real g;                              ///< Adiabatic index
};

class LocalLaxFriedrichsFlux : public InviscidFlux
{
public:
	LocalLaxFriedrichsFlux(const IdealGasPhysics *const analyticalflux);
	
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const a_real *const uleft, const a_real *const uright, const a_real* const n, 
			a_real *const flux);
	
	/** Currently computes an approximate Jacobian with frozen spectral radius.
	 * This has been found to perform no worse than the exact Jacobian for inviscid flows.
	 * \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const a_real *const uleft, const a_real *const uright, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);

	/** Computes the exact Jacobian.
	 * This has been done to make the frozen Jacobian default.
	 */
	void get_jacobian_2(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
};

/// Van-Leer flux-vector-splitting
class VanLeerFlux : public InviscidFlux
{
public:
	VanLeerFlux(const IdealGasPhysics *const analyticalflux);
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
};

/// Liou-Steffen AUSM flux-vector-splitting
/** I call this flux vector splitting (FVS) for want of a better term; even though it is not
 * FVS strictly speaking, it's close enough. It's a FVS of the convective and pressure fluxes
 * separately.
 * \warning The Jacobian does not work, except for useless small CFL numbers.
 */
class AUSMFlux : public InviscidFlux
{
public:
	AUSMFlux(const IdealGasPhysics *const analyticalflux);
	
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	
	/** \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
};

/// AUSM+ flux
/** Supposed to be better than AUSMFlux, but is not, for some reason.
 */
class AUSMPlusFlux : public InviscidFlux
{
public:
	AUSMPlusFlux(const IdealGasPhysics *const analyticalflux);
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
};

/// Abstract class for fluxes which depend on Roe-averages
class RoeAverageBasedFlux : public InviscidFlux
{
public:
	RoeAverageBasedFlux(const IdealGasPhysics *const analyticalflux);
	virtual void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux) = 0;
	virtual void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr) = 0;

protected:

	/// Computes Roe-averaged quantities
	void getRoeAverages(const a_real ul[NVARS], const a_real ur[NVARS], const a_real n[NDIM],
		const a_real vxi, const a_real vyi, const a_real Hi,
		const a_real vxj, const a_real vyj, const a_real Hj,
		a_real& Rij, a_real& rhoij, a_real& vxij, a_real& vyij, a_real &vm2ij, a_real& vnij,
		a_real& Hij, a_real& cij)
	{
		Rij = std::sqrt(ur[0]/ul[0]);
		rhoij = Rij*ul[0];
		vxij = (Rij*vxj + vxi)/(Rij + 1.0);
		vyij = (Rij*vyj + vyi)/(Rij + 1.0);
		Hij = (Rij*Hj + Hi)/(Rij + 1.0);
		vm2ij = vxij*vxij + vyij*vyij;
		vnij = vxij*n[0] + vyij*n[1];
		cij = sqrt( (g-1.0)*(Hij - vm2ij*0.5) );
	}
	
	/// Computes derivatives of Roe-averaged quantities w.r.t. conserved variables
	/** \note The output vectors are assigned to, so any prior contents are lost.
	 */
	void getJacobiansRoeAveragesWrtConserved(
		const a_real ul[NVARS], const a_real ur[NVARS], const a_real n[NDIM],
		const a_real vxi, const a_real vyi, const a_real Hi,
		const a_real vxj, const a_real vyj, const a_real Hj,
		const a_real dvxi[NVARS], const a_real dvyi[NVARS], const a_real dHi[NVARS],
		const a_real dvxj[NVARS], const a_real dvyj[NVARS], const a_real dHj[NVARS],
		a_real dRiji[NVARS], a_real drhoiji[NVARS], a_real dvxiji[NVARS], a_real dvyiji[NVARS],
		a_real dvm2iji[NVARS], a_real dvniji[NVARS], a_real dHiji[NVARS], a_real dciji[NVARS],
		a_real dRijj[NVARS], a_real drhoijj[NVARS], a_real dvxijj[NVARS], a_real dvyijj[NVARS],
		a_real dvm2ijj[NVARS], a_real dvnijj[NVARS], a_real dHijj[NVARS], a_real dcijj[NVARS] );
};

/// Roe-Pike flux-difference splitting
/** From Blazek \cite{blazek}.
 */
class RoeFlux : public RoeAverageBasedFlux
{
public:
	RoeFlux(const IdealGasPhysics *const analyticalflux);
	
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	
	/** \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
protected:
	/// Entropy fix parameter
	const a_real fixeps;
};

/// Harten Lax Van-Leer numerical flux
/** Decent for inviscid flows.
 * \cite invflux_batten
 */
class HLLFlux : public RoeAverageBasedFlux
{
	/// Computes the Jacobian of the numerical flux w.r.t. left state
	void getFluxJac_left(const a_real *const ul, const a_real *const ur, const a_real *const n, 
			a_real *const flux, a_real *const fluxd);
	/// Computes the Jacobian of the numerical flux w.r.t. right state
	void getFluxJac_right(const a_real *const ul, const a_real *const ur, const a_real *const n, 
			a_real *const flux, a_real *const fluxd);

public:
	HLLFlux(const IdealGasPhysics *const analyticalflux);
	
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	
	/** \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);

	/// Computes both the flux and the 2 flux Jacobians
	void get_flux_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux, a_real *const dfdl, a_real *const dfdr);
	
	void get_jacobian_2(const a_real *const ul, const a_real *const ur, 
			const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
};

/// Harten Lax Van-Leer numerical flux with contact restoration by Toro
/** Implemented as described by Batten et al. \cite invflux_hllc_batten
 * Good for both inviscid and viscous flows.
 */
class HLLCFlux : public RoeAverageBasedFlux
{
public:
	HLLCFlux(const IdealGasPhysics *const analyticalflux);
	
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	
	/** \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
	
	/// Computes both the flux and the 2 flux Jacobians
	void get_flux_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux, a_real *const dfdl, a_real *const dfdr);

protected:
	/// Computes the averaged state between the waves in the Riemann fan
	/** \param[in] u The state outside the Riemann fan
	 * \param[in] n Normal to the face
	 * \param[in] vn Normal velocity corresponding to u and n
	 * \param[in[ p Pressure corresponding to u
	 * \param[in] ss Signal speed separating u from the averaged state
	 * \param[in] sm Contact wave speed
	 * \param[in|out] ustr The output average state
	 */
	void getStarState(const a_real u[NVARS], const a_real n[NDIM],
		const a_real vn, const a_real p, 
		const a_real ss, const a_real sm,
		a_real *const __restrict ustr) const;

	/// Computes the averaged state between the waves in the Riemann fan
	/// and corresponding Jacobians
	/** Jacobians of the average state w.r.t. both "this" and the "other" initial states
	 * are computed.
	 * \param[in] u "This" state outside the Riemann fan
	 * \param[in] n Normal to the face
	 * \param[in] vn Normal velocity corresponding to u and n
	 * \param[in[ p Pressure corresponding to u
	 * \param[in] ss Signal speed separating u from the averaged state
	 * \param[in] sm Contact wave speed
	 * \param[in] dvn Derivatives of normal velocity corresponding to this state (vn)
	 * \param[in] dp Derivatives of pressure corresponding to this state (p)
	 * \param[in] dssi Derivatives of signal speed ss w.r.t. this (u) state
	 * \param[in] dsmi Derivatives of contact speed sm w.r.t. this (u) state
	 * \param[in] dssj Derivatives of signal speed ss w.r.t. "other" state
	 * \param[in] dsmj Derivatives of contact speed sm w.r.t. "other" state
	 * \param[in|out] ustr The output average state
	 * \param[in|out] dustri Jacobian of ustr w.r.t. this initial state
	 * \param[in|out] dustrj Jacobian of ustr w.r.t. other initial state
	 */
	void getStarStateAndJacobian(const a_real u[NVARS], const a_real n[NDIM],
		const a_real vn, const a_real p, 
		const a_real ss, const a_real sm,
		const a_real dvn[NVARS], const a_real dp[NVARS], 
		const a_real dssi[NDIM], const a_real dsmi[NDIM],
		const a_real dssj[NDIM], const a_real dsmj[NDIM],
		a_real ustr[NVARS],
		a_real dustri[NVARS][NVARS], 
		a_real dustrj[NVARS][NVARS]) const __attribute((always_inline));
};

} // end namespace acfd

#endif
