/** \file anumericalflux.hpp
 * \brief Numerical flux schemes for the compressible Euler equations.
 * \author Aditya Kashi
 * \date March 2015, September 2017
 */

#ifndef ANUMERICALFLUX_H
#define ANUMERICALFLUX_H 1

#include "aconstants.hpp"
#include "physics/aphysics.hpp"

namespace fvens {

/// Abstract class from which to derive all numerical flux classes
/** The class is such that given the left and right states and a face normal,
 * the numerical flux and its Jacobian w.r.t. left and right states is computed.
 */
template <typename scalar, typename j_real = a_real>
class InviscidFlux
{
public:
	/// Sets the physics context for the inviscid flux scheme
	InviscidFlux(const IdealGasPhysics<scalar> *const physcs);

	/** Computes flux across a face
	 * \param[in] uleft is the vector of left states for the face
	 * \param[in] uright is the vector of right states for the face
	 * \param[in] n is the normal vector to the face
	 * \param[in,out] flux contains the computed flux
	 */
	virtual void get_flux(const scalar *const uleft, const scalar *const uright,
			const scalar* const n,
			scalar *const flux) const = 0;

	/// Computes the Jacobian of inviscid flux across a face w.r.t. both left and right states
	/** dfdl is the `lower' block formed by the coupling between elements adjoining the face,
	 * while dfdr is the `upper' block.
	 * The negative of the lower block is the contribution to diagonal block of left cell, and
	 * the negative of the upper block is the contribution to diagonal block of right cell.
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	virtual void get_jacobian(const j_real *const uleft, const j_real *const uright,
	                          const j_real *const n,
	                          j_real *const dfdl, j_real *const dfdr) const = 0;

	virtual ~InviscidFlux();

protected:
	const IdealGasPhysics<scalar> *const physics;  ///< Functionality replated to gas constitutive law

	/// Gas physics for Jacobian computation
	/// This is a copy of physics above, but using scalar type.
	const IdealGasPhysics<j_real> *const jphy;

	const a_real g;                                ///< Adiabatic index
};

template <typename scalar, typename j_real = a_real>
class LocalLaxFriedrichsFlux : public InviscidFlux<scalar,j_real>
{
public:
	LocalLaxFriedrichsFlux(const IdealGasPhysics<scalar> *const analyticalflux);

	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const scalar *const uleft, const scalar *const uright, const scalar* const n,
			scalar *const flux) const;

	/** Currently computes an approximate Jacobian with frozen spectral radius.
	 * This has been found to perform no worse than the exact Jacobian for inviscid flows.
	 * \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const j_real *const uleft, const j_real *const uright, const j_real* const n,
	                  j_real *const dfdl, j_real *const dfdr) const;

	/** Computes the exact Jacobian.
	 * This has been done to make the frozen Jacobian default.
	 */
	void get_jacobian_2(const j_real *const ul, const j_real *const ur, const j_real* const n,
			j_real *const dfdl, j_real *const dfdr) const;

protected:
	using InviscidFlux<scalar,j_real>::physics;
	using InviscidFlux<scalar,j_real>::jphy;
	using InviscidFlux<scalar,j_real>::g;
};

/// Van-Leer flux-vector-splitting
template <typename scalar, typename j_real = a_real>
class VanLeerFlux : public InviscidFlux<scalar,j_real>
{
public:
	VanLeerFlux(const IdealGasPhysics<scalar> *const analyticalflux);
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const scalar *const ul, const scalar *const ur, const scalar* const n,
	              scalar *const flux) const;
	void get_jacobian(const j_real *const ul, const j_real *const ur, const j_real* const n,
	                  j_real *const dfdl, j_real *const dfdr) const;

protected:
	using InviscidFlux<scalar,j_real>::physics;
	using InviscidFlux<scalar,j_real>::jphy;
	using InviscidFlux<scalar,j_real>::g;
};

/// Liou-Steffen AUSM flux-vector-splitting
/** I call this flux vector splitting (FVS) for want of a better term; even though it is not
 * FVS strictly speaking, it's close enough. It's a FVS of the convective and pressure fluxes
 * separately.
 * \warning The Jacobian does not work; use the LLF Jacobian instead.
 */
template <typename scalar, typename j_real = a_real>
class AUSMFlux : public InviscidFlux<scalar,j_real>
{
public:
	AUSMFlux(const IdealGasPhysics<scalar> *const analyticalflux);

	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const scalar *const ul, const scalar *const ur, const scalar* const n,
	              scalar *const flux) const;

	/** \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const j_real *const ul, const j_real *const ur, const j_real* const n,
	                  j_real *const dfdl, j_real *const dfdr) const;

protected:
	using InviscidFlux<scalar,j_real>::physics;
	using InviscidFlux<scalar,j_real>::jphy;
	using InviscidFlux<scalar,j_real>::g;
};

/// Liou's AUSM+ flux
template <typename scalar, typename j_real = a_real>
class AUSMPlusFlux : public InviscidFlux<scalar,j_real>
{
public:
	AUSMPlusFlux(const IdealGasPhysics<scalar> *const analyticalflux);
	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const scalar *const ul, const scalar *const ur, const scalar* const n,
			scalar *const flux) const;

	void get_jacobian(const j_real *const ul, const j_real *const ur, const j_real* const n,
			j_real *const dfdl, j_real *const dfdr) const;

protected:
	using InviscidFlux<scalar,j_real>::physics;
	using InviscidFlux<scalar,j_real>::jphy;
	using InviscidFlux<scalar,j_real>::g;
};

/// Abstract class for fluxes which depend on Roe-averages
template <typename scalar, typename j_real = a_real>
class RoeAverageBasedFlux : public InviscidFlux<scalar,j_real>
{
public:
	RoeAverageBasedFlux(const IdealGasPhysics<scalar> *const analyticalflux);
	virtual void get_flux(const scalar *const ul, const scalar *const ur, const scalar* const n,
			scalar *const flux) const = 0;
	virtual void get_jacobian(const j_real *const ul, const j_real *const ur, const j_real* const n,
			j_real *const dfdl, j_real *const dfdr) const = 0;

protected:
	using InviscidFlux<scalar,j_real>::physics;
	using InviscidFlux<scalar,j_real>::jphy;
	using InviscidFlux<scalar,j_real>::g;

	/// Computes Roe-averaged quantities
	void getRoeAverages(const scalar ul[NVARS], const scalar ur[NVARS], const scalar n[NDIM],
		const scalar vxi, const scalar vyi, const scalar Hi,
		const scalar vxj, const scalar vyj, const scalar Hj,
		scalar& Rij, scalar& rhoij, scalar& vxij, scalar& vyij, scalar &vm2ij, scalar& vnij,
		scalar& Hij, scalar& cij) const
	{
		Rij = sqrt(ur[0]/ul[0]);
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
		const j_real ul[NVARS], const j_real ur[NVARS], const j_real n[NDIM],
		const j_real vxi, const j_real vyi, const j_real Hi,
		const j_real vxj, const j_real vyj, const j_real Hj,
		const j_real dvxi[NVARS], const j_real dvyi[NVARS], const j_real dHi[NVARS],
		const j_real dvxj[NVARS], const j_real dvyj[NVARS], const j_real dHj[NVARS],
		j_real dRiji[NVARS], j_real drhoiji[NVARS], j_real dvxiji[NVARS], j_real dvyiji[NVARS],
		j_real dvm2iji[NVARS], j_real dvniji[NVARS], j_real dHiji[NVARS], j_real dciji[NVARS],
		j_real dRijj[NVARS], j_real drhoijj[NVARS], j_real dvxijj[NVARS], j_real dvyijj[NVARS],
		j_real dvm2ijj[NVARS], j_real dvnijj[NVARS], j_real dHijj[NVARS], j_real dcijj[NVARS] ) const;
};

/// Roe-Pike flux-difference splitting
/** From Blazek \cite{blazek}.
 */
template <typename scalar, typename j_real = a_real>
class RoeFlux : public RoeAverageBasedFlux<scalar,j_real>
{
public:
	RoeFlux(const IdealGasPhysics<scalar> *const analyticalflux);

	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const scalar *const ul, const scalar *const ur, const scalar* const n,
	              scalar *const flux) const;

	/** \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const j_real *const ul, const j_real *const ur, const j_real* const n,
	                  j_real *const dfdl, j_real *const dfdr) const;

protected:
	using InviscidFlux<scalar,j_real>::physics;
	using InviscidFlux<scalar,j_real>::jphy;
	using InviscidFlux<scalar,j_real>::g;
	using RoeAverageBasedFlux<scalar,j_real>::getRoeAverages;
	using RoeAverageBasedFlux<scalar,j_real>::getJacobiansRoeAveragesWrtConserved;

	/// Entropy fix parameter
	const a_real fixeps;
};

/// Harten Lax Van-Leer numerical flux
/** Decent for inviscid flows.
 * \cite invflux_batten
 */
template <typename scalar, typename j_real = a_real>
class HLLFlux : public RoeAverageBasedFlux<scalar,j_real>
{
public:
	HLLFlux(const IdealGasPhysics<scalar> *const analyticalflux);

	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const scalar *const ul, const scalar *const ur, const scalar* const n,
	              scalar *const flux) const;

	/** Computes an approximate Jacobian of the HLL fluxes assuming frozen signal speeds
	 * \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const j_real *const ul, const j_real *const ur, const j_real* const n,
	                  j_real *const dfdl, j_real *const dfdr) const;

protected:
	using InviscidFlux<scalar,j_real>::physics;
	using InviscidFlux<scalar,j_real>::jphy;
	using InviscidFlux<scalar,j_real>::g;
	using RoeAverageBasedFlux<scalar,j_real>::getRoeAverages;
	using RoeAverageBasedFlux<scalar,j_real>::getJacobiansRoeAveragesWrtConserved;
};

/// Harten Lax Van-Leer numerical flux with contact restoration by Toro
/** Implemented as described by Batten et al. \cite invflux_hllc_batten
 * Good for both inviscid and viscous flows.
 */
template <typename scalar, typename j_real = a_real>
class HLLCFlux : public RoeAverageBasedFlux<scalar,j_real>
{
public:
	HLLCFlux(const IdealGasPhysics<scalar> *const analyticalflux);

	/** \sa InviscidFlux::get_flux
	 */
	void get_flux(const scalar *const ul, const scalar *const ur, const scalar* const n,
	              scalar *const flux) const;

	/** \sa InviscidFlux::get_jacobian
	 * \warning The output is *assigned* to the arrays dfdl and dfdr - any prior contents are lost!
	 */
	void get_jacobian(const j_real *const ul, const j_real *const ur, const j_real* const n,
	                  j_real *const dfdl, j_real *const dfdr) const;

protected:
	using InviscidFlux<scalar,j_real>::physics;
	using InviscidFlux<scalar,j_real>::jphy;
	using InviscidFlux<scalar,j_real>::g;
	using RoeAverageBasedFlux<scalar,j_real>::getRoeAverages;
	using RoeAverageBasedFlux<scalar,j_real>::getJacobiansRoeAveragesWrtConserved;

	/// Computes the averaged state between the waves in the Riemann fan
	/** \param[in] u The state outside the Riemann fan
	 * \param[in] n Normal to the face
	 * \param[in] vn Normal velocity corresponding to u and n
	 * \param[in[ p Pressure corresponding to u
	 * \param[in] ss Signal speed separating u from the averaged state
	 * \param[in] sm Contact wave speed
	 * \param[in|out] ustr The output average state
	 */
	void getStarState(const scalar u[NVARS], const scalar n[NDIM],
	                  const scalar vn, const scalar p,
	                  const scalar ss, const scalar sm,
	                  scalar *const __restrict ustr) const;

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
	void getStarStateAndJacobian(const j_real u[NVARS], const j_real n[NDIM],
		const j_real vn, const j_real p,
		const j_real ss, const j_real sm,
		const j_real dvn[NVARS], const j_real dp[NVARS],
		const j_real dssi[NDIM], const j_real dsmi[NDIM],
		const j_real dssj[NDIM], const j_real dsmj[NDIM],
		j_real ustr[NVARS],
		j_real dustri[NVARS][NVARS],
		j_real dustrj[NVARS][NVARS]) const __attribute((always_inline));
};

} // end namespace fvens

#endif
