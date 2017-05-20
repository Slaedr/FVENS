#ifndef __ALIMITER_H

#ifndef __ACONSTANTS_H
#include "aconstants.hpp"
#endif

#ifndef __AARRAY2D_H
#include "aarray2d.hpp"
#endif

#ifndef __AMESH2DH_H
#include "amesh2dh.hpp"
#endif

#define __ALIMITER_H

namespace acfd {

/// Abstract class for computing face values (for each Gauss point) from cell-centered values and gradients
/** \note Face values at boundary faces are only computed for the left (interior) side. Right side values for boundary faces need to computed elsewhere using boundary conditions.
 */
class FaceDataComputation
{
protected:
	const UMesh2dh* m;
	const Matrix* u;					///< Cell-centered flow data
	const amat::Array2d<a_real>* ug;			///< Cell-centered ghost cell flow data
	const amat::Array2d<a_real>* dudx;
	const amat::Array2d<a_real>* dudy;
	const amat::Array2d<a_real>* rb;			///< coords of cell centers of ghost cells
	const amat::Array2d<a_real>* ri;			///< coords of cell centers of real cells
	const amat::Array2d<a_real>* gr;		/// coords of gauss points of each face
	amat::Array2d<a_real>* ufl;			///< left face flow data
	amat::Array2d<a_real>* ufr;			///< right face flow data
	int ng;									///< Number of Gauss points

public:
	FaceDataComputation();
    FaceDataComputation (const UMesh2dh* mesh, const amat::Array2d<a_real>* ghost_centres, const amat::Array2d<a_real>* c_centres, 
			const amat::Array2d<a_real>* gauss_r);
    void setup(const UMesh2dh* mesh, const amat::Array2d<a_real>* ghost_centres, const amat::Array2d<a_real>* c_centres, const amat::Array2d<a_real>* gauss_r);

	virtual void compute_face_values(const Matrix *const unknowns, const amat::Array2d<a_real> *const unknow_ghost, 
			const amat::Array2d<a_real> *const x_deriv, const amat::Array2d<a_real> *const y_deriv,
			amat::Array2d<a_real> *const uface_left, amat::Array2d<a_real> *const uface_right) = 0;

	virtual ~FaceDataComputation();
};

/// Calculate values of variables at left and right sides of each face based on computed derivatives but without limiter.
/** ug (cell centered flow variables at ghost cells) are not used for this
 */
class NoLimiter : public FaceDataComputation
{
public:
	/// Constructs the NoLimiter object. \sa FaceDataComputation::FaceDataComputation.
	NoLimiter(const UMesh2dh* mesh, const amat::Array2d<a_real>* ghost_centres, const amat::Array2d<a_real>* c_centres, 
			const amat::Array2d<a_real>* gauss_r);

	void compute_face_values(const Matrix *const unknowns, const amat::Array2d<a_real> *const unknow_ghost, 
			const amat::Array2d<a_real> *const x_deriv, const amat::Array2d<a_real> *const y_deriv, 
			amat::Array2d<a_real> *const uface_left, amat::Array2d<a_real> *const uface_right);
};

/// Computes state at left and right sides of each face based on WENO-limited derivatives at each cell
/** References:
 * - Y. Xia, X. Liu and H. Luo. "A finite volume method based on a WENO reconstruction for compressible flows on hybrid grids", 52nd AIAA Aerospace Sciences Meeting, AIAA-2014-0939.
 * - M. Dumbser and M. Kaeser. "Arbitrary high order non-oscillatory finite volume schemes on unsttructured meshes for linear hyperbolic systems", J. Comput. Phys. 221 pp 693--723, 2007.
 *
 * Note that we do not take the 'oscillation indicator' as the square of the magnitude of the gradient, like (it seems) in Dumbser & Kaeser, but unlike in Xia et. al.
 */
class WENOLimiter : public FaceDataComputation
{
	amat::Array2d<a_real>* ldudx;
	amat::Array2d<a_real>* ldudy;
	a_real gamma;
	a_real lambda;
	a_real epsilon;
public:
    WENOLimiter(const UMesh2dh* mesh, const amat::Array2d<a_real>* ghost_centres, const amat::Array2d<a_real>* c_centres, const amat::Array2d<a_real>* gauss_r);

	void compute_face_values(const Matrix *const unknowns, const amat::Array2d<a_real> *const unknow_ghost,
			const amat::Array2d<a_real> *const x_deriv, const amat::Array2d<a_real> *const y_deriv,
			amat::Array2d<a_real> *const uface_left, amat::Array2d<a_real> *const uface_right);

	~WENOLimiter();
};

/// Computes face values using the MUSCL scheme with Van-Albada limiter
class VanAlbadaLimiter : public FaceDataComputation
{
    a_real eps;
    a_real k;               			/// van-Albada parameter
	amat::Array2d<a_real> phi_l;		/// left-face limiter values
	amat::Array2d<a_real> phi_r;		/// right-face limiter values

public:
	void setup(const UMesh2dh* mesh, const amat::Array2d<a_real>* ghost_centres, const amat::Array2d<a_real>* r_centres, const amat::Array2d<a_real>* gauss_r);
    
	/// Calculate values of variables at left and right sides of each face based on computed derivatives and limiter values
	void compute_face_values(const Matrix *const unknowns, const amat::Array2d<a_real> *const unknow_ghost, 
			const amat::Array2d<a_real> *const x_deriv, const amat::Array2d<a_real> *const y_deriv, 
			amat::Array2d<a_real> *const uface_left, amat::Array2d<a_real> *const uface_right);
};

} // end namespace
#endif
