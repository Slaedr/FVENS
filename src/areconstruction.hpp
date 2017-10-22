/** @file areconstruction.hpp
 * @brief Slope limiters for variable-extrapolation reconstruction
 * @author Aditya Kashi
 */

#ifndef ARECONSTRUCTION_H
#define ARECONSTRUCTION_H

#include "aconstants.hpp"
#include "aarray2d.hpp"
#include "amesh2dh.hpp"

namespace acfd {

/// Abstract class for computing face values from cell-centered values and gradients
/** \note Face values at boundary faces are only computed for the left (interior) side. 
 * Right side values for boundary faces need to computed elsewhere using boundary conditions.
 */
class SolutionReconstruction
{
protected:
	const UMesh2dh* m;
	const amat::Array2d<a_real>* ri;		///< coords of cell centers of cells
	const amat::Array2d<a_real>* gr;		///< coords of gauss points of each face
	int ng;									///< Number of Gauss points

public:
	SolutionReconstruction();

    SolutionReconstruction (const UMesh2dh* mesh,            ///< Mesh context
			const amat::Array2d<a_real>* c_centres,       ///< Cell centres
			const amat::Array2d<a_real>* gauss_r);        ///< Coords of Gauss points

    void setup(const UMesh2dh* mesh,
			const amat::Array2d<a_real>* c_centres, 
			const amat::Array2d<a_real>* gauss_r);

	virtual void compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& unknowns, 
			const amat::Array2d<a_real>& unknow_ghost, 
			const amat::Array2d<a_real>& x_deriv, const amat::Array2d<a_real>& y_deriv,
			amat::Array2d<a_real>& uface_left, amat::Array2d<a_real>& uface_right) = 0;

	virtual ~SolutionReconstruction();
};

/// Calculate values of variables at left and right sides of each face 
/// based on computed derivatives but without limiter.
/** ug (cell centered flow variables at ghost cells) are not used for this
 */
class LinearUnlimitedReconstruction : public SolutionReconstruction
{
public:
	/// Constructor. \sa SolutionReconstruction::SolutionReconstruction.
	LinearUnlimitedReconstruction(const UMesh2dh* mesh,
			const amat::Array2d<a_real>* c_centres, 
			const amat::Array2d<a_real>* gauss_r);

	void compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& unknowns, 
			const amat::Array2d<a_real>& unknow_ghost, 
			const amat::Array2d<a_real>& x_deriv, const amat::Array2d<a_real>& y_deriv, 
			amat::Array2d<a_real>& uface_left, amat::Array2d<a_real>& uface_right);
};

/// Computes state at left and right sides of each face based on WENO-limited derivatives 
/// at each cell
/** References:
 * - Y. Xia, X. Liu and H. Luo. "A finite volume method based on a WENO reconstruction 
 *   for compressible flows on hybrid grids", 52nd AIAA Aerospace Sciences Meeting, AIAA-2014-0939.
 * - M. Dumbser and M. Kaeser. "Arbitrary high order non-oscillatory finite volume schemes on 
 *   unsttructured meshes for linear hyperbolic systems", J. Comput. Phys. 221 pp 693--723, 2007.
 *
 * Note that we do not take the 'oscillation indicator' as the square of the magnitude of 
 * the gradient, like (it seems) in Dumbser & Kaeser, but unlike in Xia et. al.
 */
class WENOReconstruction : public SolutionReconstruction
{
	amat::Array2d<a_real> ldudx;
	amat::Array2d<a_real> ldudy;
	a_real gamma;
	a_real lambda;
	a_real epsilon;
public:
    WENOReconstruction(const UMesh2dh* mesh,
			const amat::Array2d<a_real>* c_centres, 
			const amat::Array2d<a_real>* gauss_r);

	void compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& unknowns, 
			const amat::Array2d<a_real>& unknow_ghost, 
			const amat::Array2d<a_real>& x_deriv, const amat::Array2d<a_real>& y_deriv, 
			amat::Array2d<a_real>& uface_left, amat::Array2d<a_real>& uface_right);
};

/// Computes face values using MUSCL reconstruciton with Van-Albada limiter
class MUSCLVanAlbada : public SolutionReconstruction
{
	const a_real eps;                       ///< Small number
	const a_real k;                         ///< MUSCL order parameter

public:
    MUSCLVanAlbada(const UMesh2dh* mesh,
			const amat::Array2d<a_real>* c_centres, 
			const amat::Array2d<a_real>* gauss_r);
    
	void compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& unknowns, 
			const amat::Array2d<a_real>& unknow_ghost, 
			const amat::Array2d<a_real>& x_deriv, const amat::Array2d<a_real>& y_deriv, 
			amat::Array2d<a_real>& uface_left, amat::Array2d<a_real>& uface_right);
};

/// Non-differentiable multidimensional slope limiter for linear reconstruction
class BarthJespersenLimiter : public SolutionReconstruction
{
public:
    BarthJespersenLimiter(const UMesh2dh* mesh, 
			const amat::Array2d<a_real>* c_centres, 
			const amat::Array2d<a_real>* gauss_r);
    
	void compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& unknowns, 
			const amat::Array2d<a_real>& unknow_ghost, 
			const amat::Array2d<a_real>& x_deriv, const amat::Array2d<a_real>& y_deriv, 
			amat::Array2d<a_real>& uface_left, amat::Array2d<a_real>& uface_right);
};

/// Differentiable modification of Barth-Jespersen limiter
class VenkatakrishnanLimiter: public SolutionReconstruction
{
	/// Parameter for adjusting limiting vs convergence
	a_real K;

	/// List of characteristic length of cells
	std::vector<a_real> clength;

public:
	/** \param[in] k_param Smaller values lead to better limiting at the expense of convergence,
	 *             higher values improve convergence at the expense of some oscillations
	 *             in the solution.
	 */
    VenkatakrishnanLimiter(const UMesh2dh* mesh, 
			const amat::Array2d<a_real>* c_centres, 
			const amat::Array2d<a_real>* gauss_r, a_real k_param);
    
	void compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& unknowns, 
			const amat::Array2d<a_real>& unknow_ghost, 
			const amat::Array2d<a_real>& x_deriv, const amat::Array2d<a_real>& y_deriv, 
			amat::Array2d<a_real>& uface_left, amat::Array2d<a_real>& uface_right);
};

/*template <int nvars>
void modifiedAverageGradient(const a_real *const dr, const a_real *const n);*/

} // end namespace
#endif
