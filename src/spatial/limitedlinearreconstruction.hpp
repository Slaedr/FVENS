/** \file
 * \brief Reconstruction schemes with limiters which are at best linear in smooth regions
 * \author Aditya Kashi
 */

#ifndef FVENS_LIMITEDLINEARRECONSTRUCTION_H
#define FVENS_LIMITEDLINEARRECONSTRUCTION_H

#include "areconstruction.hpp"

namespace fvens {

/// Computes state at left and right sides of each face based on WENO-limited derivatives 
/// at each cell
/** References: \cite xia2014, \cite dumbser2007.
 *
 * \warning In the domain decomposition framework,
 * note that this reconstruction is non-compact even in the cell-centred gradients. Cell gradients of
 * connectivity ghost cells are required.
 * 
 * If this limiter is used, then the computation of the residual in a cell requires
 * cell-centred state variables at that cell, its neighbors, its neighbors' neighbors and even
 * *their* neighbors! In other words, if only cell-centred states are communicated, three layers of
 * ghost cells are needed. If only one layer of ghost cells is used, then three messages are needed -
 * the neighbors' cell centred values, then their cell gradients and finally the shared face's
 * reconstructed value on the neighbors' side.
 *
 * Note that we do not take the 'oscillation indicator' as the square of the magnitude of 
 * the gradient, like (it seems) in Dumbser & Kaeser, but unlike in Xia et. al.
 */
template <typename scalar, int nvars>
class WENOReconstruction : public SolutionReconstruction<scalar,nvars>
{
	const a_real gamma;               ///< Exponent for oscillation indicator
	const a_real lambda;              ///< Weight of central stencil relative to biased stencils
	const a_real epsilon;             ///< Small constant to avoid division by zero
public:
	WENOReconstruction(const UMesh2dh<scalar> *const mesh,
	                   const scalar *const c_centres, 
	                   const scalar *const c_centres_ghost,
	                   const amat::Array2d<scalar>& gauss_r,
	                   const a_real central_weight);

	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const scalar *const grads,
	                         amat::Array2dMutableView<scalar> uface_left,
	                         amat::Array2dMutableView<scalar> uface_right) const;
protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::gr;
};

/// Non-differentiable multidimensional slope limiter for linear reconstruction
/** In domain decomposition frameworks, node that this reconstruction does NOT need the cell gradients
 * of connectivity ghost cells.
 */
template <typename scalar, int nvars>
class BarthJespersenLimiter : public SolutionReconstruction<scalar,nvars>
{
public:
	BarthJespersenLimiter(const UMesh2dh<scalar> *const mesh, 
	                      const scalar *const c_centres, 
	                      const scalar *const c_centres_ghost,
	                      const amat::Array2d<scalar>& gauss_r);
    
	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const scalar *const grads,
	                         amat::Array2dMutableView<scalar> uface_left,
	                         amat::Array2dMutableView<scalar> uface_right) const;
protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::gr;
};

/// Differentiable modification of Barth-Jespersen limiter
/** In domain decomposition frameworks, node that this reconstruction does NOT need the cell gradients
 * of connectivity ghost cells.
 */
template <typename scalar, int nvars>
class VenkatakrishnanLimiter: public SolutionReconstruction<scalar,nvars>
{
	/// Parameter for adjusting limiting vs convergence
	const a_real K;

	/// List of characteristic length of cells
	std::vector<scalar> clength;

public:
	/** \param[in] k_param Smaller values lead to better limiting at the expense of convergence,
	 *    higher values improve convergence at the expense of some oscillations in the solution.
	 */
	VenkatakrishnanLimiter(const UMesh2dh<scalar> *const mesh, 
	                       const scalar *const c_centres, 
	                       const scalar *const c_centres_ghost,
	                       const amat::Array2d<scalar>& gauss_r, const a_real k_param);
    
	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const scalar *const grads,
	                         amat::Array2dMutableView<scalar> uface_left,
	                         amat::Array2dMutableView<scalar> uface_right) const;
protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::gr;
};

}

#endif
