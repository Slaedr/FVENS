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
	                   const amat::Array2d<scalar>& c_centres, 
	                   const amat::Array2d<scalar>* gauss_r,
	                   const a_real central_weight);

	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradArray<scalar,nvars>& grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;
protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::gr;
	using SolutionReconstruction<scalar,nvars>::ng;
};

/// Non-differentiable multidimensional slope limiter for linear reconstruction
template <typename scalar, int nvars>
class BarthJespersenLimiter : public SolutionReconstruction<scalar,nvars>
{
public:
	BarthJespersenLimiter(const UMesh2dh<scalar> *const mesh, 
	                      const amat::Array2d<scalar>& c_centres, 
	                      const amat::Array2d<scalar>* gauss_r);
    
	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradArray<scalar,nvars>& grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;
protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::gr;
	using SolutionReconstruction<scalar,nvars>::ng;
};

/// Differentiable modification of Barth-Jespersen limiter
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
	                       const amat::Array2d<scalar>& c_centres, 
	                       const amat::Array2d<scalar>* gauss_r, const a_real k_param);
    
	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradArray<scalar,nvars>& grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;
protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::gr;
	using SolutionReconstruction<scalar,nvars>::ng;
};

}

#endif
