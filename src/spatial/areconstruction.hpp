/** @file areconstruction.hpp
 * @brief Slope limiters for variable-extrapolation reconstruction
 * @author Aditya Kashi
 */

#ifndef FVENS_ARECONSTRUCTION_H
#define FVENS_ARECONSTRUCTION_H

#include "aconstants.hpp"
#include "utilities/aarray2d.hpp"
#include "mesh/amesh2dh.hpp"

namespace fvens {

/// Abstract class for computing face values from cell-centered values and gradients
/** \note Face values at boundary faces are only computed for the left (interior) side. 
 * Right side values for boundary faces need to computed elsewhere using boundary conditions.
 */
template <typename scalar, int nvars>
class SolutionReconstruction
{
protected:
	const UMesh2dh<scalar> *const m;            ///< Mesh context
	/// Coords of cell-centres of subdomain and connectivity ghost cells
	const scalar *const ri;
	/// Coords of cell-centres of physical boundary ghost cells
	const scalar *const ribp;
	/// Coords of Gauss quadrature points of each face
	const amat::Array2d<scalar>& gr;

public:
	SolutionReconstruction (const UMesh2dh<scalar> *const  mesh,          ///< Mesh context
	                        const scalar *const c_centres,                ///< Cell centres
	                        const scalar *const c_centres_ghost,          ///< Ghost cell centres
	                        const amat::Array2d<scalar>& gauss_r);        ///< Coords of Gauss points

	virtual void compute_face_values(const MVector<scalar>& unknowns, 
	                                 const amat::Array2d<scalar>& unknow_ghost,
	                                 const scalar *const grads,
	                                 amat::Array2d<scalar>& uface_left,
	                                 amat::Array2d<scalar>& uface_right) const = 0;

	virtual ~SolutionReconstruction();
};

/// Calculate values of variables at left and right sides of each face 
/// based on computed derivatives but without limiter.
/** ug (cell centered flow variables at ghost cells) are not used for this.
 * In domain decomposition frameworks, node that this reconstruction does NOT need the cell gradients
 * of connectivity ghost cells.
 */
template <typename scalar, int nvars>
class LinearUnlimitedReconstruction : public SolutionReconstruction<scalar,nvars>
{
public:
	/// Constructor. \sa SolutionReconstruction::SolutionReconstruction.
	LinearUnlimitedReconstruction(const UMesh2dh<scalar> *const mesh,
	                              const scalar *const c_centres, 
	                              const scalar *const c_centres_ghost,
	                              const amat::Array2d<scalar>& gauss_r);

	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const scalar *const grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;

protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::gr;
};

} // end namespace
#endif
