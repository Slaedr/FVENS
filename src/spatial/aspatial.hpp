/** @file aspatial.hpp
 * @brief Common functionality for spatial discretization
 * @author Aditya Kashi
 * @date Feb 24, 2016; modified May 13 2017
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FVENS_ASPATIAL_H
#define FVENS_ASPATIAL_H 1

#include <array>
#include <tuple>
#include <petscmat.h>

#include "aconstants.hpp"
#include "utilities/aarray2d.hpp"
#include "linalg/petscutils.hpp"

#include "mesh/mesh.hpp"

#include "comm/ghostveccomm.hpp"

namespace fvens {

/// Base class for finite volume spatial discretization
template<typename scalar, int nvars>
class Spatial
{
public:
	/// Common setup required for finite volume discretizations
	/** Computes and stores cell centre coordinates, ghost cells' centres, and
	 * quadrature point coordinates.
	 */
	Spatial(const UMesh<scalar,NDIM> *const mesh);

	virtual ~Spatial();

	/// Computes the residual and local time steps
	/** By convention, we need to compute the negative of the nonlinear function whose root
	 * we want to find. Note that our nonlinear function or residual is defined (for steady problems) as
	 * the sum (over all cells) of net outgoing fluxes from each cell
	 * \f$ r(u) = \sum_K \int_{\partial K} F \hat{n} d\gamma \f$ where $f K $f denotes a cell.
	 * For pseudo time-stepping, the output should be -r(u), where the ODE is
	 * M du/dt + r(u) = 0.
	 *
	 * \param[in] u The state at which the residual is to be computed
	 * \param[in|out] residual The residual is added to this
	 * \param[in] gettimesteps Whether time-step computation is required
	 * \param[out] dtm Local time steps are stored in this
	 */
	virtual StatusCode compute_residual(const Vec u, Vec residual,
	                                    const bool gettimesteps, Vec dtm) const = 0;

	/// Computes and assembles the residual Jacobian
	StatusCode assemble_jacobian(const Vec uvec, Mat A) const;

	/// Computes the blocks of the Jacobian matrix for the flux across an interior face
	/** It is supposed to be a point-block in dr/du when we want to solve [M du/dt +] r(u) = 0.
	 * The convention is that L and U should go into the two off-diagonal blocks
	 * corresponding to the face. The negative of L and U are added to the
	 * diagonal blocks of the respective cells.
	 */
	virtual void
	compute_local_jacobian_interior(const fint iface,
	                                const freal *const ul, const freal *const ur,
	                                Eigen::Matrix <freal,nvars,nvars,Eigen::RowMajor>& L,
	                                Eigen::Matrix <freal,nvars,nvars,Eigen::RowMajor>& U
	                                ) const = 0;

	/// Computes the blocks of the Jacobian matrix for the flux across a boundary face
	/** The convention is that L is the negative of the matrix that is added to the diagonal block
	 * corresponding to the interior cell.
	 */
	virtual void compute_local_jacobian_boundary(const fint iface,
	                                             const freal *const ul,
	                                             Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& L
	                                             ) const = 0;


	/// Computes gradients of field variables and stores them in the argument
	virtual void getGradients(const Vec u, GradBlock_t<freal,NDIM,nvars> *const grads) const = 0;

	/// Exposes access to the mesh context
	const UMesh<scalar,NDIM>* mesh() const
	{
		return m;
	}

protected:
	/// Mesh context
	const UMesh<scalar,NDIM> *const m;

	/// Cell centers of both real cells and connectivity ghost cells
	Vec rcvec;

	/// Raw array accessor of cell centers of both real cells and connectivity ghost cells
	/** Remains in sync with \ref rcvec
	 * The first nelem rows correspond to real cells,
	 * the rest are ghost cells corresponding to connectivity boundaries
	 */
	ConstGhostedVecHandler<scalar> rch;

	/// Cell centres of ghost cells at physical boundaries
	amat::Array2d<scalar> rcbp;

	/// Pointer to cell-centre of first physical boundary face
	/** Required to deal with the case where a subdomain has no physical boundary faces.
	 */
	const scalar *rcbptr;

	/// Communication context for ghosted PETSc vectors
	GhostedBlockVecComm<NDIM> dimcomm;

	/// Faces' Gauss points' coords, stored a 3D array of dimensions
	/// naface x nguass x ndim (in that order)
	amat::Array2d<scalar> gr;

	/// Computes cell-centres of subdomain cells into \ref rcvec
	void update_subdomain_cell_centres();

	/// computes ghost cell centers assuming symmetry about the midpoint of the boundary face
	void compute_ghost_cell_coords_about_midpoint(amat::Array2d<scalar>& rchg);

	/// computes ghost cell centers assuming symmetry about the face
	void compute_ghost_cell_coords_about_face(amat::Array2d<scalar>& rchg);

	/// Computes a unique face gradient from cell-centred gradients using the modified average method
	/** \param ccleft Coordinates of left cell centre
	 * \param ccright Coordinates of right cell centre
	 * \param ucl The left cell-centred state
	 * \param ucr The right cell-centred state
	 * \param gradl Left cell-centred gradients (ndim x nvars flattened array)
	 * \param gradr Right cell-centred gradients (ndim x nvars flattened array)
	 * \param[out] grad Face gradient
	 */
	void getFaceGradient_modifiedAverage(const scalar *const ccleft, const scalar *const ccright,
	                                     const scalar *const ucl, const scalar *const ucr,
	                                     const scalar *const gradl, const scalar *const gradr,
	                                     scalar grad[NDIM][nvars]) const;

	/// Computes the thin-layer face gradient and its Jacobian w.r.t. the left and right states
	/** The Jacobians are computed w.r.t. whatever variables
	 * the derivatives dul and dur are computed with respect to.
	 * \param iface The \ref intfac index of the face at which the gradient Jacobian is to be computed
	 * \param ucl The left state
	 * \param ucr The right state
	 * \param dul The Jacobian of the left state w.r.t. the cell-centred conserved variables
	 * \param dur The Jacobian of the right state w.r.t. the cell-centred conserved variables
	 * \param[out] grad Face gradients
	 * \param[out] dgradl Jacobian of left cell-centred gradients
	 * \param[out] dgradr Jacobian of right cell-centred gradients
	 */
	void getFaceGradientAndJacobian_thinLayer(const scalar *const ccleft, const scalar *const ccright,
	                                          const freal *const ucl, const freal *const ucr,
	                                          const freal *const dul, const freal *const dur,
	                                          scalar grad[NDIM][nvars],
	                                          scalar dgradl[NDIM][nvars][nvars],
	                                          scalar dgradr[NDIM][nvars][nvars]) const;
};

}	// end namespace
#endif
