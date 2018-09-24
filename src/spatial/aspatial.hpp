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

#ifndef ASPATIAL_H
#define ASPATIAL_H 1

#include <array>
#include <tuple>

#include "aconstants.hpp"
#include "utilities/aarray2d.hpp"

#include "mesh/amesh2dh.hpp"

#include <petscmat.h>

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
	Spatial(const UMesh2dh<scalar> *const mesh);

	virtual ~Spatial();
	
	/// Computes the residual and local time steps
	/** By convention, we need to compute the negative of the nonlinear function whose root
	 * we want to find. For pseudo time-stepping, the output should be -r(u), where the ODE is
	 * M du/dt + r(u) = 0.
	 * \param[in] u The state at which the residual is to be computed
	 * \param[in|out] residual The residual is added to this
	 * \param[in] gettimesteps Whether time-step computation is required
	 * \param[out] dtm Local time steps are stored in this
	 */
	virtual StatusCode assemble_residual(const Vec u, Vec residual, 
			const bool gettimesteps, std::vector<a_real>& dtm) const = 0;
	
	/// Computes the Jacobian matrix of the residual r(u)
	/** It is supposed to compute dr/du when we want to solve [M du/dt +] r(u) = 0.
	 */
	virtual StatusCode compute_jacobian(const Vec u, Mat A) const = 0;

	/// Computes gradients of field variables and stores them in the argument
	virtual void getGradients(const MVector<a_real>& u,
	                          GradArray<a_real,nvars>& grads) const = 0;

	/// Sets initial conditions
	/** \param[in] fromfile True if initial data is to be read from a file
	 * \param[in] file Name of initial conditions file
	 * \param[in|out] u Vector to store the initial data in
	 */
	virtual StatusCode initializeUnknowns(Vec u) const = 0;

	/// Exposes access to the mesh context
	const UMesh2dh<scalar>* mesh() const
	{
		return m;
	}

protected:
	/// Mesh context
	const UMesh2dh<scalar> *const m;

	/// Cell centers of both real cells and ghost cells
	/** The first nelem rows correspond to real cells, 
	 * the next nelem+nbface rows are ghost cell centres, indexed by nelem+iface for face iface.
	 */
	amat::Array2d<scalar> rc;

	/// Faces' Gauss points' coords, stored a 3D array of dimensions 
	/// naface x nguass x ndim (in that order)
	amat::Array2d<scalar>* gr;
	
	/// computes ghost cell centers assuming symmetry about the midpoint of the boundary face
	void compute_ghost_cell_coords_about_midpoint(amat::Array2d<scalar>& rchg);

	/// computes ghost cell centers assuming symmetry about the face
	void compute_ghost_cell_coords_about_face(amat::Array2d<scalar>& rchg);

	/// Computes a unique face gradient from cell-centred gradients using the modified average method
	/** \param iface The \ref intfac index of the face at which the gradient is to be computed
	 * \param ucl The left cell-centred state
	 * \param ucr The right cell-centred state
	 * \param gradl Left cell-centred gradients
	 * \param gradr Right cell-centred gradients
	 * \param[out] grad Face gradients
	 */
	void getFaceGradient_modifiedAverage(const a_int iface,
		const scalar *const ucl, const scalar *const ucr,
		const scalar gradl[NDIM][nvars], const scalar gradr[NDIM][nvars], scalar grad[NDIM][nvars])
		const;

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
	void getFaceGradientAndJacobian_thinLayer(const a_int iface,
		const a_real *const ucl, const a_real *const ucr,
		const a_real *const dul, const a_real *const dur,
		a_real grad[NDIM][nvars], a_real dgradl[NDIM][nvars][nvars], a_real dgradr[NDIM][nvars][nvars])
		const;
};

}	// end namespace
#endif
