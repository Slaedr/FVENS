/** \file afactory.hpp
 * \brief Various factories for generating simulation-related objects
 * \author Aditya Kashi
 * \date 2017 October
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

#ifndef AFACTORY_H
#define AFACTORY_H

#include <string>

#include "aarray2d.hpp"
#include "spatial/anumericalflux.hpp"
#include "spatial/agradientschemes.hpp"
#include "spatial/areconstruction.hpp"
#include "spatial/flow_spatial.hpp"
#include "ode/nonlinearrelaxation.hpp"
#include "utilities/controlparser.hpp"

namespace fvens {

/// Returns a new inviscid numerical flux context
template <typename scalar>
InviscidFlux<scalar>* create_mutable_inviscidflux(const std::string& type, 
		const IdealGasPhysics<scalar> *const p) ;

/// Returns a new immutable inviscid flux context
template <typename scalar>
const InviscidFlux<scalar>* create_const_inviscidflux(const std::string& type, 
		const IdealGasPhysics<scalar> *const p) ;

/// Returns a newly-created gradient computation context
/** \param type Type of gradient scheme
 * \param m Mesh context. Currently its scalar type has to be same as that of the gradients etc
 * \param rc Array of cell centres all cells (including ghost cells); this must also currently have
 *   the same scalar type as the gradients.
 */
template <typename scalar, int nvars>
GradientScheme<scalar,nvars>* create_mutable_gradientscheme(const std::string& type, 
		const UMesh2dh<scalar> *const m, const amat::Array2d<scalar>& rc) ;

/// Returns a newly-created immutable gradient computation context
/** Parameters are as explained for \ref create_mutable_gradientscheme
 */
template <typename scalar, int nvars>
const GradientScheme<scalar,nvars>* create_const_gradientscheme(const std::string& type, 
		const UMesh2dh<scalar> *const m, const amat::Array2d<scalar>& rc) ;

/// Returns a solution reconstruction context
/** Solution reconstruction here means computing the values of the conserved variables at faces from
 * the cell-centred values and cell-centred gradients.
 * \param type Type of scheme to use for reconstructing the solution
 * \param m Mesh context. Currently its scalar type has to be same as that of the gradients etc.
 * \param rc Array of cell centres all cells (including ghost cells); this must also currently have
 *   the same scalar type as the gradients.
 * \param gr Coordinates of quadrature points at each face. They're generally the face centres.
 * \param param A parameter that controls the behaviour of some limiters.
 */
template <typename scalar, int nvars>
SolutionReconstruction<scalar,nvars>*
create_mutable_reconstruction(const std::string& type,
                              const UMesh2dh<scalar> *const m,
                              const amat::Array2d<scalar>& rc,
                              const amat::Array2d<scalar>& gr,
                              const a_real param);

/// Returns an immutable solution reconstruction context \sa create_mutable_reconstruction
template <typename scalar, int nvars>
const SolutionReconstruction<scalar,nvars>*
create_const_reconstruction(const std::string& type,
                            const UMesh2dh<scalar> *const m, const amat::Array2d<scalar>& rc,
                            const amat::Array2d<scalar>& gr, const a_real param);

/// Creates the appropriate flow solver class
/** This function is needed to instantiate the appropriate class from the \ref FlowFV template.
 */
template <typename scalar>
FlowFV_base<scalar>* create_mutable_flowSpatialDiscretization(
	const UMesh2dh<scalar> *const m,               ///< Mesh context
	const FlowPhysicsConfig& pconf,                ///< Physical data about the problem
	const FlowNumericsConfig& nconf);              ///< Options controlling the numerical method

/// Generates an immutable spatial discretization for slow problems
/** \sa create_mutable_flowSpatialDiscretization
 */
template <typename scalar>
const FlowFV_base<scalar>* create_const_flowSpatialDiscretization(
	const UMesh2dh<scalar> *const m,               ///< Mesh context
	const FlowPhysicsConfig& pconf,                ///< Physical data about the problem
	const FlowNumericsConfig& nconf);              ///< Options controlling the numerical method

/// Generates a nonlinear update (under-relaxation) scheme
template <int nvars>
const NonlinearUpdate<nvars>* create_const_nonlinearUpdateScheme(const FlowParserOptions& opts);

}
#endif
